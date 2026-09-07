from __future__ import annotations

"""B0/V1/V3 physical-error signature atlas with matched xy and xz views.

This is a presentation/sensitivity runner built on the canonical physical route.
The severity values are imported from ``run_vortex_error_showcase_v2`` so the
single-case showcase and the cross-charge atlas cannot silently diverge.

For each error family and each B0/V1/V3 case the runner emits:
  * xy fixed-laboratory/common-scale panels;
  * xy morphology-axis-centred/own-peak panels;
  * xz fixed-laboratory/common-scale panels with the tracked Bessel/vortex
    feature overlaid;
  * xz fixed-laboratory/own-peak panels so shape changes remain visible when
    throughput changes strongly.

Longitudinal maps are synthesized from the propagated angular spectrum in fixed
physical planes.  They are not image-space shifts, crops or interpolated camera
pictures.  The feature overlay is diagnostic only and never moves the xz data.

Every severity is an UNMEASURED SENSITIVITY LEVEL, not a claim about the bench.
All intensity heatmaps use inferno.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from tools.run_vortex_error_showcase_v2 import FAMILIES, _config_and_maps, _ell
from vbb_study.digital_twin.phase2a_contracts import canonical_hardware_manifest, hardware_value
from vbb_study.digital_twin.vortex_fixed_plane_bl_propagation import (
    build_bandlimited_fixed_plane_longitudinal_map,
)
from vbb_study.digital_twin.vortex_following_propagation import (
    bessel_feature_axis_path_m,
    transverse_morphology_axis,
)
from vbb_study.digital_twin.vortex_system_route import build_system_route
from vbb_study.equations.fields import make_xy_grid
from vbb_study.equations.propagation import angular_spectrum_propagate_bl

EPS = np.finfo(float).tiny
WINDOW_M = 10.0e-3
XY_SHAPE_HALFWIDTH_M = 0.55e-3
XY_LAB_HALFWIDTH_MIN_M = 0.85e-3
XY_LAB_HALFWIDTH_MAX_M = 2.0e-3
XZ_HALFWIDTH_M = 1.8e-3
XZ_Y_HALFWIDTH_M = 1.0e-3
MIN_RETAINED_SPECTRAL_POWER = 0.985


def _grid(grid_n: int) -> dict:
    return make_xy_grid(int(grid_n), WINDOW_M / int(grid_n))


def _centroid(intensity: np.ndarray, grid: dict) -> tuple[float, float]:
    I = np.maximum(np.asarray(intensity, float), 0.0)
    total = max(float(np.sum(I)), EPS)
    X = np.asarray(grid["X"], float)
    Y = np.asarray(grid["Y"], float)
    return float(np.sum(I * X) / total), float(np.sum(I * Y) / total)


def _axis_seed(family: str, value: float, centroid: tuple[float, float]) -> tuple[float, float]:
    if family == "axicon_decentre_x":
        return float(value), 0.0
    return float(centroid[0]), float(centroid[1])


def _transverse_axis(
    field: np.ndarray,
    intensity: np.ndarray,
    grid: dict,
    *,
    case_id: str,
    family: str,
    value: float,
) -> tuple[tuple[float, float], str, int, int]:
    centroid = _centroid(intensity, grid)
    seed = _axis_seed(family, value, centroid)
    try:
        axis = transverse_morphology_axis(
            field,
            grid,
            vortex_charge=_ell(case_id),
            seed_x_m=seed[0],
            seed_y_m=seed[1],
            search_radius_m=1.40e-3,
        )
        return (
            (float(axis.x_m), float(axis.y_m)),
            str(axis.method),
            int(axis.detected_topological_charge),
            int(axis.selected_singularity_count),
        )
    except RuntimeError:
        return centroid, "energy_centroid_fallback_after_unresolved_morphology_axis", 0, 0


def _crop_xy(
    intensity: np.ndarray,
    x: np.ndarray,
    *,
    centre_x_m: float,
    centre_y_m: float,
    halfwidth_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ix = np.flatnonzero(np.abs(x - float(centre_x_m)) <= float(halfwidth_m))
    iy = np.flatnonzero(np.abs(x - float(centre_y_m)) <= float(halfwidth_m))
    if ix.size < 48 or iy.size < 48:
        raise RuntimeError(
            f"atlas xy crop is under-sampled: nx={ix.size}, ny={iy.size}, halfwidth={halfwidth_m:g} m"
        )
    return (
        x[ix] - float(centre_x_m),
        x[iy] - float(centre_y_m),
        np.asarray(intensity)[np.ix_(iy, ix)],
    )


def _difference_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    ref = np.asarray(reference, float)
    cand = np.asarray(candidate, float)
    peak = max(float(np.max(ref)), EPS)
    diff = cand - ref
    return {
        "xy_rms_delta_over_nominal_peak": float(np.sqrt(np.mean(diff * diff)) / peak),
        "xy_max_abs_delta_over_nominal_peak": float(np.max(np.abs(diff)) / peak),
        "xy_intensity_correlation_to_nominal": float(np.corrcoef(ref.ravel(), cand.ravel())[0, 1]),
        "xy_peak_to_nominal_peak": float(np.max(cand) / peak),
        "xy_power_to_nominal_power": float(np.sum(cand) / max(float(np.sum(ref)), EPS)),
    }


def _lab_halfwidth(records_by_case: dict[str, list[dict]]) -> float:
    displacement = 0.0
    for records in records_by_case.values():
        for rec in records:
            ax, ay = rec["morphology_axis_m"]
            displacement = max(displacement, abs(float(ax)), abs(float(ay)))
    return float(
        np.clip(
            max(XY_LAB_HALFWIDTH_MIN_M, displacement + 0.70e-3),
            XY_LAB_HALFWIDTH_MIN_M,
            XY_LAB_HALFWIDTH_MAX_M,
        )
    )


def _save_xy_family(
    *, family_records: dict[str, list[dict]], spec, z_ref_m: float, output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    cases = ("B0", "V1", "V3")
    lab_halfwidth = _lab_halfwidth(family_records)

    for mode in ("lab_common", "morphology"):
        fig, axes = plt.subplots(3, 3, figsize=(12.4, 11.2), constrained_layout=True)
        for row, case_id in enumerate(cases):
            records = family_records[case_id]
            x = np.asarray(records[0]["grid"]["x"], float)
            row_peak = max(float(np.max(r["intensity"])) for r in records)
            for col, rec in enumerate(records):
                if mode == "lab_common":
                    xx, yy, image = _crop_xy(
                        rec["intensity"], x,
                        centre_x_m=0.0, centre_y_m=0.0,
                        halfwidth_m=lab_halfwidth,
                    )
                    image = image / max(row_peak, EPS)
                    extent = [xx[0] * 1e3, xx[-1] * 1e3, yy[0] * 1e3, yy[-1] * 1e3]
                    xlabel, ylabel = "lab x (mm)", "lab y (mm)"
                else:
                    ax, ay = rec["morphology_axis_m"]
                    xx, yy, image = _crop_xy(
                        rec["intensity"], x,
                        centre_x_m=ax, centre_y_m=ay,
                        halfwidth_m=XY_SHAPE_HALFWIDTH_M,
                    )
                    image = image / max(float(np.max(image)), EPS)
                    extent = [xx[0] * 1e6, xx[-1] * 1e6, yy[0] * 1e6, yy[-1] * 1e6]
                    xlabel, ylabel = "x from morphology axis (µm)", "y from morphology axis (µm)"
                axes[row, col].imshow(
                    image, origin="lower", extent=extent,
                    cmap="inferno", vmin=0.0, vmax=1.0,
                )
                axes[row, col].set_title(spec.value_label(float(rec["value"])), fontsize=9)
                axes[row, col].set_xlabel(xlabel, fontsize=8)
                axes[row, col].set_ylabel(ylabel, fontsize=8)
                if col == 0:
                    axes[row, col].text(
                        -0.27, 0.5, case_id,
                        transform=axes[row, col].transAxes,
                        va="center", ha="center", rotation=90,
                        fontsize=11, fontweight="bold",
                    )
        descriptor = (
            "fixed laboratory coordinates; common intensity scale within each B0/V1/V3 row"
            if mode == "lab_common"
            else "morphology-axis centred; each panel normalised to its own peak"
        )
        fig.suptitle(
            f"{spec.title} — B0 / V1 / V3 — xy at z={z_ref_m * 1e3:.0f} mm\n"
            f"{descriptor} — inferno\nUNMEASURED SENSITIVITY LEVELS",
            fontsize=13,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = "xy_lab_common" if mode == "lab_common" else "xy_morphology_normalised"
        fig.savefig(output_dir / f"{spec.key}_{stem}.png", dpi=320, bbox_inches="tight")
        fig.savefig(output_dir / f"{spec.key}_{stem}.svg", bbox_inches="tight")
        plt.close(fig)


def _save_xz_family(
    *, family_records: dict[str, list[dict]], spec, output_dir: Path, mode: str,
) -> None:
    import matplotlib.pyplot as plt

    cases = ("B0", "V1", "V3")
    fig, axes = plt.subplots(3, 3, figsize=(13.0, 10.4), constrained_layout=True)
    for row, case_id in enumerate(cases):
        records = family_records[case_id]
        row_peak = max(float(np.max(r["xz_intensity"])) for r in records)
        for col, rec in enumerate(records):
            xz = np.asarray(rec["xz_intensity"], float)
            scale = row_peak if mode == "common" else max(float(np.max(xz)), EPS)
            image = xz / max(scale, EPS)
            z = np.asarray(rec["z_m"], float)
            x = np.asarray(rec["xz_x_m"], float)
            axes[row, col].imshow(
                image.T,
                origin="lower",
                aspect="auto",
                extent=[z[0] * 1e3, z[-1] * 1e3, x[0] * 1e3, x[-1] * 1e3],
                cmap="inferno", vmin=0.0, vmax=1.0,
            )
            path = np.asarray(rec["xz_feature_path_m"], float)
            axes[row, col].plot(z * 1e3, path * 1e3, "w--", lw=0.8, alpha=0.75)
            axes[row, col].set_title(spec.value_label(float(rec["value"])), fontsize=9)
            axes[row, col].set_xlabel("z from axicon (mm)", fontsize=8)
            axes[row, col].set_ylabel("laboratory x (mm)", fontsize=8)
            if col == 0:
                axes[row, col].text(
                    -0.28, 0.5, case_id,
                    transform=axes[row, col].transAxes,
                    va="center", ha="center", rotation=90,
                    fontsize=11, fontweight="bold",
                )
    descriptor = (
        "fixed x-z laboratory plane; common intensity scale within each B0/V1/V3 row"
        if mode == "common"
        else "fixed x-z laboratory plane; each panel normalised to its own peak"
    )
    fig.suptitle(
        f"{spec.title} — B0 / V1 / V3 longitudinal signature\n"
        f"{descriptor}; dashed line = tracked Bessel/vortex feature — inferno\n"
        "UNMEASURED SENSITIVITY LEVELS",
        fontsize=13,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "xz_lab_common" if mode == "common" else "xz_own_peak"
    fig.savefig(output_dir / f"{spec.key}_{stem}.png", dpi=320, bbox_inches="tight")
    fig.savefig(output_dir / f"{spec.key}_{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def _save_strong_overview(
    *, all_records: dict[str, dict[str, list[dict]]], output_dir: Path, kind: str,
) -> None:
    import matplotlib.pyplot as plt

    cases = ("B0", "V1", "V3")
    fig, axes = plt.subplots(3, len(FAMILIES), figsize=(3.0 * len(FAMILIES), 9.2), constrained_layout=True)
    for row, case_id in enumerate(cases):
        for col, spec in enumerate(FAMILIES):
            rec = all_records[spec.key][case_id][-1]
            if kind == "xy":
                x = np.asarray(rec["grid"]["x"], float)
                ax, ay = rec["morphology_axis_m"]
                xx, yy, image = _crop_xy(
                    rec["intensity"], x,
                    centre_x_m=ax, centre_y_m=ay,
                    halfwidth_m=XY_SHAPE_HALFWIDTH_M,
                )
                image = image / max(float(np.max(image)), EPS)
                extent = [xx[0] * 1e6, xx[-1] * 1e6, yy[0] * 1e6, yy[-1] * 1e6]
                axes[row, col].imshow(
                    image, origin="lower", extent=extent,
                    cmap="inferno", vmin=0.0, vmax=1.0,
                )
                axes[row, col].set_xlabel("Δx (µm)", fontsize=7)
                axes[row, col].set_ylabel("Δy (µm)", fontsize=7)
            else:
                xz = np.asarray(rec["xz_intensity"], float)
                image = xz / max(float(np.max(xz)), EPS)
                z = np.asarray(rec["z_m"], float)
                x = np.asarray(rec["xz_x_m"], float)
                axes[row, col].imshow(
                    image.T, origin="lower", aspect="auto",
                    extent=[z[0] * 1e3, z[-1] * 1e3, x[0] * 1e3, x[-1] * 1e3],
                    cmap="inferno", vmin=0.0, vmax=1.0,
                )
                axes[row, col].plot(
                    z * 1e3, np.asarray(rec["xz_feature_path_m"]) * 1e3,
                    "w--", lw=0.65, alpha=0.75,
                )
                axes[row, col].set_xlabel("z (mm)", fontsize=7)
                axes[row, col].set_ylabel("lab x (mm)", fontsize=7)
            if row == 0:
                axes[row, col].set_title(
                    f"{spec.title}\n{spec.value_label(float(rec['value']))}",
                    fontsize=8,
                )
            if col == 0:
                axes[row, col].text(
                    -0.31, 0.5, case_id,
                    transform=axes[row, col].transAxes,
                    va="center", ha="center", rotation=90,
                    fontsize=11, fontweight="bold",
                )
    title = (
        "Strong-sensitivity xy morphology signatures — B0 / V1 / V3"
        if kind == "xy"
        else "Strong-sensitivity xz signatures — B0 / V1 / V3"
    )
    fig.suptitle(
        title + "\nEach panel own-peak normalised — inferno — UNMEASURED SENSITIVITY LEVELS",
        fontsize=13,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"strong_error_{kind}_overview.png", dpi=320, bbox_inches="tight")
    fig.savefig(output_dir / f"strong_error_{kind}_overview.svg", bbox_inches="tight")
    plt.close(fig)


def run(
    *, cases: tuple[str, ...], grid_n: int, z_ref_m: float,
    z_min_m: float, z_max_m: float, z_samples: int, output_root: Path,
) -> dict:
    if tuple(cases) != ("B0", "V1", "V3"):
        raise ValueError("signature atlas requires cases exactly B0 V1 V3 for matched rows")
    manifest = canonical_hardware_manifest()
    wavelength = float(hardware_value(manifest, "wavelength_m"))
    gamma = math.radians(float(hardware_value(manifest, "axicon_base_angle_deg")))
    grid_for_maps = _grid(grid_n)
    z_values = np.linspace(float(z_min_m), float(z_max_m), int(z_samples))
    if not np.any(np.isclose(z_values, float(z_ref_m), rtol=0.0, atol=1e-12)):
        z_values = np.sort(np.unique(np.append(z_values, float(z_ref_m))))
    xz_x = np.linspace(-XZ_HALFWIDTH_M, XZ_HALFWIDTH_M, 361)
    xz_y = np.linspace(-XZ_Y_HALFWIDTH_M, XZ_Y_HALFWIDTH_M, 181)

    all_records: dict[str, dict[str, list[dict]]] = {}
    metric_rows: list[dict] = []

    for spec in FAMILIES:
        print(f"[atlas] family={spec.key}", flush=True)
        family_records: dict[str, list[dict]] = {}
        for case_id in cases:
            print(f"  case={case_id}", flush=True)
            records: list[dict] = []
            for value in spec.values:
                print(f"    value={spec.value_label(float(value))}", flush=True)
                config, maps = _config_and_maps(
                    spec.key, float(value), grid=grid_for_maps,
                    wavelength_m=wavelength, axicon_base_angle_rad=gamma,
                )
                route = build_system_route(case_id, grid_n=grid_n, config=config, **maps)
                grid = dict(route["grid"])
                field_ref = angular_spectrum_propagate_bl(
                    route["post_axicon"], grid, wavelength, float(z_ref_m), bandlimit=True
                )
                intensity_ref = np.abs(field_ref) ** 2
                axis, axis_method, detected_charge, singularities = _transverse_axis(
                    field_ref, intensity_ref, grid,
                    case_id=case_id, family=spec.key, value=float(value),
                )
                centroid = _centroid(intensity_ref, grid)

                longitudinal = build_bandlimited_fixed_plane_longitudinal_map(
                    route["post_axicon"], grid,
                    wavelength_m=wavelength,
                    z_values_m=z_values,
                    x_coordinates_m=xz_x,
                    y_coordinates_m=xz_y,
                    fixed_x_m=0.0,
                    fixed_y_m=0.0,
                    minimum_retained_spectral_power=MIN_RETAINED_SPECTRAL_POWER,
                    source_label=f"atlas:{case_id}:{spec.key}:{float(value):.12g}",
                )
                feature_path = bessel_feature_axis_path_m(
                    longitudinal.xz_intensity,
                    longitudinal.x_coordinates_m,
                    vortex_charge=_ell(case_id),
                    search_halfwidth_m=0.65e-3,
                )
                records.append({
                    "value": float(value),
                    "grid": grid,
                    "intensity": intensity_ref,
                    "centroid_m": centroid,
                    "morphology_axis_m": axis,
                    "axis_method": axis_method,
                    "detected_topological_charge": detected_charge,
                    "selected_singularity_count": singularities,
                    "z_m": np.asarray(longitudinal.z_m, float),
                    "xz_x_m": np.asarray(longitudinal.x_coordinates_m, float),
                    "xz_intensity": np.asarray(longitudinal.xz_intensity, float),
                    "xz_feature_path_m": np.asarray(feature_path, float),
                    "minimum_retained_spectral_power_fraction": float(
                        np.min(longitudinal.support_retained_spectral_power_fraction)
                    ),
                })

            nominal = records[0]
            nominal_intensity = nominal["intensity"]
            ncx, ncy = nominal["centroid_m"]
            nax, nay = nominal["morphology_axis_m"]
            for rec in records:
                cx, cy = rec["centroid_m"]
                ax, ay = rec["morphology_axis_m"]
                path = np.asarray(rec["xz_feature_path_m"], float)
                metric_rows.append({
                    "case_id": case_id,
                    "family": spec.key,
                    "family_title": spec.title,
                    "value": rec["value"],
                    "value_label": spec.value_label(float(rec["value"])),
                    "provenance": spec.provenance,
                    "grid_n": int(grid_n),
                    "dx_m": float(rec["grid"]["dx"]),
                    "z_ref_m": float(z_ref_m),
                    "centroid_x_m": float(cx),
                    "centroid_y_m": float(cy),
                    "centroid_shift_from_nominal_m": float(math.hypot(cx - ncx, cy - ncy)),
                    "morphology_axis_x_m": float(ax),
                    "morphology_axis_y_m": float(ay),
                    "morphology_axis_shift_from_nominal_m": float(math.hypot(ax - nax, ay - nay)),
                    "axis_method": rec["axis_method"],
                    "detected_topological_charge": int(rec["detected_topological_charge"]),
                    "selected_singularity_count": int(rec["selected_singularity_count"]),
                    "xz_feature_min_m": float(np.min(path)),
                    "xz_feature_max_m": float(np.max(path)),
                    "xz_feature_peak_to_peak_m": float(np.ptp(path)),
                    "minimum_retained_spectral_power_fraction": float(
                        rec["minimum_retained_spectral_power_fraction"]
                    ),
                    **_difference_metrics(nominal_intensity, rec["intensity"]),
                })
            family_records[case_id] = records

        all_records[spec.key] = family_records
        family_dir = output_root / "figures" / "families"
        _save_xy_family(
            family_records=family_records, spec=spec,
            z_ref_m=z_ref_m, output_dir=family_dir,
        )
        _save_xz_family(
            family_records=family_records, spec=spec,
            output_dir=family_dir, mode="common",
        )
        _save_xz_family(
            family_records=family_records, spec=spec,
            output_dir=family_dir, mode="own",
        )

    overview_dir = output_root / "figures" / "overview"
    _save_strong_overview(all_records=all_records, output_dir=overview_dir, kind="xy")
    _save_strong_overview(all_records=all_records, output_dir=overview_dir, kind="xz")

    validation_dir = output_root / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    with (validation_dir / "signature_atlas_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metric_rows)

    manifest_out = {
        "outcome": "B0-V1-V3-PHYSICAL-ERROR-SIGNATURE-ATLAS",
        "cases": list(cases),
        "grid_n": int(grid_n),
        "window_m": WINDOW_M,
        "z_reference_m": float(z_ref_m),
        "z_range_m": [float(z_values[0]), float(z_values[-1])],
        "z_samples": int(z_values.size),
        "heatmap": "inferno",
        "longitudinal_model": "distance-aware Matsushima BL-ASM in fixed physical x-z/y-z planes",
        "longitudinal_feature_overlay": "tracked Bessel/vortex feature; overlay does not recenter x-z data",
        "minimum_retained_spectral_power_required": MIN_RETAINED_SPECTRAL_POWER,
        "severity_provenance": "all values are unmeasured sensitivity levels",
        "families": [
            {
                "key": spec.key,
                "title": spec.title,
                "values": list(spec.values),
                "labels": [spec.value_label(v) for v in spec.values],
                "physical_basis": spec.provenance,
            }
            for spec in FAMILIES
        ],
    }
    (validation_dir / "signature_atlas_manifest.json").write_text(
        json.dumps(manifest_out, indent=2), encoding="utf-8"
    )
    return manifest_out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", choices=("B0", "V1", "V3"), default=["B0", "V1", "V3"])
    parser.add_argument("--grid-n", type=int, default=1024)
    parser.add_argument("--z-ref-mm", type=float, default=60.0)
    parser.add_argument("--z-min-mm", type=float, default=5.0)
    parser.add_argument("--z-max-mm", type=float, default=110.0)
    parser.add_argument("--z-samples", type=int, default=61)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/error_signature_atlas"))
    args = parser.parse_args()
    if int(args.grid_n) < 768:
        raise SystemExit("presentation signature atlas requires grid_n >= 768")
    if int(args.z_samples) < 32:
        raise SystemExit("signature atlas requires at least 32 longitudinal z samples")
    result = run(
        cases=tuple(args.cases),
        grid_n=int(args.grid_n),
        z_ref_m=float(args.z_ref_mm) * 1e-3,
        z_min_m=float(args.z_min_mm) * 1e-3,
        z_max_m=float(args.z_max_mm) * 1e-3,
        z_samples=int(args.z_samples),
        output_root=args.output_root,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
