from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from vbb_study.digital_twin.phase2e_spectral_propagation import build_dense_spectral_propagation
from vbb_study.digital_twin.vortex_system_error_sweeps import (
    blocked_or_data_driven_families,
    system_sweep_registry,
)
from vbb_study.digital_twin.vortex_system_route import build_system_route
from vbb_study.equations.propagation import angular_spectrum_propagate_bl


EPS = np.finfo(float).tiny


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _metrics(I: np.ndarray, grid: dict[str, Any]) -> dict[str, float]:
    arr = np.maximum(np.asarray(I, dtype=float), 0.0)
    total = float(np.sum(arr))
    X = np.asarray(grid["X"], dtype=float)
    Y = np.asarray(grid["Y"], dtype=float)
    peak = float(np.max(arr))
    cx = float(np.sum(arr * X) / max(total, EPS))
    cy = float(np.sum(arr * Y) / max(total, EPS))
    radial_second_moment = float(
        np.sum(arr * ((X - cx) ** 2 + (Y - cy) ** 2)) / max(total, EPS)
    )
    return {
        "peak_au": peak,
        "power_au": total * float(grid["dx"]) ** 2,
        "centroid_x_m": cx,
        "centroid_y_m": cy,
        "rms_radius_m": float(np.sqrt(max(radial_second_moment, 0.0))),
    }


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=float).ravel().copy()
    bb = np.asarray(b, dtype=float).ravel().copy()
    aa -= float(np.mean(aa))
    bb -= float(np.mean(bb))
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom <= EPS:
        return 1.0 if np.allclose(aa, bb) else 0.0
    return float(np.dot(aa, bb) / denom)


def _line_metrics(
    intensity_zx: np.ndarray,
    *,
    z_m: np.ndarray,
    coordinate_m: np.ndarray,
) -> dict[str, Any]:
    """Metrics for an x-z or y-z line map without assuming the axis is fixed."""

    I = np.maximum(np.asarray(intensity_zx, dtype=float), 0.0)
    coord = np.asarray(coordinate_m, dtype=float)
    z = np.asarray(z_m, dtype=float)
    if I.shape != (z.size, coord.size):
        raise ValueError("line-map shape does not match z/transverse coordinates")

    line_power = np.sum(I, axis=1)
    centre = np.sum(I * coord[None, :], axis=1) / np.maximum(line_power, EPS)
    variance = (
        np.sum(I * (coord[None, :] - centre[:, None]) ** 2, axis=1)
        / np.maximum(line_power, EPS)
    )
    width = np.sqrt(np.maximum(variance, 0.0))
    peak_trace = np.max(I, axis=1)

    active = peak_trace >= 0.15 * max(float(np.max(peak_trace)), EPS)
    if int(np.count_nonzero(active)) >= 2:
        slope, intercept = np.polyfit(z[active], centre[active], 1)
        active_length = float(z[active][-1] - z[active][0])
        mean_width = float(np.mean(width[active]))
        centre_span = float(np.max(centre[active]) - np.min(centre[active]))
    else:
        slope = float("nan")
        intercept = float("nan")
        active_length = 0.0
        mean_width = float("nan")
        centre_span = float("nan")

    return {
        "centre_m": centre,
        "width_m": width,
        "peak_trace": peak_trace,
        "active_mask": active,
        "centre_slope_rad_approx": float(slope),
        "centre_intercept_m": float(intercept),
        "active_length_m": active_length,
        "mean_width_m": mean_width,
        "centre_span_m": centre_span,
    }


def _normalised_trace(trace: np.ndarray) -> np.ndarray:
    arr = np.maximum(np.asarray(trace, dtype=float), 0.0)
    return arr / max(float(np.max(arr)), EPS)


def _crop_square(
    arr: np.ndarray,
    x: np.ndarray,
    halfwidth: float,
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.abs(x) <= float(halfwidth)
    return x[mask], np.asarray(arr)[np.ix_(mask, mask)]


def _family_component_image(
    family: str,
    route: dict[str, Any],
    nominal: dict[str, Any],
) -> tuple[np.ndarray, str, str]:
    if family.startswith("beam_curvature"):
        phase = np.angle(
            np.asarray(route["input_beam"])
            * np.conj(np.asarray(nominal["input_beam"]))
        )
        return phase, "input beam phase difference vs nominal", "phase"

    if family.startswith("beam_"):
        return np.abs(route["input_beam"]) ** 2, "input beam intensity", "intensity"

    if family.startswith("slm"):
        return (
            np.abs(route["fourier_plane_before_iris"]) ** 2,
            "4F Fourier plane before iris",
            "intensity",
        )

    if family.startswith("fourf_iris"):
        after = (
            np.asarray(route["fourier_plane_before_iris"])
            * np.asarray(route["fourier_iris_mask"])
        )
        return np.abs(after) ** 2, "4F Fourier plane after fixed iris", "intensity"

    if family.startswith("fourf_lens2"):
        return (
            np.abs(route["post_4f_selected_order"]) ** 2,
            "4F output / selected-order plane before axicon",
            "intensity",
        )

    if family.startswith("fourf"):
        return (
            np.abs(route["fourier_plane_before_iris"]) ** 2,
            "4F Fourier plane before iris",
            "intensity",
        )

    if family.startswith("axicon_rigid_tilt"):
        phase = np.angle(
            np.asarray(route["field_on_axicon_plane"])
            * np.conj(np.asarray(nominal["field_on_axicon_plane"]))
        )
        return phase, "field phase on axicon local plane vs nominal", "phase"

    ratio = route["post_axicon_local"] / np.where(
        np.abs(route["field_on_axicon_plane"]) > 1e-9,
        route["field_on_axicon_plane"],
        1.0,
    )
    ratio0 = nominal["post_axicon_local"] / np.where(
        np.abs(nominal["field_on_axicon_plane"]) > 1e-9,
        nominal["field_on_axicon_plane"],
        1.0,
    )
    return (
        np.angle(ratio * np.conj(ratio0)),
        "axicon phase difference vs nominal",
        "phase",
    )


def _nominal_value(values: tuple[Any, ...]) -> Any:
    finite = [v for v in values if np.isfinite(float(v))]
    if any(np.isclose(float(v), 1.0) for v in finite):
        return min(finite, key=lambda v: abs(float(v) - 1.0))
    nonfinite = [v for v in values if not np.isfinite(float(v))]
    if nonfinite:
        return nonfinite[0]
    return min(values, key=lambda v: abs(float(v)))


def _same_value(a: Any, b: Any) -> bool:
    aa = float(a)
    bb = float(b)
    if not np.isfinite(aa) or not np.isfinite(bb):
        return (not np.isfinite(aa)) and (not np.isfinite(bb))
    return bool(np.isclose(aa, bb, rtol=0.0, atol=1e-15))


def _rotated_plane_diagnostics(route: dict[str, Any]) -> dict[str, float | str]:
    meta = route["metadata"]
    fourf = meta["fourf"]
    entries: dict[str, dict[str, Any]] = {
        "lens1_to": fourf["lens1"].get("lab_to_lens_plane", {}),
        "lens1_from": fourf["lens1"].get("lens_plane_to_lab", {}),
        "lens2_to": fourf["lens2"].get("lab_to_lens_plane", {}),
        "lens2_from": fourf["lens2"].get("lens_plane_to_lab", {}),
        "axicon_to": meta.get("lab_to_tilted", {}),
        "axicon_from": meta.get("tilted_to_lab", {}),
    }
    out: dict[str, float | str] = {}
    ratios: list[float] = []
    for name, entry in entries.items():
        ratio = float(entry.get("spectral_power_ratio", 1.0))
        out[f"{name}_spectral_power_ratio"] = ratio
        out[f"{name}_interpolation_model"] = str(
            entry.get("interpolation_model", "identity")
        )
        ratios.append(ratio)
    out["min_rotated_plane_spectral_power_ratio"] = float(min(ratios))
    return out


def _transverse_sampling(family: str) -> np.ndarray:
    """Use a wider diagnostic field for families that can translate/steer axes."""

    wide_tokens = (
        "lateral_decentre",
        "lens1_decentre",
        "lens2_decentre",
        "lens1_tilt",
        "lens2_tilt",
        "rigid_tilt",
    )
    if any(token in family for token in wide_tokens):
        return np.linspace(-0.65e-3, 0.65e-3, 401)
    return np.linspace(-0.18e-3, 0.18e-3, 321)


def run_family(
    family: str,
    *,
    case_id: str,
    grid_n: int,
    z_reference_m: float,
    output_root: Path,
    figure_root: Path,
) -> list[dict[str, Any]]:
    import matplotlib.pyplot as plt

    registry = system_sweep_registry()
    spec = registry[family]
    values = tuple(spec["values"])
    nominal_value = _nominal_value(values)
    nominal_config = spec["builder"](nominal_value)
    nominal_route = build_system_route(case_id, grid_n=grid_n, config=nominal_config)

    records: list[dict[str, Any]] = []
    z = np.arange(5e-3, 140e-3 + 2e-3, 2e-3)
    transverse = _transverse_sampling(family)

    for value in values:
        config = spec["builder"](value)
        route = build_system_route(case_id, grid_n=grid_n, config=config)
        grid = route["grid"]
        wavelength = float(route["metadata"]["wavelength_m"])
        propagated = angular_spectrum_propagate_bl(
            route["post_axicon"],
            dict(grid),
            wavelength,
            z_reference_m,
            n_medium=1.0,
            bandlimit=True,
            include_evanescent=True,
        )
        Ixy = np.abs(propagated) ** 2
        dense = build_dense_spectral_propagation(
            grid=grid,
            wavelength_m=wavelength,
            z_values_m=z,
            transverse_coordinates_m=transverse,
            scalar_field=route["post_axicon"],
            source_label=f"{case_id}:{family}={value}",
        )
        component, component_label, component_kind = _family_component_image(
            family,
            route,
            nominal_route,
        )
        records.append(
            {
                "value": value,
                "route": route,
                "Ixy": np.asarray(Ixy),
                "xz": np.asarray(dense.xz_intensity),
                "yz": np.asarray(dense.yz_intensity),
                "component": component,
                "component_label": component_label,
                "component_kind": component_kind,
            }
        )

    nominal_record = next(
        rec for rec in records if _same_value(rec["value"], nominal_value)
    )
    nominal_xy = np.asarray(nominal_record["Ixy"])
    nominal_xz = np.asarray(nominal_record["xz"])
    nominal_yz = np.asarray(nominal_record["yz"])
    nominal_xy_metrics = _metrics(nominal_xy, dict(nominal_route["grid"]))
    nominal_x_line = _line_metrics(
        nominal_xz,
        z_m=z,
        coordinate_m=transverse,
    )
    nominal_y_line = _line_metrics(
        nominal_yz,
        z_m=z,
        coordinate_m=transverse,
    )

    rows: list[dict[str, Any]] = []
    for rec in records:
        route = rec["route"]
        grid = dict(route["grid"])
        xy_metrics = _metrics(rec["Ixy"], grid)
        xline = _line_metrics(rec["xz"], z_m=z, coordinate_m=transverse)
        yline = _line_metrics(rec["yz"], z_m=z, coordinate_m=transverse)

        norm_peak = _normalised_trace(xline["peak_trace"])
        norm_peak0 = _normalised_trace(nominal_x_line["peak_trace"])
        residual = norm_peak - norm_peak0

        row = {
            "case_id": case_id,
            "family": family,
            "value": rec["value"],
            "nominal_value": nominal_value,
            "units": spec["units"],
            "fidelity": spec["fidelity"],
            "grid_n": grid_n,
            "z_reference_m": z_reference_m,
            "fourf_iris_selected_fraction": route["metadata"]["fourf"][
                "iris_selected_power_fraction"
            ],
            "axicon_tilt_status": route["metadata"]["axicon_tilt_status"],
            "exact_axicon_kr_m_inv": route["metadata"]["axicon"]["exact_kr_m_inv"],
            **xy_metrics,
            "output_power_ratio_to_nominal": float(
                xy_metrics["power_au"] / max(nominal_xy_metrics["power_au"], EPS)
            ),
            "xy_corr_nominal": _correlation(rec["Ixy"], nominal_xy),
            "xz_corr_nominal": _correlation(rec["xz"], nominal_xz),
            "yz_corr_nominal": _correlation(rec["yz"], nominal_yz),
            "xz_center_slope_rad_approx": xline["centre_slope_rad_approx"],
            "yz_center_slope_rad_approx": yline["centre_slope_rad_approx"],
            "xz_center_intercept_m": xline["centre_intercept_m"],
            "yz_center_intercept_m": yline["centre_intercept_m"],
            "xz_center_span_m": xline["centre_span_m"],
            "yz_center_span_m": yline["centre_span_m"],
            "xz_mean_width_m": xline["mean_width_m"],
            "yz_mean_width_m": yline["mean_width_m"],
            "xz_active_length_m": xline["active_length_m"],
            "yz_active_length_m": yline["active_length_m"],
            "axial_peak_corr_nominal": _correlation(
                xline["peak_trace"],
                nominal_x_line["peak_trace"],
            ),
            "axial_peak_residual_rms": float(np.sqrt(np.mean(residual * residual))),
            **_rotated_plane_diagnostics(route),
        }
        rows.append(row)

    x = np.asarray(nominal_route["grid"]["x"], dtype=float)
    xy_common = max(float(np.max(rec["Ixy"])) for rec in records)
    xz_common = max(float(np.max(rec["xz"])) for rec in records)
    yz_common = max(float(np.max(rec["yz"])) for rec in records)
    comp_abs = max(float(np.max(np.abs(rec["component"]))) for rec in records)

    fig, axes = plt.subplots(
        4,
        len(records),
        figsize=(3.1 * len(records), 11.2),
        constrained_layout=True,
        squeeze=False,
    )
    for col, rec in enumerate(records):
        value = rec["value"]
        comp = rec["component"]
        if rec["component_kind"] == "intensity":
            half = (
                3.0e-3
                if (family.startswith("slm") or family.startswith("fourf"))
                else 2.5e-3
            )
            xc, cc = _crop_square(comp, x, half)
            im0 = axes[0, col].imshow(
                cc / max(comp_abs, EPS),
                origin="lower",
                extent=[xc[0] * 1e3, xc[-1] * 1e3, xc[0] * 1e3, xc[-1] * 1e3],
                vmin=0,
                vmax=1,
                cmap="inferno",
            )
        else:
            xc, cc = _crop_square(comp, x, 2.5e-3)
            im0 = axes[0, col].imshow(
                cc,
                origin="lower",
                extent=[xc[0] * 1e3, xc[-1] * 1e3, xc[0] * 1e3, xc[-1] * 1e3],
                vmin=-np.pi,
                vmax=np.pi,
                cmap="twilight",
            )
        axes[0, col].set_xlabel("x (mm)")
        axes[0, col].set_ylabel("y (mm)")
        axes[0, col].set_title(str(value))

        xy_half = 0.65e-3 if np.max(np.abs(transverse)) > 0.2e-3 else 0.18e-3
        xc2, Icrop = _crop_square(rec["Ixy"], x, xy_half)
        im1 = axes[1, col].imshow(
            Icrop / max(xy_common, EPS),
            origin="lower",
            extent=[xc2[0] * 1e6, xc2[-1] * 1e6, xc2[0] * 1e6, xc2[-1] * 1e6],
            vmin=0,
            vmax=1,
            cmap="inferno",
        )
        axes[1, col].set_xlabel("x (µm)")
        axes[1, col].set_ylabel("y (µm)")

        im2 = axes[2, col].imshow(
            (rec["xz"] / max(xz_common, EPS)).T,
            origin="lower",
            aspect="auto",
            extent=[
                z[0] * 1e3,
                z[-1] * 1e3,
                transverse[0] * 1e6,
                transverse[-1] * 1e6,
            ],
            vmin=0,
            vmax=1,
            cmap="inferno",
        )
        axes[2, col].set_xlabel("z from axicon (mm)")
        axes[2, col].set_ylabel("x (µm)")

        im3 = axes[3, col].imshow(
            (rec["yz"] / max(yz_common, EPS)).T,
            origin="lower",
            aspect="auto",
            extent=[
                z[0] * 1e3,
                z[-1] * 1e3,
                transverse[0] * 1e6,
                transverse[-1] * 1e6,
            ],
            vmin=0,
            vmax=1,
            cmap="inferno",
        )
        axes[3, col].set_xlabel("z from axicon (mm)")
        axes[3, col].set_ylabel("y (µm)")

    fig.colorbar(im0, ax=axes[0, :].tolist(), label=records[0]["component_label"])
    fig.colorbar(im1, ax=axes[1, :].tolist(), label="I / sweep-global max")
    fig.colorbar(im2, ax=axes[2, :].tolist(), label="I / sweep-global max")
    fig.colorbar(im3, ax=axes[3, :].tolist(), label="I / sweep-global max")
    fig.suptitle(f"{case_id} — {family} | {spec['fidelity']}")
    path = figure_root / case_id / f"{family}_system_diagnostic.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run physically placed vortex/Bessel system-error sweeps."
    )
    parser.add_argument("--cases", nargs="+", default=["B0", "V1", "V3"])
    parser.add_argument("--families", nargs="+", default=["all"])
    parser.add_argument("--grid-n", type=int, default=1536)
    parser.add_argument("--z-mm", type=float, default=60.0)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/validation/vortex_system_errors"),
    )
    parser.add_argument(
        "--figure-root",
        type=Path,
        default=Path("outputs/figures/vortex_system_errors"),
    )
    args = parser.parse_args()

    registry = system_sweep_registry()
    families = list(registry) if args.families == ["all"] else list(args.families)
    unknown = [name for name in families if name not in registry]
    if unknown:
        raise SystemExit(f"unknown families: {unknown}")

    rows: list[dict[str, Any]] = []
    for case_id in args.cases:
        for family in families:
            print(f"running {case_id} / {family}", flush=True)
            rows.extend(
                run_family(
                    family,
                    case_id=case_id,
                    grid_n=int(args.grid_n),
                    z_reference_m=float(args.z_mm) * 1e-3,
                    output_root=args.output_root,
                    figure_root=args.figure_root,
                )
            )

    _write_csv(args.output_root / "system_error_metrics.csv", rows)
    manifest = {
        "outcome": "VORTEX-SYSTEM-ERROR-RESEARCH-SUITE",
        "report_figures_authorised": False,
        "cases": list(args.cases),
        "families": families,
        "grid_n": int(args.grid_n),
        "z_reference_mm": float(args.z_mm),
        "diagnostic_contract": (
            "component plane + xy + xz + yz; correlations, line-centre slope/"
            "intercept/width, axial modulation and rotated-plane numerical "
            "power bookkeeping are written to CSV"
        ),
        "executed_registry": {
            name: {k: v for k, v in registry[name].items() if k != "builder"}
            for name in families
        },
        "blocked_or_data_driven": blocked_or_data_driven_families(),
        "policy": (
            "Every executable family is a controlled relative-sensitivity study. "
            "Absolute bench-error magnitudes require the listed calibration data "
            "and independent validation gates. Axicon rigid tilt remains blocked "
            "for full refractive claims even when the scalar rotated-plane "
            "numerical gate passes."
        ),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "system_error_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()
