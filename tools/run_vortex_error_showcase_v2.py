from __future__ import annotations

"""High-resolution inferno error showcase, morphology-axis corrected.

This is a presentation/sensitivity runner, not a bench-calibrated prediction.
Every severity is explicitly unmeasured.  The optical operators themselves are
the canonical physical-route operators.

Two views are emitted for every error family:
  * fixed laboratory coordinates with one common intensity scale per family;
  * Bessel/vortex morphology-axis-recentred with each panel normalised to its
    own peak.  For vortex cases the morphology axis is found from phase winding,
    not from the global intensity centroid.
"""

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from vbb_study.digital_twin.phase2a_contracts import canonical_hardware_manifest, hardware_value
from vbb_study.digital_twin.vortex_beam_slm_errors import GaussianBeamError, SLMError
from vbb_study.digital_twin.vortex_explicit_4f import FourFError
from vbb_study.digital_twin.vortex_following_propagation import transverse_morphology_axis
from vbb_study.digital_twin.vortex_system_route import AxiconError, SystemErrorConfig, build_system_route
from vbb_study.digital_twin.vortex_wavefront_errors import zernike_opd_map_m
from vbb_study.equations.fields import make_xy_grid
from vbb_study.equations.propagation import angular_spectrum_propagate_bl

EPS = np.finfo(float).tiny
WINDOW_M = 10.0e-3
SHAPE_HALFWIDTH_M = 0.55e-3
WAVEFRONT_SUPPORT_RADIUS_M = 2.5e-3


@dataclass(frozen=True)
class FamilySpec:
    key: str
    title: str
    values: tuple[float, float, float]
    value_label: Callable[[float], str]
    provenance: str


FAMILIES: tuple[FamilySpec, ...] = (
    FamilySpec(
        "beam_decentre_x", "Input-beam lateral decentre",
        (0.0, 0.75e-3, 1.50e-3), lambda v: f"{v * 1e3:.2f} mm",
        "translated Gaussian before SLM1; nominal beam radius is 2 mm",
    ),
    FamilySpec(
        "fourf_iris_offset_x", "4F iris lateral offset",
        (0.0, 0.45e-3, 0.80e-3), lambda v: f"{v * 1e3:.2f} mm",
        "physical circular aperture translated in the propagated Fourier plane",
    ),
    FamilySpec(
        "slm_phase_stroke", "Dual-SLM phase-stroke underdrive",
        (1.0, 0.85, 0.70), lambda v: f"{100.0 * v:.0f}% commanded stroke",
        "sensitivity only until measured panel LUT/stroke is supplied",
    ),
    FamilySpec(
        "axicon_decentre_x", "Axicon lateral decentre",
        (0.0, 0.50e-3, 1.00e-3), lambda v: f"{v * 1e3:.2f} mm",
        "translated physical axicon sag/apex coordinates relative to the beam",
    ),
    FamilySpec(
        "axicon_round_tip_radius", "Rounded axicon apex",
        (0.0, 200e-6, 800e-6),
        lambda v: "sharp" if v == 0.0 else f"{v * 1e6:.0f} µm radius scale",
        "hyperboloidal rounded-tip sensitivity; radius converted to the route sag parameter",
    ),
    FamilySpec(
        "lens1_astigmatism", "Declared L1 astigmatism",
        (0.0, 0.35, 0.70), lambda v: f"{v:.2f} waves RMS",
        "generic declared-plane Zernike OPD sensitivity, not a measured L1 wavefront",
    ),
    FamilySpec(
        "lens1_trefoil", "Declared L1 trefoil",
        (0.0, 0.35, 0.70), lambda v: f"{v:.2f} waves RMS",
        "generic declared-plane Zernike OPD sensitivity, not a measured L1 wavefront",
    ),
)


def _ell(case_id: str) -> int:
    return {"B0": 0, "V1": 1, "V3": 3}[case_id]


def _grid(grid_n: int) -> dict:
    return make_xy_grid(int(grid_n), WINDOW_M / int(grid_n))


def _config_and_maps(
    family: str, value: float, *, grid: dict, wavelength_m: float,
    axicon_base_angle_rad: float,
) -> tuple[SystemErrorConfig, dict[str, np.ndarray]]:
    if family == "beam_decentre_x":
        return SystemErrorConfig(beam=GaussianBeamError(decentre_m=(float(value), 0.0))), {}
    if family == "fourf_iris_offset_x":
        return SystemErrorConfig(fourf=FourFError(iris_offset_m=(float(value), 0.0))), {}
    if family == "slm_phase_stroke":
        err = SLMError(phase_stroke_scale=float(value))
        return SystemErrorConfig(slm1=err, slm2=err), {}
    if family == "axicon_decentre_x":
        return SystemErrorConfig(axicon=AxiconError(decentre_m=(float(value), 0.0))), {}
    if family == "axicon_round_tip_radius":
        if float(value) == 0.0:
            return SystemErrorConfig(axicon=AxiconError(tip_model="sharp")), {}
        a = float(value) * math.tan(float(axicon_base_angle_rad))
        return SystemErrorConfig(
            axicon=AxiconError(tip_model="hyperboloidal_round", rounding_parameter_m=a)
        ), {}
    if family in {"lens1_astigmatism", "lens1_trefoil"}:
        name = "astigmatism_x" if family == "lens1_astigmatism" else "trefoil_x"
        opd = zernike_opd_map_m(
            name, grid, wavelength_m=float(wavelength_m), waves_rms=float(value),
            pupil_radius_m=WAVEFRONT_SUPPORT_RADIUS_M,
        )
        return SystemErrorConfig(), {"lens1_opd_map_m": opd}
    raise KeyError(family)


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


def _crop(
    intensity: np.ndarray, x: np.ndarray, *, centre_x_m: float, centre_y_m: float,
    halfwidth_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ix = np.flatnonzero(np.abs(x - float(centre_x_m)) <= float(halfwidth_m))
    iy = np.flatnonzero(np.abs(x - float(centre_y_m)) <= float(halfwidth_m))
    if ix.size < 32 or iy.size < 32:
        raise RuntimeError(
            f"showcase crop is under-sampled: nx={ix.size}, ny={iy.size}, halfwidth={halfwidth_m:g} m"
        )
    return (
        x[ix] - float(centre_x_m), x[iy] - float(centre_y_m),
        np.asarray(intensity)[np.ix_(iy, ix)],
    )


def _family_lab_halfwidth(records: list[dict]) -> float:
    max_axis = max(
        max(abs(float(r["morphology_axis_m"][0])), abs(float(r["morphology_axis_m"][1])))
        for r in records
    )
    return float(np.clip(max(0.75e-3, max_axis + 0.65e-3), 0.75e-3, 2.0e-3))


def _difference_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    ref = np.asarray(reference, float)
    cand = np.asarray(candidate, float)
    peak = max(float(np.max(ref)), EPS)
    diff = cand - ref
    return {
        "rms_delta_over_nominal_peak": float(np.sqrt(np.mean(diff * diff)) / peak),
        "max_abs_delta_over_nominal_peak": float(np.max(np.abs(diff)) / peak),
        "intensity_correlation_to_nominal": float(np.corrcoef(ref.ravel(), cand.ravel())[0, 1]),
        "peak_to_nominal_peak": float(np.max(cand) / peak),
        "power_to_nominal_power": float(np.sum(cand) / max(float(np.sum(ref)), EPS)),
    }


def _save_family_figure(
    records: list[dict], spec: FamilySpec, *, case_id: str, z_m: float, output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    common_peak = max(float(np.max(r["intensity"])) for r in records)
    lab_halfwidth = _family_lab_halfwidth(records)
    x = np.asarray(records[0]["grid"]["x"], float)
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.7), constrained_layout=True)
    common_im = None
    shape_im = None

    for col, rec in enumerate(records):
        label = spec.value_label(float(rec["value"]))
        xl, yl, lab = _crop(
            rec["intensity"], x, centre_x_m=0.0, centre_y_m=0.0,
            halfwidth_m=lab_halfwidth,
        )
        common_im = axes[0, col].imshow(
            lab, origin="lower",
            extent=[xl[0] * 1e3, xl[-1] * 1e3, yl[0] * 1e3, yl[-1] * 1e3],
            cmap="inferno", vmin=0.0, vmax=common_peak,
        )
        axes[0, col].set_title(label)
        axes[0, col].set_xlabel("laboratory x (mm)")
        axes[0, col].set_ylabel("laboratory y (mm)")
        axes[0, col].axhline(0.0, color="white", linewidth=0.45, alpha=0.35)
        axes[0, col].axvline(0.0, color="white", linewidth=0.45, alpha=0.35)

        axx, axy = rec["morphology_axis_m"]
        xs, ys, shape = _crop(
            rec["intensity"], x, centre_x_m=axx, centre_y_m=axy,
            halfwidth_m=SHAPE_HALFWIDTH_M,
        )
        shape = shape / max(float(np.max(shape)), EPS)
        shape_im = axes[1, col].imshow(
            shape, origin="lower",
            extent=[xs[0] * 1e6, xs[-1] * 1e6, ys[0] * 1e6, ys[-1] * 1e6],
            cmap="inferno", vmin=0.0, vmax=1.0,
        )
        axes[1, col].set_xlabel("x from morphology axis (µm)")
        axes[1, col].set_ylabel("y from morphology axis (µm)")
        axes[1, col].axhline(0.0, color="white", linewidth=0.45, alpha=0.35)
        axes[1, col].axvline(0.0, color="white", linewidth=0.45, alpha=0.35)

    if common_im is not None:
        fig.colorbar(common_im, ax=axes[0, :].tolist(), label="intensity (common family scale)")
    if shape_im is not None:
        fig.colorbar(shape_im, ax=axes[1, :].tolist(), label="I / own peak")
    fig.suptitle(
        f"{case_id} — {spec.title} — z = {z_m * 1e3:.0f} mm\n"
        "top: fixed lab frame + common absolute scale | bottom: morphology-axis-recentred + own-peak scale\n"
        "UNMEASURED SENSITIVITY STUDY — not a claim about bench error magnitude",
        fontsize=12,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{spec.key}_inferno_showcase_v2.png", dpi=320, bbox_inches="tight")
    fig.savefig(output_dir / f"{spec.key}_inferno_showcase_v2.svg", bbox_inches="tight")
    plt.close(fig)


def _save_combined(
    records_by_family: dict[str, list[dict]], *, case_id: str, z_m: float,
    output_dir: Path, mode: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(FAMILIES), 3, figsize=(13.0, 3.45 * len(FAMILIES)), constrained_layout=True)
    for row, spec in enumerate(FAMILIES):
        records = records_by_family[spec.key]
        x = np.asarray(records[0]["grid"]["x"], float)
        common_peak = max(float(np.max(r["intensity"])) for r in records)
        lab_halfwidth = _family_lab_halfwidth(records)
        for col, rec in enumerate(records):
            if mode == "lab_common":
                xx, yy, image = _crop(
                    rec["intensity"], x, centre_x_m=0.0, centre_y_m=0.0,
                    halfwidth_m=lab_halfwidth,
                )
                extent = [xx[0] * 1e3, xx[-1] * 1e3, yy[0] * 1e3, yy[-1] * 1e3]
                image = image / max(common_peak, EPS)
                xlabel, ylabel = "lab x (mm)", "lab y (mm)"
            elif mode == "shape":
                axx, axy = rec["morphology_axis_m"]
                xx, yy, image = _crop(
                    rec["intensity"], x, centre_x_m=axx, centre_y_m=axy,
                    halfwidth_m=SHAPE_HALFWIDTH_M,
                )
                extent = [xx[0] * 1e6, xx[-1] * 1e6, yy[0] * 1e6, yy[-1] * 1e6]
                image = image / max(float(np.max(image)), EPS)
                xlabel, ylabel = "Δx (µm)", "Δy (µm)"
            else:
                raise ValueError(mode)

            axes[row, col].imshow(
                image, origin="lower", extent=extent, cmap="inferno", vmin=0.0, vmax=1.0
            )
            axes[row, col].set_title(spec.value_label(float(rec["value"])), fontsize=9)
            axes[row, col].set_xlabel(xlabel, fontsize=8)
            axes[row, col].set_ylabel(ylabel, fontsize=8)
            if col == 0:
                axes[row, col].text(
                    -0.30, 0.5, spec.title, transform=axes[row, col].transAxes,
                    rotation=90, va="center", ha="center", fontsize=9, fontweight="bold",
                )

    descriptor = (
        "fixed laboratory frame; common intensity scale within each error family"
        if mode == "lab_common"
        else "topological/morphology-axis-recentred; each panel normalised to its own peak"
    )
    fig.suptitle(
        f"{case_id} physical-error sensitivity showcase — z = {z_m * 1e3:.0f} mm\n"
        f"{descriptor} — inferno\nUNMEASURED SENSITIVITY LEVELS",
        fontsize=13,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "common_scale_v2" if mode == "lab_common" else "morphology_normalised_v2"
    fig.savefig(output_dir / f"{case_id}_inferno_error_showcase_{stem}.png", dpi=320, bbox_inches="tight")
    fig.savefig(output_dir / f"{case_id}_inferno_error_showcase_{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def run(case_id: str, *, grid_n: int, z_m: float, output_root: Path) -> dict:
    manifest = canonical_hardware_manifest()
    wavelength = float(hardware_value(manifest, "wavelength_m"))
    gamma = math.radians(float(hardware_value(manifest, "axicon_base_angle_deg")))
    grid_for_maps = _grid(grid_n)
    records_by_family: dict[str, list[dict]] = {}
    metrics_rows: list[dict] = []

    for spec in FAMILIES:
        print(f"[{case_id}] {spec.key}", flush=True)
        records: list[dict] = []
        for value in spec.values:
            config, maps = _config_and_maps(
                spec.key, float(value), grid=grid_for_maps,
                wavelength_m=wavelength, axicon_base_angle_rad=gamma,
            )
            route = build_system_route(case_id, grid_n=grid_n, config=config, **maps)
            grid = dict(route["grid"])
            field = angular_spectrum_propagate_bl(
                route["post_axicon"], grid, wavelength, float(z_m), bandlimit=True
            )
            intensity = np.abs(field) ** 2
            centroid = _centroid(intensity, grid)
            seed = _axis_seed(spec.key, float(value), centroid)
            axis = transverse_morphology_axis(
                field, grid, vortex_charge=_ell(case_id),
                seed_x_m=seed[0], seed_y_m=seed[1], search_radius_m=1.25e-3,
            )
            records.append({
                "value": float(value), "field": field, "intensity": intensity,
                "grid": grid, "centroid_m": centroid,
                "morphology_axis_m": (float(axis.x_m), float(axis.y_m)),
                "axis_method": axis.method,
                "detected_topological_charge": int(axis.detected_topological_charge),
                "selected_singularity_count": int(axis.selected_singularity_count),
            })

        nominal = records[0]
        nominal_intensity = nominal["intensity"]
        ncx, ncy = nominal["centroid_m"]
        nax, nay = nominal["morphology_axis_m"]
        for rec in records:
            cx, cy = rec["centroid_m"]
            axx, axy = rec["morphology_axis_m"]
            metrics_rows.append({
                "case_id": case_id,
                "family": spec.key,
                "family_title": spec.title,
                "value": rec["value"],
                "value_label": spec.value_label(rec["value"]),
                "provenance": spec.provenance,
                "grid_n": int(grid_n),
                "dx_m": float(rec["grid"]["dx"]),
                "z_m": float(z_m),
                "centroid_x_m": float(cx),
                "centroid_y_m": float(cy),
                "centroid_shift_from_nominal_m": float(math.hypot(cx - ncx, cy - ncy)),
                "morphology_axis_x_m": float(axx),
                "morphology_axis_y_m": float(axy),
                "morphology_axis_shift_from_nominal_m": float(math.hypot(axx - nax, axy - nay)),
                "axis_method": rec["axis_method"],
                "detected_topological_charge": rec["detected_topological_charge"],
                "selected_singularity_count": rec["selected_singularity_count"],
                **_difference_metrics(nominal_intensity, rec["intensity"]),
            })

        records_by_family[spec.key] = records
        _save_family_figure(
            records, spec, case_id=case_id, z_m=z_m,
            output_dir=output_root / "figures" / case_id,
        )

    _save_combined(
        records_by_family, case_id=case_id, z_m=z_m,
        output_dir=output_root / "figures" / case_id, mode="lab_common",
    )
    _save_combined(
        records_by_family, case_id=case_id, z_m=z_m,
        output_dir=output_root / "figures" / case_id, mode="shape",
    )

    validation_dir = output_root / "validation" / case_id
    validation_dir.mkdir(parents=True, exist_ok=True)
    with (validation_dir / "showcase_metrics_v2.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metrics_rows)

    manifest_out = {
        "outcome": "VORTEX-ERROR-HIRES-INFERNO-SHOWCASE-V2",
        "case_id": case_id,
        "grid_n": int(grid_n),
        "z_m": float(z_m),
        "heatmap": "inferno",
        "lab_view": "fixed lab frame; common scale per family; adaptive crop",
        "shape_view": "topological/morphology-axis-recentred; own-peak normalised",
        "provenance": "all severities are unmeasured sensitivity levels",
        "families": [
            {
                "key": spec.key, "title": spec.title,
                "values": list(spec.values),
                "labels": [spec.value_label(v) for v in spec.values],
                "physical_basis": spec.provenance,
            }
            for spec in FAMILIES
        ],
    }
    (validation_dir / "showcase_manifest_v2.json").write_text(
        json.dumps(manifest_out, indent=2), encoding="utf-8"
    )
    return manifest_out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("B0", "V1", "V3"), default="V3")
    parser.add_argument("--grid-n", type=int, default=1024)
    parser.add_argument("--z-mm", type=float, default=60.0)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/error_showcase_hires_v2"))
    args = parser.parse_args()
    if int(args.grid_n) < 768:
        raise SystemExit("presentation showcase requires grid_n >= 768")
    manifest = run(
        args.case, grid_n=int(args.grid_n), z_m=float(args.z_mm) * 1e-3,
        output_root=args.output_root,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
