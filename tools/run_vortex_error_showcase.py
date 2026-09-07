from __future__ import annotations

"""High-resolution, presentation-oriented physical error showcase.

This runner is deliberately separate from the conservative screening/CI sweeps.
It exists to make the *consequence* of representative perturbations visually
legible without changing the underlying physical operators.

Every severity below is an UNMEASURED SENSITIVITY LEVEL.  The figures must not be
used to claim that the laboratory has any of these exact error magnitudes.

Two complementary views are saved:
  1. fixed laboratory coordinates + one common intensity scale per family;
     this preserves displacement, clipping and absolute throughput changes;
  2. centroid-recentred + own-peak normalisation; this exposes morphology changes
     after translation/brightness differences are factored out.

All intensity heatmaps use ``inferno`` by design.
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
from vbb_study.digital_twin.vortex_system_route import AxiconError, SystemErrorConfig, build_system_route
from vbb_study.digital_twin.vortex_wavefront_errors import zernike_opd_map_m
from vbb_study.equations.fields import make_xy_grid
from vbb_study.equations.propagation import angular_spectrum_propagate_bl

EPS = np.finfo(float).tiny
WINDOW_M = 10.0e-3
LAB_HALFWIDTH_M = 2.0e-3
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
        "beam_decentre_x",
        "Input-beam lateral decentre",
        (0.0, 0.75e-3, 1.50e-3),
        lambda v: f"{v * 1e3:.2f} mm",
        "translated Gaussian before SLM1; 2 mm nominal 1/e field-amplitude radius",
    ),
    FamilySpec(
        "fourf_iris_offset_x",
        "4F iris lateral offset",
        (0.0, 0.45e-3, 0.80e-3),
        lambda v: f"{v * 1e3:.2f} mm",
        "physical circular aperture translated in the propagated Fourier plane",
    ),
    FamilySpec(
        "slm_phase_stroke",
        "Dual-SLM phase-stroke underdrive",
        (1.0, 0.85, 0.70),
        lambda v: f"{100.0 * v:.0f}% commanded stroke",
        "sensitivity only until measured panel LUT/stroke is supplied",
    ),
    FamilySpec(
        "axicon_decentre_x",
        "Axicon lateral decentre",
        (0.0, 0.50e-3, 1.00e-3),
        lambda v: f"{v * 1e3:.2f} mm",
        "translated physical axicon sag/apex coordinates relative to the beam",
    ),
    FamilySpec(
        "axicon_round_tip_radius",
        "Rounded axicon apex",
        (0.0, 200e-6, 800e-6),
        lambda v: "sharp" if v == 0.0 else f"{v * 1e6:.0f} µm radius scale",
        "hyperboloidal rounded-tip sensitivity; radius converted to the route's sag parameter",
    ),
    FamilySpec(
        "lens1_astigmatism",
        "Declared L1 astigmatism",
        (0.0, 0.35, 0.70),
        lambda v: f"{v:.2f} waves RMS",
        "generic declared-plane Zernike OPD sensitivity, not a measured L1 wavefront",
    ),
    FamilySpec(
        "lens1_trefoil",
        "Declared L1 trefoil",
        (0.0, 0.35, 0.70),
        lambda v: f"{v:.2f} waves RMS",
        "generic declared-plane Zernike OPD sensitivity, not a measured L1 wavefront",
    ),
)


def _grid(grid_n: int) -> dict:
    return make_xy_grid(int(grid_n), WINDOW_M / int(grid_n))


def _config_and_maps(
    family: str,
    value: float,
    *,
    grid: dict,
    wavelength_m: float,
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
        # v3 convention: the displayed radial tip scale is converted to the
        # hyperboloidal sag parameter a = r_tip * tan(gamma).
        a = float(value) * math.tan(float(axicon_base_angle_rad))
        return SystemErrorConfig(
            axicon=AxiconError(tip_model="hyperboloidal_round", rounding_parameter_m=a)
        ), {}
    if family in {"lens1_astigmatism", "lens1_trefoil"}:
        name = "astigmatism_x" if family == "lens1_astigmatism" else "trefoil_x"
        opd = zernike_opd_map_m(
            name,
            grid,
            wavelength_m=float(wavelength_m),
            waves_rms=float(value),
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


def _crop(
    intensity: np.ndarray,
    x: np.ndarray,
    *,
    centre_x_m: float,
    centre_y_m: float,
    halfwidth_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ix = np.flatnonzero(np.abs(x - float(centre_x_m)) <= float(halfwidth_m))
    iy = np.flatnonzero(np.abs(x - float(centre_y_m)) <= float(halfwidth_m))
    if ix.size < 32 or iy.size < 32:
        raise RuntimeError(
            f"showcase crop is under-sampled: nx={ix.size}, ny={iy.size}, halfwidth={halfwidth_m:g} m"
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
    corr = float(np.corrcoef(ref.ravel(), cand.ravel())[0, 1])
    return {
        "rms_delta_over_nominal_peak": float(np.sqrt(np.mean(diff * diff)) / peak),
        "max_abs_delta_over_nominal_peak": float(np.max(np.abs(diff)) / peak),
        "intensity_correlation_to_nominal": corr,
        "peak_to_nominal_peak": float(np.max(cand) / peak),
        "power_to_nominal_power": float(np.sum(cand) / max(float(np.sum(ref)), EPS)),
    }


def _save_family_figure(
    records: list[dict],
    spec: FamilySpec,
    *,
    case_id: str,
    z_m: float,
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    common_peak = max(float(np.max(r["intensity"])) for r in records)
    x = np.asarray(records[0]["grid"]["x"], float)
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.7), constrained_layout=True)
    common_im = None
    shape_im = None
    for col, rec in enumerate(records):
        label = spec.value_label(float(rec["value"]))
        xl, yl, lab = _crop(
            rec["intensity"], x,
            centre_x_m=0.0, centre_y_m=0.0, halfwidth_m=LAB_HALFWIDTH_M,
        )
        common_im = axes[0, col].imshow(
            lab,
            origin="lower",
            extent=[xl[0] * 1e3, xl[-1] * 1e3, yl[0] * 1e3, yl[-1] * 1e3],
            cmap="inferno", vmin=0.0, vmax=common_peak,
        )
        axes[0, col].set_title(label)
        axes[0, col].set_xlabel("laboratory x (mm)")
        axes[0, col].set_ylabel("laboratory y (mm)")
        axes[0, col].axhline(0.0, linewidth=0.45, alpha=0.4)
        axes[0, col].axvline(0.0, linewidth=0.45, alpha=0.4)

        cx, cy = rec["centroid_m"]
        xs, ys, shape = _crop(
            rec["intensity"], x,
            centre_x_m=cx, centre_y_m=cy, halfwidth_m=SHAPE_HALFWIDTH_M,
        )
        shape = shape / max(float(np.max(shape)), EPS)
        shape_im = axes[1, col].imshow(
            shape,
            origin="lower",
            extent=[xs[0] * 1e6, xs[-1] * 1e6, ys[0] * 1e6, ys[-1] * 1e6],
            cmap="inferno", vmin=0.0, vmax=1.0,
        )
        axes[1, col].set_xlabel("x from energy centroid (µm)")
        axes[1, col].set_ylabel("y from energy centroid (µm)")
        axes[1, col].axhline(0.0, linewidth=0.45, alpha=0.4)
        axes[1, col].axvline(0.0, linewidth=0.45, alpha=0.4)

    if common_im is not None:
        fig.colorbar(common_im, ax=axes[0, :].tolist(), label="intensity (common family scale)")
    if shape_im is not None:
        fig.colorbar(shape_im, ax=axes[1, :].tolist(), label="I / own peak")
    fig.suptitle(
        f"{case_id} — {spec.title} — z = {z_m * 1e3:.0f} mm\n"
        "top: fixed lab frame + common absolute scale | bottom: centroid-recentred + own-peak scale\n"
        "UNMEASURED SENSITIVITY STUDY — not a claim about bench error magnitude",
        fontsize=12,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{spec.key}_inferno_showcase.png", dpi=320, bbox_inches="tight")
    fig.savefig(output_dir / f"{spec.key}_inferno_showcase.svg", bbox_inches="tight")
    plt.close(fig)


def _save_combined(records_by_family: dict[str, list[dict]], *, case_id: str, z_m: float,
                   output_dir: Path, mode: str) -> None:
    import matplotlib.pyplot as plt

    nrows = len(FAMILIES)
    fig, axes = plt.subplots(nrows, 3, figsize=(13.0, 3.45 * nrows), constrained_layout=True)
    for row, spec in enumerate(FAMILIES):
        records = records_by_family[spec.key]
        x = np.asarray(records[0]["grid"]["x"], float)
        common_peak = max(float(np.max(r["intensity"])) for r in records)
        for col, rec in enumerate(records):
            if mode == "lab_common":
                xx, yy, image = _crop(
                    rec["intensity"], x,
                    centre_x_m=0.0, centre_y_m=0.0, halfwidth_m=LAB_HALFWIDTH_M,
                )
                extent = [xx[0] * 1e3, xx[-1] * 1e3, yy[0] * 1e3, yy[-1] * 1e3]
                image = image / max(common_peak, EPS)
                xlabel = "lab x (mm)"
                ylabel = "lab y (mm)"
            elif mode == "shape":
                cx, cy = rec["centroid_m"]
                xx, yy, image = _crop(
                    rec["intensity"], x,
                    centre_x_m=cx, centre_y_m=cy, halfwidth_m=SHAPE_HALFWIDTH_M,
                )
                extent = [xx[0] * 1e6, xx[-1] * 1e6, yy[0] * 1e6, yy[-1] * 1e6]
                image = image / max(float(np.max(image)), EPS)
                xlabel = "Δx (µm)"
                ylabel = "Δy (µm)"
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
        else "centroid-recentred; each panel normalised to its own peak"
    )
    fig.suptitle(
        f"{case_id} physical-error sensitivity showcase — z = {z_m * 1e3:.0f} mm\n"
        f"{descriptor} — inferno\nUNMEASURED SENSITIVITY LEVELS",
        fontsize=13,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "inferno_error_showcase_common_scale" if mode == "lab_common" else "inferno_error_showcase_shape_normalised"
    fig.savefig(output_dir / f"{case_id}_{stem}.png", dpi=320, bbox_inches="tight")
    fig.savefig(output_dir / f"{case_id}_{stem}.svg", bbox_inches="tight")
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
            cx, cy = _centroid(intensity, grid)
            records.append({
                "value": float(value), "intensity": intensity, "grid": grid,
                "centroid_m": (cx, cy), "metadata": route["metadata"],
            })

        nominal = records[0]["intensity"]
        nominal_cx, nominal_cy = records[0]["centroid_m"]
        for rec in records:
            dm = _difference_metrics(nominal, rec["intensity"])
            cx, cy = rec["centroid_m"]
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
                "centroid_shift_from_nominal_m": float(math.hypot(cx - nominal_cx, cy - nominal_cy)),
                **dm,
            })
        records_by_family[spec.key] = records
        _save_family_figure(
            records, spec, case_id=case_id, z_m=z_m,
            output_dir=output_root / "figures" / case_id,
        )

    _save_combined(records_by_family, case_id=case_id, z_m=z_m,
                   output_dir=output_root / "figures" / case_id, mode="lab_common")
    _save_combined(records_by_family, case_id=case_id, z_m=z_m,
                   output_dir=output_root / "figures" / case_id, mode="shape")

    validation_dir = output_root / "validation" / case_id
    validation_dir.mkdir(parents=True, exist_ok=True)
    csv_path = validation_dir / "showcase_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metrics_rows)

    manifest_out = {
        "outcome": "VORTEX-ERROR-HIRES-INFERNO-SHOWCASE",
        "case_id": case_id,
        "grid_n": int(grid_n),
        "z_m": float(z_m),
        "heatmap": "inferno",
        "lab_view": "fixed lab frame; common scale per family",
        "shape_view": "centroid-recentred; own-peak normalised",
        "provenance": "all severities are unmeasured sensitivity levels",
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
    (validation_dir / "showcase_manifest.json").write_text(
        json.dumps(manifest_out, indent=2), encoding="utf-8"
    )
    return manifest_out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("B0", "V1", "V3"), default="V3")
    parser.add_argument("--grid-n", type=int, default=1024)
    parser.add_argument("--z-mm", type=float, default=60.0)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/error_showcase_hires"))
    args = parser.parse_args()
    if args.grid_n < 768:
        raise SystemExit("presentation showcase requires grid_n >= 768")
    manifest = run(
        args.case,
        grid_n=int(args.grid_n),
        z_m=float(args.z_mm) * 1e-3,
        output_root=args.output_root,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
