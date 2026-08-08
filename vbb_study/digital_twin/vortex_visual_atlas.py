"""Vortex Bessel visual-atlas sweeps built on the repaired Phase 2E route.

The atlas is diagnostic/reporting infrastructure.  It does not modify accepted
Phase 2A/2B/2C outputs.  Source-scale plots use the repaired no-additional-hard-
aperture Phase 2E route and explicit, labelled perturbations.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from vbb_study.digital_twin.phase2a_contracts import canonical_hardware_manifest, hardware_value
from vbb_study.digital_twin.phase2e_production_repair import build_nominal_source
from vbb_study.equations.propagation import angular_spectrum_propagate_bl


EPS = np.finfo(float).tiny
DEFAULT_WINDOW_M = 10e-3
DEFAULT_SCREENING_N = 1536
DEFAULT_REPORT_N = 3072
DEFAULT_Z_M = (20e-3, 40e-3, 60e-3, 80e-3, 100e-3)
DEFAULT_OUTPUT_ROOT = Path("outputs/validation/vortex_visual_atlas")
DEFAULT_FIGURE_ROOT = Path("outputs/figures/vortex_visual_atlas")


@dataclass(frozen=True)
class ZernikeSpec:
    name: str
    radial_order: int
    azimuthal_order: int
    orientation: str


ZERNIKE_REGISTRY: dict[str, ZernikeSpec] = {
    "defocus": ZernikeSpec("defocus", 2, 0, "rotationally_symmetric"),
    "astigmatism_x": ZernikeSpec("astigmatism_x", 2, 2, "cos_2theta"),
    "astigmatism_y": ZernikeSpec("astigmatism_y", 2, -2, "sin_2theta"),
    "coma_x": ZernikeSpec("coma_x", 3, 1, "cos_theta"),
    "coma_y": ZernikeSpec("coma_y", 3, -1, "sin_theta"),
    "spherical": ZernikeSpec("spherical", 4, 0, "rotationally_symmetric"),
}


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    values = [dict(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in values:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(values)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def zernike_mode(name: str, grid: Mapping[str, Any], radius_m: float) -> np.ndarray:
    """Return an RMS-normalised real Zernike-like mode inside a declared radius.

    The convention is explicit and self-contained for visual sensitivity studies;
    values are normalised numerically to unit RMS over rho<=1.
    """
    if name not in ZERNIKE_REGISTRY:
        raise ValueError(f"unknown Zernike mode {name!r}")
    x = np.asarray(grid["X"], dtype=float)
    y = np.asarray(grid["Y"], dtype=float)
    rho = np.hypot(x, y) / float(radius_m)
    theta = np.arctan2(y, x)
    if name == "defocus":
        raw = 2.0 * rho**2 - 1.0
    elif name == "astigmatism_x":
        raw = rho**2 * np.cos(2.0 * theta)
    elif name == "astigmatism_y":
        raw = rho**2 * np.sin(2.0 * theta)
    elif name == "coma_x":
        raw = (3.0 * rho**3 - 2.0 * rho) * np.cos(theta)
    elif name == "coma_y":
        raw = (3.0 * rho**3 - 2.0 * rho) * np.sin(theta)
    elif name == "spherical":
        raw = 6.0 * rho**4 - 6.0 * rho**2 + 1.0
    else:  # pragma: no cover
        raise AssertionError(name)
    mask = rho <= 1.0
    rms = float(np.sqrt(np.mean(np.square(raw[mask]))))
    return np.where(mask, raw / max(rms, EPS), 0.0)


def apply_zernike_waves(
    field: np.ndarray,
    grid: Mapping[str, Any],
    *,
    name: str,
    waves_rms: float,
    radius_m: float,
) -> np.ndarray:
    mode = zernike_mode(name, grid, radius_m)
    return np.asarray(field, dtype=np.complex128) * np.exp(1j * 2.0 * np.pi * float(waves_rms) * mode)


def build_atlas_source(
    case_id: str,
    *,
    grid_n: int,
    beam_radius_scale: float = 1.0,
    axicon_angle_scale: float = 1.0,
    axicon_decentre_m: tuple[float, float] = (0.0, 0.0),
    axicon_tilt_rad: tuple[float, float] = (0.0, 0.0),
    zernike_name: str | None = None,
    zernike_waves_rms: float = 0.0,
    aperture_model: str = "none",
) -> tuple[np.ndarray, Mapping[str, Any], dict[str, Any]]:
    """Create an explicitly perturbed source from the repaired Phase 2E baseline."""
    source, grid, meta = build_nominal_source(
        case_id,
        grid_n=int(grid_n),
        window_m=DEFAULT_WINDOW_M,
        aperture_model=aperture_model,
    )
    manifest = canonical_hardware_manifest()
    wavelength = float(meta["wavelength_m"])
    x = np.asarray(grid["X"], dtype=float)
    y = np.asarray(grid["Y"], dtype=float)

    # Beam-radius sensitivity is applied as an amplitude reweight relative to the
    # canonical Gaussian envelope so upstream SLM/4F phase content is retained.
    canonical_radius = float(hardware_value(manifest, "beam_radius_on_slm_m"))
    target_radius = canonical_radius * float(beam_radius_scale)
    r2 = x**2 + y**2
    canonical_env = np.exp(-r2 / canonical_radius**2)
    target_env = np.exp(-r2 / target_radius**2)
    field = source * target_env / np.maximum(canonical_env, 1e-12)

    # Re-map nominal axicon phase to a changed cone angle without re-running the
    # rest of the optical train.  Remove nominal radial phase, then apply perturbed.
    n_ax = float(hardware_value(manifest, "axicon_refractive_index"))
    n_m = float(hardware_value(manifest, "axicon_external_medium_index"))
    gamma0 = math.radians(float(hardware_value(manifest, "axicon_base_angle_deg")))
    k0 = 2.0 * math.pi / wavelength
    kr0 = k0 * (n_ax - n_m) * math.tan(gamma0)
    gamma = gamma0 * float(axicon_angle_scale)
    kr = k0 * (n_ax - n_m) * math.tan(gamma)
    ax, ay = axicon_decentre_m
    tx, ty = axicon_tilt_rad
    r0 = np.hypot(x, y)
    r1 = np.hypot(x - float(ax), y - float(ay))
    field *= np.exp(+1j * kr0 * r0)
    field *= np.exp(-1j * kr * r1)
    field *= np.exp(1j * k0 * (float(tx) * x + float(ty) * y))

    if zernike_name is not None and abs(float(zernike_waves_rms)) > 0.0:
        aberration_radius = 2.0 * canonical_radius
        field = apply_zernike_waves(
            field,
            grid,
            name=zernike_name,
            waves_rms=float(zernike_waves_rms),
            radius_m=aberration_radius,
        )

    metadata = {
        **meta,
        "atlas_route": "repaired_phase2e_source_scale",
        "beam_radius_scale": float(beam_radius_scale),
        "axicon_angle_scale": float(axicon_angle_scale),
        "axicon_decentre_x_m": float(ax),
        "axicon_decentre_y_m": float(ay),
        "axicon_tilt_x_rad": float(tx),
        "axicon_tilt_y_rad": float(ty),
        "zernike_name": zernike_name or "none",
        "zernike_waves_rms": float(zernike_waves_rms),
        "radial_wavevector_m_inv_atlas": float(kr),
    }
    return np.asarray(field, dtype=np.complex128), grid, metadata


def propagate_selected_planes(
    source: np.ndarray,
    grid: Mapping[str, Any],
    wavelength_m: float,
    z_values_m: Sequence[float] = DEFAULT_Z_M,
) -> list[np.ndarray]:
    planes: list[np.ndarray] = []
    for z_m in z_values_m:
        propagated = angular_spectrum_propagate_bl(
            source,
            dict(grid),
            float(wavelength_m),
            float(z_m),
            n_medium=1.0,
            bandlimit=True,
            include_evanescent=True,
        )
        planes.append(np.asarray(np.abs(propagated) ** 2, dtype=np.float32))
    return planes


def transverse_metrics(intensity: np.ndarray, grid: Mapping[str, Any]) -> dict[str, float]:
    I = np.asarray(intensity, dtype=float)
    total = float(np.sum(I))
    x = np.asarray(grid["X"], dtype=float)
    y = np.asarray(grid["Y"], dtype=float)
    cx = float(np.sum(I * x) / max(total, EPS))
    cy = float(np.sum(I * y) / max(total, EPS))
    peak = float(np.max(I))
    centre = int(I.shape[0]) // 2
    return {
        "power_au": total * float(grid["dx"]) ** 2,
        "peak_intensity_au": peak,
        "centroid_x_m": cx,
        "centroid_y_m": cy,
        "native_centre_intensity_ratio": float(I[centre, centre] / max(peak, EPS)),
    }


def parameter_registry() -> dict[str, Sequence[Any]]:
    return {
        "beam_radius_scale": (0.6, 0.8, 1.0, 1.2, 1.4),
        "axicon_angle_scale": (0.75, 0.875, 1.0, 1.125, 1.25),
        "aperture_model": ("none", "soft", "hard"),
    }


def aberration_registry() -> dict[str, Sequence[float]]:
    return {name: (-0.20, -0.10, 0.0, 0.10, 0.20) for name in ZERNIKE_REGISTRY}


def alignment_registry() -> dict[str, Sequence[float]]:
    return {
        "axicon_decentre_x_m": (-200e-6, -100e-6, 0.0, 100e-6, 200e-6),
        "axicon_tilt_x_rad": (-0.20e-3, -0.10e-3, 0.0, 0.10e-3, 0.20e-3),
    }


def run_atlas_screening(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    cases: Sequence[str] = ("B0", "V1", "V3"),
    grid_n: int = DEFAULT_SCREENING_N,
    z_m: float = 60e-3,
) -> dict[str, Any]:
    """Generate numerical sweep tables; report plots are generated separately."""
    rows: list[dict[str, Any]] = []
    for case_id in cases:
        for parameter, values in parameter_registry().items():
            for value in values:
                kwargs: dict[str, Any] = {parameter: value}
                source, grid, meta = build_atlas_source(case_id, grid_n=grid_n, **kwargs)
                plane = propagate_selected_planes(source, grid, float(meta["wavelength_m"]), (z_m,))[0]
                rows.append({"family": "parameter", "parameter": parameter, "value": value, "case_id": case_id, "grid_n": grid_n, "z_m": z_m, **transverse_metrics(plane, grid)})
        for mode, values in aberration_registry().items():
            for waves in values:
                source, grid, meta = build_atlas_source(case_id, grid_n=grid_n, zernike_name=mode, zernike_waves_rms=waves)
                plane = propagate_selected_planes(source, grid, float(meta["wavelength_m"]), (z_m,))[0]
                rows.append({"family": "aberration", "parameter": mode, "value": waves, "case_id": case_id, "grid_n": grid_n, "z_m": z_m, **transverse_metrics(plane, grid)})
        for parameter, values in alignment_registry().items():
            for value in values:
                kwargs = {}
                if parameter == "axicon_decentre_x_m":
                    kwargs["axicon_decentre_m"] = (float(value), 0.0)
                elif parameter == "axicon_tilt_x_rad":
                    kwargs["axicon_tilt_rad"] = (float(value), 0.0)
                source, grid, meta = build_atlas_source(case_id, grid_n=grid_n, **kwargs)
                plane = propagate_selected_planes(source, grid, float(meta["wavelength_m"]), (z_m,))[0]
                rows.append({"family": "alignment", "parameter": parameter, "value": value, "case_id": case_id, "grid_n": grid_n, "z_m": z_m, **transverse_metrics(plane, grid)})
    _write_csv(output_root / "atlas_screening_metrics.csv", rows)
    manifest = {
        "outcome": "VORTEX-VISUAL-ATLAS-SCREENING",
        "report_figures_authorised": False,
        "screening_grid_n": int(grid_n),
        "report_grid_n": DEFAULT_REPORT_N,
        "route": "repaired Phase 2E source-scale, nominal no additional hard aperture",
        "cases": list(cases),
        "parameter_registry": {k: list(v) for k, v in parameter_registry().items()},
        "aberration_registry": {k: list(v) for k, v in aberration_registry().items()},
        "alignment_registry": {k: list(v) for k, v in alignment_registry().items()},
        "zernike_convention": "numerically unit-RMS real modes inside radius=2*w0; phase=2*pi*waves_rms*Z",
        "next_step": "select representative sweep points and regenerate report-facing figures at N=3072",
    }
    _write_json(output_root / "atlas_screening_manifest.json", manifest)
    return manifest
