"""Vortex-Bessel visual atlas routed through the canonical physical system model.

The atlas distinguishes physical component/input errors from generic wavefront
sensitivity. Physical errors are built with ``vortex_system_route`` so they act
at their actual optical planes. Generic Zernikes reuse
``vortex_wavefront_errors`` and are explicitly applied at a declared axicon-exit
plane; they are not surrogates for a named misaligned optic.

All values in the registries are sensitivity values, not measurements. Atlas
outputs remain research diagnostics until the corresponding calibration and
reference checks pass.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from vbb_study.digital_twin.phase2a_contracts import canonical_hardware_manifest, hardware_value
from vbb_study.digital_twin.vortex_beam_slm_errors import GaussianBeamError, SLMError
from vbb_study.digital_twin.vortex_explicit_4f import FourFError
from vbb_study.digital_twin.vortex_system_route import (
    AxiconError,
    DEFAULT_WINDOW_M,
    SystemErrorConfig,
    build_system_route,
)
from vbb_study.digital_twin.vortex_wavefront_errors import unit_rms_zernike
from vbb_study.equations.propagation import angular_spectrum_propagate_bl

EPS = np.finfo(float).tiny
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
    "trefoil_x": ZernikeSpec("trefoil_x", 3, 3, "cos_3theta"),
    "trefoil_y": ZernikeSpec("trefoil_y", 3, -3, "sin_3theta"),
    "quadrafoil_x": ZernikeSpec("quadrafoil_x", 4, 4, "cos_4theta"),
    "quadrafoil_y": ZernikeSpec("quadrafoil_y", 4, -4, "sin_4theta"),
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
        writer.writeheader(); writer.writerows(values)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def zernike_mode(name: str, grid: Mapping[str, Any], radius_m: float) -> np.ndarray:
    """Return the canonical unit-RMS declared-plane Zernike mode."""
    if name not in ZERNIKE_REGISTRY:
        raise ValueError(f"unknown Zernike mode {name!r}")
    return unit_rms_zernike(name, grid, pupil_radius_m=float(radius_m))


def apply_zernike_waves(field: np.ndarray, grid: Mapping[str, Any], *, name: str,
                         waves_rms: float, radius_m: float) -> np.ndarray:
    """Apply a generic OPD sensitivity phase at the declared axicon-exit plane."""
    mode = zernike_mode(name, grid, radius_m)
    return np.asarray(field, np.complex128) * np.exp(1j * 2.0 * np.pi * float(waves_rms) * mode)


def _canonical_config(
    *,
    beam_radius_scale: float,
    input_beam_decentre_m: tuple[float, float],
    input_beam_angle_rad: tuple[float, float],
    hologram_decentre_m: tuple[float, float],
    fourier_iris_offset_fraction: float,
    axicon_angle_scale: float,
    axicon_decentre_m: tuple[float, float],
    axicon_tilt_rad: tuple[float, float],
    axicon_tip_model: str,
    axicon_rounding_parameter_m: float,
    axicon_flat_tip_radius_m: float,
) -> SystemErrorConfig:
    manifest = canonical_hardware_manifest()
    iris_radius_m = float(hardware_value(manifest, "fourier_iris_radius_m"))
    return SystemErrorConfig(
        beam=GaussianBeamError(
            radius_x_scale=float(beam_radius_scale),
            radius_y_scale=float(beam_radius_scale),
            decentre_m=tuple(map(float, input_beam_decentre_m)),
            pointing_rad=tuple(map(float, input_beam_angle_rad)),
        ),
        slm1=SLMError(pattern_offset_m=tuple(map(float, hologram_decentre_m))),
        fourf=FourFError(
            iris_offset_m=(float(fourier_iris_offset_fraction) * iris_radius_m, 0.0)
        ),
        axicon=AxiconError(
            base_angle_scale=float(axicon_angle_scale),
            decentre_m=tuple(map(float, axicon_decentre_m)),
            tilt_rad=tuple(map(float, axicon_tilt_rad)),
            tip_model=str(axicon_tip_model),
            rounding_parameter_m=float(axicon_rounding_parameter_m),
            flat_tip_radius_m=float(axicon_flat_tip_radius_m),
        ),
    )


def build_atlas_source(
    case_id: str,
    *,
    grid_n: int,
    beam_radius_scale: float = 1.0,
    input_beam_decentre_m: tuple[float, float] = (0.0, 0.0),
    input_beam_angle_rad: tuple[float, float] = (0.0, 0.0),
    hologram_decentre_m: tuple[float, float] = (0.0, 0.0),
    fourier_iris_offset_fraction: float = 0.0,
    axicon_angle_scale: float = 1.0,
    axicon_decentre_m: tuple[float, float] = (0.0, 0.0),
    axicon_tilt_rad: tuple[float, float] = (0.0, 0.0),
    axicon_tip_model: str = "sharp",
    axicon_rounding_parameter_m: float = 0.0,
    axicon_flat_tip_radius_m: float = 0.0,
    zernike_name: str | None = None,
    zernike_waves_rms: float = 0.0,
) -> tuple[np.ndarray, Mapping[str, Any], dict[str, Any]]:
    """Build an atlas source through the canonical physical route."""
    config = _canonical_config(
        beam_radius_scale=beam_radius_scale,
        input_beam_decentre_m=input_beam_decentre_m,
        input_beam_angle_rad=input_beam_angle_rad,
        hologram_decentre_m=hologram_decentre_m,
        fourier_iris_offset_fraction=fourier_iris_offset_fraction,
        axicon_angle_scale=axicon_angle_scale,
        axicon_decentre_m=axicon_decentre_m,
        axicon_tilt_rad=axicon_tilt_rad,
        axicon_tip_model=axicon_tip_model,
        axicon_rounding_parameter_m=axicon_rounding_parameter_m,
        axicon_flat_tip_radius_m=axicon_flat_tip_radius_m,
    )
    route = build_system_route(
        case_id, grid_n=int(grid_n), config=config, window_m=DEFAULT_WINDOW_M
    )
    grid = route["grid"]
    source = np.asarray(route["post_axicon"], np.complex128)
    raw_meta = dict(route["metadata"])

    if zernike_name is not None and abs(float(zernike_waves_rms)) > 0.0:
        source = apply_zernike_waves(
            source, grid, name=zernike_name, waves_rms=float(zernike_waves_rms),
            radius_m=2.0e-3,
        )
        generic_name = str(zernike_name)
        generic_waves = float(zernike_waves_rms)
    else:
        generic_name = "none"; generic_waves = 0.0

    beam_meta = dict(raw_meta["beam"])
    ax_meta = dict(raw_meta["axicon"])
    fourf_meta = dict(raw_meta["fourf"])
    meta = {
        **raw_meta,
        "route_id": "vortex_explicit_system_error_route_v1",
        "wavelength_m": float(raw_meta["wavelength_m"]),
        "input_beam_angle_rad": tuple(beam_meta["beam_pointing_rad"]),
        "input_angle_applied_plane": "before_SLM1",
        "input_beam_decentre_m": tuple(beam_meta["beam_decentre_m"]),
        "axicon_tilt_rad": tuple(raw_meta["axicon_rigid_tilt_rad"]),
        "axicon_tilt_model": str(raw_meta["axicon_tilt_status"]),
        "full_vector_snell_fresnel": False,
        "axicon_tip_model": str(ax_meta["tip_model"]),
        "rounding_parameter_m": float(ax_meta["rounding_parameter_m"]),
        "flat_tip_radius_m": float(ax_meta["flat_tip_radius_m"]),
        "fourier_iris_selected_power_fraction": float(fourf_meta["iris_selected_power_fraction"]),
        "additional_objective_pupil_application_count": 0,
        "generic_wavefront_aberration": generic_name,
        "generic_wavefront_waves_rms": generic_waves,
        "generic_wavefront_application_plane": "axicon_exit_plane",
        "atlas_evidence_status": "research_sensitivity_not_measured_validation",
    }
    return source, grid, meta


def propagate_selected_planes(source: np.ndarray, grid: Mapping[str, Any], wavelength_m: float,
                              z_values_m: Sequence[float] = DEFAULT_Z_M) -> list[np.ndarray]:
    planes: list[np.ndarray] = []
    for z_m in z_values_m:
        propagated = angular_spectrum_propagate_bl(
            source, dict(grid), float(wavelength_m), float(z_m), n_medium=1.0,
            bandlimit=True, include_evanescent=True,
        )
        planes.append(np.asarray(np.abs(propagated) ** 2, np.float32))
    return planes


def transverse_metrics(intensity: np.ndarray, grid: Mapping[str, Any]) -> dict[str, float]:
    I = np.asarray(intensity, float); total = float(np.sum(I))
    x = np.asarray(grid["X"], float); y = np.asarray(grid["Y"], float)
    cx = float(np.sum(I * x) / max(total, EPS)); cy = float(np.sum(I * y) / max(total, EPS))
    peak = float(np.max(I)); centre = int(I.shape[0]) // 2
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
        "input_beam_angle_x_rad": (-1e-3, -0.5e-3, 0.0, 0.5e-3, 1e-3),
        "input_beam_decentre_x_m": (-400e-6, -200e-6, 0.0, 200e-6, 400e-6),
    }


def manufacturing_defect_registry() -> dict[str, Sequence[Any]]:
    return {
        "axicon_rounding_parameter_m": (0.0, 2e-6, 5e-6, 10e-6, 20e-6),
        "axicon_flat_tip_radius_m": (0.0, 10e-6, 25e-6, 50e-6, 100e-6),
    }


def aberration_registry() -> dict[str, Sequence[float]]:
    return {name: (-0.50, -0.25, 0.0, 0.25, 0.50) for name in ZERNIKE_REGISTRY}


def alignment_registry() -> dict[str, Sequence[float]]:
    return {
        "axicon_decentre_x_m": (-400e-6, -200e-6, 0.0, 200e-6, 400e-6),
        "axicon_tilt_y_rad": tuple(np.deg2rad([-0.5, -0.25, 0.0, 0.25, 0.5])),
        "hologram_decentre_x_m": (-200e-6, -100e-6, 0.0, 100e-6, 200e-6),
        "fourier_iris_offset_fraction": (-0.6, -0.3, 0.0, 0.3, 0.6),
    }


def _source_kwargs(family: str, parameter: str, value: Any) -> dict[str, Any]:
    if family == "parameter":
        if parameter == "input_beam_angle_x_rad": return {"input_beam_angle_rad": (float(value), 0.0)}
        if parameter == "input_beam_decentre_x_m": return {"input_beam_decentre_m": (float(value), 0.0)}
        return {parameter: value}
    if family == "manufacturing":
        if parameter == "axicon_rounding_parameter_m":
            return {"axicon_tip_model": "sharp" if float(value) == 0.0 else "hyperboloidal_round",
                    "axicon_rounding_parameter_m": float(value)}
        if parameter == "axicon_flat_tip_radius_m":
            return {"axicon_tip_model": "sharp" if float(value) == 0.0 else "flat_blunt",
                    "axicon_flat_tip_radius_m": float(value)}
    if family == "aberration":
        return {"zernike_name": parameter, "zernike_waves_rms": float(value)}
    if family == "alignment":
        if parameter == "axicon_decentre_x_m": return {"axicon_decentre_m": (float(value), 0.0)}
        if parameter == "axicon_tilt_y_rad": return {"axicon_tilt_rad": (0.0, float(value))}
        if parameter == "hologram_decentre_x_m": return {"hologram_decentre_m": (float(value), 0.0)}
        if parameter == "fourier_iris_offset_fraction": return {"fourier_iris_offset_fraction": float(value)}
    raise ValueError((family, parameter, value))


def run_atlas_screening(*, output_root: Path = DEFAULT_OUTPUT_ROOT,
                        cases: Sequence[str] = ("B0", "V1", "V3"),
                        grid_n: int = DEFAULT_SCREENING_N, z_m: float = 60e-3) -> dict[str, Any]:
    """Generate sensitivity tables from the canonical error route."""
    rows: list[dict[str, Any]] = []
    registries = (("parameter", parameter_registry()), ("manufacturing", manufacturing_defect_registry()),
                  ("aberration", aberration_registry()), ("alignment", alignment_registry()))
    for case_id in cases:
        for family, registry in registries:
            for parameter, values in registry.items():
                for value in values:
                    source, grid, meta = build_atlas_source(
                        case_id, grid_n=grid_n, **_source_kwargs(family, parameter, value))
                    plane = propagate_selected_planes(source, grid, float(meta["wavelength_m"]), (z_m,))[0]
                    rows.append({
                        "family": family, "parameter": parameter, "value": value,
                        "case_id": case_id, "grid_n": grid_n, "z_m": z_m,
                        "route_id": meta["route_id"],
                        "axicon_tilt_model": meta.get("axicon_tilt_model", ""),
                        "full_vector_snell_fresnel": meta.get("full_vector_snell_fresnel", False),
                        **transverse_metrics(plane, grid),
                    })
    _write_csv(output_root / "atlas_screening_metrics.csv", rows)
    manifest = {
        "outcome": "VORTEX-CANONICAL-PHYSICAL-VISUAL-ATLAS-SCREENING",
        "report_figures_authorised": False,
        "screening_grid_n": int(grid_n), "report_grid_n": DEFAULT_REPORT_N,
        "route": "vbb_study.digital_twin.vortex_system_route.build_system_route",
        "cases": list(cases),
        "parameter_registry": {k: list(v) for k, v in parameter_registry().items()},
        "manufacturing_defect_registry": {k: list(v) for k, v in manufacturing_defect_registry().items()},
        "aberration_registry": {k: list(v) for k, v in aberration_registry().items()},
        "alignment_registry": {k: list(v) for k, v in alignment_registry().items()},
        "zernike_convention": "canonical unit-RMS declared-plane wavefront modes; never named-optic surrogates without evidence",
        "axicon_tilt_fidelity": "scalar rotated-angular-spectrum to/from axicon plane; full vector surface refraction required for absolute large-angle claims",
        "calibration_required": "SLM LUT/static maps/incidence, 4F geometry, axicon profile/aperture, beam and rigid-body alignment",
    }
    _write_json(output_root / "atlas_screening_manifest.json", manifest)
    return manifest
