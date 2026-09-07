"""Compatibility wrapper for the legacy vortex physical-error API.

This module used to contain a second, lower-fidelity physical route.  It is now
intentionally thin: all routed simulations are delegated to
``vortex_system_route`` so historical callers cannot silently bypass the
canonical Gaussian -> SLM1 -> SLM2/carrier -> explicit 4F/iris -> axicon model.

Important fidelity rule
-----------------------
A rigidly tilted axicon is *not* representable by multiplying a laboratory-plane
field by a single scalar transmission function.  The canonical route rotates the
angular spectrum to the optic plane, applies the local axicon transmission, then
rotates back.  Therefore ``physical_axicon_transmission`` supports only a
parallel axicon; callers requesting non-zero tilt are refused and must use
``build_physical_route_checkpoints`` / ``build_system_route``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from vbb_study.digital_twin.phase2a_contracts import canonical_hardware_manifest, hardware_value
from vbb_study.digital_twin.vortex_beam_slm_errors import GaussianBeamError, SLMError
from vbb_study.digital_twin.vortex_explicit_4f import FourFError
from vbb_study.digital_twin.vortex_system_route import (
    AxiconError,
    SystemErrorConfig,
    axicon_sag_m as _canonical_axicon_sag_m,
    build_system_route,
    physical_axicon_on_own_plane,
)

DEFAULT_WINDOW_M = 10.0e-3


@dataclass(frozen=True)
class PhysicalPerturbation:
    """Legacy parameter surface mapped onto the canonical system-error route."""

    beam_radius_scale: float = 1.0
    input_beam_decentre_m: tuple[float, float] = (0.0, 0.0)
    input_beam_angle_rad: tuple[float, float] = (0.0, 0.0)
    hologram_decentre_m: tuple[float, float] = (0.0, 0.0)
    fourier_iris_offset_fraction: float = 0.0
    axicon_base_angle_scale: float = 1.0
    axicon_decentre_m: tuple[float, float] = (0.0, 0.0)
    axicon_tilt_rad: tuple[float, float] = (0.0, 0.0)
    axicon_tip_model: str = "sharp"
    axicon_rounding_parameter_m: float = 0.0
    axicon_flat_tip_radius_m: float = 0.0

    def validate(self) -> None:
        if self.beam_radius_scale <= 0.0:
            raise ValueError("beam_radius_scale must be positive")
        if self.axicon_base_angle_scale <= 0.0:
            raise ValueError("axicon_base_angle_scale must be positive")
        if self.axicon_tip_model not in {"sharp", "hyperboloidal_round", "flat_blunt"}:
            raise ValueError("axicon_tip_model must be sharp, hyperboloidal_round, or flat_blunt")
        if self.axicon_rounding_parameter_m < 0.0 or self.axicon_flat_tip_radius_m < 0.0:
            raise ValueError("tip dimensions cannot be negative")
        sx = math.sin(float(self.input_beam_angle_rad[0]))
        sy = math.sin(float(self.input_beam_angle_rad[1]))
        if sx * sx + sy * sy >= 1.0:
            raise ValueError("input beam direction cosines are non-propagating")
        # Canonical scalar rotated-plane route owns the actual domain gate.
        AxiconError(
            base_angle_scale=float(self.axicon_base_angle_scale),
            decentre_m=tuple(map(float, self.axicon_decentre_m)),
            tilt_rad=tuple(map(float, self.axicon_tilt_rad)),
            tip_model=str(self.axicon_tip_model),
            rounding_parameter_m=float(self.axicon_rounding_parameter_m),
            flat_tip_radius_m=float(self.axicon_flat_tip_radius_m),
        ).validate()


def incident_plane_wave_phase(
    grid: Mapping[str, Any],
    wavelength_m: float,
    angle_x_rad: float,
    angle_y_rad: float,
) -> np.ndarray:
    """Exact transverse-direction-cosine plane-wave factor at the input plane."""

    sx = math.sin(float(angle_x_rad))
    sy = math.sin(float(angle_y_rad))
    if sx * sx + sy * sy >= 1.0:
        raise ValueError("requested input direction has no real longitudinal component")
    k0 = 2.0 * math.pi / float(wavelength_m)
    X = np.asarray(grid["X"], dtype=float)
    Y = np.asarray(grid["Y"], dtype=float)
    return np.exp(1j * k0 * (sx * X + sy * Y))


def axicon_sag_m(
    radius_m: np.ndarray,
    base_angle_rad: float,
    *,
    tip_model: str = "sharp",
    rounding_parameter_m: float = 0.0,
    flat_tip_radius_m: float = 0.0,
) -> np.ndarray:
    """Compatibility alias for the canonical axicon sag implementation."""

    return _canonical_axicon_sag_m(
        radius_m,
        base_angle_rad,
        tip_model=tip_model,
        rounding_parameter_m=rounding_parameter_m,
        flat_tip_radius_m=flat_tip_radius_m,
    )


def physical_axicon_transmission(
    grid: Mapping[str, Any],
    *,
    wavelength_m: float,
    refractive_index: float,
    external_index: float,
    base_angle_rad: float,
    decentre_m: tuple[float, float] = (0.0, 0.0),
    tilt_rad: tuple[float, float] = (0.0, 0.0),
    tip_model: str = "sharp",
    rounding_parameter_m: float = 0.0,
    flat_tip_radius_m: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a parallel-plane axicon transmission using the canonical model.

    Non-zero rigid tilt is deliberately rejected because a tilted optic requires
    propagation to/from the tilted plane; it is not a multiplicative lab-plane
    phase screen.
    """

    if math.hypot(*map(float, tilt_rad)) > 0.0:
        raise ValueError(
            "rigid axicon tilt cannot be represented by a single lab-plane transmission; "
            "use build_physical_route_checkpoints() or vortex_system_route.build_system_route()"
        )
    error = AxiconError(
        decentre_m=tuple(map(float, decentre_m)),
        tip_model=str(tip_model),
        rounding_parameter_m=float(rounding_parameter_m),
        flat_tip_radius_m=float(flat_tip_radius_m),
    )
    field, meta = physical_axicon_on_own_plane(
        grid,
        wavelength_m=float(wavelength_m),
        base_angle_rad=float(base_angle_rad),
        refractive_index=float(refractive_index),
        external_index=float(external_index),
        error=error,
    )
    return field, {
        **meta,
        "compatibility_wrapper": True,
        "axicon_tilt_model": "parallel_plane_only; rigid tilt delegated to canonical rotated-plane route",
        "full_vector_snell_fresnel": False,
    }


def _canonical_config(perturbation: PhysicalPerturbation) -> SystemErrorConfig:
    perturbation.validate()
    manifest = canonical_hardware_manifest()
    iris_radius_m = float(hardware_value(manifest, "fourier_iris_radius_m"))
    return SystemErrorConfig(
        beam=GaussianBeamError(
            radius_x_scale=float(perturbation.beam_radius_scale),
            radius_y_scale=float(perturbation.beam_radius_scale),
            decentre_m=tuple(map(float, perturbation.input_beam_decentre_m)),
            pointing_rad=tuple(map(float, perturbation.input_beam_angle_rad)),
        ),
        slm1=SLMError(pattern_offset_m=tuple(map(float, perturbation.hologram_decentre_m))),
        slm2=SLMError(),
        fourf=FourFError(
            iris_offset_m=(float(perturbation.fourier_iris_offset_fraction) * iris_radius_m, 0.0)
        ),
        axicon=AxiconError(
            base_angle_scale=float(perturbation.axicon_base_angle_scale),
            decentre_m=tuple(map(float, perturbation.axicon_decentre_m)),
            tilt_rad=tuple(map(float, perturbation.axicon_tilt_rad)),
            tip_model=str(perturbation.axicon_tip_model),
            rounding_parameter_m=float(perturbation.axicon_rounding_parameter_m),
            flat_tip_radius_m=float(perturbation.axicon_flat_tip_radius_m),
        ),
    )


def build_physical_route_checkpoints(
    case_id: str,
    *,
    grid_n: int,
    perturbation: PhysicalPerturbation = PhysicalPerturbation(),
    window_m: float = DEFAULT_WINDOW_M,
) -> dict[str, Any]:
    """Legacy checkpoints backed entirely by the canonical integrated route."""

    route = build_system_route(
        case_id,
        grid_n=int(grid_n),
        config=_canonical_config(perturbation),
        window_m=float(window_m),
    )
    meta = dict(route["metadata"])
    beam_meta = dict(meta["beam"])
    fourf_meta = dict(meta["fourf"])
    sx, sy, sz = beam_meta["beam_direction_cosines"]
    compatibility_meta = {
        **meta,
        "route_id": "vortex_system_route_compatibility_wrapper_v2",
        "canonical_route_id": meta.get("route_id", "vortex_explicit_system_error_route_v1"),
        "legacy_route_id": "phase2e_physical_error_route",
        "beam_radius_m": float(beam_meta["beam_radius_x_m"]),
        "input_beam_decentre_m": tuple(map(float, perturbation.input_beam_decentre_m)),
        "input_beam_angle_rad": tuple(map(float, perturbation.input_beam_angle_rad)),
        "input_direction_cosines": (float(sx), float(sy), float(sz)),
        "input_angle_applied_plane": "before_SLM1",
        "hologram_decentre_m": tuple(map(float, perturbation.hologram_decentre_m)),
        "fourier_iris_offset_fraction": float(perturbation.fourier_iris_offset_fraction),
        "first_order_efficiency": float(fourf_meta["iris_selected_power_fraction"]),
        "objective_transform_application_count": 0,
        "additional_objective_pupil_application_count": 0,
        "axicon_decentre_m": tuple(map(float, perturbation.axicon_decentre_m)),
        "axicon_tilt_rad": tuple(map(float, perturbation.axicon_tilt_rad)),
        "axicon_tip_model": str(perturbation.axicon_tip_model),
        "axicon_tilt_model": meta.get("axicon_tilt_status", "none"),
        "full_vector_snell_fresnel": False,
        "compatibility_wrapper": True,
    }
    return {
        "grid": route["grid"],
        "raw_input": route["input_beam"],
        "post_slm": route["post_slm2"],
        "post_filter": route["post_4f_selected_order"],
        "post_axicon": route["post_axicon"],
        "metadata": compatibility_meta,
    }


def build_physical_source(
    case_id: str,
    *,
    grid_n: int,
    perturbation: PhysicalPerturbation = PhysicalPerturbation(),
    window_m: float = DEFAULT_WINDOW_M,
) -> tuple[np.ndarray, Mapping[str, Any], dict[str, Any]]:
    checkpoints = build_physical_route_checkpoints(
        case_id, grid_n=grid_n, perturbation=perturbation, window_m=window_m
    )
    return checkpoints["post_axicon"], checkpoints["grid"], dict(checkpoints["metadata"])


__all__ = [
    "DEFAULT_WINDOW_M",
    "PhysicalPerturbation",
    "incident_plane_wave_phase",
    "axicon_sag_m",
    "physical_axicon_transmission",
    "build_physical_route_checkpoints",
    "build_physical_source",
]
