from __future__ import annotations

"""Physical candidate mechanisms used by experiment-to-simulation matching.

These candidates construct *complex-field operators in their declared optical
planes*.  They are not templates extracted from the rendered error atlas.
Default values are deliberately labelled as unmeasured sensitivity grids; real
parameter inference should replace them with ranges justified by calibration,
mechanical tolerances or prior measurements.
"""

from dataclasses import dataclass
from typing import Any, Mapping
import math

import numpy as np

from vbb_study.digital_twin.vortex_beam_slm_errors import GaussianBeamError, SLMError
from vbb_study.digital_twin.vortex_explicit_4f import FourFError
from vbb_study.digital_twin.vortex_system_route import AxiconError, SystemErrorConfig
from vbb_study.digital_twin.vortex_wavefront_errors import zernike_opd_map_m


WAVEFRONT_SUPPORT_RADIUS_M = 2.5e-3


@dataclass(frozen=True)
class CandidateFamily:
    key: str
    title: str
    default_values: tuple[float, ...]
    parameter_unit: str
    operator_class: str
    physical_plane: str
    provenance: str
    miao_handoff_policy: str


FAMILY_REGISTRY: dict[str, CandidateFamily] = {
    "beam_decentre_x": CandidateFamily(
        key="beam_decentre_x",
        title="Input-beam lateral decentre",
        default_values=(0.0, 0.75e-3, 1.50e-3),
        parameter_unit="m",
        operator_class="input_amplitude_translation",
        physical_plane="pre_SLM1",
        provenance="unmeasured sensitivity grid; translated Gaussian before SLM1",
        miao_handoff_policy="mechanism_specific_not_phase_retrieval_target",
    ),
    "fourf_iris_offset_x": CandidateFamily(
        key="fourf_iris_offset_x",
        title="4F iris lateral offset",
        default_values=(0.0, 0.45e-3, 0.80e-3),
        parameter_unit="m",
        operator_class="hard_amplitude_aperture_translation",
        physical_plane="4F_Fourier_plane",
        provenance="unmeasured sensitivity grid; physical circular Fourier-plane aperture",
        miao_handoff_policy="amplitude_or_clipping_not_phase_only",
    ),
    "slm_phase_stroke": CandidateFamily(
        key="slm_phase_stroke",
        title="Dual-SLM phase-stroke underdrive",
        default_values=(1.0, 0.85, 0.70),
        parameter_unit="command_scale",
        operator_class="SLM_phase_response_scale",
        physical_plane="SLM1_and_SLM2",
        provenance="unmeasured sensitivity grid until measured panel LUT/stroke is supplied",
        miao_handoff_policy="calibrate_SLM_response_before_wavefront_retrieval",
    ),
    "axicon_decentre_x": CandidateFamily(
        key="axicon_decentre_x",
        title="Axicon lateral decentre",
        default_values=(0.0, 0.50e-3, 1.00e-3),
        parameter_unit="m",
        operator_class="axicon_sag_coordinate_translation",
        physical_plane="axicon_surface",
        provenance="unmeasured sensitivity grid; translated physical axicon sag/apex coordinates",
        miao_handoff_policy="alignment_parameter_not_direct_conjugate_phase",
    ),
    "axicon_round_tip_radius": CandidateFamily(
        key="axicon_round_tip_radius",
        title="Rounded axicon apex",
        default_values=(0.0, 200e-6, 800e-6),
        parameter_unit="m_radial_scale",
        operator_class="physical_axicon_sag_defect",
        physical_plane="axicon_surface",
        provenance="unmeasured hyperboloidal rounded-tip sensitivity grid",
        miao_handoff_policy="surface_geometry_requires_mechanism_specific_model",
    ),
    "lens1_astigmatism": CandidateFamily(
        key="lens1_astigmatism",
        title="Declared L1 astigmatism",
        default_values=(0.0, 0.35, 0.70),
        parameter_unit="waves_RMS",
        operator_class="declared_plane_OPD",
        physical_plane="4F_lens1",
        provenance="generic unit-RMS Zernike OPD sensitivity; not a measured L1 wavefront",
        miao_handoff_policy="phase_retrieval_candidate_requires_plane_mapping",
    ),
    "lens1_trefoil": CandidateFamily(
        key="lens1_trefoil",
        title="Declared L1 trefoil",
        default_values=(0.0, 0.35, 0.70),
        parameter_unit="waves_RMS",
        operator_class="declared_plane_OPD",
        physical_plane="4F_lens1",
        provenance="generic unit-RMS Zernike OPD sensitivity; not a measured L1 wavefront",
        miao_handoff_policy="phase_retrieval_candidate_requires_plane_mapping",
    ),
    "lens1_quadrafoil": CandidateFamily(
        key="lens1_quadrafoil",
        title="Declared L1 quadrafoil",
        default_values=(0.0, 0.35, 0.70),
        parameter_unit="waves_RMS",
        operator_class="declared_plane_OPD",
        physical_plane="4F_lens1",
        provenance="generic unit-RMS Zernike OPD sensitivity; not a measured L1 wavefront",
        miao_handoff_policy="phase_retrieval_candidate_requires_plane_mapping",
    ),
}

# Match the seven mechanisms in the completed B0/V1/V3 atlas by default.
DEFAULT_MATCH_FAMILIES: tuple[str, ...] = (
    "beam_decentre_x",
    "fourf_iris_offset_x",
    "slm_phase_stroke",
    "axicon_decentre_x",
    "axicon_round_tip_radius",
    "lens1_astigmatism",
    "lens1_trefoil",
)


def family_spec(key: str) -> CandidateFamily:
    try:
        return FAMILY_REGISTRY[str(key)]
    except KeyError as exc:
        raise KeyError(f"unknown physical error candidate family {key!r}") from exc


def build_candidate_route_arguments(
    family: str,
    value: float,
    *,
    grid: Mapping[str, Any],
    wavelength_m: float,
    axicon_base_angle_rad: float,
) -> tuple[SystemErrorConfig, dict[str, np.ndarray]]:
    """Build canonical-route configuration and user-supplied-map arguments."""
    key = family_spec(family).key
    v = float(value)
    if key == "beam_decentre_x":
        return SystemErrorConfig(beam=GaussianBeamError(decentre_m=(v, 0.0))), {}
    if key == "fourf_iris_offset_x":
        return SystemErrorConfig(fourf=FourFError(iris_offset_m=(v, 0.0))), {}
    if key == "slm_phase_stroke":
        if v <= 0.0:
            raise ValueError("SLM phase-stroke scale must be positive")
        error = SLMError(phase_stroke_scale=v)
        return SystemErrorConfig(slm1=error, slm2=error), {}
    if key == "axicon_decentre_x":
        return SystemErrorConfig(axicon=AxiconError(decentre_m=(v, 0.0))), {}
    if key == "axicon_round_tip_radius":
        if v < 0.0:
            raise ValueError("rounded-tip radial scale cannot be negative")
        if v == 0.0:
            return SystemErrorConfig(axicon=AxiconError(tip_model="sharp")), {}
        # Route convention: a = r_tip * tan(gamma) for the hyperboloidal sag.
        a = v * math.tan(float(axicon_base_angle_rad))
        return SystemErrorConfig(
            axicon=AxiconError(tip_model="hyperboloidal_round", rounding_parameter_m=a)
        ), {}
    if key.startswith("lens1_"):
        mode = {
            "lens1_astigmatism": "astigmatism_x",
            "lens1_trefoil": "trefoil_x",
            "lens1_quadrafoil": "quadrafoil_x",
        }[key]
        opd = zernike_opd_map_m(
            mode,
            grid,
            wavelength_m=float(wavelength_m),
            waves_rms=v,
            pupil_radius_m=WAVEFRONT_SUPPORT_RADIUS_M,
        )
        return SystemErrorConfig(), {"lens1_opd_map_m": opd}
    raise AssertionError(key)


__all__ = [
    "CandidateFamily",
    "FAMILY_REGISTRY",
    "DEFAULT_MATCH_FAMILIES",
    "WAVEFRONT_SUPPORT_RADIUS_M",
    "family_spec",
    "build_candidate_route_arguments",
]
