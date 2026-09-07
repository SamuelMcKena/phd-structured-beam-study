"""Authoritative evidence policy for legacy structured-beam error simulations.

A historical module is never promoted to physical or correction evidence merely
because its filename names a physical error.  This registry records whether its
operator is canonical, calibration-limited, compatibility-only, diagnostic,
redirected, or reference-only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

Status = Literal[
    "canonical_physical", "calibration_limited", "compatibility_restricted",
    "diagnostic_only", "deprecated_redirect", "reference_only",
]


@dataclass(frozen=True)
class LegacyErrorModule:
    module: str
    status: Status
    physical_claim_allowed: bool
    correction_evidence_allowed: bool
    replacement: str
    reason: str


LEGACY_ERROR_MODULES: tuple[LegacyErrorModule, ...] = (
    LegacyErrorModule("vbb_study.digital_twin.vortex_system_route", "canonical_physical", True, True, "self", "Integrated Gaussian -> SLM1 -> SLM2/carrier -> explicit 4F/iris -> axicon route; errors act at declared optical planes."),
    LegacyErrorModule("vbb_study.digital_twin.vortex_system_error_sweeps", "canonical_physical", True, True, "vbb_study.digital_twin.vortex_system_route", "Sensitivity values are explicitly non-measured and builders use the integrated physical route."),
    LegacyErrorModule("vbb_study.digital_twin.vortex_visual_atlas", "calibration_limited", True, False, "vbb_study.digital_twin.vortex_system_route", "Atlas now routes physical errors through the canonical system model and generic Zernikes through the declared-plane OPD basis; outputs remain uncalibrated sensitivity diagnostics."),
    LegacyErrorModule("vbb_study.digital_twin.vortex_visual_atlas_figures", "calibration_limited", True, False, "vbb_study.digital_twin.vortex_visual_atlas", "Renderer consumes the canonical-route atlas but is not measured correction validation."),
    LegacyErrorModule("vbb_study.digital_twin.vortex_error_research_figures", "calibration_limited", True, False, "vbb_study.digital_twin.vortex_system_route", "Input-pointing figures have been redirected to the canonical system route; LCOS angular response and bench geometry still require calibration."),
    LegacyErrorModule("vbb_study.digital_twin.vortex_beam_slm_errors", "calibration_limited", True, True, "vbb_study.slm_model + measured LUT/static maps", "Registration, quantisation, stroke and fringing are field-plane models; panel-specific LUT/static map/fringing parameters require calibration."),
    LegacyErrorModule("vbb_study.slm_model", "canonical_physical", True, True, "self", "Separates throughput-only, resolved pixel aperture and coherent unmodulated dead-space models and enforces spatial sampling for resolved pixels."),
    LegacyErrorModule("vbb_study.digital_twin.vortex_explicit_4f", "canonical_physical", True, True, "self", "Physical propagated Fourier-plane and iris route; preferred over image-space clipping/shifting."),
    LegacyErrorModule("vbb_study.digital_twin.vortex_rotated_plane", "calibration_limited", True, True, "self", "Scalar rotated-angular-spectrum utility; not a full vector Snell/Fresnel surface solver."),
    LegacyErrorModule("vbb_study.digital_twin.vortex_rotated_plane_baseband", "reference_only", False, False, "vbb_study.digital_twin.vortex_rotated_plane", "Numerical reference helper, not a standalone bench model."),
    LegacyErrorModule("vbb_study.digital_twin.vortex_error_reference_models", "reference_only", False, False, "self", "Analytic/literature validation targets only."),
    LegacyErrorModule("vbb_study.digital_twin.vortex_physical_errors", "deprecated_redirect", False, False, "vbb_study.digital_twin.vortex_system_route", "Older route prototype; its small-angle thin-element axicon-tilt approximation is superseded by explicit rotated-plane propagation."),
    LegacyErrorModule("vbb_study.digital_twin.component_plane_pipeline", "compatibility_restricted", True, False, "vbb_study.digital_twin.vortex_system_route", "Many errors are correctly pre-propagation, but the legacy fill-factor control is uniform sqrt(FF) throughput only and the route lacks explicit 4F planes for several controls."),
    LegacyErrorModule("vbb_study.digital_twin.lab_perturbations", "diagnostic_only", False, False, "vbb_study.digital_twin.vortex_system_route", "Explicit post-engine diagnostic layer for many stack transforms; never causal propagation or correction evidence."),
    LegacyErrorModule("vbb_study.digital_twin.lab_realism_controls", "compatibility_restricted", False, False, "vbb_study.digital_twin.vortex_system_error_sweeps", "UI/control compatibility definitions; evidence must come from the routed implementation."),
    LegacyErrorModule("vbb_study.digital_twin.phase2b_visual_cases", "diagnostic_only", False, False, "vbb_study.digital_twin.vortex_system_route", "Visual-case layer is not an authoritative physical forward model."),
    LegacyErrorModule("vbb_study.digital_twin.phase2b_visual_diagnostics", "diagnostic_only", False, False, "vbb_study.digital_twin.vortex_system_route", "Display diagnostics remain post-propagation evidence only."),
    LegacyErrorModule("vbb_study.digital_twin.vortex_wavefront_errors", "canonical_physical", True, True, "self", "Declared-plane generic OPD maps through quadrafoil; not named-optic surrogates unless measured/derived for that optic."),
    LegacyErrorModule("vbb_study.digital_twin.vortex_axicon_tip_reference", "reference_only", False, False, "vbb_study.digital_twin.vortex_system_route", "Rounded-tip benchmark only."),
    LegacyErrorModule("vbb_study.digital_twin.vortex_round_tip_reference", "reference_only", False, False, "vbb_study.digital_twin.vortex_system_route", "Legacy rounded-tip reference only."),
    LegacyErrorModule("vbb_study.digital_twin.vortex_axicon_oblique_reference", "reference_only", False, False, "vbb_study.digital_twin.vortex_system_route", "Oblique-incidence literature/reference contract only."),
    LegacyErrorModule("vbb_study.digital_twin.vortex_axicon_oblique_wave", "deprecated_redirect", False, False, "vbb_study.digital_twin.vortex_system_route", "Special-purpose oblique-wave implementation superseded by the integrated route plus independent reference checks."),
    LegacyErrorModule("vbb_study.digital_twin.vortex_refractive_axicon", "deprecated_redirect", False, False, "vbb_study.digital_twin.vortex_system_route", "Older scalar refractive-axicon route retained for regression/reference comparison."),
    LegacyErrorModule("vbb_study.digital_twin.vortex_refractive_axicon_wave", "deprecated_redirect", False, False, "vbb_study.digital_twin.vortex_system_route", "Older wave-propagation variant superseded for error studies."),
    LegacyErrorModule("vbb_study.digital_twin.vector_refractive_axicon", "calibration_limited", True, False, "self", "Vector refractive-axicon sensitivity route; absolute bench claims depend on material/surface/incidence calibration."),
    LegacyErrorModule("vbb_study.digital_twin.vector_refractive_axicon_eikonal", "reference_only", False, False, "vbb_study.digital_twin.vector_refractive_axicon", "Eikonal benchmark only."),
    LegacyErrorModule("vbb_study.digital_twin.vector_tilt_study", "calibration_limited", True, False, "vbb_study.digital_twin.vector_refractive_axicon", "Vector tilt sensitivity within stated interface assumptions; not absolute bench proof."),
)


def legacy_error_rows() -> list[dict[str, object]]:
    return [asdict(row) for row in LEGACY_ERROR_MODULES]


def module_policy(module: str) -> LegacyErrorModule:
    for row in LEGACY_ERROR_MODULES:
        if row.module == module:
            return row
    raise KeyError(f"unregistered legacy error module: {module}")


def assert_correction_evidence_allowed(module: str) -> None:
    row = module_policy(module)
    if not row.correction_evidence_allowed:
        raise RuntimeError(
            f"{module} is {row.status} and is not authorised as correction evidence; "
            f"use {row.replacement}. Reason: {row.reason}"
        )


__all__ = ["Status", "LegacyErrorModule", "LEGACY_ERROR_MODULES", "legacy_error_rows",
           "module_policy", "assert_correction_evidence_allowed"]
