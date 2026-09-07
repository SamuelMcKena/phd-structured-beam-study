"""Authoritative cleanup policy for legacy structured-beam error simulations.

The repository contains multiple generations of error-modelling code.  This
registry prevents a legacy visualisation or lower-fidelity compatibility model
from being silently promoted to physical evidence.

Statuses
--------
canonical_physical
    Error is applied to the complex field at a physically declared plane and is
    the preferred route for morphology-sensitive work.
calibration_limited
    Physical operator exists, but absolute bench claims require measured
    parameters/maps/LUTs.
compatibility_restricted
    Retained for old dashboards/tests, with specific claims prohibited.
diagnostic_only
    Post-propagation or display-space diagnostic. Never correction evidence.
deprecated_redirect
    Superseded by a named canonical module.
reference_only
    Analytic/literature benchmark, not a bench forward model.
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
    LegacyErrorModule("vbb_study.digital_twin.vortex_system_route", "canonical_physical", True, True, "self", "Integrated Gaussian->SLM1->SLM2/carrier->explicit 4F/iris->axicon route; errors are introduced at declared optical planes."),
    LegacyErrorModule("vbb_study.digital_twin.vortex_system_error_sweeps", "canonical_physical", True, True, "vbb_study.digital_twin.vortex_system_route", "Sensitivity values are explicitly non-measured and builders route into the physical system model."),
    LegacyErrorModule("vbb_study.digital_twin.vortex_beam_slm_errors", "calibration_limited", True, True, "vbb_study.slm_model + measured LUT/static maps", "Registration/quantisation/stroke/fringing are field-plane models, but panel-specific LUT, static map and fringing parameters require calibration."),
    LegacyErrorModule("vbb_study.slm_model", "canonical_physical", True, True, "self", "Separates throughput-only, resolved pixel aperture and coherent unmodulated dead-space models and enforces spatial sampling for resolved pixels."),
    LegacyErrorModule("vbb_study.digital_twin.vortex_explicit_4f", "canonical_physical", True, True, "self", "Physical Fourier-plane propagation and aperture route; preferred over image-space clipping or shifting."),
    LegacyErrorModule("vbb_study.digital_twin.vortex_rotated_plane", "calibration_limited", True, True, "self", "Scalar rotated-angular-spectrum utility; suitable for scalar sensitivity but not a full vector Snell/Fresnel surface solver."),
    LegacyErrorModule("vbb_study.digital_twin.vortex_rotated_plane_baseband", "reference_only", False, False, "vbb_study.digital_twin.vortex_rotated_plane", "Numerical/reference helper for rotated-plane behaviour, not a standalone bench claim."),
    LegacyErrorModule("vbb_study.digital_twin.vortex_error_reference_models", "reference_only", False, False, "self", "Analytic/literature validation contracts only; never substitute reference equations for a propagated bench model."),
    LegacyErrorModule("vbb_study.digital_twin.vortex_physical_errors", "deprecated_redirect", False, False, "vbb_study.digital_twin.vortex_system_route", "Older physical-route prototype. Axicon tilt is a small-angle thin-element OPD approximation and is superseded by the explicit rotated-plane system route."),
    LegacyErrorModule("vbb_study.digital_twin.vortex_error_research_figures", "deprecated_redirect", False, False, "vbb_study.digital_twin.vortex_system_route + vortex_system_error_sweeps", "Figure generator still imports the older physical-route prototype; figures remain research diagnostics until redirected to the canonical route."),
    LegacyErrorModule("vbb_study.digital_twin.component_plane_pipeline", "compatibility_restricted", True, False, "vbb_study.digital_twin.vortex_system_route", "Correctly applies many perturbations before propagation, but its legacy fill-factor control is uniform sqrt(FF) throughput only and it lacks an explicit 4F plane for several controls."),
    LegacyErrorModule("vbb_study.digital_twin.lab_perturbations", "diagnostic_only", False, False, "vbb_study.digital_twin.vortex_system_route", "Module explicitly operates as a post-engine diagnostic layer for many stack transforms. It must not be cited as causal optical propagation or correction evidence."),
    LegacyErrorModule("vbb_study.digital_twin.lab_realism_controls", "compatibility_restricted", False, False, "vbb_study.digital_twin.vortex_system_error_sweeps", "Dashboard/control compatibility layer; physical claim must come from the canonical routed implementation, not the control itself."),
    LegacyErrorModule("vbb_study.digital_twin.phase2b_visual_cases", "diagnostic_only", False, False, "vbb_study.digital_twin.vortex_system_route", "Visual-case layer is not an authoritative physical forward model."),
    LegacyErrorModule("vbb_study.digital_twin.phase2b_visual_diagnostics", "diagnostic_only", False, False, "vbb_study.digital_twin.vortex_system_route", "Display/diagnostic transformations remain post-propagation evidence only."),
    LegacyErrorModule("vbb_study.digital_twin.vortex_wavefront_errors", "canonical_physical", True, True, "self", "Declared-plane OPD maps. Zernikes are generic wavefront sensitivities, not surrogates for a named optic unless measured/derived for that optic."),
    LegacyErrorModule("vbb_study.digital_twin.vortex_axicon_tip_reference", "reference_only", False, False, "vbb_study.digital_twin.vortex_system_route", "Rounded-tip reference/benchmark only; physical figures must use an explicit sag/phase transmission and propagation."),
    LegacyErrorModule("vbb_study.digital_twin.vortex_round_tip_reference", "reference_only", False, False, "vbb_study.digital_twin.vortex_system_route", "Legacy rounded-tip reference implementation; use only for cross-checks."),
    LegacyErrorModule("vbb_study.digital_twin.vortex_axicon_oblique_reference", "reference_only", False, False, "vbb_study.digital_twin.vortex_system_route", "Oblique-incidence literature/reference contract, not bench propagation."),
    LegacyErrorModule("vbb_study.digital_twin.vortex_axicon_oblique_wave", "deprecated_redirect", False, False, "vbb_study.digital_twin.vortex_system_route", "Special-purpose oblique-wave implementation is superseded by the integrated route and independent reference checks."),
    LegacyErrorModule("vbb_study.digital_twin.vortex_refractive_axicon", "deprecated_redirect", False, False, "vbb_study.digital_twin.vortex_system_route", "Older scalar refractive-axicon route; retained for regression/reference comparison only."),
    LegacyErrorModule("vbb_study.digital_twin.vortex_refractive_axicon_wave", "deprecated_redirect", False, False, "vbb_study.digital_twin.vortex_system_route", "Older wave-propagation variant; integrated system route is authoritative for error studies."),
    LegacyErrorModule("vbb_study.digital_twin.vector_refractive_axicon", "calibration_limited", True, False, "self", "Vector refractive-axicon branch is useful for vector sensitivity; absolute bench claims remain dependent on actual material/surface/incidence calibration."),
    LegacyErrorModule("vbb_study.digital_twin.vector_refractive_axicon_eikonal", "reference_only", False, False, "vbb_study.digital_twin.vector_refractive_axicon", "Eikonal/reference route is a benchmark, not the final propagated vector bench model."),
    LegacyErrorModule("vbb_study.digital_twin.vector_tilt_study", "calibration_limited", True, False, "vbb_study.digital_twin.vector_refractive_axicon", "Vector tilt sensitivity is meaningful only within the stated scalar/vector interface assumptions and measured geometry."),
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
