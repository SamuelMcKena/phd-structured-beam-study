"""
vbb_study.digital_twin
======================

Beam-to-write digital twin — layered simulation package.

Stage status:
    8B  energy_accounting, exposure_bookkeeping  [IMPLEMENTED]
    8C  field_coupling, field_fluence, figures   [IMPLEMENTED]
    8D  3D beam-to-sample visualiser             [planned]
    8E  writing trajectory / dose accumulation   [planned]
    8F+ lab realism, nonlinear, thermal          [planned]
    8G  calibration ingestion                    [planned]

Allowed model statuses at Stage 8B-8C:
    optical_prediction
    energy_accounting_prediction
    fluence_prediction
    exposure_bookkeeping
    diagnostic_preview

Forbidden at Stage 8B-8C:
    fluence_threshold_proxy
    dose_accumulation_proxy
    uncalibrated_material_response_proxy
    calibrated_material_prediction
    experimentally_validated_prediction
"""

from vbb_study.digital_twin.energy_accounting import (
    LaserSource,
    OpticalComponent,
    EnergyLedgerRow,
    EnergyLedger,
    FluenceEstimate,
    PeakIntensityEstimate,
    average_power_w,
    validate_fraction,
    compute_energy_ledger,
    fresnel_normal_incidence_transmission,
    energy_fraction_after_components,
    fluence_from_effective_area_j_cm2,
    scale_intensity_to_fluence_j_cm2,
    peak_fluence_j_cm2,
    peak_intensity_w_cm2,
)
from vbb_study.digital_twin.exposure_bookkeeping import (
    pulse_spacing_um,
    pulses_per_spot,
    dose_per_unit_length_proxy,
    static_exposure_total_energy_uJ,
    line_exposure_summary,
)
from vbb_study.digital_twin.field_coupling import (
    MissingOpticalFieldError,
    InvalidOpticalFieldError,
    UnsupportedSurfaceFieldError,
    OpticalFieldPlane,
    OpticalFieldStack,
    validate_intensity_plane,
    validate_intensity_stack,
    plane_from_arrays,
    stack_from_arrays,
    extract_plane_from_surfacefield,
    extract_stack_from_surfacefield,
)
from vbb_study.digital_twin.field_fluence import (
    FluencePlaneResult,
    FluenceStackResult,
    transverse_integral_um2,
    integrated_energy_uJ_from_fluence,
    scale_plane_to_fluence,
    scale_stack_to_fluence,
    peak_intensity_from_fluence_result,
    field_fluence_summary,
)
from vbb_study.digital_twin.field_figures import (
    plot_stage8c_field_fluence_preview,
    CaveatsRequiredError,
)
from vbb_study.digital_twin.lab_realism_controls import (
    LabStageControl,
    LabStageResult,
    LabRealismReport,
    REQUIRED_STAGE_NAMES,
    ALLOWED_STATUS_LEVELS,
    default_lab_controls,
    validate_future_physics_disabled,
    build_laser_source_from_controls,
    build_energy_components_from_controls,
    build_energy_ledger_from_controls,
    build_exposure_summary_from_controls,
    build_lab_realism_report,
)
from vbb_study.digital_twin.cockpit_dashboard import (
    build_cockpit_summary,
    compute_peak_location_diagnostics,
    choose_display_plane,
    build_warning_flags,
    plot_integrated_cockpit_dashboard,
    make_interpretation_text,
)

__all__ = [
    # energy accounting
    "LaserSource",
    "OpticalComponent",
    "EnergyLedgerRow",
    "EnergyLedger",
    "FluenceEstimate",
    "PeakIntensityEstimate",
    "average_power_w",
    "validate_fraction",
    "compute_energy_ledger",
    "fresnel_normal_incidence_transmission",
    "energy_fraction_after_components",
    "fluence_from_effective_area_j_cm2",
    "scale_intensity_to_fluence_j_cm2",
    "peak_fluence_j_cm2",
    "peak_intensity_w_cm2",
    # exposure bookkeeping
    "pulse_spacing_um",
    "pulses_per_spot",
    "dose_per_unit_length_proxy",
    "static_exposure_total_energy_uJ",
    "line_exposure_summary",
    # field coupling (Stage 8C)
    "MissingOpticalFieldError",
    "InvalidOpticalFieldError",
    "UnsupportedSurfaceFieldError",
    "OpticalFieldPlane",
    "OpticalFieldStack",
    "validate_intensity_plane",
    "validate_intensity_stack",
    "plane_from_arrays",
    "stack_from_arrays",
    "extract_plane_from_surfacefield",
    "extract_stack_from_surfacefield",
    # field fluence (Stage 8C)
    "FluencePlaneResult",
    "FluenceStackResult",
    "transverse_integral_um2",
    "integrated_energy_uJ_from_fluence",
    "scale_plane_to_fluence",
    "scale_stack_to_fluence",
    "peak_intensity_from_fluence_result",
    "field_fluence_summary",
    # field figures (Stage 8C)
    "plot_stage8c_field_fluence_preview",
    "CaveatsRequiredError",
    # lab realism controls (Stage 8C.1)
    "LabStageControl",
    "LabStageResult",
    "LabRealismReport",
    "REQUIRED_STAGE_NAMES",
    "ALLOWED_STATUS_LEVELS",
    "default_lab_controls",
    "validate_future_physics_disabled",
    "build_laser_source_from_controls",
    "build_energy_components_from_controls",
    "build_energy_ledger_from_controls",
    "build_exposure_summary_from_controls",
    "build_lab_realism_report",
    # integrated cockpit dashboard (Stage 8C.1)
    "build_cockpit_summary",
    "compute_peak_location_diagnostics",
    "choose_display_plane",
    "build_warning_flags",
    "plot_integrated_cockpit_dashboard",
    "make_interpretation_text",
]
