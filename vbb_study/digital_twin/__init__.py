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
]
