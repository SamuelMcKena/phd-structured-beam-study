"""
vbb_study.digital_twin
======================

Beam-to-write digital twin — layered simulation package.

Stage status:
    8B  energy_accounting, exposure_bookkeeping  [IMPLEMENTED]
    8C  engine1 adapter (optical fields)         [planned]
    8D  3D beam-to-sample visualiser             [planned]
    8E  writing trajectory / dose accumulation   [planned]
    8F+ lab realism, nonlinear, thermal          [planned]
    8G  calibration ingestion                    [planned]

Allowed model statuses at Stage 8B:
    energy_accounting_prediction
    fluence_prediction
    exposure_bookkeeping
    diagnostic_preview

Forbidden at Stage 8B:
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
]
