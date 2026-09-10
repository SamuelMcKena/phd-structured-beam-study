"""Versioned calibration bundle, detector transfer, and uncertainty interfaces."""

from vbb_study.calibration.detector_transfer import (
    DetectorTransferCalibration,
    DetectorTransferResult,
    apply_detector_transfer,
)
from vbb_study.calibration.full_field_uncertainty import (
    FullFieldUncertaintyConfig,
    FullFieldUncertaintyResult,
    ParameterDistribution,
    propagate_full_field_uncertainty,
)
from vbb_study.calibration.io import dump_calibration_bundle, load_calibration_bundle
from vbb_study.calibration.schema import (
    CALIBRATION_SCHEMA_VERSION,
    CalibrationBundle,
    canonical_calibration_template,
)
from vbb_study.calibration.validation import (
    CalibrationReadiness,
    CalibrationValidationReport,
    calibration_readiness_for_claim,
    validate_calibration_bundle,
)

__all__ = [
    "CALIBRATION_SCHEMA_VERSION",
    "CalibrationBundle",
    "CalibrationReadiness",
    "CalibrationValidationReport",
    "DetectorTransferCalibration",
    "DetectorTransferResult",
    "FullFieldUncertaintyConfig",
    "FullFieldUncertaintyResult",
    "ParameterDistribution",
    "apply_detector_transfer",
    "calibration_readiness_for_claim",
    "canonical_calibration_template",
    "dump_calibration_bundle",
    "load_calibration_bundle",
    "propagate_full_field_uncertainty",
    "validate_calibration_bundle",
]
