"""Phase 3B broadband simulation-to-experiment comparison bridge.

This module joins the new broadband/detector prediction to the established
calibrated camera-coordinate comparison. It never optimises camera scale,
rotation, centre, background or acceptance thresholds to make an image agree.

A numerical comparison result is not, by itself, an experimental-validation
claim. Acceptance criteria must be supplied explicitly and measurement/provenance
blockers remain visible in the result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

from vbb_study.calibration.camera_comparison import (
    CameraCalibration,
    CameraComparison,
    compare_simulation_to_camera,
)
from vbb_study.calibration.schema import CalibrationBundle, value_at

if TYPE_CHECKING:
    from vbb_study.digital_twin.phase3b_bench_broadband import Phase3BResult


@dataclass(frozen=True)
class ComparisonAcceptanceCriteria:
    min_energy_normalised_correlation: float | None = None
    max_energy_normalised_l2: float | None = None
    max_centroid_error_m: float | None = None
    min_valid_overlap_fraction: float | None = None

    def validate(self) -> None:
        if self.min_energy_normalised_correlation is not None and not 0.0 <= self.min_energy_normalised_correlation <= 1.0:
            raise ValueError("min_energy_normalised_correlation must lie in [0,1]")
        if self.max_energy_normalised_l2 is not None and self.max_energy_normalised_l2 < 0.0:
            raise ValueError("max_energy_normalised_l2 must be non-negative")
        if self.max_centroid_error_m is not None and self.max_centroid_error_m < 0.0:
            raise ValueError("max_centroid_error_m must be non-negative")
        if self.min_valid_overlap_fraction is not None and not 0.0 <= self.min_valid_overlap_fraction <= 1.0:
            raise ValueError("min_valid_overlap_fraction must lie in [0,1]")


@dataclass(frozen=True)
class Phase3BExperimentComparison:
    camera: CameraComparison
    acceptance_passed: bool | None
    acceptance_checks: Mapping[str, bool]
    blockers: tuple[str, ...]
    status: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _load_background(path: str | Path) -> np.ndarray:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    suffix = source.suffix.lower()
    if suffix == ".npy":
        values = np.load(source, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(source, allow_pickle=False) as data:
            if len(data.files) != 1:
                raise ValueError("background NPZ must contain exactly one array")
            values = data[data.files[0]]
    elif suffix in {".csv", ".txt"}:
        values = np.loadtxt(source, delimiter="," if suffix == ".csv" else None)
    else:
        raise ValueError(f"unsupported background format {suffix!r}")
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.size == 0 or np.any(~np.isfinite(array)):
        raise ValueError("camera background must be a finite non-empty 2-D array")
    return array


def camera_calibration_from_bundle(
    bundle: CalibrationBundle,
) -> tuple[CameraCalibration, tuple[str, ...]]:
    """Build camera coordinates from supplied calibration without hidden fitting."""

    blockers: list[str] = []
    scale = value_at(bundle, "camera.object_plane_scale_m_per_pixel")
    if scale in (None, ""):
        raise ValueError("camera.object_plane_scale_m_per_pixel is required for physical comparison")
    rotation_deg = value_at(bundle, "camera.rotation_deg")
    if rotation_deg in (None, ""):
        rotation_rad = 0.0
        blockers.append("camera rotation unresolved; provisional zero rotation used")
    else:
        rotation_rad = float(np.deg2rad(float(rotation_deg)))

    centre = bundle.data.get("camera", {}).get("centre_pixel")
    if centre is None:
        centre_x = centre_y = None
        blockers.append("camera optical-axis centre pixel unresolved; image centre used provisionally")
    else:
        if not isinstance(centre, (list, tuple)) or len(centre) != 2:
            raise ValueError("camera.centre_pixel must be [x, y]")
        centre_x, centre_y = float(centre[0]), float(centre[1])

    saturation = bundle.data.get("camera", {}).get("saturation_level")
    calibration = CameraCalibration(
        object_plane_scale_m_per_pixel=float(scale),
        rotation_rad=rotation_rad,
        centre_pixel_x=centre_x,
        centre_pixel_y=centre_y,
        saturation_level=None if saturation in (None, "") else float(saturation),
    )
    calibration.validate()
    return calibration, tuple(blockers)


def _evaluate_acceptance(
    metrics: Mapping[str, float],
    criteria: ComparisonAcceptanceCriteria | None,
) -> tuple[bool | None, dict[str, bool]]:
    if criteria is None:
        return None, {}
    criteria.validate()
    checks: dict[str, bool] = {}
    if criteria.min_energy_normalised_correlation is not None:
        checks["energy_normalised_correlation"] = (
            float(metrics["energy_normalised_correlation"]) >= float(criteria.min_energy_normalised_correlation)
        )
    if criteria.max_energy_normalised_l2 is not None:
        checks["energy_normalised_l2"] = float(metrics["energy_normalised_l2"]) <= float(criteria.max_energy_normalised_l2)
    if criteria.max_centroid_error_m is not None:
        checks["centroid_error_m"] = float(metrics["centroid_error_m"]) <= float(criteria.max_centroid_error_m)
    if criteria.min_valid_overlap_fraction is not None:
        checks["valid_overlap_fraction"] = float(metrics["valid_overlap_fraction"]) >= float(criteria.min_valid_overlap_fraction)
    if not checks:
        raise ValueError("acceptance criteria object contains no active criterion")
    return bool(all(checks.values())), checks


def compare_phase3b_to_camera(
    phase3b: "Phase3BResult",
    measured_intensity: np.ndarray,
    bundle: CalibrationBundle,
    *,
    measured_background: np.ndarray | float | None = None,
    acceptance_criteria: ComparisonAcceptanceCriteria | None = None,
) -> Phase3BExperimentComparison:
    """Compare broadband prediction to a measured image in calibrated coordinates.

    If a Phase 3B detector transfer exists, the comparison uses its
    ``optical_response_intensity`` stage: PSF/relative detector response have
    been applied, but additive background and saturation have not. The measured
    image is independently background-subtracted by the established camera
    comparison and measured saturated pixels are masked.
    """

    camera_calibration, camera_blockers = camera_calibration_from_bundle(bundle)
    blockers = list(phase3b.blockers) + list(camera_blockers)

    background = measured_background
    background_path = bundle.data.get("camera", {}).get("background_frame_path")
    if background is None and background_path:
        background = _load_background(str(background_path))
    if background is None:
        blockers.append("measured camera background frame not supplied")

    if bundle.data_classification != "laboratory_measurement":
        blockers.append(
            f"calibration bundle classification is {bundle.data_classification!r}, not 'laboratory_measurement'"
        )

    if phase3b.detector_transfer is None:
        simulated = phase3b.broadband.detector_integrated_intensity
        detector_stage = "ideal_broadband_intensity_no_measured_detector_transfer"
        blockers.append("measured detector PSF/response transfer not supplied")
    else:
        simulated = phase3b.detector_transfer.optical_response_intensity
        detector_stage = "measured_psf_response_before_background_and_saturation"

    camera = compare_simulation_to_camera(
        measured_intensity,
        camera_calibration,
        simulated,
        phase3b.broadband.x_m,
        phase3b.broadband.y_m,
        background=background,
    )
    acceptance_passed, checks = _evaluate_acceptance(camera.metrics, acceptance_criteria)

    if acceptance_criteria is None:
        status = "metrics_only_no_predeclared_acceptance_criteria"
    elif blockers:
        status = "acceptance_evaluated_but_calibration_blockers_remain"
    elif acceptance_passed:
        status = "predeclared_numerical_acceptance_passed"
    else:
        status = "predeclared_numerical_acceptance_failed"

    return Phase3BExperimentComparison(
        camera=camera,
        acceptance_passed=acceptance_passed,
        acceptance_checks=checks,
        blockers=tuple(dict.fromkeys(blockers)),
        status=status,
        metadata={
            "calibration_id": bundle.calibration_id,
            "data_classification": bundle.data_classification,
            "detector_comparison_stage": detector_stage,
            "automatic_registration": False,
            "automatic_scale_fit": False,
            "automatic_acceptance_threshold_fit": False,
            "experimental_validation_claimed": False,
        },
    )


__all__ = [
    "ComparisonAcceptanceCriteria",
    "Phase3BExperimentComparison",
    "camera_calibration_from_bundle",
    "compare_phase3b_to_camera",
]
