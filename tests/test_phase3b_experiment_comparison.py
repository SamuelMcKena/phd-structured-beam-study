from __future__ import annotations

import numpy as np
import pytest

from vbb_study.calibration.detector_transfer import (
    DetectorTransferCalibration,
    apply_detector_transfer,
)
from vbb_study.calibration.phase3b_experiment_comparison import (
    ComparisonAcceptanceCriteria,
    camera_calibration_from_bundle,
    compare_phase3b_to_camera,
)
from vbb_study.calibration.schema import CalibrationBundle, canonical_calibration_template
from vbb_study.digital_twin.broadband_propagation import (
    BroadbandResult,
    SpectralFieldPlane,
    spectrum_from_arrays,
)
from vbb_study.digital_twin.phase3b_bench_broadband import Phase3BResult


def _bundle(*, laboratory: bool = True) -> CalibrationBundle:
    data = canonical_calibration_template()
    data["data_classification"] = "laboratory_measurement" if laboratory else "synthetic_not_experimental"
    data["camera"]["object_plane_scale_m_per_pixel"]["value"] = 0.25e-3
    data["camera"]["rotation_deg"]["value"] = 0.0
    data["camera"]["centre_pixel"] = [4.0, 4.0]
    data["camera"]["saturation_level"] = 100.0
    return CalibrationBundle(data)


def _phase3b_result() -> tuple[Phase3BResult, np.ndarray]:
    axis = np.linspace(-1.0e-3, 1.0e-3, 9)
    X, Y = np.meshgrid(axis, axis, indexing="xy")
    optical = np.exp(-2.0 * (X * X + Y * Y) / (0.45e-3**2))
    spectrum = spectrum_from_arrays([1.00e-6, 1.05e-6], [0.4, 0.6])
    field1 = SpectralFieldPlane(
        wavelength_m=1.00e-6,
        x_m=axis,
        y_m=axis,
        Ex=np.sqrt(optical).astype(np.complex128),
    )
    field2 = SpectralFieldPlane(
        wavelength_m=1.05e-6,
        x_m=axis,
        y_m=axis,
        Ex=np.sqrt(optical).astype(np.complex128),
    )
    broadband = BroadbandResult(
        spectrum=spectrum,
        fields=(field1, field2),
        detector_integrated_intensity=optical,
        x_m=axis,
        y_m=axis,
        spectral_centroid_x_m=np.asarray([0.0, 0.0]),
        spectral_centroid_y_m=np.asarray([0.0, 0.0]),
        metadata={"test": True},
    )
    response = np.full_like(optical, 0.8)
    detector = apply_detector_transfer(
        optical,
        DetectorTransferCalibration(
            relative_response=response,
            additive_background=2.0,
            saturation_level=100.0,
            source="synthetic_known_detector",
        ),
    )
    result = Phase3BResult(
        broadband=broadband,
        detector_transfer=detector,
        spectrum_status="synthetic_known_spectrum",
        sample_material_status="sellmeier_fused_silica",
        axicon_material_status="sellmeier_known_glass",
        blockers=(),
        metadata={"experimental_validation_claimed": False},
    )
    return result, detector.optical_response_intensity


def test_detector_transfer_exposes_correct_comparison_stage() -> None:
    phase3b, optical_response = _phase3b_result()
    detector = phase3b.detector_transfer
    assert detector is not None
    np.testing.assert_allclose(detector.raw_unsaturated_intensity, optical_response + 2.0)
    np.testing.assert_allclose(detector.intensity, optical_response + 2.0)
    assert detector.metadata["comparison_stage"].startswith("optical_response_intensity")


def test_phase3b_camera_bridge_recovers_exact_known_comparison() -> None:
    phase3b, optical_response = _phase3b_result()
    bundle = _bundle(laboratory=True)
    measured_background = 2.0
    measured_raw = optical_response + measured_background
    criteria = ComparisonAcceptanceCriteria(
        min_energy_normalised_correlation=0.999999,
        max_energy_normalised_l2=1.0e-10,
        max_centroid_error_m=1.0e-12,
        min_valid_overlap_fraction=0.999,
    )
    comparison = compare_phase3b_to_camera(
        phase3b,
        measured_raw,
        bundle,
        measured_background=measured_background,
        acceptance_criteria=criteria,
    )
    assert comparison.acceptance_passed is True
    assert comparison.status == "predeclared_numerical_acceptance_passed"
    assert comparison.blockers == ()
    assert comparison.camera.metrics["energy_normalised_correlation"] == pytest.approx(1.0, abs=1e-12)
    assert comparison.camera.metrics["energy_normalised_l2"] == pytest.approx(0.0, abs=1e-12)
    assert comparison.camera.metrics["centroid_error_m"] == pytest.approx(0.0, abs=1e-12)
    assert comparison.metadata["automatic_registration"] is False
    assert comparison.metadata["automatic_scale_fit"] is False
    assert comparison.metadata["experimental_validation_claimed"] is False


def test_phase3b_camera_bridge_does_not_call_synthetic_bundle_experimentally_validated() -> None:
    phase3b, optical_response = _phase3b_result()
    bundle = _bundle(laboratory=False)
    comparison = compare_phase3b_to_camera(
        phase3b,
        optical_response,
        bundle,
        measured_background=0.0,
        acceptance_criteria=ComparisonAcceptanceCriteria(min_energy_normalised_correlation=0.99),
    )
    assert comparison.acceptance_passed is True
    assert comparison.status == "acceptance_evaluated_but_calibration_blockers_remain"
    assert any("not 'laboratory_measurement'" in blocker for blocker in comparison.blockers)
    assert comparison.metadata["experimental_validation_claimed"] is False


def test_camera_comparison_requires_physical_scale() -> None:
    data = canonical_calibration_template()
    data["camera"]["object_plane_scale_m_per_pixel"]["value"] = None
    with pytest.raises(ValueError, match="object_plane_scale_m_per_pixel"):
        camera_calibration_from_bundle(CalibrationBundle(data))


def test_missing_rotation_and_centre_are_visible_blockers_not_fitted() -> None:
    data = canonical_calibration_template()
    data["camera"]["object_plane_scale_m_per_pixel"]["value"] = 2.0e-6
    data["camera"]["rotation_deg"]["value"] = None
    data["camera"]["centre_pixel"] = None
    calibration, blockers = camera_calibration_from_bundle(CalibrationBundle(data))
    assert calibration.rotation_rad == 0.0
    assert calibration.centre_pixel_x is None
    assert calibration.centre_pixel_y is None
    assert any("rotation unresolved" in blocker for blocker in blockers)
    assert any("centre pixel unresolved" in blocker for blocker in blockers)
