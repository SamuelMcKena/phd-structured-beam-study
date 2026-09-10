from __future__ import annotations

import numpy as np
import pytest

from vbb_study.calibration.detector_transfer import (
    DetectorTransferCalibration,
    apply_detector_transfer,
)
from vbb_study.calibration.full_field_uncertainty import (
    FullFieldUncertaintyConfig,
    ParameterDistribution,
    propagate_full_field_uncertainty,
)
from vbb_study.calibration.schema import CalibrationBundle, canonical_calibration_template
from vbb_study.digital_twin.broadband_propagation import (
    SpectralFieldPlane,
    gaussian_transform_limited_spectrum,
    propagate_broadband,
    spectrum_from_arrays,
)
from vbb_study.digital_twin.phase3b_bench_broadband import (
    Phase3BConfig,
    material_model_from_calibration,
    spectrum_from_calibration,
)
from vbb_study.equations.calibrated_objective import (
    ObjectivePupilCalibration,
    apply_objective_pupil_calibration,
)
from vbb_study.equations.dispersion import ConstantIndexMaterial, FUSED_SILICA_MALITSON
from vbb_study.equations.vector_surface_refraction import refract_vectors


def test_fused_silica_sellmeier_is_dispersive_near_1030_nm() -> None:
    wavelengths = np.asarray([0.8e-6, 1.03e-6, 1.5e-6])
    n = FUSED_SILICA_MALITSON.refractive_index(wavelengths)
    assert 1.44 < n[1] < 1.46
    assert n[0] > n[1] > n[2]
    ng = float(FUSED_SILICA_MALITSON.group_index(1.03e-6))
    assert ng > float(n[1])


def test_constant_index_material_is_explicitly_constant() -> None:
    material = ConstantIndexMaterial("control", 1.5)
    np.testing.assert_allclose(material.refractive_index([0.9e-6, 1.1e-6]), [1.5, 1.5])


def test_broadband_detector_sum_uses_energy_weights() -> None:
    spectrum = spectrum_from_arrays([1.0e-6, 1.1e-6], [1.0, 3.0])
    axis = np.linspace(-1.0e-3, 1.0e-3, 9)
    X, Y = np.meshgrid(axis, axis, indexing="xy")

    def propagate(wavelength_m: float) -> SpectralFieldPlane:
        amplitude = 1.0 if wavelength_m < 1.05e-6 else 2.0
        field = amplitude * np.exp(-(X * X + Y * Y) / (0.4e-3**2))
        return SpectralFieldPlane(wavelength_m, axis, axis, field.astype(complex))

    result = propagate_broadband(spectrum, propagate)
    i1 = np.abs(propagate(1.0e-6).Ex) ** 2
    i2 = np.abs(propagate(1.1e-6).Ex) ** 2
    np.testing.assert_allclose(result.detector_integrated_intensity, 0.25 * i1 + 0.75 * i2)
    assert result.metadata["combination_rule"].startswith("incoherent")


def test_broadband_rejects_hidden_grid_change() -> None:
    spectrum = spectrum_from_arrays([1.0e-6, 1.1e-6], [1.0, 1.0])

    def propagate(wavelength_m: float) -> SpectralFieldPlane:
        n = 9 if wavelength_m < 1.05e-6 else 11
        axis = np.linspace(-1.0, 1.0, n)
        return SpectralFieldPlane(wavelength_m, axis, axis, np.ones((n, n), dtype=complex))

    with pytest.raises(ValueError, match="one declared physical output grid"):
        propagate_broadband(spectrum, propagate)


def test_transform_limited_control_is_labelled_not_measured() -> None:
    spectrum = gaussian_transform_limited_spectrum(1.03e-6, 250e-15, samples=9)
    assert spectrum.wavelengths_m.size == 9
    assert np.isclose(np.sum(spectrum.energy_weights), 1.0)
    assert spectrum.metadata["measured_spectrum"] is False


def test_objective_pupil_calibration_applies_amplitude_opd_and_mask() -> None:
    ex = np.ones((4, 4), dtype=complex)
    ey = np.zeros_like(ex)
    amplitude = np.full((4, 4), 0.5)
    opd = np.zeros((4, 4))
    opd[:, 2:] = 0.5e-6
    valid = np.ones((4, 4), dtype=bool)
    valid[0, 0] = False
    calibration = ObjectivePupilCalibration(amplitude, opd, valid)
    out_ex, out_ey, meta = apply_objective_pupil_calibration(
        ex, ey, wavelength_m=1.0e-6, calibration=calibration
    )
    assert out_ex[0, 0] == 0.0
    np.testing.assert_allclose(np.abs(out_ex[1:, :]), 0.5)
    np.testing.assert_allclose(out_ex[1, 3], -0.5 + 0.0j, atol=1e-12)
    np.testing.assert_allclose(out_ey, 0.0)
    assert meta["automatic_map_resizing"] is False


def test_detector_transfer_psf_and_saturation_are_explicit() -> None:
    image = np.zeros((9, 9), dtype=float)
    image[4, 4] = 10.0
    psf = np.ones((3, 3), dtype=float)
    calibration = DetectorTransferCalibration(psf=psf, saturation_level=0.8)
    result = apply_detector_transfer(image, calibration)
    # FFT convolution leaves roundoff-sized values outside the compact PSF
    # support, so test the physically meaningful saturation support rather than
    # exact floating-point zeros.
    assert np.count_nonzero(result.saturated_mask) == 9
    assert float(np.max(result.intensity)) == pytest.approx(0.8)
    assert result.metadata["saturated_fraction"] == pytest.approx(9.0 / 81.0)
    assert result.metadata["automatic_fit_to_measurement"] is False


def test_full_field_uncertainty_reruns_field() -> None:
    axis = np.linspace(-1.0, 1.0, 15)
    X, Y = np.meshgrid(axis, axis, indexing="xy")

    class Plane:
        def __init__(self, intensity: np.ndarray) -> None:
            self.intensity = intensity
            self.x_m = axis
            self.y_m = axis

    def simulate(parameters: dict[str, float]) -> Plane:
        width = parameters["width"]
        return Plane(np.exp(-(X * X + Y * Y) / (width * width)))

    result = propagate_full_field_uncertainty(
        [ParameterDistribution("width", 0.5, 0.02, lower_bound=0.1)],
        simulate,
        config=FullFieldUncertaintyConfig(samples=30, random_seed=7),
        metric_functions={"peak": lambda plane: float(np.max(plane.intensity))},
    )
    assert result.failed_samples == 0
    assert result.mean_intensity.shape == (15, 15)
    assert np.any(result.standard_deviation_intensity > 0.0)
    assert result.metadata["automatic_registration"] is False


def test_vector_surface_refraction_normal_incidence_energy_closure() -> None:
    result = refract_vectors(
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, -1.0]),
        n_incident=1.0,
        n_transmitted=1.5,
    )
    np.testing.assert_allclose(result.transmitted_direction, [0.0, 0.0, 1.0], atol=1e-14)
    np.testing.assert_allclose(result.reflected_direction, [0.0, 0.0, -1.0], atol=1e-14)
    assert float(result.R_s + result.T_s) == pytest.approx(1.0, abs=1e-12)
    assert float(result.R_p + result.T_p) == pytest.approx(1.0, abs=1e-12)


def test_vector_surface_refraction_obeys_snell() -> None:
    theta_i = np.deg2rad(30.0)
    result = refract_vectors(
        np.array([np.sin(theta_i), 0.0, np.cos(theta_i)]),
        np.array([0.0, 0.0, -1.0]),
        n_incident=1.0,
        n_transmitted=1.5,
    )
    theta_t = np.arctan2(result.transmitted_direction[0], result.transmitted_direction[2])
    assert float(theta_t) == pytest.approx(np.arcsin(np.sin(theta_i) / 1.5), abs=1e-12)


def test_phase3b_material_policy_does_not_assign_unknown_axicon_glass() -> None:
    data = canonical_calibration_template()
    data["axicon"]["refractive_index"]["value"] = None
    bundle = CalibrationBundle(data)
    sample, sample_status = material_model_from_calibration(bundle, role="sample")
    axicon, axicon_status = material_model_from_calibration(bundle, role="axicon")
    assert sample is FUSED_SILICA_MALITSON
    assert sample_status == "sellmeier_fused_silica"
    assert axicon is None
    assert axicon_status.startswith("blocked")


def test_phase3b_synthetic_spectrum_requires_explicit_opt_in() -> None:
    data = canonical_calibration_template()
    data["laser"]["pulse_duration_s"]["value"] = 250e-15
    bundle = CalibrationBundle(data)
    with pytest.raises(ValueError, match="measured-spectrum run blocked"):
        spectrum_from_calibration(bundle, Phase3BConfig())
    spectrum, status = spectrum_from_calibration(
        bundle,
        Phase3BConfig(require_measured_spectrum=False, allow_transform_limited_control=True, transform_limited_samples=9),
    )
    assert spectrum.wavelengths_m.size == 9
    assert status == "synthetic_transform_limited_control_not_measurement"
