from __future__ import annotations

import numpy as np

from vbb_study.calibration.schema import CalibrationBundle, canonical_calibration_template
from vbb_study.digital_twin.broadband_propagation import (
    SpectralFieldPlane,
    spectrum_from_arrays,
    spectrum_from_wavelength_density,
)
from vbb_study.digital_twin.phase3b_bench_broadband import Phase3BConfig, run_phase3b_broadband


def test_wavelength_density_uses_trapezoidal_cell_weights_on_nonuniform_grid() -> None:
    wavelengths = np.asarray([1.0e-6, 2.0e-6, 4.0e-6])
    spectrum = spectrum_from_wavelength_density(wavelengths, [1.0, 1.0, 1.0])
    # Trapezoidal point widths are [0.5, 1.5, 1.0] um over this grid.
    np.testing.assert_allclose(spectrum.energy_weights, [1.0 / 6.0, 0.5, 1.0 / 3.0])
    assert spectrum.metadata["wavelength_grid_nonuniform"] is True
    assert "density_integrated" in spectrum.metadata["spectral_value_interpretation"]


def test_explicit_energy_weights_are_not_reweighted_by_wavelength_spacing() -> None:
    spectrum = spectrum_from_arrays([1.0e-6, 2.0e-6, 4.0e-6], [1.0, 1.0, 2.0])
    np.testing.assert_allclose(spectrum.energy_weights, [0.25, 0.25, 0.5])
    assert spectrum.metadata["spectral_value_interpretation"] == "integrated_discrete_energy_weights"


def test_phase3b_blocks_broadband_calibration_claim_for_unknown_constant_index_axicon() -> None:
    data = canonical_calibration_template()
    data["laser"]["pulse_duration_s"]["value"] = 250e-15
    # The legacy template has a single assumed axicon index but no glass ID.
    # Phase 3B may propagate it as a labelled comparison, but must not call it
    # calibrated broadband dispersion.
    bundle = CalibrationBundle(data)
    axis = np.linspace(-1.0e-4, 1.0e-4, 7)

    def propagate(context):
        return SpectralFieldPlane(
            wavelength_m=context.wavelength_m,
            x_m=axis,
            y_m=axis,
            Ex=np.ones((axis.size, axis.size), dtype=np.complex128),
            metadata={"test_context": dict(context.metadata)},
        )

    result = run_phase3b_broadband(
        bundle,
        propagate,
        config=Phase3BConfig(
            require_measured_spectrum=False,
            allow_transform_limited_control=True,
            transform_limited_samples=5,
        ),
    )
    assert result.axicon_material_status == "constant_index_no_dispersion"
    assert any("axicon wavelength dispersion unresolved" in blocker for blocker in result.blockers)
    assert result.metadata["phase2h_two_surface_refractive_axicon_available"] is True
    assert result.metadata["full_volume_fdtd_fem_available"] is False
    assert result.metadata["experimental_validation_claimed"] is False
