from __future__ import annotations

import numpy as np
import pytest

from vbb_study.digital_twin.vortex_explicit_4f import (
    FourFError,
    LensError,
    explicit_4f_relay,
    nominal_order_position_m,
)
from vbb_study.equations.fields import make_xy_grid
from vbb_study.digital_twin.vortex_system_route import (
    build_multirate_system_route,
    fourier_resample_fixed_window,
)


def test_canonical_plus_one_iris_centre_is_about_1p929_mm() -> None:
    x, y = nominal_order_position_m(
        wavelength_m=1029e-9,
        focal_length_m=0.300,
        carrier_cpm=6250.0,
    )
    assert np.isclose(x, 1.9294149e-3, rtol=0.0, atol=1e-9)
    assert y == 0.0


def test_iris_error_offset_is_relative_to_nominal_plus_one_order() -> None:
    grid = make_xy_grid(128, 10e-3 / 128)
    X = np.asarray(grid["X"])
    Y = np.asarray(grid["Y"])
    field = np.exp(-(X**2 + Y**2) / (1.5e-3) ** 2) * np.exp(1j * 2*np.pi*6250.0*X)
    route = explicit_4f_relay(
        field,
        grid,
        wavelength_m=1029e-9,
        nominal_focal_length_m=0.05,
        nominal_iris_radius_m=1.5e-3,
        nominal_carrier_cpm=6250.0,
        error=FourFError(iris_offset_m=(100e-6, 0.0)),
    )
    nominal = route["metadata"]["nominal_selected_order_centre_m"]
    centre = route["metadata"]["physical_iris_centre_m"]
    assert np.isclose(centre[0] - nominal[0], 100e-6, atol=1e-12)


def test_rigid_lens_tilt_uses_rotated_plane_backend() -> None:
    grid = make_xy_grid(128, 10e-3 / 128)
    field = np.exp(-(np.asarray(grid["R"]) / 1.5e-3) ** 2).astype(complex)
    route = explicit_4f_relay(
        field,
        grid,
        wavelength_m=1029e-9,
        nominal_focal_length_m=0.05,
        nominal_iris_radius_m=2e-3,
        nominal_carrier_cpm=0.0,
        error=FourFError(lens1=LensError(tilt_rad=(0.0, 1e-3))),
    )
    assert route["metadata"]["lens1"]["tilt_status"] == "scalar_rotated_angular_spectrum_thin_lens"
    assert route["metadata"]["lens1"]["tilt_fidelity"] == "scalar_paraxial_rotated_plane"


def test_parallel_lens_keeps_identity_tilt_status() -> None:
    grid = make_xy_grid(128, 10e-3 / 128)
    field = np.exp(-(np.asarray(grid["R"]) / 1.5e-3) ** 2).astype(complex)
    route = explicit_4f_relay(
        field,
        grid,
        wavelength_m=1029e-9,
        nominal_focal_length_m=0.05,
        nominal_iris_radius_m=2e-3,
    )
    assert route["metadata"]["lens1"]["tilt_status"] == "none"
    assert route["metadata"]["lens2"]["tilt_status"] == "none"


def test_fixed_window_fourier_resampling_preserves_a_bandlimited_field() -> None:
    grid = make_xy_grid(64, 10e-3 / 64)
    field = np.exp(-(np.asarray(grid["R"]) / 1.2e-3) ** 2).astype(complex)
    fine = fourier_resample_fixed_window(field, 128)
    fine_grid = make_xy_grid(128, 10e-3 / 128)
    expected = np.exp(-(np.asarray(fine_grid["R"]) / 1.2e-3) ** 2)
    assert fine.shape == (128, 128)
    assert np.allclose(np.real(fine), expected, atol=2e-6, rtol=2e-6)
    assert np.max(np.abs(np.imag(fine))) < 2e-6


def test_multirate_route_reports_distinct_relay_and_axicon_sampling() -> None:
    route = build_multirate_system_route(
        "V3", relay_grid_n=128, propagation_grid_n=256, window_m=10e-3,
    )
    assert route["post_axicon"].shape == (256, 256)
    assert route["relay_route"]["post_4f_selected_order"].shape == (128, 128)
    assert route["metadata"]["route_id"] == "vortex_explicit_system_error_multirate_route_v2"
    assert route["metadata"]["propagation_dx_m"] < route["metadata"]["relay_dx_m"]


def test_multirate_route_rejects_coarsening_at_axicon_handoff() -> None:
    with pytest.raises(ValueError, match="propagation_grid_n"):
        build_multirate_system_route(
            "V3", relay_grid_n=256, propagation_grid_n=128, window_m=10e-3,
        )
