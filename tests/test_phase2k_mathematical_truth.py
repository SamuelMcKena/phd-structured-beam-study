"""Phase 2K mathematical truth tests.

These tests are intentionally reference-first rather than regression-first.  A
historical numerical baseline is not accepted merely because the old code
reproduces it.  Each check compares the implementation with an analytic law or
an independently written propagation calculation.
"""

from __future__ import annotations

import math

import numpy as np
import scipy.special as sp

from vbb_study.digital_twin.vortex_error_reference_models import (
    exact_refractive_axicon_kr_m_inv,
)
from vbb_study.equations.scalar_bessel import (
    axicon_cone_angle_exact_rad,
    bessel_gauss_field,
    bessel_gauss_ring_radius_m,
    ring_radius_from_jprime_zero_m,
    transverse_wavevector_from_axicon,
)


def _fresnel_fft(field: np.ndarray, dx_m: float, z_m: float, k_m_inv: float) -> np.ndarray:
    """Independent paraxial transfer-function propagation used only in tests."""

    n = int(field.shape[0])
    fx = np.fft.fftfreq(n, d=float(dx_m))
    kx = 2.0 * np.pi * fx
    kx_grid, ky_grid = np.meshgrid(kx, kx)
    transfer = np.exp(-1j * (kx_grid**2 + ky_grid**2) * float(z_m) / (2.0 * float(k_m_inv)))
    return np.fft.ifft2(np.fft.fft2(field) * transfer)


def _grid(n: int = 384, dx_m: float = 0.35e-6):
    x = (np.arange(n, dtype=float) - n // 2) * float(dx_m)
    xg, yg = np.meshgrid(x, x)
    return xg, yg, np.hypot(xg, yg), np.arctan2(yg, xg)


def test_bessel_gauss_z0_is_exact_requested_waist_field() -> None:
    _, _, radius, phi = _grid(192, 0.45e-6)
    ell = 3
    kr = 0.85e6
    waist = 18.0e-6
    got = bessel_gauss_field(radius, phi, ell=ell, kr_m_inv=kr, waist_m=waist)
    expected = sp.jv(ell, kr * radius) * np.exp(-(radius**2) / waist**2) * np.exp(1j * ell * phi)
    assert np.allclose(got, expected, rtol=2.0e-13, atol=2.0e-13)


def test_bessel_gauss_analytic_propagation_matches_independent_fresnel_fft() -> None:
    xg, yg, radius, phi = _grid()
    wavelength = 1029.0e-9
    n_medium = 1.0
    k = 2.0 * np.pi * n_medium / wavelength
    ell = 3
    kr = 0.75e6
    waist = 16.0e-6
    z = 22.0e-6

    initial = bessel_gauss_field(
        radius,
        phi,
        ell=ell,
        kr_m_inv=kr,
        waist_m=waist,
        z_m=0.0,
    )
    numeric = _fresnel_fft(initial, float(xg[0, 1] - xg[0, 0]), z, k)
    analytic = bessel_gauss_field(
        radius,
        phi,
        ell=ell,
        kr_m_inv=kr,
        waist_m=waist,
        z_m=z,
        wavelength0_m=wavelength,
        n_medium=n_medium,
    )

    # Compare only where the finite beam carries meaningful power, avoiding a
    # relative-error metric dominated by machine noise in the tails.
    mask = np.abs(analytic) >= 1.0e-5 * float(np.max(np.abs(analytic)))
    rel_l2 = float(np.linalg.norm((numeric - analytic)[mask]) / np.linalg.norm(analytic[mask]))
    assert rel_l2 < 2.0e-4


def test_nonzero_z_bessel_gauss_requires_wavelength() -> None:
    _, _, radius, phi = _grid(64, 0.8e-6)
    try:
        bessel_gauss_field(
            radius,
            phi,
            ell=1,
            kr_m_inv=0.5e6,
            waist_m=15.0e-6,
            z_m=1.0e-6,
        )
    except ValueError as exc:
        assert "wavelength0_m" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("non-zero-z BG propagation silently omitted wavelength")


def test_finite_bg_ring_is_not_assumed_to_be_jprime_peak() -> None:
    ell = 3
    kr = 1.0e6
    finite_waist = 4.0e-6
    finite = bessel_gauss_ring_radius_m(ell, kr, finite_waist)
    pure = ring_radius_from_jprime_zero_m(ell, kr)
    # Gaussian apodization pulls the finite-energy maximum inward.
    assert finite < pure
    assert (pure - finite) / pure > 1.0e-3

    # In the weak-apodization limit the finite BG peak must converge to the
    # infinite-Bessel J'_ell reference.
    almost_flat = bessel_gauss_ring_radius_m(ell, kr, 2.0e-3)
    assert abs(almost_flat - pure) / pure < 2.0e-5


def test_exact_snell_axicon_kr_matches_independent_digital_twin_reference() -> None:
    wavelength = 1029.0e-9
    k0 = 2.0 * np.pi / wavelength
    n_ax = 1.458
    n_ext = 1.0
    gamma = math.radians(2.0)
    kr_scalar_reference = transverse_wavevector_from_axicon(
        k0,
        n_ax,
        n_ext,
        gamma,
        mode="snell_exact",
    )
    kr_digital_twin = exact_refractive_axicon_kr_m_inv(
        wavelength_m=wavelength,
        base_angle_rad=gamma,
        refractive_index=n_ax,
        external_index=n_ext,
    )
    assert math.isclose(kr_scalar_reference, kr_digital_twin, rel_tol=2.0e-14, abs_tol=0.0)


def test_axicon_thin_phase_uses_vacuum_k0_not_medium_k() -> None:
    wavelength = 1029.0e-9
    k0 = 2.0 * np.pi / wavelength
    n_ax = 1.70
    n_medium = 1.45
    gamma = math.radians(1.5)
    expected = k0 * (n_ax - n_medium) * math.tan(gamma)
    got = transverse_wavevector_from_axicon(k0, n_ax, n_medium, gamma, mode="tan")
    assert math.isclose(got, expected, rel_tol=2.0e-15, abs_tol=0.0)


def test_axicon_angle_convention_obeys_snell_law() -> None:
    n_ax = 1.458
    n_ext = 1.0
    gamma = math.radians(2.0)
    theta = axicon_cone_angle_exact_rad(n_ax, n_ext, gamma)
    lhs = n_ax * math.sin(gamma)
    rhs = n_ext * math.sin(gamma + theta)
    assert math.isclose(lhs, rhs, rel_tol=2.0e-14, abs_tol=2.0e-14)
