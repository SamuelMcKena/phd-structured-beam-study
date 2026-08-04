"""Scalar Bessel and Bessel-Gauss equations used by the study.

This module is the compact formula home for the Baliyan-Nishchal scalar field
definitions.  The larger engine in ``bessel_twin_core`` still owns propagation,
device realism, and case assembly; here I keep the equations that should be
readable next to the theory notes.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
from scipy import optimize, special as sp

from .fields import gaussian_amplitude

TWOPI = 2.0 * math.pi
EPS = 1.0e-30


def medium_wavenumber_m_inv(wavelength0_m: float, n_medium: float = 1.0) -> float:
    """Return ``k = 2*pi*n/lambda0`` in rad/m.

    I pass the vacuum wavelength and refractive index explicitly so the caller
    cannot silently mix air-scale and in-material Bessel lengths.
    """

    return TWOPI * float(n_medium) / max(float(wavelength0_m), EPS)


def transverse_wavevector_from_axicon(
    k_m_inv: float,
    n_axicon: float,
    n_medium: float,
    gamma_rad: float,
    *,
    mode: str = "tan",
) -> float:
    """Return the axicon transverse wavevector ``k_r`` in rad/m.

    This is the equation behind Baliyan-Nishchal eq. 3 after writing the
    conical phase as ``exp(-i k_r r)``.  ``mode='tan'`` keeps the exact
    ``tan(gamma)`` term used by the engine; ``mode='small_angle'`` exposes the
    paper's ``sin(theta) ~= (n-1) gamma`` approximation for sanity checks.
    """

    if str(mode).lower().strip() == "small_angle":
        return float(k_m_inv) * (float(n_axicon) - float(n_medium)) * float(gamma_rad)
    return float(k_m_inv) * (float(n_axicon) - float(n_medium)) * math.tan(float(gamma_rad))


def cone_angle_from_kr(kr_m_inv: float, k_m_inv: float) -> float:
    """Return the cone angle ``theta`` in radians from ``sin(theta)=k_r/k``."""

    ratio = np.clip(float(kr_m_inv) / max(float(k_m_inv), EPS), -1.0, 1.0)
    return float(math.asin(ratio))


def longitudinal_wavevector_m_inv(k_m_inv: float, kr_m_inv: float) -> float:
    """Return ``k_z = sqrt(k^2 - k_r^2)`` in rad/m for a propagating cone."""

    return float(math.sqrt(max(float(k_m_inv) ** 2 - float(kr_m_inv) ** 2, 0.0)))


def bessel_gauss_field(
    R_m: Any,
    Phi_rad: Any,
    *,
    ell: int,
    kr_m_inv: float,
    waist_m: float,
    z_m: float = 0.0,
    wavelength0_m: float | None = None,
    n_medium: float = 1.0,
    amplitude: complex = 1.0,
) -> np.ndarray:
    """Return the scalar Bessel-Gauss field from Baliyan-Nishchal eq. 1.

    The implemented form is
    ``E = A J_l(k_r r) exp(i l phi) exp(-r^2/w0^2) exp(i k_z z)``.  If
    ``wavelength0_m`` is omitted I leave out the axial carrier because many
    transverse diagnostics only need the envelope.
    """

    R = np.asarray(R_m, dtype=float)
    Phi = np.asarray(Phi_rad, dtype=float)
    ell_i = int(ell)
    envelope = sp.jv(abs(ell_i), float(kr_m_inv) * R) * np.exp(-(R**2) / max(float(waist_m), EPS) ** 2)
    azimuth = np.exp(1j * ell_i * Phi)
    axial = 1.0
    if wavelength0_m is not None:
        k = medium_wavenumber_m_inv(float(wavelength0_m), float(n_medium))
        kz = longitudinal_wavevector_m_inv(k, float(kr_m_inv))
        axial = np.exp(1j * kz * float(z_m))
    return np.asarray(amplitude, dtype=complex) * envelope * azimuth * axial


def first_jprime_zero(ell: int, order: int = 1) -> float:
    """Return the selected zero of ``J'_ell`` as a dimensionless radius.

    I use this to define the main annular peak for ``ell > 0`` because the
    vortex makes the on-axis value intentionally dark.
    """

    ell_abs = abs(int(ell))
    order_i = int(order)
    if ell_abs == 0:
        raise ValueError("The l=0 mode has a core, not a vortex ring.")
    if order_i < 1:
        raise ValueError("order must be at least 1.")
    return float(sp.jnp_zeros(ell_abs, order_i)[order_i - 1])


def ring_radius_from_jprime_zero_m(ell: int, kr_m_inv: float, order: int = 1) -> float:
    """Return the ``ell > 0`` Bessel-ring radius in metres."""

    return first_jprime_zero(ell, order=order) / max(float(kr_m_inv), EPS)


def j0_first_null_radius_m(kr_m_inv: float) -> float:
    """Return the first zero of ``J_0(k_r r)`` in metres."""

    return float(sp.jn_zeros(0, 1)[0] / max(float(kr_m_inv), EPS))


def l0_hwhm_core_radius_m(kr_m_inv: float) -> float:
    """Return the l=0 core half-width-at-half-maximum radius in metres.

    I solve ``J_0(x)^2 = 1/2`` rather than using the first null because the
    glossary treats the bright l=0 core as an HWHM object.
    """

    root = optimize.brentq(lambda x: float(sp.jv(0, x) ** 2 - 0.5), 0.0, float(sp.jn_zeros(0, 1)[0]))
    return float(root / max(float(kr_m_inv), EPS))


def non_diffracting_length_m(
    waist_m: float,
    kr_m_inv: float,
    *,
    wavelength0_m: float,
    n_medium: float = 1.0,
) -> float:
    """Return Baliyan-Nishchal eq. 5, ``z_max = w0 / (k_r/k)`` in metres.

    I compute ``k`` in the propagation medium, so this is equivalent to
    ``2*pi*w0/(lambda_medium*k_r)``.
    """

    k = medium_wavenumber_m_inv(float(wavelength0_m), float(n_medium))
    return float(float(waist_m) * k / max(float(kr_m_inv), EPS))


def normalized_radial_intensity(ell: int, radius_m: Any, kr_m_inv: float) -> np.ndarray:
    """Return ``J_l(k_r r)^2`` normalized to its finite-array maximum."""

    radius = np.asarray(radius_m, dtype=float)
    I = np.abs(sp.jv(abs(int(ell)), float(kr_m_inv) * radius)) ** 2
    scale = float(np.nanmax(I)) if I.size else 0.0
    return I / max(scale, EPS)


# ---------------------------------------------------------------------------
# Conical axicon input field (distinct from the Bessel-Gauss target)
# ---------------------------------------------------------------------------


def conical_axicon_input_field(
    R_m: Any,
    Phi_rad: Any,
    *,
    ell: int,
    kr_m_inv: float,
    waist_m: float,
    amplitude: complex = 1.0,
) -> np.ndarray:
    """Return the Gaussian-apodized conical axicon-phase input field.

    This is the *source/input* field that an axicon imparts on a Gaussian
    beam.  Its transverse form is::

        E(r, phi) = A * exp(-r^2/w0^2) * exp(i(-k_r r + ell phi))

    This field is intentionally different from the true Bessel-Gauss target
    :func:`bessel_gauss_field`, which has a ``J_ell(k_r r)`` envelope.
    The conical axicon input field propagates into a Bessel-Gauss field over
    the non-diffracting length, but at z = 0 it has a bright Gaussian centre
    rather than the correct Bessel oscillations.

    Use :func:`bessel_gauss_field` when a mathematical Bessel-Gauss
    reference is needed.  Use this function when modelling the SLM or
    axicon output plane.

    Parameters
    ----------
    R_m, Phi_rad:
        Transverse coordinate grids (metres, radians).
    ell:
        Topological charge.  ``ell = 0`` gives a simple Gaussian times a
        converging conical phase.
    kr_m_inv:
        Transverse wavevector (axicon ring radius in k-space), rad/m.
    waist_m:
        Gaussian beam ``1/e`` amplitude radius, m.
    amplitude:
        Overall complex amplitude scale factor.
    """

    R = np.asarray(R_m, dtype=float)
    Phi = np.asarray(Phi_rad, dtype=float)
    envelope = np.exp(-(R**2) / max(float(waist_m), EPS) ** 2)
    conical_phase = np.exp(1j * (-float(kr_m_inv) * R + int(ell) * Phi))
    return complex(amplitude) * envelope * conical_phase


def build_conical_axicon_field_ideal(
    grid: dict[str, Any],
    design: Any,
    laser: Any,
    include_vortex: bool = True,
) -> np.ndarray:
    """Return the Gaussian-apodized conical axicon-phase input field.

    This field is ``exp(-r^2/w0^2) exp(i(-k_r r + ell phi))``. It is useful as
    an idealized axicon input/source field, but it is not the true
    ``J_ell(k_r r)`` Bessel-Gauss target envelope.
    """

    amp = gaussian_amplitude(grid["R"], max(design.w0_sample_m, grid["dx"]))
    phase = -design.kr_sample_m_inv * grid["R"]
    if include_vortex:
        phase = phase + design.ell * grid["PHI"]
    return amp * np.exp(1j * phase)


def build_bessel_gauss_field_ideal(
    grid: dict[str, Any],
    design: Any,
    laser: Any | None = None,
    include_vortex: bool = True,
) -> np.ndarray:
    """Return the true scalar Bessel-Gauss target field.

    The transverse envelope is
    ``J_|ell|(k_r r) exp(i ell phi) exp(-r^2/w0^2)``. For ``ell > 0`` the
    on-axis amplitude is therefore near zero, unlike the conical axicon input
    field above.
    """

    ell = int(design.ell) if include_vortex else 0
    envelope = sp.jv(abs(ell), design.kr_sample_m_inv * grid["R"])
    envelope = envelope * np.exp(-(grid["R"] ** 2) / max(design.w0_sample_m, grid["dx"], EPS) ** 2)
    phase = np.exp(1j * ell * grid["PHI"])
    return envelope * phase


def build_sample_field_ideal(
    grid: dict[str, Any],
    design: Any,
    laser: Any,
    include_vortex: bool = True,
) -> np.ndarray:
    """Compatibility wrapper for the legacy conical axicon-phase field.

    Despite the old generic name, this returns
    :func:`build_conical_axicon_field_ideal`, not the true ``J_ell`` target.
    Use :func:`build_bessel_gauss_field_ideal` when a Bessel-Gauss target field
    is required.
    """

    warnings.warn(
        "build_sample_field_ideal() returns the legacy conical axicon-phase field; "
        "use build_conical_axicon_field_ideal() or build_bessel_gauss_field_ideal() explicitly.",
        DeprecationWarning,
        stacklevel=2,
    )
    return build_conical_axicon_field_ideal(grid, design, laser, include_vortex=include_vortex)


# ---------------------------------------------------------------------------
# Vortex ring radius (readable alias)
# ---------------------------------------------------------------------------


def vortex_ring_radius_m(ell: int, kr_m_inv: float, order: int = 1) -> float:
    """Return the main bright-ring radius for a vortex Bessel beam in metres.

    For ``ell > 0`` the on-axis amplitude of the Bessel function is zero.
    The first bright ring sits at the first maximum of ``|J_ell(k_r r)|``,
    which occurs at the first zero of ``J'_ell``.

    This is a readable alias for :func:`ring_radius_from_jprime_zero_m`.

    Parameters
    ----------
    ell:
        Topological charge.  Must be > 0.
    kr_m_inv:
        Transverse wavevector, rad/m.
    order:
        Ring order (1 = innermost main ring; default).
    """

    return ring_radius_from_jprime_zero_m(ell, kr_m_inv, order=order)
