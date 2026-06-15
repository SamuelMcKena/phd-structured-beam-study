"""Propagation and sampling equations for scalar-field checks.

The heavy propagators live in ``bessel_twin_core``.  Here I keep the compact
expressions for medium wavelength, ASM longitudinal wavenumber, Matsushima-style
band limiting, and Nyquist margins so the notebooks can report the same
quantities with one vocabulary.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

TWOPI = 2.0 * math.pi
EPS = 1.0e-30


def medium_wavelength_m(wavelength0_m: float, n_medium: float = 1.0) -> float:
    """Return the wavelength inside a medium in metres."""

    return float(wavelength0_m) / max(float(n_medium), EPS)


def medium_wavenumber_m_inv(wavelength0_m: float, n_medium: float = 1.0) -> float:
    """Return ``k = 2*pi/lambda_medium`` in rad/m."""

    return TWOPI / max(medium_wavelength_m(wavelength0_m, n_medium), EPS)


def fft_frequency_grid(N: int, dx_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Return centered FFT frequency grids in cycles/m."""

    freq = np.fft.fftshift(np.fft.fftfreq(int(N), d=float(dx_m)))
    return np.meshgrid(freq, freq, indexing="xy")


def asm_longitudinal_wavenumber_m_inv(
    FX_cpm: Any,
    FY_cpm: Any,
    *,
    wavelength_m: float,
    include_evanescent: bool = True,
) -> np.ndarray:
    """Return the ASM ``k_z`` array in rad/m for frequency grids in cycles/m."""

    FX = np.asarray(FX_cpm, dtype=float)
    FY = np.asarray(FY_cpm, dtype=float)
    arg = TWOPI**2 * ((1.0 / max(float(wavelength_m), EPS)) ** 2 - FX**2 - FY**2)
    root = np.sqrt(np.abs(arg))
    if include_evanescent:
        return np.where(arg >= 0.0, root, 1j * root)
    return np.where(arg >= 0.0, root, 0.0)


def matsushima_bandlimit_mask(
    FX_cpm: Any,
    FY_cpm: Any,
    *,
    wavelength_m: float,
    z_m: float,
    N: int,
    dx_m: float,
) -> np.ndarray:
    """Return the square band-limit mask used by BL-ASM.

    This matches the compact sampling condition used in the engine:
    ``u_lim = 1/(lambda * sqrt((2*du*z)^2 + 1))`` with
    ``du = 1/(N*dx)``.
    """

    FX = np.asarray(FX_cpm, dtype=float)
    FY = np.asarray(FY_cpm, dtype=float)
    du = 1.0 / (int(N) * float(dx_m))
    u_lim = 1.0 / (max(float(wavelength_m), EPS) * math.sqrt((2.0 * du * abs(float(z_m))) ** 2 + 1.0))
    return (np.abs(FX) <= u_lim) & (np.abs(FY) <= u_lim)


def nyquist_spatial_frequency_cpm(dx_m: float) -> float:
    """Return the grid Nyquist spatial frequency in cycles/m."""

    return 0.5 / max(float(dx_m), EPS)


def radial_period_m(kr_m_inv: float) -> float:
    """Return the radial Bessel fringe period ``2*pi/k_r`` in metres."""

    return TWOPI / max(float(kr_m_inv), EPS)


def samples_per_radial_period(dx_m: float, kr_m_inv: float) -> float:
    """Return how many transverse samples cover one ``2*pi/k_r`` period."""

    return radial_period_m(float(kr_m_inv)) / max(float(dx_m), EPS)


def sampling_margin_for_transverse_wavevector(dx_m: float, kr_m_inv: float) -> float:
    """Return a Nyquist margin for a transverse wavevector.

    Values greater than one mean ``k_r/(2*pi)`` lies inside the sampling
    Nyquist limit; smaller values are a warning that the radial oscillation is
    under-sampled.
    """

    kr_cpm = float(kr_m_inv) / TWOPI
    return nyquist_spatial_frequency_cpm(float(dx_m)) / max(abs(kr_cpm), EPS)


def discrete_power(values: Any, dx_m: float) -> float:
    """Return ``sum(|U|^2) dx^2`` for a sampled transverse field."""

    U = np.asarray(values)
    return float(np.sum(np.abs(U) ** 2) * float(dx_m) * float(dx_m))
