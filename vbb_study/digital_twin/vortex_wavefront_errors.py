"""Declared-plane wavefront-error maps for controlled aberration studies.

These Zernikes are OPD specifications, not surrogates for a named misaligned
optic.  A map is only physically interpretable when its plane is declared.
The basis includes the azimuthal orders needed by the Miao synthetic benchmark:
astigmatism (|m|=2), trefoil (|m|=3), quadrafoil (|m|=4), plus defocus, coma
and spherical for general wavefront-sensitivity work.
"""
from __future__ import annotations

import numpy as np
from typing import Any, Mapping

EPS = np.finfo(float).tiny
ZERNIKE_NAMES = (
    "defocus", "astigmatism_x", "astigmatism_y", "coma_x", "coma_y",
    "trefoil_x", "trefoil_y", "quadrafoil_x", "quadrafoil_y", "spherical",
)


def unit_rms_zernike(name: str, grid: Mapping[str, Any], *, pupil_radius_m: float,
                     centre_m: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    if name not in ZERNIKE_NAMES:
        raise ValueError(f"unsupported Zernike {name!r}")
    if float(pupil_radius_m) <= 0.0:
        raise ValueError("pupil_radius_m must be positive")
    X = np.asarray(grid["X"], float) - float(centre_m[0])
    Y = np.asarray(grid["Y"], float) - float(centre_m[1])
    rho = np.hypot(X, Y) / float(pupil_radius_m)
    theta = np.arctan2(Y, X)
    if name == "defocus": raw = 2.0 * rho**2 - 1.0
    elif name == "astigmatism_x": raw = rho**2 * np.cos(2.0 * theta)
    elif name == "astigmatism_y": raw = rho**2 * np.sin(2.0 * theta)
    elif name == "coma_x": raw = (3.0 * rho**3 - 2.0 * rho) * np.cos(theta)
    elif name == "coma_y": raw = (3.0 * rho**3 - 2.0 * rho) * np.sin(theta)
    elif name == "trefoil_x": raw = rho**3 * np.cos(3.0 * theta)
    elif name == "trefoil_y": raw = rho**3 * np.sin(3.0 * theta)
    elif name == "quadrafoil_x": raw = rho**4 * np.cos(4.0 * theta)
    elif name == "quadrafoil_y": raw = rho**4 * np.sin(4.0 * theta)
    else: raw = 6.0 * rho**4 - 6.0 * rho**2 + 1.0
    mask = rho <= 1.0
    rms = float(np.sqrt(np.mean(raw[mask] ** 2)))
    return np.where(mask, raw / max(rms, EPS), 0.0)


def zernike_opd_map_m(name: str, grid: Mapping[str, Any], *, wavelength_m: float,
                      waves_rms: float, pupil_radius_m: float,
                      centre_m: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    return float(waves_rms) * float(wavelength_m) * unit_rms_zernike(
        name, grid, pupil_radius_m=float(pupil_radius_m), centre_m=centre_m)


def opd_to_phase_rad(opd_map_m: np.ndarray, wavelength_m: float) -> np.ndarray:
    if float(wavelength_m) <= 0.0:
        raise ValueError("wavelength_m must be positive")
    return 2.0 * np.pi * np.asarray(opd_map_m, float) / float(wavelength_m)
