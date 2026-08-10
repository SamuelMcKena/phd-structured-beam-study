"""Carrier-tracked rotated angular spectrum for large oblique plane angles.

The standard sampled field cannot represent a many-degree optical carrier on a
10 mm / O(10^3) grid.  This module keeps the absolute carrier wavevector as
metadata and samples only the baseband envelope.  Spectral rotation and the
Matsushima solid-angle Jacobian are evaluated with the *absolute* wavevectors,
so large plane rotations do not require aliasing the carrier onto the FFT grid.

This is still a scalar tilted-plane coordinate transform.  It is a numerical
building block for an oblique thin-axicon wave model, not a substitute for a
full vector two-surface Fresnel/Snell solver.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from vbb_study.digital_twin.vortex_rotated_plane import (
    DEFAULT_INTERPOLATION_ORDER,
    _spline_uniform_complex,
    rotation_matrix,
)
from vbb_study.equations.fields import fft2c, ifft2c


EPS = np.finfo(float).tiny


def rotate_baseband_angular_spectrum(
    baseband_field: np.ndarray,
    grid: Mapping[str, Any],
    *,
    wavelength_m: float,
    tilt_x_rad: float,
    tilt_y_rad: float,
    source_spectral_center_cpm: tuple[float, float],
    inverse: bool = False,
    interpolation_order: int = DEFAULT_INTERPOLATION_ORDER,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Rotate a sampled envelope while tracking its absolute carrier exactly."""

    envelope = np.asarray(baseband_field, dtype=np.complex128)
    tx = float(tilt_x_rad)
    ty = float(tilt_y_rad)
    fsx, fsy = map(float, source_spectral_center_cpm)
    inv_lam = 1.0 / float(wavelength_m)
    source_center_sq = fsx * fsx + fsy * fsy
    if source_center_sq >= inv_lam * inv_lam:
        raise ValueError("source carrier is non-propagating")
    fsz = math.sqrt(max(inv_lam * inv_lam - source_center_sq, 0.0))
    source_center = np.asarray([fsx, fsy, fsz], dtype=float)

    if tx == 0.0 and ty == 0.0:
        return envelope.copy(), {
            "tilt_x_rad": 0.0,
            "tilt_y_rad": 0.0,
            "inverse": bool(inverse),
            "source_spectral_center_cpm": [fsx, fsy],
            "destination_spectral_center_cpm": [fsx, fsy],
            "spectral_power_ratio": 1.0,
            "spectral_clipped_fraction": 0.0,
            "interpolation_model": "identity",
            "carrier_representation": "analytic_unsampled",
            "fidelity": "exact_identity_baseband",
        }

    R = rotation_matrix(tx, ty)
    if inverse:
        R = R.T
    destination_center = R.T @ source_center
    fdx = float(destination_center[0])
    fdy = float(destination_center[1])

    foff = np.asarray(grid["FX"], dtype=float)
    goff = np.asarray(grid["FY"], dtype=float)
    axis = np.asarray(grid.get("fx", foff[0]), dtype=float)
    if axis.ndim != 1 or axis.size != foff.shape[1]:
        axis = np.asarray(foff[0], dtype=float)

    spectrum = fft2c(envelope)
    fx_destination = fdx + foff
    fy_destination = fdy + goff
    destination_transverse_sq = fx_destination * fx_destination + fy_destination * fy_destination
    propagating_destination = destination_transverse_sq < inv_lam * inv_lam
    fz_destination = np.sqrt(
        np.maximum(inv_lam * inv_lam - destination_transverse_sq, 0.0)
    )

    fx_source = (
        R[0, 0] * fx_destination
        + R[0, 1] * fy_destination
        + R[0, 2] * fz_destination
    )
    fy_source = (
        R[1, 0] * fx_destination
        + R[1, 1] * fy_destination
        + R[1, 2] * fz_destination
    )
    fz_source = (
        R[2, 0] * fx_destination
        + R[2, 1] * fy_destination
        + R[2, 2] * fz_destination
    )
    valid = propagating_destination & (fz_source > 0.0)

    sampled = _spline_uniform_complex(
        spectrum,
        axis,
        fx_source - fsx,
        fy_source - fsy,
        order=int(interpolation_order),
    )
    jacobian = np.zeros_like(fz_destination)
    jacobian[valid] = np.abs(fz_source[valid]) / np.maximum(fz_destination[valid], EPS)
    rotated = np.where(valid, sampled * jacobian, 0.0)
    output_envelope = ifft2c(rotated)

    source_power = float(np.sum(np.abs(spectrum) ** 2))
    rotated_power = float(np.sum(np.abs(rotated) ** 2))
    ratio = rotated_power / max(source_power, EPS)
    return np.asarray(output_envelope, dtype=np.complex128), {
        "tilt_x_rad": tx,
        "tilt_y_rad": ty,
        "inverse": bool(inverse),
        "source_spectral_center_cpm": [fsx, fsy],
        "destination_spectral_center_cpm": [fdx, fdy],
        "source_spectral_center_z_cpm": float(fsz),
        "destination_spectral_center_z_cpm": float(destination_center[2]),
        "spectral_power_ratio": float(ratio),
        "spectral_clipped_fraction": float(max(0.0, 1.0 - ratio)),
        "interpolation_model": f"spline_order_{int(interpolation_order)}",
        "jacobian_model": "abs(fz_source/fz_destination)",
        "carrier_representation": "analytic_unsampled",
        "rotation_matrix": R.tolist(),
        "fidelity": "scalar_rotated_angular_spectrum_carrier_tracked_baseband",
    }


__all__ = ["rotate_baseband_angular_spectrum"]
