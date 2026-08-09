"""Scalar rotated-angular-spectrum utilities for rigidly tilted optical planes.

The implementation follows the coordinate-rotation idea of Matsushima,
Schimmel & Wyrowski, JOSA A 20, 1755-1762 (2003): the plane-wave spectrum is
rotated in Fourier space and resampled on the regular spectral grid of the
new plane.  The Jacobian ``|fz/fz'|`` is included from invariance of solid angle.

This is a scalar propagating-wave model.  It does not replace vector Fresnel /
Snell treatment at a strongly tilted refractive interface.  Its purpose here is
to stop representing a rigidly tilted optic by an arbitrary phase ramp on an
untilted plane.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from vbb_study.equations.fields import fft2c, ifft2c


EPS = np.finfo(float).tiny


def rotation_matrix(tilt_x_rad: float, tilt_y_rad: float) -> np.ndarray:
    """Return R mapping coordinates in the tilted plane frame into lab frame."""

    tx = float(tilt_x_rad)
    ty = float(tilt_y_rad)
    cx, sx = math.cos(tx), math.sin(tx)
    cy, sy = math.cos(ty), math.sin(ty)
    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=float)
    ry = np.asarray([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=float)
    return ry @ rx


def _bilinear_uniform_complex(
    data: np.ndarray,
    coordinate: np.ndarray,
    xq: np.ndarray,
    yq: np.ndarray,
) -> np.ndarray:
    """Bilinear interpolation of a complex array on a square uniform grid."""

    arr = np.asarray(data, dtype=np.complex128)
    c = np.asarray(coordinate, dtype=float)
    if c.ndim != 1 or arr.shape != (c.size, c.size):
        raise ValueError("expected square data on one shared coordinate axis")
    step = float(c[1] - c[0])
    if not np.allclose(np.diff(c), step, rtol=1e-9, atol=1e-12 * max(1.0, abs(step))):
        raise ValueError("spectral coordinate must be uniform")

    ix = (np.asarray(xq, dtype=float) - c[0]) / step
    iy = (np.asarray(yq, dtype=float) - c[0]) / step
    i0 = np.floor(ix).astype(np.int64)
    j0 = np.floor(iy).astype(np.int64)
    tx = ix - i0
    ty = iy - j0
    valid = (i0 >= 0) & (j0 >= 0) & (i0 < c.size - 1) & (j0 < c.size - 1)

    out = np.zeros(ix.shape, dtype=np.complex128)
    if not np.any(valid):
        return out
    iv = i0[valid]
    jv = j0[valid]
    ax = tx[valid]
    ay = ty[valid]
    v00 = arr[jv, iv]
    v10 = arr[jv, iv + 1]
    v01 = arr[jv + 1, iv]
    v11 = arr[jv + 1, iv + 1]
    out[valid] = (
        (1.0 - ax) * (1.0 - ay) * v00
        + ax * (1.0 - ay) * v10
        + (1.0 - ax) * ay * v01
        + ax * ay * v11
    )
    return out


def rotate_angular_spectrum(
    field: np.ndarray,
    grid: Mapping[str, Any],
    *,
    wavelength_m: float,
    tilt_x_rad: float,
    tilt_y_rad: float,
    inverse: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Resample a scalar angular spectrum between lab and tilted planes.

    The two planes intersect at their origins.  ``inverse=False`` maps a field
    represented on the lab plane onto the tilted plane.  ``inverse=True`` maps
    the tilted-plane field back to the lab plane.
    """

    tx = float(tilt_x_rad)
    ty = float(tilt_y_rad)
    if tx == 0.0 and ty == 0.0:
        return np.asarray(field, dtype=np.complex128).copy(), {
            "tilt_x_rad": 0.0,
            "tilt_y_rad": 0.0,
            "inverse": bool(inverse),
            "spectral_clipped_fraction": 0.0,
            "jacobian_model": "identity",
        }

    R = rotation_matrix(tx, ty)
    if inverse:
        R = R.T

    f = np.asarray(grid["FX"], dtype=float)
    g = np.asarray(grid["FY"], dtype=float)
    axis = np.asarray(grid.get("fx", np.unique(f[0])), dtype=float)
    if axis.ndim != 1 or axis.size != f.shape[1]:
        axis = np.asarray(grid["FX"][0], dtype=float)

    inv_lam = 1.0 / float(wavelength_m)
    transverse_sq_prime = f * f + g * g
    propagating_prime = transverse_sq_prime < inv_lam * inv_lam
    h_prime = np.sqrt(np.maximum(inv_lam * inv_lam - transverse_sq_prime, 0.0))

    # For each regular spectral sample in the destination frame, rotate its
    # wavevector into the source frame and interpolate the source spectrum.
    fx_src = R[0, 0] * f + R[0, 1] * g + R[0, 2] * h_prime
    fy_src = R[1, 0] * f + R[1, 1] * g + R[1, 2] * h_prime
    fz_src = R[2, 0] * f + R[2, 1] * g + R[2, 2] * h_prime
    source_propagating = fz_src > 0.0

    spectrum = fft2c(np.asarray(field, dtype=np.complex128))
    sampled = _bilinear_uniform_complex(spectrum, axis, fx_src, fy_src)
    valid = propagating_prime & source_propagating

    # From dfx dfy / fz = dfx' dfy' / fz' under rotation.
    jacobian = np.zeros_like(h_prime)
    jacobian[valid] = np.abs(fz_src[valid]) / np.maximum(h_prime[valid], EPS)
    rotated = np.where(valid, sampled * jacobian, 0.0)
    output = ifft2c(rotated)

    source_power = float(np.sum(np.abs(spectrum) ** 2))
    kept_power = float(np.sum(np.abs(rotated) ** 2))
    return np.asarray(output, dtype=np.complex128), {
        "tilt_x_rad": tx,
        "tilt_y_rad": ty,
        "inverse": bool(inverse),
        "spectral_clipped_fraction": float(max(0.0, 1.0 - kept_power / max(source_power, EPS))),
        "jacobian_model": "abs(fz_source/fz_destination)",
        "rotation_matrix": R.tolist(),
        "fidelity": "scalar_rotated_angular_spectrum_propagating_components",
    }


def lab_to_tilted_plane(
    field: np.ndarray,
    grid: Mapping[str, Any],
    *,
    wavelength_m: float,
    tilt_x_rad: float,
    tilt_y_rad: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    return rotate_angular_spectrum(
        field,
        grid,
        wavelength_m=wavelength_m,
        tilt_x_rad=tilt_x_rad,
        tilt_y_rad=tilt_y_rad,
        inverse=False,
    )


def tilted_to_lab_plane(
    field: np.ndarray,
    grid: Mapping[str, Any],
    *,
    wavelength_m: float,
    tilt_x_rad: float,
    tilt_y_rad: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    return rotate_angular_spectrum(
        field,
        grid,
        wavelength_m=wavelength_m,
        tilt_x_rad=tilt_x_rad,
        tilt_y_rad=tilt_y_rad,
        inverse=True,
    )
