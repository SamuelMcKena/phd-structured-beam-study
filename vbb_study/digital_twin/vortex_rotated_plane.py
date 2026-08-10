"""Scalar rotated-angular-spectrum utilities for rigidly tilted optical planes.

The implementation follows the coordinate-rotation idea of Matsushima,
Schimmel & Wyrowski, JOSA A 20, 1755-1762 (2003): the plane-wave spectrum is
rotated in Fourier space and resampled on the regular spectral grid of the new
plane.  The Jacobian ``|fz/fz'|`` is included from invariance of solid angle.

A previous research implementation used bilinear spectral interpolation.  That
was adequate to expose morphology changes but introduced unphysical numerical
power loss in repeated tilted-plane transforms.  The production research path
therefore uses cubic-spline interpolation of the real and imaginary spectrum,
with explicit spectral-power bookkeeping and round-trip validation gates.

This remains a scalar propagating-wave model.  It does not replace vector
Fresnel/Snell treatment at a strongly tilted refractive interface.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
from scipy.ndimage import map_coordinates

from vbb_study.equations.fields import fft2c, ifft2c


EPS = np.finfo(float).tiny
DEFAULT_INTERPOLATION_ORDER = 3


def rotation_matrix(tilt_x_rad: float, tilt_y_rad: float) -> np.ndarray:
    """Return R mapping coordinates in the tilted plane frame into lab frame."""

    tx = float(tilt_x_rad)
    ty = float(tilt_y_rad)
    cx, sx = math.cos(tx), math.sin(tx)
    cy, sy = math.cos(ty), math.sin(ty)
    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=float)
    ry = np.asarray([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=float)
    return ry @ rx


def _spline_uniform_complex(
    data: np.ndarray,
    coordinate: np.ndarray,
    xq: np.ndarray,
    yq: np.ndarray,
    *,
    order: int = DEFAULT_INTERPOLATION_ORDER,
) -> np.ndarray:
    """Spline-resample complex data on a square, uniform spectral grid.

    Real and imaginary components are interpolated independently.  Cubic
    interpolation is the default because the old bilinear map was found to
    attenuate smooth spectra significantly during forward/inverse plane
    rotations.  Samples outside the represented spectrum are set to zero rather
    than extrapolated.
    """

    arr = np.asarray(data, dtype=np.complex128)
    c = np.asarray(coordinate, dtype=float)
    if c.ndim != 1 or arr.shape != (c.size, c.size):
        raise ValueError("expected square data on one shared coordinate axis")
    if c.size < 4:
        raise ValueError("spectral interpolation requires at least four samples")
    step = float(c[1] - c[0])
    if not np.allclose(
        np.diff(c),
        step,
        rtol=1e-9,
        atol=1e-12 * max(1.0, abs(step)),
    ):
        raise ValueError("spectral coordinate must be uniform")
    order_i = int(order)
    if order_i < 1 or order_i > 5:
        raise ValueError("spline interpolation order must lie in [1, 5]")

    ix = (np.asarray(xq, dtype=float) - c[0]) / step
    iy = (np.asarray(yq, dtype=float) - c[0]) / step
    query = np.vstack([iy.ravel(), ix.ravel()])

    real = map_coordinates(
        np.real(arr),
        query,
        order=order_i,
        mode="constant",
        cval=0.0,
        prefilter=order_i > 1,
    ).reshape(ix.shape)
    imag = map_coordinates(
        np.imag(arr),
        query,
        order=order_i,
        mode="constant",
        cval=0.0,
        prefilter=order_i > 1,
    ).reshape(ix.shape)
    return np.asarray(real + 1j * imag, dtype=np.complex128)


def rotate_angular_spectrum(
    field: np.ndarray,
    grid: Mapping[str, Any],
    *,
    wavelength_m: float,
    tilt_x_rad: float,
    tilt_y_rad: float,
    inverse: bool = False,
    interpolation_order: int = DEFAULT_INTERPOLATION_ORDER,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Resample a scalar angular spectrum between lab and tilted planes.

    The two planes intersect at their origins. ``inverse=False`` maps a field on
    the lab plane onto the tilted plane; ``inverse=True`` performs the inverse
    coordinate rotation.

    ``spectral_power_ratio`` is a numerical diagnostic, not an optical
    transmission coefficient.  A rigid coordinate transformation should remain
    close to unity for a sufficiently sampled, fully represented spectrum.
    """

    tx = float(tilt_x_rad)
    ty = float(tilt_y_rad)
    if tx == 0.0 and ty == 0.0:
        return np.asarray(field, dtype=np.complex128).copy(), {
            "tilt_x_rad": 0.0,
            "tilt_y_rad": 0.0,
            "inverse": bool(inverse),
            "spectral_clipped_fraction": 0.0,
            "spectral_power_ratio": 1.0,
            "interpolation_model": "identity",
            "jacobian_model": "identity",
            "fidelity": "exact_identity_parallel_plane",
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

    # For every regular spectral sample in the destination frame, rotate its
    # wavevector into the source frame and interpolate the source spectrum.
    fx_src = R[0, 0] * f + R[0, 1] * g + R[0, 2] * h_prime
    fy_src = R[1, 0] * f + R[1, 1] * g + R[1, 2] * h_prime
    fz_src = R[2, 0] * f + R[2, 1] * g + R[2, 2] * h_prime
    source_propagating = fz_src > 0.0

    spectrum = fft2c(np.asarray(field, dtype=np.complex128))
    sampled = _spline_uniform_complex(
        spectrum,
        axis,
        fx_src,
        fy_src,
        order=int(interpolation_order),
    )
    valid = propagating_prime & source_propagating

    # dfx dfy / fz is invariant under rotation of the wavevector sphere.
    jacobian = np.zeros_like(h_prime)
    jacobian[valid] = np.abs(fz_src[valid]) / np.maximum(h_prime[valid], EPS)
    rotated = np.where(valid, sampled * jacobian, 0.0)
    output = ifft2c(rotated)

    source_power = float(np.sum(np.abs(spectrum) ** 2))
    rotated_power = float(np.sum(np.abs(rotated) ** 2))
    power_ratio = rotated_power / max(source_power, EPS)
    interpolation_model = f"spline_order_{int(interpolation_order)}"
    return np.asarray(output, dtype=np.complex128), {
        "tilt_x_rad": tx,
        "tilt_y_rad": ty,
        "inverse": bool(inverse),
        # Backwards-compatible field: this is a numerical spectral-power
        # deficit diagnostic, not necessarily true physical clipping.
        "spectral_clipped_fraction": float(max(0.0, 1.0 - power_ratio)),
        "spectral_power_ratio": float(power_ratio),
        "interpolation_model": interpolation_model,
        "jacobian_model": "abs(fz_source/fz_destination)",
        "rotation_matrix": R.tolist(),
        "fidelity": (
            "scalar_rotated_angular_spectrum_propagating_components_"
            f"{interpolation_model}"
        ),
    }


def lab_to_tilted_plane(
    field: np.ndarray,
    grid: Mapping[str, Any],
    *,
    wavelength_m: float,
    tilt_x_rad: float,
    tilt_y_rad: float,
    interpolation_order: int = DEFAULT_INTERPOLATION_ORDER,
) -> tuple[np.ndarray, dict[str, Any]]:
    return rotate_angular_spectrum(
        field,
        grid,
        wavelength_m=wavelength_m,
        tilt_x_rad=tilt_x_rad,
        tilt_y_rad=tilt_y_rad,
        inverse=False,
        interpolation_order=interpolation_order,
    )


def tilted_to_lab_plane(
    field: np.ndarray,
    grid: Mapping[str, Any],
    *,
    wavelength_m: float,
    tilt_x_rad: float,
    tilt_y_rad: float,
    interpolation_order: int = DEFAULT_INTERPOLATION_ORDER,
) -> tuple[np.ndarray, dict[str, Any]]:
    return rotate_angular_spectrum(
        field,
        grid,
        wavelength_m=wavelength_m,
        tilt_x_rad=tilt_x_rad,
        tilt_y_rad=tilt_y_rad,
        inverse=True,
        interpolation_order=interpolation_order,
    )


__all__ = [
    "DEFAULT_INTERPOLATION_ORDER",
    "lab_to_tilted_plane",
    "rotate_angular_spectrum",
    "rotation_matrix",
    "tilted_to_lab_plane",
]
