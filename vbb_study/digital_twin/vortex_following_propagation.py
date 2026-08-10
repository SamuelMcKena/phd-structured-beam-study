"""Beam-following dense propagation diagnostics for system-error figures.

A fixed laboratory x-z/y-z slice is useful for steering, but it is a poor beam
morphology diagnostic once the beam is translated: the orthogonal slice can
simply miss the beam and appear blank.  This module evaluates the same discrete
angular spectrum on transverse coordinates that follow a supplied x(z), y(z)
beam axis.  No image interpolation is used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from vbb_study.equations.fields import fft2c
from vbb_study.equations.propagation import (
    asm_longitudinal_wavenumber_m_inv,
    bandlimit_mask_matsushima,
)


EPS = np.finfo(float).tiny


@dataclass(frozen=True)
class BeamFollowingPropagationMap:
    transverse_offset_m: np.ndarray = field(repr=False, compare=False)
    z_m: np.ndarray = field(repr=False, compare=False)
    x_axis_m: np.ndarray = field(repr=False, compare=False)
    y_axis_m: np.ndarray = field(repr=False, compare=False)
    xz_intensity: np.ndarray = field(repr=False, compare=False)
    yz_intensity: np.ndarray = field(repr=False, compare=False)
    metadata: Mapping[str, Any]


def _phase_matrix(
    coordinates_m: np.ndarray,
    *,
    n: int,
    dx_m: float,
    centre_coordinate_m: float,
) -> np.ndarray:
    frequency_index = np.arange(int(n), dtype=float) - int(n) / 2.0
    sample_coordinate = (
        np.asarray(coordinates_m, dtype=float) - float(centre_coordinate_m)
    ) / float(dx_m)
    return np.exp(2j * np.pi * np.outer(frequency_index, sample_coordinate) / float(n))


def _field_line_x(
    spectrum: np.ndarray,
    transfer: np.ndarray,
    *,
    x_coordinates_m: np.ndarray,
    y_coordinate_m: float,
    n: int,
    dx_m: float,
    centre_coordinate_m: float,
) -> np.ndarray:
    phase_x = _phase_matrix(
        np.asarray(x_coordinates_m, dtype=float),
        n=n,
        dx_m=dx_m,
        centre_coordinate_m=centre_coordinate_m,
    )
    phase_y = _phase_matrix(
        np.asarray([float(y_coordinate_m)]),
        n=n,
        dx_m=dx_m,
        centre_coordinate_m=centre_coordinate_m,
    )[:, 0]
    propagated = np.asarray(spectrum, dtype=np.complex128) * transfer
    line_spectrum = phase_y @ propagated
    return np.asarray(line_spectrum @ phase_x / float(n * n))


def _field_line_y(
    spectrum: np.ndarray,
    transfer: np.ndarray,
    *,
    y_coordinates_m: np.ndarray,
    x_coordinate_m: float,
    n: int,
    dx_m: float,
    centre_coordinate_m: float,
) -> np.ndarray:
    phase_y = _phase_matrix(
        np.asarray(y_coordinates_m, dtype=float),
        n=n,
        dx_m=dx_m,
        centre_coordinate_m=centre_coordinate_m,
    )
    phase_x = _phase_matrix(
        np.asarray([float(x_coordinate_m)]),
        n=n,
        dx_m=dx_m,
        centre_coordinate_m=centre_coordinate_m,
    )[:, 0]
    propagated = np.asarray(spectrum, dtype=np.complex128) * transfer
    line_spectrum = propagated @ phase_x
    return np.asarray(line_spectrum @ phase_y / float(n * n))


def line_centroid_m(intensity: np.ndarray, coordinates_m: Sequence[float]) -> np.ndarray:
    """Intensity centroid of every z row in a longitudinal line map."""

    values = np.maximum(np.asarray(intensity, dtype=float), 0.0)
    coordinate = np.asarray(coordinates_m, dtype=float)
    if values.ndim != 2 or values.shape[1] != coordinate.size:
        raise ValueError("line intensity shape does not match coordinates")
    power = np.sum(values, axis=1)
    centroid = np.sum(values * coordinate[None, :], axis=1) / np.maximum(power, EPS)
    return np.asarray(centroid, dtype=float)


def robust_axis_path_m(
    intensity: np.ndarray,
    coordinates_m: Sequence[float],
    *,
    peak_floor_fraction: float = 0.05,
) -> np.ndarray:
    """Track a line centroid while avoiding noise-only z rows.

    Rows whose peak is below ``peak_floor_fraction`` of the global line-map peak
    are linearly filled from neighbouring valid rows.  This prevents a vanishing
    far-tail from throwing the beam-following slice to arbitrary coordinates.
    """

    values = np.maximum(np.asarray(intensity, dtype=float), 0.0)
    centroid = line_centroid_m(values, coordinates_m)
    peak = np.max(values, axis=1)
    valid = peak >= float(peak_floor_fraction) * max(float(np.max(peak)), EPS)
    index = np.arange(values.shape[0], dtype=float)
    if int(np.count_nonzero(valid)) < 2:
        return np.zeros(values.shape[0], dtype=float)
    filled = np.interp(index, index[valid], centroid[valid])
    return np.asarray(filled, dtype=float)


def build_beam_following_propagation(
    *,
    grid: Mapping[str, Any],
    wavelength_m: float,
    z_values_m: Sequence[float],
    transverse_offsets_m: Sequence[float],
    scalar_field: np.ndarray,
    x_axis_m: Sequence[float] | float = 0.0,
    y_axis_m: Sequence[float] | float = 0.0,
    n_medium: float = 1.0,
    source_label: str,
) -> BeamFollowingPropagationMap:
    """Evaluate x-z and y-z maps in coordinates relative to a tracked axis.

    For every z plane the x-z line is sampled at ``y=y_axis(z)`` over
    ``x=x_axis(z)+offset``.  The y-z line is sampled analogously at
    ``x=x_axis(z)``.  This is a direct Fourier-series field synthesis from the
    angular spectrum and therefore does not shift/interpolate rendered images.
    """

    n = int(grid["N"])
    dx_m = float(grid["dx"])
    field = np.asarray(scalar_field, dtype=np.complex128)
    if field.shape != (n, n):
        raise ValueError("scalar field must match the declared square grid")
    z = np.asarray(z_values_m, dtype=float)
    offsets = np.asarray(transverse_offsets_m, dtype=float)
    if z.ndim != 1 or offsets.ndim != 1:
        raise ValueError("z and transverse offsets must be one-dimensional")
    if z.size < 16 or offsets.size < 64:
        raise ValueError("beam-following diagnostics require at least 64x16 samples")

    def _path(value: Sequence[float] | float) -> np.ndarray:
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            return np.full(z.size, float(arr), dtype=float)
        if arr.shape != z.shape:
            raise ValueError("axis path must be scalar or match z_values_m")
        return arr

    x_axis = _path(x_axis_m)
    y_axis = _path(y_axis_m)
    spectrum = fft2c(field)
    kz = asm_longitudinal_wavenumber_m_inv(
        np.asarray(grid["FX"]),
        np.asarray(grid["FY"]),
        wavelength_m=float(wavelength_m) / float(n_medium),
        include_evanescent=True,
    )
    native = np.asarray(grid["x"], dtype=float)
    centre_coordinate = float(native[n // 2])

    xz = np.empty((z.size, offsets.size), dtype=float)
    yz = np.empty_like(xz)
    for iz, zz in enumerate(z):
        transfer = np.exp(1j * kz * float(zz))
        transfer *= bandlimit_mask_matsushima(
            dict(grid), float(wavelength_m), float(zz), n_medium=float(n_medium)
        )
        x_line = _field_line_x(
            spectrum,
            transfer,
            x_coordinates_m=x_axis[iz] + offsets,
            y_coordinate_m=float(y_axis[iz]),
            n=n,
            dx_m=dx_m,
            centre_coordinate_m=centre_coordinate,
        )
        y_line = _field_line_y(
            spectrum,
            transfer,
            y_coordinates_m=y_axis[iz] + offsets,
            x_coordinate_m=float(x_axis[iz]),
            n=n,
            dx_m=dx_m,
            centre_coordinate_m=centre_coordinate,
        )
        xz[iz] = np.abs(x_line) ** 2
        yz[iz] = np.abs(y_line) ** 2

    return BeamFollowingPropagationMap(
        transverse_offset_m=offsets,
        z_m=z,
        x_axis_m=x_axis,
        y_axis_m=y_axis,
        xz_intensity=xz,
        yz_intensity=yz,
        metadata={
            "outcome": "BEAM-FOLLOWING-DENSE-SPECTRAL-PROPAGATION",
            "source_label": str(source_label),
            "sampling_model": "direct_Fourier_series_no_image_interpolation",
            "axis_model": "supplied_xz_yz_path",
        },
    )


__all__ = [
    "BeamFollowingPropagationMap",
    "build_beam_following_propagation",
    "line_centroid_m",
    "robust_axis_path_m",
]
