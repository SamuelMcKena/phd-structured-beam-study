"""Coordinate-aware physical-grid alignment for Python/Zemax comparisons."""
from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .models import FieldPlane


def _interpolate(values: np.ndarray, x: np.ndarray, y: np.ndarray, x_new: np.ndarray, y_new: np.ndarray) -> np.ndarray:
    yy, xx = np.meshgrid(y_new, x_new, indexing="ij")
    points = np.column_stack([yy.ravel(), xx.ravel()])
    arr = np.asarray(values)
    if np.iscomplexobj(arr):
        real = RegularGridInterpolator((y, x), arr.real, bounds_error=True)(points)
        imag = RegularGridInterpolator((y, x), arr.imag, bounds_error=True)(points)
        return (real + 1j * imag).reshape(y_new.size, x_new.size)
    out = RegularGridInterpolator((y, x), arr, bounds_error=True)(points)
    return out.reshape(y_new.size, x_new.size)


def resample_plane(plane: FieldPlane, x_new: np.ndarray, y_new: np.ndarray, *, method: str = "linear") -> FieldPlane:
    if method != "linear":
        raise ValueError("only coordinate-aware linear interpolation is currently supported")
    x_new = np.asarray(x_new, dtype=float)
    y_new = np.asarray(y_new, dtype=float)
    changes: dict[str, object] = {"x_m": x_new, "y_m": y_new}
    for name in ("Ex", "Ey", "Ez", "intensity", "phase"):
        value = getattr(plane, name)
        changes[name] = None if value is None else _interpolate(value, plane.x_m, plane.y_m, x_new, y_new)
    changes["metadata"] = {**dict(plane.metadata), "resampled": True, "interpolation_method": method, "original_x_extent_m": [float(plane.x_m[0]), float(plane.x_m[-1])], "original_y_extent_m": [float(plane.y_m[0]), float(plane.y_m[-1])]}
    return plane.copy_with(**changes)


def align_field_planes(python_plane: FieldPlane, zemax_plane: FieldPlane, *, comparison_grid: Literal["python", "zemax"] = "python") -> tuple[FieldPlane, FieldPlane, dict[str, object]]:
    if not np.isclose(python_plane.wavelength_m, zemax_plane.wavelength_m, rtol=1e-6, atol=1e-12):
        raise ValueError("Python and Zemax planes have different wavelengths")
    xmin = max(float(python_plane.x_m[0]), float(zemax_plane.x_m[0]))
    xmax = min(float(python_plane.x_m[-1]), float(zemax_plane.x_m[-1]))
    ymin = max(float(python_plane.y_m[0]), float(zemax_plane.y_m[0]))
    ymax = min(float(python_plane.y_m[-1]), float(zemax_plane.y_m[-1]))
    if xmin >= xmax or ymin >= ymax:
        raise ValueError("Python and Zemax physical grids do not overlap")
    source = python_plane if comparison_grid == "python" else zemax_plane
    x_common = source.x_m[(source.x_m >= xmin) & (source.x_m <= xmax)]
    y_common = source.y_m[(source.y_m >= ymin) & (source.y_m <= ymax)]
    if x_common.size < 2 or y_common.size < 2:
        raise ValueError("overlap contains too few samples for a physical comparison")
    py = resample_plane(python_plane, x_common, y_common) if not (np.array_equal(x_common, python_plane.x_m) and np.array_equal(y_common, python_plane.y_m)) else python_plane
    ze = resample_plane(zemax_plane, x_common, y_common) if not (np.array_equal(x_common, zemax_plane.x_m) and np.array_equal(y_common, zemax_plane.y_m)) else zemax_plane
    metadata = {"comparison_grid": comparison_grid, "comparison_x_extent_m": [float(x_common[0]), float(x_common[-1])], "comparison_y_extent_m": [float(y_common[0]), float(y_common[-1])], "comparison_shape": [int(y_common.size), int(x_common.size)], "interpolation_method": "linear", "registration_applied": False, "removed_degrees_of_freedom": []}
    return py, ze, metadata
