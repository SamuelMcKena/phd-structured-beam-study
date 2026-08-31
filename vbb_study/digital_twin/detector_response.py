"""Detector-response utilities for comparing numerical fields with camera data.

The q=20 BeamGage acquisition is sampled by a 5.5 um-pitch camera, whereas
several report figures are rendered on a much finer display grid.  Comparing a
high-resolution numerical intensity directly with an interpolated camera image
therefore gives the inverse solver access to spatial detail that the detector
never measured.

This module applies a simple, explicit square-pixel response: each detector
sample is the mean intensity over one pixel area (sub-pixel quadrature on the
numerical grid), followed by sampling on the detector lattice.  The sampled
camera image can then be interpolated to a requested display grid for plotting
or metric evaluation.  No free optical blur is introduced here.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

EPS = np.finfo(float).tiny


def detector_axis_for_display(axis_m: np.ndarray, pixel_pitch_m: float) -> np.ndarray:
    """Return a detector-centre lattice that fully covers ``axis_m``.

    The lattice is centred on zero because beam-frame comparisons place the
    measured core at the numerical optical axis.  A future calibrated camera
    origin may be supplied by shifting the returned detector coordinates before
    sampling.
    """
    axis = np.asarray(axis_m, dtype=float)
    if axis.ndim != 1 or axis.size < 2:
        raise ValueError("axis_m must be a one-dimensional coordinate vector")
    pitch = float(pixel_pitch_m)
    if not np.isfinite(pitch) or pitch <= 0:
        raise ValueError("pixel_pitch_m must be positive")
    limit = float(np.max(np.abs(axis))) + pitch
    half = int(np.ceil(limit / pitch))
    return np.arange(-half, half + 1, dtype=float) * pitch


def integrate_and_sample_square_pixels(
    intensity_stack: np.ndarray,
    native_axis_m: np.ndarray,
    detector_axis_m: np.ndarray,
    *,
    pixel_pitch_m: float,
    quadrature_n: int = 3,
) -> np.ndarray:
    """Integrate numerical intensity over square detector pixels.

    Parameters
    ----------
    intensity_stack:
        Array with shape ``(z, y, x)`` on a square Cartesian grid.
    native_axis_m:
        Physical coordinate vector for both native transverse axes.
    detector_axis_m:
        Detector pixel-centre coordinates for both camera axes.
    pixel_pitch_m:
        Square pixel pitch/width.
    quadrature_n:
        Number of equally weighted sub-samples per pixel dimension.  Three is
        sufficient for the present 5.5 um detector / ~4.9 um propagation grid
        and avoids pretending the detector has finer measured information.
    """
    stack = np.asarray(intensity_stack, dtype=float)
    x = np.asarray(native_axis_m, dtype=float)
    det = np.asarray(detector_axis_m, dtype=float)
    if stack.ndim != 3 or stack.shape[1:] != (x.size, x.size):
        raise ValueError("intensity_stack must have shape (z, n, n) matching native_axis_m")
    if x.size < 2 or not np.all(np.diff(x) > 0):
        raise ValueError("native_axis_m must be strictly increasing")
    if det.ndim != 1 or det.size < 2 or not np.all(np.diff(det) > 0):
        raise ValueError("detector_axis_m must be a strictly increasing vector")
    pitch = float(pixel_pitch_m)
    qn = int(quadrature_n)
    if pitch <= 0 or qn < 1:
        raise ValueError("pixel pitch and quadrature_n must be positive")

    dx = float(x[1] - x[0])
    # Midpoint quadrature over one physical square pixel.
    offsets = ((np.arange(qn, dtype=float) + 0.5) / qn - 0.5) * pitch
    sampled = np.zeros((stack.shape[0], det.size, det.size), dtype=float)
    for dy in offsets:
        ycoord = (det + dy - x[0]) / dx
        for dxoff in offsets:
            xcoord = (det + dxoff - x[0]) / dx
            yy, xx = np.meshgrid(ycoord, xcoord, indexing="ij")
            for iz, image in enumerate(stack):
                sampled[iz] += ndimage.map_coordinates(
                    image, [yy, xx], order=1, mode="constant", cval=0.0,
                )
    sampled /= float(qn * qn)
    return sampled


def interpolate_detector_to_display(
    detector_stack: np.ndarray,
    detector_axis_m: np.ndarray,
    display_axis_m: np.ndarray,
) -> np.ndarray:
    """Interpolate detector samples to a display/metric grid.

    This interpolation does not create new measured spatial information; it is
    only a common plotting/metric coordinate system, matching how the BMG data
    are already displayed in the q=20 workflow.
    """
    stack = np.asarray(detector_stack, dtype=float)
    det = np.asarray(detector_axis_m, dtype=float)
    display = np.asarray(display_axis_m, dtype=float)
    if stack.ndim != 3 or stack.shape[1:] != (det.size, det.size):
        raise ValueError("detector_stack shape must match detector_axis_m")
    step = float(det[1] - det[0])
    coord = (display - det[0]) / step
    yy, xx = np.meshgrid(coord, coord, indexing="ij")
    return np.stack([
        ndimage.map_coordinates(image, [yy, xx], order=1, mode="constant", cval=0.0)
        for image in stack
    ])


def sample_camera_response(
    intensity_stack: np.ndarray,
    native_axis_m: np.ndarray,
    display_axis_m: np.ndarray,
    *,
    pixel_pitch_m: float,
    quadrature_n: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply pixel-area integration, detector sampling, then display interpolation."""
    detector_axis = detector_axis_for_display(display_axis_m, pixel_pitch_m)
    detector = integrate_and_sample_square_pixels(
        intensity_stack, native_axis_m, detector_axis,
        pixel_pitch_m=pixel_pitch_m, quadrature_n=quadrature_n,
    )
    shown = interpolate_detector_to_display(detector, detector_axis, display_axis_m)
    return shown, detector_axis


def plane_normalise(stack: np.ndarray) -> np.ndarray:
    a = np.maximum(np.asarray(stack, dtype=float), 0.0)
    if a.ndim != 3:
        raise ValueError("stack must have shape (z,y,x)")
    return a / np.maximum(np.max(a, axis=(1, 2), keepdims=True), EPS)
