"""Observation-frame utilities for propagation-stack comparison.

The camera sees the *relative* motion between the optical field and the sampled
camera/stage coordinate system.  Without an independently measured reference
axis versus stage position, a nearly linear trajectory in a Bessel z scan cannot
be uniquely assigned to beam pointing, residual carrier, optic tilt, or stage
runout.

This module therefore treats the affine part of the measured trajectory as an
explicit observation-frame nuisance transform.  It is kept separate from the
optical error model and must never be exported as an SLM correction.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import ndimage

EPS = np.finfo(float).tiny


@dataclass(frozen=True)
class AffineTrajectory:
    """Affine relative beam/camera trajectory in physical transverse units."""

    z: np.ndarray
    measured_yx: np.ndarray
    fitted_yx: np.ndarray
    residual_yx: np.ndarray
    slope_yx_per_z: np.ndarray
    intercept_yx: np.ndarray
    rms_residual_yx: np.ndarray


def fit_affine_trajectory(z: np.ndarray, measured_yx: np.ndarray, *, centre_fit: bool = True) -> AffineTrajectory:
    """Fit y(z), x(z) independently with one straight line.

    Parameters
    ----------
    z:
        One-dimensional longitudinal coordinate.  Units are arbitrary but must
        be used consistently with ``measured_yx`` when interpreting the slope.
    measured_yx:
        Array with shape ``(n_z, 2)`` containing [y, x] positions.
    centre_fit:
        When true, subtract the median fitted position so the transform is
        expressed around the common comparison frame rather than an arbitrary
        intercept at z=0.
    """
    z = np.asarray(z, float).ravel()
    yx = np.asarray(measured_yx, float)
    if yx.shape != (z.size, 2):
        raise ValueError("measured_yx must have shape (len(z), 2)")
    fitted = np.empty_like(yx)
    slopes = np.empty(2, float)
    intercepts = np.empty(2, float)
    for column in range(2):
        slope, intercept = np.polyfit(z, yx[:, column], 1)
        fitted[:, column] = slope * z + intercept
        slopes[column] = slope
        intercepts[column] = intercept
    if centre_fit:
        fitted -= np.median(fitted, axis=0, keepdims=True)
    residual = yx - fitted
    # The measured coordinates supplied by the q20 analysis are already centred
    # about their median.  Remove any tiny constant residual before reporting RMS.
    residual_for_rms = residual - np.median(residual, axis=0, keepdims=True)
    rms = np.sqrt(np.mean(residual_for_rms**2, axis=0))
    return AffineTrajectory(
        z=z,
        measured_yx=yx,
        fitted_yx=fitted,
        residual_yx=residual,
        slope_yx_per_z=slopes,
        intercept_yx=intercepts,
        rms_residual_yx=rms,
    )


def shift_stack_by_trajectory(
    stack: np.ndarray,
    axis_coordinate: np.ndarray,
    trajectory_yx: np.ndarray,
    *,
    inverse: bool = False,
    order: int = 1,
    renormalise_planes: bool = True,
) -> np.ndarray:
    """Translate every intensity plane by a supplied physical [y,x] trajectory.

    This is an observation-coordinate transform.  It does not alter phase and is
    not a substitute for propagating a genuine optical pointing or tilt model.
    It is useful when the relative camera/beam walk is known but its physical
    ownership (stage versus optics) is not yet calibrated.
    """
    a = np.asarray(stack, float)
    axis = np.asarray(axis_coordinate, float).ravel()
    yx = np.asarray(trajectory_yx, float)
    if a.ndim != 3 or a.shape[0] != yx.shape[0] or yx.shape[1:] != (2,):
        raise ValueError("stack must be (z,y,x) and trajectory_yx must be (z,2)")
    if axis.size != a.shape[1] or a.shape[1] != a.shape[2]:
        raise ValueError("axis_coordinate must match the square transverse stack")
    if axis.size < 2:
        raise ValueError("axis_coordinate needs at least two samples")
    du = float(np.median(np.diff(axis)))
    if not np.allclose(np.diff(axis), du, rtol=1e-7, atol=max(abs(du), 1.0)*1e-12):
        raise ValueError("axis_coordinate must be uniformly sampled")
    sign = -1.0 if inverse else 1.0
    out = np.empty_like(a, dtype=float)
    for iz, (dy, dx) in enumerate(yx):
        out[iz] = ndimage.shift(
            a[iz],
            shift=(sign * float(dy) / du, sign * float(dx) / du),
            order=int(order),
            mode="constant",
            cval=0.0,
            prefilter=(int(order) > 1),
        )
    if renormalise_planes:
        peak = np.maximum(out.reshape(out.shape[0], -1).max(axis=1), EPS)
        out = out / peak[:, None, None]
    return out


def ring_chord_branches(radius: np.ndarray, centre_yx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return fixed-axis x- and y-slice branch coordinates for translated rings.

    For a circular ring of radius R centred at (yc,xc), the laboratory y=0 XZ
    slice intersects at x = xc +/- sqrt(R^2-yc^2).  Likewise the x=0 YZ slice
    intersects at y = yc +/- sqrt(R^2-xc^2).  Planes with no geometrical
    intersection are returned as NaN.
    """
    r = np.asarray(radius, float)
    yx = np.asarray(centre_yx, float)
    if yx.shape != (r.size, 2):
        raise ValueError("centre_yx must have shape (len(radius), 2)")
    yc, xc = yx[:, 0], yx[:, 1]
    xhalf2 = r*r - yc*yc
    yhalf2 = r*r - xc*xc
    xhalf = np.where(xhalf2 >= 0.0, np.sqrt(np.maximum(xhalf2, 0.0)), np.nan)
    yhalf = np.where(yhalf2 >= 0.0, np.sqrt(np.maximum(yhalf2, 0.0)), np.nan)
    xbranches = np.column_stack((xc - xhalf, xc + xhalf))
    ybranches = np.column_stack((yc - yhalf, yc + yhalf))
    return xbranches, ybranches
