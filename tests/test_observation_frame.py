from __future__ import annotations

import numpy as np

from vbb_study.digital_twin.observation_frame import (
    fit_affine_trajectory,
    ring_chord_branches,
    shift_stack_by_trajectory,
)


def _ring(axis, radius):
    X, Y = np.meshgrid(axis, axis, indexing="xy")
    R = np.hypot(X, Y)
    return np.exp(-0.5 * ((R - radius) / 2.0) ** 2)


def test_affine_trajectory_recovers_known_slope() -> None:
    z = np.arange(-17.0, 1.0)
    y = -5.35 * z - 44.0
    x = 0.83 * z + 7.0
    measured = np.column_stack((y, x))
    fit = fit_affine_trajectory(z, measured)
    assert np.allclose(fit.slope_yx_per_z, (-5.35, 0.83), atol=1e-12)
    assert np.allclose(fit.rms_residual_yx, 0.0, atol=1e-12)


def test_shift_stack_by_trajectory_roundtrip_for_smooth_ring() -> None:
    axis = np.linspace(-180.0, 180.0, 241)
    base = _ring(axis, 45.0)
    stack = np.repeat(base[None], 4, axis=0)
    trajectory = np.asarray([[0.0, 0.0], [8.0, -4.0], [-12.0, 6.0], [4.0, 10.0]])
    moved = shift_stack_by_trajectory(stack, axis, trajectory, renormalise_planes=False)
    restored = shift_stack_by_trajectory(moved, axis, trajectory, inverse=True, renormalise_planes=False)
    # Bilinear interpolation is not exactly reversible, but the central field
    # must remain highly correlated after the observation-frame round trip.
    for original, recovered in zip(stack, restored):
        r = np.corrcoef(original.ravel(), recovered.ravel())[0, 1]
        assert r > 0.995


def test_ring_chord_geometry_explains_apparent_hourglass() -> None:
    radius = np.full(5, 45.0)
    centres = np.asarray([
        [-40.0, 0.0],
        [-20.0, 0.0],
        [0.0, 0.0],
        [20.0, 0.0],
        [40.0, 0.0],
    ])
    xbranches, ybranches = ring_chord_branches(radius, centres)
    xhalf = 0.5 * (xbranches[:, 1] - xbranches[:, 0])
    assert np.isclose(xhalf[2], 45.0)
    assert xhalf[0] < xhalf[1] < xhalf[2]
    assert xhalf[4] < xhalf[3] < xhalf[2]
    # The YZ branches translate with the centre while retaining nearly the
    # full ring separation because x-centre is zero.
    assert np.allclose(ybranches[:, 1] - ybranches[:, 0], 90.0)
