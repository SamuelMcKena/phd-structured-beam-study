"""Explicit parallel-plane 4F relay for physical error studies.

Unlike the historical collapsed ``FFT -> circular mask -> IFFT`` contract, this
module propagates through two thin lenses and a physical Fourier-plane iris.  It
therefore supports lens/iris despace, lens decentre, finite lens aperture and
measured/prescribed lens wavefront error maps.

Lens *tilt* is intentionally not approximated here.  Rigid tilt changes the
orientation of the optical plane and belongs to the rotated-angular-spectrum
backend in ``vortex_rotated_plane.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from vbb_study.equations.propagation import angular_spectrum_propagate_bl


EPS = np.finfo(float).tiny
TWOPI = 2.0 * np.pi


@dataclass(frozen=True)
class LensError:
    focal_length_scale: float = 1.0
    decentre_m: tuple[float, float] = (0.0, 0.0)
    clear_radius_m: float | None = None

    def validate(self) -> None:
        if self.focal_length_scale <= 0.0:
            raise ValueError("lens focal_length_scale must be positive")
        if self.clear_radius_m is not None and self.clear_radius_m <= 0.0:
            raise ValueError("lens clear radius must be positive")


@dataclass(frozen=True)
class FourFError:
    """Errors of a nominal symmetric 4F relay with fixed object/output planes.

    Positive ``lens1_axial_shift_m`` moves L1 away from the SLM/input plane and
    therefore reduces the L1->iris distance by the same amount.  Positive
    ``lens2_axial_shift_m`` moves L2 away from the iris and reduces the final
    L2->output distance.  This keeps the nominal SLM, iris and output planes
    fixed while physically moving the lenses.
    """

    lens1: LensError = LensError()
    lens2: LensError = LensError()
    lens1_axial_shift_m: float = 0.0
    lens2_axial_shift_m: float = 0.0
    iris_offset_m: tuple[float, float] = (0.0, 0.0)
    iris_radius_scale: float = 1.0

    def validate(self, nominal_focal_length_m: float) -> None:
        self.lens1.validate()
        self.lens2.validate()
        if self.iris_radius_scale <= 0.0:
            raise ValueError("iris_radius_scale must be positive")
        f = float(nominal_focal_length_m)
        distances = (
            f + float(self.lens1_axial_shift_m),
            f - float(self.lens1_axial_shift_m),
            f + float(self.lens2_axial_shift_m),
            f - float(self.lens2_axial_shift_m),
        )
        if min(distances) <= 0.0:
            raise ValueError("4F lens shift produces a non-positive propagation distance")


def field_power(field: np.ndarray, grid: Mapping[str, Any]) -> float:
    dx = float(grid["dx"])
    return float(np.sum(np.abs(np.asarray(field, dtype=np.complex128)) ** 2) * dx * dx)


def thin_lens_transmission(
    grid: Mapping[str, Any],
    *,
    wavelength_m: float,
    focal_length_m: float,
    decentre_m: tuple[float, float] = (0.0, 0.0),
    clear_radius_m: float | None = None,
    opd_map_m: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Ideal paraxial thin lens plus optional physical aperture and OPD map."""

    X = np.asarray(grid["X"], dtype=float) - float(decentre_m[0])
    Y = np.asarray(grid["Y"], dtype=float) - float(decentre_m[1])
    k0 = TWOPI / float(wavelength_m)
    phase = -k0 * (X * X + Y * Y) / (2.0 * float(focal_length_m))
    transmission = np.exp(1j * phase)
    if clear_radius_m is not None:
        transmission = transmission * ((X * X + Y * Y) <= float(clear_radius_m) ** 2)
    if opd_map_m is not None:
        opd = np.asarray(opd_map_m, dtype=float)
        if opd.shape != transmission.shape:
            raise ValueError("lens OPD map shape does not match simulation grid")
        transmission = transmission * np.exp(1j * k0 * opd)
        opd_status = "measured_or_user_supplied"
    else:
        opd_status = "none"
    return np.asarray(transmission, dtype=np.complex128), {
        "focal_length_m": float(focal_length_m),
        "decentre_m": tuple(map(float, decentre_m)),
        "clear_radius_m": None if clear_radius_m is None else float(clear_radius_m),
        "opd_map_status": opd_status,
        "lens_tilt_model": "not_in_parallel_plane_backend",
    }


def physical_iris(
    grid: Mapping[str, Any],
    *,
    radius_m: float,
    offset_m: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    X = np.asarray(grid["X"], dtype=float) - float(offset_m[0])
    Y = np.asarray(grid["Y"], dtype=float) - float(offset_m[1])
    return (X * X + Y * Y <= float(radius_m) ** 2).astype(float)


def _propagate(
    field: np.ndarray,
    grid: Mapping[str, Any],
    wavelength_m: float,
    distance_m: float,
) -> np.ndarray:
    if abs(float(distance_m)) < 1e-15:
        return np.asarray(field, dtype=np.complex128)
    return np.asarray(
        angular_spectrum_propagate_bl(
            np.asarray(field, dtype=np.complex128),
            dict(grid),
            float(wavelength_m),
            float(distance_m),
            n_medium=1.0,
            bandlimit=True,
            include_evanescent=True,
        ),
        dtype=np.complex128,
    )


def explicit_4f_relay(
    field_in: np.ndarray,
    grid: Mapping[str, Any],
    *,
    wavelength_m: float,
    nominal_focal_length_m: float,
    nominal_iris_radius_m: float,
    error: FourFError = FourFError(),
    lens1_opd_map_m: np.ndarray | None = None,
    lens2_opd_map_m: np.ndarray | None = None,
) -> dict[str, Any]:
    """Propagate through SLM/object plane -> L1 -> iris -> L2 -> output plane."""

    error.validate(float(nominal_focal_length_m))
    f = float(nominal_focal_length_m)
    d_obj_l1 = f + float(error.lens1_axial_shift_m)
    d_l1_iris = f - float(error.lens1_axial_shift_m)
    d_iris_l2 = f + float(error.lens2_axial_shift_m)
    d_l2_out = f - float(error.lens2_axial_shift_m)

    f1 = f * float(error.lens1.focal_length_scale)
    f2 = f * float(error.lens2.focal_length_scale)
    lens1, lens1_meta = thin_lens_transmission(
        grid,
        wavelength_m=wavelength_m,
        focal_length_m=f1,
        decentre_m=error.lens1.decentre_m,
        clear_radius_m=error.lens1.clear_radius_m,
        opd_map_m=lens1_opd_map_m,
    )
    lens2, lens2_meta = thin_lens_transmission(
        grid,
        wavelength_m=wavelength_m,
        focal_length_m=f2,
        decentre_m=error.lens2.decentre_m,
        clear_radius_m=error.lens2.clear_radius_m,
        opd_map_m=lens2_opd_map_m,
    )

    u0 = np.asarray(field_in, dtype=np.complex128)
    p0 = field_power(u0, grid)
    pre_l1 = _propagate(u0, grid, wavelength_m, d_obj_l1)
    post_l1 = pre_l1 * lens1
    pre_iris = _propagate(post_l1, grid, wavelength_m, d_l1_iris)

    iris_radius = float(nominal_iris_radius_m) * float(error.iris_radius_scale)
    iris = physical_iris(grid, radius_m=iris_radius, offset_m=error.iris_offset_m)
    pre_iris_power = field_power(pre_iris, grid)
    post_iris = pre_iris * iris
    selected_fraction = field_power(post_iris, grid) / max(pre_iris_power, EPS)

    pre_l2 = _propagate(post_iris, grid, wavelength_m, d_iris_l2)
    post_l2 = pre_l2 * lens2
    output = _propagate(post_l2, grid, wavelength_m, d_l2_out)

    return {
        "input": u0,
        "pre_lens1": pre_l1,
        "post_lens1": post_l1,
        "fourier_plane_before_iris": pre_iris,
        "iris_mask": iris,
        "fourier_plane_after_iris": post_iris,
        "pre_lens2": pre_l2,
        "post_lens2": post_l2,
        "output": output,
        "metadata": {
            "model": "explicit_parallel_plane_4f_ASM",
            "nominal_focal_length_m": f,
            "distances_m": {
                "object_to_lens1": d_obj_l1,
                "lens1_to_iris": d_l1_iris,
                "iris_to_lens2": d_iris_l2,
                "lens2_to_output": d_l2_out,
            },
            "lens1": lens1_meta,
            "lens2": lens2_meta,
            "iris_radius_m": iris_radius,
            "iris_offset_m": tuple(map(float, error.iris_offset_m)),
            "iris_selected_power_fraction": float(selected_fraction),
            "input_power": float(p0),
            "output_power": float(field_power(output, grid)),
            "lens_tilt_status": "requires_rotated_plane_backend",
        },
    }
