from __future__ import annotations

import math

import numpy as np

from vbb_study.digital_twin.vector_refractive_axicon_eikonal import (
    build_tilted_vector_refractive_axicon_field,
)
from vbb_study.digital_twin.vortex_refractive_axicon import (
    RefractiveAxiconGeometry,
    trace_refractive_axicon_bundle,
)
from vbb_study.equations.fields import make_xy_grid
from vbb_study.vector_field import VectorField


WAVELENGTH = 1.029e-6
N_AX = 1.458
N_EXT = 1.0


def _geometry() -> RefractiveAxiconGeometry:
    return RefractiveAxiconGeometry(
        base_angle_rad=math.radians(2.0),
        clear_radius_m=1.2e-3,
        centre_thickness_m=2.0e-3,
        refractive_index=N_AX,
        external_index=N_EXT,
    )


def _plane_wave(n: int = 128, window_m: float = 3.0e-3) -> VectorField:
    grid = make_xy_grid(n, window_m / n)
    ex = np.full((n, n), 1.0 / np.sqrt(2.0), dtype=np.complex128)
    ey = 1j * ex
    return VectorField(
        ex=ex,
        ey=ey,
        ez=np.zeros_like(ex),
        grid=grid,
        wavelength_m=WAVELENGTH,
        medium_index=N_EXT,
    )


def test_plane_wave_vector_geometry_matches_independent_scalar_two_surface_tracer_at_oblique_tilts() -> None:
    """The vector solver must not invent a different ray geometry.

    For a uniform plane wave, polarization changes Fresnel amplitudes but not the
    isotropic Snell ray path.  This test compares the Phase-2H ray bundle directly
    against the pre-existing independently validated scalar two-surface tracer at
    0, 5 and 10 degrees.  Only common valid entrance samples are compared.
    """

    geometry = _geometry()
    source = _plane_wave()
    X = np.asarray(source.grid["X"], dtype=float)
    Y = np.asarray(source.grid["Y"], dtype=float)
    incident_lab = np.asarray([0.0, 0.0, 1.0])

    for tilt_deg in (0.0, 5.0, 10.0):
        tilt = math.radians(tilt_deg)
        vector = build_tilted_vector_refractive_axicon_field(
            source,
            geometry=geometry,
            tilt_x_rad=0.0,
            tilt_y_rad=tilt,
            reference_gap_m=0.20e-3,
            output_n=256,
            output_window_m=3.0e-3,
        )
        scalar = trace_refractive_axicon_bundle(
            X,
            Y,
            geometry=geometry,
            tilt_x_rad=0.0,
            tilt_y_rad=tilt,
            incident_direction_lab=incident_lab,
            reference_gap_m=0.20e-3,
            apex_exclusion_radius_m=0.0,
        )
        common = np.asarray(vector.geometry_bundle.valid, dtype=bool) & np.asarray(scalar.valid, dtype=bool)
        assert np.count_nonzero(common) > 1000

        vector_out = np.asarray(vector.outgoing_direction_lab, dtype=float)[common]
        scalar_out = np.asarray(scalar.outgoing_lab, dtype=float)[common]
        angular_vector_error = np.linalg.norm(vector_out - scalar_out, axis=1)
        assert float(np.max(angular_vector_error)) < 2e-8

        vector_exit = np.asarray(vector.geometry_bundle.exit_point_lab_m, dtype=float)[common]
        scalar_exit = np.asarray(scalar.exit_point_lab_m, dtype=float)[common]
        exit_position_error = np.linalg.norm(vector_exit - scalar_exit, axis=1)
        assert float(np.max(exit_position_error)) < 2e-9

        # Interface geometry is polarization independent; both implementations
        # must also agree on the physical path through the glass.
        vector_distance = np.asarray(vector.geometry_bundle.internal_distance_m, dtype=float)[common]
        scalar_distance = np.asarray(scalar.internal_distance_m, dtype=float)[common]
        assert float(np.max(np.abs(vector_distance - scalar_distance))) < 2e-9
