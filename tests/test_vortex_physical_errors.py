from __future__ import annotations

import math

import numpy as np

from vbb_study.digital_twin.vortex_physical_errors import (
    PhysicalPerturbation,
    axicon_sag_m,
    build_physical_route_checkpoints,
    incident_plane_wave_phase,
    physical_axicon_transmission,
)
from vbb_study.digital_twin.vortex_system_route import build_system_route
from vbb_study.equations.fields import make_xy_grid


def _normalised_overlap(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.complex128).ravel()
    bb = np.asarray(b, dtype=np.complex128).ravel()
    denom = np.linalg.norm(aa) * np.linalg.norm(bb)
    return float(abs(np.vdot(aa, bb)) / max(float(denom), np.finfo(float).tiny))


def test_nominal_legacy_api_is_exact_canonical_wrapper() -> None:
    legacy = build_physical_route_checkpoints("V1", grid_n=256)
    canonical = build_system_route("V1", grid_n=256)
    assert legacy["metadata"]["compatibility_wrapper"] is True
    assert legacy["metadata"]["canonical_route_id"] == canonical["metadata"]["route_id"]
    assert _normalised_overlap(legacy["post_axicon"], canonical["post_axicon"]) > 1.0 - 1e-12


def test_input_angle_is_applied_before_slm_and_changes_upstream_field() -> None:
    nominal = build_physical_route_checkpoints("B0", grid_n=256)
    tilted = build_physical_route_checkpoints(
        "B0", grid_n=256, perturbation=PhysicalPerturbation(input_beam_angle_rad=(1.0e-3, 0.0))
    )
    assert tilted["metadata"]["input_angle_applied_plane"] == "before_SLM1"
    assert not np.allclose(nominal["raw_input"], tilted["raw_input"])
    assert not np.allclose(nominal["post_filter"], tilted["post_filter"])


def test_input_plane_wave_uses_direction_cosines() -> None:
    grid = make_xy_grid(128, 10e-3 / 128)
    wavelength = 1029e-9
    angle = 0.01
    phase = incident_plane_wave_phase(grid, wavelength, angle, 0.0)
    x = np.asarray(grid["x"], dtype=float)
    row = phase[64]
    measured_step = np.angle(row[65] * np.conj(row[64]))
    expected_step = 2.0 * math.pi / wavelength * math.sin(angle) * (x[65] - x[64])
    expected_step = math.atan2(math.sin(expected_step), math.cos(expected_step))
    assert np.isclose(measured_step, expected_step, atol=1e-12)


def test_beam_radius_change_is_rebuilt_before_slm() -> None:
    small = build_physical_route_checkpoints(
        "B0", grid_n=256, perturbation=PhysicalPerturbation(beam_radius_scale=0.7)
    )
    large = build_physical_route_checkpoints(
        "B0", grid_n=256, perturbation=PhysicalPerturbation(beam_radius_scale=1.3)
    )
    assert small["metadata"]["beam_radius_m"] < large["metadata"]["beam_radius_m"]
    assert not np.allclose(np.abs(small["raw_input"]), np.abs(large["raw_input"]))
    assert not np.allclose(small["post_filter"], large["post_filter"])


def test_hyperboloidal_tip_has_zero_central_slope_and_conical_asymptote() -> None:
    gamma = math.radians(2.0)
    r = np.linspace(0.0, 3e-3, 20001)
    rounded = axicon_sag_m(r, gamma, tip_model="hyperboloidal_round", rounding_parameter_m=10e-6)
    sharp = axicon_sag_m(r, gamma, tip_model="sharp")
    central_slope = (rounded[1] - rounded[0]) / (r[1] - r[0])
    outer_slope = (rounded[-1] - rounded[-101]) / (r[-1] - r[-101])
    assert abs(central_slope) < 0.05 * math.tan(gamma)
    assert np.isclose(outer_slope, math.tan(gamma), rtol=0.0, atol=3e-4)
    assert np.max(np.abs((rounded[-100:] - rounded[-1]) - (sharp[-100:] - sharp[-1]))) < 1e-7


def test_flat_blunt_tip_has_zero_sag_in_declared_flat_radius() -> None:
    r = np.linspace(0.0, 500e-6, 1001)
    sag = axicon_sag_m(
        r, math.radians(2.0), tip_model="flat_blunt", flat_tip_radius_m=100e-6
    )
    assert np.allclose(sag[r <= 100e-6], 0.0)
    assert np.any(sag[r > 100e-6] > 0.0)


def test_parallel_axicon_transmission_supports_physical_decentre() -> None:
    grid = make_xy_grid(256, 10e-3 / 256)
    nominal, _ = physical_axicon_transmission(
        grid, wavelength_m=1029e-9, refractive_index=1.458,
        external_index=1.0, base_angle_rad=math.radians(2.0)
    )
    shifted, _ = physical_axicon_transmission(
        grid, wavelength_m=1029e-9, refractive_index=1.458,
        external_index=1.0, base_angle_rad=math.radians(2.0), decentre_m=(200e-6, 0.0)
    )
    assert not np.allclose(nominal, shifted)
    assert np.allclose(np.abs(nominal), 1.0)
    assert np.allclose(np.abs(shifted), 1.0)


def test_standalone_transmission_refuses_rigid_axicon_tilt() -> None:
    grid = make_xy_grid(128, 10e-3 / 128)
    try:
        physical_axicon_transmission(
            grid, wavelength_m=1029e-9, refractive_index=1.458,
            external_index=1.0, base_angle_rad=math.radians(2.0),
            tilt_rad=(0.0, math.radians(0.25)),
        )
    except ValueError as exc:
        assert "tilt" in str(exc).lower() and "route" in str(exc).lower()
    else:  # pragma: no cover
        raise AssertionError("standalone lab-plane transmission must refuse rigid axicon tilt")


def test_legacy_route_delegates_axicon_tilt_to_rotated_plane_canonical_route() -> None:
    nominal = build_physical_route_checkpoints("B0", grid_n=256)
    tilted = build_physical_route_checkpoints(
        "B0", grid_n=256,
        perturbation=PhysicalPerturbation(axicon_tilt_rad=(0.0, math.radians(0.25))),
    )
    assert tilted["metadata"]["compatibility_wrapper"] is True
    assert tilted["metadata"]["axicon_tilt_model"] == "scalar_rotated_angular_spectrum"
    assert not np.allclose(nominal["post_axicon"], tilted["post_axicon"])


def test_large_axicon_tilt_is_refused_by_canonical_fidelity_gate() -> None:
    perturbation = PhysicalPerturbation(axicon_tilt_rad=(math.radians(20.0), 0.0))
    try:
        perturbation.validate()
    except ValueError as exc:
        assert "20" in str(exc) or "tilt" in str(exc).lower()
    else:  # pragma: no cover
        raise AssertionError("large scalar axicon tilt should be rejected")
