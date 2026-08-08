from __future__ import annotations

import numpy as np

from vbb_study.digital_twin.vortex_visual_atlas import (
    ZERNIKE_REGISTRY,
    aberration_registry,
    alignment_registry,
    apply_zernike_waves,
    build_atlas_source,
    parameter_registry,
    zernike_mode,
)


def test_expected_zernike_modes_are_registered() -> None:
    assert set(ZERNIKE_REGISTRY) == {
        "defocus",
        "astigmatism_x",
        "astigmatism_y",
        "coma_x",
        "coma_y",
        "spherical",
    }


def test_zernike_modes_are_finite_and_near_unit_rms_inside_declared_radius() -> None:
    _, grid, _ = build_atlas_source("B0", grid_n=256)
    radius_m = 2.0e-3
    rho = np.asarray(grid["R"], dtype=float) / radius_m
    mask = rho <= 1.0
    for name in ZERNIKE_REGISTRY:
        mode = zernike_mode(name, grid, radius_m)
        assert mode.shape == (256, 256)
        assert np.all(np.isfinite(mode))
        rms = float(np.sqrt(np.mean(mode[mask] ** 2)))
        assert np.isclose(rms, 1.0, rtol=0.0, atol=1e-12)


def test_zero_wave_aberration_does_not_change_field() -> None:
    source, grid, _ = build_atlas_source("V1", grid_n=256)
    changed = apply_zernike_waves(source, grid, name="coma_x", waves_rms=0.0, radius_m=2e-3)
    assert np.array_equal(source, changed)


def test_nonzero_coma_changes_phase_not_pointwise_amplitude() -> None:
    source, grid, _ = build_atlas_source("V1", grid_n=256)
    changed = apply_zernike_waves(source, grid, name="coma_x", waves_rms=0.1, radius_m=2e-3)
    assert not np.allclose(source, changed)
    assert np.allclose(np.abs(source), np.abs(changed), rtol=1e-12, atol=1e-12)


def test_nominal_atlas_source_uses_repaired_phase2e_route() -> None:
    source, grid, meta = build_atlas_source("V3", grid_n=256)
    assert source.shape == (256, 256)
    assert grid["N"] == 256
    assert meta["atlas_route"] == "repaired_phase2e_source_scale"
    assert meta["aperture_model"] == "none"
    assert meta["historical_objective_pupil_application_count"] == 0


def test_registries_have_nominal_zero_or_one_reference_points() -> None:
    assert 1.0 in parameter_registry()["beam_radius_scale"]
    assert 1.0 in parameter_registry()["axicon_angle_scale"]
    assert "none" in parameter_registry()["aperture_model"]
    assert all(0.0 in values for values in aberration_registry().values())
    assert all(0.0 in values for values in alignment_registry().values())
