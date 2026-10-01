"""Focused tests for the two-SLM registration architecture comparison."""

from __future__ import annotations

import numpy as np
import pytest

from vbb_study.digital_twin.slm_registration_architectures import (
    ARCHITECTURES,
    RegistrationState,
    architecture_phase_components,
    architecture_registration_states_1d,
    errors_from_registration,
    pure_registration_error,
    unit_cell_offsets,
    canonical_registration_hardware,
    _charge_from_case_id,
)


def test_generic_vortex_case_parser_supports_report_charges() -> None:
    assert _charge_from_case_id("B0") == 0
    assert _charge_from_case_id("V1") == 1
    assert _charge_from_case_id("V10") == 10
    assert _charge_from_case_id("V20") == 20
    with pytest.raises(ValueError):
        _charge_from_case_id("bad")



def test_architecture_names_are_explicit() -> None:
    assert ARCHITECTURES == ("upstream_vortex", "downstream_vortex")


def test_pure_registration_keeps_hologram_centre_fixed() -> None:
    err = pure_registration_error(2.0e-6, -3.0e-6)
    total_x = err.panel_translation_m[0] + err.pattern_offset_m[0]
    total_y = err.panel_translation_m[1] + err.pattern_offset_m[1]
    assert total_x == pytest.approx(0.0)
    assert total_y == pytest.approx(0.0)
    assert err.lattice_offset_m == pytest.approx((-2.0e-6, 3.0e-6))


def test_two_panel_registration_state_is_independent() -> None:
    state = RegistrationState(
        slm1_dx_m=1.0e-6,
        slm1_dy_m=2.0e-6,
        slm2_dx_m=-3.0e-6,
        slm2_dy_m=4.0e-6,
    )
    e1, e2 = errors_from_registration(state)
    assert e1.lattice_offset_m == pytest.approx((-1.0e-6, -2.0e-6))
    assert e2.lattice_offset_m == pytest.approx((3.0e-6, -4.0e-6))
    assert np.add(e1.panel_translation_m, e1.pattern_offset_m) == pytest.approx((0.0, 0.0))
    assert np.add(e2.panel_translation_m, e2.pattern_offset_m) == pytest.approx((0.0, 0.0))


def test_canonical_slm2_carrier_is_confirmed_20_pixel_blaze() -> None:
    hw = canonical_registration_hardware()
    pitch = float(hw["pixel_pitch_m"])
    carrier = float(hw["carrier_cpm"])
    period_px = 1.0 / (carrier * pitch)
    assert carrier == pytest.approx(6250.0)
    assert period_px == pytest.approx(20.0)



def test_slm2_carries_carrier_blaze_in_both_architectures() -> None:
    x = np.asarray([[0.0, 1.0e-6, 2.0e-6]])
    y = np.zeros_like(x)
    carrier = 6250.0

    for architecture in ARCHITECTURES:
        _, slm2, roles = architecture_phase_components(
            architecture,
            charge=0,
            carrier_cpm=carrier,
            correction_command=None,
        )
        phase = slm2(x, y)
        expected = 2.0 * np.pi * carrier * x
        assert phase == pytest.approx(expected)
        assert roles["SLM2_carrier_blaze_present"] is True
        assert "carrier_blaze" in roles["SLM2"]


def test_phase_ownership_upstream_vortex() -> None:
    x = np.asarray([[1.0e-6, 0.0]])
    y = np.asarray([[0.0, 1.0e-6]])

    def corr(xp, yp):
        return 0.25 + 0.0 * xp + 0.0 * yp

    slm1, slm2, roles = architecture_phase_components(
        "upstream_vortex",
        charge=3,
        carrier_cpm=1000.0,
        correction_command=corr,
    )
    assert roles["vortex_owner"] == "SLM1"
    assert roles["correction_owner"] == "SLM2"
    assert tuple(roles["SLM1"]) == ("vortex",)
    assert tuple(roles["SLM2"]) == ("correction", "carrier_blaze")

    assert slm1(x, y)[0, 0] == pytest.approx(0.0)
    assert slm1(x, y)[0, 1] == pytest.approx(3.0 * np.pi / 2.0)
    expected_slm2 = corr(x, y) + 2.0 * np.pi * 1000.0 * x
    assert slm2(x, y) == pytest.approx(expected_slm2)


def test_phase_ownership_downstream_vortex() -> None:
    x = np.asarray([[1.0e-6, 0.0]])
    y = np.asarray([[0.0, 1.0e-6]])

    def corr(xp, yp):
        return 0.5 + 0.0 * xp + 0.0 * yp

    slm1, slm2, roles = architecture_phase_components(
        "downstream_vortex",
        charge=3,
        carrier_cpm=1000.0,
        correction_command=corr,
    )
    assert roles["vortex_owner"] == "SLM2"
    assert roles["correction_owner"] == "SLM1"
    assert tuple(roles["SLM1"]) == ("correction",)
    assert tuple(roles["SLM2"]) == ("vortex", "carrier_blaze")

    assert slm1(x, y) == pytest.approx(corr(x, y))
    expected_slm2 = (
        3.0 * np.arctan2(y, x)
        + 2.0 * np.pi * 1000.0 * x
    )
    assert slm2(x, y) == pytest.approx(expected_slm2)


def test_flat_correction_is_explicit_physics_isolation_baseline() -> None:
    _, _, roles = architecture_phase_components(
        "downstream_vortex",
        charge=10,
        carrier_cpm=6250.0,
        correction_command=None,
    )
    assert roles["correction_status"] == "flat_zero_physics_isolation"


def test_unit_cell_offsets_cover_distinct_periodic_states_only() -> None:
    pitch = 8.0e-6
    offsets = unit_cell_offsets(pitch, subdivisions=8)
    assert len(offsets) == 8
    assert offsets[0] == pytest.approx(0.0)
    assert offsets[-1] == pytest.approx(7.0 * pitch / 8.0)
    assert np.all(offsets < pitch)


def test_common_and_differential_sweeps_are_opposites() -> None:
    offsets = np.asarray([0.0, 2.0e-6])
    common = architecture_registration_states_1d(
        offsets,
        panel="common",
        axis="x",
    )[-1]
    differential = architecture_registration_states_1d(
        offsets,
        panel="differential",
        axis="x",
    )[-1]

    assert common.slm1_dx_m == pytest.approx(2.0e-6)
    assert common.slm2_dx_m == pytest.approx(2.0e-6)
    assert differential.slm1_dx_m == pytest.approx(2.0e-6)
    assert differential.slm2_dx_m == pytest.approx(-2.0e-6)


def test_diagonal_sweep_moves_x_and_y_equally() -> None:
    state = architecture_registration_states_1d(
        np.asarray([1.0e-6]),
        panel="slm2",
        axis="diagonal",
    )[0]
    assert state.slm2_dx_m == pytest.approx(1.0e-6)
    assert state.slm2_dy_m == pytest.approx(1.0e-6)
