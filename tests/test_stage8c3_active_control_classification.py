"""Stage 8C.3 lab-control classification tests."""

import pytest

from vbb_study.digital_twin.lab_perturbations import (
    AFFECTED_OUTPUTS,
    CONTROL_CLASSIFICATIONS,
    classification_for_control,
    classify_lab_controls,
    enabled_uncoupled_controls,
)
from vbb_study.digital_twin.lab_realism_controls import (
    build_lab_realism_report,
    default_lab_controls,
    validate_future_physics_disabled,
)


def test_every_default_lab_control_has_classification_and_affected_outputs():
    controls = default_lab_controls()
    rows = classify_lab_controls(controls)
    assert len(rows) == len(controls)
    assert {row.control for row in rows} == set(controls)
    for row in rows:
        assert row.classification in CONTROL_CLASSIFICATIONS
        assert row.affects
        assert set(row.affects) <= AFFECTED_OUTPUTS
        assert isinstance(row.implemented, bool)
        assert row.downstream_response_expected


def test_required_controls_have_expected_classifications_and_outputs():
    controls = default_lab_controls()
    examples = {
        "enable_beam_tilt": ("physics_active", {"phase", "angular_spectrum", "field"}),
        "enable_pulse_energy_jitter": ("energy_active", {"energy_ledger", "fluence"}),
        "enable_sample_tilt": ("geometry_active", {"sample_geometry"}),
        "enable_camera_crop": ("diagnostic_active", {"metadata"}),
        "physical_axicon_angle_error_deg": ("warning_only", {"warnings", "metadata"}),
        "polarisation_state": ("metadata_only", {"metadata"}),
        "enable_material_response": ("future_not_implemented", {"future_stage", "warnings"}),
    }
    for control, (classification, affects) in examples.items():
        row = classification_for_control(control, controls)
        assert row.classification == classification
        assert affects <= set(row.affects)


def test_future_only_controls_remain_disabled_and_raise_if_enabled():
    controls = default_lab_controls()
    rows = classify_lab_controls(controls)
    future = [row for row in rows if row.classification == "future_not_implemented"]
    assert future
    assert all(not row.enabled for row in future)

    controls["enable_material_response"] = True
    with pytest.raises(NotImplementedError):
        validate_future_physics_disabled(controls)


def test_enabled_uncoupled_controls_are_reported():
    controls = default_lab_controls()
    controls["enable_first_order_filter_decentre"] = True
    uncoupled = enabled_uncoupled_controls(controls)
    names = {row.control for row in uncoupled}
    assert "enable_first_order_filter_decentre" in names
    assert all(not row.implemented for row in uncoupled)


def test_lab_realism_report_rows_include_classification_columns():
    rows = build_lab_realism_report(default_lab_controls()).to_rows()
    assert rows
    for row in rows:
        assert "control_classifications" in row
        assert "affected_outputs" in row
        assert row["control_classifications"]
        assert row["affected_outputs"]
