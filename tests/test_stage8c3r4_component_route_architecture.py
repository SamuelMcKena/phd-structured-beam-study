"""Stage 8C.3 component-owned route architecture tests."""

from __future__ import annotations

from vbb_study.digital_twin.route_aware_axicon import (
    EXECUTED_PHYSICAL_AXICON_COMPONENT_IDS,
    RouteAwareAxiconConfig,
    build_route_component_declarations,
    build_route_perturbation_records,
    route_inspection_rows,
    run_route_aware_axicon_pipeline,
)


CFG = RouteAwareAxiconConfig.fast(grid_N=112, dx_um=0.6, n_z=10)


def test_executed_route_is_ordered_component_and_segment_chain():
    run = run_route_aware_axicon_pipeline(config=CFG)
    ids = tuple(c.component_id for c in run.component_chain)
    assert ids == EXECUTED_PHYSICAL_AXICON_COMPONENT_IDS
    assert tuple(r.component_id for r in run.route_inspection_records) == ids
    assert any(c.component_type == "propagation_segment" for c in run.component_chain)
    assert any(c.distance_to_next_element_mm > 0 for c in run.component_chain)
    for row in route_inspection_rows(run):
        assert "incoming_field_metrics" in row
        assert "outgoing_field_metrics" in row
        assert "energy before (uJ)" in row
        assert "energy after (uJ)" in row
        assert "centroid before (um)" in row
        assert "angle after (mrad)" in row
        assert "model status" in row


def test_component_declarations_separate_supported_and_unrepresented_optics():
    declarations = {c.component_id: c for c in build_route_component_declarations(CFG)}
    assert declarations["physical_axicon"].represented_by_current_engine is True
    assert declarations["physical_axicon"].physical_model_available is True
    assert "decentre_x_um" in declarations["physical_axicon"].misalignment_modes_currently_supported
    assert "axial_offset_um" in declarations["physical_axicon"].misalignment_modes_currently_supported
    assert declarations["SLM1"].represented_by_current_engine is False
    assert declarations["SLM1"].status == "warning_only"
    assert declarations["Fourier_filter"].physical_model_available is False
    assert declarations["objective_pupil"].status == "future_not_implemented"


def test_physical_axicon_decentre_is_component_pose_error_at_axicon_plane():
    run = run_route_aware_axicon_pipeline({"physical_axicon_centre_x_um": 7.0}, config=CFG)
    axicon_row = [r for r in run.route_inspection_records if r.component_id == "physical_axicon"][0]
    assert axicon_row.actual_pose_error["decentre_x_um"] == 7.0
    assert axicon_row.component_type == "physical_axicon"
    assert "after_objective_boundary" in axicon_row.downstream_consequences
    assert run.axicon_incidence_metrics["relative_beam_to_axicon_offset_x_um"] < -6.0
    rec = [r for r in build_route_perturbation_records(run.config) if r.parameter_name == "physical_axicon_centre_x_um"][0]
    assert rec.component_id == "physical_axicon"
    assert rec.implementation_plane == "physical_axicon_plane"


def test_aperture_decentre_is_applied_at_aperture_component_and_changes_energy():
    base = run_route_aware_axicon_pipeline(config=CFG)
    dec = run_route_aware_axicon_pipeline(
        {"input_aperture_radius_um": 18.0, "input_aperture_centre_x_um": 14.0},
        config=CFG,
    )
    ap_row = [r for r in dec.route_inspection_records if r.component_id == "input_aperture"][0]
    assert ap_row.actual_pose_error["decentre_x_um"] == 14.0
    assert ap_row.aperture_overlap is not None
    assert ap_row.aperture_overlap < 0.99
    assert dec.propagated_stack.transmitted_fraction < base.propagated_stack.transmitted_fraction


def test_axicon_axial_offset_edits_adjacent_segment_distances():
    run = run_route_aware_axicon_pipeline({"physical_axicon_axial_offset_um": 75.0}, config=CFG)
    rows = {r.component_id: r for r in run.route_inspection_records}
    assert abs(rows["source_to_physical_axicon"].distance_to_next_element_mm - (CFG.pre_axicon_distance_mm + 0.075)) < 1e-12
    assert abs(rows["physical_axicon_to_after_objective"].distance_to_next_element_mm - (CFG.axicon_to_objective_distance_mm - 0.075)) < 1e-12
    axicon = rows["physical_axicon"]
    assert axicon.actual_pose_error["axial_offset_um"] == 75.0


def test_field_state_controls_are_boundary_conditions_with_approximation_labels():
    run = run_route_aware_axicon_pipeline(
        {"field_tilt_x_mrad": 9.0, "field_tilt_location": "before_physical_axicon"},
        config=CFG,
    )
    boundary = [r for r in run.route_inspection_records if r.component_id == "physical_axicon_input_boundary"][0]
    assert boundary.model_status == "boundary_condition_active"
    rec = [r for r in run.perturbation_records if r.parameter_name == "field_tilt_x_mrad"][0]
    assert rec.component_id == "physical_axicon_input_boundary"
    assert rec.boundary_plane == "before_physical_axicon"
    assert "phase ramp" in rec.physical_approximation
    assert "upstream" in rec.upstream_hardware_error_could_emulate
    assert "physical_axicon" in rec.downstream_components_consume


def test_no_stage8d_or_material_response_in_component_route():
    run = run_route_aware_axicon_pipeline(config=CFG)
    blob = str(run.metadata).lower() + str(run.propagated_stack.metadata).lower()
    assert run.final_export_allowed is False
    assert run.model_status == "optical_prediction"
    assert "no_material_model" in str(run.reference_plane_state.metadata)
    banned = ("stage8d", "plasma", "ablation", "damage", "dose_accumulation", "thermal")
    assert not any(term in blob for term in banned)
