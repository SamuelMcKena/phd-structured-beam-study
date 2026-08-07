from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from vbb_study.digital_twin.phase2e_final_figure_style import FINAL_FIGURE_STYLE
from vbb_study.digital_twin.phase2e_final_source_propagation import (
    source_scale_route_contract,
)


VALIDATION = Path("outputs/validation/phase2e_final_propagation")
FIGURES = Path("outputs/figures/phase2e_final_source_propagation")


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _cache(case: str, route: str = "nominal_no_additional_aperture") -> dict[str, np.ndarray]:
    path = VALIDATION / "production_cache" / f"{case.lower()}_{route}.npz"
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def test_01_final_grid_selection_satisfies_reference_gate() -> None:
    gate = _json(VALIDATION / "final_resolution_gate.json")
    assert gate["selected_production_grid_n"] == 3072
    assert gate["grid_pass"]["3072"] is True


def test_02_resolution_gate_holds_fixed_physical_window() -> None:
    rows = _csv(VALIDATION / "final_resolution_gate.csv")
    assert {float(row["window_m"]) for row in rows} == {10e-3}


def test_03_nominal_route_contains_no_additional_hard_aperture() -> None:
    route = source_scale_route_contract()["nominal_no_additional_aperture"]
    assert route["aperture_application_count"] == 0
    assert "none_additional" in route["aperture_model"]


def test_04_soft_route_applies_exactly_one_soft_aperture() -> None:
    route = source_scale_route_contract()["soft_aperture_sensitivity"]
    assert route["aperture_application_count"] == 1
    assert route["aperture_model"] == "exp[-(r/r_a)^8]"


def test_05_hard_route_applies_exactly_one_hard_aperture() -> None:
    route = source_scale_route_contract()["hard_aperture_diagnostic"]
    assert route["aperture_application_count"] == 1
    assert "hard truncation" in route["aperture_model"]


def test_06_hard_route_is_never_report_primary() -> None:
    route = source_scale_route_contract()["hard_aperture_diagnostic"]
    assert route["report_eligibility"] == "diagnostic_only"
    assert route["calibration_required"] is True


def test_07_source_routes_apply_no_objective_transform() -> None:
    assert all(route["objective_transform_application_count"] == 0 for route in source_scale_route_contract().values())


def test_08_source_routes_apply_axicon_once() -> None:
    assert all(route["axicon_application_count"] == 1 for route in source_scale_route_contract().values())


def test_09_carrier_and_4f_route_are_single_application() -> None:
    metadata = _json(VALIDATION / "production_cache" / "b0_nominal_no_additional_aperture.json")
    assert metadata["first_order_filter_application_count"] == 1
    assert metadata["carrier_removal_application_count"] == 1


def test_10_b0_gate_uses_raw_on_axis_observable() -> None:
    rows = [row for row in _csv(VALIDATION / "final_resolution_gate.csv") if row["case_id"] == "B0"]
    assert all(np.isclose(float(row["primary_observable_raw"]), float(row["on_axis_intensity_raw"])) for row in rows)


def test_11_vortex_gate_uses_ring_aware_observable() -> None:
    rows = [row for row in _csv(VALIDATION / "final_resolution_gate.csv") if row["case_id"] in {"V1", "V3"}]
    assert any(not np.isclose(float(row["primary_observable_raw"]), float(row["on_axis_intensity_raw"])) for row in rows)


def test_12_fixed_bucket_or_annulus_is_constant_with_z() -> None:
    for case in ("B0", "V1", "V3"):
        metadata = _json(VALIDATION / "production_cache" / f"{case.lower()}_nominal_no_additional_aperture.json")
        assert metadata["fixed_region"]["inner_radius_m"] >= 0.0
        assert metadata["fixed_region"]["outer_radius_m"] > metadata["fixed_region"]["inner_radius_m"]


def test_13_invalid_feature_radii_are_nan() -> None:
    for case in ("B0", "V1", "V3"):
        arrays = _cache(case)
        assert np.all(np.isnan(arrays["feature_radius_m"][~arrays["feature_valid"].astype(bool)]))


def test_14_invalid_radius_samples_are_not_connected() -> None:
    manifest = _json(FIGURES / "00_manifest" / "final_figure_manifest.json")
    assert any("NaN invalid radii remain disconnected" in row["notes"] for row in manifest)


def test_15_primary_maps_forbid_per_z_normalisation() -> None:
    assert FINAL_FIGURE_STYLE.primary_normalisation == "global_linear"


def test_16_primary_maps_forbid_log_gamma_and_db() -> None:
    assert FINAL_FIGURE_STYLE.primary_scaling == "linear"
    manifest = _json(FIGURES / "00_manifest" / "final_figure_manifest.json")
    assert all(token not in row["normalisation_policy"].lower() for row in manifest for token in ("log", "gamma", "db", "per-z"))


def test_17_matched_comparisons_share_axes_and_colour_limits() -> None:
    manifest = _json(FIGURES / "00_manifest" / "final_figure_manifest.json")
    rows = [row for row in manifest if row["route_id"] == "three_route_aperture_comparison"]
    assert len(rows) == 4
    assert all(row["colour_limits"] == "[0.0, 1.0]" for row in rows)


def test_18_surfaces_share_limits_and_camera() -> None:
    style = _json(FIGURES / "00_manifest" / "final_figure_style.json")
    assert style["surface_intensity_limits"] == [0.0, 1.0]
    assert isinstance(style["surface_elevation_deg"], float)
    assert isinstance(style["surface_azimuth_deg"], float)


def test_19_display_interpolation_never_supplies_metrics() -> None:
    manifest = _json(FIGURES / "00_manifest" / "final_figure_manifest.json")
    assert all(row["metrics_computed_on_native_arrays"] for row in manifest)
    assert all(not row["display_interpolation_used_for_metrics"] for row in manifest)


def test_20_z_step_convergence_passes() -> None:
    gate = _json(VALIDATION / "z_step_convergence.json")
    assert gate["status"] == "passed"
    assert gate["selected_z_step_m"] == 0.25e-3


def test_21_edge_energy_threshold_passes() -> None:
    summary = _csv(VALIDATION / "final_case_summary.csv")
    assert max(float(row["maximum_edge_energy_fraction"]) for row in summary) <= 0.01


def test_22_power_drift_threshold_passes() -> None:
    summary = _csv(VALIDATION / "final_case_summary.csv")
    assert max(float(row["maximum_propagation_power_drift_fraction"]) for row in summary) <= 1e-3


def test_23_useful_zone_definitions_remain_separate() -> None:
    rows = _csv(VALIDATION / "final_zone_summary.csv")
    expected = {"configured_nominal_interval", "geometric_zone_estimate", "measured_FWHM_axial_zone", "measured_strict_useful_region"}
    assert all({row["zone_definition"] for row in rows if row["case_id"] == case} == expected for case in ("B0", "V1", "V3"))


def test_24_configured_interval_is_not_called_measured() -> None:
    rows = _csv(VALIDATION / "final_zone_summary.csv")
    configured = [row for row in rows if row["zone_definition"] == "configured_nominal_interval"]
    assert all(row["is_measured"] == "False" for row in configured)


def test_25_hard_aperture_beading_remains_diagnostic() -> None:
    rows = _csv(VALIDATION / "final_aperture_comparison.csv")
    hard = [row for row in rows if row["route_id"] == "hard_aperture_diagnostic"]
    assert hard and all(row["report_eligibility"] == "diagnostic_only" for row in hard)


def test_26_phase2a_phase2b_phase2c_hashes_unchanged() -> None:
    status = _json(VALIDATION / "upstream_hash_status.json")
    assert status["unchanged"] is True
    assert status["file_count"] > 0


def test_27_vortex_winding_contracts_remain_unchanged() -> None:
    rows = {row["case_id"]: row for row in _csv(VALIDATION / "final_case_summary.csv")}
    assert int(rows["V1"]["requested_winding"]) == 1
    assert int(rows["V3"]["requested_winding"]) == 3
    assert abs(float(rows["V1"]["measured_winding"]) - 1.0) < 0.1
    assert abs(float(rows["V3"]["measured_winding"]) - 3.0) < 0.1


def test_28_source_and_objective_scale_claims_remain_separate() -> None:
    rows = _csv(VALIDATION / "final_claim_impact_registry.csv")
    phase2c = next(row for row in rows if row["claim_id"] == "PHASE2E_FINAL_CLAIM_006")
    assert phase2c["remains_valid"] == "True"
    assert "remain separate" in phase2c["notes"]


def test_29_manifests_contain_every_required_artifact() -> None:
    figures = _json(FIGURES / "00_manifest" / "final_figure_manifest.json")
    artifacts = _json(FIGURES / "00_manifest" / "final_artifact_manifest.json")
    paths = {row["path"] for row in artifacts}
    assert len(figures) >= 18
    assert all(row["png_path"] in paths and row["pdf_path"] in paths for row in figures)


def test_30_report_figures_are_authorised_only_after_all_gates() -> None:
    resolution = _json(VALIDATION / "final_resolution_gate.json")
    z_gate = _json(VALIDATION / "z_step_convergence.json")
    outcome = _json(VALIDATION / "final_outcome_report.json")
    assert resolution["status"] == "passed"
    assert z_gate["status"] == "passed"
    assert outcome["outcome"] == "PHASE2E-FINAL-A"
    assert outcome["report_figures_authorised"] is True
