"""Stage 8C.3 notebook wiring tests."""

import ast
import json
from pathlib import Path


ROOT = Path(__file__).parent.parent
NB_PATH = ROOT / "notebooks" / "digital_twin" / "00_full_beam_to_write_cockpit_MVP.ipynb"


def _load() -> dict:
    return json.loads(NB_PATH.read_text(encoding="utf-8"))


def _source() -> str:
    nb = _load()
    return "\n".join(
        "".join(cell.get("source", [])) for cell in nb["cells"]
    )


def test_notebook_valid_and_stage8c2_single_dashboard_cell_preserved():
    nb = _load()
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell.get("source", [])))
    n = sum(
        1 for cell in nb["cells"]
        if cell.get("cell_type") == "code"
        and "plot_integrated_cockpit_dashboard(" in "".join(cell.get("source", []))
        and "fig =" in "".join(cell.get("source", []))
    )
    assert n == 1


def test_active_lab_realism_table_present_with_required_columns():
    src = _source()
    assert "Active Lab-Realism Controls" in src
    for header in [
        '"control"',
        '"value"',
        '"enabled"',
        '"classification"',
        '"affects"',
        '"implemented"',
        '"downstream_response_expected"',
    ]:
        assert header in src
    assert "physics_active" in src
    assert "warning_only" in src
    assert "future_not_implemented" in src


def test_poynting_vector_clarification_present():
    src = _source()
    assert (
        "In the current scalar optical model, direct Poynting-vector editing "
        "is not a primitive active input."
    ) in src
    assert "input beam tilt / angular-spectrum phase ramp" in src


def test_baseline_vs_perturbed_section_and_preview_path_present():
    src = _source()
    assert "Baseline vs Perturbed Misalignment Sanity Check" in src
    assert "plot_baseline_vs_perturbed_comparison" in src
    assert "metric_delta_table" in src
    assert "stage8c3_baseline_vs_perturbed_preview.png" in src
    assert "selected_plane_index=diagnostics" in src
    assert "display(comparison_fig)" in src


def test_sensitivity_sweep_section_and_preview_path_present():
    src = _source()
    assert "Stage 8C.3 - Active perturbation sensitivity sweep" in src
    assert "plot_misalignment_sensitivity_sweep" in src
    assert "stage8c3_misalignment_sensitivity_sweep_preview.png" in src
    assert "build_stage8c3_sensitivity_scenarios" in src
    assert "severe_worse_than_mild" in src
    assert "Active perturbations are diagnostic lab-realism controls" in src


def test_dashboard_receives_active_perturbation_result():
    src = _source()
    assert 'perturbation_result=globals().get("perturbation_result")' in src
    assert 'degradation_metrics=globals().get("perturbed_degradation_metrics")' in src


def test_notebook_does_not_introduce_material_response_overclaim():
    src = _source().lower()
    assert "enable_material_response = false" in src
    assert "material response disabled" in src or "material response is disabled" in src
    assert "no material response is predicted" in src or "material response disabled" in src
