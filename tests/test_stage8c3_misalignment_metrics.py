"""Stage 8C.3 degradation metric and comparison-plot tests."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from pathlib import Path

from vbb_study.digital_twin.active_realism_metrics import (
    build_stage8c3_sensitivity_scenarios,
    compute_misalignment_sensitivity_sweep,
    compute_degradation_metrics,
    metric_delta_table,
    plot_baseline_vs_perturbed_comparison,
    plot_misalignment_sensitivity_sweep,
)
from vbb_study.digital_twin.field_coupling import stack_from_arrays
from vbb_study.digital_twin.field_fluence import scale_stack_to_fluence
from vbb_study.digital_twin.lab_perturbations import apply_lab_perturbations_to_stack


MINIMUM_METRICS = {
    "centroid_x_um",
    "centroid_y_um",
    "peak_x_um",
    "peak_y_um",
    "peak_z_um",
    "ring_circularity_score",
    "azimuthal_uniformity_score",
    "side_lobe_imbalance",
    "core_fill_fraction",
    "central_darkness_contrast",
    "peak_fluence_change_fraction",
    "target_depth_peak_fluence",
    "central_roi_peak_fluence",
    "captured_power_drift_fraction",
    "pupil_clipped_power_fraction",
    "first_order_selected_fraction",
    "symmetry_score",
    "baseline_similarity_score",
}


def _stack():
    x = np.linspace(-20.0, 20.0, 61)
    y = x.copy()
    z = np.linspace(-60.0, 180.0, 25)
    X, Y = np.meshgrid(x, y, indexing="xy")
    R = np.hypot(X, Y)
    planes = []
    for zi in z:
        amp = np.exp(-((zi - 80.0) / 80.0) ** 2) + 0.2
        planes.append(amp * np.exp(-((R - 5.5) ** 2) / (2.0 * 1.8**2)) + 0.12 * amp * np.exp(-(R**2) / (2.0 * 10.0**2)))
    return stack_from_arrays(np.asarray(planes), x, y, z, source_status="unit_test_fixture")


def test_degradation_metrics_include_required_keys():
    stack = _stack()
    fluence = scale_stack_to_fluence(stack, 40.0)
    metrics = compute_degradation_metrics(stack, fluence, target_depth_um=80.0)
    assert MINIMUM_METRICS <= set(metrics)
    assert "peak_trajectory_x_span_um" in metrics


def test_metric_delta_table_reports_required_metrics():
    stack = _stack()
    fluence = scale_stack_to_fluence(stack, 40.0)
    baseline = compute_degradation_metrics(stack, fluence, target_depth_um=80.0)
    result = apply_lab_perturbations_to_stack(
        stack,
        {
            "enable_vortex_centre_offset": True,
            "vortex_centre_offset_x_um": 4.0,
            "enable_zero_order_leakage": True,
            "zero_order_leakage_fraction": 0.08,
        },
    )
    perturbed_fluence = scale_stack_to_fluence(result.perturbed_stack, 40.0)
    perturbed = compute_degradation_metrics(
        result.perturbed_stack,
        perturbed_fluence,
        baseline_stack=stack,
        baseline_fluence_result=fluence,
        target_depth_um=80.0,
        perturbation_metadata=result.metadata,
    )
    table = metric_delta_table(baseline, perturbed)
    assert MINIMUM_METRICS <= {row["metric"] for row in table}
    assert perturbed["baseline_similarity_score"] < 1.0


def test_baseline_vs_perturbed_plot_uses_shared_scaling():
    stack = _stack()
    fluence = scale_stack_to_fluence(stack, 40.0)
    result = apply_lab_perturbations_to_stack(
        stack,
        {
            "enable_beam_decentre": True,
            "beam_decentre_x_um": 3.0,
            "enable_zernike_aberrations": True,
            "zernike_coma_x_waves": 0.2,
        },
    )
    perturbed_fluence = scale_stack_to_fluence(result.perturbed_stack, 40.0)
    fig = plot_baseline_vs_perturbed_comparison(
        stack,
        result.perturbed_stack,
        fluence,
        perturbed_fluence,
        selected_plane_index=10,
        show_caveats=True,
    )
    clims = {ax.get_title(): ax.images[0].get_clim() for ax in fig.axes if ax.images}
    assert clims["baseline XY fluence"] == clims["perturbed XY fluence"]
    assert clims["baseline XZ fluence"] == clims["perturbed XZ fluence"]
    assert fig.stage8c3_metadata["final_export_allowed"] is False
    assert fig.stage8c3_metadata["figure_status"] == "diagnostic_allowed"
    plt.close(fig)


def test_sensitivity_scenarios_cover_required_stage8c3b_cases():
    scenarios = build_stage8c3_sensitivity_scenarios()
    assert {
        "phase_center_offset",
        "beam_tilt",
        "pupil_clipping",
        "zero_order_leakage",
    } <= set(scenarios)
    assert "vortex_centre_offset_x_um" in scenarios["phase_center_offset"].severe_controls
    assert "beam_tilt_x_mrad" in scenarios["beam_tilt"].severe_controls
    assert "pupil_decentre_x_um" in scenarios["pupil_clipping"].severe_controls
    assert "zero_order_leakage_fraction" in scenarios["zero_order_leakage"].severe_controls


def test_sensitivity_sweep_metrics_show_mild_and_severe_degradation():
    stack = _stack()
    for scenario_key in build_stage8c3_sensitivity_scenarios():
        sweep = compute_misalignment_sensitivity_sweep(
            stack,
            40.0,
            scenario=scenario_key,
            target_depth_um=80.0,
            central_roi_half_width_um=8.0,
        )
        metric = sweep.scenario.degradation_metric
        assert sweep.mild_metrics[metric] != sweep.baseline_metrics[metric]
        assert sweep.severe_metrics[metric] != sweep.baseline_metrics[metric]
        assert sweep.metadata["severe_worse_than_mild"] is True
        assert (
            sweep.severe_metrics["baseline_similarity_score"]
            < sweep.mild_metrics["baseline_similarity_score"]
        )


def test_sensitivity_sweep_plot_uses_shared_scales_and_saves():
    stack = _stack()
    out = Path("outputs/figures/digital_twin/_stage8c3_test_sensitivity_sweep_preview.png")
    fig = plot_misalignment_sensitivity_sweep(
        stack,
        40.0,
        scenario="phase_center_offset",
        target_depth_um=80.0,
        central_roi_half_width_um=8.0,
        output_path=out,
        show_caveats=True,
        dpi=80,
    )
    assert out.is_file()
    images = [ax.images[0] for ax in fig.axes if ax.images]
    assert len(images) >= 12
    assert images[0].get_clim() == images[1].get_clim() == images[2].get_clim()
    assert images[4].get_clim() == images[5].get_clim() == images[6].get_clim()
    assert images[8].get_clim() == images[9].get_clim() == images[10].get_clim()
    assert fig.stage8c3_metadata["stage"] == "stage8c3b_misalignment_sensitivity_sweep"
    assert fig.stage8c3_metadata["final_export_allowed"] is False
    assert fig.stage8c3_metadata["severe_worse_than_mild"] is True
    plt.close(fig)
