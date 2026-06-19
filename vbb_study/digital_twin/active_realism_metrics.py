"""Degradation metrics and baseline-vs-perturbed plots for Stage 8C.3."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import matplotlib
matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt  # noqa: E402

from vbb_study.digital_twin.field_coupling import OpticalFieldStack
from vbb_study.digital_twin.field_fluence import FluenceStackResult, scale_stack_to_fluence
from vbb_study.digital_twin.lab_perturbations import apply_lab_perturbations_to_stack

FINAL_EXPORT_ALLOWED = False
FIGURE_STATUS = "diagnostic_allowed"
MODEL_STATUS = "diagnostic_preview"


@dataclass(frozen=True)
class SensitivityScenario:
    """One Stage 8C.3B active-perturbation severity sweep scenario."""

    key: str
    title: str
    scenario_name: str
    mild_controls: Mapping[str, Any]
    severe_controls: Mapping[str, Any]
    mild_label: str
    severe_label: str
    degradation_metric: str
    worse_direction: str
    expected_visible_change: str


@dataclass(frozen=True)
class SensitivitySweepResult:
    """Computed fields/metrics for a Stage 8C.3B sensitivity sweep."""

    scenario: SensitivityScenario
    baseline_stack: OpticalFieldStack
    mild_stack: OpticalFieldStack
    severe_stack: OpticalFieldStack
    baseline_fluence: FluenceStackResult
    mild_fluence: FluenceStackResult
    severe_fluence: FluenceStackResult
    baseline_metrics: Mapping[str, float]
    mild_metrics: Mapping[str, float]
    severe_metrics: Mapping[str, float]
    mild_delta_rows: tuple[Mapping[str, float | str], ...]
    severe_delta_rows: tuple[Mapping[str, float | str], ...]
    shared_scales: Mapping[str, float]
    selected_plane_index: int
    central_roi_half_width_um: float
    final_export_allowed: bool = False
    figure_status: str = FIGURE_STATUS
    model_status: str = MODEL_STATUS
    metadata: Mapping[str, Any] = field(default_factory=dict)


def compute_degradation_metrics(
    stack: OpticalFieldStack,
    fluence_result: FluenceStackResult | None = None,
    *,
    baseline_metrics: Mapping[str, Any] | None = None,
    baseline_stack: OpticalFieldStack | None = None,
    baseline_fluence_result: FluenceStackResult | None = None,
    plane_index: int | None = None,
    target_depth_um: float | None = None,
    central_roi_half_width_um: float = 10.0,
    perturbation_metadata: Mapping[str, Any] | None = None,
    first_order_selected_fraction: float | None = None,
) -> dict[str, float]:
    """Compute robust scalar degradation metrics from a field/fluence stack."""
    if not isinstance(stack, OpticalFieldStack):
        raise TypeError(f"stack must be OpticalFieldStack; got {type(stack).__name__}.")
    I = np.asarray(stack.intensity_zyx, dtype=float)
    x = np.asarray(stack.x_um, dtype=float)
    y = np.asarray(stack.y_um, dtype=float)
    z = np.asarray(stack.z_um, dtype=float)
    if plane_index is None:
        if target_depth_um is None:
            plane_index = int(np.argmax(np.max(I, axis=(1, 2))))
        else:
            plane_index = _nearest_index(z, float(target_depth_um))
    plane_index = int(np.clip(plane_index, 0, I.shape[0] - 1))
    plane = I[plane_index]

    centroid_x, centroid_y = _centroid(plane, x, y)
    centroid_x_by_z, centroid_y_by_z = _centroid_by_z(I, x, y)
    peak_i = int(np.argmax(I))
    pz, py, px = (int(v) for v in np.unravel_index(peak_i, I.shape))
    peak_value = float(I[pz, py, px])
    peak_x = float(x[px])
    peak_y = float(y[py])
    peak_z = float(z[pz])
    peak_x_by_z, peak_y_by_z = _peak_xy_by_z(I, x, y)

    ring_circularity = _ring_circularity_score(plane, x, y, centroid_x, centroid_y)
    azimuth = _azimuthal_uniformity_score(plane, x, y, centroid_x, centroid_y)
    side_lobe_imbalance = _side_lobe_imbalance(plane, x, y)
    core_fill = _core_fill_fraction(plane, x, y)
    central_darkness = float(np.clip(1.0 - core_fill, 0.0, 1.0))
    symmetry = _symmetry_score(plane)
    captured_drift = _captured_power_drift(fluence_result)
    target_peak = _target_depth_peak(fluence_result, z, target_depth_um)
    roi_peak = _central_roi_peak(fluence_result, x, y, central_roi_half_width_um, plane_index)

    pupil_clip = float((perturbation_metadata or {}).get("pupil_clipped_power_fraction", 0.0) or 0.0)
    first_order = first_order_selected_fraction
    if first_order is None:
        first_order = float((perturbation_metadata or {}).get("first_order_selected_fraction", np.nan))

    peak_fluence = _global_peak_fluence(fluence_result, fallback=peak_value)
    baseline_similarity = 1.0
    if baseline_stack is not None:
        baseline_similarity = _stack_similarity(baseline_stack, stack)
    elif baseline_metrics is not None and np.isfinite(float(baseline_metrics.get("peak_x_um", np.nan))):
        dx = peak_x - float(baseline_metrics.get("peak_x_um", 0.0))
        dy = peak_y - float(baseline_metrics.get("peak_y_um", 0.0))
        baseline_similarity = float(np.exp(-np.hypot(dx, dy) / max(float(np.ptp(x)), 1e-9)))

    peak_change = 0.0
    if baseline_fluence_result is not None:
        base_peak = _global_peak_fluence(baseline_fluence_result, fallback=np.nan)
        if np.isfinite(base_peak) and base_peak != 0.0:
            peak_change = float((peak_fluence - base_peak) / base_peak)
    elif baseline_metrics is not None:
        base_peak = float(baseline_metrics.get("global_peak_fluence", baseline_metrics.get("peak_intensity_value", np.nan)))
        if np.isfinite(base_peak) and base_peak != 0.0:
            peak_change = float((peak_fluence - base_peak) / base_peak)

    return {
        "centroid_x_um": float(centroid_x),
        "centroid_y_um": float(centroid_y),
        "peak_x_um": peak_x,
        "peak_y_um": peak_y,
        "peak_z_um": peak_z,
        "peak_intensity_value": peak_value,
        "centroid_x_span_um": _finite_span(centroid_x_by_z),
        "centroid_y_span_um": _finite_span(centroid_y_by_z),
        "peak_trajectory_x_span_um": _finite_span(peak_x_by_z),
        "peak_trajectory_y_span_um": _finite_span(peak_y_by_z),
        "ring_circularity_score": ring_circularity,
        "azimuthal_uniformity_score": azimuth,
        "side_lobe_imbalance": side_lobe_imbalance,
        "core_fill_fraction": core_fill,
        "central_darkness_contrast": central_darkness,
        "peak_fluence_change_fraction": peak_change,
        "target_depth_peak_fluence": target_peak,
        "central_roi_peak_fluence": roi_peak,
        "captured_power_drift_fraction": captured_drift,
        "pupil_clipped_power_fraction": pupil_clip,
        "first_order_selected_fraction": float(first_order) if first_order is not None else float("nan"),
        "symmetry_score": symmetry,
        "baseline_similarity_score": baseline_similarity,
        "global_peak_fluence": peak_fluence,
    }


def metric_delta_table(
    baseline_metrics: Mapping[str, Any],
    perturbed_metrics: Mapping[str, Any],
) -> list[dict[str, float | str]]:
    rows = []
    for key in [
        "centroid_x_um",
        "centroid_y_um",
        "peak_x_um",
        "peak_y_um",
        "peak_z_um",
        "centroid_x_span_um",
        "centroid_y_span_um",
        "peak_trajectory_x_span_um",
        "peak_trajectory_y_span_um",
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
    ]:
        b = float(baseline_metrics.get(key, np.nan))
        p = float(perturbed_metrics.get(key, np.nan))
        rows.append({"metric": key, "baseline": b, "perturbed": p, "delta": p - b})
    return rows


def build_stage8c3_sensitivity_scenarios() -> dict[str, SensitivityScenario]:
    """Return the required Stage 8C.3B mild/severe perturbation scenarios."""
    return {
        "phase_center_offset": SensitivityScenario(
            key="phase_center_offset",
            title="Scenario A - SLM/vortex/axicon centre offset",
            scenario_name="SLM/vortex/axicon centre offset",
            mild_controls={
                "enable_vortex_centre_offset": True,
                "vortex_centre_offset_x_um": 3.0,
                "enable_axicon_centre_offset": True,
                "axicon_centre_offset_x_um": 3.0,
            },
            severe_controls={
                "enable_vortex_centre_offset": True,
                "vortex_centre_offset_x_um": 8.0,
                "enable_axicon_centre_offset": True,
                "axicon_centre_offset_x_um": 8.0,
            },
            mild_label="vortex x=3 um; axicon x=3 um",
            severe_label="vortex x=8 um; axicon x=8 um",
            degradation_metric="azimuthal_uniformity_score",
            worse_direction="lower",
            expected_visible_change="ring lopsided; dark core shifts/fills; symmetry and azimuthal uniformity drop",
        ),
        "beam_tilt": SensitivityScenario(
            key="beam_tilt",
            title="Scenario B - beam tilt / pointing",
            scenario_name="Beam tilt / pointing",
            mild_controls={
                "enable_beam_tilt": True,
                "beam_tilt_x_mrad": 1.5,
            },
            severe_controls={
                "enable_beam_tilt": True,
                "beam_tilt_x_mrad": 5.0,
            },
            mild_label="beam tilt x=1.5 mrad",
            severe_label="beam tilt x=5.0 mrad",
            degradation_metric="centroid_x_span_um",
            worse_direction="higher",
            expected_visible_change="XZ path skews; peak trajectory span increases",
        ),
        "pupil_clipping": SensitivityScenario(
            key="pupil_clipping",
            title="Scenario C - pupil clipping / decentre",
            scenario_name="Objective pupil clipping / decentre",
            mild_controls={
                "enable_pupil_clipping": True,
                "pupil_decentre_x_um": 3.0,
                "pupil_radius_um": 11.0,
            },
            severe_controls={
                "enable_pupil_clipping": True,
                "pupil_decentre_x_um": 8.0,
                "pupil_radius_um": 7.0,
            },
            mild_label="pupil radius=11 um; decentre x=3 um",
            severe_label="pupil radius=7 um; decentre x=8 um",
            degradation_metric="pupil_clipped_power_fraction",
            worse_direction="higher",
            expected_visible_change="asymmetric side lobes; clipped-power fraction rises; peak fluence changes",
        ),
        "zero_order_leakage": SensitivityScenario(
            key="zero_order_leakage",
            title="Scenario D - zero-order leakage",
            scenario_name="Zero-order leakage",
            mild_controls={
                "enable_zero_order_leakage": True,
                "zero_order_leakage_fraction": 0.02,
            },
            severe_controls={
                "enable_zero_order_leakage": True,
                "zero_order_leakage_fraction": 0.10,
            },
            mild_label="zero-order leakage=0.02",
            severe_label="zero-order leakage=0.10",
            degradation_metric="core_fill_fraction",
            worse_direction="higher",
            expected_visible_change="hollow core fills; central darkness contrast worsens",
        ),
    }


def compute_misalignment_sensitivity_sweep(
    baseline_stack: OpticalFieldStack,
    pulse_energy_uJ: float,
    *,
    scenario: str | SensitivityScenario = "phase_center_offset",
    baseline_fluence: FluenceStackResult | None = None,
    selected_plane_index: int | None = None,
    target_depth_um: float | None = None,
    central_roi_half_width_um: float = 8.0,
) -> SensitivitySweepResult:
    """Compute aligned/mild/severe stacks, fluence maps, and degradation metrics."""
    if not isinstance(baseline_stack, OpticalFieldStack):
        raise TypeError(f"baseline_stack must be OpticalFieldStack; got {type(baseline_stack).__name__}.")
    scenario_def = _scenario_from_value(scenario)
    baseline_fluence = baseline_fluence or scale_stack_to_fluence(baseline_stack, pulse_energy_uJ)
    F0 = np.asarray(baseline_fluence.fluence_zyx_j_cm2, dtype=float)
    if selected_plane_index is None:
        if target_depth_um is None:
            selected_plane_index = int(np.argmax(np.max(F0, axis=(1, 2))))
        else:
            selected_plane_index = _nearest_index(np.asarray(baseline_stack.z_um, dtype=float), target_depth_um)
    selected_plane_index = int(np.clip(selected_plane_index, 0, F0.shape[0] - 1))

    mild_result = apply_lab_perturbations_to_stack(baseline_stack, scenario_def.mild_controls)
    severe_result = apply_lab_perturbations_to_stack(baseline_stack, scenario_def.severe_controls)
    mild_fluence = scale_stack_to_fluence(mild_result.perturbed_stack, pulse_energy_uJ)
    severe_fluence = scale_stack_to_fluence(severe_result.perturbed_stack, pulse_energy_uJ)

    baseline_metrics = compute_degradation_metrics(
        baseline_stack,
        baseline_fluence,
        plane_index=selected_plane_index,
        target_depth_um=target_depth_um,
        central_roi_half_width_um=central_roi_half_width_um,
    )
    mild_metrics = compute_degradation_metrics(
        mild_result.perturbed_stack,
        mild_fluence,
        baseline_stack=baseline_stack,
        baseline_fluence_result=baseline_fluence,
        plane_index=selected_plane_index,
        target_depth_um=target_depth_um,
        central_roi_half_width_um=central_roi_half_width_um,
        perturbation_metadata=mild_result.metadata,
    )
    severe_metrics = compute_degradation_metrics(
        severe_result.perturbed_stack,
        severe_fluence,
        baseline_stack=baseline_stack,
        baseline_fluence_result=baseline_fluence,
        plane_index=selected_plane_index,
        target_depth_um=target_depth_um,
        central_roi_half_width_um=central_roi_half_width_um,
        perturbation_metadata=severe_result.metadata,
    )
    shared_scales = _sweep_shared_scales(
        baseline_stack,
        baseline_fluence,
        mild_fluence,
        severe_fluence,
        selected_plane_index=selected_plane_index,
        central_roi_half_width_um=central_roi_half_width_um,
    )
    metric = scenario_def.degradation_metric
    mild_worse = _is_metric_worse(
        float(mild_metrics.get(metric, np.nan)),
        float(baseline_metrics.get(metric, np.nan)),
        scenario_def.worse_direction,
    )
    severe_worse_than_mild = _is_metric_worse(
        float(severe_metrics.get(metric, np.nan)),
        float(mild_metrics.get(metric, np.nan)),
        scenario_def.worse_direction,
    )
    return SensitivitySweepResult(
        scenario=scenario_def,
        baseline_stack=baseline_stack,
        mild_stack=mild_result.perturbed_stack,
        severe_stack=severe_result.perturbed_stack,
        baseline_fluence=baseline_fluence,
        mild_fluence=mild_fluence,
        severe_fluence=severe_fluence,
        baseline_metrics=baseline_metrics,
        mild_metrics=mild_metrics,
        severe_metrics=severe_metrics,
        mild_delta_rows=tuple(metric_delta_table(baseline_metrics, mild_metrics)),
        severe_delta_rows=tuple(metric_delta_table(baseline_metrics, severe_metrics)),
        shared_scales=shared_scales,
        selected_plane_index=selected_plane_index,
        central_roi_half_width_um=float(central_roi_half_width_um),
        metadata={
            "stage": "stage8c3b_misalignment_sensitivity_sweep",
            "scenario": scenario_def.key,
            "mild_worse_than_baseline": mild_worse,
            "severe_worse_than_mild": severe_worse_than_mild,
            "mild_baseline_similarity": float(mild_metrics.get("baseline_similarity_score", np.nan)),
            "severe_baseline_similarity": float(severe_metrics.get("baseline_similarity_score", np.nan)),
            "degradation_metric": metric,
        },
    )


def plot_misalignment_sensitivity_sweep(
    baseline_stack: OpticalFieldStack,
    pulse_energy_uJ: float,
    *,
    scenario: str | SensitivityScenario = "phase_center_offset",
    baseline_fluence: FluenceStackResult | None = None,
    selected_plane_index: int | None = None,
    target_depth_um: float | None = None,
    central_roi_half_width_um: float = 8.0,
    output_path: str | Path | None = None,
    show_caveats: bool = True,
    dpi: int = 180,
    title: str = "Stage 8C.3B Active Perturbation Sensitivity Sweep",
) -> "matplotlib.figure.Figure":
    """Render a polished aligned/mild/severe/difference diagnostic sweep."""
    if output_path is not None and not show_caveats:
        raise ValueError("Refusing to save Stage 8C.3B sweep without caveats.")
    sweep = compute_misalignment_sensitivity_sweep(
        baseline_stack,
        pulse_energy_uJ,
        scenario=scenario,
        baseline_fluence=baseline_fluence,
        selected_plane_index=selected_plane_index,
        target_depth_um=target_depth_um,
        central_roi_half_width_um=central_roi_half_width_um,
    )
    scenario_def = sweep.scenario
    x = np.asarray(baseline_stack.x_um, dtype=float)
    y = np.asarray(baseline_stack.y_um, dtype=float)
    z = np.asarray(baseline_stack.z_um, dtype=float)
    selected_i = int(sweep.selected_plane_index)
    yc = _nearest_index(y, 0.0)
    roi_x, roi_y = _roi_indices(x, y, sweep.central_roi_half_width_um)
    F0 = np.asarray(sweep.baseline_fluence.fluence_zyx_j_cm2, dtype=float)
    Fm = np.asarray(sweep.mild_fluence.fluence_zyx_j_cm2, dtype=float)
    Fs = np.asarray(sweep.severe_fluence.fluence_zyx_j_cm2, dtype=float)

    xy_maps = [F0[selected_i], Fm[selected_i], Fs[selected_i]]
    xz_maps = [F0[:, yc, :].T, Fm[:, yc, :].T, Fs[:, yc, :].T]
    roi_maps = [
        F0[selected_i][np.ix_(roi_y, roi_x)],
        Fm[selected_i][np.ix_(roi_y, roi_x)],
        Fs[selected_i][np.ix_(roi_y, roi_x)],
    ]
    xy_diff = xy_maps[2] - xy_maps[0]
    xz_diff = xz_maps[2] - xz_maps[0]
    roi_diff = roi_maps[2] - roi_maps[0]

    fig = plt.figure(figsize=(18.2, 12.6), facecolor="white")
    gs = fig.add_gridspec(
        4,
        4,
        height_ratios=[1.08, 1.03, 1.03, 0.95],
        width_ratios=[1.0, 1.0, 1.0, 1.08],
        left=0.090,
        right=0.965,
        top=0.830,
        bottom=0.055,
        hspace=0.42,
        wspace=0.20,
    )
    fig.suptitle(
        f"{title}\n{scenario_def.title}",
        x=0.045,
        y=0.982,
        ha="left",
        va="top",
        fontsize=18,
        fontweight="bold",
    )
    _badge(fig, 0.075, 0.895, "DIAGNOSTIC ONLY", "#0d47a1", "#e3f2fd")
    _badge(fig, 0.195, 0.895, "NO MATERIAL RESPONSE", "#4a148c", "#f3e5f5")
    _badge(fig, 0.355, 0.895, "SHARED COLOUR SCALES", "#1b5e20", "#e8f5e9")
    fig.text(
        0.52,
        0.898,
        f"Mild: {scenario_def.mild_label}    Severe: {scenario_def.severe_label}",
        fontsize=10.5,
        ha="left",
        va="center",
        color="#263238",
    )
    fig.text(
        0.52,
        0.872,
        "Active perturbations are optical/fluence lab-realism diagnostics only; "
        "they do not predict material modification.",
        fontsize=10,
        ha="left",
        va="center",
        color="#37474f",
    )

    axes = np.empty((4, 4), dtype=object)
    for r in range(4):
        for c in range(4):
            axes[r, c] = fig.add_subplot(gs[r, c])

    extent_xy = _extent_from_coords(x, y)
    extent_xz = (float(np.min(z)), float(np.max(z)), float(np.min(x)), float(np.max(x)))
    extent_roi = _extent_from_coords(x[roi_x], y[roi_y])
    col_titles = ["Aligned baseline", "Mild perturbation", "Severe perturbation", "Severe - baseline"]

    xy_images = []
    for idx, arr in enumerate(xy_maps):
        im = axes[0, idx].imshow(
            arr,
            origin="lower",
            extent=extent_xy,
            cmap="viridis",
            vmin=0.0,
            vmax=sweep.shared_scales["xy_vmax"],
            aspect="equal",
        )
        xy_images.append(im)
        axes[0, idx].set_title(col_titles[idx], fontsize=11, fontweight="bold")
        axes[0, idx].set_xlabel("x (um)")
        axes[0, idx].set_ylabel("y (um)")
    im_diff = axes[0, 3].imshow(
        xy_diff,
        origin="lower",
        extent=extent_xy,
        cmap="coolwarm",
        vmin=-sweep.shared_scales["xy_diff_absmax"],
        vmax=sweep.shared_scales["xy_diff_absmax"],
        aspect="equal",
    )
    axes[0, 3].set_title(col_titles[3], fontsize=11, fontweight="bold")
    axes[0, 3].set_xlabel("x (um)")
    axes[0, 3].set_ylabel("y (um)")
    fig.colorbar(xy_images[-1], ax=list(axes[0, 0:3]), fraction=0.022, pad=0.010, label="XY fluence (J/cm^2)")
    fig.colorbar(im_diff, ax=axes[0, 3], fraction=0.046, pad=0.025, label="delta J/cm^2")

    xz_images = []
    for idx, arr in enumerate(xz_maps):
        im = axes[1, idx].imshow(
            arr,
            origin="lower",
            extent=extent_xz,
            cmap="viridis",
            vmin=0.0,
            vmax=sweep.shared_scales["xz_vmax"],
            aspect="auto",
        )
        xz_images.append(im)
        axes[1, idx].axvline(z[selected_i], color="white", lw=1.4, ls="--", alpha=0.9)
        axes[1, idx].set_xlabel("z (um)")
        axes[1, idx].set_ylabel("x (um)")
    im_diff = axes[1, 3].imshow(
        xz_diff,
        origin="lower",
        extent=extent_xz,
        cmap="coolwarm",
        vmin=-sweep.shared_scales["xz_diff_absmax"],
        vmax=sweep.shared_scales["xz_diff_absmax"],
        aspect="auto",
    )
    axes[1, 3].axvline(z[selected_i], color="black", lw=1.2, ls="--", alpha=0.75)
    axes[1, 3].set_xlabel("z (um)")
    axes[1, 3].set_ylabel("x (um)")
    fig.colorbar(xz_images[-1], ax=list(axes[1, 0:3]), fraction=0.022, pad=0.010, label="XZ fluence (J/cm^2)")
    fig.colorbar(im_diff, ax=axes[1, 3], fraction=0.046, pad=0.025, label="delta J/cm^2")

    roi_images = []
    for idx, arr in enumerate(roi_maps):
        im = axes[2, idx].imshow(
            arr,
            origin="lower",
            extent=extent_roi,
            cmap="magma",
            vmin=0.0,
            vmax=sweep.shared_scales["roi_vmax"],
            aspect="equal",
        )
        roi_images.append(im)
        axes[2, idx].set_xlabel("x (um)")
        axes[2, idx].set_ylabel("y (um)")
    im_diff = axes[2, 3].imshow(
        roi_diff,
        origin="lower",
        extent=extent_roi,
        cmap="coolwarm",
        vmin=-sweep.shared_scales["roi_diff_absmax"],
        vmax=sweep.shared_scales["roi_diff_absmax"],
        aspect="equal",
    )
    axes[2, 3].set_xlabel("x (um)")
    axes[2, 3].set_ylabel("y (um)")
    fig.colorbar(roi_images[-1], ax=list(axes[2, 0:3]), fraction=0.022, pad=0.010, label="ROI fluence (J/cm^2)")
    fig.colorbar(im_diff, ax=axes[2, 3], fraction=0.046, pad=0.025, label="delta J/cm^2")

    row_labels = [
        "XY fluence @ selected plane",
        "XZ fluence (y = 0 slice)",
        f"Central ROI/core (+/-{sweep.central_roi_half_width_um:g} um)",
        "Metric deltas / degradation indicators",
    ]
    for r, label in enumerate(row_labels):
        axes[r, 0].text(
            -0.48,
            0.5,
            label,
            transform=axes[r, 0].transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="#263238",
        )

    _metric_card(
        axes[3, 0],
        "Aligned baseline",
        _baseline_metric_lines(sweep.baseline_metrics),
        "#eceff1",
        "#37474f",
    )
    _metric_card(
        axes[3, 1],
        "Mild perturbation",
        _degradation_metric_lines(sweep.baseline_metrics, sweep.mild_metrics),
        "#fff8e1",
        "#f57f17",
    )
    _metric_card(
        axes[3, 2],
        "Severe perturbation",
        _degradation_metric_lines(sweep.baseline_metrics, sweep.severe_metrics),
        "#ffebee",
        "#b71c1c",
    )
    _metric_card(
        axes[3, 3],
        "What changed and why",
        _sweep_summary_lines(sweep),
        "#e8f5e9" if bool(sweep.metadata.get("severe_worse_than_mild")) else "#fff3e0",
        "#1b5e20" if bool(sweep.metadata.get("severe_worse_than_mild")) else "#e65100",
    )

    fig.stage8c3_metadata = {  # type: ignore[attr-defined]
        "stage": "stage8c3b_misalignment_sensitivity_sweep",
        "figure_status": FIGURE_STATUS,
        "model_status": MODEL_STATUS,
        "final_export_allowed": False,
        "scenario": scenario_def.key,
        "shared_xy_vmax": float(sweep.shared_scales["xy_vmax"]),
        "shared_xz_vmax": float(sweep.shared_scales["xz_vmax"]),
        "shared_roi_vmax": float(sweep.shared_scales["roi_vmax"]),
        "mild_baseline_similarity": float(sweep.metadata["mild_baseline_similarity"]),
        "severe_baseline_similarity": float(sweep.metadata["severe_baseline_similarity"]),
        "severe_worse_than_mild": bool(sweep.metadata["severe_worse_than_mild"]),
        "degradation_metric": scenario_def.degradation_metric,
    }
    fig.stage8c3_sensitivity_result = sweep  # type: ignore[attr-defined]

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            out,
            dpi=dpi,
            bbox_inches="tight",
            metadata={
                "Title": title,
                "stage": "stage8c3b_misalignment_sensitivity_sweep",
                "scenario": scenario_def.key,
                "figure_status": FIGURE_STATUS,
                "model_status": MODEL_STATUS,
                "final_export_allowed": "False",
                "Description": "Stage 8C.3B active perturbation severity sweep; optical fluence diagnostic only.",
            },
        )
    return fig


def plot_baseline_vs_perturbed_comparison(
    baseline_stack: OpticalFieldStack,
    perturbed_stack: OpticalFieldStack,
    baseline_fluence: FluenceStackResult,
    perturbed_fluence: FluenceStackResult,
    *,
    selected_plane_index: int | None = None,
    output_path: str | Path | None = None,
    show_caveats: bool = True,
    dpi: int = 180,
    title: str = "Stage 8C.3 Baseline vs Perturbed Misalignment Sanity Check",
) -> "matplotlib.figure.Figure":
    """Plot baseline/perturbed XY and XZ maps with shared colour scaling."""
    if output_path is not None and not show_caveats:
        raise ValueError("Refusing to save Stage 8C.3 comparison without caveats.")
    F0 = np.asarray(baseline_fluence.fluence_zyx_j_cm2, dtype=float)
    F1 = np.asarray(perturbed_fluence.fluence_zyx_j_cm2, dtype=float)
    z = np.asarray(baseline_stack.z_um, dtype=float)
    x = np.asarray(baseline_stack.x_um, dtype=float)
    y = np.asarray(baseline_stack.y_um, dtype=float)
    if selected_plane_index is None:
        selected_plane_index = int(np.argmax(np.max(F0, axis=(1, 2))))
    selected_plane_index = int(np.clip(selected_plane_index, 0, F0.shape[0] - 1))
    yc = _nearest_index(y, 0.0)

    xy0 = F0[selected_plane_index]
    xy1 = F1[selected_plane_index]
    xz0 = F0[:, yc, :].T
    xz1 = F1[:, yc, :].T
    xy_vmax = float(np.nanmax([np.nanmax(xy0), np.nanmax(xy1)]))
    xz_vmax = float(np.nanmax([np.nanmax(xz0), np.nanmax(xz1)]))
    xy_diff = xy1 - xy0
    xz_diff = xz1 - xz0
    diff_v = max(float(np.nanmax(np.abs(xy_diff))), float(np.nanmax(np.abs(xz_diff))), 1e-12)

    fig, axes = plt.subplots(3, 3, figsize=(15, 13), constrained_layout=True)
    fig.suptitle(
        f"{title}\nshared colour scales; optical fluence diagnostic only; final_export_allowed=False",
        fontsize=14,
    )
    extent_xy = _extent_from_coords(x, y)
    extent_xz = (float(np.min(z)), float(np.max(z)), float(np.min(x)), float(np.max(x)))

    im = axes[0, 0].imshow(xy0, origin="lower", extent=extent_xy, cmap="viridis", vmin=0, vmax=xy_vmax)
    axes[0, 0].set_title("baseline XY fluence")
    fig.colorbar(im, ax=axes[0, 0], label="J/cm^2")
    im = axes[0, 1].imshow(xy1, origin="lower", extent=extent_xy, cmap="viridis", vmin=0, vmax=xy_vmax)
    axes[0, 1].set_title("perturbed XY fluence")
    fig.colorbar(im, ax=axes[0, 1], label="J/cm^2")
    im = axes[0, 2].imshow(xy_diff, origin="lower", extent=extent_xy, cmap="coolwarm", vmin=-diff_v, vmax=diff_v)
    axes[0, 2].set_title("difference XY map")
    fig.colorbar(im, ax=axes[0, 2], label="delta J/cm^2")

    im = axes[1, 0].imshow(xz0, origin="lower", extent=extent_xz, aspect="auto", cmap="viridis", vmin=0, vmax=xz_vmax)
    axes[1, 0].set_title("baseline XZ fluence")
    fig.colorbar(im, ax=axes[1, 0], label="J/cm^2")
    im = axes[1, 1].imshow(xz1, origin="lower", extent=extent_xz, aspect="auto", cmap="viridis", vmin=0, vmax=xz_vmax)
    axes[1, 1].set_title("perturbed XZ fluence")
    fig.colorbar(im, ax=axes[1, 1], label="J/cm^2")
    im = axes[1, 2].imshow(xz_diff, origin="lower", extent=extent_xz, aspect="auto", cmap="coolwarm", vmin=-diff_v, vmax=diff_v)
    axes[1, 2].set_title("difference XZ map")
    fig.colorbar(im, ax=axes[1, 2], label="delta J/cm^2")

    b_metrics = compute_degradation_metrics(baseline_stack, baseline_fluence)
    p_metrics = compute_degradation_metrics(
        perturbed_stack,
        perturbed_fluence,
        baseline_stack=baseline_stack,
        baseline_fluence_result=baseline_fluence,
    )
    rows = metric_delta_table(b_metrics, p_metrics)
    axes[2, 0].axis("off")
    axes[2, 1].axis("off")
    axes[2, 2].axis("off")
    table_rows = rows[:10]
    table = axes[2, 0].table(
        cellText=[
            [r["metric"], f"{r['baseline']:.3g}", f"{r['perturbed']:.3g}", f"{r['delta']:.3g}"]
            for r in table_rows
        ],
        colLabels=["metric", "baseline", "perturbed", "delta"],
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(2.9, 1.25)
    axes[2, 0].set_title("metric delta table", loc="left")
    axes[2, 1].text(
        0.0, 0.9,
        "Interpretation:\n"
        "The baseline and perturbed maps use identical axes and identical colour scales.\n"
        "Difference panels show the active perturbation response directly.\n"
        "This is optical fluence only; no material response is predicted.",
        va="top", ha="left", fontsize=10,
    )
    axes[2, 2].text(
        0.0, 0.9,
        f"selected z = {z[selected_plane_index]:.3g} um\n"
        f"XY shared vmax = {xy_vmax:.3g} J/cm^2\n"
        f"XZ shared vmax = {xz_vmax:.3g} J/cm^2\n"
        f"baseline_similarity_score = {p_metrics['baseline_similarity_score']:.3g}",
        va="top", ha="left", fontsize=10, family="monospace",
    )

    for ax in axes[:2].ravel():
        ax.set_xlabel("x or z (um)")
        ax.set_ylabel("y or x (um)")

    fig.stage8c3_metadata = {  # type: ignore[attr-defined]
        "stage": "stage8c3_active_lab_realism",
        "figure_status": FIGURE_STATUS,
        "model_status": MODEL_STATUS,
        "final_export_allowed": False,
        "shared_xy_vmax": xy_vmax,
        "shared_xz_vmax": xz_vmax,
    }
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            out,
            dpi=dpi,
            bbox_inches="tight",
            metadata={
                "Title": title,
                "stage": "stage8c3_active_lab_realism",
                "figure_status": FIGURE_STATUS,
                "model_status": MODEL_STATUS,
                "final_export_allowed": "False",
                "Description": "Stage 8C.3 baseline-vs-perturbed optical fluence diagnostic only.",
            },
        )
    return fig


def _scenario_from_value(scenario: str | SensitivityScenario) -> SensitivityScenario:
    if isinstance(scenario, SensitivityScenario):
        return scenario
    scenarios = build_stage8c3_sensitivity_scenarios()
    key = str(scenario)
    if key not in scenarios:
        raise KeyError(f"Unknown Stage 8C.3B sensitivity scenario {key!r}; allowed: {sorted(scenarios)}.")
    return scenarios[key]


def _sweep_shared_scales(
    stack: OpticalFieldStack,
    baseline_fluence: FluenceStackResult,
    mild_fluence: FluenceStackResult,
    severe_fluence: FluenceStackResult,
    *,
    selected_plane_index: int,
    central_roi_half_width_um: float,
) -> dict[str, float]:
    x = np.asarray(stack.x_um, dtype=float)
    y = np.asarray(stack.y_um, dtype=float)
    yc = _nearest_index(y, 0.0)
    roi_x, roi_y = _roi_indices(x, y, central_roi_half_width_um)
    maps = [
        np.asarray(baseline_fluence.fluence_zyx_j_cm2, dtype=float),
        np.asarray(mild_fluence.fluence_zyx_j_cm2, dtype=float),
        np.asarray(severe_fluence.fluence_zyx_j_cm2, dtype=float),
    ]
    xy = [F[selected_plane_index] for F in maps]
    xz = [F[:, yc, :].T for F in maps]
    roi = [F[selected_plane_index][np.ix_(roi_y, roi_x)] for F in maps]
    return {
        "xy_vmax": max(float(np.nanmax(a)) for a in xy),
        "xz_vmax": max(float(np.nanmax(a)) for a in xz),
        "roi_vmax": max(float(np.nanmax(a)) for a in roi),
        "xy_diff_absmax": max(float(np.nanmax(np.abs(xy[2] - xy[0]))), 1e-12),
        "xz_diff_absmax": max(float(np.nanmax(np.abs(xz[2] - xz[0]))), 1e-12),
        "roi_diff_absmax": max(float(np.nanmax(np.abs(roi[2] - roi[0]))), 1e-12),
    }


def _roi_indices(
    x: np.ndarray,
    y: np.ndarray,
    half_width_um: float,
) -> tuple[np.ndarray, np.ndarray]:
    half = float(half_width_um)
    roi_x = np.flatnonzero(np.abs(x) <= half)
    roi_y = np.flatnonzero(np.abs(y) <= half)
    if roi_x.size == 0:
        roi_x = np.array([_nearest_index(x, 0.0)])
    if roi_y.size == 0:
        roi_y = np.array([_nearest_index(y, 0.0)])
    return roi_x, roi_y


def _is_metric_worse(value: float, reference: float, direction: str) -> bool:
    if not np.isfinite(value) or not np.isfinite(reference):
        return False
    if direction == "higher":
        return value > reference
    if direction == "lower":
        return value < reference
    raise ValueError(f"Unknown worse direction {direction!r}.")


def _badge(fig: Any, x: float, y: float, text: str, edge: str, face: str) -> None:
    fig.text(
        x,
        y,
        text,
        ha="left",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color=edge,
        bbox=dict(boxstyle="round,pad=0.32", facecolor=face, edgecolor=edge, lw=1.2),
    )


def _metric_card(ax: Any, title: str, lines: list[str], face: str, edge: str) -> None:
    ax.set_axis_off()
    ax.add_patch(
        plt.Rectangle(
            (0.0, 0.0),
            1.0,
            1.0,
            transform=ax.transAxes,
            facecolor=face,
            edgecolor=edge,
            lw=1.4,
            clip_on=False,
        )
    )
    ax.text(
        0.035,
        0.93,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
        color=edge,
    )
    ax.text(
        0.035,
        0.82,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.8,
        family="monospace",
        color="#263238",
        linespacing=1.18,
    )


def _baseline_metric_lines(metrics: Mapping[str, float]) -> list[str]:
    return [
        f"centroid      : ({_fmt_num(metrics.get('centroid_x_um'))}, {_fmt_num(metrics.get('centroid_y_um'))}) um",
        f"peak xyz      : ({_fmt_num(metrics.get('peak_x_um'))}, {_fmt_num(metrics.get('peak_y_um'))}, {_fmt_num(metrics.get('peak_z_um'))})",
        f"peak fluence  : {_fmt_num(metrics.get('global_peak_fluence'))}",
        f"az uniformity : {_fmt_num(metrics.get('azimuthal_uniformity_score'))}",
        f"ring circular : {_fmt_num(metrics.get('ring_circularity_score'))}",
        f"core fill     : {_fmt_num(metrics.get('core_fill_fraction'))}",
        f"symmetry      : {_fmt_num(metrics.get('symmetry_score'))}",
        f"pupil clipped : {_fmt_pct_value(metrics.get('pupil_clipped_power_fraction'))}",
        f"baseline sim. : {_fmt_num(metrics.get('baseline_similarity_score'))}",
    ]


def _degradation_metric_lines(
    baseline: Mapping[str, float],
    metrics: Mapping[str, float],
) -> list[str]:
    centroid_shift = _xy_shift(baseline, metrics, "centroid_x_um", "centroid_y_um")
    peak_shift_xy = _xy_shift(baseline, metrics, "peak_x_um", "peak_y_um")
    peak_shift_z = float(metrics.get("peak_z_um", np.nan)) - float(baseline.get("peak_z_um", np.nan))
    return [
        f"centroid shift: {_fmt_num(centroid_shift)} um",
        f"peak shift    : {_fmt_num(peak_shift_xy)} um xy, dz={_fmt_num(peak_shift_z)}",
        f"peak fluence  : {_fmt_signed_pct(metrics.get('peak_fluence_change_fraction'))}",
        f"az uniformity : {_fmt_num(metrics.get('azimuthal_uniformity_score'))} ({_fmt_delta(metrics, baseline, 'azimuthal_uniformity_score')})",
        f"ring circular : {_fmt_num(metrics.get('ring_circularity_score'))} ({_fmt_delta(metrics, baseline, 'ring_circularity_score')})",
        f"core fill     : {_fmt_num(metrics.get('core_fill_fraction'))} ({_fmt_delta(metrics, baseline, 'core_fill_fraction')})",
        f"symmetry      : {_fmt_num(metrics.get('symmetry_score'))} ({_fmt_delta(metrics, baseline, 'symmetry_score')})",
        f"pupil clipped : {_fmt_pct_value(metrics.get('pupil_clipped_power_fraction'))}",
        f"baseline sim. : {_fmt_num(metrics.get('baseline_similarity_score'))}",
    ]


def _sweep_summary_lines(sweep: SensitivitySweepResult) -> list[str]:
    metric = sweep.scenario.degradation_metric
    base = float(sweep.baseline_metrics.get(metric, np.nan))
    mild = float(sweep.mild_metrics.get(metric, np.nan))
    severe = float(sweep.severe_metrics.get(metric, np.nan))
    return [
        f"scenario      : {sweep.scenario.scenario_name}",
        f"expected      : {sweep.scenario.expected_visible_change}",
        f"severity metric: {metric}",
        f"baseline/mild/severe:",
        f"  {_fmt_num(base)} -> {_fmt_num(mild)} -> {_fmt_num(severe)}",
        f"severe worse  : {bool(sweep.metadata.get('severe_worse_than_mild'))}",
        f"similarity    : mild {_fmt_num(sweep.mild_metrics.get('baseline_similarity_score'))}",
        f"                severe {_fmt_num(sweep.severe_metrics.get('baseline_similarity_score'))}",
        "claim boundary: optical fluence only",
        "no material response predicted",
    ]


def _xy_shift(
    baseline: Mapping[str, float],
    metrics: Mapping[str, float],
    x_key: str,
    y_key: str,
) -> float:
    dx = float(metrics.get(x_key, np.nan)) - float(baseline.get(x_key, np.nan))
    dy = float(metrics.get(y_key, np.nan)) - float(baseline.get(y_key, np.nan))
    return float(np.hypot(dx, dy))


def _fmt_num(value: Any) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(v):
        return "n/a"
    if abs(v) >= 100:
        return f"{v:.3g}"
    if abs(v) >= 10:
        return f"{v:.3f}"
    return f"{v:.4f}"


def _fmt_delta(
    metrics: Mapping[str, float],
    baseline: Mapping[str, float],
    key: str,
) -> str:
    value = float(metrics.get(key, np.nan)) - float(baseline.get(key, np.nan))
    if not np.isfinite(value):
        return "d=n/a"
    return f"d={value:+.3g}"


def _fmt_pct_value(value: Any) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(v):
        return "n/a"
    return f"{100.0 * v:.2f}%"


def _fmt_signed_pct(value: Any) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(v):
        return "n/a"
    return f"{100.0 * v:+.2f}%"


def _centroid(plane: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    total = float(np.sum(plane))
    if total <= 0:
        return float("nan"), float("nan")
    X, Y = np.meshgrid(x, y, indexing="xy")
    return float(np.sum(plane * X) / total), float(np.sum(plane * Y) / total)


def _centroid_by_z(I: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xs: list[float] = []
    ys: list[float] = []
    for plane in np.asarray(I, dtype=float):
        cx, cy = _centroid(plane, x, y)
        xs.append(cx)
        ys.append(cy)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def _peak_xy_by_z(I: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    px: list[float] = []
    py: list[float] = []
    for plane in np.asarray(I, dtype=float):
        iy, ix = (int(v) for v in np.unravel_index(int(np.argmax(plane)), plane.shape))
        px.append(float(x[ix]))
        py.append(float(y[iy]))
    return np.asarray(px, dtype=float), np.asarray(py, dtype=float)


def _finite_span(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.max(finite) - np.min(finite))


def _ring_circularity_score(plane: np.ndarray, x: np.ndarray, y: np.ndarray, cx: float, cy: float) -> float:
    X, Y = np.meshgrid(x, y, indexing="xy")
    R = np.hypot(X - cx, Y - cy)
    threshold = 0.55 * float(np.max(plane)) if plane.size else 0.0
    mask = plane >= threshold
    if np.count_nonzero(mask) < 4:
        return 0.0
    radii = R[mask]
    score = 1.0 - float(np.std(radii) / max(np.mean(radii), 1e-12))
    return float(np.clip(score, 0.0, 1.0))


def _azimuthal_uniformity_score(plane: np.ndarray, x: np.ndarray, y: np.ndarray, cx: float, cy: float) -> float:
    X, Y = np.meshgrid(x, y, indexing="xy")
    R = np.hypot(X - cx, Y - cy)
    theta = (np.arctan2(Y - cy, X - cx) + 2.0 * np.pi) % (2.0 * np.pi)
    idx = np.unravel_index(int(np.argmax(plane)), plane.shape)
    r0 = float(R[idx])
    dr = max(0.18 * r0, np.mean(np.abs(np.diff(x))) if x.size > 1 else 1.0)
    mask = (R >= max(0.0, r0 - dr)) & (R <= r0 + dr)
    if np.count_nonzero(mask) < 8:
        return 0.0
    bins = np.linspace(0.0, 2.0 * np.pi, 17)
    vals = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        b = mask & (theta >= lo) & (theta < hi)
        if np.any(b):
            vals.append(float(np.mean(plane[b])))
    if not vals or np.mean(vals) <= 0:
        return 0.0
    score = 1.0 - float(np.std(vals) / np.mean(vals))
    return float(np.clip(score, 0.0, 1.0))


def _side_lobe_imbalance(plane: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    X, Y = np.meshgrid(x, y, indexing="xy")
    left = float(np.sum(plane[X < 0]))
    right = float(np.sum(plane[X >= 0]))
    up = float(np.sum(plane[Y >= 0]))
    down = float(np.sum(plane[Y < 0]))
    denom1 = max(left + right, 1e-12)
    denom2 = max(up + down, 1e-12)
    return float(max(abs(left - right) / denom1, abs(up - down) / denom2))


def _core_fill_fraction(plane: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    X, Y = np.meshgrid(x, y, indexing="xy")
    R = np.hypot(X, Y)
    idx = np.unravel_index(int(np.argmax(plane)), plane.shape)
    r_peak = max(float(R[idx]), max(float(np.mean(np.abs(np.diff(x)))) if x.size > 1 else 1.0, 1e-9))
    mask = R <= max(0.25 * r_peak, 1e-9)
    if not np.any(mask) or float(np.max(plane)) <= 0:
        return 0.0
    return float(np.clip(np.mean(plane[mask]) / float(np.max(plane)), 0.0, 1.0))


def _symmetry_score(plane: np.ndarray) -> float:
    if plane.size == 0 or float(np.max(plane)) <= 0:
        return 0.0
    flip_x = np.fliplr(plane)
    flip_y = np.flipud(plane)
    rms = 0.5 * (
        np.sqrt(np.mean((plane - flip_x) ** 2)) + np.sqrt(np.mean((plane - flip_y) ** 2))
    )
    return float(np.clip(1.0 - rms / max(float(np.max(plane)), 1e-12), 0.0, 1.0))


def _captured_power_drift(fluence_result: FluenceStackResult | None) -> float:
    if fluence_result is None:
        return float("nan")
    return float(fluence_result.propagation_energy_drift_fraction)


def _target_depth_peak(fluence_result: FluenceStackResult | None, z: np.ndarray, target_depth_um: float | None) -> float:
    if fluence_result is None:
        return float("nan")
    idx = int(np.argmax(fluence_result.peak_fluence_by_z_j_cm2)) if target_depth_um is None else _nearest_index(z, target_depth_um)
    return float(np.max(fluence_result.fluence_zyx_j_cm2[idx]))


def _central_roi_peak(
    fluence_result: FluenceStackResult | None,
    x: np.ndarray,
    y: np.ndarray,
    half_width: float,
    plane_index: int,
) -> float:
    if fluence_result is None:
        return float("nan")
    xi = np.flatnonzero(np.abs(x) <= float(half_width))
    yi = np.flatnonzero(np.abs(y) <= float(half_width))
    if xi.size == 0 or yi.size == 0:
        return float("nan")
    return float(np.max(fluence_result.fluence_zyx_j_cm2[int(plane_index)][np.ix_(yi, xi)]))


def _global_peak_fluence(fluence_result: FluenceStackResult | None, fallback: float) -> float:
    if fluence_result is None:
        return float(fallback)
    return float(np.max(fluence_result.fluence_zyx_j_cm2))


def _stack_similarity(baseline: OpticalFieldStack, perturbed: OpticalFieldStack) -> float:
    a = np.asarray(baseline.intensity_zyx, dtype=float)
    b = np.asarray(perturbed.intensity_zyx, dtype=float)
    if a.shape != b.shape:
        return float("nan")
    a_norm = a.ravel() / max(float(np.linalg.norm(a.ravel())), 1e-12)
    b_norm = b.ravel() / max(float(np.linalg.norm(b.ravel())), 1e-12)
    relative_distance = float(np.linalg.norm(a_norm - b_norm) / np.sqrt(2.0))
    return float(np.clip(1.0 - relative_distance, 0.0, 1.0))


def _nearest_index(coords: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(np.asarray(coords, dtype=float) - float(value))))


def _extent_from_coords(x_um: np.ndarray, y_um: np.ndarray) -> tuple[float, float, float, float]:
    x = np.asarray(x_um, dtype=float)
    y = np.asarray(y_um, dtype=float)
    dx = float(np.mean(np.abs(np.diff(x)))) if x.size > 1 else 1.0
    dy = float(np.mean(np.abs(np.diff(y)))) if y.size > 1 else 1.0
    return (
        float(np.min(x) - 0.5 * dx),
        float(np.max(x) + 0.5 * dx),
        float(np.min(y) - 0.5 * dy),
        float(np.max(y) + 0.5 * dy),
    )
