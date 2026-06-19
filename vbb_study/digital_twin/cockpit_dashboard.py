"""Integrated Stage 8C.1 cockpit dashboard and field diagnostics.

The helpers here choose safe display planes, expose central-ROI and target-depth
fluence metrics, and render the beam-to-write cockpit. They do not implement
material response and do not change optical propagation physics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

import matplotlib
matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt  # noqa: E402

from vbb_study.digital_twin.energy_accounting import EnergyLedger
from vbb_study.digital_twin.exposure_bookkeeping import line_exposure_summary
from vbb_study.digital_twin.field_coupling import OpticalFieldStack
from vbb_study.digital_twin.field_figures import CaveatsRequiredError
from vbb_study.digital_twin.field_fluence import (
    FluenceStackResult,
    peak_intensity_from_fluence_result,
)

STAGE = "stage8c1_integrated_cockpit_mvp"
MODEL_STATUS = "diagnostic_preview"
FIGURE_STATUS = "diagnostic_allowed"
FINAL_EXPORT_ALLOWED = False


def choose_display_plane(
    z_um: np.ndarray,
    *,
    selected_z_mode: str = "target_depth",
    target_depth_um: float = 0.0,
    custom_z_um: float | None = None,
    optical_peak_z_um: float | None = None,
    global_peak_near_boundary: bool = False,
) -> dict[str, Any]:
    """Choose the cockpit display plane without promoting crop-edge artifacts."""
    z = np.asarray(z_um, dtype=float)
    if z.ndim != 1 or z.size == 0:
        raise ValueError("z_um must be a non-empty 1D coordinate array.")

    mode = str(selected_z_mode or "target_depth")
    if mode == "optical_peak":
        if global_peak_near_boundary:
            idx = _nearest_index(z, target_depth_um)
            reason = "target_depth_safe: global optical peak is near crop boundary"
        else:
            target = target_depth_um if optical_peak_z_um is None else float(optical_peak_z_um)
            idx = _nearest_index(z, target)
            reason = "optical_peak"
    elif mode == "sample_surface":
        idx = _nearest_index(z, 0.0)
        reason = "sample_surface"
    elif mode == "custom":
        if custom_z_um is None:
            raise ValueError("custom_z_um is required when selected_z_mode='custom'.")
        idx = _nearest_index(z, float(custom_z_um))
        reason = "custom"
    else:
        idx = _nearest_index(z, float(target_depth_um))
        reason = "target_depth"

    return {
        "selected_plane_index": int(idx),
        "selected_plane_z_um": float(z[idx]),
        "selected_plane_reason": reason,
        "requested_mode": mode,
    }


def compute_peak_location_diagnostics(
    stack: OpticalFieldStack,
    fluence_result: FluenceStackResult,
    *,
    target_depth_um: float = 0.0,
    central_roi_half_width_um: float = 10.0,
    selected_z_mode: str = "target_depth",
    custom_z_um: float | None = None,
    boundary_margin_px: int = 1,
    pulse_duration_fs: float | None = None,
) -> dict[str, Any]:
    """Compute global, central-ROI, target-depth, and surface fluence diagnostics."""
    if not isinstance(stack, OpticalFieldStack):
        raise TypeError(f"stack must be OpticalFieldStack; got {type(stack).__name__}.")
    if not isinstance(fluence_result, FluenceStackResult):
        raise TypeError(
            f"fluence_result must be FluenceStackResult; got {type(fluence_result).__name__}."
        )

    F = np.asarray(fluence_result.fluence_zyx_j_cm2, dtype=float)
    if F.shape != stack.intensity_zyx.shape:
        raise ValueError("fluence stack shape does not match optical stack shape.")
    if not np.all(np.isfinite(F)):
        raise ValueError("fluence stack contains non-finite values.")

    nz, ny, nx = F.shape
    z = np.asarray(stack.z_um, dtype=float)
    x = np.asarray(stack.x_um, dtype=float)
    y = np.asarray(stack.y_um, dtype=float)

    global_flat = int(np.argmax(F))
    gz_i, gy_i, gx_i = (int(v) for v in np.unravel_index(global_flat, F.shape))
    global_peak_value = float(F[gz_i, gy_i, gx_i])
    distance_px = int(min(gy_i, ny - 1 - gy_i, gx_i, nx - 1 - gx_i))
    near_boundary = bool(distance_px <= int(boundary_margin_px))

    roi_x_idx = np.flatnonzero(np.abs(x) <= float(central_roi_half_width_um))
    roi_y_idx = np.flatnonzero(np.abs(y) <= float(central_roi_half_width_um))
    if roi_x_idx.size == 0:
        roi_x_idx = np.array([_nearest_index(x, 0.0)])
    if roi_y_idx.size == 0:
        roi_y_idx = np.array([_nearest_index(y, 0.0)])
    roi = F[:, roi_y_idx[:, None], roi_x_idx]
    roi_flat = int(np.argmax(roi))
    rz_i, ry_local, rx_local = (int(v) for v in np.unravel_index(roi_flat, roi.shape))
    ry_i = int(roi_y_idx[ry_local])
    rx_i = int(roi_x_idx[rx_local])

    central_roi_peak_by_z = np.max(roi, axis=(1, 2))
    target_i = _nearest_index(z, float(target_depth_um))
    surface_i = _nearest_index(z, 0.0)

    selected = choose_display_plane(
        z,
        selected_z_mode=selected_z_mode,
        target_depth_um=float(target_depth_um),
        custom_z_um=custom_z_um,
        optical_peak_z_um=float(z[gz_i]),
        global_peak_near_boundary=near_boundary,
    )
    selected_i = int(selected["selected_plane_index"])

    drift = float(fluence_result.propagation_energy_drift_fraction)
    warning_messages: list[str] = []
    if near_boundary:
        warning_messages.append("global optical peak is near the crop boundary")
    if np.isfinite(drift) and drift > 0.20:
        warning_messages.append(f"raw captured-power drift is high ({drift:.1%})")
    warning_level = "caution" if warning_messages else "pass"

    peak_intensity = None
    if pulse_duration_fs is not None:
        peak_intensity = peak_intensity_from_fluence_result(fluence_result, pulse_duration_fs)

    return {
        "global_peak_value": global_peak_value,
        "global_peak_x_um": float(x[gx_i]),
        "global_peak_y_um": float(y[gy_i]),
        "global_peak_z_um": float(z[gz_i]),
        "global_peak_index_zyx": (gz_i, gy_i, gx_i),
        "global_peak_distance_to_boundary_px": distance_px,
        "global_peak_near_boundary": near_boundary,
        "central_roi_peak_value": float(F[rz_i, ry_i, rx_i]),
        "central_roi_peak_x_um": float(x[rx_i]),
        "central_roi_peak_y_um": float(y[ry_i]),
        "central_roi_peak_z_um": float(z[rz_i]),
        "central_roi_peak_index_zyx": (rz_i, ry_i, rx_i),
        "central_roi_peak_by_z_j_cm2": central_roi_peak_by_z,
        "target_depth_peak_value": float(np.max(F[target_i])),
        "target_depth_z_um": float(z[target_i]),
        "target_depth_index": int(target_i),
        "sample_surface_peak_value": float(np.max(F[surface_i])),
        "sample_surface_z_um": float(z[surface_i]),
        "sample_surface_index": int(surface_i),
        "selected_plane_z_um": selected["selected_plane_z_um"],
        "selected_plane_index": selected_i,
        "selected_plane_reason": selected["selected_plane_reason"],
        "selected_plane_peak_value": float(np.max(F[selected_i])),
        "captured_power_drift_fraction": drift,
        "raw_captured_power_fraction_by_z": fluence_result.raw_captured_power_fraction_by_z,
        "peak_intensity_w_cm2": peak_intensity,
        "warning_level": warning_level,
        "warning_messages": warning_messages,
    }


def build_warning_flags(
    *,
    energy_ledger: EnergyLedger | None = None,
    exposure_summary: Mapping[str, Any] | None = None,
    diagnostics: Mapping[str, Any] | None = None,
    first_order_geometry_valid: bool | None = None,
    pupil_clipping_fraction: float | None = None,
    power_limit_exceeded: bool | None = None,
    pulse_overlap_fraction: float | None = None,
    captured_power_drift_fraction: float | None = None,
) -> dict[str, dict[str, str]]:
    """Build cockpit warning flags for hardware-feasibility panels."""
    flags = {
        "captured_power_drift": {"status": "pass", "message": "raw captured-power drift acceptable"},
        "crop_edge_peak": {"status": "pass", "message": "global peak is not near crop edge"},
        "first_order_geometry": {"status": "pass", "message": "first-order geometry valid"},
        "pupil_clipping": {"status": "pass", "message": "pupil clipping not detected"},
        "power_limit": {"status": "pass", "message": "average power within configured limit"},
        "pulse_overlap": {"status": "pass", "message": "pulse spacing supports continuous scan bookkeeping"},
    }

    diag = dict(diagnostics or {})
    drift = captured_power_drift_fraction
    if drift is None:
        drift = diag.get("captured_power_drift_fraction")
    if drift is not None and np.isfinite(float(drift)) and float(drift) > 0.20:
        flags["captured_power_drift"] = {
            "status": "caution",
            "message": f"raw captured-power drift {float(drift):.1%}",
        }

    if bool(diag.get("global_peak_near_boundary", False)):
        flags["crop_edge_peak"] = {
            "status": "caution",
            "message": "global peak is near crop boundary; do not use as headline plane",
        }

    if first_order_geometry_valid is False:
        flags["first_order_geometry"] = {
            "status": "caution",
            "message": "first-order geometry is not currently valid",
        }

    if pupil_clipping_fraction is not None and np.isfinite(float(pupil_clipping_fraction)):
        if float(pupil_clipping_fraction) > 0:
            flags["pupil_clipping"] = {
                "status": "caution",
                "message": f"pupil clipping fraction {float(pupil_clipping_fraction):.3g}",
            }

    if power_limit_exceeded is None and energy_ledger is not None:
        power_limit_exceeded = any("exceeds limit" in str(w).lower() for w in energy_ledger.ledger_warnings)
    if power_limit_exceeded:
        flags["power_limit"] = {"status": "fail", "message": "average power exceeds configured limit"}

    if exposure_summary is not None:
        pulse_overlap_fraction = exposure_summary.get("overlap_fraction", pulse_overlap_fraction)
        if any("gap" in str(w).lower() or "dotted" in str(w).lower() for w in exposure_summary.get("warnings", [])):
            flags["pulse_overlap"] = {
                "status": "caution",
                "message": "pulse spacing warning from exposure bookkeeping",
            }
    if pulse_overlap_fraction is not None and np.isfinite(float(pulse_overlap_fraction)):
        if float(pulse_overlap_fraction) < 0:
            flags["pulse_overlap"] = {"status": "caution", "message": "pulse spacing exceeds effective diameter"}

    return flags


def build_cockpit_summary(
    *,
    controls: Mapping[str, Any],
    energy_ledger: EnergyLedger | None = None,
    exposure_summary: Mapping[str, Any] | None = None,
    diagnostics: Mapping[str, Any] | None = None,
    lab_report: Any | None = None,
) -> dict[str, Any]:
    """Collect cockpit inputs and computed outputs into one summary mapping."""
    if exposure_summary is None and energy_ledger is not None:
        exposure_summary = line_exposure_summary(
            energy_ledger.energy_at_sample_uJ,
            controls.get("repetition_rate_Hz", 25_000.0),
            controls.get("scan_speed_mm_s", 1.0),
            controls.get("line_length_um", 500.0),
            controls.get("effective_diameter_um", 3.0),
        )
    return {
        "controls": dict(controls),
        "energy_ledger": energy_ledger,
        "exposure_summary": dict(exposure_summary or {}),
        "diagnostics": dict(diagnostics or {}),
        "lab_report": lab_report,
        "warning_flags": build_warning_flags(
            energy_ledger=energy_ledger,
            exposure_summary=exposure_summary,
            diagnostics=diagnostics,
        ),
        "model_status": MODEL_STATUS,
        "final_export_allowed": FINAL_EXPORT_ALLOWED,
    }


def plot_integrated_cockpit_dashboard(
    stack: OpticalFieldStack,
    fluence_result: FluenceStackResult,
    *,
    controls: Mapping[str, Any] | None = None,
    energy_ledger: EnergyLedger | None = None,
    exposure_summary: Mapping[str, Any] | None = None,
    lab_report: Any | None = None,
    diagnostics: Mapping[str, Any] | None = None,
    output_path: str | Path | None = None,
    show_caveats: bool = True,
    dpi: int = 180,
    display_scaling: str = "percentile",
    display_percentile_clip: tuple[float, float] = (0.5, 99.5),
    title: str = "Stage 8C.1 Integrated Beam-to-Write Cockpit MVP",
) -> "matplotlib.figure.Figure":
    """Render the integrated optical/energy/exposure cockpit dashboard."""
    controls = dict(controls or {})
    if output_path is not None and not show_caveats:
        raise CaveatsRequiredError(
            "Refusing to save Stage 8C.1 dashboard with show_caveats=False."
        )
    if diagnostics is None:
        diagnostics = compute_peak_location_diagnostics(
            stack,
            fluence_result,
            target_depth_um=float(controls.get("focus_depth_um", 0.0)),
            central_roi_half_width_um=float(controls.get("central_roi_half_width_um", 10.0)),
            selected_z_mode=str(controls.get("selected_z_mode", controls.get("selected_z_um", "target_depth"))),
            custom_z_um=controls.get("custom_z_um"),
            pulse_duration_fs=controls.get("pulse_duration_fs"),
        )
    diagnostics = dict(diagnostics)
    if exposure_summary is None and energy_ledger is not None:
        exposure_summary = line_exposure_summary(
            energy_ledger.energy_at_sample_uJ,
            float(controls.get("repetition_rate_Hz", 25_000.0)),
            float(controls.get("scan_speed_mm_s", 1.0)),
            float(controls.get("line_length_um", 500.0)),
            float(controls.get("effective_diameter_um", 3.0)),
        )
    exposure_summary = dict(exposure_summary or {})
    warning_flags = build_warning_flags(
        energy_ledger=energy_ledger,
        exposure_summary=exposure_summary,
        diagnostics=diagnostics,
    )

    fig, axes = plt.subplots(4, 3, figsize=(17, 17), constrained_layout=True)
    fig.suptitle(
        f"{title}\noptical/fluence diagnostic only - no material response",
        fontsize=15,
    )

    selected_idx = int(diagnostics["selected_plane_index"])
    z = np.asarray(stack.z_um, dtype=float)
    x = np.asarray(stack.x_um, dtype=float)
    y = np.asarray(stack.y_um, dtype=float)
    extent_xy = _extent_from_coords(x, y)
    F = fluence_result.fluence_zyx_j_cm2

    _text_panel(axes[0, 0], _experiment_summary_text(controls, diagnostics), "Experiment Request")
    _plot_energy_ledger(axes[0, 1], energy_ledger)
    _text_panel(axes[0, 2], _exposure_text(controls, exposure_summary), "Exposure Bookkeeping")

    _text_panel(axes[1, 0], _warning_flags_text(warning_flags), "Lab Realism / Feasibility")
    im = axes[1, 1].imshow(
        _display_map(stack.intensity_zyx[selected_idx], display_scaling, display_percentile_clip),
        origin="lower",
        extent=extent_xy,
        cmap="inferno",
        aspect="equal",
    )
    axes[1, 1].set_title(f"XY optical intensity @ z={z[selected_idx]:.3g} um")
    axes[1, 1].set_xlabel("x (um)")
    axes[1, 1].set_ylabel("y (um)")
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

    im = axes[1, 2].imshow(
        _display_map(F[selected_idx], display_scaling, display_percentile_clip),
        origin="lower",
        extent=extent_xy,
        cmap="viridis",
        aspect="equal",
    )
    axes[1, 2].set_title("XY optical fluence @ selected plane (J/cm^2)")
    axes[1, 2].set_xlabel("x (um)")
    axes[1, 2].set_ylabel("y (um)")
    fig.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04, label="J/cm^2")

    roi_half = float(controls.get("central_roi_half_width_um", 10.0))
    xi = np.flatnonzero(np.abs(x) <= roi_half)
    yi = np.flatnonzero(np.abs(y) <= roi_half)
    if xi.size == 0:
        xi = np.array([_nearest_index(x, 0.0)])
    if yi.size == 0:
        yi = np.array([_nearest_index(y, 0.0)])
    roi_f = F[selected_idx][np.ix_(yi, xi)]
    im = axes[2, 0].imshow(
        _display_map(roi_f, display_scaling, display_percentile_clip),
        origin="lower",
        extent=_extent_from_coords(x[xi], y[yi]),
        cmap="viridis",
        aspect="equal",
    )
    axes[2, 0].set_title("Central ROI fluence zoom")
    axes[2, 0].set_xlabel("x (um)")
    axes[2, 0].set_ylabel("y (um)")
    fig.colorbar(im, ax=axes[2, 0], fraction=0.046, pad=0.04, label="J/cm^2")

    yc = _nearest_index(y, 0.0)
    xz = F[:, yc, :].T
    im = axes[2, 1].imshow(
        _display_map(xz, display_scaling, display_percentile_clip),
        origin="lower",
        extent=(float(np.min(z)), float(np.max(z)), float(np.min(x)), float(np.max(x))),
        cmap="viridis",
        aspect="auto",
    )
    axes[2, 1].set_title("XZ optical fluence with lab markers")
    axes[2, 1].set_xlabel("z (um)")
    axes[2, 1].set_ylabel("x (um)")
    for value, color, label, style in [
        (0.0, "#1f77b4", "surface z=0", "-"),
        (float(controls.get("focus_depth_um", 0.0)), "#ff7f0e", "target depth", "--"),
        (float(diagnostics["selected_plane_z_um"]), "#2ca02c", "selected z", "-."),
        (float(diagnostics["global_peak_z_um"]), "#d62728", "global peak z", ":"),
    ]:
        axes[2, 1].axvline(value, color=color, lw=1.2, ls=style, label=label)
    axes[2, 1].legend(fontsize=7)
    fig.colorbar(im, ax=axes[2, 1], fraction=0.046, pad=0.04, label="J/cm^2")

    axes[2, 2].plot(z, fluence_result.peak_fluence_by_z_j_cm2, label="global peak by z")
    if "central_roi_peak_by_z_j_cm2" in diagnostics:
        axes[2, 2].plot(z, diagnostics["central_roi_peak_by_z_j_cm2"], ls="--", label="central ROI peak by z")
    axes[2, 2].axvline(float(diagnostics["selected_plane_z_um"]), color="#2ca02c", lw=1.2, label="selected")
    axes[2, 2].set_title("Peak fluence + raw captured-power drift")
    axes[2, 2].set_xlabel("z (um)")
    axes[2, 2].set_ylabel("peak fluence (J/cm^2)")
    axr = axes[2, 2].twinx()
    axr.plot(z, fluence_result.raw_captured_power_fraction_by_z, color="#9467bd", ls=":", label="raw captured fraction")
    axr.set_ylabel("raw captured fraction")
    axes[2, 2].legend(fontsize=7, loc="upper left")
    axr.legend(fontsize=7, loc="upper right")

    warning_text = _diagnostic_warning_text(diagnostics, show_caveats)
    _text_panel(axes[3, 0], warning_text, "Warnings / Interpretation Boundary")
    _text_panel(axes[3, 1], _lab_report_text(lab_report), "Stage Report Snapshot")
    _text_panel(axes[3, 2], _future_disabled_text(), "Future Physics Disabled")

    fig.stage8c1_metadata = {  # type: ignore[attr-defined]
        "stage": STAGE,
        "figure_status": FIGURE_STATUS,
        "model_status": MODEL_STATUS,
        "final_export_allowed": False,
        "selected_plane_z_um": diagnostics["selected_plane_z_um"],
        "warning_level": diagnostics.get("warning_level", "pass"),
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
                "stage": STAGE,
                "figure_status": FIGURE_STATUS,
                "model_status": MODEL_STATUS,
                "final_export_allowed": "False",
                "Description": "Stage 8C.1 optical/fluence diagnostic only; no material response.",
            },
        )
    return fig


def make_interpretation_text(
    *,
    controls: Mapping[str, Any] | None = None,
    energy_ledger: EnergyLedger | None = None,
    exposure_summary: Mapping[str, Any] | None = None,
    diagnostics: Mapping[str, Any] | None = None,
    lab_report: Any | None = None,
) -> str:
    """Return plain-language cockpit interpretation with the Stage 8C.1 boundary."""
    controls = dict(controls or {})
    exposure_summary = dict(exposure_summary or {})
    diagnostics = dict(diagnostics or {})

    input_energy = controls.get("pulse_energy_before_optics_uJ", "not available")
    sample_energy = "not available"
    avg_power = "not available"
    limit_state = "not available"
    if energy_ledger is not None:
        sample_energy = f"{energy_ledger.energy_at_sample_uJ:.4g}"
        avg_power = f"{energy_ledger.average_power_at_sample_W:.4g}"
        limit_state = "above" if any("exceeds limit" in str(w).lower() for w in energy_ledger.ledger_warnings) else "below"

    first_order = "valid" if float(controls.get("selected_first_order_fraction", 0.0)) > 0 else "not valid"
    route = "valid" if controls.get("generation_method", "holographic") in {"holographic", "physical"} else "not valid"
    pupil = "not available from current engine"
    pulse_spacing = exposure_summary.get("pulse_spacing_um", "not available")
    pulses_per_spot = exposure_summary.get("pulses_per_spot", "not available")
    selected_f = diagnostics.get("selected_plane_peak_value", diagnostics.get("global_peak_value", "not available"))
    central_f = diagnostics.get("central_roi_peak_value", "not available")
    target_f = diagnostics.get("target_depth_peak_value", "not available")
    peak_i = diagnostics.get("peak_intensity_w_cm2", "not available")
    edge = "is" if diagnostics.get("global_peak_near_boundary") else "is not"
    drift = diagnostics.get("captured_power_drift_fraction", "not available")
    if isinstance(drift, (float, int)) and np.isfinite(float(drift)):
        drift_text = f"{100.0 * float(drift):.2f}%"
    else:
        drift_text = str(drift)
    lab_state = _lab_status(lab_report)

    return "\n".join(
        [
            f"Energy at sample is {sample_energy} uJ from {input_energy} uJ before optics.",
            f"Average power is {limit_state} the configured limit; sample average power is {avg_power} W.",
            f"The optical route is {route}.",
            f"The first-order geometry is {first_order}.",
            f"Pupil fill/clipping status is {pupil}.",
            f"Pulse spacing is {_fmt(pulse_spacing)} um, giving {_fmt(pulses_per_spot)} pulses per spot.",
            f"For the selected plane, peak optical fluence is {_fmt(selected_f)} J/cm^2.",
            f"Central ROI peak is {_fmt(central_f)} J/cm^2.",
            f"Target-depth peak is {_fmt(target_f)} J/cm^2.",
            f"Peak intensity estimate is {_fmt(peak_i)} W/cm^2, assuming no nonlinear reshaping.",
            f"The global field peak {edge} near the crop boundary.",
            f"Captured-power drift is {drift_text}.",
            f"Lab realism status is {lab_state}.",
            "This output is optical fluence only; no material modification is predicted.",
        ]
    )


def _nearest_index(coords: np.ndarray, value: float) -> int:
    arr = np.asarray(coords, dtype=float)
    return int(np.argmin(np.abs(arr - float(value))))


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


def _display_map(arr: np.ndarray, scaling: str, percentile_clip: tuple[float, float]) -> np.ndarray:
    data = np.asarray(arr, dtype=float)
    if scaling == "log":
        floor = np.nanmax(data) * 1e-12 if np.nanmax(data) > 0 else 1e-12
        return np.log10(np.maximum(data, floor))
    if scaling == "percentile":
        lo, hi = np.nanpercentile(data, percentile_clip)
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            return np.clip(data, lo, hi)
    return data


def _plot_energy_ledger(ax: Any, ledger: EnergyLedger | None) -> None:
    if ledger is None:
        _text_panel(ax, "not available from current engine", "Energy Ledger")
        return
    labels = ["input"] + [row.component_name[:12] for row in ledger.rows]
    first_energy = ledger.rows[0].energy_in_uJ if ledger.rows else ledger.energy_at_sample_uJ
    values = [first_energy] + [row.energy_out_uJ for row in ledger.rows]
    ax.bar(np.arange(len(values)), values, color="#2374ab")
    ax.set_xticks(np.arange(len(labels)), labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("pulse energy (uJ)")
    ax.set_title("Energy Ledger")
    ax.axhline(ledger.energy_at_sample_uJ, color="#d62728", ls=":", lw=1, label="sample")
    ax.legend(fontsize=7)


def _text_panel(ax: Any, text: str, title: str) -> None:
    ax.set_title(title)
    ax.axis("off")
    ax.text(0.02, 0.98, text, va="top", ha="left", family="monospace", fontsize=8.5, wrap=True)


def _experiment_summary_text(controls: Mapping[str, Any], diagnostics: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            f"Laser: {controls.get('wavelength_nm', 'na')} nm, {controls.get('pulse_duration_fs', 'na')} fs, {controls.get('repetition_rate_Hz', 'na')} Hz",
            f"Pulse energy before optics: {controls.get('pulse_energy_before_optics_uJ', 'na')} uJ",
            f"Route: {controls.get('generation_method', 'holographic')} VBB, ell={controls.get('ell', 'na')}",
            f"Sample: {controls.get('material_name', 'na')}, n={controls.get('refractive_index', 'na')}",
            f"Focus target: {controls.get('focus_depth_um', 'na')} um",
            f"Selected plane: {diagnostics.get('selected_plane_z_um', 'na')} um",
            f"Reason: {diagnostics.get('selected_plane_reason', 'na')}",
            "Claim: optical fluence only",
        ]
    )


def _exposure_text(controls: Mapping[str, Any], exposure: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            f"writing_mode: {controls.get('writing_mode', 'na')}",
            f"scan axis: {controls.get('scan_axis', 'na')}",
            f"scan speed: {controls.get('scan_speed_mm_s', 'na')} mm/s",
            f"pulse spacing: {_fmt(exposure.get('pulse_spacing_um', 'na'))} um",
            f"pulses/spot: {_fmt(exposure.get('pulses_per_spot', 'na'))}",
            f"line time: {_fmt(exposure.get('line_duration_s', 'na'))} s",
            f"total pulses: {exposure.get('total_pulses_on_line', 'na')}",
            "Status: exposure bookkeeping only",
        ]
    )


def _warning_flags_text(flags: Mapping[str, Mapping[str, str]]) -> str:
    lines = []
    for name, info in flags.items():
        mark = {"pass": "OK", "caution": "WARN", "fail": "FAIL"}.get(info.get("status"), "INFO")
        lines.append(f"{mark:4s} {name}: {info.get('message', '')}")
    return "\n".join(lines)


def _diagnostic_warning_text(diagnostics: Mapping[str, Any], show_caveats: bool) -> str:
    lines = [
        f"Status: {diagnostics.get('warning_level', 'pass')}",
        f"global_peak_near_boundary: {diagnostics.get('global_peak_near_boundary')}",
        f"global_peak_distance_to_boundary_px: {diagnostics.get('global_peak_distance_to_boundary_px')}",
        f"selected_plane_reason: {diagnostics.get('selected_plane_reason')}",
        f"captured_power_drift: {_fmt(diagnostics.get('captured_power_drift_fraction'))}",
    ]
    for msg in diagnostics.get("warning_messages", []) or []:
        lines.append(f"warning: {msg}")
    if show_caveats:
        lines.extend(
            [
                "",
                "Plain meaning:",
                "Optical fluence is shown for inspection.",
                "It is not absorbed energy, material change, ablation,",
                "a written feature, or a calibrated prediction.",
            ]
        )
    return "\n".join(lines)


def _lab_report_text(lab_report: Any | None) -> str:
    if lab_report is None:
        return "not available from current engine"
    rows = lab_report.to_rows() if hasattr(lab_report, "to_rows") else []
    lines = []
    for row in rows[:10]:
        lines.append(f"{row['status_level'][:12]:12s} {row['stage_name']}")
    if len(rows) > 10:
        lines.append(f"... +{len(rows) - 10} stages")
    return "\n".join(lines)


def _future_disabled_text() -> str:
    return "\n".join(
        [
            "Material response: disabled until calibration",
            "Dose accumulation: Stage 8E",
            "Threshold maps: future proxy/calibration stage",
            "Microscope proxy: later",
            "Waveguide prediction: later, requires dn calibration",
            "Surface ablation: later, separate model",
            "Nonlinear propagation: later, requires material constants",
            "Thermal accumulation: later, requires thermal model/constants",
        ]
    )


def _lab_status(lab_report: Any | None) -> str:
    if lab_report is None or not hasattr(lab_report, "stages"):
        return "not available"
    levels = [stage.status_level for stage in lab_report.stages]
    for level in ("fail", "caution", "missing", "diagnostic_only", "disabled_future", "pass"):
        if level in levels:
            return level
    return "not available"


def _fmt(value: Any) -> str:
    if isinstance(value, (float, int)):
        if np.isfinite(float(value)):
            return f"{float(value):.4g}"
    return str(value)

