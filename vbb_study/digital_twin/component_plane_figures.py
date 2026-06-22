"""Stage 8C.3R.1 free-space reference-plane validation figures.

Three diagnostic figures (optical/fluence only, n = 1.0, no material model):

  * ``plot_reference_plane_energy_axis_validation`` - energy waterfall, raw
    power/transmission, commanded-vs-actual axis, XZ centre trajectories,
    expected-vs-measured tilt slope, FOV/crop reliability, claim boundary.
  * ``plot_individual_sensitivity_atlas`` - one upstream error family per row
    (XY, XZ, difference, metrics + translation/deformation/clipping/contamination
    classification + numerical-reliability state).
  * ``plot_fov_convergence_check`` - baseline + perturbations, standard vs
    expanded grid/FOV, with reliability labels.

``final_export_allowed=False`` everywhere.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import matplotlib
matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt  # noqa: E402

from vbb_study.digital_twin.component_plane_pipeline import (
    ComponentPlaneConfig,
    run_component_plane_pipeline,
)
from vbb_study.digital_twin.component_plane_metrics import (
    run_component_plane_scenario,
    classify_translation_vs_deformation,
    compute_axis_tracking,
    stack_to_fluence,
    _nearest_index,
)
from vbb_study.digital_twin.component_plane_validation import (
    compute_energy_audit,
    validate_beam_tilt,
    fov_convergence_check,
    zero_control_equivalence,
)

FIGURE_STATUS = "diagnostic_allowed"
MODEL_STATUS = "optical_prediction"
_DIAG_NOTE = "Diagnostic sensitivity sweep. Not an experimentally measured laboratory tolerance."

# Individual upstream-error families (NOT the combined stress test).
DEFAULT_ATLAS_SCENARIOS = (
    "vortex_misregistration",
    "axicon_misregistration",
    "input_decentre_slm_area",
    "beam_tilt_pupil",
    "pupil_decentre_clip",
    "zernike_defocus",
    "zernike_astigmatism",
    "zernike_coma",
    "zernike_spherical",
    "zero_order_leakage",
)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _badges(fig: Any, y: float = 0.93) -> None:
    for bx, txt, ec, fc in [
        (0.045, "DIAGNOSTIC ONLY", "#0d47a1", "#e3f2fd"),
        (0.165, "NO MATERIAL RESPONSE", "#4a148c", "#f3e5f5"),
        (0.330, "FREE-SPACE n=1.0", "#1b5e20", "#e8f5e9"),
        (0.455, "COMPONENT-PLANE PHYSICS", "#bf360c", "#fff3e0"),
    ]:
        fig.text(bx, y, txt, ha="left", va="center", fontsize=9, fontweight="bold",
                 color=ec, bbox=dict(boxstyle="round,pad=0.3", facecolor=fc, edgecolor=ec, lw=1.1))


def _edge_ticks(ax: Any, ext: tuple, ring_x: float, ring_y: float) -> None:
    xmin, xmax, ymin, ymax = ext
    ty, tx = 0.05 * (ymax - ymin), 0.05 * (xmax - xmin)
    ax.plot([0, 0], [ymin, ymin + ty], color="white", lw=2.0, zorder=6, clip_on=False)
    ax.plot([ring_x, ring_x], [ymin, ymin + ty], color="cyan", lw=2.0, zorder=7, clip_on=False)
    ax.plot([xmin, xmin + tx], [0, 0], color="white", lw=2.0, zorder=6, clip_on=False)
    ax.plot([xmin, xmin + tx], [ring_y, ring_y], color="cyan", lw=2.0, zorder=7, clip_on=False)


def _card(ax: Any, title: str, lines: list[str], face: str, edge: str, fs: float = 8.0) -> None:
    ax.set_axis_off()
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                               facecolor=face, edgecolor=edge, lw=1.3, clip_on=False))
    ax.text(0.04, 0.95, title, transform=ax.transAxes, fontsize=fs + 1.5,
            fontweight="bold", va="top", ha="left", color=edge)
    ax.text(0.04, 0.83, "\n".join(lines), transform=ax.transAxes, fontsize=fs,
            va="top", ha="left", family="monospace", color="#1a1a1a")


def _single_run_reliability(ax_metrics: Mapping[str, Any]) -> str:
    oof = float(ax_metrics["out_of_frame_fraction"])
    margin = float(ax_metrics["field_of_view_margin_um"])
    if oof > 0.02 or margin < 0.0:
        return "invalid_out_of_frame"
    if oof > 0.005:
        return "caution_crop_limited"
    return "numerically_reliable"


def _ext(x: np.ndarray, y: np.ndarray) -> tuple:
    return float(x.min()), float(x.max()), float(y.min()), float(y.max())


def _save(fig: Any, output_path: str | Path | None, dpi: int, title: str, stage: str) -> None:
    if output_path is None:
        return
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", metadata={
        "Title": title, "stage": stage, "final_export_allowed": "False",
        "Description": "Stage 8C.3R.1 free-space reference-plane diagnostic; n=1.0; no material response.",
    })


# ---------------------------------------------------------------------------
# Figure 1 - energy + axis validation
# ---------------------------------------------------------------------------


def plot_reference_plane_energy_axis_validation(
    scenario: str = "pupil_decentre_clip",
    *,
    config: ComponentPlaneConfig | None = None,
    tilt_mrad: float = 24.0,
    output_path: str | Path | None = None,
    dpi: int = 165,
) -> "matplotlib.figure.Figure":
    """Figure 1: free-space energy + axis validation."""
    config = config or ComponentPlaneConfig()
    res = run_component_plane_scenario(scenario, config=config)
    sel = int(res.selected_plane_index)
    base_au = compute_energy_audit(res.baseline, fov_reliable=True)
    mild_au = compute_energy_audit(res.mild, baseline_run=res.baseline, fov_reliable=True)
    sev_au = compute_energy_audit(res.severe, baseline_run=res.baseline, fov_reliable=True)
    tilt = validate_beam_tilt(tilt_mrad, config=config)
    fov = fov_convergence_check({}, config=config)
    equiv = zero_control_equivalence(config)

    sev_st = res.severe.propagated_stack
    x = np.asarray(sev_st.x_um, float); z = np.asarray(sev_st.z_um, float)
    Fs = np.asarray(stack_to_fluence(sev_st).fluence_zyx_j_cm2, float)
    ext = _ext(x, x)

    fig = plt.figure(figsize=(17.5, 13.0), facecolor="white")
    gs = fig.add_gridspec(3, 3, left=0.07, right=0.965, top=0.86, bottom=0.06,
                          hspace=0.42, wspace=0.30, height_ratios=[1.0, 1.05, 0.95])
    fig.suptitle("Stage 8C.3R.1 Free-Space Reference-Plane Energy & Axis Validation\n"
                 f"{res.scenario.title}  (n=1.0 free-space reference; no material model)",
                 x=0.045, y=0.975, ha="left", va="top", fontsize=16, fontweight="bold")
    _badges(fig)

    # (0,0:2) energy waterfall
    axw = fig.add_subplot(gs[0, :2])
    planes = ["input", "after input", "after SLM", "after pupil = ref plane"]
    def chain(au):
        led = au["per_plane_ledger"]
        return [au["input_pulse_energy_uJ"]] + [r["energy_after_uJ"] for r in led]
    for au, lab, col in [(base_au, "baseline", "#37474f"), (mild_au, "mild", "#f57f17"), (sev_au, "severe", "#b71c1c")]:
        axw.plot(range(len(planes)), chain(au)[:len(planes)], "-o", color=col, label=lab, lw=2)
    axw.set_xticks(range(len(planes))); axw.set_xticklabels(planes, fontsize=8.5)
    axw.set_ylabel("pulse energy (uJ)"); axw.set_title("Energy waterfall through component planes", fontsize=11, fontweight="bold")
    axw.grid(alpha=0.3); axw.legend(fontsize=9)

    # (0,2) transmission + encircled bars
    axb = fig.add_subplot(gs[0, 2])
    labels = ["baseline", "mild", "severe"]
    transm = [base_au["transmitted_fraction"], mild_au["transmitted_fraction"], sev_au["transmitted_fraction"]]
    enc = [base_au["encircled_energy_fraction_near_peak"], mild_au["encircled_energy_fraction_near_peak"], sev_au["encircled_energy_fraction_near_peak"]]
    xb = np.arange(3)
    axb.bar(xb - 0.2, transm, 0.4, label="transmitted frac", color="#1565c0")
    axb.bar(xb + 0.2, enc, 0.4, label="encircled@peak", color="#ef6c00")
    axb.set_xticks(xb); axb.set_xticklabels(labels, fontsize=8.5); axb.set_ylim(0, 1.05)
    axb.set_title("Transmission & encircled energy", fontsize=10.5, fontweight="bold"); axb.legend(fontsize=8)

    # (1,0) severe XY fluence + axis edge ticks
    axxy = fig.add_subplot(gs[1, 0])
    im = axxy.imshow(Fs[sel], origin="lower", extent=ext, cmap="viridis", aspect="equal")
    _edge_ticks(axxy, ext, res.severe_axis["ring_centre_x_um"], res.severe_axis["ring_centre_y_um"])
    axxy.set_title("Severe reference-plane XY fluence\n(white=commanded, cyan=measured ticks)", fontsize=10, fontweight="bold")
    axxy.set_xlabel("x (um)"); axxy.set_ylabel("y (um)")
    fig.colorbar(im, ax=axxy, fraction=0.046, pad=0.02, label="J/cm^2")

    # (1,1) XZ centre trajectories
    axt = fig.add_subplot(gs[1, 1])
    for axm, lab, col in [(res.baseline_axis, "baseline", "#37474f"), (res.mild_axis, "mild", "#f57f17"), (res.severe_axis, "severe", "#b71c1c")]:
        axt.plot(z, np.asarray(axm["centre_trajectory_x_um"], float), "-", color=col, lw=1.8, label=lab)
    axt.axhline(0.0, color="0.6", ls=":", lw=1.0)
    axt.set_xlabel("z (um)"); axt.set_ylabel("beam-centre x (um)")
    axt.set_title("Commanded (x=0) vs actual centre trajectory", fontsize=10.5, fontweight="bold")
    axt.grid(alpha=0.3); axt.legend(fontsize=8)

    # (1,2) tilt expected vs measured
    axti = fig.add_subplot(gs[1, 2])
    tilt_run = run_component_plane_pipeline({"enable_beam_tilt": True, "beam_tilt_x_mrad": tilt_mrad}, config=config)
    tax = compute_axis_tracking(tilt_run.propagated_stack)
    zt = np.asarray(tilt_run.propagated_stack.z_um, float)
    meas = np.asarray(tax["centre_trajectory_x_um"], float)
    z0, z1 = tilt["valid_z_fit_range_um"]
    axti.plot(zt, meas, "o", ms=3, color="#00838f", label="measured centre")
    zz = np.linspace(z0, z1, 20)
    axti.plot(zz, meas[_nearest_index(zt, z0)] + tilt["expected_slope_x"] * (zz - z0),
              "--", color="#c62828", lw=2, label="analytical kx/kz")
    axti.set_xlabel("z (um)"); axti.set_ylabel("centre x (um)")
    axti.set_title(f"Tilt {tilt_mrad:g} mrad: expected vs measured\nrel err {tilt['relative_slope_error']*100:.1f}%  ({tilt['grid_pixels_of_displacement']:.1f} px)",
                   fontsize=10, fontweight="bold")
    axti.grid(alpha=0.3); axti.legend(fontsize=8)

    # (2,0) energy audit card
    _card(fig.add_subplot(gs[2, 0]), "Energy / normalisation audit", [
        f"input energy        : {base_au['input_pulse_energy_uJ']:.2f} uJ",
        "                      base / mild / severe",
        f"transmitted frac    : {base_au['transmitted_fraction']:.3f}/{mild_au['transmitted_fraction']:.3f}/{sev_au['transmitted_fraction']:.3f}",
        f"ref-plane energy uJ : {base_au['sample_pulse_energy_uJ']:.2f}/{mild_au['sample_pulse_energy_uJ']:.2f}/{sev_au['sample_pulse_energy_uJ']:.2f}",
        f"peak fluence J/cm2  : {base_au['peak_fluence_J_cm2']:.1f}/{mild_au['peak_fluence_J_cm2']:.1f}/{sev_au['peak_fluence_J_cm2']:.1f}",
        f"peak/ref-energy     : {base_au['peak_to_reference_energy_ratio']:.3f}/{mild_au['peak_to_reference_energy_ratio']:.3f}/{sev_au['peak_to_reference_energy_ratio']:.3f}",
        f"renorm factor       : {sev_au['renormalisation_factor']:.1f}  (no pre-clip renorm)",
        f"energy_accounting_valid : {sev_au['energy_accounting_valid']}",
        f"severe peak rise    : {sev_au['peak_rise_status']}",
    ], "#eceff1", "#37474f")

    # (2,1) axis card
    sa = res.severe_axis
    _card(fig.add_subplot(gs[2, 1]), "Commanded vs actual beam axis (severe)", [
        "commanded axis      : (0.000, 0.000) um",
        f"ring centre         : ({sa['ring_centre_x_um']:.3f}, {sa['ring_centre_y_um']:.3f}) um",
        f"intensity centroid  : ({sa['intensity_centroid_x_um']:.3f}, {sa['intensity_centroid_y_um']:.3f}) um",
        f"peak point          : ({sa['peak_x_um']:.3f}, {sa['peak_y_um']:.3f}) um",
        f"ring fit quality    : {sa['ring_fit_quality']:.3f}",
        f"radial axis error   : {sa['radial_axis_error_um']:.3f} um",
        f"steering x/y        : {sa['beam_steering_angle_x_mrad']:.2f}/{sa['beam_steering_angle_y_mrad']:.2f} mrad",
        f"trajectory fit R2   : {sa['trajectory_fit_quality']:.3f}",
        f"FOV margin          : {sa['field_of_view_margin_um']:.2f} um",
    ], "#e3f2fd", "#0d47a1")

    # (2,2) tilt + FOV + equivalence + claim
    _card(fig.add_subplot(gs[2, 2]), "Validation & claim boundary", [
        f"zero-control equivalent : {equiv['equivalent_within_tolerance']}",
        f"  cfield/I/F sim ~       {equiv['complex_field_similarity']:.6f}",
        f"  ring meas/anal um      {equiv['ring_radius_measured_um']:.2f}/{equiv['ring_radius_analytical_um']:.2f}",
        f"tilt {tilt_mrad:g} mrad rel err   {tilt['relative_slope_error']*100:.1f}% ({tilt['agrees_within_tolerance']})",
        f"  px displacement        {tilt['grid_pixels_of_displacement']:.1f}",
        f"FOV reliability         {fov['metric_convergence_status']}",
        f"  peak conv rel diff     {fov['peak_fluence_rel_diff']*100:.2f}%",
        "",
        "claim: optical fluence diagnostic only;",
        "free-space n=1.0; no material model;",
        "final_export_allowed=False.",
    ], "#e8f5e9", "#1b5e20")

    fig.cpr1_metadata = {  # type: ignore[attr-defined]
        "stage": "stage8c3r1_reference_plane_validation", "scenario": scenario,
        "equivalent": bool(equiv["equivalent_within_tolerance"]),
        "tilt_rel_err": float(tilt["relative_slope_error"]),
        "fov_status": fov["metric_convergence_status"],
        "severe_peak_rise_status": sev_au["peak_rise_status"],
        "final_export_allowed": False,
    }
    _save(fig, output_path, dpi, "Stage 8C.3R.1 Reference-Plane Energy & Axis Validation",
          "stage8c3r1_reference_plane_validation")
    return fig


# ---------------------------------------------------------------------------
# Figure 2 - individual sensitivity atlas
# ---------------------------------------------------------------------------


def plot_individual_sensitivity_atlas(
    scenarios: Sequence[str] = DEFAULT_ATLAS_SCENARIOS,
    *,
    config: ComponentPlaneConfig | None = None,
    output_path: str | Path | None = None,
    dpi: int = 150,
) -> "matplotlib.figure.Figure":
    """Figure 2: one upstream-error family per row (baseline/severe/diff + metrics)."""
    config = config or ComponentPlaneConfig()
    results = [run_component_plane_scenario(k, config=config) for k in scenarios]
    nrow = len(results)

    fig = plt.figure(figsize=(15.0, 2.55 * nrow + 1.4), facecolor="white")
    gs = fig.add_gridspec(nrow, 4, left=0.10, right=0.975,
                          top=1.0 - 0.9 / (2.55 * nrow + 1.4), bottom=0.04,
                          hspace=0.55, wspace=0.22, width_ratios=[1, 1, 1, 1.25])
    fig.suptitle("Stage 8C.3R.1 Individual Diagnostic Sensitivity Atlas  "
                 "(free-space n=1.0; no material model)\n" + _DIAG_NOTE,
                 x=0.04, y=0.995, ha="left", va="top", fontsize=14, fontweight="bold")

    for r, res in enumerate(results):
        sel = int(res.selected_plane_index)
        b_st, s_st = res.baseline.propagated_stack, res.severe.propagated_stack
        x = np.asarray(b_st.x_um, float); z = np.asarray(b_st.z_um, float)
        yc = _nearest_index(x, 0.0)
        Fb = np.asarray(stack_to_fluence(b_st).fluence_zyx_j_cm2, float)
        Fs = np.asarray(stack_to_fluence(s_st).fluence_zyx_j_cm2, float)
        ext = _ext(x, x); ext_xz = (float(z.min()), float(z.max()), float(x.min()), float(x.max()))
        vmax = max(float(Fb[sel].max()), float(Fs[sel].max()))
        diff = Fs[sel] - Fb[sel]
        dabs = max(float(np.abs(diff).max()), 1e-12)

        ax0 = fig.add_subplot(gs[r, 0])
        ax0.imshow(Fb[sel], origin="lower", extent=ext, cmap="viridis", vmin=0, vmax=vmax, aspect="equal")
        ax0.set_ylabel(res.scenario.key, fontsize=8.5, fontweight="bold")
        if r == 0: ax0.set_title("baseline XY", fontsize=10, fontweight="bold")
        ax1 = fig.add_subplot(gs[r, 1])
        im1 = ax1.imshow(Fs[sel], origin="lower", extent=ext, cmap="viridis", vmin=0, vmax=vmax, aspect="equal")
        _edge_ticks(ax1, ext, res.severe_axis["ring_centre_x_um"], res.severe_axis["ring_centre_y_um"])
        if r == 0: ax1.set_title("severe XY + axis ticks", fontsize=10, fontweight="bold")
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.02)
        ax2 = fig.add_subplot(gs[r, 2])
        ax2.imshow(Fs[:, yc, :].T, origin="lower", extent=ext_xz, cmap="viridis", aspect="auto")
        if r == 0: ax2.set_title("severe XZ", fontsize=10, fontweight="bold")
        ax2.set_xlabel("z (um)" if r == nrow - 1 else "")

        cls = res.severe_class; en_b = res.baseline_energy; en_s = res.severe_energy
        verdict = []
        if cls["translation_dominated_boolean"]: verdict.append("translation")
        if cls["residual_shape_deformation_score"] > 0.15 and not cls["translation_dominated_boolean"]: verdict.append("deformation")
        if cls["throughput_loss_fraction"] > 0.05: verdict.append("clipping")
        if cls["core_contamination_fraction"] > 0.02: verdict.append("core-contam")
        rel = _single_run_reliability(res.severe_axis)
        axc = fig.add_subplot(gs[r, 3])
        _card(axc, res.scenario.title, [
            f"severe: {res.scenario.severe_label}",
            f"transmit {en_s['transmitted_fraction']:.3f}  peakF {en_s['peak_fluence_J_cm2']:.0f}",
            f"reg sim {cls['registered_similarity_score']:.2f}  resid {cls['residual_shape_deformation_score']:.2f}",
            f"cshift {cls['centroid_shift_um']:.2f}um  thru_loss {cls['throughput_loss_fraction']:.2f}",
            f"core-contam {cls['core_contamination_fraction']:.3f}",
            f"class: {', '.join(verdict) if verdict else 'near-baseline'}",
            f"reliability: {rel}",
        ], "#fafafa", "#455a64", fs=7.4)

    fig.cpr1_metadata = {"stage": "stage8c3r1_individual_sensitivity_atlas",  # type: ignore[attr-defined]
                         "scenarios": list(scenarios), "final_export_allowed": False}
    _save(fig, output_path, dpi, "Stage 8C.3R.1 Individual Sensitivity Atlas",
          "stage8c3r1_individual_sensitivity_atlas")
    return fig


# ---------------------------------------------------------------------------
# Figure 3 - FOV convergence
# ---------------------------------------------------------------------------


def plot_fov_convergence_check(
    cases: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    config: ComponentPlaneConfig | None = None,
    output_path: str | Path | None = None,
    dpi: int = 160,
) -> "matplotlib.figure.Figure":
    """Figure 3: standard vs expanded grid/FOV reliability for several cases."""
    config = config or ComponentPlaneConfig()
    if cases is None:
        cases = {
            "baseline": {},
            "beam tilt 24 mrad": {"enable_beam_tilt": True, "beam_tilt_x_mrad": 24.0},
            "pupil clip+decentre": {"enable_pupil_clipping": True, "pupil_radius_um": 12.0, "pupil_decentre_x_um": 12.0},
            "decentre+SLM area": {"enable_beam_decentre": True, "beam_decentre_x_um": 16.0,
                                  "enable_slm_active_area": True, "slm_active_width_um": 30.0, "slm_active_height_um": 30.0},
        }
    checks = {name: fov_convergence_check(ctrl, config=config) for name, ctrl in cases.items()}
    # Deliberately undersized FOV to demonstrate the reliability labels trigger.
    from dataclasses import replace as _replace
    undersized = _replace(config, grid_N=max(64, int(config.grid_N * 0.35)))
    checks["undersized FOV (demo)"] = fov_convergence_check({}, config=undersized)
    names = list(checks)

    fig = plt.figure(figsize=(15.0, 8.6), facecolor="white")
    gs = fig.add_gridspec(2, 2, left=0.07, right=0.97, top=0.82, bottom=0.30,
                          hspace=0.45, wspace=0.25)
    fig.suptitle("Stage 8C.3R.1 Crop / FOV Convergence Check  (standard vs expanded grid; free-space n=1.0)",
                 x=0.045, y=0.965, ha="left", va="top", fontsize=15, fontweight="bold")
    _badges(fig, y=0.90)

    xb = np.arange(len(names))
    axp = fig.add_subplot(gs[0, 0])
    axp.bar(xb - 0.2, [checks[n]["standard"]["peak_fluence"] for n in names], 0.4, label="standard", color="#1565c0")
    axp.bar(xb + 0.2, [checks[n]["expanded"]["peak_fluence"] for n in names], 0.4, label="expanded", color="#ef6c00")
    axp.set_xticks(xb); axp.set_xticklabels(names, fontsize=7.5, rotation=15)
    axp.set_ylabel("peak fluence J/cm^2"); axp.set_title("Peak fluence: standard vs expanded FOV", fontsize=10.5, fontweight="bold")
    axp.legend(fontsize=8)

    axr = fig.add_subplot(gs[0, 1])
    axr.bar(xb - 0.2, [checks[n]["peak_fluence_rel_diff"] * 100 for n in names], 0.4, label="peak rel diff %", color="#6a1b9a")
    axr.bar(xb + 0.2, [checks[n]["out_of_frame_fraction"] * 100 for n in names], 0.4, label="out-of-frame %", color="#c62828")
    axr.axhline(5.0, color="0.4", ls="--", lw=1, label="5% convergence")
    axr.set_xticks(xb); axr.set_xticklabels(names, fontsize=7.5, rotation=15)
    axr.set_ylabel("%"); axr.set_title("Convergence & FOV spill", fontsize=10.5, fontweight="bold"); axr.legend(fontsize=8)

    axm = fig.add_subplot(gs[1, :])
    axm.imshow(np.asarray([[checks[n]["peak_position_diff_um"] for n in names],
                           [checks[n]["captured_power_drift_diff"] for n in names]]), aspect="auto", cmap="Blues")
    axm.set_yticks([0, 1]); axm.set_yticklabels(["peak pos diff (um)", "drift diff"], fontsize=8)
    axm.set_xticks(xb); axm.set_xticklabels(names, fontsize=8)
    for j, n in enumerate(names):
        axm.text(j, 0, f"{checks[n]['peak_position_diff_um']:.2f}", ha="center", va="center", fontsize=8)
        axm.text(j, 1, f"{checks[n]['captured_power_drift_diff']:.3f}", ha="center", va="center", fontsize=8)
    axm.set_title("Standard-vs-expanded differences", fontsize=10.5, fontweight="bold")

    rows = []
    for n in names:
        c = checks[n]
        rows.append(f"{n:22s} N {c['standard_grid_N']}->{c['expanded_grid_N']}  "
                    f"peakRel {c['peak_fluence_rel_diff']*100:5.2f}%  oof {c['out_of_frame_fraction']*100:5.2f}%  "
                    f"margin {c['field_of_view_margin_um']:6.1f}um  -> {c['metric_convergence_status']}")
    fig.text(0.07, 0.245, "Reliability labels (numerically_reliable / caution_crop_limited / invalid_out_of_frame):",
             fontsize=9.5, fontweight="bold", va="top")
    fig.text(0.07, 0.215, "\n".join(rows), fontsize=8.4, va="top", family="monospace")
    fig.text(0.07, 0.045, "Diagnostic only; free-space n=1.0; no material model; final_export_allowed=False.",
             fontsize=9, va="top", color="#1b5e20", fontweight="bold")

    fig.cpr1_metadata = {"stage": "stage8c3r1_fov_convergence",  # type: ignore[attr-defined]
                         "reliability": {n: checks[n]["metric_convergence_status"] for n in names},
                         "final_export_allowed": False}
    _save(fig, output_path, dpi, "Stage 8C.3R.1 FOV Convergence Check", "stage8c3r1_fov_convergence")
    return fig
