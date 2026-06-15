"""Stage-C train visualisation and figure rebuild helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap

import bessel_twin_core as bt
from . import vbb_axicon, vbb_polarized_train, vbb_regime, vbb_studies, vbb_style

Method = Literal["holographic", "physical"]


@dataclass
class TrainFrame:
    label: str
    field: np.ndarray
    grid: dict[str, Any]
    metadata: dict[str, Any]


def _normalised_amplitude(field: np.ndarray) -> np.ndarray:
    amp = np.abs(np.asarray(field, dtype=complex))
    peak = float(np.nanmax(amp)) if amp.size else 0.0
    return amp / peak if peak > 0.0 else amp


def _phase(field: np.ndarray) -> np.ndarray:
    return np.angle(np.asarray(field, dtype=complex))


def _extent_um(grid: dict[str, Any]) -> list[float]:
    x = np.asarray(grid["x"], dtype=float) / bt.um
    y = np.asarray(grid.get("y", grid["x"]), dtype=float) / bt.um
    return [float(x[0]), float(x[-1]), float(y[0]), float(y[-1])]


def _crop_center(arr: np.ndarray, grid: dict[str, Any], max_pixels: int = 192) -> tuple[np.ndarray, dict[str, Any]]:
    data = np.asarray(arr)
    n0, n1 = data.shape[-2:]
    size = int(min(max_pixels, n0, n1))
    y0 = max(0, (n0 - size) // 2)
    x0 = max(0, (n1 - size) // 2)
    cropped = data[y0 : y0 + size, x0 : x0 + size]
    x = np.asarray(grid["x"], dtype=float)[x0 : x0 + size]
    y_source = np.asarray(grid.get("y", grid["x"]), dtype=float)
    y = y_source[y0 : y0 + size]
    X, Y = np.meshgrid(x, y, indexing="xy")
    return cropped, {"x": x, "y": y, "X": X, "Y": Y, "R": np.hypot(X, Y), "PHI": np.arctan2(Y, X), "dx": grid["dx"], "N": size}


def _air_config_and_design(config: bt.TwinConfig) -> tuple[bt.TwinConfig, bt.BeamDesign]:
    air = vbb_studies.beam_air_config(config)
    design = bt.compute_design_from_targets(air.laser, air.target, air.material)
    return air, design


def holographic_train_frames(config: bt.TwinConfig, *, lab: bool = True) -> tuple[list[TrainFrame], dict[str, Any]]:
    air, design = _air_config_and_design(replace(config, generation_method="holographic"))
    slm = bt.build_realistic_slm_field(
        air,
        design,
        include_quantization=lab,
        include_fill_factor=lab,
        include_active_aperture=lab,
        include_blaze=air.include_blaze,
    )
    grid = slm["grid"]
    amp0 = bt.gaussian_amplitude(grid["R"], air.laser.beam_radius_on_slm_m)
    frames = [TrainFrame("input Gaussian on SLM", amp0.astype(complex), grid, {"plane": "SLM"})]
    frames.append(TrainFrame("after SLM1 hologram", slm["U"], grid, {"plane": "SLM", "lab": lab}))
    if air.include_first_order_isolation and air.include_blaze:
        filter_geometry = bt.first_order_filter_geometry(grid, air.slm, design)
        order = bt.isolate_first_order(
            slm["U"],
            grid,
            air.slm,
            filter_radius_lpmm=filter_geometry["effective_filter_radius_lpmm"],
        )
        order.update(filter_geometry)
        U = order["U_selected"]
    else:
        order = {"selected_fraction": 1.0}
        U = slm["U"]
    frames.append(TrainFrame("after first-order isolation", U, grid, {"selected_fraction": float(order["selected_fraction"])}))
    pupil = (grid["R"] <= air.objective.pupil_radius_m).astype(float)
    U_pupil = U * pupil
    frames.append(TrainFrame("objective pupil", U_pupil, grid, {"plane": "pupil"}))
    U_focus, focal_grid = bt.focus_to_focal_plane(U_pupil, grid, air.laser, air.objective)
    frames.append(TrainFrame("focused surface seed", U_focus, focal_grid, {"plane": "surface_in_air"}))
    mask = {
        "slm1_phase": np.asarray(slm["phase"], dtype=float),
        "slm1_grid": grid,
        "slm2_phase": np.zeros_like(np.asarray(slm["phase"], dtype=float)),
        "slm2_grid": grid,
        "slm2_residual_before_rad": None,
        "slm2_residual_after_rad": None,
    }
    return frames, mask


def physical_train_frames(config: bt.TwinConfig, *, lab: bool = True) -> tuple[list[TrainFrame], dict[str, Any]]:
    physical_cfg = config.physical_axicon
    if not lab:
        physical_cfg = replace(physical_cfg, slm2_stroke_levels=None, slm2_conjugate_mode="full")
    air, design = _air_config_and_design(replace(config, generation_method="physical", physical_axicon=physical_cfg))
    grid = bt.make_xy_grid(int(air.grid.ideal_N), float(air.grid.ideal_dx_m))
    amp = np.exp(-(grid["R"] ** 2) / max(float(design.w0_sample_m), bt.EPS) ** 2)
    frames = [TrainFrame("input Gaussian on SLM1", amp.astype(complex), grid, {"plane": "SLM1"})]
    charge = int(design.ell if physical_cfg.slm1_vortex_charge is None else physical_cfg.slm1_vortex_charge)
    slm1_phase = charge * grid["PHI"]
    U1 = amp * np.exp(1j * slm1_phase)
    frames.append(
        TrainFrame(
            "SLM1 vortex only",
            U1,
            grid,
            {"charge": charge, "plane": "SLM1", "phase_components": ("vortex",), "contains_axicon": False},
        )
    )
    z = float(physical_cfg.inter_slm_z_m)
    if abs(z) <= bt.EPS:
        U2 = U1.copy()
    else:
        prop = bt.make_bl_asm_propagator(U1, grid, air.laser.wavelength_m, n_medium=float(physical_cfg.inter_slm_n), bandlimit=True)
        U2 = prop(z)
    frames.append(TrainFrame("before SLM2", U2, grid, {"inter_slm_z_m": z}))
    U2_flat, diag = vbb_axicon.slm2_conjugate(
        U2,
        mode=physical_cfg.slm2_conjugate_mode,
        stroke_levels=physical_cfg.slm2_stroke_levels,
        return_diagnostics=True,
    )
    slm2_phase = np.angle(U2_flat / (U2 + bt.EPS))
    frames.append(
        TrainFrame(
            "SLM2 correction mask",
            np.exp(1j * slm2_phase),
            grid,
            {
                **dict(diag),
                "plane": "SLM2",
                "phase_components": ("conjugate_flattening", "optional_amplitude_correction"),
                "contains_axicon": False,
            },
        )
    )
    frames.append(
        TrainFrame(
            "after SLM2 flattening",
            U2_flat,
            grid,
            {
                **dict(diag),
                "plane": "SLM2",
                "phase_components": ("conjugate_flattening", "optional_amplitude_correction"),
                "contains_axicon": False,
            },
        )
    )

    n_axicon = float(design.n_axicon if physical_cfg.n_axicon is None else physical_cfg.n_axicon)
    axicon_medium_n = float(physical_cfg.axicon_medium_n)
    k_r = float(design.kr_sample_m_inv)
    if physical_cfg.axicon_base_angle_deg is None:
        denom = air.laser.k0 * (n_axicon - axicon_medium_n)
        gamma_rad = np.arctan(k_r / max(abs(denom), bt.EPS))
        if denom < 0.0:
            gamma_rad = -gamma_rad
        gamma_deg = float(np.rad2deg(gamma_rad))
    else:
        gamma_deg = float(physical_cfg.axicon_base_angle_deg)
        k_r = float(air.laser.k0 * (n_axicon - axicon_medium_n) * np.tan(np.deg2rad(gamma_deg)))
    axicon_phase = -abs(k_r) * np.asarray(grid["R"], dtype=float)
    axicon_field = np.exp(1j * axicon_phase)
    frames.append(
        TrainFrame(
            "physical axicon phase (continuous)",
            axicon_field,
            grid,
            {"plane": "physical_axicon", "pixelated": False, "k_r_m_inv": float(abs(k_r)), "base_angle_deg": gamma_deg},
        )
    )
    tp, ts = vbb_polarized_train.fresnel_sp_coefficients(n_axicon, axicon_medium_n, gamma_deg)
    fresnel_amp = np.sqrt(max(0.5 * (abs(tp) ** 2 + abs(ts) ** 2), 0.0))
    if physical_cfg.axicon_aperture_radius_m is None:
        aperture = np.ones_like(np.asarray(grid["R"], dtype=float))
    else:
        aperture = (np.asarray(grid["R"], dtype=float) <= float(physical_cfg.axicon_aperture_radius_m)).astype(float)
    U3 = np.sqrt(max(float(physical_cfg.axicon_transmission), 0.0)) * fresnel_amp * aperture * axicon_field * U2_flat
    frames.append(
        TrainFrame(
            "after physical axicon",
            U3,
            grid,
            {
                "method": "physical",
                "plane": "post_axicon",
                "k_r_m_inv": float(abs(k_r)),
                "physical_axicon_pixelated": False,
                "scalar_fresnel_amplitude": float(fresnel_amp),
            },
        )
    )
    mask = {
        "slm1_phase": bt.phase_wrap(slm1_phase),
        "slm1_grid": grid,
        "slm1_phase_components": ("vortex",),
        "slm1_contains_axicon": False,
        "slm2_phase": bt.phase_wrap(slm2_phase),
        "slm2_grid": grid,
        "slm2_phase_components": ("conjugate_flattening", "optional_amplitude_correction"),
        "slm2_contains_axicon": False,
        "axicon_phase": bt.phase_wrap(axicon_phase),
        "axicon_grid": grid,
        "axicon_phase_components": ("continuous_conical", "fresnel_sp"),
        "axicon_phase_is_pixelated": False,
        "axicon_k_r_m_inv": float(abs(k_r)),
        "slm2_residual_before_rad": float(diag["residual_phase_rms_before_rad"]),
        "slm2_residual_after_rad": float(diag["residual_phase_rms_after_rad"]),
    }
    return frames, mask


def build_train_frames(config: bt.TwinConfig, *, method: Method = "holographic", lab: bool = True) -> tuple[list[TrainFrame], dict[str, Any]]:
    if method == "holographic":
        return holographic_train_frames(config, lab=lab)
    if method == "physical":
        return physical_train_frames(config, lab=lab)
    raise ValueError(f"Unsupported method: {method!r}")


def plot_train_visualiser(
    config: bt.TwinConfig,
    *,
    method: Method = "holographic",
    output_dir: str | Path = "Publication_Study/outputs/figures/stage_c",
) -> Path:
    vbb_style.apply_style()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    lab_frames, lab_masks = build_train_frames(config, method=method, lab=True)
    ideal_frames, ideal_masks = build_train_frames(config, method=method, lab=False)
    n = max(len(lab_frames), len(ideal_frames))
    fig, axes = plt.subplots(4, n, figsize=(3.0 * n, 10.0), constrained_layout=True)
    for col in range(n):
        for row, frames in enumerate((ideal_frames, lab_frames)):
            if col >= len(frames):
                axes[2 * row, col].axis("off")
                axes[2 * row + 1, col].axis("off")
                continue
            field, grid = _crop_center(frames[col].field, frames[col].grid)
            amp = _normalised_amplitude(field)
            ph = _phase(field)
            plane_label = str(frames[col].metadata.get("plane", "plotted")).replace("_", " ")
            axes[2 * row, col].imshow(vbb_style.display_scale(amp, gamma=0.70), extent=_extent_um(grid), origin="lower", cmap=vbb_style.INTENSITY_CMAP)
            axes[2 * row, col].set_title(frames[col].label if row == 0 else "")
            axes[2 * row, col].set_xlabel(f"x [um, {plane_label} grid]")
            axes[2 * row, col].set_ylabel(("ideal amp" if row == 0 else "lab amp") + f"\ny [um, {plane_label} grid]")
            im = axes[2 * row + 1, col].imshow(ph, extent=_extent_um(grid), origin="lower", cmap=vbb_style.PHASE_CMAP, vmin=-np.pi, vmax=np.pi)
            axes[2 * row + 1, col].set_xlabel(f"x [um, {plane_label} grid]")
            axes[2 * row + 1, col].set_ylabel(("ideal phase" if row == 0 else "lab phase") + f"\ny [um, {plane_label} grid]")
    fig.colorbar(im, ax=axes[:, -1], shrink=0.70, label="phase [rad, cyclic]")
    before = lab_masks.get("slm2_residual_before_rad")
    after = lab_masks.get("slm2_residual_after_rad")
    subtitle = "" if before is None else f" SLM2 residual RMS {before:.3g} -> {after:.3g} rad"
    fig.suptitle(f"{method} train visualiser: ideal vs lab.{subtitle}")
    path = out / f"stage_c_train_{method}.png"
    return vbb_style.save_figure(
        fig,
        path,
        f"Per-stage {method} beam train. Amplitude is gamma displayed without changing the data; phase uses a cyclic twilight colormap. {subtitle.strip()}",
        metadata={"method": method, "slm2_residual_before_rad": before, "slm2_residual_after_rad": after},
    )


def plot_air_axial_trace(
    result: dict[str, Any],
    *,
    output_dir: str | Path = "Publication_Study/outputs/figures/stage_c",
    subject: str | None = None,
) -> Path:
    vbb_style.apply_style()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics = result["metrics"]
    volume = result["volume"]
    design = result["design"]
    z_um = np.asarray(volume["z"], dtype=float) / bt.um
    trace = np.asarray(volume["peak"], dtype=float)
    trace_n = trace / (float(np.max(trace)) + bt.EPS)
    surface_um = float(result["surface_field"].z_surface_m / bt.um)
    focus_um = float(z_um[int(volume["peak_index"])])
    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    ax.plot(z_um, trace_n, color="#0072B2", lw=2.0)
    ax.axhline(0.5, color="0.35", ls=":", lw=1.0)
    ax.axvspan(metrics["zone_start_um"], metrics["zone_end_um"], color="#009E73", alpha=0.18, label="axial FWHM plateau")
    ax.axvline(surface_um, color="#D55E00", lw=1.4, label="surface")
    ax.axvline(focus_um, color="#CC79A7", lw=1.1, ls="--", label="peak/focus")
    ax.set_xlabel("z [um, surface-in-air propagation]")
    ax.set_ylabel("normalised peak intensity [a.u.]")
    ax.set_title(f"Air axial trace ell={design.ell}, zone={metrics['bessel_zone_um']:.1f} um")
    ax.legend(frameon=False)
    path = out / f"stage_c_air_axial_trace_{subject or metrics.get('case_id', 'case')}.png"
    return vbb_style.save_figure(
        fig,
        path,
        "Corrected air beam axial trace. The scan spans both sides of the measured non-diffracting plateau, and the surface hand-off plane is inside that plateau.",
        metadata={
            "case_id": metrics.get("case_id"),
            "bessel_zone_um": metrics["bessel_zone_um"],
            "surface_z_um": surface_um,
            "focus_z_um": focus_um,
        },
    )


def _zero_order_leakage_from_order(
    result: dict[str, Any],
    *,
    dc_window_lpmm: float = 1.0,
) -> tuple[float, float]:
    order = result.get("order", {})
    if "A" not in order or "order_mask" not in order or "pupil_grid" not in result:
        return float("nan"), float("nan")
    A = np.asarray(order["A"])
    P = np.abs(A) ** 2
    total = float(np.sum(P)) + bt.EPS
    grid = result["pupil_grid"]
    FX = np.asarray(grid["FX"], dtype=float)
    FY = np.asarray(grid["FY"], dtype=float)
    selected = np.asarray(order["order_mask"], dtype=bool)
    dc = np.hypot(FX, FY) <= float(dc_window_lpmm) * 1.0e3
    leaked = float(np.sum(P[selected & dc]))
    dc_power = float(np.sum(P[dc])) + bt.EPS
    return float(leaked / total), float(leaked / dc_power)


def holographic_carrier_filter_sweep(
    config: bt.TwinConfig,
    *,
    blaze_period_px_values: Sequence[int] = (12, 14, 16, 18, 20, 22, 24, 28, 32),
    filter_radius_lpmm_values: Sequence[float] = (2.5, 4.0, 5.25, 6.0),
    zero_order_window_lpmm: float = 1.0,
    zero_order_leakage_threshold: float = 1.0e-4,
) -> pd.DataFrame:
    """Bounded holographic carrier/filter fairness sweep for one nominal case."""

    rows: list[dict[str, Any]] = []
    base = replace(config, generation_method="holographic")
    for period_px in blaze_period_px_values:
        for radius_lpmm in filter_radius_lpmm_values:
            slm = replace(base.slm, blaze_period_px=int(period_px), first_order_filter_radius_lpmm=float(radius_lpmm))
            cfg = replace(base, slm=slm)
            result = bt.run_case(cfg, preset=str(cfg.grid.label or "fast"), path="realistic", case_id=f"carrier_p{period_px}_r{radius_lpmm:g}")
            metrics = result["metrics"]
            leak_total, leak_dc = _zero_order_leakage_from_order(result, dc_window_lpmm=zero_order_window_lpmm)
            rows.append(
                {
                    "blaze_period_px": int(period_px),
                    "carrier_lpmm": float(slm.carrier_lpmm),
                    "configured_filter_radius_lpmm": float(radius_lpmm),
                    "effective_filter_radius_lpmm": float(metrics["first_order_effective_filter_radius_lpmm"]),
                    "recommended_filter_radius_lpmm": float(metrics["first_order_recommended_filter_radius_lpmm"]),
                    "axicon_cone_radius_lpmm": float(metrics["first_order_axicon_cone_radius_lpmm"]),
                    "first_order_selected_fraction": float(metrics["first_order_selected_fraction"]),
                    "zero_order_leakage_total_fraction": leak_total,
                    "zero_order_leakage_dc_fraction": leak_dc,
                    "bessel_zone_um": float(metrics["bessel_zone_um"]),
                    "zone_capped": bool(metrics["zone_capped"]),
                    "first_order_geometry_valid": bool(metrics["first_order_geometry_valid"]),
                    "phase_sampling_label": str(metrics["phase_sampling_label"]),
                    "focal_sampling_label": str(metrics["focal_sampling_label"]),
                    "validity_valid": bool(result["validity_report"]["valid"]),
                    "surface_z_um": float(metrics.get("surface_z_um", np.nan)),
                }
            )
    table = pd.DataFrame(rows)
    ok = (
        table["first_order_geometry_valid"].astype(bool)
        & (~table["zone_capped"].astype(bool))
        & (table["zero_order_leakage_total_fraction"] <= float(zero_order_leakage_threshold))
        & (table["first_order_selected_fraction"] > 0.0)
    )
    table["is_eligible_optimum"] = ok
    table["is_optimum"] = False
    if ok.any():
        eligible = table.loc[ok].sort_values(
            ["first_order_selected_fraction", "bessel_zone_um"],
            ascending=[False, False],
        )
        table.loc[eligible.index[0], "is_optimum"] = True
    return table


def plot_holographic_carrier_filter_tradeoff(
    table: pd.DataFrame,
    config: bt.TwinConfig,
    *,
    output_dir: str | Path = "Publication_Study/outputs/figures/stage_c",
    zero_order_leakage_threshold: float = 1.0e-4,
) -> Path:
    """Plot carrier/filter efficiency, zero-order leakage, and zone length."""

    vbb_style.apply_style()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 8.6), sharex=True, constrained_layout=True)
    for radius, group in table.groupby("configured_filter_radius_lpmm"):
        group = group.sort_values("carrier_lpmm")
        label = f"filter {radius:g} lp/mm"
        axes[0].plot(group["carrier_lpmm"], group["first_order_selected_fraction"], marker="o", label=label)
        axes[1].semilogy(group["carrier_lpmm"], np.maximum(group["zero_order_leakage_total_fraction"], 1.0e-12), marker="o", label=label)
        axes[2].plot(group["carrier_lpmm"], group["bessel_zone_um"], marker="o", label=label)
    opt = table.loc[table.get("is_optimum", False).astype(bool)] if "is_optimum" in table else pd.DataFrame()
    if not opt.empty:
        xopt = float(opt.iloc[0]["carrier_lpmm"])
        for ax in axes:
            ax.axvline(xopt, color="#009E73", lw=1.4, ls="--", label="chosen optimum" if ax is axes[0] else None)
    air = vbb_studies.beam_air_config(config)
    design = bt.compute_design_from_targets(air.laser, air.target, air.material)
    dx_eff = air.slm.pixel_pitch_m * max(1, int(air.grid.device_downsample))
    simulation_carrier_wall = 1.0 / (2.0 * dx_eff) / 1.0e3 - abs(design.kr_slm_m_inv) / bt.TWOPI / 1.0e3
    slm_carrier_wall = 1.0 / (2.0 * air.slm.pixel_pitch_m) / 1.0e3 - abs(design.kr_slm_m_inv) / bt.TWOPI / 1.0e3
    for ax in axes:
        ax.axvline(simulation_carrier_wall, color="#D55E00", lw=1.1, ls=":", label="simulation Nyquist wall" if ax is axes[0] else None)
        ax.axvline(slm_carrier_wall, color="0.35", lw=1.0, ls="-.", label="SLM pixel Nyquist wall" if ax is axes[0] else None)
        ax.grid(alpha=0.25)
    axes[1].axhline(float(zero_order_leakage_threshold), color="0.25", lw=0.9, ls="--", label="leakage threshold")
    axes[0].set_ylabel("first-order selected fraction")
    axes[1].set_ylabel("zero-order leakage / total")
    axes[2].set_ylabel("canonical zone [um]")
    axes[2].set_xlabel("blaze carrier [lp/mm, SLM/free-space plane]")
    axes[0].set_title("Holographic carrier/filter fairness sweep")
    axes[0].legend(frameon=False, ncols=2, fontsize=8)
    axes[1].legend(frameon=False, fontsize=8)
    path = out / "stage_c5_holographic_carrier_filter_tradeoff.png"
    return vbb_style.save_figure(
        fig,
        path,
        "Bounded carrier/filter sweep for the nominal ell=3 holographic beam. The chosen optimum maximises first-order selected fraction subject to low zero-order leakage and an uncapped canonical axial zone.",
        metadata={
            "zero_order_leakage_threshold": float(zero_order_leakage_threshold),
            "simulation_carrier_wall_lpmm": float(simulation_carrier_wall),
            "slm_pixel_carrier_wall_lpmm": float(slm_carrier_wall),
        },
    )


def method_comparison_table(
    config: bt.TwinConfig,
    *,
    regimes: Sequence[str] = ("general", "limits"),
    methods: Sequence[Method] = ("holographic", "physical"),
    path: bt.PathKind = "realistic",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for regime_name in regimes:
        cfg_reg = vbb_regime.config_for_regime(config, regime_name)
        for method in methods:
            cfg = replace(cfg_reg, generation_method=method)
            actual_path = path if method == "holographic" else "ideal"
            result = bt.run_case(cfg, preset=str(cfg.grid.label or "fast"), path=actual_path, case_id=f"{regime_name}_{method}")
            meta = result.get("axicon_metadata", {})
            rows.append(
                {
                    "regime": regime_name,
                    "method": method,
                    "path": actual_path,
                    "bessel_zone_um": result["metrics"]["bessel_zone_um"],
                    "feature_diameter_um": result["metrics"]["feature_diameter_um"],
                    "peak_fluence_J_cm2": result["metrics"]["peak_fluence_J_cm2"],
                    "side_to_core_peak_ratio": result["metrics"]["side_to_core_peak_ratio"],
                    "throughput_or_efficiency": result["metrics"].get("first_order_selected_fraction", np.nan),
                    "realised_k_r_m_inv": result.get("axicon_result", None).k_r if result.get("axicon_result") is not None else result["design"].kr_sample_m_inv,
                    "slm2_residual_before_rad": meta.get("slm2_residual_phase_rms_before_rad", np.nan),
                    "slm2_residual_after_rad": meta.get("slm2_residual_phase_rms_after_rad", np.nan),
                    "validity_valid": result["validity_report"]["valid"],
                }
            )
    return pd.DataFrame(rows)


def plot_sampling_qa(
    config: bt.TwinConfig,
    *,
    output_dir: str | Path = "Publication_Study/outputs/figures/stage_c",
) -> Path:
    vbb_style.apply_style()
    table = vbb_regime.validity_map(config, "limits")
    cores = sorted(table["target_core_diameter_um"].unique())
    zones = sorted(table["target_bessel_length_um"].unique())
    status = np.zeros((len(cores), len(zones)), dtype=int)
    for i, core in enumerate(cores):
        for j, zone in enumerate(zones):
            row = table[(table["target_core_diameter_um"] == core) & (table["target_bessel_length_um"] == zone)].iloc[0]
            status[i, j] = 2 if bool(row["valid"]) else 0
    cmap = ListedColormap(["#D55E00", "#E69F00", "#009E73"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    fig, ax = plt.subplots(figsize=(7.0, 4.6), constrained_layout=True)
    im = ax.imshow(status, origin="lower", aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(len(zones)), [f"{z:g}" for z in zones])
    ax.set_yticks(range(len(cores)), [f"{c:g}" for c in cores])
    ax.set_xlabel("target Bessel length [um, surface plane]")
    ax.set_ylabel("target core diameter [um, surface plane]")
    ax.set_title("Limits-regime sampling QA")
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["fail", "marginal", "pass"])
    path = Path(output_dir) / "stage_c_sampling_qa_limits.png"
    return vbb_style.save_figure(
        fig,
        path,
        "Sampling QA map from the real Stage-B/C per-cell validity report. The limits grid contains a mixed valid/invalid envelope rather than a flat placeholder.",
        metadata={"valid_count": int(table["valid"].sum()), "invalid_count": int((~table["valid"]).sum())},
    )


__all__ = [
    "TrainFrame",
    "build_train_frames",
    "holographic_carrier_filter_sweep",
    "holographic_train_frames",
    "method_comparison_table",
    "physical_train_frames",
    "plot_air_axial_trace",
    "plot_holographic_carrier_filter_tradeoff",
    "plot_sampling_qa",
    "plot_train_visualiser",
]
