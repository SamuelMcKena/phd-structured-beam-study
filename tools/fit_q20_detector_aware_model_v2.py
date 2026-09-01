"""Detector-aware q=20 real-data model v2.

This is a new evidence branch derivative; it does not overwrite accepted q=20
artifacts.  It addresses two limitations exposed by the poster audit:

1. The previous absolute-z registration was selected from a restricted 30--50 mm
   search using an axisymmetric morphology objective.  The resulting nominal
   route approaches the end of its modelled Bessel zone over the final measured
   planes even though the real BeamGage field remains stable.  Here absolute z
   is re-estimated on TRAIN planes only with the detector-aware full-intensity
   objective, then frozen before held-out scoring.
2. The previous residual model used only radially invariant angular phase modes.
   Miao et al. (Opt. Express 30, 11360--11371, 2022) reconstruct a general input
   wavefront and report low-order Zernike content (including astigmatism and
   trefoil) as physically useful correction coordinates.  We therefore test a
   compact Miao-informed smooth phase basis (defocus, astigmatism, coma,
   trefoil, spherical) together with a deliberately low-order log-amplitude
   nuisance basis.  The amplitude terms are DIAGNOSTIC: a phase-only SLM cannot
   directly correct them.

All optical candidates are propagated through the already accepted bench route
and measured 5.5 um square-pixel detector response.  Even z planes are used for
model fitting; odd planes are held out.  No hardware SLM mask is emitted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "notebooks" / "experimental" / "axicon_aberration_correction"
for p in (ROOT, EXP):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from real_bmg_digital_twin_correction import (  # noqa: E402
    AxiconError,
    FourFError,
    PIXEL_M,
    RELAY_N,
    FIT_N,
    FIT_WINDOW_M,
    SystemErrorConfig,
)
from vbb_study.digital_twin.detector_response import (  # noqa: E402
    plane_normalise,
    sample_camera_response,
)
from vbb_study.digital_twin.observation_frame import (  # noqa: E402
    fit_affine_trajectory,
    shift_stack_by_trajectory,
)
from vbb_study.digital_twin.vortex_continuous_propagation import (  # noqa: E402
    build_fixed_support_spectrum,
    native_field_at_z,
)
from vbb_study.digital_twin.vortex_system_route import build_multirate_system_route  # noqa: E402

EPS = np.finfo(float).tiny
THERMAL = "inferno"
Q = 20
PHASE_BASIS_NAMES = (
    "defocus",
    "astig_cos",
    "astig_sin",
    "coma_cos",
    "coma_sin",
    "trefoil_cos",
    "trefoil_sin",
    "spherical",
)
AMPLITUDE_BASIS_NAMES = ("dipole_cos", "dipole_sin", "quadrupole_cos", "quadrupole_sin")


def savefig(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=500, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def score(predicted: np.ndarray, measured: np.ndarray, axis_um: np.ndarray) -> dict:
    X, Y = np.meshgrid(axis_um, axis_um, indexing="xy")
    roi = np.hypot(X, Y) <= 145.0
    pn, mn = plane_normalise(predicted), plane_normalise(measured)
    rows = []
    for iz in range(len(pn)):
        a, b = pn[iz][roi], mn[iz][roi]
        rows.append({
            "pearson_r": float(np.corrcoef(a, b)[0, 1]),
            "nrmse": float(np.sqrt(np.mean((a - b) ** 2))),
        })
    return {
        "mean_pearson_r": float(np.mean([row["pearson_r"] for row in rows])),
        "mean_nrmse": float(np.mean([row["nrmse"] for row in rows])),
        "max_nrmse": float(np.max([row["nrmse"] for row in rows])),
        "per_plane": rows,
    }


def robust_objective(metrics: dict) -> float:
    # Mean error is primary; correlation and the worst plane prevent a solution
    # that wins on average by allowing the end of the scan to collapse.
    return float(
        metrics["mean_nrmse"]
        + 0.035 * (1.0 - metrics["mean_pearson_r"])
        + 0.12 * metrics["max_nrmse"]
    )


def detector_render(native: np.ndarray, x_m: np.ndarray, axis_um: np.ndarray) -> np.ndarray:
    shown, _ = sample_camera_response(
        native,
        np.asarray(x_m, float),
        np.asarray(axis_um, float) * 1e-6,
        pixel_pitch_m=PIXEL_M,
        quadrature_n=3,
    )
    return plane_normalise(shown)


def native_intensity(prop, z_abs_m: np.ndarray) -> np.ndarray:
    return np.asarray([
        np.abs(np.asarray(native_field_at_z(prop, float(z)), complex)) ** 2
        for z in np.asarray(z_abs_m, float)
    ], np.float32)


def build_context(source_dir: Path) -> dict:
    source_dir = Path(source_dir)
    d = np.load(source_dir / "rerender_arrays.npz")
    measured = plane_normalise(np.asarray(d["measured"], float))
    axis_um = np.asarray(d["axis_um"], float)
    z_rel = np.asarray(d["z_relative_mm"], float)

    path = pd.read_csv(source_dir / "measured_beam_path.csv")
    yx = path[["y_relative_um", "x_relative_um"]].to_numpy(float)
    affine = fit_affine_trajectory(z_rel, yx, centre_fit=True)
    data = plane_normalise(shift_stack_by_trajectory(measured, axis_um, affine.fitted_yx, inverse=True))

    source_summary = json.loads((source_dir / "run_summary.json").read_text(encoding="utf-8"))
    scale = float(source_summary["effective_axicon"]["effective_scale_relative_to_repository_2deg_assumption"])
    config = SystemErrorConfig(
        fourf=FourFError(iris_radius_scale=1.0),
        axicon=AxiconError(base_angle_scale=scale),
    )
    route = build_multirate_system_route(
        f"V{Q}",
        relay_grid_n=RELAY_N,
        propagation_grid_n=FIT_N,
        window_m=FIT_WINDOW_M,
        config=config,
    )
    grid = route["grid"]
    wavelength = float(route["metadata"]["wavelength_m"])
    baseline_field = np.asarray(route["post_axicon"], np.complex128)
    # Build once over the entire registration search.  Candidate residuals later
    # rebuild the spectrum because they change the complex field, not the bench.
    baseline_prop = build_fixed_support_spectrum(
        baseline_field,
        grid,
        wavelength_m=wavelength,
        z_max_m=0.060,
        minimum_retained_spectral_power=0.98,
    )
    return {
        "data": data,
        "axis_um": axis_um,
        "z_rel": z_rel,
        "config": config,
        "route": route,
        "grid": grid,
        "wavelength": wavelength,
        "baseline_field": baseline_field,
        "baseline_prop": baseline_prop,
        "affine": affine,
    }


def render_baseline(context: dict, z_abs_m: np.ndarray) -> np.ndarray:
    native = native_intensity(context["baseline_prop"], z_abs_m)
    return detector_render(native, np.asarray(context["grid"]["x"], float), context["axis_um"])


def scan_absolute_z(context: dict, train: np.ndarray, out: Path) -> tuple[float, pd.DataFrame]:
    z_rel = context["z_rel"]
    data = context["data"]
    axis_um = context["axis_um"]
    # Use five train planes spanning the complete acquisition so the selected z
    # cannot hide a Bessel-zone collapse at the end of the stack.
    scan_ids = np.asarray([train[0], train[2], train[4], train[6], train[-1]], dtype=int)

    records: list[dict] = []
    def evaluate(z0_mm: float, stage: str) -> None:
        z_abs = (float(z0_mm) + z_rel[scan_ids]) * 1e-3
        pred = render_baseline(context, z_abs)
        m = score(pred, data[scan_ids], axis_um)
        records.append({
            "stage": stage,
            "z0_mm": float(z0_mm),
            "mean_pearson_r": m["mean_pearson_r"],
            "mean_nrmse": m["mean_nrmse"],
            "max_nrmse": m["max_nrmse"],
            "objective": robust_objective(m),
        })

    for z0 in np.arange(18.0, 50.01, 2.0):
        evaluate(float(z0), "coarse")
    coarse = min((r for r in records if r["stage"] == "coarse"), key=lambda r: r["objective"])
    for z0 in np.arange(max(17.5, coarse["z0_mm"] - 2.0), min(50.5, coarse["z0_mm"] + 2.01), 0.5):
        if not any(abs(r["z0_mm"] - float(z0)) < 1e-9 for r in records):
            evaluate(float(z0), "refine")

    table = pd.DataFrame(records).sort_values("z0_mm").reset_index(drop=True)
    best = table.iloc[int(table.objective.argmin())]
    table["selected"] = np.isclose(table.z0_mm, float(best.z0_mm))
    table.to_csv(out / "absolute_z_registration_train_only.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    ax.plot(table.z0_mm, table.objective, "o-", lw=1.6, ms=4, label="train objective")
    ax.axvline(float(best.z0_mm), ls="--", lw=1.5, label=f"selected z0 = {float(best.z0_mm):.1f} mm")
    ax.set(xlabel="absolute z at relative z = 0 (mm)", ylabel="robust detector-aware objective",
           title="Train-only absolute-z registration")
    ax.grid(alpha=.22); ax.legend(frameon=False)
    savefig(fig, out / "01_absolute_z_registration_train_only")
    return float(best.z0_mm), table


def _normalise_basis(arr: np.ndarray, support: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, float)
    scale = float(np.max(np.abs(a[support])))
    return a / max(scale, EPS)


def residual_bases(grid: dict, radius_m: float = 1.60e-3) -> tuple[list[np.ndarray], list[np.ndarray]]:
    X = np.asarray(grid["X"], float)
    Y = np.asarray(grid["Y"], float)
    R = np.hypot(X, Y)
    theta = np.arctan2(Y, X)
    rho = R / float(radius_m)
    # Smoothly suppress basis excursions beyond the annuli sampled by the real
    # scan.  This avoids an artificial hard pupil edge in the fitted residual.
    env = np.exp(-0.5 * (R / 2.05e-3) ** 8)
    support = R <= 2.0e-3

    phase = [
        env * (2.0 * rho**2 - 1.0),
        env * rho**2 * np.cos(2.0 * theta),
        env * rho**2 * np.sin(2.0 * theta),
        env * (3.0 * rho**3 - 2.0 * rho) * np.cos(theta),
        env * (3.0 * rho**3 - 2.0 * rho) * np.sin(theta),
        env * rho**3 * np.cos(3.0 * theta),
        env * rho**3 * np.sin(3.0 * theta),
        env * (6.0 * rho**4 - 6.0 * rho**2 + 1.0),
    ]
    amp = [
        env * rho * np.cos(theta),
        env * rho * np.sin(theta),
        env * rho**2 * np.cos(2.0 * theta),
        env * rho**2 * np.sin(2.0 * theta),
    ]
    return (
        [_normalise_basis(v, support) for v in phase],
        [_normalise_basis(v, support) for v in amp],
    )


def maps_from_coefficients(
    pcoef: np.ndarray,
    acoef: np.ndarray,
    phase_basis: list[np.ndarray],
    amp_basis: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phase = np.zeros_like(phase_basis[0], dtype=float)
    for c, basis in zip(np.asarray(pcoef, float), phase_basis):
        phase += float(c) * basis
    logamp = np.zeros_like(amp_basis[0], dtype=float)
    for c, basis in zip(np.asarray(acoef, float), amp_basis):
        logamp += float(c) * basis
    logamp = np.clip(logamp, -0.45, 0.45)
    amplitude = np.exp(logamp)
    return phase, logamp, amplitude


def simulate_residual(
    context: dict,
    z_abs_m: np.ndarray,
    pcoef: np.ndarray,
    acoef: np.ndarray,
    phase_basis: list[np.ndarray],
    amp_basis: list[np.ndarray],
) -> np.ndarray:
    phase, _, amp = maps_from_coefficients(pcoef, acoef, phase_basis, amp_basis)
    field = context["baseline_field"] * amp * np.exp(1j * phase)
    prop = build_fixed_support_spectrum(
        field,
        context["grid"],
        wavelength_m=context["wavelength"],
        z_max_m=max(0.002, float(np.max(np.abs(z_abs_m))) + 0.002),
        minimum_retained_spectral_power=0.98,
    )
    native = native_intensity(prop, z_abs_m)
    return detector_render(native, np.asarray(context["grid"]["x"], float), context["axis_um"])


def weighted_system(current: np.ndarray, derivatives: list[np.ndarray], target: np.ndarray, axis_um: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X, Y = np.meshgrid(axis_um, axis_um, indexing="xy")
    R = np.hypot(X, Y)
    roi = (R >= 12.0) & (R <= 140.0)
    current_n = plane_normalise(current)
    target_n = plane_normalise(target)
    J, y = [], []
    for iz in range(len(current_n)):
        t = target_n[iz][roi]
        residual = t - current_n[iz][roi]
        weight = np.sqrt(0.18 + 0.82 * np.sqrt(np.clip(t, 0.0, 1.0)))
        J.append(weight[:, None] * np.column_stack([d[iz][roi] for d in derivatives]))
        y.append(weight * residual)
    return np.vstack(J), np.concatenate(y)


def fit_complex_residual(context: dict, z_abs: np.ndarray, train: np.ndarray, held: np.ndarray, out: Path) -> dict:
    data = context["data"]
    axis_um = context["axis_um"]
    phase_basis, amp_basis = residual_bases(context["grid"])
    pcoef = np.zeros(len(phase_basis), float)
    acoef = np.zeros(len(amp_basis), float)
    current = render_baseline(context, z_abs[train])
    history = []

    for iteration in range(2):
        derivatives: list[np.ndarray] = []
        dp = 0.12 if iteration == 0 else 0.08
        da = 0.08 if iteration == 0 else 0.05
        for j in range(len(pcoef)):
            pp, pm = pcoef.copy(), pcoef.copy()
            pp[j] += dp; pm[j] -= dp
            plus = simulate_residual(context, z_abs[train], pp, acoef, phase_basis, amp_basis)
            minus = simulate_residual(context, z_abs[train], pm, acoef, phase_basis, amp_basis)
            derivatives.append((plus - minus) / (2.0 * dp))
        for j in range(len(acoef)):
            ap, am = acoef.copy(), acoef.copy()
            ap[j] += da; am[j] -= da
            plus = simulate_residual(context, z_abs[train], pcoef, ap, phase_basis, amp_basis)
            minus = simulate_residual(context, z_abs[train], pcoef, am, phase_basis, amp_basis)
            derivatives.append((plus - minus) / (2.0 * da))

        J, y = weighted_system(current, derivatives, data[train], axis_um)
        ridge = np.concatenate([
            np.full(len(pcoef), 8e-3, float),
            np.full(len(acoef), 2.0e-2, float),
        ])
        lhs = J.T @ J + np.diag(ridge)
        step = np.linalg.solve(lhs, J.T @ y)
        pstep = np.clip(step[:len(pcoef)], -0.32, 0.32)
        astep = np.clip(step[len(pcoef):], -0.14, 0.14)

        base_metrics = score(current, data[train], axis_um)
        best = (robust_objective(base_metrics), 0.0, pcoef.copy(), acoef.copy(), current)
        for strength in (1.0, 0.60, 0.35):
            ptry = np.clip(pcoef + strength * pstep, -0.90, 0.90)
            atry = np.clip(acoef + strength * astep, -0.35, 0.35)
            pred = simulate_residual(context, z_abs[train], ptry, atry, phase_basis, amp_basis)
            met = score(pred, data[train], axis_um)
            obj = robust_objective(met)
            if obj < best[0]:
                best = (obj, strength, ptry, atry, pred)
        _, strength, pcoef, acoef, current = best
        met = score(current, data[train], axis_um)
        history.append({
            "iteration": iteration + 1,
            "accepted_strength": float(strength),
            "phase_coefficients_rad": pcoef.tolist(),
            "log_amplitude_coefficients": acoef.tolist(),
            "train": met,
        })
        if strength == 0.0:
            break

    all_pred = simulate_residual(context, z_abs, pcoef, acoef, phase_basis, amp_basis)
    phase_only_pred = simulate_residual(context, z_abs, pcoef, np.zeros_like(acoef), phase_basis, amp_basis)
    baseline_all = render_baseline(context, z_abs)
    phase_map, logamp_map, amp_map = maps_from_coefficients(pcoef, acoef, phase_basis, amp_basis)

    result = {
        "phase_basis": list(PHASE_BASIS_NAMES),
        "amplitude_basis": list(AMPLITUDE_BASIS_NAMES),
        "phase_coefficients_rad": pcoef.tolist(),
        "log_amplitude_coefficients": acoef.tolist(),
        "iterations": history,
        "baseline_train": score(baseline_all[train], data[train], axis_um),
        "baseline_heldout": score(baseline_all[held], data[held], axis_um),
        "phase_only_same_phase_train": score(phase_only_pred[train], data[train], axis_um),
        "phase_only_same_phase_heldout": score(phase_only_pred[held], data[held], axis_um),
        "complex_train": score(all_pred[train], data[train], axis_um),
        "complex_heldout": score(all_pred[held], data[held], axis_um),
        "amplitude_terms_are_hardware_correctable_by_phase_only_slm": False,
    }

    np.savez_compressed(
        out / "model_v2_stacks.npz",
        measured_beam_frame=data.astype(np.float32),
        detector_nominal=baseline_all.astype(np.float32),
        phase_only_residual=phase_only_pred.astype(np.float32),
        complex_residual=all_pred.astype(np.float32),
        axis_um=axis_um,
        z_relative_mm=context["z_rel"],
        phase_residual_rad=phase_map.astype(np.float32),
        log_amplitude_residual=logamp_map.astype(np.float32),
        amplitude_residual=amp_map.astype(np.float32),
        x_model_m=np.asarray(context["grid"]["x"], float),
        phase_coefficients_rad=pcoef,
        log_amplitude_coefficients=acoef,
    )

    # Poster-grade complete-stack comparison.  Every plane is shown so weak
    # regions cannot be hidden by representative-plane selection.
    extent = [axis_um[0], axis_um[-1], axis_um[0], axis_um[-1]]
    fig, axs = plt.subplots(3, 6, figsize=(15.5, 8.1), constrained_layout=True)
    for ax, iz in zip(axs.ravel(), range(len(context["z_rel"]))):
        ax.imshow(all_pred[iz], origin="lower", extent=extent, cmap=THERMAL, vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"{context['z_rel'][iz]:.0f} mm", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    fig.suptitle("Detector-aware q=20 model v2: fitted complex residual over all 18 planes", fontsize=14, fontweight="bold")
    savefig(fig, out / "02_model_v2_all_planes")

    ids = [0, 4, 8, 12, 16]
    fig, axs = plt.subplots(2, len(ids), figsize=(13.0, 5.35), constrained_layout=True)
    for col, iz in enumerate(ids):
        for row, stack in enumerate((data, all_pred)):
            axs[row, col].imshow(stack[iz], origin="lower", extent=extent, cmap=THERMAL, vmin=0, vmax=1, interpolation="nearest")
            axs[row, col].set_aspect("equal"); axs[row, col].set_xticks([]); axs[row, col].set_yticks([])
            if row == 0:
                axs[row, col].set_title(f"z = {context['z_rel'][iz]:.0f} mm", fontsize=10)
            if col == 0:
                axs[row, col].set_ylabel("MEASURED" if row == 0 else "MODEL V2", fontsize=10, fontweight="bold")
    fig.suptitle("Real BeamGage morphology versus detector-aware model v2", fontsize=14, fontweight="bold")
    savefig(fig, out / "03_measured_vs_model_v2_selected_planes")

    held_z = context["z_rel"][held]
    models = [
        ("nominal", baseline_all),
        ("phase part", phase_only_pred),
        ("phase + amplitude", all_pred),
    ]
    fig, axs = plt.subplots(2, 1, figsize=(8.2, 6.2), sharex=True, constrained_layout=True)
    for label, stack in models:
        s = score(stack[held], data[held], axis_um)
        axs[0].plot(held_z, [v["pearson_r"] for v in s["per_plane"]], "o-", lw=1.5, label=label)
        axs[1].plot(held_z, [v["nrmse"] for v in s["per_plane"]], "o-", lw=1.5, label=label)
    axs[0].set(ylabel="Pearson r", title="Odd z planes: held out from z registration and residual fitting")
    axs[1].set(xlabel="relative z (mm)", ylabel="NRMSE")
    for ax in axs:
        ax.grid(alpha=.22); ax.legend(frameon=False, ncol=3, fontsize=8)
    savefig(fig, out / "04_heldout_model_v2_metrics")

    model_x_mm = np.asarray(context["grid"]["x"], float) * 1e3
    pextent = [model_x_mm[0], model_x_mm[-1], model_x_mm[0], model_x_mm[-1]]
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 4.2), constrained_layout=True)
    im0 = axs[0].imshow(phase_map, origin="lower", extent=pextent, cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axs[0].set(title="Inferred phase residual", xlabel="x (mm)", ylabel="y (mm)", xlim=(-2.2, 2.2), ylim=(-2.2, 2.2), aspect="equal")
    fig.colorbar(im0, ax=axs[0], label="phase (rad)", fraction=.046)
    im1 = axs[1].imshow(amp_map, origin="lower", extent=pextent, cmap="viridis", vmin=.70, vmax=1.30)
    axs[1].set(title="Inferred amplitude nuisance", xlabel="x (mm)", ylabel="y (mm)", xlim=(-2.2, 2.2), ylim=(-2.2, 2.2), aspect="equal")
    fig.colorbar(im1, ax=axs[1], label="relative amplitude", fraction=.046)
    fig.suptitle("Miao-informed compact complex residual at the axicon-input plane", fontsize=13, fontweight="bold")
    savefig(fig, out / "05_inferred_complex_residual")

    return {"result": result, "stacks": (baseline_all, phase_only_pred, all_pred)}


def run(source_dir: Path, out: Path) -> dict:
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    context = build_context(Path(source_dir))
    n = len(context["z_rel"])
    train = np.arange(0, n, 2, dtype=int)
    held = np.arange(1, n, 2, dtype=int)

    z0_mm, zscan = scan_absolute_z(context, train, out)
    z_abs = (z0_mm + context["z_rel"]) * 1e-3
    fit = fit_complex_residual(context, z_abs, train, held, out)
    result = fit["result"]

    previous_zscan = pd.read_csv(Path(source_dir) / "full_route_z_registration_scan.csv")
    previous_z0 = float(previous_zscan.loc[previous_zscan.selected.astype(bool), "value"].iloc[0])
    summary = {
        "study": "q20 detector-aware model v2: train-only z registration + Miao-informed complex residual",
        "comparison_frame": "affine beam/camera walk removed before optical fitting",
        "detector_model": "5.5 um square pixel, 3x3 midpoint area integration, no free blur",
        "data_split": {
            "train_indices": train.tolist(),
            "heldout_indices": held.tolist(),
            "z_registration_uses_heldout": False,
            "residual_fit_uses_heldout": False,
        },
        "absolute_z_registration": {
            "previous_model_bound_z0_mm": previous_z0,
            "selected_train_only_detector_aware_z0_mm": z0_mm,
            "search_mm": [18.0, 50.0],
            "status": "model-bound nuisance registration; independent bench distance still required",
        },
        "residual_model": result,
        "miao_connection": {
            "paper": "B. Miao et al., Opt. Express 30, 11360-11371 (2022), doi:10.1364/OE.454796",
            "borrowed_principle": "intensity-only Bessel wavefront retrieval motivates low-order pupil phase coordinates and explicit separation of radial/non-axisymmetric aberration",
            "difference": "this fit evaluates candidates through the bench-matched SLM/4F/axicon/detector digital twin and retains held-out z planes",
        },
        "hardware_ready": False,
        "hardware_boundary": (
            "The fitted phase/amplitude maps are diagnostic at the axicon-input plane. Amplitude terms are not directly correctable by a phase-only SLM, and no new SLM2 mask is emitted until the phase-correctable component is separately propagated upstream with calibrated coordinates/LUT."
        ),
    }
    (out / "model_v2_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=EXP / "outputs" / "digital_twin_correction")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "validation" / "q20_detector_aware_model_v2")
    args = parser.parse_args()
    run(args.source_dir, args.out)


if __name__ == "__main__":
    main()
