"""Fit the real q=20 residual morphology after removing unresolved affine walk.

The fixed-camera BMG stack contains a nearly affine relative beam/camera motion
whose physical ownership is not calibrated.  This script keeps that trajectory
as an observation-frame nuisance and fits only the residual beam morphology in a
beam-following frame.

A compact SLM2 angular phase basis is estimated with a finite-difference
Gauss--Newton step through the complete multirate optical route.  The positive
phase must improve reconstruction of *held-out* BMG planes before its conjugate
is allowed to be described as a conditional correction prediction.
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
for path in (ROOT, EXP):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from real_bmg_digital_twin_correction import (
    AxiconError,
    FourFError,
    SystemErrorConfig,
    propagate_route,
    sample_model,
)
from vbb_study.digital_twin.observation_frame import fit_affine_trajectory, shift_stack_by_trajectory
from vbb_study.digital_twin.residual_phase_fit import angular_phase_from_coefficients

EPS = np.finfo(float).tiny
THERMAL = "inferno"
MODES = (1, 2, 3)


def normalise(stack: np.ndarray) -> np.ndarray:
    a = np.maximum(np.asarray(stack, float), 0.0)
    return a / np.maximum(a.reshape(a.shape[0], -1).max(axis=1)[:, None, None], EPS)


def metric_stack(pred: np.ndarray, data: np.ndarray, axis_um: np.ndarray, indices: np.ndarray) -> dict:
    X, Y = np.meshgrid(axis_um, axis_um, indexing="xy")
    roi = np.hypot(X, Y) <= 145.0
    rows = []
    pn, dn = normalise(pred), normalise(data)
    for iz in np.asarray(indices, int):
        a = pn[int(iz)][roi]
        b = dn[int(iz)][roi]
        rows.append({
            "index": int(iz),
            "pearson_r": float(np.corrcoef(a, b)[0, 1]),
            "nrmse": float(np.sqrt(np.mean((a-b)**2))),
        })
    return {
        "mean_pearson_r": float(np.mean([r["pearson_r"] for r in rows])),
        "mean_nrmse": float(np.mean([r["nrmse"] for r in rows])),
        "planes": rows,
    }


def savefig(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def select_nominal_iris(source_dir: Path) -> tuple[float, dict]:
    """Retain nominal iris when the old screen does not identify a unique value."""
    table = pd.read_csv(source_dir / "physical_parameter_objective_scans.csv")
    iris = table[table.parameter == "4F iris radius scale"].copy()
    if iris.empty:
        return 1.0, {"reason": "no prior iris screen; use nominal"}
    best = iris.loc[iris.objective.idxmin()]
    nominal = iris.iloc[np.argmin(np.abs(iris.value.to_numpy(float)-1.0))]
    rel = float(nominal.objective / max(float(best.objective), EPS) - 1.0)
    if rel <= 0.05:
        return 1.0, {
            "reason": "nominal retained because its objective is within 5% of broad minimum",
            "best_screen_value": float(best.value),
            "nominal_relative_objective_increase": rel,
        }
    return float(best.value), {
        "reason": "prior screen separates nominal by more than 5%",
        "best_screen_value": float(best.value),
        "nominal_relative_objective_increase": rel,
    }


def simulate(config: SystemErrorConfig, z_abs_m: np.ndarray, axis_um: np.ndarray, phase_slm2=None):
    native, meta = propagate_route(config, z_abs_m, phase_slm2=phase_slm2)
    shown = sample_model(native, meta["x_m"], axis_um)
    del native
    return shown, meta


def build_weighted_linear_system(
    baseline: np.ndarray,
    derivatives: list[np.ndarray],
    measured: np.ndarray,
    axis_um: np.ndarray,
    train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    X, Y = np.meshgrid(axis_um, axis_um, indexing="xy")
    R = np.hypot(X, Y)
    # Keep the vortex ring and useful nearby sidelobes, but do not let the large
    # dark outer area dominate a least-squares fit.
    roi = (R >= 12.0) & (R <= 135.0)
    b0 = normalise(baseline)
    data = normalise(measured)
    pieces_J, pieces_y = [], []
    for iz in np.asarray(train, int):
        target = data[int(iz)][roi]
        residual = target - b0[int(iz)][roi]
        weight = np.sqrt(0.20 + 0.80*np.sqrt(np.clip(target, 0.0, 1.0)))
        columns = [np.asarray(d[int(iz)][roi], float) for d in derivatives]
        pieces_J.append(weight[:, None] * np.column_stack(columns))
        pieces_y.append(weight * residual)
    return np.vstack(pieces_J), np.concatenate(pieces_y)


def ridge_step(J: np.ndarray, y: np.ndarray, *, ridge: float = 2e-3, bound: float = 0.9) -> np.ndarray:
    lhs = J.T @ J + float(ridge) * np.eye(J.shape[1])
    rhs = J.T @ y
    coeff = np.linalg.solve(lhs, rhs)
    return np.clip(coeff, -float(bound), float(bound))


def finite_difference_jacobian(
    config: SystemErrorConfig,
    z_abs_m: np.ndarray,
    axis_um: np.ndarray,
    theta_slm2: np.ndarray,
    centre_coeff: np.ndarray,
    *,
    delta: float,
) -> tuple[list[np.ndarray], list[dict]]:
    derivatives, records = [], []
    ncoef = centre_coeff.size
    for j in range(ncoef):
        cp = centre_coeff.copy(); cm = centre_coeff.copy()
        cp[j] += delta; cm[j] -= delta
        pp = angular_phase_from_coefficients(theta_slm2, cp, modes=MODES)
        pm = angular_phase_from_coefficients(theta_slm2, cm, modes=MODES)
        plus, _ = simulate(config, z_abs_m, axis_um, pp)
        minus, _ = simulate(config, z_abs_m, axis_um, pm)
        derivative = (normalise(plus) - normalise(minus)) / (2.0*delta)
        derivatives.append(derivative.astype(np.float32))
        records.append({"coefficient_index": j, "mode": MODES[j//2],
                        "component": "cos" if j % 2 == 0 else "sin", "delta_rad": float(delta)})
    return derivatives, records


def run(source_dir: Path, out: Path) -> dict:
    source_dir, out = Path(source_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    d = np.load(source_dir / "rerender_arrays.npz")
    measured = normalise(d["measured"])
    axis_um = np.asarray(d["axis_um"], float)
    z_rel_mm = np.asarray(d["z_relative_mm"], float)

    path = pd.read_csv(source_dir / "measured_beam_path.csv")
    measured_yx = path[["y_relative_um", "x_relative_um"]].to_numpy(float)
    affine = fit_affine_trajectory(z_rel_mm, measured_yx, centre_fit=True)
    measured_bf = shift_stack_by_trajectory(measured, axis_um, affine.fitted_yx, inverse=True)

    kinfo = json.loads((source_dir / "measured_k_perp_calibration.json").read_text(encoding="utf-8"))
    run_summary = json.loads((source_dir / "run_summary.json").read_text(encoding="utf-8"))
    scale = float(run_summary["effective_axicon"]["effective_scale_relative_to_repository_2deg_assumption"])
    ztable = pd.read_csv(source_dir / "full_route_z_registration_scan.csv")
    z0_mm = float(ztable.loc[ztable.selected.astype(bool), "value"].iloc[0])
    z_abs_m = (z0_mm + z_rel_mm) * 1e-3
    iris, iris_policy = select_nominal_iris(source_dir)
    config = SystemErrorConfig(
        fourf=FourFError(iris_radius_scale=float(iris)),
        axicon=AxiconError(base_angle_scale=float(scale)),
    )

    train = np.arange(0, len(z_abs_m), 2, dtype=int)
    held = np.arange(1, len(z_abs_m), 2, dtype=int)
    baseline, meta = simulate(config, z_abs_m, axis_um)
    theta = np.asarray(meta["theta_slm2"], float)
    before_train = metric_stack(baseline, measured_bf, axis_um, train)
    before_held = metric_stack(baseline, measured_bf, axis_um, held)

    # First Gauss--Newton linearization about zero phase.
    c0 = np.zeros(2*len(MODES), float)
    jac1, jac_records = finite_difference_jacobian(
        config, z_abs_m, axis_um, theta, c0, delta=0.18)
    J1, y1 = build_weighted_linear_system(baseline, jac1, measured_bf, axis_um, train)
    step1 = ridge_step(J1, y1, ridge=3e-3, bound=0.8)
    phase1 = angular_phase_from_coefficients(theta, step1, modes=MODES)
    pred1, _ = simulate(config, z_abs_m, axis_um, phase1)
    score1_train = metric_stack(pred1, measured_bf, axis_um, train)
    score1_held = metric_stack(pred1, measured_bf, axis_um, held)

    # A second local linearization is only accepted if the first nonlinear
    # evaluation improved held-out correlation *and* RMSE.  This prevents a
    # second step from polishing training planes after the physical signal has
    # already stopped generalising.
    second_attempted = bool(
        score1_held["mean_pearson_r"] > before_held["mean_pearson_r"]
        and score1_held["mean_nrmse"] < before_held["mean_nrmse"]
    )
    candidates = [("step1", step1, pred1, score1_train, score1_held)]
    if second_attempted:
        jac2, _ = finite_difference_jacobian(
            config, z_abs_m, axis_um, theta, step1, delta=0.12)
        J2, y2 = build_weighted_linear_system(pred1, jac2, measured_bf, axis_um, train)
        delta2 = ridge_step(J2, y2, ridge=4e-3, bound=0.45)
        step2 = np.clip(step1 + delta2, -1.0, 1.0)
        phase2 = angular_phase_from_coefficients(theta, step2, modes=MODES)
        pred2, _ = simulate(config, z_abs_m, axis_um, phase2)
        candidates.append(("step2", step2, pred2,
                           metric_stack(pred2, measured_bf, axis_um, train),
                           metric_stack(pred2, measured_bf, axis_um, held)))

    # Select using held-out performance only.  Require both metrics to move in
    # the right direction relative to nominal; otherwise retain zero residual.
    valid = [item for item in candidates
             if item[4]["mean_pearson_r"] > before_held["mean_pearson_r"]
             and item[4]["mean_nrmse"] < before_held["mean_nrmse"]]
    if valid:
        selected = max(valid, key=lambda item: item[4]["mean_pearson_r"] - 0.5*item[4]["mean_nrmse"])
        name, coeff, error_reconstruction, after_train, after_held = selected
    else:
        name, coeff, error_reconstruction = "nominal", c0, baseline
        after_train, after_held = before_train, before_held

    closure_gain_r = float(after_held["mean_pearson_r"] - before_held["mean_pearson_r"])
    closure_reduction_e = float(before_held["mean_nrmse"] - after_held["mean_nrmse"])
    closure_supported = bool(closure_gain_r >= 0.02 and closure_reduction_e >= 0.003)
    phase_error = angular_phase_from_coefficients(theta, coeff, modes=MODES)
    phase_correction = -phase_error

    # The baseline is the conditional corrected-model field only if the positive
    # phase reconstruction is independently supported.  No synthetic 'perfect'
    # correction column is emitted when closure is weak.
    conditional_correction = baseline if closure_supported else None

    fit_rows = []
    for label, _, _, tr, he in candidates:
        fit_rows.append({
            "candidate": label,
            "train_mean_r": tr["mean_pearson_r"], "train_mean_nrmse": tr["mean_nrmse"],
            "heldout_mean_r": he["mean_pearson_r"], "heldout_mean_nrmse": he["mean_nrmse"],
        })
    pd.DataFrame(fit_rows).to_csv(out / "beam_frame_residual_candidates.csv", index=False)
    pd.DataFrame(jac_records).to_csv(out / "beam_frame_jacobian_columns.csv", index=False)
    np.save(out / "beam_frame_predicted_slm2_error_phase_rad.npy", phase_error.astype(np.float32))
    np.save(out / "beam_frame_predicted_slm2_conjugate_phase_rad.npy", phase_correction.astype(np.float32))
    np.savez_compressed(out / "beam_frame_residual_stacks.npz",
                        measured_beam_frame=measured_bf.astype(np.float32),
                        nominal=baseline.astype(np.float32),
                        error_reconstruction=error_reconstruction.astype(np.float32),
                        z_relative_mm=z_rel_mm, axis_um=axis_um,
                        coefficients_rad=coeff.astype(float))

    # Compact scientific diagnostic.
    rep = int(np.argmin(np.abs(z_rel_mm + 10.0)))
    extent = [axis_um[0], axis_um[-1], axis_um[0], axis_um[-1]]
    fig, axs = plt.subplots(2, 3, figsize=(13.5, 8.2), constrained_layout=True)
    fields = [
        (measured_bf[rep], "BMG morphology\naffine walk removed"),
        (baseline[rep], "nominal digital twin"),
        (error_reconstruction[rep], "fitted positive error"),
    ]
    for col, (image, title) in enumerate(fields):
        axs[0,col].imshow(image, origin="lower", extent=extent, cmap=THERMAL, vmin=0, vmax=1)
        axs[0,col].set_aspect("equal"); axs[0,col].set(title=title, xlabel="x (um)", ylabel="y (um)")
    for label, stack, style in (("BMG", measured_bf, "-"), ("nominal", baseline, "--"), ("fitted error", error_reconstruction, ":")):
        axs[1,0].plot(axis_um, stack[rep, len(axis_um)//2, :], style, lw=1.4, label=label)
    axs[1,0].set(title="horizontal section", xlabel="x (um)", ylabel="plane-normalized intensity", xlim=(-130,130)); axs[1,0].grid(alpha=.2); axs[1,0].legend(fontsize=8)
    # per-plane held/train correlation
    all_idx = np.arange(len(z_rel_mm))
    b_all = metric_stack(baseline, measured_bf, axis_um, all_idx)
    e_all = metric_stack(error_reconstruction, measured_bf, axis_um, all_idx)
    br = [r["pearson_r"] for r in b_all["planes"]]; er = [r["pearson_r"] for r in e_all["planes"]]
    axs[1,1].plot(z_rel_mm, br, "o-", ms=3, label="nominal")
    axs[1,1].plot(z_rel_mm, er, "o-", ms=3, label="positive error reconstruction")
    axs[1,1].scatter(z_rel_mm[held], np.asarray(er)[held], s=45, facecolors="none", edgecolors="k", label="held-out")
    axs[1,1].set(title="reconstruction agreement", xlabel="relative z (mm)", ylabel="Pearson r"); axs[1,1].grid(alpha=.2); axs[1,1].legend(fontsize=7)
    # phase on SLM2 numerical relay grid
    im = axs[1,2].imshow(phase_error, origin="lower", cmap="twilight_shifted", vmin=-np.pi, vmax=np.pi)
    axs[1,2].set(title="fitted residual phase\nSLM2 numerical plane", xlabel="relay-grid x", ylabel="relay-grid y")
    fig.colorbar(im, ax=axs[1,2], label="phase (rad)", shrink=.8)
    fig.suptitle("Real q=20 residual morphology fit in the beam-following frame")
    savefig(fig, out / "08_beam_frame_full_route_residual_fit")

    summary = {
        "status": "POSITIVE_ERROR_RECONSTRUCTION_SUPPORTED" if closure_supported else "RESIDUAL_PHASE_NOT_SUPPORTED_BY_HELDOUT_CLOSURE",
        "comparison_frame": "real BMG after removing only affine relative beam/camera walk",
        "observation_walk": {
            "y_slope_um_per_mm": float(affine.slope_yx_per_z[0]),
            "x_slope_um_per_mm": float(affine.slope_yx_per_z[1]),
            "physical_ownership_resolved": False,
            "used_as_slm_correction": False,
        },
        "physical_model": {
            "measured_k_perp_m_inv": float(kinfo["robust_k_perp_m_inv"]),
            "axicon_base_angle_scale": scale,
            "model_bound_relative_zero_absolute_mm": z0_mm,
            "iris_radius_scale": iris,
            "iris_selection_policy": iris_policy,
        },
        "phase_model": {"modes": list(MODES), "selected_candidate": name,
                        "coefficients_cos_sin_rad": coeff.tolist(),
                        "fit_plane": "numerical SLM2 relay grid; programmed q theta excluded"},
        "nominal": {"train": before_train, "heldout": before_held},
        "positive_error_reconstruction": {"train": after_train, "heldout": after_held,
                                          "heldout_correlation_gain": closure_gain_r,
                                          "heldout_nrmse_reduction": closure_reduction_e,
                                          "supported": closure_supported},
        "conditional_correction": {
            "available": conditional_correction is not None,
            "meaning": "if and only if the positive fitted error is causal, its exact conjugate cancels that modeled phase and returns the nominal physical baseline",
            "post_slm_measurement": False,
        },
        "hardware_ready": False,
    }
    (out / "beam_frame_residual_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path,
                        default=EXP / "outputs" / "digital_twin_correction")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "outputs" / "validation" / "q20_beam_frame_residual_v1")
    args = parser.parse_args()
    run(args.source_dir, args.out)


if __name__ == "__main__":
    main()
