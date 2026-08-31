"""Compare candidate residual-phase planes for the real q=20 BMG stack.

The measured affine beam/camera walk is first removed as an unresolved
observation-frame nuisance.  Identical low-order angular phase bases are then
linearised through the complete multirate route at either (a) SLM2 before the
4F relay or (b) the selected-order field immediately before the physical
axicon.  A real aberration introduced downstream of the relay should not be
forced onto SLM2 merely because SLM2 is the intended actuator.

This is a model-selection diagnostic.  It does not establish hardware
conjugacy or produce an experimentally validated correction.
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
    AxiconError, FourFError, SystemErrorConfig, propagate_route, sample_model,
)
from vbb_study.digital_twin.observation_frame import fit_affine_trajectory, shift_stack_by_trajectory
from vbb_study.digital_twin.residual_phase_fit import angular_phase_from_coefficients

EPS = np.finfo(float).tiny
MODES = (1, 2, 3)
THERMAL = "inferno"


def normalise(stack):
    a = np.maximum(np.asarray(stack, float), 0.0)
    return a / np.maximum(a.reshape(a.shape[0], -1).max(axis=1)[:, None, None], EPS)


def score(predicted, measured, axis_um, indices):
    X, Y = np.meshgrid(axis_um, axis_um, indexing="xy")
    roi = np.hypot(X, Y) <= 145.0
    pn, mn = normalise(predicted), normalise(measured)
    rows = []
    for iz in np.asarray(indices, int):
        a, b = pn[iz][roi], mn[iz][roi]
        rows.append((float(np.corrcoef(a, b)[0,1]), float(np.sqrt(np.mean((a-b)**2)))))
    return float(np.mean([x[0] for x in rows])), float(np.mean([x[1] for x in rows]))


def weighted_system(baseline, derivatives, measured, axis_um, train):
    X, Y = np.meshgrid(axis_um, axis_um, indexing="xy")
    R = np.hypot(X, Y)
    roi = (R >= 12.0) & (R <= 135.0)
    b0, data = normalise(baseline), normalise(measured)
    J, y = [], []
    for iz in np.asarray(train, int):
        target = data[iz][roi]
        residual = target - b0[iz][roi]
        w = np.sqrt(0.20 + 0.80*np.sqrt(np.clip(target, 0.0, 1.0)))
        J.append(w[:,None]*np.column_stack([d[iz][roi] for d in derivatives]))
        y.append(w*residual)
    return np.vstack(J), np.concatenate(y)


def ridge(J, y, ridge_value=3e-3):
    return np.clip(np.linalg.solve(J.T@J + ridge_value*np.eye(J.shape[1]), J.T@y), -0.9, 0.9)


def phase_grid(meta, plane):
    if plane == "slm2":
        return np.asarray(meta["theta_slm2"], float)
    x = np.asarray(meta["x_m"], float)
    X, Y = np.meshgrid(x, x, indexing="xy")
    return np.arctan2(Y, X)


def simulate(config, z_abs_m, axis_um, *, plane, coefficients, theta):
    phase = angular_phase_from_coefficients(theta, coefficients, modes=MODES)
    kwargs = {"phase_slm2": phase} if plane == "slm2" else {"phase_axicon_input": phase}
    native, meta = propagate_route(config, z_abs_m, **kwargs)
    shown = sample_model(native, meta["x_m"], axis_um)
    del native
    return shown


def savefig(fig, stem):
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def run(source_dir: Path, out: Path):
    source_dir, out = Path(source_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    d = np.load(source_dir/"rerender_arrays.npz")
    measured = normalise(d["measured"])
    axis_um = np.asarray(d["axis_um"], float)
    z_rel = np.asarray(d["z_relative_mm"], float)
    path = pd.read_csv(source_dir/"measured_beam_path.csv")
    yx = path[["y_relative_um", "x_relative_um"]].to_numpy(float)
    affine = fit_affine_trajectory(z_rel, yx, centre_fit=True)
    data = shift_stack_by_trajectory(measured, axis_um, affine.fitted_yx, inverse=True)

    summary0 = json.loads((source_dir/"run_summary.json").read_text(encoding="utf-8"))
    scale = float(summary0["effective_axicon"]["effective_scale_relative_to_repository_2deg_assumption"])
    zscan = pd.read_csv(source_dir/"full_route_z_registration_scan.csv")
    z0 = float(zscan.loc[zscan.selected.astype(bool), "value"].iloc[0])
    z_abs = (z0 + z_rel)*1e-3
    # The previous iris screen has a broad minimum: nominal 1.0 is only 2.66%
    # above the best 1.15 candidate.  Keep nominal rather than turning a weak
    # screen into a measured hardware parameter.
    config = SystemErrorConfig(
        fourf=FourFError(iris_radius_scale=1.0),
        axicon=AxiconError(base_angle_scale=scale),
    )
    native, meta = propagate_route(config, z_abs)
    baseline = sample_model(native, meta["x_m"], axis_um); del native
    train = np.arange(0, len(z_abs), 2, dtype=int)
    held = np.arange(1, len(z_abs), 2, dtype=int)
    btr = score(baseline, data, axis_um, train)
    bhe = score(baseline, data, axis_um, held)

    records = []
    stacks = {"nominal": baseline}
    coeff_records = {}
    for plane in ("slm2", "axicon_input"):
        theta = phase_grid(meta, plane)
        c0 = np.zeros(2*len(MODES), float)
        derivatives = []
        delta = 0.18
        for j in range(c0.size):
            cp, cm = c0.copy(), c0.copy()
            cp[j] += delta; cm[j] -= delta
            plus = simulate(config, z_abs, axis_um, plane=plane, coefficients=cp, theta=theta)
            minus = simulate(config, z_abs, axis_um, plane=plane, coefficients=cm, theta=theta)
            derivatives.append((normalise(plus)-normalise(minus))/(2*delta))
        J, y = weighted_system(baseline, derivatives, data, axis_um, train)
        coeff = ridge(J, y)
        predicted = simulate(config, z_abs, axis_um, plane=plane, coefficients=coeff, theta=theta)
        tr = score(predicted, data, axis_um, train)
        he = score(predicted, data, axis_um, held)
        records.append({
            "phase_plane": plane,
            "train_r_before": btr[0], "train_r_after": tr[0],
            "train_nrmse_before": btr[1], "train_nrmse_after": tr[1],
            "heldout_r_before": bhe[0], "heldout_r_after": he[0],
            "heldout_nrmse_before": bhe[1], "heldout_nrmse_after": he[1],
            "heldout_delta_r": he[0]-bhe[0],
            "heldout_delta_nrmse": he[1]-bhe[1],
        })
        stacks[plane] = predicted
        coeff_records[plane] = coeff.tolist()
    table = pd.DataFrame(records)
    table.to_csv(out/"residual_phase_plane_comparison.csv", index=False)

    best = table.sort_values(["heldout_r_after", "heldout_nrmse_after"], ascending=[False, True]).iloc[0]
    best_plane = str(best.phase_plane)
    supported = bool(float(best.heldout_delta_r) >= 0.02 and float(best.heldout_delta_nrmse) <= -0.003)

    rep = int(np.argmin(np.abs(z_rel+10.0)))
    extent = [axis_um[0], axis_um[-1], axis_um[0], axis_um[-1]]
    fig, axs = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    display = [
        (data, "BMG beam-frame"),
        (baseline, "nominal"),
        (stacks["slm2"], "phase fitted at SLM2"),
        (stacks["axicon_input"], "phase fitted at axicon input"),
    ]
    for col, (stack, title) in enumerate(display):
        axs[0,col].imshow(stack[rep], origin="lower", extent=extent, cmap=THERMAL, vmin=0, vmax=1)
        axs[0,col].set_aspect("equal"); axs[0,col].set(title=title, xlabel="x (um)", ylabel="y (um)")
        axs[1,col].plot(axis_um, data[rep,len(axis_um)//2,:], lw=1.5, label="BMG")
        axs[1,col].plot(axis_um, stack[rep,len(axis_um)//2,:], "--", lw=1.4, label=title)
        axs[1,col].set(xlim=(-130,130), ylim=(0,1.05), xlabel="x (um)", ylabel="plane-normalized intensity")
        axs[1,col].grid(alpha=.2); axs[1,col].legend(fontsize=7)
    fig.suptitle("Where does the residual q=20 phase belong? Same modes, same data, same forward model")
    savefig(fig, out/"09_residual_phase_plane_comparison")

    fig, axs = plt.subplots(1, 2, figsize=(11,4.5), constrained_layout=True)
    names = ["nominal", "SLM2", "axicon input"]
    rvals = [bhe[0], float(table[table.phase_plane=="slm2"].heldout_r_after.iloc[0]),
             float(table[table.phase_plane=="axicon_input"].heldout_r_after.iloc[0])]
    evals = [bhe[1], float(table[table.phase_plane=="slm2"].heldout_nrmse_after.iloc[0]),
             float(table[table.phase_plane=="axicon_input"].heldout_nrmse_after.iloc[0])]
    axs[0].bar(names, rvals); axs[0].set(ylabel="held-out Pearson r", title="Residual reconstruction")
    axs[1].bar(names, evals); axs[1].set(ylabel="held-out normalized RMSE", title="Residual reconstruction")
    for ax in axs: ax.tick_params(axis="x", rotation=15); ax.grid(axis="y", alpha=.2)
    fig.suptitle("Positive-error reconstruction on z planes excluded from fitting")
    savefig(fig, out/"10_residual_phase_plane_metrics")

    summary = {
        "comparison_frame": "affine relative beam/camera walk removed from measured BMG only",
        "baseline": {"heldout_r": bhe[0], "heldout_nrmse": bhe[1]},
        "planes": {row["phase_plane"]: row for row in records},
        "coefficients_cos_sin_rad": coeff_records,
        "best_phase_plane": best_plane,
        "positive_error_reconstruction_supported": supported,
        "support_rule": "held-out Pearson gain >=0.02 and normalized RMSE reduction >=0.003",
        "hardware_mapping_inferred": False,
    }
    (out/"residual_phase_plane_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=EXP/"outputs"/"digital_twin_correction")
    parser.add_argument("--out", type=Path, default=ROOT/"outputs"/"validation"/"q20_phase_plane_comparison")
    args = parser.parse_args()
    run(args.source_dir, args.out)

if __name__ == "__main__":
    main()
