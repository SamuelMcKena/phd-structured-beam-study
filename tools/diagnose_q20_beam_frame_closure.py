"""Re-score existing real-q20 positive error reconstructions in the beam frame.

No optical model is re-fit here.  The purpose is to determine whether the weak
fixed-camera closure of the existing Miao/full-model phase estimates was mainly
caused by the unresolved affine relative beam/camera walk, or whether the phase
estimates themselves fail to reproduce the residual measured morphology.
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vbb_study.digital_twin.observation_frame import fit_affine_trajectory, shift_stack_by_trajectory

EPS = np.finfo(float).tiny
THERMAL = "inferno"


def normalise(stack):
    a = np.maximum(np.asarray(stack, float), 0.0)
    p = np.maximum(a.reshape(a.shape[0], -1).max(axis=1), EPS)
    return a / p[:, None, None]


def metrics(predicted, measured, axis_um):
    X, Y = np.meshgrid(axis_um, axis_um, indexing="xy")
    roi = np.hypot(X, Y) <= 160.0
    rs, es = [], []
    for p, m in zip(normalise(predicted), normalise(measured)):
        pv, mv = p[roi], m[roi]
        rs.append(float(np.corrcoef(pv, mv)[0, 1]))
        es.append(float(np.sqrt(np.mean((pv-mv)**2))))
    return np.asarray(rs), np.asarray(es)


def savefig(fig, stem):
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def run(source_dir: Path, out: Path):
    source_dir, out = Path(source_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    d = np.load(source_dir / "rerender_arrays.npz")
    axis = np.asarray(d["axis_um"], float)
    z = np.asarray(d["z_relative_mm"], float)
    measured = normalise(d["measured"])
    baseline = normalise(d["calibrated_physical_baseline"])
    target = normalise(d["full_route_target"])
    miao_error = normalise(d["miao_error_reconstruction"])
    full_error = normalise(d["full_error_reconstruction"])
    miao_corrected = normalise(d["miao_only_same_model"])
    assisted_miao = normalise(d["physical_plus_miao"])

    path = pd.read_csv(source_dir / "measured_beam_path.csv")
    yx = path[["y_relative_um", "x_relative_um"]].to_numpy(float)
    affine = fit_affine_trajectory(z, yx, centre_fit=True)
    measured_bf = shift_stack_by_trajectory(measured, axis, affine.fitted_yx, inverse=True)

    candidates = {
        "nominal baseline": baseline,
        "Miao positive-error reconstruction": miao_error,
        "full-model positive-error reconstruction": full_error,
    }
    rows = []
    scores = {}
    for name, stack in candidates.items():
        r_fixed, e_fixed = metrics(stack, measured, axis)
        r_bf, e_bf = metrics(stack, measured_bf, axis)
        scores[name] = (r_fixed, e_fixed, r_bf, e_bf)
        for iz in range(len(z)):
            rows.append({
                "model": name, "z_index": iz, "z_relative_mm": float(z[iz]),
                "fixed_camera_pearson_r": float(r_fixed[iz]), "fixed_camera_nrmse": float(e_fixed[iz]),
                "beam_frame_pearson_r": float(r_bf[iz]), "beam_frame_nrmse": float(e_bf[iz]),
            })
    pd.DataFrame(rows).to_csv(out / "beam_frame_error_closure_metrics.csv", index=False)

    # Corrected predictions are assessed against the same calibrated target; this
    # is conditional forward-model performance, not a post-mask measurement.
    corrected_scores = {}
    for name, stack in {
        "Miao-only correction prediction": miao_corrected,
        "physical fit + Miao prediction": assisted_miao,
        "physical baseline / exact full-model cancellation": baseline,
    }.items():
        rr, ee = metrics(stack, target, axis)
        corrected_scores[name] = {"mean_r_to_target": float(np.mean(rr)), "mean_nrmse_to_target": float(np.mean(ee))}

    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    for name, (_, _, rr, ee) in scores.items():
        axs[0].plot(z, rr, "o-", ms=3, label=name)
        axs[1].plot(z, ee, "o-", ms=3, label=name)
    axs[0].set(xlabel="relative z (mm)", ylabel="Pearson r", title="Positive-error reconstruction vs detrended BMG")
    axs[1].set(xlabel="relative z (mm)", ylabel="normalized RMSE", title="Positive-error reconstruction vs detrended BMG")
    for ax in axs: ax.grid(alpha=.25); ax.legend(fontsize=7.5)
    fig.suptitle("Does the existing retrieved phase explain the residual optical morphology once affine walk is removed?")
    savefig(fig, out / "06_existing_error_closure_in_beam_frame")

    rep = int(np.argmin(np.abs(z + 10.0)))
    extent = [axis[0], axis[-1], axis[0], axis[-1]]
    fig, axs = plt.subplots(1, 4, figsize=(15, 4), constrained_layout=True)
    fields = [
        (measured_bf[rep], "BMG after affine walk removal"),
        (baseline[rep], "nominal baseline"),
        (miao_error[rep], "Miao positive error"),
        (full_error[rep], "full-model positive error"),
    ]
    for ax, (image, title) in zip(axs, fields):
        ax.imshow(image, origin="lower", extent=extent, cmap=THERMAL, vmin=0, vmax=1)
        ax.set_aspect("equal"); ax.set(title=title, xlabel="x (um)", ylabel="y (um)")
    fig.suptitle(f"Beam-frame error reconstruction at z={z[rep]:g} mm")
    savefig(fig, out / "07_beam_frame_error_reconstruction_xy")

    summary = {"comparison_frame": "BMG with only affine relative trajectory removed; model fields remain in selected-order beam frame",
               "affine_walk_is_not_assigned_to_optics": True,
               "error_reconstruction": {}, "conditional_correction_predictions": corrected_scores}
    for name, (rf, ef, rb, eb) in scores.items():
        summary["error_reconstruction"][name] = {
            "fixed_camera_mean_r": float(np.mean(rf)), "fixed_camera_mean_nrmse": float(np.mean(ef)),
            "beam_frame_mean_r": float(np.mean(rb)), "beam_frame_mean_nrmse": float(np.mean(eb)),
        }
    baseline_bf_r = summary["error_reconstruction"]["nominal baseline"]["beam_frame_mean_r"]
    baseline_bf_e = summary["error_reconstruction"]["nominal baseline"]["beam_frame_mean_nrmse"]
    for key in ("Miao positive-error reconstruction", "full-model positive-error reconstruction"):
        item = summary["error_reconstruction"][key]
        item["closure_improves_over_nominal"] = bool(item["beam_frame_mean_r"] > baseline_bf_r and item["beam_frame_mean_nrmse"] < baseline_bf_e)
    (out / "beam_frame_closure_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path,
                        default=ROOT / "notebooks" / "experimental" / "axicon_aberration_correction" / "outputs" / "digital_twin_correction")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "validation" / "q20_observation_frame")
    args = parser.parse_args()
    run(args.source_dir, args.out)


if __name__ == "__main__":
    main()
