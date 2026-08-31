"""Refit the real q=20 residual with the camera inside the forward model.

The BeamGage stack is sampled by 5.5 um detector pixels and later interpolated
to a finer display grid.  Earlier inverse fits compared that interpolated data
directly with a finely rendered numerical field.  This script first removes
only the unresolved affine beam/camera walk, then applies an explicit square
pixel response to every numerical prediction before fitting low-order residual
phase.

Two candidate phase planes are tested with identical m=1..3 angular bases:
SLM2 and the selected-order field immediately before the axicon.  Coefficients
are fitted on even z planes only.  Odd z planes are never used by the fit and
are reported as held-out evidence.  No SLM hardware mask is emitted.
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
    PIXEL_M,
    SystemErrorConfig,
    propagate_route,
    sample_model,
)
from vbb_study.digital_twin.detector_response import plane_normalise, sample_camera_response
from vbb_study.digital_twin.observation_frame import fit_affine_trajectory, shift_stack_by_trajectory
from vbb_study.digital_twin.residual_phase_fit import angular_phase_from_coefficients

EPS = np.finfo(float).tiny
MODES = (1, 2, 3)
THERMAL = "inferno"


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
        "per_plane": rows,
    }


def weighted_system(
    baseline: np.ndarray,
    derivatives: list[np.ndarray],
    measured: np.ndarray,
    axis_um: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    X, Y = np.meshgrid(axis_um, axis_um, indexing="xy")
    R = np.hypot(X, Y)
    roi = (R >= 12.0) & (R <= 135.0)
    b0, data = plane_normalise(baseline), plane_normalise(measured)
    J, y = [], []
    for iz in range(len(b0)):
        target = data[iz][roi]
        residual = target - b0[iz][roi]
        # Bright ring pixels carry more information without discarding the
        # lower-intensity sidelobe morphology.
        weight = np.sqrt(0.20 + 0.80 * np.sqrt(np.clip(target, 0.0, 1.0)))
        J.append(weight[:, None] * np.column_stack([d[iz][roi] for d in derivatives]))
        y.append(weight * residual)
    return np.vstack(J), np.concatenate(y)


def ridge_update(J: np.ndarray, y: np.ndarray, ridge_value: float = 3e-3) -> np.ndarray:
    lhs = J.T @ J + float(ridge_value) * np.eye(J.shape[1])
    return np.linalg.solve(lhs, J.T @ y)


def phase_grid(meta: dict, plane: str) -> np.ndarray:
    if plane == "slm2":
        return np.asarray(meta["theta_slm2"], float)
    x = np.asarray(meta["x_m"], float)
    X, Y = np.meshgrid(x, x, indexing="xy")
    return np.arctan2(Y, X)


def detector_render(native: np.ndarray, x_m: np.ndarray, axis_um: np.ndarray) -> np.ndarray:
    shown, _ = sample_camera_response(
        native,
        np.asarray(x_m, float),
        np.asarray(axis_um, float) * 1e-6,
        pixel_pitch_m=PIXEL_M,
        quadrature_n=3,
    )
    return plane_normalise(shown)


def simulate(
    config: SystemErrorConfig,
    z_abs_m: np.ndarray,
    axis_um: np.ndarray,
    *,
    plane: str,
    coefficients: np.ndarray,
    theta: np.ndarray,
) -> np.ndarray:
    phase = angular_phase_from_coefficients(theta, coefficients, modes=MODES)
    kwargs = {"phase_slm2": phase} if plane == "slm2" else {"phase_axicon_input": phase}
    native, meta = propagate_route(config, z_abs_m, **kwargs)
    shown = detector_render(native, meta["x_m"], axis_um)
    del native
    return shown


def fit_one_plane(
    config: SystemErrorConfig,
    z_train_m: np.ndarray,
    z_all_m: np.ndarray,
    axis_um: np.ndarray,
    train_data: np.ndarray,
    all_data: np.ndarray,
    baseline_train: np.ndarray,
    baseline_all: np.ndarray,
    theta: np.ndarray,
    plane: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    ncoef = 2 * len(MODES)
    coeff = np.zeros(ncoef, float)
    delta = 0.16
    iteration_records = []

    # Two train-only Gauss-Newton updates.  The held-out planes are not touched
    # until the coefficients have been frozen.
    current_train = baseline_train
    for iteration in range(2):
        derivatives = []
        for j in range(ncoef):
            cp, cm = coeff.copy(), coeff.copy()
            cp[j] += delta
            cm[j] -= delta
            plus = simulate(config, z_train_m, axis_um, plane=plane, coefficients=cp, theta=theta)
            minus = simulate(config, z_train_m, axis_um, plane=plane, coefficients=cm, theta=theta)
            derivatives.append((plus - minus) / (2.0 * delta))
        J, y = weighted_system(current_train, derivatives, train_data, axis_um)
        step = ridge_update(J, y)
        # Keep the local linearisation in a low-order, low-amplitude regime.
        step = np.clip(step, -0.45, 0.45)
        coeff = np.clip(coeff + step, -0.90, 0.90)
        current_train = simulate(
            config, z_train_m, axis_um, plane=plane, coefficients=coeff, theta=theta,
        )
        s = score(current_train, train_data, axis_um)
        iteration_records.append({
            "iteration": iteration + 1,
            "coefficients_rad": coeff.tolist(),
            "train_mean_pearson_r": s["mean_pearson_r"],
            "train_mean_nrmse": s["mean_nrmse"],
        })
        delta = 0.10

    predicted_all = simulate(
        config, z_all_m, axis_um, plane=plane, coefficients=coeff, theta=theta,
    )
    train_ids = np.arange(0, len(all_data), 2, dtype=int)
    held_ids = np.arange(1, len(all_data), 2, dtype=int)
    before_train = score(baseline_all[train_ids], all_data[train_ids], axis_um)
    before_held = score(baseline_all[held_ids], all_data[held_ids], axis_um)
    after_train = score(predicted_all[train_ids], all_data[train_ids], axis_um)
    after_held = score(predicted_all[held_ids], all_data[held_ids], axis_um)
    result = {
        "phase_plane": plane,
        "modes": list(MODES),
        "coefficients_cos_sin_rad": coeff.tolist(),
        "iterations": iteration_records,
        "train_before": before_train,
        "train_after": after_train,
        "heldout_before": before_held,
        "heldout_after": after_held,
        "heldout_delta_r": after_held["mean_pearson_r"] - before_held["mean_pearson_r"],
        "heldout_delta_nrmse": after_held["mean_nrmse"] - before_held["mean_nrmse"],
    }
    return coeff, predicted_all, result


def savefig(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def run(source_dir: Path, out: Path) -> dict:
    source_dir, out = Path(source_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)

    d = np.load(source_dir / "rerender_arrays.npz")
    measured = plane_normalise(np.asarray(d["measured"], float))
    axis_um = np.asarray(d["axis_um"], float)
    z_rel = np.asarray(d["z_relative_mm"], float)
    path = pd.read_csv(source_dir / "measured_beam_path.csv")
    yx = path[["y_relative_um", "x_relative_um"]].to_numpy(float)
    affine = fit_affine_trajectory(z_rel, yx, centre_fit=True)
    data = shift_stack_by_trajectory(measured, axis_um, affine.fitted_yx, inverse=True)
    data = plane_normalise(data)

    source_summary = json.loads((source_dir / "run_summary.json").read_text(encoding="utf-8"))
    scale = float(source_summary["effective_axicon"]["effective_scale_relative_to_repository_2deg_assumption"])
    zscan = pd.read_csv(source_dir / "full_route_z_registration_scan.csv")
    z0 = float(zscan.loc[zscan.selected.astype(bool), "value"].iloc[0])
    z_abs = (z0 + z_rel) * 1e-3
    config = SystemErrorConfig(
        fourf=FourFError(iris_radius_scale=1.0),
        axicon=AxiconError(base_angle_scale=scale),
    )

    native, meta = propagate_route(config, z_abs)
    direct = plane_normalise(sample_model(native, meta["x_m"], axis_um))
    detector_nominal = detector_render(native, meta["x_m"], axis_um)
    del native

    train = np.arange(0, len(z_abs), 2, dtype=int)
    held = np.arange(1, len(z_abs), 2, dtype=int)
    direct_train = score(direct[train], data[train], axis_um)
    direct_held = score(direct[held], data[held], axis_um)
    detector_train = score(detector_nominal[train], data[train], axis_um)
    detector_held = score(detector_nominal[held], data[held], axis_um)

    plane_results = {}
    predictions = {}
    coeffs = {}
    for plane in ("slm2", "axicon_input"):
        theta = phase_grid(meta, plane)
        c, prediction, result = fit_one_plane(
            config,
            z_abs[train],
            z_abs,
            axis_um,
            data[train],
            data,
            detector_nominal[train],
            detector_nominal,
            theta,
            plane,
        )
        coeffs[plane] = c
        predictions[plane] = prediction
        plane_results[plane] = result

    # Select the diagnostic phase plane only from the training objective.  The
    # odd-z result then remains a genuine external check of that selection.
    selected_plane = min(
        plane_results,
        key=lambda p: plane_results[p]["train_after"]["mean_nrmse"],
    )
    selected = plane_results[selected_plane]
    strong_support = bool(
        selected["heldout_delta_r"] >= 0.02
        and selected["heldout_delta_nrmse"] <= -0.003
    )
    directional_support = bool(
        selected["heldout_delta_r"] > 0.0
        and selected["heldout_delta_nrmse"] < 0.0
    )

    rows = []
    for label, stack in (
        ("direct_high_resolution_nominal", direct),
        ("detector_aware_nominal", detector_nominal),
        ("detector_aware_slm2_phase", predictions["slm2"]),
        ("detector_aware_axicon_input_phase", predictions["axicon_input"]),
    ):
        tr, he = score(stack[train], data[train], axis_um), score(stack[held], data[held], axis_um)
        rows.append({
            "model": label,
            "train_mean_pearson_r": tr["mean_pearson_r"],
            "train_mean_nrmse": tr["mean_nrmse"],
            "heldout_mean_pearson_r": he["mean_pearson_r"],
            "heldout_mean_nrmse": he["mean_nrmse"],
        })
    pd.DataFrame(rows).to_csv(out / "detector_aware_model_comparison.csv", index=False)

    np.savez_compressed(
        out / "detector_aware_residual_stacks.npz",
        measured_beam_frame=data.astype(np.float32),
        direct_nominal=direct.astype(np.float32),
        detector_nominal=detector_nominal.astype(np.float32),
        slm2_positive_error=predictions["slm2"].astype(np.float32),
        axicon_input_positive_error=predictions["axicon_input"].astype(np.float32),
        axis_um=axis_um,
        z_relative_mm=z_rel,
        slm2_coefficients_rad=coeffs["slm2"],
        axicon_input_coefficients_rad=coeffs["axicon_input"],
    )

    rep = int(np.argmin(np.abs(z_rel + 10.0)))
    extent = [axis_um[0], axis_um[-1], axis_um[0], axis_um[-1]]
    displays = [
        (data, "BMG, beam-following frame"),
        (direct, "numerical field, no detector"),
        (detector_nominal, "nominal + 5.5 um detector"),
        (predictions[selected_plane], f"detector + fitted phase ({selected_plane})"),
    ]
    fig, axs = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    mid = len(axis_um) // 2
    for col, (stack, title) in enumerate(displays):
        axs[0, col].imshow(
            stack[rep], origin="lower", extent=extent, cmap=THERMAL,
            vmin=0, vmax=1, interpolation="nearest",
        )
        axs[0, col].set(title=title, xlabel="x (um)", ylabel="y (um)", aspect="equal")
        axs[1, col].plot(axis_um, data[rep, mid], lw=1.5, label="BMG")
        axs[1, col].plot(axis_um, stack[rep, mid], "--", lw=1.4, label=title)
        axs[1, col].set(
            xlim=(-130, 130), ylim=(0, 1.05), xlabel="x (um)",
            ylabel="plane-normalized intensity",
        )
        axs[1, col].grid(alpha=.2)
        axs[1, col].legend(fontsize=7)
    fig.suptitle("q=20 inverse with the measured 5.5 um camera sampling inside the forward model")
    savefig(fig, out / "11_detector_aware_residual_fit")

    labels = ["high-res nominal", "camera-aware nominal", "SLM2 phase", "axicon-input phase"]
    rvals = [row["heldout_mean_pearson_r"] for row in rows]
    evals = [row["heldout_mean_nrmse"] for row in rows]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    axs[0].bar(labels, rvals)
    axs[0].set(title="Held-out intensity correlation", ylabel="Pearson r")
    axs[1].bar(labels, evals)
    axs[1].set(title="Held-out intensity error", ylabel="normalized RMSE")
    for ax in axs:
        ax.tick_params(axis="x", rotation=18)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Odd z planes were excluded from detector-aware phase fitting")
    savefig(fig, out / "12_detector_aware_heldout_metrics")

    summary = {
        "comparison_frame": "measured affine beam/camera walk removed before optical residual fitting",
        "detector_model": {
            "pixel_pitch_um": float(PIXEL_M * 1e6),
            "pixel_shape": "square",
            "pixel_area_quadrature": "3x3 midpoint samples",
            "additional_free_blur_fitted": False,
            "display_interpolation_creates_measurement_information": False,
        },
        "high_resolution_nominal": {"train": direct_train, "heldout": direct_held},
        "detector_aware_nominal": {"train": detector_train, "heldout": detector_held},
        "phase_models": plane_results,
        "selected_phase_plane_by_training_nrmse": selected_plane,
        "selected_heldout_directional_improvement": directional_support,
        "selected_heldout_strong_support": strong_support,
        "strong_support_rule": "held-out Pearson gain >=0.02 and normalized RMSE reduction >=0.003",
        "hardware_phase_map_emitted": False,
        "reason_hardware_map_blocked": (
            "positive-error recreation must be separated from camera response and the SLM2/input-plane "
            "coordinate, conjugacy, parity and 1030-nm LUT remain uncalibrated"
        ),
    }
    (out / "detector_aware_residual_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir", type=Path,
        default=EXP / "outputs" / "digital_twin_correction",
    )
    parser.add_argument(
        "--out", type=Path,
        default=ROOT / "outputs" / "validation" / "q20_detector_aware_residual",
    )
    args = parser.parse_args()
    run(args.source_dir, args.out)


if __name__ == "__main__":
    main()
