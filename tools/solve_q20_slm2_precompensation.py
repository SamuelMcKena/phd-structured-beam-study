"""Model-internal SLM2 precompensation for the held-out-supported q=20 residual.

The detector-aware inverse recovered a low-order phase at the selected-order
field immediately before the axicon.  That phase is a diagnosis plane, not the
actuator plane.  This script therefore solves a separate inverse problem:
choose a zero-winding low-order SLM2 phase whose propagation through the
explicit carrier + 4F + fixed +1 iris produces the *conjugate* of the recovered
axicon-input phase.

The closure test is deliberately non-trivial.  The recovered positive error is
applied at the axicon input while the correction is applied upstream at SLM2;
both then pass through their respective physical parts of the digital twin.
The corrected prediction is compared with the nominal detector-aware q=20
field.  No native hardware mask is emitted because SLM2 coordinate registration
and the 1030-nm LUT are not yet bench calibrated.
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
    FIT_WINDOW_M,
    PIXEL_M,
    Q,
    RELAY_N,
    FIT_N,
    SystemErrorConfig,
    propagate_route,
)
from vbb_study.digital_twin.detector_response import plane_normalise, sample_camera_response
from vbb_study.digital_twin.observation_frame import fit_affine_trajectory, shift_stack_by_trajectory
from vbb_study.digital_twin.residual_phase_fit import angular_phase_from_coefficients
from vbb_study.digital_twin.vortex_system_route import build_multirate_system_route
from vbb_study.viz_fields import phase_winding

EPS = np.finfo(float).tiny
SLM_MODES = (1, 2, 3, 4)
THERMAL = "inferno"


def route(config: SystemErrorConfig, *, slm2_phase: np.ndarray | None = None,
          axicon_phase: np.ndarray | None = None) -> dict:
    return build_multirate_system_route(
        f"V{Q}",
        relay_grid_n=RELAY_N,
        propagation_grid_n=FIT_N,
        window_m=FIT_WINDOW_M,
        config=config,
        slm2_static_phase_map_rad=slm2_phase,
        axicon_input_phase_map_rad=axicon_phase,
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


def intensity_score(a: np.ndarray, b: np.ndarray, axis_um: np.ndarray) -> dict:
    X, Y = np.meshgrid(axis_um, axis_um, indexing="xy")
    roi = np.hypot(X, Y) <= 145.0
    an, bn = plane_normalise(a), plane_normalise(b)
    rs, es = [], []
    for ia, ib in zip(an, bn):
        av, bv = ia[roi], ib[roi]
        rs.append(float(np.corrcoef(av, bv)[0, 1]))
        es.append(float(np.sqrt(np.mean((av - bv) ** 2))))
    return {"mean_pearson_r": float(np.mean(rs)), "mean_nrmse": float(np.mean(es)),
            "per_plane_pearson_r": rs, "per_plane_nrmse": es}


def complex_overlap(a: np.ndarray, b: np.ndarray, weight: np.ndarray) -> float:
    w = np.asarray(weight, float)
    aa, bb = np.asarray(a, complex), np.asarray(b, complex)
    num = abs(np.sum(w * np.conj(aa) * bb))
    den = np.sqrt(np.sum(w * np.abs(aa) ** 2) * np.sum(w * np.abs(bb) ** 2))
    return float(num / max(float(den), EPS))


def solve_relay_phase(config: SystemErrorConfig, error_phase: np.ndarray) -> tuple[np.ndarray, dict]:
    base = route(config)
    f0 = np.asarray(base["field_on_axicon_plane"], complex)
    target = f0 * np.exp(-1j * np.asarray(error_phase, float))
    relay_grid = base["relay_route"]["grid"]
    theta_slm = np.arctan2(np.asarray(relay_grid["Y"], float), np.asarray(relay_grid["X"], float))
    x = np.asarray(base["grid"]["x"], float)
    X, Y = np.meshgrid(x, x, indexing="xy")
    R = np.hypot(X, Y)
    intensity = np.abs(f0) ** 2
    weight = intensity / max(float(np.max(intensity)), EPS)
    mask = (R <= 2.4e-3) & (weight >= 2e-3)
    # Decimate only the linear solve; every nonlinear route evaluation remains
    # full-resolution.  This keeps the complex least-squares matrix tractable.
    ids = np.zeros_like(mask)
    ids[::6, ::6] = True
    fit_mask = mask & ids
    w = np.sqrt(weight[fit_mask])

    coeff = np.zeros(2 * len(SLM_MODES), float)
    history = []
    delta = 0.14
    for iteration in range(2):
        current_phase = angular_phase_from_coefficients(theta_slm, coeff, modes=SLM_MODES)
        current_route = route(config, slm2_phase=current_phase)
        current = np.asarray(current_route["field_on_axicon_plane"], complex)
        residual = target - current
        columns = []
        for j in range(coeff.size):
            cp, cm = coeff.copy(), coeff.copy()
            cp[j] += delta
            cm[j] -= delta
            pp = angular_phase_from_coefficients(theta_slm, cp, modes=SLM_MODES)
            pm = angular_phase_from_coefficients(theta_slm, cm, modes=SLM_MODES)
            fp = np.asarray(route(config, slm2_phase=pp)["field_on_axicon_plane"], complex)
            fm = np.asarray(route(config, slm2_phase=pm)["field_on_axicon_plane"], complex)
            columns.append((fp - fm) / (2.0 * delta))
        C = np.column_stack([col[fit_mask] for col in columns])
        rv = residual[fit_mask]
        A = np.vstack([w[:, None] * C.real, w[:, None] * C.imag])
        b = np.concatenate([w * rv.real, w * rv.imag])
        ridge = 2e-3
        step = np.linalg.solve(A.T @ A + ridge * np.eye(A.shape[1]), A.T @ b)
        step = np.clip(step, -0.40, 0.40)
        coeff = np.clip(coeff + step, -0.90, 0.90)
        trial_phase = angular_phase_from_coefficients(theta_slm, coeff, modes=SLM_MODES)
        trial = np.asarray(route(config, slm2_phase=trial_phase)["field_on_axicon_plane"], complex)
        corrected_with_error = trial * np.exp(1j * error_phase)
        history.append({
            "iteration": iteration + 1,
            "coefficients_rad": coeff.tolist(),
            "target_field_overlap": complex_overlap(trial[mask], target[mask], weight[mask]),
            "closure_field_overlap": complex_overlap(corrected_with_error[mask], f0[mask], weight[mask]),
        })
        delta = 0.09

    phase = angular_phase_from_coefficients(theta_slm, coeff, modes=SLM_MODES)
    solved_route = route(config, slm2_phase=phase)
    fsolve = np.asarray(solved_route["field_on_axicon_plane"], complex)
    closed = fsolve * np.exp(1j * error_phase)
    # Remove one global piston before reporting phase RMS.
    phasor = closed[mask] * np.conj(f0[mask])
    piston = np.angle(np.sum(weight[mask] * phasor))
    phase_error = np.angle(phasor * np.exp(-1j * piston))
    amplitude_ratio = np.abs(closed[mask]) / np.maximum(np.abs(f0[mask]), EPS)
    summary = {
        "slm_modes": list(SLM_MODES),
        "coefficient_order": [item for m in SLM_MODES for item in (f"cos{m}", f"sin{m}")],
        "coefficients_rad": coeff.tolist(),
        "iterations": history,
        "target_field_overlap": complex_overlap(fsolve[mask], target[mask], weight[mask]),
        "closure_field_overlap": complex_overlap(closed[mask], f0[mask], weight[mask]),
        "closure_phase_rms_rad": float(np.sqrt(np.average(phase_error ** 2, weights=weight[mask]))),
        "closure_amplitude_ratio_rms_from_unity": float(np.sqrt(np.average((amplitude_ratio - 1.0) ** 2, weights=weight[mask]))),
        "fit_support_fraction": float(np.mean(mask)),
        "hardware_coordinate_map_used": False,
    }
    return phase, summary


def savefig(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def run(source_dir: Path, candidate_json: Path, out: Path) -> dict:
    source_dir, candidate_json, out = Path(source_dir), Path(candidate_json), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    candidate = json.loads(candidate_json.read_text(encoding="utf-8"))
    if not candidate.get("heldout_support_pass", False):
        raise RuntimeError("refusing SLM2 closure for a residual that failed held-out recreation")

    d = np.load(source_dir / "rerender_arrays.npz")
    measured = plane_normalise(np.asarray(d["measured"], float))
    axis_um = np.asarray(d["axis_um"], float)
    z_rel = np.asarray(d["z_relative_mm"], float)
    path = pd.read_csv(source_dir / "measured_beam_path.csv")
    yx = path[["y_relative_um", "x_relative_um"]].to_numpy(float)
    affine = fit_affine_trajectory(z_rel, yx, centre_fit=True)
    measured_bf = plane_normalise(shift_stack_by_trajectory(measured, axis_um, affine.fitted_yx, inverse=True))

    source_summary = json.loads((source_dir / "run_summary.json").read_text(encoding="utf-8"))
    scale = float(source_summary["effective_axicon"]["effective_scale_relative_to_repository_2deg_assumption"])
    zscan = pd.read_csv(source_dir / "full_route_z_registration_scan.csv")
    z0 = float(zscan.loc[zscan.selected.astype(bool), "value"].iloc[0])
    z_abs = (z0 + z_rel) * 1e-3
    config = SystemErrorConfig(
        fourf=FourFError(iris_radius_scale=1.0),
        axicon=AxiconError(base_angle_scale=scale),
    )

    base_route = route(config)
    x_ax = np.asarray(base_route["grid"]["x"], float)
    X, Y = np.meshgrid(x_ax, x_ax, indexing="xy")
    theta_ax = np.arctan2(Y, X)
    error_phase = angular_phase_from_coefficients(
        theta_ax,
        np.asarray(candidate["coefficients_rad"], float),
        modes=tuple(candidate["angular_modes"]),
    )
    slm2_phase, relay_summary = solve_relay_phase(config, error_phase)

    # Full longitudinal closure: same positive error used by the held-out
    # recreation, but correction is applied upstream at SLM2 and passes through
    # the explicit 4F selected-order route before it meets the error plane.
    nominal_native, nominal_meta = propagate_route(config, z_abs)
    error_native, error_meta = propagate_route(config, z_abs, phase_axicon_input=error_phase)
    corrected_native, corrected_meta = propagate_route(
        config, z_abs, phase_slm2=slm2_phase, phase_axicon_input=error_phase,
    )
    nominal = detector_render(nominal_native, nominal_meta["x_m"], axis_um)
    error_prediction = detector_render(error_native, error_meta["x_m"], axis_um)
    corrected = detector_render(corrected_native, corrected_meta["x_m"], axis_um)
    del nominal_native, error_native, corrected_native

    error_to_nominal = intensity_score(error_prediction, nominal, axis_um)
    corrected_to_nominal = intensity_score(corrected, nominal, axis_um)
    measured_to_error = intensity_score(error_prediction, measured_bf, axis_um)

    nominal_route = route(config)
    corrected_route = route(config, slm2_phase=slm2_phase, axicon_phase=error_phase)
    winding_nominal = {}
    winding_corrected = {}
    for radius_mm in (0.70, 1.05, 1.40):
        key = f"radius_{radius_mm:.2f}_mm"
        winding_nominal[key] = float(phase_winding(
            nominal_route["post_axicon"], nominal_route["grid"], radius_mm * 1e-3, n_phi=720,
        ))
        winding_corrected[key] = float(phase_winding(
            corrected_route["post_axicon"], corrected_route["grid"], radius_mm * 1e-3, n_phi=720,
        ))
    winding_preserved = bool(all(
        abs(winding_corrected[k] - winding_nominal[k]) <= 0.25 for k in winding_nominal
    ))

    np.savez_compressed(
        out / "model_coordinate_slm2_precompensation.npz",
        slm2_phase_rad=slm2_phase.astype(np.float32),
        relay_x_m=np.asarray(base_route["relay_route"]["grid"]["x"], float),
        error_phase_axicon_input_rad=error_phase.astype(np.float32),
        axicon_x_m=x_ax,
        z_relative_mm=z_rel,
        display_axis_um=axis_um,
        nominal_detector=nominal.astype(np.float32),
        positive_error_detector=error_prediction.astype(np.float32),
        predicted_corrected_detector=corrected.astype(np.float32),
        measured_beam_frame=measured_bf.astype(np.float32),
    )

    rep = int(np.argmin(np.abs(z_rel + 10.0)))
    extent = [axis_um[0], axis_um[-1], axis_um[0], axis_um[-1]]
    displays = [
        (measured_bf, "measured BMG, beam frame"),
        (error_prediction, "recovered positive error"),
        (corrected, "predicted after SLM2 precompensation"),
        (nominal, "nominal q=20 target"),
    ]
    fig, axs = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    mid = len(axis_um) // 2
    for col, (stack, title) in enumerate(displays):
        axs[0, col].imshow(stack[rep], origin="lower", extent=extent, cmap=THERMAL,
                           vmin=0, vmax=1, interpolation="nearest")
        axs[0, col].set(title=title, xlabel="x (um)", ylabel="y (um)", aspect="equal")
        reference = nominal if col >= 2 else measured_bf
        ref_label = "nominal" if col >= 2 else "BMG"
        axs[1, col].plot(axis_um, reference[rep, mid], lw=1.5, label=ref_label)
        axs[1, col].plot(axis_um, stack[rep, mid], "--", lw=1.4, label=title)
        axs[1, col].set(xlim=(-130, 130), ylim=(0, 1.05), xlabel="x (um)",
                        ylabel="plane-normalized intensity")
        axs[1, col].grid(alpha=.2); axs[1, col].legend(fontsize=7)
    fig.suptitle("q=20 model closure: recovered axicon-input error corrected from upstream SLM2")
    savefig(fig, out / "13_slm2_precompensation_closure")

    fig, axs = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)
    relay_x = np.asarray(base_route["relay_route"]["grid"]["x"], float) * 1e3
    ax_x = x_ax * 1e3
    axs[0].imshow(slm2_phase, origin="lower", extent=[relay_x[0], relay_x[-1], relay_x[0], relay_x[-1]],
                  cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axs[0].set(title="model-coordinate SLM2 precompensation", xlabel="x (mm)", ylabel="y (mm)")
    axs[1].imshow(error_phase, origin="lower", extent=[ax_x[0], ax_x[-1], ax_x[0], ax_x[-1]],
                  cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axs[1].set(title="recovered positive phase at axicon input", xlabel="x (mm)", ylabel="y (mm)")
    labels = ["error vs nominal", "corrected vs nominal"]
    axs[2].bar(labels, [error_to_nominal["mean_nrmse"], corrected_to_nominal["mean_nrmse"]])
    axs[2].set(title="full-z closure error", ylabel="normalized RMSE")
    axs[2].tick_params(axis="x", rotation=15); axs[2].grid(axis="y", alpha=.2)
    fig.suptitle("Correction is solved through the explicit 4F relay; this is not a hardware-ready SLM mask")
    savefig(fig, out / "14_slm2_precompensation_phase_and_metrics")

    summary = {
        "status": "model_internal_cross_plane_correction_closure",
        "source_candidate": candidate,
        "slm2_inverse": relay_summary,
        "positive_error_recreation_vs_measured": measured_to_error,
        "positive_error_vs_nominal": error_to_nominal,
        "predicted_corrected_vs_nominal": corrected_to_nominal,
        "winding_nominal": winding_nominal,
        "winding_corrected": winding_corrected,
        "winding_preserved_within_0p25_turn": winding_preserved,
        "closure_pass": bool(
            corrected_to_nominal["mean_pearson_r"] >= 0.95
            and corrected_to_nominal["mean_nrmse"] <= 0.05
            and winding_preserved
        ),
        "closure_rule": "corrected vs nominal mean Pearson r >=0.95, NRMSE <=0.05, and q winding preserved within 0.25 turn",
        "hardware_ready": False,
        "hardware_blockers": candidate["hardware_blockers"],
        "hardware_mask_emitted": False,
    }
    (out / "slm2_precompensation_closure_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path,
                        default=EXP / "outputs" / "digital_twin_correction")
    parser.add_argument("--candidate-json", type=Path,
                        default=EXP / "candidates" / "q20_detector_aware_axicon_residual_candidate.json")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "outputs" / "validation" / "q20_slm2_precompensation")
    args = parser.parse_args()
    run(args.source_dir, args.candidate_json, args.out)


if __name__ == "__main__":
    main()
