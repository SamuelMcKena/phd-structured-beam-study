"""Detector-domain SLM2 precompensation for the supported q=20 residual.

The first cross-plane closure solved for the conjugate *complex field* at the
axicon input.  That produced excellent phase agreement but substantial amplitude
error after the finite +1-order iris, which in turn made the propagated main
ring asymmetric and washed out the surrounding Bessel rings.

This v2 solve targets the quantity we actually care about: detector-resolved
intensity over the longitudinal stack.  A compact zero-winding SLM2 phase basis
is propagated through the explicit carrier + 4F + fixed iris, combined with the
held-out-supported positive error at the axicon input, propagated to all z
planes, integrated over the measured 5.5 um camera pixels, and compared with the
nominal q=20 detector prediction.

The fit uses alternating z planes and is scored on the untouched planes.  The
loss contains a non-zero floor so the radial sidelobes matter rather than the
bright principal ring dominating the solve.  The result remains a model-space
precompensation candidate, not a hardware-ready SLM mask.
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
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "notebooks" / "experimental" / "axicon_aberration_correction"
for path in (ROOT, EXP):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from real_bmg_digital_twin_correction import (
    AxiconError, FourFError, FIT_WINDOW_M, PIXEL_M, Q, RELAY_N, FIT_N,
    SystemErrorConfig, propagate_route,
)
from vbb_study.digital_twin.detector_response import plane_normalise, sample_camera_response
from vbb_study.digital_twin.residual_phase_fit import angular_phase_from_coefficients
from vbb_study.digital_twin.vortex_system_route import build_multirate_system_route
from vbb_study.viz_fields import phase_winding

EPS = np.finfo(float).tiny
THERMAL = "inferno"
ANGULAR_MODES = (1, 2, 3, 4)
RADIAL_ORDERS = (0, 1)


def route(config: SystemErrorConfig, *, slm2_phase=None, axicon_phase=None):
    return build_multirate_system_route(
        f"V{Q}", relay_grid_n=RELAY_N, propagation_grid_n=FIT_N,
        window_m=FIT_WINDOW_M, config=config,
        slm2_static_phase_map_rad=slm2_phase,
        axicon_input_phase_map_rad=axicon_phase,
    )


def detector_render(native, x_m, axis_um):
    shown, _ = sample_camera_response(
        native, np.asarray(x_m, float), np.asarray(axis_um, float)*1e-6,
        pixel_pitch_m=PIXEL_M, quadrature_n=3,
    )
    return plane_normalise(shown)


def phase_basis(relay_grid):
    X = np.asarray(relay_grid["X"], float)
    Y = np.asarray(relay_grid["Y"], float)
    theta = np.arctan2(Y, X)
    R = np.hypot(X, Y)
    # Beam illumination is about 2 mm radius; scale radial terms around this
    # support and clip only the basis coordinate, not the optical field.
    rho = np.clip(R / 2.0e-3, 0.0, 1.6)
    basis, names = [], []
    # Radially varying angular terms give the phase-only actuator enough freedom
    # to compensate amplitude changes induced by finite selected-order filtering.
    for m in ANGULAR_MODES:
        for p in RADIAL_ORDERS:
            radial = rho**p
            basis.append(radial*np.cos(m*theta)); names.append(f"r{p}_cos{m}")
            basis.append(radial*np.sin(m*theta)); names.append(f"r{p}_sin{m}")
    # Two zero-winding radial terms allow small longitudinal/radial cleanup
    # without changing the programmed q=20 topological charge.
    basis.append(rho**2); names.append("r2_axisymmetric")
    basis.append(rho**4); names.append("r4_axisymmetric")
    return np.stack(basis), names


def phase_from_coefficients(basis, coeff):
    return np.tensordot(np.asarray(coeff, float), np.asarray(basis, float), axes=(0, 0))


def intensity_metrics(predicted, target, axis_um, indices):
    X, Y = np.meshgrid(axis_um, axis_um, indexing="xy")
    roi = np.hypot(X, Y) <= 145.0
    pn, tn = plane_normalise(predicted), plane_normalise(target)
    rows = []
    for iz in np.asarray(indices, int):
        a, b = pn[iz][roi], tn[iz][roi]
        rows.append((float(np.corrcoef(a, b)[0,1]), float(np.sqrt(np.mean((a-b)**2)))))
    return float(np.mean([r for r,_ in rows])), float(np.mean([e for _,e in rows]))


def radial_profile(image, axis_um, dr_um=2.0, rmax_um=140.0):
    X, Y = np.meshgrid(axis_um, axis_um, indexing="xy")
    R = np.hypot(X, Y)
    edges = np.arange(0.0, rmax_um+dr_um, dr_um)
    idx = np.digitize(R.ravel(), edges)-1
    good = (idx >= 0) & (idx < len(edges)-1)
    sums = np.bincount(idx[good], weights=np.asarray(image, float).ravel()[good], minlength=len(edges)-1)
    num = np.bincount(idx[good], minlength=len(edges)-1)
    return 0.5*(edges[:-1]+edges[1:]), sums/np.maximum(num, 1)


def structure_metrics(stack, target, axis_um, indices):
    X, Y = np.meshgrid(axis_um, axis_um, indexing="xy")
    R = np.hypot(X, Y)
    sn, tn = plane_normalise(stack), plane_normalise(target)
    radial_corr, side_rmse, ring_cv, peak_imbalance = [], [], [], []
    mid = len(axis_um)//2
    for iz in np.asarray(indices, int):
        rr, ps = radial_profile(sn[iz], axis_um)
        _, pt = radial_profile(tn[iz], axis_um)
        rp = float(rr[(rr >= 20) & (rr <= 70)][np.argmax(pt[(rr >= 20) & (rr <= 70)])])
        maskp = (rr >= 10) & (rr <= 135)
        radial_corr.append(float(np.corrcoef(ps[maskp], pt[maskp])[0,1]))
        side = maskp & (np.abs(rr-rp) >= 8.0)
        side_rmse.append(float(np.sqrt(np.mean((ps[side]-pt[side])**2))))
        ann = np.abs(R-rp) <= 5.5
        vals = sn[iz][ann]
        ring_cv.append(float(np.std(vals)/max(np.mean(vals), EPS)))
        cut = sn[iz, mid]
        left = np.max(cut[(axis_um >= -70) & (axis_um <= -20)])
        right = np.max(cut[(axis_um >= 20) & (axis_um <= 70)])
        peak_imbalance.append(float(abs(left-right)/max(0.5*(left+right), EPS)))
    return {
        "mean_radial_profile_corr": float(np.mean(radial_corr)),
        "mean_sidelobe_profile_rmse": float(np.mean(side_rmse)),
        "mean_principal_ring_azimuth_cv": float(np.mean(ring_cv)),
        "mean_opposite_peak_imbalance": float(np.mean(peak_imbalance)),
    }


def weighted_system(current, derivatives, target, axis_um, indices):
    X, Y = np.meshgrid(axis_um, axis_um, indexing="xy")
    R = np.hypot(X, Y)
    roi = R <= 145.0
    cur, tar = plane_normalise(current), plane_normalise(target)
    J, y = [], []
    for iz in np.asarray(indices, int):
        t = tar[iz][roi]
        residual = t-cur[iz][roi]
        # A large floor deliberately keeps sidelobes/radial rings in the solve.
        w = np.sqrt(0.55 + 0.45*np.sqrt(np.clip(t, 0.0, 1.0)))
        J.append(w[:,None]*np.column_stack([d[iz][roi] for d in derivatives]))
        y.append(w*residual)
    return np.vstack(J), np.concatenate(y)


def objective_tuple(stack, target, axis_um, indices):
    r, e = intensity_metrics(stack, target, axis_um, indices)
    s = structure_metrics(stack, target, axis_um, indices)
    # Lower is better. Keep image fidelity dominant while explicitly penalizing
    # loss of radial rings and main-ring asymmetry.
    value = (
        e + 0.12*(1.0-r) +
        0.20*s["mean_sidelobe_profile_rmse"] +
        0.025*s["mean_principal_ring_azimuth_cv"] +
        0.025*s["mean_opposite_peak_imbalance"]
    )
    return float(value), r, e, s


def savefig(fig, stem):
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def run(source_dir: Path, candidate_json: Path, out: Path):
    source_dir, candidate_json, out = Path(source_dir), Path(candidate_json), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    candidate = json.loads(candidate_json.read_text(encoding="utf-8"))
    if not candidate.get("heldout_support_pass", False):
        raise RuntimeError("residual phase candidate did not pass held-out recreation")

    d = np.load(source_dir/"rerender_arrays.npz")
    axis_um = np.asarray(d["axis_um"], float)
    z_rel = np.asarray(d["z_relative_mm"], float)
    source_summary = json.loads((source_dir/"run_summary.json").read_text(encoding="utf-8"))
    scale = float(source_summary["effective_axicon"]["effective_scale_relative_to_repository_2deg_assumption"])
    zscan = pd.read_csv(source_dir/"full_route_z_registration_scan.csv")
    z0 = float(zscan.loc[zscan.selected.astype(bool), "value"].iloc[0])
    z_abs = (z0+z_rel)*1e-3
    config = SystemErrorConfig(
        fourf=FourFError(iris_radius_scale=1.0),
        axicon=AxiconError(base_angle_scale=scale),
    )

    base = route(config)
    xax = np.asarray(base["grid"]["x"], float)
    Xax, Yax = np.meshgrid(xax, xax, indexing="xy")
    theta_ax = np.arctan2(Yax, Xax)
    error_phase = angular_phase_from_coefficients(
        theta_ax, np.asarray(candidate["coefficients_rad"], float),
        modes=tuple(candidate["angular_modes"]),
    )
    basis, names = phase_basis(base["relay_route"]["grid"])

    nominal_native, nominal_meta = propagate_route(config, z_abs)
    target = detector_render(nominal_native, nominal_meta["x_m"], axis_um)
    del nominal_native

    train = np.arange(0, len(z_abs), 2, dtype=int)
    held = np.arange(1, len(z_abs), 2, dtype=int)
    coeff = np.zeros(len(names), float)
    history = []
    delta = 0.12

    def simulate(c):
        phase = phase_from_coefficients(basis, c)
        native, meta = propagate_route(config, z_abs, phase_slm2=phase, phase_axicon_input=error_phase)
        shown = detector_render(native, meta["x_m"], axis_um)
        del native
        return shown

    current = simulate(coeff)
    for iteration in range(3):
        derivatives = []
        for j in range(len(coeff)):
            cp, cm = coeff.copy(), coeff.copy()
            cp[j] += delta; cm[j] -= delta
            plus, minus = simulate(cp), simulate(cm)
            derivatives.append((plus-minus)/(2.0*delta))
        J, y = weighted_system(current, derivatives, target, axis_um, train)
        ridge = 8e-3
        step = np.linalg.solve(J.T@J + ridge*np.eye(J.shape[1]), J.T@y)
        step = np.clip(step, -0.32, 0.32)
        base_obj = objective_tuple(current, target, axis_um, train)[0]
        chosen = None
        for alpha in (1.0, 0.6, 0.3, 0.15):
            trial_c = np.clip(coeff + alpha*step, -1.15, 1.15)
            trial = simulate(trial_c)
            obj = objective_tuple(trial, target, axis_um, train)[0]
            if chosen is None or obj < chosen[0]:
                chosen = (obj, trial_c, trial, alpha)
        if chosen[0] >= base_obj:
            history.append({"iteration": iteration+1, "accepted": False, "objective": base_obj})
            break
        coeff, current = chosen[1], chosen[2]
        train_obj, tr_r, tr_e, tr_s = objective_tuple(current, target, axis_um, train)
        held_obj, he_r, he_e, he_s = objective_tuple(current, target, axis_um, held)
        history.append({
            "iteration": iteration+1, "accepted": True, "alpha": float(chosen[3]),
            "coefficients_rad": coeff.tolist(),
            "train_objective": train_obj, "train_r": tr_r, "train_nrmse": tr_e,
            "heldout_objective": held_obj, "heldout_r": he_r, "heldout_nrmse": he_e,
            "heldout_structure": he_s,
        })
        delta = 0.085

    corrected = current
    positive_native, positive_meta = propagate_route(config, z_abs, phase_axicon_input=error_phase)
    positive = detector_render(positive_native, positive_meta["x_m"], axis_um)
    del positive_native

    baseline_held = intensity_metrics(positive, target, axis_um, held)
    corrected_held = intensity_metrics(corrected, target, axis_um, held)
    baseline_structure = structure_metrics(positive, target, axis_um, held)
    corrected_structure = structure_metrics(corrected, target, axis_um, held)

    slm_phase = phase_from_coefficients(basis, coeff)
    nominal_route = route(config)
    corrected_route = route(config, slm2_phase=slm_phase, axicon_phase=error_phase)
    winding_nominal, winding_corrected = {}, {}
    for radius_mm in (0.70, 1.05, 1.40):
        key = f"radius_{radius_mm:.2f}_mm"
        winding_nominal[key] = float(phase_winding(nominal_route["post_axicon"], nominal_route["grid"], radius_mm*1e-3, n_phi=720))
        winding_corrected[key] = float(phase_winding(corrected_route["post_axicon"], corrected_route["grid"], radius_mm*1e-3, n_phi=720))
    winding_ok = bool(all(abs(winding_corrected[k]-winding_nominal[k]) <= 0.25 for k in winding_nominal))

    # Stronger closure criteria than v1: intensity fidelity plus radial/ring structure.
    closure_pass = bool(
        corrected_held[0] >= 0.95 and corrected_held[1] <= 0.05 and winding_ok and
        corrected_structure["mean_radial_profile_corr"] >= 0.97 and
        corrected_structure["mean_opposite_peak_imbalance"] <= 0.08
    )

    summary = {
        "status": "detector_domain_model_internal_precompensation_v2",
        "basis_names": names,
        "coefficients_rad": coeff.tolist(),
        "iterations": history,
        "heldout_positive_error_vs_nominal": {"mean_pearson_r": baseline_held[0], "mean_nrmse": baseline_held[1], **baseline_structure},
        "heldout_corrected_vs_nominal": {"mean_pearson_r": corrected_held[0], "mean_nrmse": corrected_held[1], **corrected_structure},
        "winding_nominal": winding_nominal,
        "winding_corrected": winding_corrected,
        "winding_preserved_within_0p25_turn": winding_ok,
        "closure_pass": closure_pass,
        "closure_rule": "held-out r>=0.95, NRMSE<=0.05, radial profile r>=0.97, opposite-peak imbalance<=0.08, q winding preserved",
        "hardware_ready": False,
        "hardware_blockers": candidate.get("hardware_blockers", []),
    }
    (out/"detector_domain_slm2_closure_v2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(
        out/"detector_domain_slm2_closure_v2.npz",
        slm2_phase_rad=slm_phase.astype(np.float32), error_phase_axicon_input_rad=error_phase.astype(np.float32),
        z_relative_mm=z_rel, axis_um=axis_um,
        nominal_detector=target.astype(np.float32), positive_error_detector=positive.astype(np.float32),
        predicted_corrected_detector=corrected.astype(np.float32),
    )

    rep = int(np.argmin(np.abs(z_rel+10.0)))
    extent = [axis_um[0], axis_um[-1], axis_um[0], axis_um[-1]]
    mid = len(axis_um)//2
    fig, axs = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for col, (stack, title) in enumerate(((target, "nominal q=20 target"), (positive, "recovered error prediction"), (corrected, "SLM2 precompensation v2"))):
        axs[0,col].imshow(stack[rep], origin="lower", extent=extent, cmap=THERMAL, vmin=0, vmax=1, interpolation="nearest")
        axs[0,col].set(title=title, xlabel="x (um)", ylabel="y (um)", aspect="equal")
        axs[1,col].plot(axis_um, target[rep,mid], lw=1.6, label="nominal")
        axs[1,col].plot(axis_um, stack[rep,mid], "--", lw=1.4, label=title)
        axs[1,col].set(xlim=(-140,140), ylim=(0,1.05), xlabel="x (um)", ylabel="normalized intensity")
        axs[1,col].grid(alpha=.2); axs[1,col].legend(fontsize=7)
    fig.suptitle("q=20 detector-domain precompensation: preserve the main ring and radial sidelobes")
    savefig(fig, out/"15_detector_domain_slm2_closure_v2")

    fig, axs = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    rr, pnom = radial_profile(target[rep], axis_um)
    _, perr = radial_profile(positive[rep], axis_um)
    _, pcorr = radial_profile(corrected[rep], axis_um)
    axs[0].plot(rr, pnom, label="nominal", lw=1.7)
    axs[0].plot(rr, perr, label="error", lw=1.2)
    axs[0].plot(rr, pcorr, label="corrected v2", lw=1.4)
    axs[0].set(xlabel="radius (um)", ylabel="azimuthal mean intensity", xlim=(0,140), title="Radial ring structure")
    axs[0].grid(alpha=.2); axs[0].legend(fontsize=8)
    labels = ["error", "corrected v2"]
    axs[1].bar(labels, [baseline_structure["mean_opposite_peak_imbalance"], corrected_structure["mean_opposite_peak_imbalance"]])
    axs[1].set(title="Held-out left/right peak imbalance", ylabel="fraction")
    axs[2].bar(labels, [baseline_structure["mean_sidelobe_profile_rmse"], corrected_structure["mean_sidelobe_profile_rmse"]])
    axs[2].set(title="Held-out sidelobe-profile error", ylabel="RMSE")
    for ax in axs[1:]: ax.grid(axis="y", alpha=.2)
    fig.suptitle("Symmetry and sidelobe preservation are explicit closure metrics")
    savefig(fig, out/"16_detector_domain_ring_structure_metrics_v2")

    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=EXP/"outputs"/"digital_twin_correction")
    parser.add_argument("--candidate-json", type=Path, default=EXP/"candidates"/"q20_detector_aware_axicon_residual_candidate.json")
    parser.add_argument("--out", type=Path, default=ROOT/"outputs"/"validation"/"q20_detector_domain_slm2_v2")
    args = parser.parse_args()
    run(args.source_dir, args.candidate_json, args.out)

if __name__ == "__main__":
    main()
