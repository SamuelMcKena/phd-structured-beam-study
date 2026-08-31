"""Symmetry- and ring-constrained detector-domain SLM2 precompensation for q=20.

v2 restored overall detector-domain fidelity and radial sidelobes but allowed the
principal annulus to become more left/right imbalanced.  v3 therefore puts the
missing observables directly inside the inverse system instead of only checking
them after the solve:

* detector-resolved intensity over alternating training z planes;
* x/y mirror residuals relative to the nominal digital-twin target;
* azimuthally averaged radial-profile residuals so the secondary Bessel rings
  cannot be traded away for the principal ring;
* a richer but still zero-winding SLM2 basis with radial orders p=0,1,2.

The held-out z planes remain untouched until scoring.  Topological charge is
checked only on nominally valid q=20 loops; a phase-winding loop crossing a dark
zero is not treated as a meaningful topological diagnostic.
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
TOOLS = ROOT / "tools"
for path in (ROOT, EXP, TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from real_bmg_digital_twin_correction import (
    AxiconError, FourFError, FIT_WINDOW_M, Q, RELAY_N, FIT_N,
    SystemErrorConfig, propagate_route,
)
from vbb_study.digital_twin.residual_phase_fit import angular_phase_from_coefficients
from vbb_study.viz_fields import phase_winding
from optimize_q20_slm2_detector_closure_v2 import (
    route, detector_render, intensity_metrics, radial_profile, phase_from_coefficients, savefig,
)

EPS = np.finfo(float).tiny
THERMAL = "inferno"
ANGULAR_MODES = (1, 2, 3, 4)
RADIAL_ORDERS = (0, 1, 2)


def phase_basis_v3(relay_grid):
    X = np.asarray(relay_grid["X"], float)
    Y = np.asarray(relay_grid["Y"], float)
    theta = np.arctan2(Y, X)
    R = np.hypot(X, Y)
    rho = np.clip(R / 2.0e-3, 0.0, 1.6)
    basis, names = [], []
    for m in ANGULAR_MODES:
        for p in RADIAL_ORDERS:
            radial = rho ** p
            basis.append(radial*np.cos(m*theta)); names.append(f"r{p}_cos{m}")
            basis.append(radial*np.sin(m*theta)); names.append(f"r{p}_sin{m}")
    basis.append(rho**2); names.append("r2_axisymmetric")
    basis.append(rho**4); names.append("r4_axisymmetric")
    return np.stack(basis), names


def seed_coefficients(seed_json: Path, names: list[str]) -> np.ndarray:
    seed = json.loads(Path(seed_json).read_text(encoding="utf-8"))
    mapping = dict(zip(seed["basis_names"], seed["coefficients_rad"]))
    return np.asarray([float(mapping.get(name, 0.0)) for name in names], float)


def radial_bin_contract(axis_um, dr_um=2.0, rmax_um=140.0):
    X, Y = np.meshgrid(axis_um, axis_um, indexing="xy")
    R = np.hypot(X, Y)
    edges = np.arange(0.0, rmax_um+dr_um, dr_um)
    ids = np.digitize(R.ravel(), edges)-1
    good = (ids >= 0) & (ids < len(edges)-1)
    counts = np.bincount(ids[good], minlength=len(edges)-1).astype(float)
    radius = 0.5*(edges[:-1]+edges[1:])
    return R, ids, good, np.maximum(counts, 1.0), radius


def radial_vector(image, ids, good, counts):
    sums = np.bincount(ids[good], weights=np.asarray(image, float).ravel()[good], minlength=len(counts))
    return sums / counts


def structure_metrics_v3(stack, target, axis_um, indices):
    X, Y = np.meshgrid(axis_um, axis_um, indexing="xy")
    R = np.hypot(X, Y)
    sn = np.maximum(np.asarray(stack, float), 0.0)
    tn = np.maximum(np.asarray(target, float), 0.0)
    sn /= np.maximum(sn.reshape(sn.shape[0], -1).max(axis=1)[:, None, None], EPS)
    tn /= np.maximum(tn.reshape(tn.shape[0], -1).max(axis=1)[:, None, None], EPS)
    radial_corr, side_rmse, ring_cv, peak_imbalance, mirror_x, mirror_y = [], [], [], [], [], []
    mid = len(axis_um)//2
    for iz in np.asarray(indices, int):
        rr, ps = radial_profile(sn[iz], axis_um)
        _, pt = radial_profile(tn[iz], axis_um)
        sel = (rr >= 20) & (rr <= 70)
        rp = float(rr[sel][np.argmax(pt[sel])])
        maskp = (rr >= 10) & (rr <= 135)
        radial_corr.append(float(np.corrcoef(ps[maskp], pt[maskp])[0,1]))
        side = maskp & (np.abs(rr-rp) >= 8.0)
        side_rmse.append(float(np.sqrt(np.mean((ps[side]-pt[side])**2))))
        ann = np.abs(R-rp) <= 5.5
        vals = sn[iz][ann]
        ring_cv.append(float(np.std(vals)/max(np.mean(vals), EPS)))
        cut = sn[iz, mid]
        left = float(np.max(cut[(axis_um >= -70) & (axis_um <= -20)]))
        right = float(np.max(cut[(axis_um >= 20) & (axis_um <= 70)]))
        peak_imbalance.append(abs(left-right)/max(0.5*(left+right), EPS))
        ann_roi = (R >= max(10.0, rp-28.0)) & (R <= min(135.0, rp+55.0))
        sx = (sn[iz] - sn[iz, :, ::-1]) - (tn[iz] - tn[iz, :, ::-1])
        sy = (sn[iz] - sn[iz, ::-1, :]) - (tn[iz] - tn[iz, ::-1, :])
        mirror_x.append(float(np.sqrt(np.mean(sx[ann_roi]**2))))
        mirror_y.append(float(np.sqrt(np.mean(sy[ann_roi]**2))))
    return {
        "mean_radial_profile_corr": float(np.mean(radial_corr)),
        "mean_sidelobe_profile_rmse": float(np.mean(side_rmse)),
        "mean_principal_ring_azimuth_cv": float(np.mean(ring_cv)),
        "mean_opposite_peak_imbalance": float(np.mean(peak_imbalance)),
        "max_opposite_peak_imbalance": float(np.max(peak_imbalance)),
        "mean_x_mirror_residual_rmse": float(np.mean(mirror_x)),
        "mean_y_mirror_residual_rmse": float(np.mean(mirror_y)),
    }


def augmented_system(current, derivatives, target, axis_um, indices):
    X, Y = np.meshgrid(axis_um, axis_um, indexing="xy")
    R, ids, good, counts, rr = radial_bin_contract(axis_um)
    roi = R <= 145.0
    ring_roi = (R >= 20.0) & (R <= 120.0)
    cur = np.maximum(np.asarray(current, float), 0.0)
    tar = np.maximum(np.asarray(target, float), 0.0)
    cur /= np.maximum(cur.reshape(cur.shape[0], -1).max(axis=1)[:, None, None], EPS)
    tar /= np.maximum(tar.reshape(tar.shape[0], -1).max(axis=1)[:, None, None], EPS)
    J, y = [], []
    for iz in np.asarray(indices, int):
        t = tar[iz][roi]
        residual = t-cur[iz][roi]
        w = np.sqrt(0.48 + 0.52*np.sqrt(np.clip(t, 0.0, 1.0)))
        J.append(w[:,None]*np.column_stack([d[iz][roi] for d in derivatives]))
        y.append(w*residual)

        for flip_axis, scale in ((1, 0.90), (0, 0.55)):
            if flip_axis == 1:
                c_diff = cur[iz]-cur[iz, :, ::-1]
                t_diff = tar[iz]-tar[iz, :, ::-1]
                dcols = [d[iz]-d[iz, :, ::-1] for d in derivatives]
            else:
                c_diff = cur[iz]-cur[iz, ::-1, :]
                t_diff = tar[iz]-tar[iz, ::-1, :]
                dcols = [d[iz]-d[iz, ::-1, :] for d in derivatives]
            J.append(scale*np.column_stack([dc[ring_roi] for dc in dcols]))
            y.append(scale*(t_diff-c_diff)[ring_roi])

        pc = radial_vector(cur[iz], ids, good, counts)
        pt = radial_vector(tar[iz], ids, good, counts)
        radial_sel = (rr >= 8.0) & (rr <= 138.0)
        dprof = np.column_stack([radial_vector(d[iz], ids, good, counts)[radial_sel] for d in derivatives])
        radial_scale = 12.0
        J.append(radial_scale*dprof)
        y.append(radial_scale*(pt-pc)[radial_sel])
    return np.vstack(J), np.concatenate(y)


def objective(stack, target, axis_um, indices):
    r, e = intensity_metrics(stack, target, axis_um, indices)
    s = structure_metrics_v3(stack, target, axis_um, indices)
    value = (
        e + 0.10*(1.0-r) +
        0.28*s["mean_sidelobe_profile_rmse"] +
        0.14*s["mean_opposite_peak_imbalance"] +
        0.05*s["max_opposite_peak_imbalance"] +
        0.08*s["mean_x_mirror_residual_rmse"] +
        0.04*s["mean_y_mirror_residual_rmse"]
    )
    return float(value), r, e, s


def regularization_diagonal(names):
    vals = []
    for name in names:
        if name.startswith("r2_") and not name.endswith("axisymmetric"):
            vals.append(3.0)
        elif name.startswith("r1_"):
            vals.append(1.6)
        elif "axisymmetric" in name:
            vals.append(2.2)
        else:
            vals.append(1.0)
    return np.asarray(vals, float)


def valid_winding_loops(nominal_route, corrected_route):
    radii_mm = np.arange(0.70, 1.51, 0.10)
    nom, cor, valid = {}, {}, []
    for radius_mm in radii_mm:
        key = f"radius_{radius_mm:.2f}_mm"
        wn = float(phase_winding(nominal_route["post_axicon"], nominal_route["grid"], radius_mm*1e-3, n_phi=720))
        wc = float(phase_winding(corrected_route["post_axicon"], corrected_route["grid"], radius_mm*1e-3, n_phi=720))
        nom[key], cor[key] = wn, wc
        if abs(wn-Q) <= 0.25:
            valid.append(key)
    preserved = bool(len(valid) >= 2 and all(abs(cor[k]-nom[k]) <= 0.25 for k in valid))
    return nom, cor, valid, preserved


def run(source_dir: Path, residual_json: Path, seed_json: Path, out: Path):
    source_dir, residual_json, seed_json, out = map(Path, (source_dir, residual_json, seed_json, out))
    out.mkdir(parents=True, exist_ok=True)
    candidate = json.loads(residual_json.read_text(encoding="utf-8"))
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
    config = SystemErrorConfig(fourf=FourFError(iris_radius_scale=1.0), axicon=AxiconError(base_angle_scale=scale))

    base = route(config)
    xax = np.asarray(base["grid"]["x"], float)
    Xax, Yax = np.meshgrid(xax, xax, indexing="xy")
    theta_ax = np.arctan2(Yax, Xax)
    error_phase = angular_phase_from_coefficients(theta_ax, np.asarray(candidate["coefficients_rad"], float), modes=tuple(candidate["angular_modes"]))
    basis, names = phase_basis_v3(base["relay_route"]["grid"])
    coeff = seed_coefficients(seed_json, names)

    nominal_native, nominal_meta = propagate_route(config, z_abs)
    target = detector_render(nominal_native, nominal_meta["x_m"], axis_um)
    del nominal_native

    positive_native, positive_meta = propagate_route(config, z_abs, phase_axicon_input=error_phase)
    positive = detector_render(positive_native, positive_meta["x_m"], axis_um)
    del positive_native

    train = np.arange(0, len(z_abs), 2, dtype=int)
    held = np.arange(1, len(z_abs), 2, dtype=int)

    def simulate(c):
        phase = phase_from_coefficients(basis, c)
        native, meta = propagate_route(config, z_abs, phase_slm2=phase, phase_axicon_input=error_phase)
        shown = detector_render(native, meta["x_m"], axis_um)
        del native
        return shown

    seed_stack = simulate(coeff)
    current = seed_stack
    history = []
    delta = 0.075
    reg_diag = regularization_diagonal(names)
    for iteration in range(3):
        derivatives = []
        for j in range(len(coeff)):
            cp, cm = coeff.copy(), coeff.copy()
            cp[j] += delta; cm[j] -= delta
            plus, minus = simulate(cp), simulate(cm)
            derivatives.append((plus-minus)/(2.0*delta))
        J, y = augmented_system(current, derivatives, target, axis_um, train)
        ridge = 1.4e-2
        lhs = J.T@J + ridge*np.diag(reg_diag)
        step = np.linalg.solve(lhs, J.T@y)
        step = np.clip(step, -0.22, 0.22)
        base_obj = objective(current, target, axis_um, train)[0]
        chosen = None
        for alpha in (1.0, 0.55, 0.30, 0.15, 0.075):
            trial_c = np.clip(coeff + alpha*step, -1.20, 1.20)
            trial = simulate(trial_c)
            obj = objective(trial, target, axis_um, train)[0]
            if chosen is None or obj < chosen[0]:
                chosen = (obj, trial_c, trial, alpha)
        if chosen[0] >= base_obj:
            history.append({"iteration": iteration+1, "accepted": False, "train_objective": base_obj})
            break
        coeff, current = chosen[1], chosen[2]
        tr = objective(current, target, axis_um, train)
        he = objective(current, target, axis_um, held)
        history.append({
            "iteration": iteration+1, "accepted": True, "alpha": float(chosen[3]),
            "coefficients_rad": coeff.tolist(),
            "train_objective": tr[0], "train_r": tr[1], "train_nrmse": tr[2], "train_structure": tr[3],
            "heldout_objective": he[0], "heldout_r": he[1], "heldout_nrmse": he[2], "heldout_structure": he[3],
        })
        delta = 0.055

    corrected = current
    seed_held = intensity_metrics(seed_stack, target, axis_um, held)
    corrected_held = intensity_metrics(corrected, target, axis_um, held)
    positive_held = intensity_metrics(positive, target, axis_um, held)
    target_structure = structure_metrics_v3(target, target, axis_um, held)
    seed_structure = structure_metrics_v3(seed_stack, target, axis_um, held)
    corrected_structure = structure_metrics_v3(corrected, target, axis_um, held)
    positive_structure = structure_metrics_v3(positive, target, axis_um, held)

    slm_phase = phase_from_coefficients(basis, coeff)
    nominal_route = route(config)
    corrected_route = route(config, slm2_phase=slm_phase, axicon_phase=error_phase)
    winding_nominal, winding_corrected, valid_loops, winding_ok = valid_winding_loops(nominal_route, corrected_route)

    closure_pass = bool(
        corrected_held[0] >= 0.95 and corrected_held[1] <= 0.05 and
        corrected_structure["mean_radial_profile_corr"] >= 0.995 and
        corrected_structure["mean_opposite_peak_imbalance"] <= 0.08 and
        corrected_structure["max_opposite_peak_imbalance"] <= 0.18 and
        winding_ok
    )

    summary = {
        "status": "symmetry_and_ring_constrained_model_internal_precompensation_v3",
        "source_seed": str(seed_json),
        "basis_names": names,
        "coefficients_rad": coeff.tolist(),
        "iterations": history,
        "heldout_positive_error_vs_nominal": {"mean_pearson_r": positive_held[0], "mean_nrmse": positive_held[1], **positive_structure},
        "heldout_v2_seed_vs_nominal": {"mean_pearson_r": seed_held[0], "mean_nrmse": seed_held[1], **seed_structure},
        "heldout_corrected_v3_vs_nominal": {"mean_pearson_r": corrected_held[0], "mean_nrmse": corrected_held[1], **corrected_structure},
        "heldout_nominal_self_structure": target_structure,
        "winding_nominal_all_tested_loops": winding_nominal,
        "winding_corrected_all_tested_loops": winding_corrected,
        "nominally_valid_q20_winding_loops": valid_loops,
        "winding_preserved_on_valid_loops": winding_ok,
        "closure_pass": closure_pass,
        "closure_rule": "held-out r>=0.95, NRMSE<=0.05, radial r>=0.995, mean opposite-peak imbalance<=0.08, max held-out imbalance<=0.18, q=20 preserved on nominally valid winding loops",
        "hardware_ready": False,
        "hardware_blockers": candidate.get("hardware_blockers", []),
    }
    (out/"detector_domain_slm2_closure_v3_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(
        out/"detector_domain_slm2_closure_v3.npz",
        slm2_phase_rad=slm_phase.astype(np.float32), error_phase_axicon_input_rad=error_phase.astype(np.float32),
        z_relative_mm=z_rel, axis_um=axis_um,
        nominal_detector=target.astype(np.float32), positive_error_detector=positive.astype(np.float32),
        v2_seed_detector=seed_stack.astype(np.float32), predicted_corrected_detector=corrected.astype(np.float32),
    )

    rep = int(np.argmin(np.abs(z_rel+10.0)))
    extent = [axis_um[0], axis_um[-1], axis_um[0], axis_um[-1]]
    mid = len(axis_um)//2
    fig, axs = plt.subplots(2, 4, figsize=(17.5, 8.2), constrained_layout=True)
    displays = ((target, "nominal q=20 target"), (positive, "recovered error prediction"), (seed_stack, "SLM2 v2 seed"), (corrected, "SLM2 symmetry-constrained v3"))
    for col, (stack, title) in enumerate(displays):
        axs[0,col].imshow(stack[rep], origin="lower", extent=extent, cmap=THERMAL, vmin=0, vmax=1, interpolation="nearest")
        axs[0,col].set(title=title, xlabel="x (um)", ylabel="y (um)", aspect="equal")
        axs[1,col].plot(axis_um, target[rep,mid], lw=1.6, label="nominal")
        axs[1,col].plot(axis_um, stack[rep,mid], "--", lw=1.4, label=title)
        axs[1,col].set(xlim=(-140,140), ylim=(0,1.05), xlabel="x (um)", ylabel="normalized intensity")
        axs[1,col].grid(alpha=.2); axs[1,col].legend(fontsize=7)
    fig.suptitle("q=20 SLM2 precompensation v3: symmetry and Bessel-ring structure inside the inverse")
    savefig(fig, out/"17_detector_domain_slm2_closure_v3")

    fig, axs = plt.subplots(1, 4, figsize=(18, 4.5), constrained_layout=True)
    rr, pnom = radial_profile(target[rep], axis_um)
    _, perr = radial_profile(positive[rep], axis_um)
    _, pseed = radial_profile(seed_stack[rep], axis_um)
    _, pcorr = radial_profile(corrected[rep], axis_um)
    for p, label in ((pnom,"nominal"),(perr,"error"),(pseed,"v2"),(pcorr,"v3")):
        axs[0].plot(rr,p,label=label,lw=1.4)
    axs[0].set(xlabel="radius (um)", ylabel="azimuthal mean intensity", xlim=(0,140), title="Radial Bessel-ring structure")
    axs[0].grid(alpha=.2); axs[0].legend(fontsize=8)
    labels = ["error","v2","v3","nominal"]
    vals = [positive_structure, seed_structure, corrected_structure, target_structure]
    axs[1].bar(labels,[v["mean_opposite_peak_imbalance"] for v in vals]); axs[1].set(title="Held-out mean L/R imbalance",ylabel="fraction")
    axs[2].bar(labels,[v["max_opposite_peak_imbalance"] for v in vals]); axs[2].set(title="Held-out worst L/R imbalance",ylabel="fraction")
    axs[3].bar(labels,[v["mean_sidelobe_profile_rmse"] for v in vals]); axs[3].set(title="Held-out sidelobe-profile error",ylabel="RMSE")
    for ax in axs[1:]: ax.grid(axis="y",alpha=.2)
    fig.suptitle("v3 does not get credit for overall correlation if it destroys symmetry or sidelobes")
    savefig(fig, out/"18_detector_domain_ring_structure_metrics_v3")

    fig, axs = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for col,(stack,title) in enumerate(((target,"nominal"),(positive,"error"),(corrected,"v3 corrected"))):
        axs[0,col].imshow(stack[:,mid,:],origin="lower",aspect="auto",cmap=THERMAL,vmin=0,vmax=1,extent=[axis_um[0],axis_um[-1],z_rel[0],z_rel[-1]])
        axs[0,col].set(title=f"XZ | {title}",xlabel="x (um)",ylabel="relative z (mm)")
        axs[1,col].imshow(stack[:,:,mid],origin="lower",aspect="auto",cmap=THERMAL,vmin=0,vmax=1,extent=[axis_um[0],axis_um[-1],z_rel[0],z_rel[-1]])
        axs[1,col].set(title=f"YZ | {title}",xlabel="y (um)",ylabel="relative z (mm)")
    fig.suptitle("q=20 detector-domain longitudinal closure across all 18 measured z coordinates")
    savefig(fig, out/"19_detector_domain_xz_yz_closure_v3")

    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=EXP/"outputs"/"digital_twin_correction")
    parser.add_argument("--residual-json", type=Path, default=EXP/"candidates"/"q20_detector_aware_axicon_residual_candidate.json")
    parser.add_argument("--seed-json", type=Path, default=EXP/"candidates"/"q20_detector_domain_slm2_v2_candidate.json")
    parser.add_argument("--out", type=Path, default=ROOT/"outputs"/"validation"/"q20_detector_domain_slm2_v3")
    args = parser.parse_args()
    run(args.source_dir, args.residual_json, args.seed_json, args.out)

if __name__ == "__main__":
    main()
