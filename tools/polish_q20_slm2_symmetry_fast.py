"""Targeted symmetry polish for the q=20 detector-domain SLM2 precompensation.

This is a fast follow-up to v2/v3.  v2 already restored the radial Bessel-ring
structure very well, but its principal annulus became more left/right
imbalanced.  Rather than re-optimising every SLM2 degree of freedom, this solve
freezes the radial-good v2 solution and adjusts only the phase terms that are
odd under x reflection and can therefore change left/right imbalance:

    cos(m theta) for odd m, sin(m theta) for even m, for radial orders p=0,1.

Finite-difference sensitivities are evaluated only on the declared training
z-planes.  The final candidate is then propagated on the complete 18-plane
stack and scored on untouched held-out planes.  The solve includes detector
intensity, horizontal-profile, x-mirror, and radial-profile residuals, and will
not accept a step that materially degrades the already-good radial rings.

The output is still a model-space SLM2 candidate, not a hardware-ready mask.
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
for p in (ROOT, EXP, TOOLS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from real_bmg_digital_twin_correction import (
    AxiconError, FourFError, Q, SystemErrorConfig, propagate_route,
)
from vbb_study.digital_twin.residual_phase_fit import angular_phase_from_coefficients
from optimize_q20_slm2_detector_closure_v2 import (
    route, detector_render, intensity_metrics, radial_profile,
    phase_basis, phase_from_coefficients, savefig,
)
from optimize_q20_slm2_detector_closure_v3 import (
    structure_metrics_v3, valid_winding_loops,
)

EPS = np.finfo(float).tiny
THERMAL = "inferno"

# These are exactly the p=0,1 basis terms that are odd under x reflection.
ACTIVE_NAMES = (
    "r0_cos1", "r1_cos1",
    "r0_sin2", "r1_sin2",
    "r0_cos3", "r1_cos3",
    "r0_sin4", "r1_sin4",
)


def seed_coefficients(seed_json: Path, names: list[str]) -> np.ndarray:
    seed = json.loads(Path(seed_json).read_text(encoding="utf-8"))
    lookup = dict(zip(seed["basis_names"], seed["coefficients_rad"]))
    return np.asarray([float(lookup.get(n, 0.0)) for n in names], float)


def radial_contract(axis_um: np.ndarray, dr_um: float = 2.0, rmax_um: float = 140.0):
    X, Y = np.meshgrid(axis_um, axis_um, indexing="xy")
    R = np.hypot(X, Y)
    edges = np.arange(0.0, rmax_um + dr_um, dr_um)
    ids = np.digitize(R.ravel(), edges) - 1
    good = (ids >= 0) & (ids < len(edges)-1)
    counts = np.bincount(ids[good], minlength=len(edges)-1).astype(float)
    rr = 0.5*(edges[:-1] + edges[1:])
    return X, Y, R, ids, good, np.maximum(counts, 1.0), rr


def radial_vector(image: np.ndarray, ids, good, counts):
    sums = np.bincount(ids[good], weights=np.asarray(image, float).ravel()[good], minlength=len(counts))
    return sums / counts


def normalise(stack):
    a = np.maximum(np.asarray(stack, float), 0.0)
    return a / np.maximum(a.reshape(a.shape[0], -1).max(axis=1)[:, None, None], EPS)


def reduced_system(current, derivatives, target, axis_um):
    """Build a small LS system aimed specifically at the remaining x-asymmetry."""
    X, Y, R, ids, good, counts, rr = radial_contract(axis_um)
    cur, tar = normalise(current), normalise(target)
    roi = R <= 145.0
    ring_roi = (R >= 18.0) & (R <= 115.0)
    mid = len(axis_um)//2
    cut_sel = (axis_um >= -125.0) & (axis_um <= 125.0)
    radial_sel = (rr >= 8.0) & (rr <= 138.0)
    J_parts, y_parts = [], []
    for iz in range(len(cur)):
        # Keep global detector fidelity present but let the symmetry observables dominate.
        t = tar[iz][roi]
        w = np.sqrt(0.30 + 0.70*np.sqrt(np.clip(t, 0.0, 1.0)))
        J_parts.append(0.45*w[:,None]*np.column_stack([d[iz][roi] for d in derivatives]))
        y_parts.append(0.45*w*(t-cur[iz][roi]))

        # Direct left/right mirror residual in the annulus and secondary-ring region.
        c_diff = cur[iz] - cur[iz, :, ::-1]
        t_diff = tar[iz] - tar[iz, :, ::-1]
        dcols = [d[iz] - d[iz, :, ::-1] for d in derivatives]
        J_parts.append(1.65*np.column_stack([dc[ring_roi] for dc in dcols]))
        y_parts.append(1.65*(t_diff-c_diff)[ring_roi])

        # The exact horizontal section the user is judging visually.
        J_parts.append(2.2*np.column_stack([d[iz, mid, cut_sel] for d in derivatives]))
        y_parts.append(2.2*(tar[iz, mid, cut_sel]-cur[iz, mid, cut_sel]))

        # Preserve the already-good radial Bessel-ring structure.
        pc = radial_vector(cur[iz], ids, good, counts)
        pt = radial_vector(tar[iz], ids, good, counts)
        dprof = np.column_stack([
            radial_vector(d[iz], ids, good, counts)[radial_sel] for d in derivatives
        ])
        J_parts.append(9.0*dprof)
        y_parts.append(9.0*(pt-pc)[radial_sel])
    return np.vstack(J_parts), np.concatenate(y_parts)


def score(stack, target, axis_um):
    idx = np.arange(len(stack), dtype=int)
    r, e = intensity_metrics(stack, target, axis_um, idx)
    s = structure_metrics_v3(stack, target, axis_um, idx)
    # Strongly prioritise the left/right annulus while protecting radial rings.
    value = (
        e + 0.08*(1.0-r) +
        0.40*s["mean_opposite_peak_imbalance"] +
        0.10*s["max_opposite_peak_imbalance"] +
        0.18*s["mean_sidelobe_profile_rmse"] +
        0.08*s["mean_x_mirror_residual_rmse"]
    )
    return float(value), float(r), float(e), s


def run(source_dir: Path, residual_json: Path, seed_json: Path, out: Path):
    source_dir, residual_json, seed_json, out = map(Path, (source_dir, residual_json, seed_json, out))
    out.mkdir(parents=True, exist_ok=True)
    residual = json.loads(residual_json.read_text(encoding="utf-8"))
    if not residual.get("heldout_support_pass", False):
        raise RuntimeError("residual phase candidate did not pass held-out recreation")

    d = np.load(source_dir/"rerender_arrays.npz")
    axis_um = np.asarray(d["axis_um"], float)
    z_rel = np.asarray(d["z_relative_mm"], float)
    summary = json.loads((source_dir/"run_summary.json").read_text(encoding="utf-8"))
    scale = float(summary["effective_axicon"]["effective_scale_relative_to_repository_2deg_assumption"])
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
        theta_ax,
        np.asarray(residual["coefficients_rad"], float),
        modes=tuple(residual["angular_modes"]),
    )
    basis, names = phase_basis(base["relay_route"]["grid"])
    coeff = seed_coefficients(seed_json, names)
    active = np.asarray([names.index(n) for n in ACTIVE_NAMES], int)

    # One full target render; training derivatives use only alternating planes.
    nominal_native, nominal_meta = propagate_route(config, z_abs)
    target = detector_render(nominal_native, nominal_meta["x_m"], axis_um)
    del nominal_native
    train_idx = np.arange(0, len(z_abs), 2, dtype=int)
    held_idx = np.arange(1, len(z_abs), 2, dtype=int)

    def phase(c):
        return phase_from_coefficients(basis, c)

    def simulate_subset(c, ids):
        native, meta = propagate_route(
            config, z_abs[np.asarray(ids, int)],
            phase_slm2=phase(c), phase_axicon_input=error_phase,
        )
        shown = detector_render(native, meta["x_m"], axis_um)
        del native
        return shown

    seed_full = simulate_subset(coeff, np.arange(len(z_abs)))
    seed_train = seed_full[train_idx]
    current_train = seed_train.copy()
    seed_train_score = score(seed_train, target[train_idx], axis_um)
    history = []
    delta = 0.055

    for iteration in range(2):
        derivatives = []
        for j in active:
            cp, cm = coeff.copy(), coeff.copy()
            cp[j] += delta; cm[j] -= delta
            plus = simulate_subset(cp, train_idx)
            minus = simulate_subset(cm, train_idx)
            derivatives.append((plus-minus)/(2.0*delta))
        J, y = reduced_system(current_train, derivatives, target[train_idx], axis_um)
        ridge = 2.5e-2
        step = np.linalg.solve(J.T@J + ridge*np.eye(len(active)), J.T@y)
        step = np.clip(step, -0.16, 0.16)
        base_score = score(current_train, target[train_idx], axis_um)
        chosen = None
        for alpha in (1.0, 0.55, 0.30, 0.15, 0.075):
            trial_c = coeff.copy()
            trial_c[active] = np.clip(trial_c[active] + alpha*step, -1.2, 1.2)
            trial = simulate_subset(trial_c, train_idx)
            sc = score(trial, target[train_idx], axis_um)
            # Never trade away the v2 ring structure just to improve symmetry.
            radial_ok = sc[3]["mean_radial_profile_corr"] >= seed_train_score[3]["mean_radial_profile_corr"] - 5e-4
            side_ok = sc[3]["mean_sidelobe_profile_rmse"] <= seed_train_score[3]["mean_sidelobe_profile_rmse"]*1.08
            if radial_ok and side_ok and (chosen is None or sc[0] < chosen[0]):
                chosen = (sc[0], trial_c, trial, alpha, sc)
        if chosen is None or chosen[0] >= base_score[0]:
            history.append({"iteration": iteration+1, "accepted": False, "train_score": base_score})
            break
        coeff, current_train = chosen[1], chosen[2]
        history.append({
            "iteration": iteration+1, "accepted": True, "alpha": float(chosen[3]),
            "train_score": chosen[4],
            "active_coefficients_rad": {names[j]: float(coeff[j]) for j in active},
        })
        delta = 0.040

    corrected = simulate_subset(coeff, np.arange(len(z_abs)))
    seed_held = score(seed_full[held_idx], target[held_idx], axis_um)
    corrected_held = score(corrected[held_idx], target[held_idx], axis_um)

    nominal_route = route(config)
    corrected_route = route(config, slm2_phase=phase(coeff), axicon_phase=error_phase)
    wnom, wcorr, valid_loops, winding_ok = valid_winding_loops(nominal_route, corrected_route)

    s = corrected_held[3]
    closure_pass = bool(
        corrected_held[1] >= 0.95 and corrected_held[2] <= 0.05 and
        s["mean_radial_profile_corr"] >= 0.995 and
        s["mean_opposite_peak_imbalance"] <= 0.08 and
        s["max_opposite_peak_imbalance"] <= 0.18 and winding_ok
    )
    result = {
        "status": "targeted_symmetry_polish_from_v2",
        "active_basis_names": list(ACTIVE_NAMES),
        "basis_names": names,
        "coefficients_rad": coeff.tolist(),
        "iterations": history,
        "heldout_v2_seed": {
            "objective": seed_held[0], "mean_pearson_r": seed_held[1],
            "mean_nrmse": seed_held[2], **seed_held[3],
        },
        "heldout_symmetry_polished": {
            "objective": corrected_held[0], "mean_pearson_r": corrected_held[1],
            "mean_nrmse": corrected_held[2], **corrected_held[3],
        },
        "winding_nominal": wnom, "winding_corrected": wcorr,
        "nominally_valid_q20_winding_loops": valid_loops,
        "winding_preserved_on_valid_loops": winding_ok,
        "closure_pass": closure_pass,
        "closure_rule": "held-out r>=0.95, NRMSE<=0.05, radial r>=0.995, mean L/R imbalance<=0.08, max L/R imbalance<=0.18, q=20 preserved",
        "hardware_ready": False,
        "hardware_blockers": residual.get("hardware_blockers", []),
    }
    (out/"q20_slm2_symmetry_polish_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.savez_compressed(
        out/"q20_slm2_symmetry_polish.npz",
        slm2_phase_rad=phase(coeff).astype(np.float32),
        error_phase_axicon_input_rad=error_phase.astype(np.float32),
        z_relative_mm=z_rel, axis_um=axis_um,
        nominal_detector=target.astype(np.float32),
        v2_seed_detector=seed_full.astype(np.float32),
        predicted_corrected_detector=corrected.astype(np.float32),
    )

    rep = int(np.argmin(np.abs(z_rel+10.0)))
    extent = [axis_um[0], axis_um[-1], axis_um[0], axis_um[-1]]
    mid = len(axis_um)//2
    fig, axs = plt.subplots(2, 3, figsize=(14.5, 8.2), constrained_layout=True)
    for col,(stack,title) in enumerate(((target,"nominal target"),(seed_full,"v2 seed"),(corrected,"symmetry-polished"))):
        axs[0,col].imshow(stack[rep], origin="lower", extent=extent, cmap=THERMAL, vmin=0, vmax=1, interpolation="nearest")
        axs[0,col].set(title=title, xlabel="x (um)", ylabel="y (um)", aspect="equal")
        axs[1,col].plot(axis_um,target[rep,mid],lw=1.7,label="nominal")
        axs[1,col].plot(axis_um,stack[rep,mid],"--",lw=1.4,label=title)
        axs[1,col].set(xlim=(-140,140),ylim=(0,1.05),xlabel="x (um)",ylabel="normalized intensity")
        axs[1,col].grid(alpha=.2); axs[1,col].legend(fontsize=8)
    fig.suptitle("q=20 SLM2 targeted symmetry polish: preserve the recovered Bessel rings")
    savefig(fig,out/"20_q20_slm2_symmetry_polish")

    fig,axs=plt.subplots(1,3,figsize=(14,4.4),constrained_layout=True)
    rr,pn=radial_profile(target[rep],axis_um); _,ps=radial_profile(seed_full[rep],axis_um); _,pc=radial_profile(corrected[rep],axis_um)
    axs[0].plot(rr,pn,label="nominal",lw=1.7); axs[0].plot(rr,ps,label="v2",lw=1.3); axs[0].plot(rr,pc,label="polished",lw=1.4)
    axs[0].set(xlim=(0,140),xlabel="radius (um)",ylabel="azimuthal mean intensity",title="Radial Bessel-ring structure"); axs[0].grid(alpha=.2); axs[0].legend()
    labels=["v2","polished"]
    axs[1].bar(labels,[seed_held[3]["mean_opposite_peak_imbalance"],corrected_held[3]["mean_opposite_peak_imbalance"]]); axs[1].set(title="Held-out mean L/R imbalance",ylabel="fraction")
    axs[2].bar(labels,[seed_held[3]["mean_sidelobe_profile_rmse"],corrected_held[3]["mean_sidelobe_profile_rmse"]]); axs[2].set(title="Held-out sidelobe-profile RMSE",ylabel="RMSE")
    for ax in axs[1:]: ax.grid(axis="y",alpha=.2)
    fig.suptitle("Symmetry polish is rejected if radial-ring fidelity degrades")
    savefig(fig,out/"21_q20_slm2_symmetry_polish_metrics")

    print(json.dumps(result, indent=2))
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source-dir",type=Path,default=EXP/"outputs"/"digital_twin_correction")
    ap.add_argument("--residual-json",type=Path,default=EXP/"candidates"/"q20_detector_aware_axicon_residual_candidate.json")
    ap.add_argument("--seed-json",type=Path,default=EXP/"candidates"/"q20_detector_domain_slm2_v2_candidate.json")
    ap.add_argument("--out",type=Path,default=ROOT/"outputs"/"validation"/"q20_slm2_symmetry_polish")
    args=ap.parse_args(); run(args.source_dir,args.residual_json,args.seed_json,args.out)

if __name__=="__main__":
    main()
