"""Hybrid q=20 correction v6: complex axicon-input closure + Miao concentric validation.

This solver attacks a limitation of the earlier detector-only precompensation.
The real-data inverse supports BOTH phase and diagnostic amplitude nuisance at
the selected-order field immediately before the axicon.  A phase-only SLM2
cannot locally cancel amplitude, but the phase-only pattern is followed by a
finite 4F spatial filter; that relay can convert some phase structure into
selected-order complex-amplitude redistribution.  Therefore v6 first solves
for a smooth, zero-winding SLM2 phase that makes the *errored complex field at
the axicon input* approach the nominal complex field.  It then freezes that
phase family and chooses a scalar strength using the independent Miao/data
concentric multi-z target from v5.

The correction remains numerical model space only.  No corrected BeamGage
frame is emitted or claimed, and the native SLM2 coordinate transform/LUT
remain bench-calibration blockers.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
EXP = ROOT / "notebooks" / "experimental" / "axicon_aberration_correction"
for p in (ROOT, TOOLS, EXP):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import solve_q20_slm2_hybrid_miao_concentric_v5 as v5  # noqa: E402
from optimize_q20_slm2_detector_closure_v2 import phase_from_coefficients  # noqa: E402
from real_bmg_digital_twin_correction import FIT_N, FIT_WINDOW_M, Q, RELAY_N  # noqa: E402
from vbb_study.digital_twin.vortex_system_route import build_multirate_system_route  # noqa: E402

EPS = np.finfo(float).tiny
THERMAL = "inferno"
ALPHAS = np.asarray([0.35, 0.50, 0.65, 0.80, 0.95, 1.10], float)
FIELD_DECIMATE = 16
MAX_FIELD_ITER = 2


def savefig(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=500, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def rich_zero_winding_basis(grid: dict) -> tuple[np.ndarray, list[str]]:
    """Smooth continuous phase coordinates; adding them cannot itself encode charge."""
    X = np.asarray(grid["X"], float)
    Y = np.asarray(grid["Y"], float)
    R = np.hypot(X, Y)
    th = np.arctan2(Y, X)
    rho = R / 2.0e-3
    env = np.exp(-0.5 * (R / 2.55e-3) ** 8)
    basis: list[np.ndarray] = []
    names: list[str] = []
    for m in range(1, 7):
        for p in (0, 1):
            radial = env * rho**p
            basis.append(radial * np.cos(m * th)); names.append(f"r{p}_cos{m}")
            basis.append(radial * np.sin(m * th)); names.append(f"r{p}_sin{m}")
    # Axisymmetric longitudinal/radial cleanup.  No piston and no theta ramp.
    basis.append(env * rho**2); names.append("r2_axisymmetric")
    basis.append(env * rho**4); names.append("r4_axisymmetric")
    arr = np.stack(basis).astype(np.float32)
    # Normalize each coordinate on the illuminated support for comparable ridge penalty.
    support = R <= 2.2e-3
    for i in range(len(arr)):
        arr[i] /= max(float(np.max(np.abs(arr[i][support]))), EPS)
    return arr, names


def align_to_target(field: np.ndarray, target: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Remove irrelevant global complex scalar for complex-field comparison."""
    y = np.asarray(field, complex)
    t = np.asarray(target, complex)
    w = np.asarray(weight, float)
    den = np.sum(w * np.abs(y) ** 2)
    gamma = np.sum(w * np.conj(y) * t) / max(float(np.real(den)), EPS)
    return gamma * y


def complex_metric(field: np.ndarray, target: np.ndarray, weight: np.ndarray) -> dict:
    y = align_to_target(field, target, weight)
    t = np.asarray(target, complex)
    w = np.asarray(weight, float)
    den = max(float(np.sum(w * np.abs(t) ** 2)), EPS)
    e = float(np.sqrt(np.sum(w * np.abs(y - t) ** 2) / den))
    ov_num = abs(np.sum(w * np.conj(y) * t))
    ov_den = np.sqrt(np.sum(w * np.abs(y) ** 2) * np.sum(w * np.abs(t) ** 2))
    return {"weighted_complex_nrmse": e, "weighted_field_overlap": float(ov_num / max(float(ov_den), EPS))}


def solve_complex_closure(config, pcoef: np.ndarray, acoef: np.ndarray, out: Path):
    nominal = build_multirate_system_route(
        f"V{Q}", relay_grid_n=RELAY_N, propagation_grid_n=FIT_N,
        window_m=FIT_WINDOW_M, config=config,
    )
    target = np.asarray(nominal["field_on_axicon_plane"], complex)
    x = np.asarray(nominal["grid"]["x"], float)
    X, Y = np.meshgrid(x, x, indexing="xy")
    R = np.hypot(X, Y)
    amp2 = np.abs(target) ** 2
    amp2 /= max(float(np.max(amp2)), EPS)
    weight = np.where(R <= 2.25e-3, 0.04 + 0.96 * np.sqrt(amp2), 0.0)

    relay_grid = nominal["relay_route"]["grid"]
    basis, names = rich_zero_winding_basis(relay_grid)
    coeff = np.zeros(len(names), float)

    def simulate(c: np.ndarray) -> np.ndarray:
        phase = phase_from_coefficients(basis, c)
        rr = v5.build_route(config, FIT_N, slm2_phase=phase, pcoef=pcoef, acoef=acoef)
        return np.asarray(rr["field_on_axicon_plane"], complex)

    current = simulate(coeff)
    before = complex_metric(current, target, weight)
    history: list[dict] = []
    sl = (slice(None, None, FIELD_DECIMATE), slice(None, None, FIELD_DECIMATE))
    wt = weight[sl]
    good = wt > 0.0
    sw = np.sqrt(wt[good])
    tsub = target[sl][good]

    for iteration in range(MAX_FIELD_ITER):
        cur_aligned = align_to_target(current, target, weight)
        csub = cur_aligned[sl][good]
        residual = tsub - csub
        delta = 0.10 if iteration == 0 else 0.065
        derivatives: list[np.ndarray] = []
        for j in range(len(coeff)):
            cp, cm = coeff.copy(), coeff.copy()
            cp[j] += delta; cm[j] -= delta
            yp = simulate(cp); ym = simulate(cm)
            yp = align_to_target(yp, target, weight)
            ym = align_to_target(ym, target, weight)
            derivatives.append(((yp - ym) / (2.0 * delta))[sl][good])
            del yp, ym
        Dc = np.column_stack(derivatives)
        J = np.vstack([sw[:, None] * Dc.real, sw[:, None] * Dc.imag])
        rhs = np.concatenate([sw * residual.real, sw * residual.imag])
        ridge = 1.8e-2
        step = np.linalg.solve(J.T @ J + ridge * np.eye(J.shape[1]), J.T @ rhs)
        step = np.clip(step, -0.32, 0.32)
        base_obj = complex_metric(current, target, weight)["weighted_complex_nrmse"]
        chosen = (base_obj, coeff.copy(), current, 0.0)
        for strength in (1.0, 0.65, 0.40, 0.22):
            trial_c = np.clip(coeff + strength * step, -1.20, 1.20)
            trial = simulate(trial_c)
            obj = complex_metric(trial, target, weight)["weighted_complex_nrmse"]
            if obj < chosen[0]:
                chosen = (obj, trial_c, trial, strength)
        accepted = chosen[3] > 0.0
        coeff, current = chosen[1], chosen[2]
        history.append({
            "iteration": iteration + 1,
            "accepted": bool(accepted),
            "step_strength": float(chosen[3]),
            "complex_metrics": complex_metric(current, target, weight),
            "coefficients_rad": coeff.tolist(),
        })
        del derivatives, Dc, J, rhs, step
        gc.collect()
        if not accepted:
            break

    phase = phase_from_coefficients(basis, coeff)
    after = complex_metric(current, target, weight)
    np.save(out / "v6_complex_closure_full_strength_slm2_phase_rad.npy", phase.astype(np.float32))
    return phase, coeff, names, before, after, history


def run(source_dir: Path, miao_dir: Path, crosscheck_json: Path, candidate_json: Path, out: Path) -> dict:
    source_dir = Path(source_dir); miao_dir = Path(miao_dir); out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    candidate = json.loads(Path(candidate_json).read_text(encoding="utf-8"))
    cross = json.loads(Path(crosscheck_json).read_text(encoding="utf-8"))
    pcoef, acoef, residual_source = v5.candidate_coefficients(candidate, cross)

    context = v5.v2.build_context(source_dir)
    data = v5.normalise(context["data"])
    z_rel = np.asarray(context["z_rel"], float)
    z0 = float(candidate["physical_nuisance"]["selected_z0_mm"])
    z_abs = (z0 + z_rel) * 1e-3
    md = np.load(miao_dir / "miao_benchmark_arrays.npz")
    miao_pred = np.asarray(md["predicted"], float)
    sym_target, target = v5.hybrid_target(data, miao_pred, v5.AXIS_UM)
    config = v5.v4.config_from_candidate(candidate, source_dir)

    full_phase, coeff, names, before, after, history = solve_complex_closure(config, pcoef, acoef, out)

    sweep = []
    for alpha in ALPHAS:
        rec = v5.evaluate_alpha(float(alpha), v5.SWEEP_N, v5.INNER_VALID, config, z_abs, target, full_phase, pcoef, acoef)
        sweep.append(rec)
        print(json.dumps({"alpha": float(alpha), "validation": rec}, indent=2))
    passing = [s for s in sweep if s["topology_q20_all_contours"]]
    if not passing:
        raise RuntimeError("v6 found no q=20 topology-preserving strength")
    selected = min(passing, key=lambda s: s["corrected_objective"])
    alpha_star = float(selected["alpha"])
    (out / "v6_strength_sweep.json").write_text(json.dumps(sweep, indent=2) + "\n", encoding="utf-8")

    prod = v5.production(config, z_abs, z_rel, target, sym_target, miao_pred, full_phase, alpha_star, pcoef, acoef, out)
    held = prod["metrics"]["legacy_heldout"]
    pos = held["detector_positive"]; cor = held["detector_corrected"]
    cv_red = 1.0 - cor["mean_principal_ring_azimuth_cv"] / max(pos["mean_principal_ring_azimuth_cv"], EPS)
    mirror_red = 1.0 - cor["mirror_rmse"] / max(pos["mirror_rmse"], EPS)
    gate = bool(
        prod["topology_q20_all_contours"]
        and cv_red >= 0.35
        and mirror_red >= 0.20
        and cor["mean_radial_profile_corr"] >= 0.90
    )
    result = {
        "study": "q20 phase-only SLM2 complex axicon-input closure followed by Miao/data concentric validation",
        "residual_source": residual_source,
        "basis_names": names,
        "coefficients_full_strength_rad": coeff.tolist(),
        "complex_field_before": before,
        "complex_field_after": after,
        "field_solve_history": history,
        "selected_alpha": alpha_star,
        "selected_validation": selected,
        "production": prod,
        "legacy_heldout_cv_reduction_fraction": float(cv_red),
        "legacy_heldout_mirror_reduction_fraction": float(mirror_red),
        "legacy_heldout_radial_profile_corr": float(cor["mean_radial_profile_corr"]),
        "passes_concentricity_gate": gate,
        "hardware_ready": False,
        "corrected_camera_evidence": "none; corrected stacks are numerical predictions only",
    }
    (out / "v6_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    # Compact figure focused on the two independent ideas being combined.
    labels = ["complex NRMSE", "1 - overlap"]
    bvals = [before["weighted_complex_nrmse"], 1.0 - before["weighted_field_overlap"]]
    avals = [after["weighted_complex_nrmse"], 1.0 - after["weighted_field_overlap"]]
    xx = np.arange(2); width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    ax.bar(xx - width/2, bvals, width, label="diagnosed field")
    ax.bar(xx + width/2, avals, width, label="after SLM2 complex-closure solve")
    ax.set_xticks(xx, labels); ax.set_ylabel("error (lower is better)")
    ax.set_title("Phase-only SLM2 uses the finite 4F filter to approach complex-field closure")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=.2)
    savefig(fig, out / "v6_complex_closure_metrics")

    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source-dir", type=Path, default=EXP / "outputs" / "digital_twin_correction")
    p.add_argument("--miao-dir", type=Path, default=ROOT / "outputs" / "validation" / "q20_miao_bessel_modal")
    p.add_argument("--crosscheck-json", type=Path, default=ROOT / "outputs" / "validation" / "q20_miao_initializer_crosscheck" / "miao_initializer_crosscheck.json")
    p.add_argument("--candidate-json", type=Path, default=ROOT / "outputs" / "validation" / "q20_detector_aware_model_v3" / "q20_detector_aware_model_v3_summary.json")
    p.add_argument("--out", type=Path, default=ROOT / "outputs" / "validation" / "q20_slm2_complex_closure_miao_v6")
    a = p.parse_args()
    run(a.source_dir, a.miao_dir, a.crosscheck_json, a.candidate_json, a.out)


if __name__ == "__main__":
    main()
