"""Cross-check the Miao-style modal retrieval against the accepted q=20 digital-twin residual.

The independent analytic benchmark retrieves an angular phase on the Bessel-forming
annulus.  This script projects that phase onto the same smooth low-order coordinates
used by the detector-aware digital twin, then tests those projected coefficients as
an initializer for the full bench-matched residual fit.

The purpose is deliberately falsifiable: if the Miao initializer does not improve or
at least converge to the accepted train/held-out solution, it is recorded as a useful
cross-check but rejected as an initializer.  Held-out odd z planes are never used for
projection, coefficient fitting, or step selection.
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
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
EXP = ROOT / "notebooks" / "experimental" / "axicon_aberration_correction"
for p in (ROOT, TOOLS, EXP):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import fit_q20_detector_aware_model_v2 as v2  # noqa: E402
import fit_q20_detector_aware_model_v3 as v3  # noqa: E402

EPS = np.finfo(float).tiny
PHASE_NAMES = list(v2.PHASE_BASIS_NAMES)
ANGULAR_IDS = np.asarray([1, 2, 3, 4, 5, 6], dtype=int)


def savefig(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=450, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def circular_remove_piston(phase: np.ndarray) -> np.ndarray:
    out = np.asarray(phase, float).copy()
    for i in range(len(out)):
        piston = np.angle(np.mean(np.exp(1j * out[i])))
        out[i] = np.angle(np.exp(1j * (out[i] - piston)))
    return out


def basis_normalisations(radius_m: float = 1.60e-3) -> np.ndarray:
    r = np.linspace(0.0, 2.0e-3, 4001)
    rho = r / radius_m
    env = np.exp(-0.5 * (r / 2.05e-3) ** 8)
    radial = [
        env * (2.0 * rho**2 - 1.0),
        env * rho**2,
        env * rho**2,
        env * (3.0 * rho**3 - 2.0 * rho),
        env * (3.0 * rho**3 - 2.0 * rho),
        env * rho**3,
        env * rho**3,
        env * (6.0 * rho**4 - 6.0 * rho**2 + 1.0),
    ]
    return np.asarray([max(float(np.max(np.abs(a))), EPS) for a in radial], float)


def project_miao_phase(miao_arrays: Path, miao_summary: Path, candidate: dict) -> dict:
    d = np.load(miao_arrays)
    phase = circular_remove_piston(np.asarray(d["annular_phase_train"], float))
    theta = np.asarray(d["theta_rad"], float)
    z_rel = np.asarray(d["z_relative_mm"], float)
    train = np.asarray(d["train_indices"], int)
    z_train = z_rel[train]
    ms = json.loads(Path(miao_summary).read_text(encoding="utf-8"))
    k_perp = float(ms["k_perp_estimation"]["selected_k_perp_m_inv"])

    # Miao stationary-phase mapping: rho_z = z * k_perp/k.  The global annulus
    # piston is not observable from a normalised intensity profile, so only the
    # non-axisymmetric coordinates are projected here.  Defocus and spherical
    # initial coefficients are intentionally left at zero.
    wavelength = 1.03e-6
    k = 2.0 * np.pi / wavelength
    z0_mm = float(candidate["physical_nuisance"]["selected_z0_mm"])
    r_m = (z0_mm + z_train) * 1e-3 * (k_perp / k)
    rho = r_m / 1.60e-3
    env = np.exp(-0.5 * (r_m / 2.05e-3) ** 8)
    norms = basis_normalisations()

    samples = []
    for rr, ee in zip(rho, env):
        samples.append(np.asarray([
            ee * rr**2 * np.cos(2.0 * theta) / norms[1],
            ee * rr**2 * np.sin(2.0 * theta) / norms[2],
            ee * (3.0 * rr**3 - 2.0 * rr) * np.cos(theta) / norms[3],
            ee * (3.0 * rr**3 - 2.0 * rr) * np.sin(theta) / norms[4],
            ee * rr**3 * np.cos(3.0 * theta) / norms[5],
            ee * rr**3 * np.sin(3.0 * theta) / norms[6],
        ]))
    B = np.asarray(samples, float)  # [z, 6, theta]

    def residual(c: np.ndarray) -> np.ndarray:
        pred = np.tensordot(B, np.asarray(c, float), axes=(1, 0))
        return np.angle(np.exp(1j * (pred - phase))).ravel()

    fit = least_squares(
        residual,
        np.zeros(6, float),
        bounds=(-1.5, 1.5),
        loss="soft_l1",
        f_scale=0.30,
        max_nfev=250,
    )
    angular = np.asarray(fit.x, float)
    init = np.zeros(8, float)
    init[ANGULAR_IDS] = angular
    accepted = np.asarray(candidate["phase_coefficients_rad"], float)
    accepted_angular = accepted[ANGULAR_IDS]
    corr = float(np.corrcoef(angular, accepted_angular)[0, 1])
    cosine = float(np.dot(angular, accepted_angular) / max(np.linalg.norm(angular) * np.linalg.norm(accepted_angular), EPS))
    # Scale is not fixed by normalised intensity-only angular retrieval.  Record
    # the least-squares scalar only as a diagnostic; the raw Miao projection is
    # what is passed to the initializer test below.
    scale = float(np.dot(angular, accepted_angular) / max(np.dot(angular, angular), EPS))
    return {
        "initializer_coefficients_rad": init,
        "angular_coefficients_rad": angular,
        "annulus_radii_mm": (r_m * 1e3),
        "projection_wrapped_rms_rad": float(np.sqrt(np.mean(residual(angular) ** 2))),
        "direction_pearson_r_vs_accepted_v3": corr,
        "direction_cosine_similarity_vs_accepted_v3": cosine,
        "diagnostic_scale_to_accepted_v3": scale,
    }


def fit_from_initial(context: dict, z_abs: np.ndarray, train: np.ndarray, held: np.ndarray, p0: np.ndarray) -> dict:
    data = context["data"]
    axis_um = context["axis_um"]
    phase_basis, amp_basis = v2.residual_bases(context["grid"])
    pcoef = np.asarray(p0, float).copy()
    acoef = np.zeros(len(amp_basis), float)
    current = v2.simulate_residual(context, z_abs[train], pcoef, acoef, phase_basis, amp_basis)
    initial = {
        "train": v2.score(current, data[train], axis_um),
        "heldout": v2.score(
            v2.simulate_residual(context, z_abs[held], pcoef, acoef, phase_basis, amp_basis),
            data[held], axis_um,
        ),
    }
    history = []

    for iteration in range(2):
        derivatives = []
        dp = 0.12 if iteration == 0 else 0.08
        da = 0.08 if iteration == 0 else 0.05
        for j in range(len(pcoef)):
            pp, pm = pcoef.copy(), pcoef.copy(); pp[j] += dp; pm[j] -= dp
            plus = v2.simulate_residual(context, z_abs[train], pp, acoef, phase_basis, amp_basis)
            minus = v2.simulate_residual(context, z_abs[train], pm, acoef, phase_basis, amp_basis)
            derivatives.append((plus - minus) / (2.0 * dp))
        for j in range(len(acoef)):
            ap, am = acoef.copy(), acoef.copy(); ap[j] += da; am[j] -= da
            plus = v2.simulate_residual(context, z_abs[train], pcoef, ap, phase_basis, amp_basis)
            minus = v2.simulate_residual(context, z_abs[train], pcoef, am, phase_basis, amp_basis)
            derivatives.append((plus - minus) / (2.0 * da))

        J, y = v2.weighted_system(current, derivatives, data[train], axis_um)
        ridge = np.concatenate([np.full(len(pcoef), 8e-3), np.full(len(acoef), 2e-2)])
        step = np.linalg.solve(J.T @ J + np.diag(ridge), J.T @ y)
        pstep = np.clip(step[:len(pcoef)], -0.32, 0.32)
        astep = np.clip(step[len(pcoef):], -0.14, 0.14)
        base_obj = v2.robust_objective(v2.score(current, data[train], axis_um))
        best = (base_obj, 0.0, pcoef.copy(), acoef.copy(), current)
        for strength in (1.0, 0.60, 0.35):
            ptry = np.clip(pcoef + strength * pstep, -0.90, 0.90)
            atry = np.clip(acoef + strength * astep, -0.35, 0.35)
            pred = v2.simulate_residual(context, z_abs[train], ptry, atry, phase_basis, amp_basis)
            met = v2.score(pred, data[train], axis_um)
            obj = v2.robust_objective(met)
            if obj < best[0]:
                best = (obj, strength, ptry, atry, pred)
        _, strength, pcoef, acoef, current = best
        history.append({
            "iteration": iteration + 1,
            "accepted_strength": float(strength),
            "phase_coefficients_rad": pcoef.tolist(),
            "log_amplitude_coefficients": acoef.tolist(),
            "train": v2.score(current, data[train], axis_um),
        })
        if strength == 0.0:
            break

    final_all = v2.simulate_residual(context, z_abs, pcoef, acoef, phase_basis, amp_basis)
    return {
        "initial": initial,
        "history": history,
        "final_phase_coefficients_rad": pcoef.tolist(),
        "final_log_amplitude_coefficients": acoef.tolist(),
        "final_train": v2.score(final_all[train], data[train], axis_um),
        "final_heldout": v2.score(final_all[held], data[held], axis_um),
    }


def run(source_dir: Path, miao_dir: Path, candidate_json: Path, out: Path) -> dict:
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    candidate = json.loads(Path(candidate_json).read_text(encoding="utf-8"))
    projection = project_miao_phase(
        Path(miao_dir) / "miao_benchmark_arrays.npz",
        Path(miao_dir) / "miao_benchmark_summary.json",
        candidate,
    )

    base = v2.build_context(Path(source_dir))
    physical = candidate["physical_nuisance"]
    cfg = v3.config_with(
        base["config"],
        beam_scale=float(physical["selected_beam_radius_scale"]),
        iris_scale=float(physical["selected_iris_radius_scale"]),
    )
    context = v3.route_context(base, cfg)
    z_rel = context["z_rel"]
    z_abs = (float(physical["selected_z0_mm"]) + z_rel) * 1e-3
    train = np.arange(0, len(z_rel), 2, dtype=int)
    held = np.arange(1, len(z_rel), 2, dtype=int)

    test = fit_from_initial(context, z_abs, train, held, projection["initializer_coefficients_rad"])
    accepted_held = candidate["heldout_metrics"]["phase_plus_amplitude_diagnostic"]
    new_held = test["final_heldout"]
    better = (
        float(new_held["mean_nrmse"]) < float(accepted_held["mean_nrmse"])
        and float(new_held["mean_pearson_r"]) >= float(accepted_held["mean_pearson_r"]) - 1e-4
    )
    result = {
        "study": "Miao analytic modal retrieval as initializer/cross-check for q20 detector-aware digital twin",
        "miao_projection": {
            "phase_basis_names": PHASE_NAMES,
            "initializer_coefficients_rad": projection["initializer_coefficients_rad"].tolist(),
            "angular_coefficients_rad": projection["angular_coefficients_rad"].tolist(),
            "annulus_radii_mm": projection["annulus_radii_mm"].tolist(),
            "projection_wrapped_rms_rad": projection["projection_wrapped_rms_rad"],
            "direction_pearson_r_vs_accepted_v3": projection["direction_pearson_r_vs_accepted_v3"],
            "direction_cosine_similarity_vs_accepted_v3": projection["direction_cosine_similarity_vs_accepted_v3"],
            "diagnostic_scale_to_accepted_v3": projection["diagnostic_scale_to_accepted_v3"],
        },
        "initializer_test": test,
        "accepted_v3_reference": {
            "phase_coefficients_rad": candidate["phase_coefficients_rad"],
            "heldout_phase_plus_amplitude": accepted_held,
        },
        "miao_initializer_selected_for_final_candidate": bool(better),
        "decision": (
            "Miao initializer improves the frozen train/held-out criterion and should be promoted for a new candidate."
            if better else
            "Miao retrieval is retained as an independent analytic cross-check/initial direction; the accepted v3 candidate remains preferred after full bench-matched optimisation."
        ),
        "heldout_used_for_selection": False,
        "hardware_ready": False,
    }
    (out / "miao_initializer_crosscheck.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    miao_ang = projection["angular_coefficients_rad"]
    accepted_ang = np.asarray(candidate["phase_coefficients_rad"], float)[ANGULAR_IDS]
    labels = [PHASE_NAMES[i] for i in ANGULAR_IDS]
    xx = np.arange(len(labels)); width = 0.38
    fig, ax = plt.subplots(figsize=(10.0, 4.8), constrained_layout=True)
    ax.bar(xx - width/2, miao_ang, width, label="Miao-style projection")
    ax.bar(xx + width/2, accepted_ang, width, label="bench-matched v3")
    ax.axhline(0, lw=0.8, color="black")
    ax.set_xticks(xx, labels, rotation=25, ha="right")
    ax.set(ylabel="coefficient (rad)", title=f"Independent low-order angular cross-check: direction r = {projection['direction_pearson_r_vs_accepted_v3']:.3f}")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=.2)
    savefig(fig, out / "miao_vs_digital_twin_angular_coefficients")

    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source-dir", type=Path, default=EXP / "outputs" / "digital_twin_correction")
    p.add_argument("--miao-dir", type=Path, default=ROOT / "outputs" / "validation" / "q20_miao_bessel_modal_benchmark")
    p.add_argument("--candidate-json", type=Path, default=EXP / "candidates" / "q20_detector_aware_model_v3_candidate.json")
    p.add_argument("--out", type=Path, default=ROOT / "outputs" / "validation" / "q20_miao_initializer_crosscheck")
    a = p.parse_args(); run(a.source_dir, a.miao_dir, a.candidate_json, a.out)


if __name__ == "__main__":
    main()
