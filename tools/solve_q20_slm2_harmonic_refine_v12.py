"""q=20 correction v12: topology-safe angular-harmonic refinement.

Purpose
-------
v10/v11 demonstrated that a propagated multi-plane target is necessary, but
v11 only attenuates/radially guards the v10 command.  It cannot create a new
solution when the remaining error is angular.  The visible failure is a
cross/fan modulation of the outer Bessel rings, so v12 puts that failure mode
explicitly into the optimisation.

v12 therefore:
  * regenerates the frozen Miao-initialised v10 multi-plane command;
  * finds a topology-safe weak continuation seed on INNER validation planes;
  * adds physically interpretable even angular SLM2 modes (m=2,4,6,8;
    cosine/sine quadratures) through the real finite-4F/+1-iris/axicon route;
  * coordinate-descends the modal coefficients using a loss that includes the
    existing multi-ring concentricity score plus explicit m=2 and m=4
    propagated detector-plane harmonic energy;
  * rejects every candidate that fails q=20 on any 1.0--1.5 mm winding contour;
  * evaluates the legacy odd planes only after the modal solve is frozen;
  * validates the chosen command at 4096 samples.

This is a numerical model-space correction study only.  No corrected panel is
post-correction BeamGage data, and the output is not a hardware-ready mask until
SLM2 LUT/bench-coordinate calibration is available.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
EXP = ROOT / "notebooks" / "experimental" / "axicon_aberration_correction"
for p in (ROOT, TOOLS, EXP):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import solve_q20_slm2_hybrid_miao_concentric_v5 as v5  # noqa: E402
import solve_q20_slm2_ifta_circular_v9 as v9  # noqa: E402
import solve_q20_slm2_multiplane_circular_v10 as v10  # noqa: E402
import solve_q20_slm2_topology_guarded_multiplane_v11 as v11  # noqa: E402
from real_bmg_digital_twin_correction import FIT_WINDOW_M, RELAY_N  # noqa: E402
from vbb_study.digital_twin.vortex_system_route import build_system_route  # noqa: E402

EPS = np.finfo(float).tiny
THERMAL = "inferno"
PROD_N = 4096

# Conservative continuation of the v10 command.  These are model-space search
# strengths, not measured SLM efficiencies.
SEED_ALPHAS = np.asarray([0.00, 0.04, 0.07, 0.10, 0.14, 0.18, 0.23, 0.30], float)

# The cross/fan failure is dominated visually by even angular structure.  m=2
# and m=4 are searched first; m=6/8 provide a small residual clean-up basis.
MODE_ORDER = ((4, "cos"), (4, "sin"), (2, "cos"), (2, "sin"),
              (6, "cos"), (6, "sin"), (8, "cos"), (8, "sin"))
STEP_SCHEDULE_RAD = (0.30, 0.18, 0.10, 0.055)
APERTURE_RADIUS_M = 2.15e-3
TAPER_INNER_FRACTION = 0.78
MAX_COMMAND_RMS_RAD = 1.10
MAX_MODE_COEFF_RAD = 0.85

# Explicit low-order angular-energy weights.  The generic multi-ring metric
# remains dominant, but m=4 is deliberately expensive because that is the
# observed cross-like morphology we are trying to eliminate.
W_M2 = 0.55
W_M4 = 1.15
W_COMMAND_RMS = 0.012


def savefig(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=500, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def smoothstep01(x: np.ndarray) -> np.ndarray:
    t = np.clip(np.asarray(x, float), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def harmonic_basis(grid: dict, m: int, quadrature: str) -> np.ndarray:
    """Unit-RMS, smoothly apodised angular mode on the illuminated SLM2 pupil."""
    X = np.asarray(grid["X"], float)
    Y = np.asarray(grid["Y"], float)
    R = np.asarray(grid["R"], float)
    theta = np.arctan2(Y, X)
    rho = np.clip(R / APERTURE_RADIUS_M, 0.0, 1.0)

    # Keep the correction smooth at the pupil edge.  A weak radial power makes
    # these Zernike-like rather than pure azimuthal phase gratings.
    taper = 1.0 - smoothstep01((rho - TAPER_INNER_FRACTION) / max(1.0 - TAPER_INNER_FRACTION, EPS))
    radial = rho ** max(2, min(int(m), 6))
    if quadrature == "cos":
        angular = np.cos(float(m) * theta)
    elif quadrature == "sin":
        angular = np.sin(float(m) * theta)
    else:
        raise ValueError(quadrature)
    b = radial * angular * taper
    support = R <= APERTURE_RADIUS_M
    b -= float(np.mean(b[support]))
    rms = float(np.sqrt(np.mean(b[support] ** 2)))
    return b / max(rms, EPS)


def polar_mode_energy(stack: np.ndarray, axis_um: np.ndarray, mode: int) -> float:
    """Weighted angular Fourier energy of bright rings in a detector stack.

    The field is sampled on many circular contours.  Dark radii are suppressed
    by radial-mean weighting, so the metric is driven by actual Bessel rings
    rather than empty background.  A perfectly concentric intensity train has
    energy near zero for every non-zero angular mode.
    """
    images = np.asarray(stack, float)
    axis = np.asarray(axis_um, float)
    if images.ndim != 3 or images.shape[1:] != (axis.size, axis.size):
        raise ValueError("stack/axis mismatch")
    coord0 = float(np.interp(0.0, axis, np.arange(axis.size, dtype=float)))
    d_um = float(np.mean(np.diff(axis)))
    rmax_um = 0.78 * float(min(abs(axis[0]), abs(axis[-1])))
    radii_um = np.linspace(max(18.0, 0.08 * rmax_um), rmax_um, 72)
    theta = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False)
    phase = np.exp(-1j * float(mode) * theta)
    vals = []
    weights = []
    for image in images:
        I = np.maximum(np.asarray(image, float), 0.0)
        I /= max(float(np.max(I)), EPS)
        for r_um in radii_um:
            rr = float(r_um / d_um)
            yy = coord0 + rr * np.sin(theta)
            xx = coord0 + rr * np.cos(theta)
            ring = ndimage.map_coordinates(I, [yy, xx], order=1, mode="constant", cval=0.0)
            mean = float(np.mean(ring))
            if mean < 0.025:
                continue
            coeff = abs(np.mean(ring * phase)) / max(mean, EPS)
            vals.append(float(coeff * coeff))
            weights.append(float(mean ** 1.5))
    if not vals:
        return float("inf")
    return float(np.average(np.asarray(vals), weights=np.asarray(weights)))


def validation_stack(route: dict, z_abs: np.ndarray) -> np.ndarray:
    ids = np.asarray(v10.VALID_PLANES, int)
    return v5.detector_stack(route, np.asarray(z_abs, float)[ids])


def objective_from_stack(stack: np.ndarray, target_det: np.ndarray, command: np.ndarray) -> tuple[float, dict]:
    ids = np.arange(len(v10.VALID_PLANES), dtype=int)
    target = np.asarray(target_det, float)[np.asarray(v10.VALID_PLANES, int)]
    multiring_score, multi = v10.multiring_objective(stack, target, ids)
    e2 = polar_mode_energy(stack, v5.AXIS_UM, 2)
    e4 = polar_mode_energy(stack, v5.AXIS_UM, 4)
    rms = float(np.sqrt(np.mean(np.asarray(command, float) ** 2)))
    score = float(multiring_score + W_M2 * e2 + W_M4 * e4 + W_COMMAND_RMS * rms)
    return score, {
        "combined_score": score,
        "multiring_score": float(multiring_score),
        "m2_energy": float(e2),
        "m4_energy": float(e4),
        "command_rms_rad": rms,
        "multiring": multi,
    }


def evaluate(config, z_abs, target_det, command, pcoef, acoef) -> dict:
    command = np.asarray(command, float)
    if float(np.sqrt(np.mean(command ** 2))) > MAX_COMMAND_RMS_RAD:
        return {"valid": False, "reason": "command_rms_limit", "score": float("inf")}
    route = v10.slm2_command_route(config, v10.OPT_N, command, pcoef, acoef)
    winding, top_ok = v10.topology(route)
    if not top_ok:
        return {"valid": False, "reason": "q20_topology", "score": float("inf"), "winding": winding}
    stack = validation_stack(route, z_abs)
    score, metrics = objective_from_stack(stack, target_det, command)
    return {"valid": True, "reason": "ok", "score": float(score), "metrics": metrics, "winding": winding}


def choose_topology_safe_seed(config, z_abs, target_det, base_command, pcoef, acoef, out: Path):
    rows = []
    best = None
    for alpha in SEED_ALPHAS:
        cmd = float(alpha) * np.asarray(base_command, float)
        ev = evaluate(config, z_abs, target_det, cmd, pcoef, acoef)
        row = {"alpha": float(alpha), **{k: v for k, v in ev.items() if k != "stack"}}
        rows.append(row)
        print(json.dumps(row, indent=2))
        if ev.get("valid") and (best is None or float(ev["score"]) < float(best[1]["score"])):
            best = (cmd, ev, float(alpha))
    (out / "v12_seed_sweep.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    if best is None:
        zero = np.zeros_like(base_command, dtype=float)
        ev = evaluate(config, z_abs, target_det, zero, pcoef, acoef)
        if not ev.get("valid"):
            raise RuntimeError("even the zero-command diagnosed model does not preserve q=20")
        best = (zero, ev, 0.0)
    return best


def coordinate_descent(config, z_abs, target_det, seed_command, pcoef, acoef, grid: dict, out: Path):
    bases = {(m, q): harmonic_basis(grid, m, q) for m, q in MODE_ORDER}
    coeff = {(m, q): 0.0 for m, q in MODE_ORDER}
    command = np.asarray(seed_command, float).copy()
    current = evaluate(config, z_abs, target_det, command, pcoef, acoef)
    if not current.get("valid"):
        raise RuntimeError("v12 seed must preserve q=20")
    history = [{"round": -1, "step_rad": 0.0, "mode": "seed", "accepted": True,
                "score": float(current["score"]), "metrics": current["metrics"],
                "coefficients_rad": {f"m{m}_{q}": float(c) for (m, q), c in coeff.items()}}]

    for round_id, step in enumerate(STEP_SCHEDULE_RAD):
        improved_round = False
        for m, q in MODE_ORDER:
            key = (m, q)
            local_best = (float(current["score"]), 0.0, current, command)
            for direction in (-1.0, 1.0):
                trial_coeff = float(coeff[key] + direction * step)
                if abs(trial_coeff) > MAX_MODE_COEFF_RAD:
                    continue
                trial = command + float(direction * step) * bases[key]
                ev = evaluate(config, z_abs, target_det, trial, pcoef, acoef)
                row = {
                    "round": int(round_id), "step_rad": float(step), "mode": f"m{m}_{q}",
                    "direction": float(direction), "trial_coefficient_rad": trial_coeff,
                    "valid": bool(ev.get("valid", False)), "reason": ev.get("reason"),
                    "score": None if not np.isfinite(ev.get("score", np.inf)) else float(ev["score"]),
                    "metrics": ev.get("metrics"), "winding": ev.get("winding"),
                }
                history.append(row)
                print(json.dumps(row, indent=2))
                if ev.get("valid") and float(ev["score"]) + 2e-4 < local_best[0]:
                    local_best = (float(ev["score"]), float(direction), ev, trial)
            if local_best[1] != 0.0:
                coeff[key] = float(coeff[key] + local_best[1] * step)
                command = np.asarray(local_best[3], float)
                current = local_best[2]
                improved_round = True
                history.append({
                    "round": int(round_id), "step_rad": float(step), "mode": f"m{m}_{q}",
                    "accepted": True, "coefficient_rad": float(coeff[key]),
                    "score": float(current["score"]), "metrics": current["metrics"],
                })
        # Continue to finer steps even if one scale stalls; this is useful when
        # topology clips a large step but permits a smaller one.
        history.append({"round": int(round_id), "round_complete": True,
                        "improved": bool(improved_round), "score": float(current["score"]),
                        "coefficients_rad": {f"m{m}_{q}": float(c) for (m, q), c in coeff.items()}})

    (out / "v12_harmonic_search_history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    np.save(out / "v12_model_space_slm2_command_rad.npy", np.asarray(command, np.float32))
    return command, current, coeff, history


def plot_validation_before_after(config, z_abs, target_det, seed_command, command, pcoef, acoef, out: Path) -> None:
    seed_route = v10.slm2_command_route(config, v10.OPT_N, seed_command, pcoef, acoef)
    corr_route = v10.slm2_command_route(config, v10.OPT_N, command, pcoef, acoef)
    ids = np.asarray(v10.VALID_PLANES, int)
    before = v5.detector_stack(seed_route, np.asarray(z_abs)[ids])
    after = v5.detector_stack(corr_route, np.asarray(z_abs)[ids])
    target = np.asarray(target_det)[ids]
    ext = [v5.AXIS_UM[0], v5.AXIS_UM[-1], v5.AXIS_UM[0], v5.AXIS_UM[-1]]
    fig, axs = plt.subplots(3, len(ids), figsize=(13.5, 9.2), constrained_layout=True)
    for col, iz in enumerate(ids):
        for row, (stack, label) in enumerate(((before, "topology-safe seed"), (target, "concentric target"), (after, "v12 harmonic refined"))):
            axs[row, col].imshow(stack[col], origin="lower", extent=ext, cmap=THERMAL,
                                 vmin=0, vmax=1, interpolation="nearest")
            axs[row, col].set_aspect("equal"); axs[row, col].set_xticks([]); axs[row, col].set_yticks([])
            if row == 0:
                axs[row, col].set_title(f"plane {int(iz)}")
            if col == 0:
                axs[row, col].set_ylabel(label, fontweight="bold")
    fig.suptitle("q=20 v12: explicit angular-harmonic suppression on propagated validation planes",
                 fontsize=14, fontweight="bold")
    savefig(fig, out / "v12_validation_harmonic_before_after")


def run(source_dir: Path, candidate_json: Path, residual_json: Path, out: Path) -> dict:
    source_dir = Path(source_dir); out = Path(out); out.mkdir(parents=True, exist_ok=True)
    config, candidate = v9.config_from_files(candidate_json, source_dir)
    pcoef, acoef, _ = v9.frozen_residual(residual_json)
    summary0 = json.loads((source_dir / "run_summary.json").read_text(encoding="utf-8"))
    z_rel = np.asarray(summary0["data"]["z_relative_mm"], float)
    z0 = float(candidate["physical_nuisance"]["selected_z0_mm"])
    z_abs = (z0 + z_rel) * 1e-3

    # v10 contributes a propagated multi-plane starting direction.  It is not
    # accepted blindly; v12 independently finds a weak topology-safe seed.
    _, target_amp, _, _ = v10.nominal_target(config, z_abs)
    base_command, v10_history, v9_history, _, prop_meta, v10_best = v10.optimise_multiplane(
        config, z_abs, target_amp, pcoef, acoef, out / "v10_regenerated"
    )
    target_det = v10.evaluation_target(config, z_abs)
    seed_command, seed_eval, seed_alpha = choose_topology_safe_seed(
        config, z_abs, target_det, base_command, pcoef, acoef, out
    )

    base_route = build_system_route("V20", grid_n=RELAY_N, config=config, window_m=FIT_WINDOW_M)
    relay_grid = base_route["grid"]
    command, final_eval, coeff, history = coordinate_descent(
        config, z_abs, target_det, seed_command, pcoef, acoef, relay_grid, out
    )
    plot_validation_before_after(config, z_abs, target_det, seed_command, command, pcoef, acoef, out)

    # One expensive 4096 production validation after all tuning is frozen.
    prod = v11.production(config, z_abs, z_rel, target_det, command, pcoef, acoef, out)
    held = prod["metrics"]["legacy_heldout"]
    pm = held["detector_positive_multiring"]; cm = held["detector_corrected_multiring"]
    reductions = {
        "multiring_score": float(1.0 - held["detector_corrected_multiring_score"] / max(held["detector_positive_multiring_score"], EPS)),
        "ring_intensity_cv": float(1.0 - cm["mean_ring_intensity_cv"] / max(pm["mean_ring_intensity_cv"], EPS)),
        "angular_harmonic_energy": float(1.0 - cm["mean_angular_harmonic_energy"] / max(pm["mean_angular_harmonic_energy"], EPS)),
        "ring_radius_std": float(1.0 - cm["mean_ring_radius_std_um"] / max(pm["mean_ring_radius_std_um"], EPS)),
    }

    # Explicit m=2/m=4 audit on the legacy odd-plane detector prediction.
    arrays = np.load(out / "v11_4096_display_arrays.npz")
    legacy = np.asarray(v10.LEGACY_HELD, int)
    bstack = np.asarray(arrays["detector_positive"])[legacy]
    cstack = np.asarray(arrays["detector_corrected"])[legacy]
    harmonic_audit = {
        "positive_m2_energy": polar_mode_energy(bstack, v5.AXIS_UM, 2),
        "corrected_m2_energy": polar_mode_energy(cstack, v5.AXIS_UM, 2),
        "positive_m4_energy": polar_mode_energy(bstack, v5.AXIS_UM, 4),
        "corrected_m4_energy": polar_mode_energy(cstack, v5.AXIS_UM, 4),
    }
    harmonic_audit["m2_reduction_fraction"] = float(1.0 - harmonic_audit["corrected_m2_energy"] / max(harmonic_audit["positive_m2_energy"], EPS))
    harmonic_audit["m4_reduction_fraction"] = float(1.0 - harmonic_audit["corrected_m4_energy"] / max(harmonic_audit["positive_m4_energy"], EPS))

    acceptance = {
        "q20_topology_preserved": bool(prod["topology_q20_all_contours"]),
        "legacy_heldout_reductions": reductions,
        "legacy_heldout_specific_harmonics": harmonic_audit,
        "passes_concentricity_gate": bool(
            prod["topology_q20_all_contours"]
            and reductions["multiring_score"] >= 0.25
            and reductions["ring_intensity_cv"] >= 0.18
            and reductions["angular_harmonic_energy"] >= 0.22
            and reductions["ring_radius_std"] >= 0.10
            and harmonic_audit["m4_reduction_fraction"] >= 0.35
            and harmonic_audit["m2_reduction_fraction"] >= 0.15
        ),
    }

    result = {
        "status": "q20_topology_safe_harmonic_refinement_v12",
        "seed_alpha": float(seed_alpha),
        "seed_validation": seed_eval,
        "final_validation": final_eval,
        "modal_coefficients_rad": {f"m{m}_{q}": float(c) for (m, q), c in coeff.items()},
        "production_validation": prod,
        "acceptance": acceptance,
        "target_policy": "explicit concentric nominal multi-plane target; measured angular pattern is never rewarded",
        "optimisation_policy": "topology-safe SLM2 m=2/4/6/8 modal coordinate descent through the full finite-4F + axicon route",
        "topology_policy": "hard q=20 gate on every 1.0--1.5 mm contour for every accepted candidate",
        "v10_best_iteration": v10_best,
        "v10_optimisation_history": v10_history,
        "v9_initialiser_history": v9_history,
        "propagation_support": prop_meta,
        "hardware_ready": False,
        "evidence_boundary": "corrected fields are numerical model-space predictions only; no corrected BeamGage image is experimental evidence",
    }
    (out / "v12_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source-dir", type=Path, default=EXP / "outputs" / "digital_twin_correction")
    p.add_argument("--candidate-json", type=Path, default=EXP / "candidates" / "q20_detector_aware_model_v3_candidate.json")
    p.add_argument("--residual-json", type=Path, default=EXP / "candidates" / "q20_miao_initialized_complex_residual_v1.json")
    p.add_argument("--out", type=Path, default=ROOT / "outputs" / "validation" / "q20_slm2_harmonic_refine_v12")
    a = p.parse_args()
    run(a.source_dir, a.candidate_json, a.residual_json, a.out)


if __name__ == "__main__":
    main()
