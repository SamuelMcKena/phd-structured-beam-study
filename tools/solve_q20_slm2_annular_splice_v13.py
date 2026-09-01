"""q=20 correction v13: annular-spliced topology-safe modal refinement.

Motivation
----------
A single global correction can be insufficient for a long Bessel focus because
axial positions are fed predominantly by different annular regions of the
conical-wave aperture.  Miao et al. retrieve aberration on axicon/input annuli,
and recent Bessel-focus adaptive-optics work uses annular Zernike-like modes and
space-varying/annularly spliced correction.  v13 adapts that principle to the
bench-matched q=20 digital twin.

This solver remains deliberately conservative:
  * the frozen Miao-initialised residual is not re-fit;
  * measured angular intensity structure is never used as a correction target;
  * a weak v10 multi-plane direction supplies only a topology-safe seed;
  * one static SLM2 command is built from overlapping radial annuli carrying
    m=2 astigmatism, m=3 trefoil and m=4 fourfold quadratures;
  * each proposed step is propagated through the commanded SLM2 -> finite 4F
    + +1 iris -> frozen residual -> refractive axicon -> multi-z route;
  * q=20 winding on every 1.0--1.5 mm contour is a hard gate;
  * final acceptance is based on visible multi-ring circularity and explicit
    m=2/m=3/m=4 detector-plane harmonic suppression at 4096 samples.

The radial bands below are MODEL-SPACE control bands on the SLM2 grid.  They are
not claimed as a measured SLM2-to-axicon conjugacy map.  A hardware-ready mask
still requires SLM LUT/coordinate calibration and experimental closure.
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
import solve_q20_slm2_harmonic_refine_v12 as v12  # noqa: E402
from real_bmg_digital_twin_correction import FIT_WINDOW_M, RELAY_N  # noqa: E402
from vbb_study.digital_twin.vortex_system_route import build_system_route  # noqa: E402

EPS = np.finfo(float).tiny
THERMAL = "inferno"
PROD_N = 4096

# Overlapping smooth control annuli.  The overlap prevents phase seams and lets
# coordinate descent synthesize a continuous depth-dependent correction.
ANNULI_M = (
    ("inner", 0.35e-3, 1.05e-3, 1.25e-3),
    ("middle", 0.90e-3, 1.45e-3, 1.70e-3),
    ("outer", 1.35e-3, 1.90e-3, 2.20e-3),
)
MODES = (2, 3, 4)
QUADRATURES = ("cos", "sin")
STEP_SCHEDULE_RAD = (0.26, 0.15, 0.085, 0.045)
MAX_COEFF_RAD = 0.75
MAX_COMMAND_RMS_RAD = 1.15

# m=3 is included because trefoil is a reported important axicon/Bessel
# aberration; m=4 remains the strongest explicit penalty for the observed cross.
W_M2 = 0.45
W_M3 = 0.50
W_M4 = 1.10
W_RMS = 0.010


def savefig(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=500, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def smoothstep01(x: np.ndarray) -> np.ndarray:
    t = np.clip(np.asarray(x, float), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def annular_window(R: np.ndarray, inner_m: float, peak_m: float, outer_m: float) -> np.ndarray:
    """C1 smooth annular bump: zero at inner/outer, unity around peak."""
    if not (0.0 <= inner_m < peak_m < outer_m):
        raise ValueError("annular radii must satisfy inner < peak < outer")
    rise = smoothstep01((R - inner_m) / max(peak_m - inner_m, EPS))
    fall = 1.0 - smoothstep01((R - peak_m) / max(outer_m - peak_m, EPS))
    return np.clip(rise * fall, 0.0, 1.0)


def annular_mode_basis(grid: dict, band, m: int, quadrature: str) -> np.ndarray:
    X = np.asarray(grid["X"], float)
    Y = np.asarray(grid["Y"], float)
    R = np.asarray(grid["R"], float)
    theta = np.arctan2(Y, X)
    _, rin, rpeak, rout = band
    window = annular_window(R, float(rin), float(rpeak), float(rout))
    angular = np.cos(float(m) * theta) if quadrature == "cos" else np.sin(float(m) * theta)
    b = window * angular
    support = window > 0.05
    if not np.any(support):
        raise RuntimeError("empty annular control support")
    b -= float(np.average(b[support], weights=np.maximum(window[support], 1e-6)))
    rms = float(np.sqrt(np.average(b[support] ** 2, weights=np.maximum(window[support], 1e-6))))
    return b / max(rms, EPS)


def objective_from_stack(stack: np.ndarray, target_det: np.ndarray, command: np.ndarray) -> tuple[float, dict]:
    ids = np.arange(len(v10.VALID_PLANES), dtype=int)
    target = np.asarray(target_det, float)[np.asarray(v10.VALID_PLANES, int)]
    multiring_score, multi = v10.multiring_objective(stack, target, ids)
    e2 = v12.polar_mode_energy(stack, v5.AXIS_UM, 2)
    e3 = v12.polar_mode_energy(stack, v5.AXIS_UM, 3)
    e4 = v12.polar_mode_energy(stack, v5.AXIS_UM, 4)
    rms = float(np.sqrt(np.mean(np.asarray(command, float) ** 2)))
    score = float(multiring_score + W_M2 * e2 + W_M3 * e3 + W_M4 * e4 + W_RMS * rms)
    return score, {
        "combined_score": score,
        "multiring_score": float(multiring_score),
        "m2_energy": float(e2), "m3_energy": float(e3), "m4_energy": float(e4),
        "command_rms_rad": rms, "multiring": multi,
    }


def evaluate(config, z_abs, target_det, command, pcoef, acoef) -> dict:
    command = np.asarray(command, float)
    if float(np.sqrt(np.mean(command ** 2))) > MAX_COMMAND_RMS_RAD:
        return {"valid": False, "reason": "command_rms_limit", "score": float("inf")}
    route = v10.slm2_command_route(config, v10.OPT_N, command, pcoef, acoef)
    winding, top_ok = v10.topology(route)
    if not top_ok:
        return {"valid": False, "reason": "q20_topology", "score": float("inf"), "winding": winding}
    ids = np.asarray(v10.VALID_PLANES, int)
    stack = v5.detector_stack(route, np.asarray(z_abs, float)[ids])
    score, metrics = objective_from_stack(stack, target_det, command)
    return {"valid": True, "reason": "ok", "score": float(score), "metrics": metrics, "winding": winding}


def topology_safe_seed(config, z_abs, target_det, base_command, pcoef, acoef, out: Path):
    # Reuse v12's conservative alpha sweep; v13's novelty is the annular modal
    # refinement, not a different initialiser.
    return v12.choose_topology_safe_seed(config, z_abs, target_det, base_command, pcoef, acoef, out)


def coordinate_descent(config, z_abs, target_det, seed_command, pcoef, acoef, grid: dict, out: Path):
    keys = [(band[0], m, q) for band in ANNULI_M for m in MODES for q in QUADRATURES]
    bases = {(band[0], m, q): annular_mode_basis(grid, band, m, q)
             for band in ANNULI_M for m in MODES for q in QUADRATURES}
    coeff = {key: 0.0 for key in keys}
    command = np.asarray(seed_command, float).copy()
    current = evaluate(config, z_abs, target_det, command, pcoef, acoef)
    if not current.get("valid"):
        raise RuntimeError("v13 seed is not topology safe")
    history = [{"round": -1, "accepted": True, "mode": "seed", "score": float(current["score"]),
                "metrics": current["metrics"]}]

    # m=4 first in each band attacks the visible cross; m=2/m=3 then remove
    # astigmatic/trefoil residuals which can otherwise alias into ring wobble.
    ordered_keys = []
    for band in ANNULI_M:
        name = band[0]
        for m in (4, 2, 3):
            for q in QUADRATURES:
                ordered_keys.append((name, m, q))

    for round_id, step in enumerate(STEP_SCHEDULE_RAD):
        improved = False
        for key in ordered_keys:
            local = (float(current["score"]), 0.0, current, command)
            for direction in (-1.0, 1.0):
                trial_coeff = float(coeff[key] + direction * step)
                if abs(trial_coeff) > MAX_COEFF_RAD:
                    continue
                trial = command + float(direction * step) * bases[key]
                ev = evaluate(config, z_abs, target_det, trial, pcoef, acoef)
                row = {
                    "round": int(round_id), "step_rad": float(step),
                    "band": key[0], "m": int(key[1]), "quadrature": key[2],
                    "direction": float(direction), "trial_coefficient_rad": trial_coeff,
                    "valid": bool(ev.get("valid", False)), "reason": ev.get("reason"),
                    "score": None if not np.isfinite(ev.get("score", np.inf)) else float(ev["score"]),
                    "metrics": ev.get("metrics"), "winding": ev.get("winding"),
                }
                history.append(row); print(json.dumps(row, indent=2))
                if ev.get("valid") and float(ev["score"]) + 2e-4 < local[0]:
                    local = (float(ev["score"]), float(direction), ev, trial)
            if local[1] != 0.0:
                coeff[key] = float(coeff[key] + local[1] * step)
                command = np.asarray(local[3], float)
                current = local[2]
                improved = True
                history.append({"round": int(round_id), "accepted": True, "band": key[0],
                                "m": int(key[1]), "quadrature": key[2],
                                "coefficient_rad": float(coeff[key]), "score": float(current["score"]),
                                "metrics": current["metrics"]})
        history.append({"round": int(round_id), "round_complete": True, "improved": bool(improved),
                        "score": float(current["score"]),
                        "coefficients_rad": {f"{b}_m{m}_{q}": float(v) for (b, m, q), v in coeff.items()}})

    (out / "v13_annular_search_history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    np.save(out / "v13_model_space_slm2_command_rad.npy", np.asarray(command, np.float32))
    return command, current, coeff, history


def plot_validation(config, z_abs, target_det, seed_command, command, pcoef, acoef, out: Path):
    ids = np.asarray(v10.VALID_PLANES, int)
    seed_route = v10.slm2_command_route(config, v10.OPT_N, seed_command, pcoef, acoef)
    corr_route = v10.slm2_command_route(config, v10.OPT_N, command, pcoef, acoef)
    before = v5.detector_stack(seed_route, np.asarray(z_abs)[ids])
    after = v5.detector_stack(corr_route, np.asarray(z_abs)[ids])
    target = np.asarray(target_det)[ids]
    ext = [v5.AXIS_UM[0], v5.AXIS_UM[-1], v5.AXIS_UM[0], v5.AXIS_UM[-1]]
    fig, axs = plt.subplots(3, len(ids), figsize=(13.5, 9.2), constrained_layout=True)
    for c, iz in enumerate(ids):
        for r, (stack, label) in enumerate(((before, "topology-safe seed"),
                                            (target, "concentric target"),
                                            (after, "v13 annular-spliced"))):
            axs[r, c].imshow(stack[c], origin="lower", extent=ext, cmap=THERMAL,
                             vmin=0, vmax=1, interpolation="nearest")
            axs[r, c].set_aspect("equal"); axs[r, c].set_xticks([]); axs[r, c].set_yticks([])
            if r == 0: axs[r, c].set_title(f"plane {int(iz)}")
            if c == 0: axs[r, c].set_ylabel(label, fontweight="bold")
    fig.suptitle("q=20 v13: annular-spliced astigmatism/trefoil/fourfold correction",
                 fontsize=14, fontweight="bold")
    savefig(fig, out / "v13_validation_annular_before_after")


def run(source_dir: Path, candidate_json: Path, residual_json: Path, out: Path) -> dict:
    source_dir = Path(source_dir); out = Path(out); out.mkdir(parents=True, exist_ok=True)
    config, candidate = v9.config_from_files(candidate_json, source_dir)
    pcoef, acoef, _ = v9.frozen_residual(residual_json)
    summary0 = json.loads((source_dir / "run_summary.json").read_text(encoding="utf-8"))
    z_rel = np.asarray(summary0["data"]["z_relative_mm"], float)
    z0 = float(candidate["physical_nuisance"]["selected_z0_mm"])
    z_abs = (z0 + z_rel) * 1e-3

    _, target_amp, _, _ = v10.nominal_target(config, z_abs)
    base_command, v10_history, v9_history, _, prop_meta, v10_best = v10.optimise_multiplane(
        config, z_abs, target_amp, pcoef, acoef, out / "v10_regenerated"
    )
    target_det = v10.evaluation_target(config, z_abs)
    seed_command, seed_eval, seed_alpha = topology_safe_seed(
        config, z_abs, target_det, base_command, pcoef, acoef, out
    )

    base_route = build_system_route("V20", grid_n=RELAY_N, config=config, window_m=FIT_WINDOW_M)
    command, final_eval, coeff, history = coordinate_descent(
        config, z_abs, target_det, seed_command, pcoef, acoef, base_route["grid"], out
    )
    plot_validation(config, z_abs, target_det, seed_command, command, pcoef, acoef, out)

    prod = v11.production(config, z_abs, z_rel, target_det, command, pcoef, acoef, out)
    held = prod["metrics"]["legacy_heldout"]
    pm = held["detector_positive_multiring"]; cm = held["detector_corrected_multiring"]
    reductions = {
        "multiring_score": float(1.0 - held["detector_corrected_multiring_score"] / max(held["detector_positive_multiring_score"], EPS)),
        "ring_intensity_cv": float(1.0 - cm["mean_ring_intensity_cv"] / max(pm["mean_ring_intensity_cv"], EPS)),
        "angular_harmonic_energy": float(1.0 - cm["mean_angular_harmonic_energy"] / max(pm["mean_angular_harmonic_energy"], EPS)),
        "ring_radius_std": float(1.0 - cm["mean_ring_radius_std_um"] / max(pm["mean_ring_radius_std_um"], EPS)),
    }
    arrays = np.load(out / "v11_4096_display_arrays.npz")
    legacy = np.asarray(v10.LEGACY_HELD, int)
    bstack = np.asarray(arrays["detector_positive"])[legacy]
    cstack = np.asarray(arrays["detector_corrected"])[legacy]
    harmonic = {}
    for m in (2, 3, 4):
        bp = v12.polar_mode_energy(bstack, v5.AXIS_UM, m)
        cp = v12.polar_mode_energy(cstack, v5.AXIS_UM, m)
        harmonic[f"positive_m{m}_energy"] = float(bp)
        harmonic[f"corrected_m{m}_energy"] = float(cp)
        harmonic[f"m{m}_reduction_fraction"] = float(1.0 - cp / max(bp, EPS))

    acceptance = {
        "q20_topology_preserved": bool(prod["topology_q20_all_contours"]),
        "legacy_heldout_reductions": reductions,
        "legacy_heldout_specific_harmonics": harmonic,
        "passes_concentricity_gate": bool(
            prod["topology_q20_all_contours"]
            and reductions["multiring_score"] >= 0.28
            and reductions["ring_intensity_cv"] >= 0.20
            and reductions["angular_harmonic_energy"] >= 0.24
            and reductions["ring_radius_std"] >= 0.10
            and harmonic["m4_reduction_fraction"] >= 0.40
            and harmonic["m2_reduction_fraction"] >= 0.15
            and harmonic["m3_reduction_fraction"] >= 0.10
        ),
    }

    result = {
        "status": "q20_annular_spliced_modal_refinement_v13",
        "seed_alpha": float(seed_alpha), "seed_validation": seed_eval,
        "final_validation": final_eval,
        "annular_control_bands_m": [{"name": n, "inner_m": a, "peak_m": b, "outer_m": c}
                                    for n, a, b, c in ANNULI_M],
        "modal_coefficients_rad": {f"{b}_m{m}_{q}": float(v) for (b, m, q), v in coeff.items()},
        "production_validation": prod, "acceptance": acceptance,
        "target_policy": "explicit concentric nominal multi-plane target; measured angular structure is never rewarded",
        "optimisation_policy": "overlapping annular m=2/m=3/m=4 coordinate descent through commanded SLM2 + finite-4F/+1 iris + axicon route",
        "topology_policy": "hard q=20 gate on every 1.0--1.5 mm contour for every accepted candidate",
        "annular_provenance": "radial bands are model-space control bands inspired by annular Bessel AO; no measured SLM2-to-axicon conjugacy is claimed",
        "v10_best_iteration": v10_best, "v10_optimisation_history": v10_history,
        "v9_initialiser_history": v9_history, "propagation_support": prop_meta,
        "hardware_ready": False,
        "evidence_boundary": "corrected fields are numerical model-space predictions only; no corrected BeamGage image is experimental evidence",
    }
    (out / "v13_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2)); return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source-dir", type=Path, default=EXP / "outputs" / "digital_twin_correction")
    p.add_argument("--candidate-json", type=Path, default=EXP / "candidates" / "q20_detector_aware_model_v3_candidate.json")
    p.add_argument("--residual-json", type=Path, default=EXP / "candidates" / "q20_miao_initialized_complex_residual_v1.json")
    p.add_argument("--out", type=Path, default=ROOT / "outputs" / "validation" / "q20_slm2_annular_splice_v13")
    a = p.parse_args(); run(a.source_dir, a.candidate_json, a.residual_json, a.out)


if __name__ == "__main__":
    main()
