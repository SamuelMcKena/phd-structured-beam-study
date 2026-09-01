"""Hybrid Miao + full-route phase-only SLM2 correction for real q=20 data.

Scientific purpose
------------------
The previous q=20 correction candidates were primarily rewarded for numerical
closure to a nominal/detected image.  That can improve Pearson/NRMSE while
leaving the principal vortex-Bessel annulus visibly non-concentric.  This
solver makes annular concentricity an explicit multi-plane objective.

The target is deliberately hybrid:

* the real pre-correction BeamGage stack supplies the radial intensity content;
  each measured plane is azimuthally symmetrised so the target preserves the
  measured ring radii/relative radial structure without preserving the angular
  distortion we are trying to remove;
* the adapted Miao et al. Bessel-modal reconstruction supplies an independent
  intensity-only concentric annular prior;
* the Miao-initialised detector-aware residual fit supplies the phase and
  diagnostic amplitude nuisance at the selected-order field immediately before
  the axicon;
* the correction itself remains phase-only at SLM2 and is propagated through
  the explicit carrier + finite 4F/+1 iris + refractive-axicon route.

Hyper-parameter/strength selection uses an inner split of the even-index planes.
Odd-index planes are reported only after the correction is frozen.  Because the
broader project has already inspected those odd planes in earlier model audits,
they are described as a legacy held-out check rather than a pristine blind test.

No corrected BeamGage image is experimental evidence.  All corrected images
emitted here are numerical model-space predictions pending bench coordinate/LUT
calibration and a measured post-correction z-stack.
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
import pandas as pd
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
EXP = ROOT / "notebooks" / "experimental" / "axicon_aberration_correction"
for p in (ROOT, TOOLS, EXP):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import fit_q20_detector_aware_model_v2 as v2  # noqa: E402
import fit_q20_detector_aware_model_v3 as v3  # noqa: E402
import solve_q20_slm2_precompensation_v4 as v4  # noqa: E402
from optimize_q20_slm2_detector_closure_v2 import (  # noqa: E402
    phase_basis,
    phase_from_coefficients,
    structure_metrics,
)
from real_bmg_digital_twin_correction import (  # noqa: E402
    FIT_N,
    FIT_WINDOW_M,
    PIXEL_M,
    Q,
    RELAY_N,
)
from vbb_study.digital_twin.detector_response import plane_normalise, sample_camera_response  # noqa: E402
from vbb_study.digital_twin.phase2a_contracts import canonical_hardware_manifest, hardware_value  # noqa: E402
from vbb_study.digital_twin.vortex_continuous_propagation import build_fixed_support_spectrum, native_field_at_z  # noqa: E402
from vbb_study.digital_twin.vortex_system_route import build_multirate_system_route  # noqa: E402
from vbb_study.viz_fields import phase_winding  # noqa: E402

EPS = np.finfo(float).tiny
THERMAL = "inferno"
AXIS_UM = np.linspace(-180.0, 180.0, 241)
INNER_TRAIN = np.asarray([0, 4, 8, 12, 16], dtype=int)
INNER_VALID = np.asarray([2, 6, 10, 14], dtype=int)
LEGACY_HELD = np.asarray([1, 3, 5, 7, 9, 11, 13, 15, 17], dtype=int)
MIAO_BLEND = 0.35
SWEEP_N = 2048
PROD_N = 4096
ALPHAS = np.asarray([0.45, 0.60, 0.75, 0.90, 1.00, 1.10], float)
WINDING_RADII_MM = (1.0, 1.1, 1.2, 1.3, 1.4, 1.5)


def savefig(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=500, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def normalise(a: np.ndarray) -> np.ndarray:
    return plane_normalise(np.maximum(np.asarray(a, float), 0.0))


def radial_symmetrise_plane(image: np.ndarray, axis_um: np.ndarray, dr_um: float = 1.0) -> np.ndarray:
    """Return a smooth radial reconstruction of one measured intensity plane."""
    image = np.asarray(image, float)
    X, Y = np.meshgrid(axis_um, axis_um, indexing="xy")
    R = np.hypot(X, Y)
    edges = np.arange(0.0, float(np.max(R)) + dr_um, dr_um)
    centres = 0.5 * (edges[:-1] + edges[1:])
    ids = np.digitize(R.ravel(), edges) - 1
    vals = image.ravel()
    sums = np.bincount(np.clip(ids, 0, len(centres)-1), weights=vals, minlength=len(centres))
    counts = np.bincount(np.clip(ids, 0, len(centres)-1), minlength=len(centres))
    prof = sums / np.maximum(counts, 1)
    prof = ndimage.gaussian_filter1d(prof, sigma=0.8, mode="nearest")
    out = np.interp(R.ravel(), centres, prof, left=prof[0], right=prof[-1]).reshape(R.shape)
    return out / max(float(np.max(out)), EPS)


def hybrid_target(measured: np.ndarray, miao_pred: np.ndarray, axis_um: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sym = np.stack([radial_symmetrise_plane(p, axis_um) for p in normalise(measured)])
    miao = normalise(miao_pred)
    if miao.shape != sym.shape:
        raise ValueError(f"Miao target shape {miao.shape} != measured shape {sym.shape}")
    target = normalise((1.0 - MIAO_BLEND) * sym + MIAO_BLEND * miao)
    return sym, target


def candidate_coefficients(candidate: dict, crosscheck: dict) -> tuple[np.ndarray, np.ndarray, str]:
    """Use the train-derived Miao-initialised residual when available."""
    test = crosscheck.get("initializer_test", {})
    p = test.get("final_phase_coefficients_rad")
    a = test.get("final_log_amplitude_coefficients")
    if p is not None and a is not None:
        return np.asarray(p, float), np.asarray(a, float), "miao_initialised_train_optimised"
    return (
        np.asarray(candidate["phase_coefficients_rad"], float),
        np.asarray(candidate["log_amplitude_coefficients"], float),
        "accepted_v3_fallback",
    )


def residual_maps(grid: dict, pcoef: np.ndarray, acoef: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pb, ab = v2.residual_bases(grid)
    phase, _, amp = v2.maps_from_coefficients(pcoef, acoef, pb, ab)
    return phase, amp


def build_route(config, propagation_n: int, *, slm2_phase: np.ndarray | None, pcoef: np.ndarray, acoef: np.ndarray):
    nominal = build_multirate_system_route(
        f"V{Q}", relay_grid_n=RELAY_N, propagation_grid_n=int(propagation_n),
        window_m=FIT_WINDOW_M, config=config,
        slm2_static_phase_map_rad=slm2_phase,
    )
    err_phase, err_amp = residual_maps(nominal["grid"], pcoef, acoef)
    return build_multirate_system_route(
        f"V{Q}", relay_grid_n=RELAY_N, propagation_grid_n=int(propagation_n),
        window_m=FIT_WINDOW_M, config=config,
        slm2_static_phase_map_rad=slm2_phase,
        axicon_input_phase_map_rad=err_phase,
        axicon_input_amplitude_map=err_amp,
    )


def detector_stack(route_result: dict, z_abs: np.ndarray) -> np.ndarray:
    wl = float(hardware_value(canonical_hardware_manifest(), "wavelength_m"))
    prop = build_fixed_support_spectrum(
        route_result["post_axicon"], route_result["grid"], wavelength_m=wl,
        z_max_m=max(0.002, float(np.max(np.abs(z_abs))) + 0.002),
        minimum_retained_spectral_power=0.98,
    )
    native = np.stack([np.abs(native_field_at_z(prop, float(z))) ** 2 for z in np.asarray(z_abs, float)])
    shown, _ = sample_camera_response(
        native, np.asarray(route_result["grid"]["x"], float), AXIS_UM * 1e-6,
        pixel_pitch_m=PIXEL_M, quadrature_n=3,
    )
    return normalise(shown)


def optical_stack(route_result: dict, z_abs: np.ndarray) -> np.ndarray:
    wl = float(hardware_value(canonical_hardware_manifest(), "wavelength_m"))
    prop = build_fixed_support_spectrum(
        route_result["post_axicon"], route_result["grid"], wavelength_m=wl,
        z_max_m=max(0.002, float(np.max(np.abs(z_abs))) + 0.002),
        minimum_retained_spectral_power=0.99,
    )
    x = np.asarray(route_result["grid"]["x"], float)
    pix = np.interp(AXIS_UM * 1e-6, x, np.arange(len(x)))
    yy, xx = np.meshgrid(pix, pix, indexing="ij")
    out = []
    for z in np.asarray(z_abs, float):
        field = native_field_at_z(prop, float(z))
        I = np.abs(field) ** 2
        crop = ndimage.map_coordinates(I, [yy, xx], order=1, mode="constant", cval=0.0)
        out.append(crop / max(float(np.max(crop)), EPS))
    return np.stack(out)


def mirror_rmse(stack: np.ndarray, ids: np.ndarray | None = None) -> float:
    arr = normalise(stack)
    if ids is not None:
        arr = arr[np.asarray(ids, int)]
    X, Y = np.meshgrid(AXIS_UM, AXIS_UM, indexing="xy")
    R = np.hypot(X, Y)
    mask = (R >= 18.0) & (R <= 145.0)
    vals = []
    for im in arr:
        ex = np.sqrt(np.mean((im[mask] - im[:, ::-1][mask]) ** 2))
        ey = np.sqrt(np.mean((im[mask] - im[::-1, :][mask]) ** 2))
        vals.append(0.5 * (ex + ey))
    return float(np.mean(vals))


def concentric_metrics(pred: np.ndarray, target: np.ndarray, ids: np.ndarray) -> dict:
    ids = np.asarray(ids, int)
    p = normalise(pred)[ids]
    t = normalise(target)[ids]
    X, Y = np.meshgrid(AXIS_UM, AXIS_UM, indexing="xy")
    roi = np.hypot(X, Y) <= 145.0
    rs, es = [], []
    for a, b in zip(p, t):
        av, bv = a[roi], b[roi]
        rs.append(float(np.corrcoef(av, bv)[0, 1]))
        es.append(float(np.sqrt(np.mean((av - bv) ** 2))))
    struct = structure_metrics(p, t, AXIS_UM, np.arange(len(p), dtype=int))
    return {
        "mean_r": float(np.mean(rs)),
        "mean_nrmse": float(np.mean(es)),
        "mirror_rmse": mirror_rmse(p),
        **{k: float(v) for k, v in struct.items()},
    }


def objective(m: dict) -> float:
    """Concentricity-first objective; scalar fidelity is deliberately secondary."""
    return float(
        0.55 * m["mean_nrmse"]
        + 0.10 * (1.0 - m["mean_r"])
        + 1.25 * m["mean_principal_ring_azimuth_cv"]
        + 0.45 * m["mean_sidelobe_profile_rmse"]
        + 0.30 * m["mean_opposite_peak_imbalance"]
        + 0.75 * m["mirror_rmse"]
    )


def weighted_linear_system(current: np.ndarray, derivatives: list[np.ndarray], target: np.ndarray, ids: np.ndarray):
    X, Y = np.meshgrid(AXIS_UM, AXIS_UM, indexing="xy")
    R = np.hypot(X, Y)
    roi = (R >= 12.0) & (R <= 145.0)
    J, y = [], []
    cur, tar = normalise(current), normalise(target)
    for iz in np.asarray(ids, int):
        t = tar[iz][roi]
        residual = t - cur[iz][roi]
        # Boost the principal-annulus/sidelobe region while retaining the full ring train.
        weight = np.sqrt(0.25 + 0.75 * np.sqrt(np.clip(t, 0.0, 1.0)))
        J.append(weight[:, None] * np.column_stack([d[iz][roi] for d in derivatives]))
        y.append(weight * residual)
    return np.vstack(J), np.concatenate(y)


def optimise_compact(config, z_abs: np.ndarray, target: np.ndarray, pcoef: np.ndarray, acoef: np.ndarray, out: Path):
    relay = build_multirate_system_route(
        f"V{Q}", relay_grid_n=RELAY_N, propagation_grid_n=FIT_N,
        window_m=FIT_WINDOW_M, config=config,
    )
    basis, names = phase_basis(relay["relay_route"]["grid"])
    coeff = np.zeros(len(names), float)

    def simulate(c: np.ndarray) -> np.ndarray:
        phase = phase_from_coefficients(basis, c)
        rr = build_route(config, FIT_N, slm2_phase=phase, pcoef=pcoef, acoef=acoef)
        return detector_stack(rr, z_abs)

    current = simulate(coeff)
    history = []
    for iteration in range(4):
        base_train = concentric_metrics(current, target, INNER_TRAIN)
        base_obj = objective(base_train)
        derivatives = []
        delta = 0.11 if iteration == 0 else 0.07
        for j in range(len(coeff)):
            cp, cm = coeff.copy(), coeff.copy()
            cp[j] += delta; cm[j] -= delta
            plus, minus = simulate(cp), simulate(cm)
            derivatives.append((plus - minus) / (2.0 * delta))
            del plus, minus
        J, y = weighted_linear_system(current, derivatives, target, INNER_TRAIN)
        ridge = 2.0e-2
        step = np.linalg.solve(J.T @ J + ridge * np.eye(J.shape[1]), J.T @ y)
        step = np.clip(step, -0.26, 0.26)
        chosen = (base_obj, coeff.copy(), current, 0.0, base_train)
        for strength in (1.0, 0.65, 0.40, 0.22):
            trial_c = np.clip(coeff + strength * step, -1.1, 1.1)
            trial = simulate(trial_c)
            met = concentric_metrics(trial, target, INNER_TRAIN)
            obj = objective(met)
            if obj < chosen[0]:
                chosen = (obj, trial_c, trial, strength, met)
        accepted = chosen[3] > 0.0
        coeff, current = chosen[1], chosen[2]
        history.append({
            "iteration": iteration + 1,
            "accepted": bool(accepted),
            "step_strength": float(chosen[3]),
            "train_objective": float(chosen[0]),
            "train_metrics": chosen[4],
            "inner_validation_metrics": concentric_metrics(current, target, INNER_VALID),
            "coefficients_rad": coeff.tolist(),
        })
        del derivatives, J, y, step
        gc.collect()
        if not accepted:
            break

    phase = phase_from_coefficients(basis, coeff)
    np.save(out / "hybrid_compact_slm2_full_strength_phase_rad.npy", phase.astype(np.float32))
    return phase, coeff, names, current, history


def winding(route_result: dict) -> tuple[dict, bool]:
    vals, ok = {}, True
    for rmm in WINDING_RADII_MM:
        w = float(phase_winding(route_result["post_axicon"], route_result["grid"], rmm * 1e-3, n_phi=720))
        vals[f"radius_{rmm:.1f}_mm"] = w
        ok &= abs(w - float(Q)) <= 0.25
    return vals, bool(ok)


def evaluate_alpha(alpha: float, N: int, ids: np.ndarray, config, z_abs: np.ndarray, target: np.ndarray, full_phase: np.ndarray, pcoef: np.ndarray, acoef: np.ndarray):
    phase = float(alpha) * full_phase
    positive = build_route(config, N, slm2_phase=None, pcoef=pcoef, acoef=acoef)
    corrected = build_route(config, N, slm2_phase=phase, pcoef=pcoef, acoef=acoef)
    pdet = detector_stack(positive, z_abs)
    cdet = detector_stack(corrected, z_abs)
    pm = concentric_metrics(pdet, target, ids)
    cm = concentric_metrics(cdet, target, ids)
    wd, top_ok = winding(corrected)
    return {
        "alpha": float(alpha),
        "grid_n": int(N),
        "topology_q20_all_contours": bool(top_ok),
        "winding": wd,
        "positive": pm,
        "corrected": cm,
        "corrected_objective": objective(cm),
        "principal_ring_cv_reduction_fraction": float(1.0 - cm["mean_principal_ring_azimuth_cv"] / max(pm["mean_principal_ring_azimuth_cv"], EPS)),
        "mirror_rmse_reduction_fraction": float(1.0 - cm["mirror_rmse"] / max(pm["mirror_rmse"], EPS)),
    }


def production(config, z_abs, z_rel, target, sym_target, miao_pred, full_phase, alpha, pcoef, acoef, out):
    positive = build_route(config, PROD_N, slm2_phase=None, pcoef=pcoef, acoef=acoef)
    corrected = build_route(config, PROD_N, slm2_phase=float(alpha) * full_phase, pcoef=pcoef, acoef=acoef)
    pdet = detector_stack(positive, z_abs)
    cdet = detector_stack(corrected, z_abs)
    popt = optical_stack(positive, z_abs)
    copt = optical_stack(corrected, z_abs)
    wp, _ = winding(positive)
    wc, top_ok = winding(corrected)
    groups = {
        "inner_train": INNER_TRAIN,
        "inner_validation": INNER_VALID,
        "legacy_heldout": LEGACY_HELD,
        "all_planes": np.arange(len(z_rel), dtype=int),
    }
    metrics = {}
    for name, ids in groups.items():
        metrics[name] = {
            "detector_positive": concentric_metrics(pdet, target, ids),
            "detector_corrected": concentric_metrics(cdet, target, ids),
            "optical_positive": concentric_metrics(popt, target, ids),
            "optical_corrected": concentric_metrics(copt, target, ids),
        }
    np.savez_compressed(
        out / "hybrid_concentric_4096_display_arrays.npz",
        axis_um=AXIS_UM, z_relative_mm=z_rel,
        measured_sym_target=sym_target.astype(np.float32),
        miao_target=normalise(miao_pred).astype(np.float32),
        hybrid_target=target.astype(np.float32),
        detector_positive=pdet.astype(np.float32), detector_corrected=cdet.astype(np.float32),
        optical_positive=popt.astype(np.float32), optical_corrected=copt.astype(np.float32),
    )
    np.save(out / "model_space_slm2_phase_hybrid_concentric_rad.npy", (float(alpha) * full_phase).astype(np.float32))

    # Poster-grade multi-plane visual: diagnosed model error -> Miao/data concentric target -> correction.
    ids = [1, 5, 9, 13, 17]
    fig, axs = plt.subplots(3, len(ids), figsize=(15.2, 8.9), constrained_layout=True)
    ext = [AXIS_UM[0], AXIS_UM[-1], AXIS_UM[0], AXIS_UM[-1]]
    rows = [(pdet, "diagnosed model error"), (target, "concentric target"), (cdet, "hybrid corrected prediction")]
    for col, iz in enumerate(ids):
        for row, (stack, label) in enumerate(rows):
            axs[row, col].imshow(stack[iz], origin="lower", extent=ext, cmap=THERMAL, vmin=0, vmax=1, interpolation="nearest")
            axs[row, col].set_aspect("equal")
            axs[row, col].set_xticks([]); axs[row, col].set_yticks([])
            if row == 0:
                axs[row, col].set_title(f"z = {z_rel[iz]:.0f} mm", fontsize=10)
            if col == 0:
                axs[row, col].set_ylabel(label, fontsize=10, fontweight="bold")
    fig.suptitle("Real-data-supported q=20 correction: Miao concentric prior + full optical route", fontsize=14, fontweight="bold")
    savefig(fig, out / "poster_hybrid_concentric_multiplane")

    # Optical-only before/after with common scale, deliberately showing ring morphology.
    ids2 = [5, 11, 17]
    fig, axs = plt.subplots(2, len(ids2), figsize=(11.2, 7.0), constrained_layout=True)
    for col, iz in enumerate(ids2):
        for row, stack in enumerate((popt, copt)):
            axs[row, col].imshow(stack[iz], origin="lower", extent=ext, cmap=THERMAL, vmin=0, vmax=1, interpolation="nearest")
            axs[row, col].set_aspect("equal"); axs[row, col].set_xticks([]); axs[row, col].set_yticks([])
            if row == 0: axs[row, col].set_title(f"z = {z_rel[iz]:.0f} mm")
            if col == 0: axs[row, col].set_ylabel("positive error" if row == 0 else "corrected", fontweight="bold")
    fig.suptitle("4096-grid optical field: annular concentricity before and after phase-only SLM2 correction", fontsize=13, fontweight="bold")
    savefig(fig, out / "poster_hybrid_optical_before_after")

    return {
        "production_grid_n": PROD_N,
        "selected_alpha": float(alpha),
        "topology_q20_all_contours": bool(top_ok),
        "winding_positive": wp,
        "winding_corrected": wc,
        "metrics": metrics,
    }


def run(source_dir: Path, miao_dir: Path, crosscheck_json: Path, candidate_json: Path, out: Path) -> dict:
    source_dir = Path(source_dir); out = Path(out); out.mkdir(parents=True, exist_ok=True)
    candidate = json.loads(Path(candidate_json).read_text(encoding="utf-8"))
    cross = json.loads(Path(crosscheck_json).read_text(encoding="utf-8"))
    pcoef, acoef, residual_source = candidate_coefficients(candidate, cross)

    context = v2.build_context(source_dir)
    data = normalise(context["data"])
    z_rel = np.asarray(context["z_rel"], float)
    z0 = float(candidate["physical_nuisance"]["selected_z0_mm"])
    z_abs = (z0 + z_rel) * 1e-3
    md = np.load(Path(miao_dir) / "miao_benchmark_arrays.npz")
    miao_pred = np.asarray(md["predicted"], float)
    sym_target, target = hybrid_target(data, miao_pred, AXIS_UM)

    config = v4.config_from_candidate(candidate, source_dir)
    full_phase, coeff, names, fit_pred, history = optimise_compact(
        config, z_abs, target, pcoef, acoef, out,
    )

    sweep = []
    for alpha in ALPHAS:
        s = evaluate_alpha(float(alpha), SWEEP_N, INNER_VALID, config, z_abs, target, full_phase, pcoef, acoef)
        sweep.append(s)
        print(json.dumps({"alpha": float(alpha), "validation": s}, indent=2))
    passing = [s for s in sweep if s["topology_q20_all_contours"]]
    if not passing:
        raise RuntimeError("No q=20 topology-preserving hybrid correction strength found")
    selected = min(passing, key=lambda s: s["corrected_objective"])
    alpha_star = float(selected["alpha"])
    (out / "hybrid_strength_sweep.json").write_text(json.dumps(sweep, indent=2) + "\n", encoding="utf-8")

    prod = production(config, z_abs, z_rel, target, sym_target, miao_pred, full_phase, alpha_star, pcoef, acoef, out)
    held = prod["metrics"]["legacy_heldout"]
    pos, cor = held["detector_positive"], held["detector_corrected"]
    cv_reduction = 1.0 - cor["mean_principal_ring_azimuth_cv"] / max(pos["mean_principal_ring_azimuth_cv"], EPS)
    mirror_reduction = 1.0 - cor["mirror_rmse"] / max(pos["mirror_rmse"], EPS)
    acceptance = {
        "q20_topology_preserved": bool(prod["topology_q20_all_contours"]),
        "legacy_heldout_principal_ring_cv_reduction_fraction": float(cv_reduction),
        "legacy_heldout_mirror_rmse_reduction_fraction": float(mirror_reduction),
        "legacy_heldout_radial_profile_corr": float(cor["mean_radial_profile_corr"]),
        "legacy_heldout_target_r": float(cor["mean_r"]),
        "legacy_heldout_target_nrmse": float(cor["mean_nrmse"]),
        "passes_concentricity_gate": bool(
            prod["topology_q20_all_contours"]
            and cv_reduction >= 0.30
            and mirror_reduction >= 0.20
            and cor["mean_radial_profile_corr"] >= 0.90
        ),
    }
    result = {
        "status": "hybrid_miao_full_route_concentric_q20_model_candidate_v5",
        "residual_source": residual_source,
        "miao_blend_fraction": MIAO_BLEND,
        "split": {
            "inner_train_even_indices": INNER_TRAIN.tolist(),
            "inner_validation_even_indices": INNER_VALID.tolist(),
            "legacy_heldout_odd_indices": LEGACY_HELD.tolist(),
            "note": "odd planes were not used by this correction solver, but were inspected in earlier project model audits and are therefore not claimed as pristine blind data",
        },
        "diagnostic_phase_coefficients_rad": pcoef.tolist(),
        "diagnostic_log_amplitude_coefficients": acoef.tolist(),
        "compact_basis_names": names,
        "compact_coefficients_rad": coeff.tolist(),
        "optimisation_history": history,
        "selected_validation_strength": selected,
        "production_validation": prod,
        "acceptance": acceptance,
        "beam_radius_boundary_policy": "v3 beam-radius nuisance is retained only as model registration; separate audit remained boundary-limited through 1.60x, so no fitted beam radius is promoted as a physical bench measurement",
        "hardware_ready": False,
        "hardware_blockers": candidate.get("hardware_blockers", [
            "SLM2-to-axicon coordinate transform/conjugacy not independently calibrated",
            "SLM2 parity/rotation/scale/centre not independently measured",
            "SLM2 1030 nm grey-to-phase LUT/stroke not calibrated",
            "no post-correction measured BeamGage z-stack",
        ]),
        "evidence_boundary": "corrected fields are numerical model-space predictions only; no corrected BeamGage frame is experimental evidence",
    }
    (out / "hybrid_concentric_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source-dir", type=Path, default=EXP / "outputs" / "digital_twin_correction")
    p.add_argument("--miao-dir", type=Path, default=ROOT / "outputs" / "validation" / "q20_miao_bessel_modal_benchmark")
    p.add_argument("--crosscheck-json", type=Path, default=ROOT / "outputs" / "validation" / "q20_miao_initializer_crosscheck" / "miao_initializer_crosscheck.json")
    p.add_argument("--candidate-json", type=Path, default=EXP / "candidates" / "q20_detector_aware_model_v3_candidate.json")
    p.add_argument("--out", type=Path, default=ROOT / "outputs" / "validation" / "q20_slm2_hybrid_miao_concentric_v5")
    a = p.parse_args()
    run(a.source_dir, a.miao_dir, a.crosscheck_json, a.candidate_json, a.out)


if __name__ == "__main__":
    main()
