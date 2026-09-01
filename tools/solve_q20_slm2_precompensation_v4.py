"""Topology-aware phase-only SLM2 precompensation for the corrected q=20 model.

This solve supersedes the poster-era alpha=0.40 candidate, which was based on
the earlier 48 mm axial registration and angular-only residual.  The positive
error used here is the PHASE-CORRECTABLE part of the train/heldout validated v3
model at the selected-order field immediately before the axicon.  Diagnostic
amplitude nuisance terms are intentionally excluded: a phase-only SLM cannot
directly cancel them.

A compact zero-winding SLM2 basis is optimized through the explicit carrier +
4F + fixed +1 iris + axicon route against the nominal q=20 detector stack.  The
fit uses even z planes and is scored on odd z planes.  The resulting phase map
is then strength-regularised on a higher-resolution optical model, and only a
candidate that preserves q=20 on every 1.0--1.5 mm winding contour can be
selected.  The final selected strength is validated at N=4096.

This remains a numerical model-space precompensation candidate until SLM2-to-
axicon coordinates/conjugacy and the 1030 nm phase LUT are bench-calibrated and
a post-correction BeamGage z-stack is acquired.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
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
EXP = ROOT / "notebooks" / "experimental" / "axicon_aberration_correction"
TOOLS = ROOT / "tools"
for p in (ROOT, EXP, TOOLS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import fit_q20_detector_aware_model_v2 as model_v2  # noqa: E402
from optimize_q20_slm2_detector_closure_v2 import (  # noqa: E402
    phase_basis,
    phase_from_coefficients,
    weighted_system,
    objective_tuple,
    intensity_metrics,
    structure_metrics,
)
from real_bmg_digital_twin_correction import (  # noqa: E402
    AxiconError,
    FIT_N,
    FIT_WINDOW_M,
    PIXEL_M,
    Q,
    RELAY_N,
    SystemErrorConfig,
)
from vbb_study.digital_twin.detector_response import plane_normalise, sample_camera_response  # noqa: E402
from vbb_study.digital_twin.vortex_beam_slm_errors import GaussianBeamError  # noqa: E402
from vbb_study.digital_twin.vortex_continuous_propagation import build_fixed_support_spectrum, native_field_at_z  # noqa: E402
from vbb_study.digital_twin.vortex_system_route import FourFError, build_multirate_system_route  # noqa: E402
from vbb_study.digital_twin.phase2a_contracts import canonical_hardware_manifest, hardware_value  # noqa: E402
from vbb_study.viz_fields import phase_winding  # noqa: E402

EPS = np.finfo(float).tiny
THERMAL = "inferno"
AXIS_UM = np.linspace(-180.0, 180.0, 241)
WINDING_RADII_MM = (1.0, 1.1, 1.2, 1.3, 1.4, 1.5)
SWEEP_N = 3072
PROD_N = 4096
ALPHAS = np.asarray([0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00], float)


def savefig(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=500, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def config_from_candidate(candidate: dict, source_dir: Path) -> SystemErrorConfig:
    source_summary = json.loads((Path(source_dir) / "run_summary.json").read_text(encoding="utf-8"))
    scale = float(source_summary["effective_axicon"]["effective_scale_relative_to_repository_2deg_assumption"])
    physical = candidate["physical_nuisance"]
    return SystemErrorConfig(
        beam=GaussianBeamError(
            radius_x_scale=float(physical["selected_beam_radius_scale"]),
            radius_y_scale=float(physical["selected_beam_radius_scale"]),
        ),
        fourf=FourFError(iris_radius_scale=float(physical["selected_iris_radius_scale"])),
        axicon=AxiconError(base_angle_scale=scale),
    )


def route(config: SystemErrorConfig, N: int, *, slm2_phase=None, axicon_phase=None):
    return build_multirate_system_route(
        f"V{Q}", relay_grid_n=RELAY_N, propagation_grid_n=int(N),
        window_m=FIT_WINDOW_M, config=config,
        slm2_static_phase_map_rad=slm2_phase,
        axicon_input_phase_map_rad=axicon_phase,
    )


def detector_render_native(native: np.ndarray, x_m: np.ndarray) -> np.ndarray:
    shown, _ = sample_camera_response(
        np.asarray(native, float), np.asarray(x_m, float), AXIS_UM * 1e-6,
        pixel_pitch_m=PIXEL_M, quadrature_n=3,
    )
    return plane_normalise(shown)


def field_stack(route_result: dict, zabs: np.ndarray) -> list[np.ndarray]:
    wl = float(hardware_value(canonical_hardware_manifest(), "wavelength_m"))
    prop = build_fixed_support_spectrum(
        route_result["post_axicon"], route_result["grid"],
        wavelength_m=wl, z_max_m=max(abs(np.asarray(zabs, float))),
        minimum_retained_spectral_power=0.99,
    )
    return [np.asarray(native_field_at_z(prop, float(z)), complex) for z in np.asarray(zabs, float)]


def detector_stack(route_result: dict, zabs: np.ndarray) -> np.ndarray:
    fields = field_stack(route_result, zabs)
    native = np.stack([np.abs(f) ** 2 for f in fields])
    out = detector_render_native(native, np.asarray(route_result["grid"]["x"], float))
    del fields, native
    return out


def phase_residual_on_grid(grid: dict, candidate: dict) -> np.ndarray:
    pb, _ = model_v2.residual_bases(grid)
    pcoef = np.asarray(candidate["phase_coefficients_rad"], float)
    phase = np.zeros_like(pb[0], dtype=float)
    for c, basis in zip(pcoef, pb):
        phase += float(c) * basis
    return phase


def optimise_compact(config: SystemErrorConfig, candidate: dict, zabs: np.ndarray, out: Path):
    base = route(config, FIT_N)
    error_phase = phase_residual_on_grid(base["grid"], candidate)
    target = detector_stack(base, zabs)
    positive_route = route(config, FIT_N, axicon_phase=error_phase)
    positive = detector_stack(positive_route, zabs)

    basis, names = phase_basis(base["relay_route"]["grid"])
    train = np.arange(0, len(zabs), 2, dtype=int)
    held = np.arange(1, len(zabs), 2, dtype=int)
    coeff = np.zeros(len(names), float)
    delta = 0.12
    current = positive
    history = []

    def simulate(c: np.ndarray) -> np.ndarray:
        phase = phase_from_coefficients(basis, c)
        r = route(config, FIT_N, slm2_phase=phase, axicon_phase=error_phase)
        return detector_stack(r, zabs)

    for iteration in range(3):
        derivatives = []
        for j in range(len(coeff)):
            cp, cm = coeff.copy(), coeff.copy()
            cp[j] += delta; cm[j] -= delta
            plus, minus = simulate(cp), simulate(cm)
            derivatives.append((plus - minus) / (2.0 * delta))
        J, y = weighted_system(current, derivatives, target, AXIS_UM, train)
        # Radial/high-order basis terms are useful but should not dominate the
        # inferred low-order correction.  A moderate ridge keeps the map smooth.
        ridge = 1.2e-2
        step = np.linalg.solve(J.T @ J + ridge * np.eye(J.shape[1]), J.T @ y)
        step = np.clip(step, -0.28, 0.28)
        base_obj = objective_tuple(current, target, AXIS_UM, train)[0]
        chosen = None
        for strength in (1.0, 0.65, 0.40, 0.22):
            trial_c = np.clip(coeff + strength * step, -1.0, 1.0)
            trial = simulate(trial_c)
            obj = objective_tuple(trial, target, AXIS_UM, train)[0]
            if chosen is None or obj < chosen[0]:
                chosen = (obj, trial_c, trial, strength)
        if chosen is None or chosen[0] >= base_obj:
            history.append({"iteration": iteration + 1, "accepted": False, "train_objective": base_obj})
            break
        coeff, current = chosen[1], chosen[2]
        tr = objective_tuple(current, target, AXIS_UM, train)
        he = objective_tuple(current, target, AXIS_UM, held)
        history.append({
            "iteration": iteration + 1,
            "accepted": True,
            "step_strength": float(chosen[3]),
            "coefficients_rad": coeff.tolist(),
            "train_objective": tr[0], "train_r": tr[1], "train_nrmse": tr[2],
            "heldout_objective": he[0], "heldout_r": he[1], "heldout_nrmse": he[2],
            "heldout_structure": he[3],
        })
        delta = 0.08

    phase = phase_from_coefficients(basis, coeff)
    baseline_he = intensity_metrics(positive, target, AXIS_UM, held)
    full_he = intensity_metrics(current, target, AXIS_UM, held)
    result = {
        "basis_names": names,
        "coefficients_rad": coeff.tolist(),
        "iterations": history,
        "positive_error_heldout": {"r": baseline_he[0], "nrmse": baseline_he[1], **structure_metrics(positive, target, AXIS_UM, held)},
        "full_strength_heldout": {"r": full_he[0], "nrmse": full_he[1], **structure_metrics(current, target, AXIS_UM, held)},
    }
    np.save(out / "slm2_compact_full_strength_phase_rad.npy", phase.astype(np.float32))
    (out / "compact_solve_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return phase, error_phase, result


def norm(a):
    a = np.maximum(np.asarray(a, float), 0.0)
    return a / max(float(a.max()), EPS)


def native_crop(field: np.ndarray, x_m: np.ndarray) -> np.ndarray:
    I = np.abs(np.asarray(field)) ** 2
    pix = np.interp(AXIS_UM * 1e-6, np.asarray(x_m, float), np.arange(len(x_m)))
    yy, xx = np.meshgrid(pix, pix, indexing="ij")
    return norm(ndimage.map_coordinates(I, [yy, xx], order=1, mode="constant", cval=0.0))


def detector_crop(field: np.ndarray, x_m: np.ndarray) -> np.ndarray:
    shown, _ = sample_camera_response(
        (np.abs(np.asarray(field)) ** 2)[None], np.asarray(x_m, float), AXIS_UM * 1e-6,
        pixel_pitch_m=PIXEL_M, quadrature_n=3,
    )
    return norm(shown[0])


def image_metrics(a, b):
    X, Y = np.meshgrid(AXIS_UM, AXIS_UM, indexing="xy")
    roi = np.hypot(X, Y) <= 145.0
    av, bv = np.asarray(a)[roi], np.asarray(b)[roi]
    return float(np.corrcoef(av, bv)[0, 1]), float(np.sqrt(np.mean((av - bv) ** 2)))


def mirror_metrics(im):
    X, Y = np.meshgrid(AXIS_UM, AXIS_UM, indexing="xy")
    R = np.hypot(X, Y); mask = (R >= 20.0) & (R <= 140.0)
    return (
        float(np.sqrt(np.mean((im[mask] - im[:, ::-1][mask]) ** 2))),
        float(np.sqrt(np.mean((im[mask] - im[::-1, :][mask]) ** 2))),
    )


def winding(route_result: dict) -> tuple[dict, bool]:
    values, ok = {}, True
    for rmm in WINDING_RADII_MM:
        w = float(phase_winding(route_result["post_axicon"], route_result["grid"], rmm * 1e-3, n_phi=720))
        values[f"radius_{rmm:.1f}_mm"] = w
        ok &= abs(w - float(Q)) <= 0.25
    return values, bool(ok)


def evaluate_alpha(alpha: float, N: int, config: SystemErrorConfig, zabs: np.ndarray, full_phase: np.ndarray, candidate: dict):
    nominal = route(config, N)
    err = phase_residual_on_grid(nominal["grid"], candidate)
    positive = route(config, N, axicon_phase=err)
    corrected = route(config, N, slm2_phase=float(alpha) * full_phase, axicon_phase=err)
    wl = float(hardware_value(canonical_hardware_manifest(), "wavelength_m"))
    props = []
    for rr in (nominal, positive, corrected):
        props.append(build_fixed_support_spectrum(
            rr["post_axicon"], rr["grid"], wavelength_m=wl,
            z_max_m=max(abs(zabs)), minimum_retained_spectral_power=0.99,
        ))
    x = np.asarray(nominal["grid"]["x"], float)
    rows = []; stacks = [[], [], []]; dstacks = [[], [], []]
    for iz, zz in enumerate(zabs):
        fields = [native_field_at_z(p, float(zz)) for p in props]
        optical = [native_crop(f, x) for f in fields]
        detector = [detector_crop(f, x) for f in fields]
        for j in range(3):
            stacks[j].append(optical[j]); dstacks[j].append(detector[j])
        rp, ep = image_metrics(optical[1], optical[0]); rc, ec = image_metrics(optical[2], optical[0])
        rpd, epd = image_metrics(detector[1], detector[0]); rcd, ecd = image_metrics(detector[2], detector[0])
        mx, my = mirror_metrics(optical[2])
        rows.append({
            "z_relative_mm": float((zabs[iz] - zabs[-1]) * 1e3),
            "positive_optical_r": rp, "positive_optical_nrmse": ep,
            "corrected_optical_r": rc, "corrected_optical_nrmse": ec,
            "positive_detector_r": rpd, "positive_detector_nrmse": epd,
            "corrected_detector_r": rcd, "corrected_detector_nrmse": ecd,
            "corrected_xmirror_rmse": mx, "corrected_ymirror_rmse": my,
        })
        del fields
    df = pd.DataFrame(rows)
    wd, top_ok = winding(corrected)
    summary = {
        "alpha": float(alpha), "grid_n": int(N), "topology_q20_all_contours": top_ok,
        "winding": wd,
        "mean_positive_optical_r": float(df.positive_optical_r.mean()),
        "mean_positive_optical_nrmse": float(df.positive_optical_nrmse.mean()),
        "mean_corrected_optical_r": float(df.corrected_optical_r.mean()),
        "mean_corrected_optical_nrmse": float(df.corrected_optical_nrmse.mean()),
        "mean_positive_detector_r": float(df.positive_detector_r.mean()),
        "mean_positive_detector_nrmse": float(df.positive_detector_nrmse.mean()),
        "mean_corrected_detector_r": float(df.corrected_detector_r.mean()),
        "mean_corrected_detector_nrmse": float(df.corrected_detector_nrmse.mean()),
        "mean_corrected_xmirror_rmse": float(df.corrected_xmirror_rmse.mean()),
        "mean_corrected_ymirror_rmse": float(df.corrected_ymirror_rmse.mean()),
    }
    return summary, df, [np.stack(s) for s in stacks], [np.stack(s) for s in dstacks]


def run(source_dir: Path, candidate_json: Path, out: Path) -> dict:
    source_dir = Path(source_dir); out = Path(out); out.mkdir(parents=True, exist_ok=True)
    candidate = json.loads(Path(candidate_json).read_text(encoding="utf-8"))
    config = config_from_candidate(candidate, source_dir)
    zrel = np.arange(-17.0, 1.0)
    z0 = float(candidate["physical_nuisance"]["selected_z0_mm"])
    zabs = (z0 + zrel) * 1e-3

    full_phase, _, compact = optimise_compact(config, candidate, zabs, out)

    sweep = []
    for alpha in ALPHAS:
        s, _, _, _ = evaluate_alpha(float(alpha), SWEEP_N, config, zabs, full_phase, candidate)
        sweep.append(s); print(json.dumps(s))
    sweep_df = pd.DataFrame([{k:v for k,v in s.items() if k != "winding"} for s in sweep])
    sweep_df.to_csv(out / "strength_sweep_3072.csv", index=False)
    (out / "strength_sweep_3072.json").write_text(json.dumps(sweep, indent=2) + "\n", encoding="utf-8")

    passing = [s for s in sweep if s["topology_q20_all_contours"]]
    if not passing:
        raise RuntimeError("No q=20 topology-preserving correction strength was found")
    # Prefer optical closure; detector closure is a secondary tie-break.  This
    # avoids a camera-sampled optimum that hides optical ring degradation.
    selected = max(passing, key=lambda s: (s["mean_corrected_optical_r"], -s["mean_corrected_optical_nrmse"], s["mean_corrected_detector_r"]))
    alpha_star = float(selected["alpha"])

    prod, prod_df, optical, detector = evaluate_alpha(alpha_star, PROD_N, config, zabs, full_phase, candidate)
    prod_df.to_csv(out / "selected_4096_metrics_vs_z.csv", index=False)
    np.save(out / "model_space_slm2_phase_selected_rad.npy", (alpha_star * full_phase).astype(np.float32))
    np.savez_compressed(
        out / "selected_4096_display_arrays.npz",
        axis_um=AXIS_UM, z_relative_mm=zrel,
        optical_nominal=optical[0].astype(np.float32),
        optical_positive_error=optical[1].astype(np.float32),
        optical_corrected=optical[2].astype(np.float32),
        detector_nominal=detector[0].astype(np.float32),
        detector_positive_error=detector[1].astype(np.float32),
        detector_corrected=detector[2].astype(np.float32),
    )

    result = {
        "status": "q20_topology_aware_slm2_precompensation_v4_model_candidate",
        "diagnostic_source": str(Path(candidate_json)),
        "selected_alpha": alpha_star,
        "compact_solve": compact,
        "sweep_grid_n": SWEEP_N,
        "production_grid_n": PROD_N,
        "production_validation": prod,
        "hardware_ready": False,
        "hardware_blockers": [
            "SLM2 to axicon/input-plane conjugacy and coordinate transform not independently calibrated",
            "SLM2 parity/rotation/scale/centre not independently measured",
            "SLM2 1030 nm grey-to-phase LUT/stroke not calibrated",
            "no post-correction measured BeamGage z-stack",
        ],
        "evidence_boundary": "correction is numerical model-space prediction only; no corrected BeamGage image is experimental evidence",
    }
    (out / "selected_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    # Poster-grade evidence: positive model error -> phase-only SLM2 correction -> nominal target.
    rep = int(np.argmin(abs(zrel + 10.0)))
    fig, axs = plt.subplots(2, 3, figsize=(12.8, 8.2), constrained_layout=True)
    titles = ["Nominal target", "Diagnosed phase error", "SLM2-corrected prediction"]
    for j, title in enumerate(titles):
        axs[0, j].imshow(optical[j][rep], origin="lower", extent=[-180,180,-180,180], cmap=THERMAL, vmin=0, vmax=1)
        axs[0, j].set(title=title, xlabel="x (um)", ylabel="y (um)", aspect="equal")
    mid = len(AXIS_UM)//2
    axs[1, 0].plot(AXIS_UM, optical[0][rep, mid], lw=2.0, label="nominal")
    axs[1, 0].plot(AXIS_UM, optical[1][rep, mid], lw=1.5, label="positive error")
    axs[1, 0].plot(AXIS_UM, optical[2][rep, mid], lw=1.5, label="corrected")
    axs[1, 0].set(xlim=(-130,130), ylim=(0,1.05), xlabel="x (um)", ylabel="normalised intensity", title="Central cut")
    axs[1, 0].legend(frameon=False, fontsize=8); axs[1,0].grid(alpha=.2)
    axs[1, 1].plot(sweep_df.alpha, sweep_df.mean_corrected_optical_r, "o-", label="optical r")
    axs[1, 1].plot(sweep_df.alpha, sweep_df.mean_corrected_detector_r, "s--", label="detector r")
    good = sweep_df.topology_q20_all_contours.astype(bool).to_numpy()
    axs[1, 1].scatter(sweep_df.alpha[good], sweep_df.mean_corrected_optical_r[good], s=70, facecolors="none", edgecolors="k", label="q=20 preserved")
    axs[1, 1].axvline(alpha_star, ls="--", lw=1.4)
    axs[1, 1].set(xlabel="correction strength alpha", ylabel="mean correlation", title="Topology-aware strength selection")
    axs[1, 1].grid(alpha=.2); axs[1,1].legend(frameon=False, fontsize=8)
    radii = np.asarray(WINDING_RADII_MM)
    vals = np.asarray([prod["winding"][f"radius_{r:.1f}_mm"] for r in radii])
    axs[1, 2].plot(radii, vals, "o-", lw=2.0)
    axs[1, 2].axhline(Q, ls="--", lw=1.5)
    axs[1, 2].set(xlabel="contour radius (mm)", ylabel="phase winding", title="4096-grid topology validation", ylim=(Q-1,Q+1))
    axs[1, 2].grid(alpha=.2)
    fig.suptitle("q=20 phase-only SLM2 precompensation: full optical route, topology constrained", fontsize=14, fontweight="bold")
    savefig(fig, out / "poster_correction_evidence")

    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source-dir", type=Path, default=EXP / "outputs" / "digital_twin_correction")
    p.add_argument("--candidate-json", type=Path, default=EXP / "candidates" / "q20_detector_aware_model_v3_candidate.json")
    p.add_argument("--out", type=Path, default=ROOT / "outputs" / "validation" / "q20_slm2_precompensation_v4")
    a = p.parse_args(); run(a.source_dir, a.candidate_json, a.out)


if __name__ == "__main__":
    main()
