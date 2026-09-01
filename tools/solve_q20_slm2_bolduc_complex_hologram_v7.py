"""q=20 correction v7: complex-amplitude precompensation with one phase-only SLM.

Miao et al. provide the intensity-only Bessel-wavefront prior.  Our independent
bench-matched inverse additionally supports a low-order phase + diagnostic
amplitude nuisance at the selected-order field immediately before the axicon.
The earlier SLM2 solvers only added a smooth phase map to a fixed-depth carrier.
That leaves an avoidable limitation: SLM2 is already followed by a Fourier-plane
order selector, so the carrier can also encode local first-order diffraction
efficiency and therefore complex amplitude.

This script tests the exact phase-only complex-field encoding of Bolduc et al.,
Opt. Lett. 38, 3546-3549 (2013), DOI 10.1364/OL.38.003546.  For a desired local
complex correction A exp(i Phi), their first-order hologram is

    Psi = M * mod(F + carrier, 2*pi)
    M   = 1 + sinc^{-1}(A)/pi
    F   = Phi - pi*M

with sinc(x)=sin(x)/x and the inverse branch x in [-pi,0].  The existing finite
4F iris explicitly selects the +1 order, making this architecture directly
relevant to the present bench model.

The desired correction is derived only from the Miao-initialised, detector-aware
phase+amplitude nuisance.  Correction strength is selected on the inner
validation split against the Miao/data concentric target.  The legacy odd planes
are reported after freezing.  All corrected frames are numerical predictions,
not post-correction BeamGage measurements.
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

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
EXP = ROOT / "notebooks" / "experimental" / "axicon_aberration_correction"
for p in (ROOT, TOOLS, EXP):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import solve_q20_slm2_hybrid_miao_concentric_v5 as v5  # noqa: E402
from real_bmg_digital_twin_correction import FIT_WINDOW_M, Q, RELAY_N  # noqa: E402
from vbb_study.digital_twin.phase2a_canonical import _panel_from_manifest  # noqa: E402
from vbb_study.digital_twin.phase2a_contracts import canonical_hardware_manifest, hardware_value  # noqa: E402
from vbb_study.digital_twin.vortex_beam_slm_errors import actual_slm_phase, transformed_pattern_coordinates  # noqa: E402
from vbb_study.digital_twin.vortex_system_route import (  # noqa: E402
    build_system_route,
    fourier_resample_fixed_window,
    physical_axicon_on_own_plane,
)
from vbb_study.digital_twin.vortex_explicit_4f import explicit_4f_relay  # noqa: E402
from vbb_study.equations.fields import make_xy_grid  # noqa: E402
from vbb_study.slm_model import apply_slm, pixelate  # noqa: E402

EPS = np.finfo(float).tiny
TWOPI = 2.0 * np.pi
THERMAL = "inferno"
SWEEP_N = 2048
PROD_N = 4096
ALPHAS = np.asarray([0.20, 0.35, 0.50, 0.65, 0.80, 0.95, 1.00], float)


def savefig(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=500, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def inverse_sinc_0_1(a: np.ndarray) -> np.ndarray:
    """Inverse unnormalised sinc on x in [-pi,0], where sinc rises 0 -> 1."""
    aa = np.clip(np.asarray(a, float), 0.0, 1.0)
    x = np.linspace(-np.pi, 0.0, 20001)
    s = np.ones_like(x)
    nz = np.abs(x) > 1e-12
    s[nz] = np.sin(x[nz]) / x[nz]
    return np.interp(aa, s, x)


def correction_transfer(grid: dict, pcoef: np.ndarray, acoef: np.ndarray, alpha: float) -> np.ndarray:
    """Desired pre-error complex multiplier on the selected-order axicon input."""
    phase_err, amp_err = v5.residual_maps(grid, pcoef, acoef)
    amp = np.maximum(np.asarray(amp_err, float), 1e-4)
    C = np.power(amp, -float(alpha)) * np.exp(-1j * float(alpha) * np.asarray(phase_err, float))
    X = np.asarray(grid["X"], float); Y = np.asarray(grid["Y"], float)
    support = np.hypot(X, Y) <= 2.25e-3
    scale = max(float(np.max(np.abs(C[support]))), EPS)
    C = C / scale
    return np.asarray(C, complex)


def bolduc_hologram(grid: dict, config, pcoef: np.ndarray, acoef: np.ndarray, alpha: float) -> tuple[np.ndarray, dict]:
    """Return the phase-only hologram command including the carrier.

    A parallel ideal 4F relay images the SLM plane with x,y inversion.  The core
    route explicitly records that convention, so the desired correction is
    inverted before encoding rather than fitted from camera data.
    """
    manifest = canonical_hardware_manifest()
    carrier = float(hardware_value(manifest, "carrier_frequency_cpm"))
    C_ax = correction_transfer(grid, pcoef, acoef, alpha)
    C_slm = C_ax[::-1, ::-1]
    A = np.clip(np.abs(C_slm), 0.0, 1.0)
    Phi = np.angle(C_slm)
    inv = inverse_sinc_0_1(A)
    M = 1.0 + inv / np.pi
    F = Phi - np.pi * M
    x2, _ = transformed_pattern_coordinates(grid, config.slm2)
    carrier_phase = TWOPI * carrier * x2
    psi = M * np.mod(F + carrier_phase, TWOPI)
    return np.asarray(psi, float), {
        "alpha": float(alpha),
        "amplitude_min": float(np.min(A)),
        "amplitude_mean": float(np.mean(A)),
        "modulation_depth_M_min": float(np.min(M)),
        "modulation_depth_M_mean": float(np.mean(M)),
        "coordinate_parity": "xy inversion from explicit 4F image convention",
        "encoding_reference": "Bolduc et al., Opt. Lett. 38, 3546-3549 (2013), DOI 10.1364/OL.38.003546",
    }


def custom_route(config, propagation_n: int, pcoef: np.ndarray, acoef: np.ndarray, alpha: float) -> tuple[dict, dict]:
    manifest = canonical_hardware_manifest()
    wavelength = float(hardware_value(manifest, "wavelength_m"))
    pixel_pitch = float(hardware_value(manifest, "slm_pixel_pitch_m"))
    carrier = float(hardware_value(manifest, "carrier_frequency_cpm"))
    f4f = float(hardware_value(manifest, "fourf_focal_length_m"))
    iris_radius = float(hardware_value(manifest, "fourier_iris_radius_m"))
    n_ax = float(hardware_value(manifest, "axicon_refractive_index"))
    n_ext = float(hardware_value(manifest, "axicon_external_medium_index"))
    gamma0 = math.radians(float(hardware_value(manifest, "axicon_base_angle_deg")))

    # Reuse the exact Gaussian + SLM1 route.  Its ordinary SLM2/4F result is ignored.
    base = build_system_route(f"V{Q}", grid_n=RELAY_N, config=config, window_m=FIT_WINDOW_M)
    grid = base["grid"]
    panel = _panel_from_manifest(manifest)
    psi, hmeta = bolduc_hologram(grid, config, pcoef, acoef, alpha)
    command = pixelate(psi, grid, panel)
    actual2, slm2_meta = actual_slm_phase(
        command, grid, error=config.slm2, pixel_pitch_m=pixel_pitch,
        lut_phase_rad=None, static_phase_map_rad=None,
    )
    slm2 = apply_slm(
        np.asarray(base["post_slm1"], complex), actual2, grid, panel,
        phase_is_prepared=True, quantise_phase=False, apply_fill_factor=True,
        apply_carrier=False,
    )
    relay = explicit_4f_relay(
        slm2.total, grid,
        wavelength_m=wavelength, nominal_focal_length_m=f4f,
        nominal_iris_radius_m=iris_radius, nominal_carrier_cpm=carrier,
        error=config.fourf,
    )
    X = np.asarray(grid["X"], float)
    selected = np.asarray(relay["output"], complex) * np.exp(1j * TWOPI * carrier * X)

    if int(propagation_n) == RELAY_N:
        fine_grid = grid
    else:
        fine_grid = make_xy_grid(int(propagation_n), FIT_WINDOW_M / int(propagation_n))
    field = fourier_resample_fixed_window(selected, int(propagation_n))
    phase_err, amp_err = v5.residual_maps(fine_grid, pcoef, acoef)
    field = field * np.asarray(amp_err, float) * np.exp(1j * np.asarray(phase_err, float))
    ax_t, ax_meta = physical_axicon_on_own_plane(
        fine_grid, wavelength_m=wavelength, base_angle_rad=gamma0,
        refractive_index=n_ax, external_index=n_ext, error=config.axicon,
        surface_height_error_m=None,
    )
    post = np.asarray(field * ax_t, complex)
    return {
        "grid": fine_grid,
        "field_on_axicon_plane": np.asarray(field, complex),
        "post_axicon": post,
        "relay_route": {"grid": grid, "post_4f_selected_order": selected},
        "metadata": {
            "route_id": "q20_bolduc_complex_hologram_v7",
            "hologram": hmeta,
            "slm2": slm2_meta,
            "axicon": ax_meta,
        },
    }, hmeta


def evaluate(alpha: float, N: int, ids: np.ndarray, config, z_abs, target, pcoef, acoef) -> dict:
    # Positive error reference stays on the canonical carrier route.
    positive = v5.build_route(config, N, slm2_phase=None, pcoef=pcoef, acoef=acoef)
    corrected, hmeta = custom_route(config, N, pcoef, acoef, alpha)
    pdet = v5.detector_stack(positive, z_abs)
    cdet = v5.detector_stack(corrected, z_abs)
    pm = v5.concentric_metrics(pdet, target, ids)
    cm = v5.concentric_metrics(cdet, target, ids)
    wd, top_ok = v5.winding(corrected)
    return {
        "alpha": float(alpha), "grid_n": int(N), "hologram": hmeta,
        "topology_q20_all_contours": bool(top_ok), "winding": wd,
        "positive": pm, "corrected": cm,
        "corrected_objective": v5.objective(cm),
        "principal_ring_cv_reduction_fraction": float(1.0 - cm["mean_principal_ring_azimuth_cv"] / max(pm["mean_principal_ring_azimuth_cv"], EPS)),
        "mirror_rmse_reduction_fraction": float(1.0 - cm["mirror_rmse"] / max(pm["mirror_rmse"], EPS)),
    }


def production(alpha, config, z_abs, z_rel, target, sym_target, miao_pred, pcoef, acoef, out):
    positive = v5.build_route(config, PROD_N, slm2_phase=None, pcoef=pcoef, acoef=acoef)
    corrected, hmeta = custom_route(config, PROD_N, pcoef, acoef, alpha)
    pdet = v5.detector_stack(positive, z_abs); cdet = v5.detector_stack(corrected, z_abs)
    popt = v5.optical_stack(positive, z_abs); copt = v5.optical_stack(corrected, z_abs)
    wp, _ = v5.winding(positive); wc, top_ok = v5.winding(corrected)
    groups = {
        "inner_train": v5.INNER_TRAIN,
        "inner_validation": v5.INNER_VALID,
        "legacy_heldout": v5.LEGACY_HELD,
        "all_planes": np.arange(len(z_rel), dtype=int),
    }
    metrics = {}
    for name, ids in groups.items():
        metrics[name] = {
            "detector_positive": v5.concentric_metrics(pdet, target, ids),
            "detector_corrected": v5.concentric_metrics(cdet, target, ids),
            "optical_positive": v5.concentric_metrics(popt, target, ids),
            "optical_corrected": v5.concentric_metrics(copt, target, ids),
        }
    np.savez_compressed(
        out / "bolduc_v7_4096_display_arrays.npz",
        axis_um=v5.AXIS_UM, z_relative_mm=z_rel,
        measured_sym_target=sym_target.astype(np.float32), miao_target=v5.normalise(miao_pred).astype(np.float32),
        hybrid_target=target.astype(np.float32), detector_positive=pdet.astype(np.float32), detector_corrected=cdet.astype(np.float32),
        optical_positive=popt.astype(np.float32), optical_corrected=copt.astype(np.float32),
    )

    ext = [v5.AXIS_UM[0], v5.AXIS_UM[-1], v5.AXIS_UM[0], v5.AXIS_UM[-1]]
    ids = [1, 5, 9, 13, 17]
    fig, axs = plt.subplots(4, len(ids), figsize=(15.5, 11.2), constrained_layout=True)
    rows = [(pdet, "diagnosed model"), (v5.normalise(miao_pred), "Miao-style"), (target, "hybrid target"), (cdet, "Bolduc corrected")]
    for col, iz in enumerate(ids):
        for row, (stack, label) in enumerate(rows):
            axs[row, col].imshow(stack[iz], origin="lower", extent=ext, cmap=THERMAL, vmin=0, vmax=1, interpolation="nearest")
            axs[row, col].set_aspect("equal"); axs[row, col].set_xticks([]); axs[row, col].set_yticks([])
            if row == 0: axs[row, col].set_title(f"z = {z_rel[iz]:.0f} mm")
            if col == 0: axs[row, col].set_ylabel(label, fontweight="bold")
    fig.suptitle("q=20: Miao/data diagnosis -> complex-amplitude phase-only hologram -> predicted correction", fontsize=14, fontweight="bold")
    savefig(fig, out / "poster_bolduc_v7_multiplane")

    ids2 = [5, 11, 17]
    fig, axs = plt.subplots(2, 3, figsize=(11.2, 7.0), constrained_layout=True)
    for col, iz in enumerate(ids2):
        for row, stack in enumerate((popt, copt)):
            axs[row, col].imshow(stack[iz], origin="lower", extent=ext, cmap=THERMAL, vmin=0, vmax=1, interpolation="nearest")
            axs[row, col].set_aspect("equal"); axs[row, col].set_xticks([]); axs[row, col].set_yticks([])
            if row == 0: axs[row, col].set_title(f"z = {z_rel[iz]:.0f} mm")
            if col == 0: axs[row, col].set_ylabel("diagnosed" if row == 0 else "Bolduc corrected", fontweight="bold")
    fig.suptitle("4096-grid optical field: before/after complex-amplitude SLM2 encoding", fontsize=13, fontweight="bold")
    savefig(fig, out / "poster_bolduc_v7_optical_before_after")

    return {
        "production_grid_n": PROD_N, "selected_alpha": float(alpha), "hologram": hmeta,
        "topology_q20_all_contours": bool(top_ok), "winding_positive": wp, "winding_corrected": wc,
        "metrics": metrics,
    }


def run(source_dir: Path, miao_dir: Path, crosscheck_json: Path, candidate_json: Path, out: Path) -> dict:
    source_dir = Path(source_dir); miao_dir = Path(miao_dir); out = Path(out); out.mkdir(parents=True, exist_ok=True)
    candidate = json.loads(Path(candidate_json).read_text(encoding="utf-8"))
    cross = json.loads(Path(crosscheck_json).read_text(encoding="utf-8"))
    pcoef, acoef, residual_source = v5.candidate_coefficients(candidate, cross)
    context = v5.v2.build_context(source_dir)
    data = v5.normalise(context["data"]); z_rel = np.asarray(context["z_rel"], float)
    z0 = float(candidate["physical_nuisance"]["selected_z0_mm"]); z_abs = (z0 + z_rel) * 1e-3
    md = np.load(miao_dir / "miao_benchmark_arrays.npz"); miao_pred = np.asarray(md["predicted"], float)
    sym_target, target = v5.hybrid_target(data, miao_pred, v5.AXIS_UM)
    config = v5.v4.config_from_candidate(candidate, source_dir)

    sweep = []
    for alpha in ALPHAS:
        rec = evaluate(float(alpha), SWEEP_N, v5.INNER_VALID, config, z_abs, target, pcoef, acoef)
        sweep.append(rec); print(json.dumps(rec, indent=2))
    passing = [s for s in sweep if s["topology_q20_all_contours"]]
    if not passing: raise RuntimeError("Bolduc v7 found no q=20 topology-preserving correction strength")
    selected = min(passing, key=lambda s: s["corrected_objective"]); alpha_star = float(selected["alpha"])
    (out / "bolduc_v7_strength_sweep.json").write_text(json.dumps(sweep, indent=2) + "\n", encoding="utf-8")
    prod = production(alpha_star, config, z_abs, z_rel, target, sym_target, miao_pred, pcoef, acoef, out)
    held = prod["metrics"]["legacy_heldout"]; pos = held["detector_positive"]; cor = held["detector_corrected"]
    cv_red = 1.0 - cor["mean_principal_ring_azimuth_cv"] / max(pos["mean_principal_ring_azimuth_cv"], EPS)
    mirror_red = 1.0 - cor["mirror_rmse"] / max(pos["mirror_rmse"], EPS)
    gate = bool(prod["topology_q20_all_contours"] and cv_red >= 0.35 and mirror_red >= 0.20 and cor["mean_radial_profile_corr"] >= 0.90)
    result = {
        "study": "q20 Miao/data inverse plus Bolduc exact complex-amplitude phase-only hologram",
        "reference": "Bolduc et al., Opt. Lett. 38, 3546-3549 (2013), DOI 10.1364/OL.38.003546",
        "residual_source": residual_source, "selected_alpha": alpha_star, "selected_validation": selected,
        "production": prod, "legacy_heldout_cv_reduction_fraction": float(cv_red),
        "legacy_heldout_mirror_reduction_fraction": float(mirror_red),
        "legacy_heldout_radial_profile_corr": float(cor["mean_radial_profile_corr"]),
        "passes_concentricity_gate": gate, "hardware_ready": False,
        "hardware_blockers": ["SLM2-to-axicon coordinate calibration", "1030-nm grey-to-phase LUT/stroke", "measured post-correction z-stack"],
        "corrected_camera_evidence": "none; corrected stacks are numerical predictions only",
    }
    (out / "bolduc_v7_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2)); return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source-dir", type=Path, default=EXP / "outputs" / "digital_twin_correction")
    p.add_argument("--miao-dir", type=Path, default=ROOT / "outputs" / "validation" / "q20_miao_bessel_modal")
    p.add_argument("--crosscheck-json", type=Path, default=ROOT / "outputs" / "validation" / "q20_miao_initializer_crosscheck" / "miao_initializer_crosscheck.json")
    p.add_argument("--candidate-json", type=Path, default=ROOT / "outputs" / "validation" / "q20_detector_aware_model_v3" / "q20_detector_aware_model_v3_summary.json")
    p.add_argument("--out", type=Path, default=ROOT / "outputs" / "validation" / "q20_slm2_bolduc_complex_hologram_v7")
    a = p.parse_args(); run(a.source_dir, a.miao_dir, a.crosscheck_json, a.candidate_json, a.out)


if __name__ == "__main__":
    main()
