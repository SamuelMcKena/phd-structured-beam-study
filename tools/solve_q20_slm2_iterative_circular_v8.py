"""q=20 correction v8: iterative circular-field closure through the finite 4F relay.

Why v8 exists
-------------
v5 made annular symmetry part of the score, but its Gauss-Newton update still
solved a pixelwise intensity residual and barely changed the principal-ring
azimuthal CV. v7 used Bolduc complex-amplitude encoding directly from the
retrieved nuisance; it reduced mirror/CV error but split the high-order vortex
inside the existing topology contours. v8 changes both the target and the
solver:

1. The actuator target is no longer a measured/Miao intensity blend. The
   selected-order field is projected onto an explicitly circular q=20 complex
   field: radial amplitude + radial residual phase + q*theta. Measured angular
   structure is therefore never rewarded by the correction objective.
2. The one-SLM complex-amplitude encoding is calibrated against the numerical
   4F convention (parity and phase sign) before use.
3. The desired complex transfer is iteratively updated after propagating through
   the finite +1-order iris. This is a model-space closed loop rather than a
   one-shot conjugate map.
4. A smooth inner topology guard keeps the correction unity across the previous
   q=20 winding audit region; correction acts mainly on the Miao-relevant outer
   annulus.
5. Strength is frozen on the even-plane inner-validation split and the legacy
   odd planes are reported afterwards. Corrected images remain numerical model
   predictions only, not post-correction BeamGage measurements.
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
import solve_q20_slm2_bolduc_complex_hologram_v7 as v7  # noqa: E402
from real_bmg_digital_twin_correction import FIT_WINDOW_M, Q, RELAY_N  # noqa: E402
from vbb_study.digital_twin.phase2a_canonical import _panel_from_manifest  # noqa: E402
from vbb_study.digital_twin.phase2a_contracts import canonical_hardware_manifest, hardware_value  # noqa: E402
from vbb_study.digital_twin.vortex_beam_slm_errors import actual_slm_phase, transformed_pattern_coordinates  # noqa: E402
from vbb_study.digital_twin.vortex_explicit_4f import explicit_4f_relay  # noqa: E402
from vbb_study.digital_twin.vortex_system_route import build_system_route, fourier_resample_fixed_window, physical_axicon_on_own_plane  # noqa: E402
from vbb_study.equations.fields import make_xy_grid  # noqa: E402
from vbb_study.slm_model import apply_slm, pixelate  # noqa: E402

EPS = np.finfo(float).tiny
TWOPI = 2.0 * np.pi
THERMAL = "inferno"
SWEEP_N = 2048
PROD_N = 4096
ALPHAS = np.asarray([0.35, 0.50, 0.65, 0.80, 0.95, 1.10, 1.25], float)
GUARD_R0_M = 1.55e-3
GUARD_R1_M = 1.85e-3
OUTER_R0_M = 2.75e-3
OUTER_R1_M = 3.15e-3
CLOSURE_ITERS = 4
CLOSURE_BETA = 0.72


def savefig(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=500, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def smoothstep01(x: np.ndarray) -> np.ndarray:
    t = np.clip(np.asarray(x, float), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def topology_guard(grid: dict) -> np.ndarray:
    R = np.hypot(np.asarray(grid["X"], float), np.asarray(grid["Y"], float))
    inner = smoothstep01((R - GUARD_R0_M) / (GUARD_R1_M - GUARD_R0_M))
    outer = 1.0 - smoothstep01((R - OUTER_R0_M) / (OUTER_R1_M - OUTER_R0_M))
    return np.clip(inner * outer, 0.0, 1.0)


def radial_complex_projection(field: np.ndarray, grid: dict, dr_m: float = 8e-6) -> np.ndarray:
    U = np.asarray(field, complex)
    X = np.asarray(grid["X"], float); Y = np.asarray(grid["Y"], float)
    R = np.hypot(X, Y); theta = np.arctan2(Y, X)
    edges = np.arange(0.0, float(np.max(R)) + dr_m, dr_m)
    centres = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.digitize(R.ravel(), edges) - 1, 0, len(centres)-1)
    amp = np.abs(U).ravel()
    counts = np.bincount(idx, minlength=len(centres))
    amp_prof = np.bincount(idx, weights=amp, minlength=len(centres)) / np.maximum(counts, 1)
    amp_prof = ndimage.gaussian_filter1d(amp_prof, sigma=1.0, mode="nearest")
    dev = U * np.exp(-1j * float(Q) * theta)
    w = np.maximum(np.abs(U), 1e-9 * max(float(np.max(np.abs(U))), 1.0))
    cr = np.bincount(idx, weights=(w * np.real(dev)).ravel(), minlength=len(centres))
    ci = np.bincount(idx, weights=(w * np.imag(dev)).ravel(), minlength=len(centres))
    ww = np.bincount(idx, weights=w.ravel(), minlength=len(centres))
    phasor = (cr + 1j * ci) / np.maximum(ww, EPS)
    phasor = ndimage.gaussian_filter1d(np.real(phasor), 1.0) + 1j * ndimage.gaussian_filter1d(np.imag(phasor), 1.0)
    phase_prof = np.unwrap(np.angle(phasor))
    amp_r = np.interp(R.ravel(), centres, amp_prof, left=amp_prof[0], right=amp_prof[-1]).reshape(R.shape)
    phase_r = np.interp(R.ravel(), centres, phase_prof, left=phase_prof[0], right=phase_prof[-1]).reshape(R.shape)
    target = amp_r * np.exp(1j * (float(Q) * theta + phase_r))
    scale = np.linalg.norm(U.ravel()) / max(np.linalg.norm(target.ravel()), EPS)
    return np.asarray(target * scale, complex)


def normalise_transfer(C: np.ndarray, grid: dict) -> np.ndarray:
    C = np.asarray(C, complex)
    R = np.hypot(np.asarray(grid["X"], float), np.asarray(grid["Y"], float))
    support = R <= OUTER_R0_M
    mx = max(float(np.max(np.abs(C[support]))), 1.0)
    C = C / mx
    A = np.clip(np.abs(C), 0.12, 1.0)
    return A * np.exp(1j * np.angle(C))


def parity_map(C: np.ndarray, parity: str) -> np.ndarray:
    if parity == "xy": return C[::-1, ::-1]
    if parity == "x": return C[:, ::-1]
    if parity == "y": return C[::-1, :]
    if parity == "none": return C
    raise ValueError(parity)


def encode_transfer(grid: dict, config, C_ax: np.ndarray, *, parity: str, phase_sign: int) -> tuple[np.ndarray, dict]:
    manifest = canonical_hardware_manifest()
    carrier = float(hardware_value(manifest, "carrier_frequency_cpm"))
    C_slm = parity_map(np.asarray(C_ax, complex), parity)
    A = np.clip(np.abs(C_slm), 0.0, 1.0)
    Phi = float(phase_sign) * np.angle(C_slm)
    inv = v7.inverse_sinc_0_1(A)
    M = 1.0 + inv / np.pi
    F = Phi - np.pi * M
    x2, _ = transformed_pattern_coordinates(grid, config.slm2)
    carrier_phase = TWOPI * carrier * x2
    psi = M * np.mod(F + carrier_phase, TWOPI)
    return np.asarray(psi, float), {
        "parity": parity, "phase_sign": int(phase_sign),
        "amplitude_min": float(np.min(A)), "amplitude_mean": float(np.mean(A)),
        "modulation_depth_M_min": float(np.min(M)), "modulation_depth_M_mean": float(np.mean(M)),
        "encoding_reference": "Bolduc et al., Opt. Lett. 38, 3546-3549 (2013), DOI 10.1364/OL.38.003546",
    }


def custom_route(config, propagation_n: int, C_ax_relay: np.ndarray, pcoef: np.ndarray, acoef: np.ndarray, *, parity: str, phase_sign: int) -> tuple[dict, dict]:
    manifest = canonical_hardware_manifest()
    wavelength = float(hardware_value(manifest, "wavelength_m"))
    pixel_pitch = float(hardware_value(manifest, "slm_pixel_pitch_m"))
    carrier = float(hardware_value(manifest, "carrier_frequency_cpm"))
    f4f = float(hardware_value(manifest, "fourf_focal_length_m"))
    iris_radius = float(hardware_value(manifest, "fourier_iris_radius_m"))
    n_ax = float(hardware_value(manifest, "axicon_refractive_index"))
    n_ext = float(hardware_value(manifest, "axicon_external_medium_index"))
    gamma0 = math.radians(float(hardware_value(manifest, "axicon_base_angle_deg")))
    base = build_system_route(f"V{Q}", grid_n=RELAY_N, config=config, window_m=FIT_WINDOW_M)
    grid = base["grid"]
    panel = _panel_from_manifest(manifest)
    psi, hmeta = encode_transfer(grid, config, C_ax_relay, parity=parity, phase_sign=phase_sign)
    command = pixelate(psi, grid, panel)
    actual2, slm2_meta = actual_slm_phase(command, grid, error=config.slm2, pixel_pitch_m=pixel_pitch, lut_phase_rad=None, static_phase_map_rad=None)
    slm2 = apply_slm(np.asarray(base["post_slm1"], complex), actual2, grid, panel, phase_is_prepared=True, quantise_phase=False, apply_fill_factor=True, apply_carrier=False)
    relay = explicit_4f_relay(slm2.total, grid, wavelength_m=wavelength, nominal_focal_length_m=f4f, nominal_iris_radius_m=iris_radius, nominal_carrier_cpm=carrier, error=config.fourf)
    X = np.asarray(grid["X"], float)
    selected = np.asarray(relay["output"], complex) * np.exp(1j * TWOPI * carrier * X)
    if int(propagation_n) == RELAY_N:
        fine_grid = grid; field = selected.copy()
    else:
        fine_grid = make_xy_grid(int(propagation_n), FIT_WINDOW_M / int(propagation_n))
        field = fourier_resample_fixed_window(selected, int(propagation_n))
    phase_err, amp_err = v5.residual_maps(fine_grid, pcoef, acoef)
    field_after_error = field * np.asarray(amp_err, float) * np.exp(1j * np.asarray(phase_err, float))
    ax_t, ax_meta = physical_axicon_on_own_plane(fine_grid, wavelength_m=wavelength, base_angle_rad=gamma0, refractive_index=n_ax, external_index=n_ext, error=config.axicon, surface_height_error_m=None)
    post = np.asarray(field_after_error * ax_t, complex)
    return {"grid": fine_grid, "selected_before_residual": np.asarray(field, complex), "field_on_axicon_plane": np.asarray(field_after_error, complex), "post_axicon": post, "relay_route": {"grid": grid, "post_4f_selected_order": selected}, "metadata": {"route_id": "q20_iterative_circular_bolduc_v8", "hologram": hmeta, "slm2": slm2_meta, "axicon": ax_meta}}, hmeta


def complex_overlap(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    aa = np.asarray(a, complex)[mask].ravel(); bb = np.asarray(b, complex)[mask].ravel()
    return float(abs(np.vdot(aa, bb)) / max(np.linalg.norm(aa) * np.linalg.norm(bb), EPS))


def choose_encoding_convention(config) -> tuple[str, int, list[dict]]:
    base = build_system_route(f"V{Q}", grid_n=RELAY_N, config=config, window_m=FIT_WINDOW_M)
    grid = base["grid"]; U0 = np.asarray(base["field_on_axicon_plane"], complex)
    X = np.asarray(grid["X"], float); Y = np.asarray(grid["Y"], float)
    R = np.hypot(X, Y); th = np.arctan2(Y, X); g = topology_guard(grid)
    Ctest = (1.0 - 0.10 * g * (0.5 + 0.5*np.cos(2.0*th))) * np.exp(1j * 0.28 * g * np.cos(th))
    Ctest = normalise_transfer(Ctest, grid)
    target = U0 * Ctest
    mask = (R >= GUARD_R1_M) & (R <= OUTER_R0_M) & (np.abs(U0) >= 0.08 * float(np.max(np.abs(U0))))
    rows = []; zeros_p = np.zeros(8, float); zeros_a = np.zeros(4, float)
    for parity in ("xy", "x", "y", "none"):
        for sign in (+1, -1):
            rr, _ = custom_route(config, RELAY_N, Ctest, zeros_p, zeros_a, parity=parity, phase_sign=sign)
            ov = complex_overlap(rr["selected_before_residual"], target, mask)
            ia = np.abs(rr["selected_before_residual"])**2; ib = np.abs(target)**2
            ic = float(np.corrcoef(ia[mask].ravel(), ib[mask].ravel())[0,1])
            rows.append({"parity": parity, "phase_sign": sign, "complex_overlap": ov, "intensity_corr": ic})
    best = max(rows, key=lambda r: (r["complex_overlap"], r["intensity_corr"]))
    return str(best["parity"]), int(best["phase_sign"]), rows


def fitted_ratio_update(target: np.ndarray, actual: np.ndarray, grid: dict) -> tuple[np.ndarray, dict]:
    U0 = np.asarray(target, complex); U = np.asarray(actual, complex)
    X = np.asarray(grid["X"], float); Y = np.asarray(grid["Y"], float); R = np.hypot(X, Y)
    ratio = U0 / np.where(np.abs(U) > 1e-12 * max(float(np.max(np.abs(U))), 1.0), U, 1.0 + 0j)
    phase = np.angle(ratio); logamp = np.log(np.clip(np.abs(ratio), 0.55, 1.80))
    basis, names = v5.phase_basis(grid)
    sl = (slice(None, None, 4), slice(None, None, 4)); amp0 = np.abs(U0)[sl]
    mask = (R[sl] >= GUARD_R1_M) & (R[sl] <= OUTER_R0_M) & (amp0 >= 0.06 * float(np.max(np.abs(U0))))
    B = np.stack([b[sl][mask] for b in basis], axis=1)
    w = np.sqrt(np.clip(amp0[mask] / max(float(np.max(amp0[mask])), EPS), 0.08, 1.0))
    ridge = 4.0e-2
    WB = w[:,None] * B
    lhs = WB.T @ WB + ridge * np.eye(B.shape[1])
    cp = np.linalg.solve(lhs, WB.T @ (w * phase[sl][mask]))
    ca = np.linalg.solve(lhs, WB.T @ (w * logamp[sl][mask]))
    phase_fit = np.clip(np.tensordot(cp, basis, axes=(0,0)), -0.60, 0.60)
    amp_fit = np.clip(np.tensordot(ca, basis, axes=(0,0)), -0.28, 0.28)
    g = topology_guard(grid); upd = np.exp(g * (amp_fit + 1j * phase_fit))
    return np.asarray(upd, complex), {"basis_names": names, "phase_coefficients": cp.tolist(), "log_amplitude_coefficients": ca.tolist(), "phase_fit_rms_rad": float(np.sqrt(np.mean(phase_fit[sl][mask]**2))), "logamp_fit_rms": float(np.sqrt(np.mean(amp_fit[sl][mask]**2)))}


def initial_transfer(base_field: np.ndarray, circular_target: np.ndarray, grid: dict, pcoef: np.ndarray, acoef: np.ndarray) -> np.ndarray:
    phase_err, amp_err = v5.residual_maps(grid, pcoef, acoef)
    actual = np.asarray(base_field, complex) * np.asarray(amp_err, float) * np.exp(1j*np.asarray(phase_err, float))
    ratio = np.asarray(circular_target, complex) / np.where(np.abs(actual) > 1e-12 * max(float(np.max(np.abs(actual))), 1.0), actual, 1.0+0j)
    g = topology_guard(grid)
    C = np.exp(g * (np.clip(np.log(np.clip(np.abs(ratio), 0.55, 1.80)), -0.45, 0.45) + 1j*np.angle(ratio)))
    return normalise_transfer(C, grid)


def iterate_transfer(config, pcoef: np.ndarray, acoef: np.ndarray, parity: str, phase_sign: int) -> tuple[np.ndarray, list[dict], np.ndarray]:
    base = build_system_route(f"V{Q}", grid_n=RELAY_N, config=config, window_m=FIT_WINDOW_M)
    grid = base["grid"]; base_field = np.asarray(base["field_on_axicon_plane"], complex)
    target = radial_complex_projection(base_field, grid)
    C = initial_transfer(base_field, target, grid, pcoef, acoef)
    X = np.asarray(grid["X"], float); Y = np.asarray(grid["Y"], float); R = np.hypot(X,Y)
    mask = (R >= GUARD_R1_M) & (R <= OUTER_R0_M) & (np.abs(target) >= 0.06*float(np.max(np.abs(target))))
    history = []
    for it in range(CLOSURE_ITERS):
        rr, hmeta = custom_route(config, RELAY_N, C, pcoef, acoef, parity=parity, phase_sign=phase_sign)
        actual = np.asarray(rr["field_on_axicon_plane"], complex)
        ov = complex_overlap(actual, target, mask)
        amp_nrmse = float(np.sqrt(np.mean((np.abs(actual[mask])/max(float(np.max(np.abs(actual[mask]))),EPS) - np.abs(target[mask])/max(float(np.max(np.abs(target[mask]))),EPS))**2)))
        upd, fitmeta = fitted_ratio_update(target, actual, grid)
        C = normalise_transfer(C * np.exp(CLOSURE_BETA * (np.log(np.clip(np.abs(upd),0.25,4.0)) + 1j*np.angle(upd))), grid)
        history.append({"iteration": it+1, "complex_overlap_before_update": ov, "amplitude_nrmse_before_update": amp_nrmse, "hologram": hmeta, "ratio_fit": fitmeta})
    return C, history, target


def scale_transfer(C: np.ndarray, grid: dict, alpha: float) -> np.ndarray:
    g = topology_guard(grid); loga = np.log(np.clip(np.abs(C), 0.12, 1.0)); ph = np.angle(C)
    return normalise_transfer(np.exp(g * float(alpha) * (loga + 1j*ph)), grid)


def concentric_nominal_target(config, N: int, z_abs: np.ndarray) -> np.ndarray:
    nom = v5.build_route(config, N, slm2_phase=None, pcoef=np.zeros(8,float), acoef=np.zeros(4,float))
    ndet = v5.detector_stack(nom, z_abs)
    return np.stack([v5.radial_symmetrise_plane(p, v5.AXIS_UM) for p in ndet])


def evaluate(alpha: float, N: int, ids: np.ndarray, config, z_abs, target, C_full, pcoef, acoef, parity, phase_sign) -> dict:
    relay_grid = build_system_route(f"V{Q}", grid_n=RELAY_N, config=config, window_m=FIT_WINDOW_M)["grid"]
    C = scale_transfer(C_full, relay_grid, alpha)
    positive = v5.build_route(config, N, slm2_phase=None, pcoef=pcoef, acoef=acoef)
    corrected, hmeta = custom_route(config, N, C, pcoef, acoef, parity=parity, phase_sign=phase_sign)
    pdet = v5.detector_stack(positive, z_abs); cdet = v5.detector_stack(corrected, z_abs)
    pm = v5.concentric_metrics(pdet, target, ids); cm = v5.concentric_metrics(cdet, target, ids)
    wd, top_ok = v5.winding(corrected)
    return {"alpha": float(alpha), "grid_n": int(N), "hologram": hmeta, "topology_q20_all_contours": bool(top_ok), "winding": wd, "positive": pm, "corrected": cm, "corrected_objective": v5.objective(cm), "principal_ring_cv_reduction_fraction": float(1.0 - cm["mean_principal_ring_azimuth_cv"] / max(pm["mean_principal_ring_azimuth_cv"], EPS)), "mirror_rmse_reduction_fraction": float(1.0 - cm["mirror_rmse"] / max(pm["mirror_rmse"], EPS))}


def production(alpha, config, z_abs, z_rel, target, C_full, pcoef, acoef, parity, phase_sign, out):
    relay_grid = build_system_route(f"V{Q}", grid_n=RELAY_N, config=config, window_m=FIT_WINDOW_M)["grid"]
    C = scale_transfer(C_full, relay_grid, alpha)
    positive = v5.build_route(config, PROD_N, slm2_phase=None, pcoef=pcoef, acoef=acoef)
    corrected, hmeta = custom_route(config, PROD_N, C, pcoef, acoef, parity=parity, phase_sign=phase_sign)
    pdet = v5.detector_stack(positive, z_abs); cdet = v5.detector_stack(corrected, z_abs)
    popt = v5.optical_stack(positive, z_abs); copt = v5.optical_stack(corrected, z_abs)
    wc, top_ok = v5.winding(corrected)
    groups = {"inner_train": v5.INNER_TRAIN, "inner_validation": v5.INNER_VALID, "legacy_heldout": v5.LEGACY_HELD, "all_planes": np.arange(len(z_rel), dtype=int)}
    metrics = {}
    for name, ids in groups.items():
        metrics[name] = {"detector_positive": v5.concentric_metrics(pdet, target, ids), "detector_corrected": v5.concentric_metrics(cdet, target, ids), "optical_positive": v5.concentric_metrics(popt, target, ids), "optical_corrected": v5.concentric_metrics(copt, target, ids)}
    np.savez_compressed(out/"iterative_circular_v8_4096_display_arrays.npz", axis_um=v5.AXIS_UM, z_relative_mm=z_rel, concentric_nominal_target=target.astype(np.float32), detector_positive=pdet.astype(np.float32), detector_corrected=cdet.astype(np.float32), optical_positive=popt.astype(np.float32), optical_corrected=copt.astype(np.float32))
    np.save(out/"model_space_slm2_complex_transfer_v8.npy", C.astype(np.complex64))
    ext = [v5.AXIS_UM[0], v5.AXIS_UM[-1], v5.AXIS_UM[0], v5.AXIS_UM[-1]]
    ids = [1,5,9,13,17]
    fig, axs = plt.subplots(3, len(ids), figsize=(15.5, 8.8), constrained_layout=True)
    rows = [(pdet,"diagnosed model"),(target,"explicit concentric target"),(cdet,"iterative corrected")]
    for col, iz in enumerate(ids):
        for row,(stack,label) in enumerate(rows):
            axs[row,col].imshow(stack[iz], origin="lower", extent=ext, cmap=THERMAL, vmin=0, vmax=1, interpolation="nearest")
            axs[row,col].set_aspect("equal"); axs[row,col].set_xticks([]); axs[row,col].set_yticks([])
            if row==0: axs[row,col].set_title(f"z = {z_rel[iz]:.0f} mm")
            if col==0: axs[row,col].set_ylabel(label, fontweight="bold")
    fig.suptitle("q=20 v8: measured diagnosis -> explicitly circular target -> iterative finite-4F correction", fontsize=14, fontweight="bold")
    savefig(fig, out/"poster_iterative_circular_v8_multiplane")
    ids2=[5,11,17]
    fig, axs = plt.subplots(2,3,figsize=(11.2,7.0),constrained_layout=True)
    for col,iz in enumerate(ids2):
        for row,stack in enumerate((popt,copt)):
            axs[row,col].imshow(stack[iz],origin="lower",extent=ext,cmap=THERMAL,vmin=0,vmax=1,interpolation="nearest")
            axs[row,col].set_aspect("equal"); axs[row,col].set_xticks([]); axs[row,col].set_yticks([])
            if row==0: axs[row,col].set_title(f"z = {z_rel[iz]:.0f} mm")
            if col==0: axs[row,col].set_ylabel("diagnosed" if row==0 else "iterative corrected",fontweight="bold")
    fig.suptitle("4096-grid optical field: concentricity-first iterative correction",fontsize=13,fontweight="bold")
    savefig(fig,out/"poster_iterative_circular_v8_optical_before_after")
    return {"production_grid_n":PROD_N,"selected_alpha":float(alpha),"topology_q20_all_contours":bool(top_ok),"winding_corrected":wc,"metrics":metrics,"hologram":hmeta}


def run(source_dir: Path, crosscheck_json: Path, candidate_json: Path, out: Path) -> dict:
    source_dir=Path(source_dir); out=Path(out); out.mkdir(parents=True,exist_ok=True)
    candidate=json.loads(Path(candidate_json).read_text(encoding="utf-8")); cross=json.loads(Path(crosscheck_json).read_text(encoding="utf-8"))
    pcoef,acoef,residual_source=v5.candidate_coefficients(candidate,cross)
    context=v5.v2.build_context(source_dir); z_rel=np.asarray(context["z_rel"],float)
    z0=float(candidate["physical_nuisance"]["selected_z0_mm"]); z_abs=(z0+z_rel)*1e-3
    config=v5.v4.config_from_candidate(candidate,source_dir)
    parity,phase_sign,convention_rows=choose_encoding_convention(config)
    C_full,closure_history,circular_field_target=iterate_transfer(config,pcoef,acoef,parity,phase_sign)
    np.save(out/"iterative_circular_full_transfer.npy",C_full.astype(np.complex64)); np.save(out/"circular_q20_selected_order_target.npy",circular_field_target.astype(np.complex64))
    target=concentric_nominal_target(config,SWEEP_N,z_abs)
    sweep=[]
    for alpha in ALPHAS:
        s=evaluate(float(alpha),SWEEP_N,v5.INNER_VALID,config,z_abs,target,C_full,pcoef,acoef,parity,phase_sign)
        sweep.append(s); print(json.dumps({"alpha":float(alpha),"validation":s},indent=2))
    (out/"iterative_circular_strength_sweep.json").write_text(json.dumps(sweep,indent=2)+"\n",encoding="utf-8")
    passing=[s for s in sweep if s["topology_q20_all_contours"]]
    selected=min(passing,key=lambda s:s["corrected_objective"]) if passing else min(sweep,key=lambda s:s["corrected_objective"])
    alpha_star=float(selected["alpha"])
    prod=production(alpha_star,config,z_abs,z_rel,target,C_full,pcoef,acoef,parity,phase_sign,out)
    held=prod["metrics"]["legacy_heldout"]; pos=held["detector_positive"]; cor=held["detector_corrected"]
    cvred=float(1.0-cor["mean_principal_ring_azimuth_cv"]/max(pos["mean_principal_ring_azimuth_cv"],EPS)); mirred=float(1.0-cor["mirror_rmse"]/max(pos["mirror_rmse"],EPS))
    acceptance={"q20_topology_preserved":bool(prod["topology_q20_all_contours"]),"legacy_heldout_principal_ring_cv_reduction_fraction":cvred,"legacy_heldout_mirror_rmse_reduction_fraction":mirred,"legacy_heldout_radial_profile_corr":float(cor["mean_radial_profile_corr"]),"legacy_heldout_target_r":float(cor["mean_r"]),"legacy_heldout_target_nrmse":float(cor["mean_nrmse"]),"passes_concentricity_gate":bool(prod["topology_q20_all_contours"] and cvred>=0.30 and mirred>=0.25 and cor["mean_radial_profile_corr"]>=0.92)}
    result={"status":"q20_iterative_circular_complex_hologram_candidate_v8","residual_source":residual_source,"encoding_convention":{"selected_parity":parity,"selected_phase_sign":phase_sign,"calibration_rows":convention_rows},"topology_guard_m":{"inner_unity_to":GUARD_R0_M,"full_from":GUARD_R1_M,"outer_full_to":OUTER_R0_M,"outer_zero_from":OUTER_R1_M},"closure_history":closure_history,"selected_validation_strength":selected,"production_validation":prod,"acceptance":acceptance,"target_policy":"explicitly circular nominal radial detector target; no measured angular structure enters the correction target","hardware_ready":False,"evidence_boundary":"corrected fields are numerical model-space predictions only; no corrected BeamGage frame is experimental evidence"}
    (out/"iterative_circular_v8_summary.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8"); print(json.dumps(result,indent=2)); return result


def main()->None:
    p=argparse.ArgumentParser()
    p.add_argument("--source-dir",type=Path,default=EXP/"outputs"/"digital_twin_correction")
    p.add_argument("--crosscheck-json",type=Path,default=ROOT/"outputs"/"validation"/"q20_miao_initializer_crosscheck"/"miao_initializer_crosscheck.json")
    p.add_argument("--candidate-json",type=Path,default=EXP/"candidates"/"q20_detector_aware_model_v3_candidate.json")
    p.add_argument("--out",type=Path,default=ROOT/"outputs"/"validation"/"q20_slm2_iterative_circular_v8")
    a=p.parse_args(); run(a.source_dir,a.crosscheck_json,a.candidate_json,a.out)

if __name__=="__main__":
    main()
