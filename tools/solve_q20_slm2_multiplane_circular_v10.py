"""q=20 correction v10: multi-plane alternating-projection SLM2 synthesis.

Why v10 exists
--------------
The v9 finite-4F IFTA substantially improved the selected-order complex field
and preserved q=20, but the propagated outer Bessel rings retained obvious
cross/fan angular structure.  That failure is expected when the actuator is
optimised against only one transverse complex field: a small residual field
error can remain visually large after axicon propagation.

v10 therefore constrains several *propagated* planes simultaneously.  Each
iteration performs the complete numerical route

    SLM2 command -> physical pixelation/quantisation -> finite 4F/+1 iris
    -> fixed-window resampling -> frozen detector-supported complex residual
    -> refractive axicon -> several z planes,

replaces only the amplitudes in the central Bessel region by explicitly radial
(concentric) nominal amplitudes, back-propagates every constrained plane through
the adjoint free-space/axicon/residual/resampling/4F operators, averages the
returned fields, and re-imposes a phase-only SLM2 command.

The frozen Miao-initialised residual is never re-fit.  The experimental odd z
planes are not used by the correction iterations.  Inner even validation planes
select the best iteration.  The final odd-plane result is a legacy held-out
model check, not pristine blind evidence because the broader project inspected
those planes in earlier model audits.

Unlike v4/v5/v9, the correction is inserted into the *commanded* SLM2 hologram
before pixelation and phase quantisation rather than through the static-map hook.
This is still not hardware-ready until the SLM2 LUT and bench coordinate map are
measured.  All corrected images are numerical predictions, not post-correction
BeamGage data.
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
from real_bmg_digital_twin_correction import FIT_WINDOW_M, PIXEL_M, Q, RELAY_N  # noqa: E402
from vbb_study.digital_twin.concentricity_metrics import stack_metrics as multiring_stack_metrics  # noqa: E402
from vbb_study.digital_twin.detector_response import plane_normalise, sample_camera_response  # noqa: E402
from vbb_study.digital_twin.phase2a_canonical import _panel_from_manifest  # noqa: E402
from vbb_study.digital_twin.phase2a_contracts import canonical_hardware_manifest, hardware_value  # noqa: E402
from vbb_study.digital_twin.vortex_beam_slm_errors import actual_slm_phase, transformed_pattern_coordinates  # noqa: E402
from vbb_study.digital_twin.vortex_continuous_propagation import build_fixed_support_spectrum  # noqa: E402
from vbb_study.digital_twin.vortex_explicit_4f import explicit_4f_relay  # noqa: E402
from vbb_study.digital_twin.vortex_system_route import (  # noqa: E402
    build_multirate_system_route,
    build_system_route,
    fourier_resample_fixed_window,
    physical_axicon_on_own_plane,
)
from vbb_study.equations.fields import fft2c, ifft2c, make_xy_grid  # noqa: E402
from vbb_study.slm_model import apply_slm, pixelate  # noqa: E402
from vbb_study.viz_fields import phase_winding  # noqa: E402

EPS = np.finfo(float).tiny
TWOPI = 2.0 * np.pi
THERMAL = "inferno"
OPT_N = 2048
PROD_N = 4096
ITERATIONS = 24
TRAIN_PLANES = np.asarray([0, 4, 8, 12, 16], dtype=int)
VALID_PLANES = np.asarray([2, 6, 10, 14], dtype=int)
LEGACY_HELD = np.asarray([1, 3, 5, 7, 9, 11, 13, 15, 17], dtype=int)
BETA_SCHEDULE = (0.28, 0.40, 0.52, 0.62)
COMMAND_STRENGTHS = np.asarray([1.00, 0.90, 0.80, 0.70, 0.60], float)
CONTROL_RADIUS_M = 190e-6
PHASE_CLIP_RAD = 1.35


def savefig(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=500, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def wrap_pm_pi(a: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * np.asarray(a, float)))


def smooth_phase(phase: np.ndarray, sigma: float = 0.70) -> np.ndarray:
    """Low-pass a phase map without unwrapping branch cuts."""
    z = ndimage.gaussian_filter(np.exp(1j * np.asarray(phase, float)), sigma=float(sigma), mode="nearest")
    return np.angle(z)


def radial_symmetrise_native(image: np.ndarray, grid: dict, *, dr_samples: float = 0.75) -> np.ndarray:
    """Radial projection on the native optical grid."""
    I = np.maximum(np.asarray(image, float), 0.0)
    R = np.asarray(grid["R"], float)
    dr = max(float(grid["dx"]) * float(dr_samples), 0.25e-6)
    edges = np.arange(0.0, float(np.max(R)) + dr, dr)
    centres = 0.5 * (edges[:-1] + edges[1:])
    idx = np.digitize(R.ravel(), edges) - 1
    good = (idx >= 0) & (idx < centres.size)
    sums = np.bincount(idx[good], weights=I.ravel()[good], minlength=centres.size)
    num = np.bincount(idx[good], minlength=centres.size)
    prof = sums / np.maximum(num, 1)
    prof = ndimage.gaussian_filter1d(prof, sigma=0.65, mode="nearest")
    return np.interp(R.ravel(), centres, prof, left=prof[0], right=prof[-1]).reshape(I.shape)


def fixed_propagation_operators(reference_post_axicon: np.ndarray, grid: dict, z_abs: np.ndarray):
    wl = float(hardware_value(canonical_hardware_manifest(), "wavelength_m"))
    p = build_fixed_support_spectrum(
        reference_post_axicon,
        grid,
        wavelength_m=wl,
        z_max_m=max(0.002, float(np.max(np.abs(z_abs))) + 0.002),
        minimum_retained_spectral_power=0.98,
    )
    support = np.asarray(p.support_mask, bool)
    kz = np.asarray(p.kz_m_inv, complex)

    def forward(field0: np.ndarray, z: float) -> np.ndarray:
        H = np.exp(1j * kz * float(z))
        return ifft2c(fft2c(np.asarray(field0, complex)) * support * H)

    def adjoint(fieldz: np.ndarray, z: float) -> np.ndarray:
        H = np.exp(1j * kz * float(z))
        return ifft2c(fft2c(np.asarray(fieldz, complex)) * support * np.conj(H))

    return forward, adjoint, {
        "retained_spectral_power_fraction": float(p.retained_spectral_power_fraction),
        "z_max_m": float(p.z_max_m),
    }


def resample_adjoint(fine_field: np.ndarray, input_n: int) -> np.ndarray:
    """Adjoint of fourier_resample_fixed_window for a fixed physical window."""
    fine = np.asarray(fine_field, complex)
    output_n = int(fine.shape[0]); input_n = int(input_n)
    if fine.shape != (output_n, output_n) or output_n < input_n:
        raise ValueError("fine field must be square and at least input_n")
    if output_n == input_n:
        return fine.copy()
    spectrum = np.fft.fftshift(np.fft.fft2(fine))
    start = (output_n - input_n) // 2
    cropped = spectrum[start:start+input_n, start:start+input_n].copy()
    delta_samples = 0.5 * (float(input_n) / output_n - 1.0)
    freq = np.fft.fftshift(np.fft.fftfreq(input_n, d=1.0))
    fy, fx = np.meshgrid(freq, freq, indexing="ij")
    phase = np.exp(1j * TWOPI * delta_samples * (fx + fy))
    cropped *= np.conj(phase)
    return np.fft.ifft2(np.fft.ifftshift(cropped))


def slm2_command_route(config, propagation_n: int, correction_command_rad: np.ndarray, pcoef: np.ndarray, acoef: np.ndarray):
    """Build the route with correction added to the actual SLM2 command.

    The correction is pixelated and quantised with the carrier rather than being
    injected via the static phase-map hook.
    """
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
    corr = np.asarray(correction_command_rad, float)
    if corr.shape != (RELAY_N, RELAY_N):
        raise ValueError("correction command must match relay grid")
    panel = _panel_from_manifest(manifest)
    x2, _ = transformed_pattern_coordinates(grid, config.slm2)
    command = pixelate(TWOPI * carrier * x2 + corr, grid, panel)
    actual2, slm2_meta = actual_slm_phase(
        command,
        grid,
        error=config.slm2,
        pixel_pitch_m=pixel_pitch,
        lut_phase_rad=None,
        static_phase_map_rad=None,
    )
    slm2 = apply_slm(
        np.asarray(base["post_slm1"], complex), actual2, grid, panel,
        phase_is_prepared=True, quantise_phase=False, apply_fill_factor=True,
        apply_carrier=False,
    )
    relay = explicit_4f_relay(
        slm2.total, grid,
        wavelength_m=wavelength,
        nominal_focal_length_m=f4f,
        nominal_iris_radius_m=iris_radius,
        nominal_carrier_cpm=carrier,
        error=config.fourf,
    )
    X = np.asarray(grid["X"], float)
    selected = np.asarray(relay["output"], complex) * np.exp(1j * TWOPI * carrier * X)

    N = int(propagation_n)
    if N == RELAY_N:
        fine_grid = grid
        selected_fine = selected
    else:
        fine_grid = make_xy_grid(N, FIT_WINDOW_M / N)
        selected_fine = fourier_resample_fixed_window(selected, N)
    phase_err, amp_err = v5.residual_maps(fine_grid, pcoef, acoef)
    residual = np.asarray(amp_err, float) * np.exp(1j * np.asarray(phase_err, float))
    field_on_axicon = selected_fine * residual
    ax_t, ax_meta = physical_axicon_on_own_plane(
        fine_grid,
        wavelength_m=wavelength,
        base_angle_rad=gamma0,
        refractive_index=n_ax,
        external_index=n_ext,
        error=config.axicon,
        surface_height_error_m=None,
    )
    post_axicon = field_on_axicon * ax_t
    return {
        "grid": fine_grid,
        "post_axicon": np.asarray(post_axicon, complex),
        "field_on_axicon_plane": np.asarray(field_on_axicon, complex),
        "selected_order_pre_residual": np.asarray(selected_fine, complex),
        "residual_multiplier": np.asarray(residual, complex),
        "axicon_transmission": np.asarray(ax_t, complex),
        "relay_route": {
            "grid": grid,
            "post_slm1": np.asarray(base["post_slm1"], complex),
            "post_slm2": np.asarray(slm2.total, complex),
            "post_4f_selected_order": np.asarray(selected, complex),
        },
        "metadata": {
            "route_id": "q20_slm2_command_phase_multiplane_v10",
            "slm2": slm2_meta,
            "axicon": ax_meta,
        },
    }


def nominal_target(config, z_abs: np.ndarray):
    nominal = build_multirate_system_route(
        f"V{Q}", relay_grid_n=RELAY_N, propagation_grid_n=OPT_N,
        window_m=FIT_WINDOW_M, config=config,
    )
    fwd, _, prop_meta = fixed_propagation_operators(nominal["post_axicon"], nominal["grid"], z_abs)
    target_amp = []
    for z in np.asarray(z_abs, float):
        u = fwd(nominal["post_axicon"], float(z))
        sym = radial_symmetrise_native(np.abs(u) ** 2, nominal["grid"])
        target_amp.append(np.sqrt(np.maximum(sym, 0.0)))
    return nominal, np.stack(target_amp), fwd, prop_meta


def sample_display_stack(fields: list[np.ndarray], grid: dict) -> np.ndarray:
    axis = np.asarray(v5.AXIS_UM, float)
    x = np.asarray(grid["x"], float)
    pix = np.interp(axis * 1e-6, x, np.arange(len(x), dtype=float))
    yy, xx = np.meshgrid(pix, pix, indexing="ij")
    out = []
    for u in fields:
        I = np.abs(np.asarray(u, complex)) ** 2
        crop = ndimage.map_coordinates(I, [yy, xx], order=1, mode="constant", cval=0.0)
        out.append(crop / max(float(np.max(crop)), EPS))
    return np.stack(out)


def detector_stack_from_fields(fields: list[np.ndarray], grid: dict) -> np.ndarray:
    native = np.stack([np.abs(np.asarray(u, complex)) ** 2 for u in fields])
    shown, _ = sample_camera_response(
        native, np.asarray(grid["x"], float), v5.AXIS_UM * 1e-6,
        pixel_pitch_m=PIXEL_M, quadrature_n=3,
    )
    return plane_normalise(shown)


def multiring_objective(stack: np.ndarray, target: np.ndarray, ids: np.ndarray) -> tuple[float, dict]:
    m = multiring_stack_metrics(stack, target, v5.AXIS_UM, np.asarray(ids, int))
    # Radius wobble is measured in microns; rescale it to dimensionless order.
    score = (
        1.00 * m["mean_ring_intensity_cv"]
        + 0.85 * m["mean_angular_harmonic_energy"]
        + 0.055 * m["mean_ring_radius_std_um"]
        + 0.012 * m["mean_ring_radius_peak_to_peak_um"]
    )
    return float(score), m


def topology(route_result: dict) -> tuple[dict, bool]:
    vals, ok = {}, True
    for rmm in v5.WINDING_RADII_MM:
        w = float(phase_winding(route_result["post_axicon"], route_result["grid"], rmm*1e-3, n_phi=720))
        vals[f"radius_{rmm:.1f}_mm"] = w
        ok &= abs(w - float(Q)) <= 0.25
    return vals, bool(ok)


def command_from_backprop(back_slm: np.ndarray, base: dict, current: np.ndarray, iteration: int) -> np.ndarray:
    post1 = np.asarray(base["post_slm1"], complex)
    canonical = np.asarray(base["post_slm2"], complex)
    mask = np.abs(post1) >= 1e-5 * float(np.max(np.abs(post1)))
    desired_factor = np.ones_like(post1, complex)
    desired_factor[mask] = np.asarray(back_slm, complex)[mask] / post1[mask]
    canonical_factor = np.ones_like(post1, complex)
    canonical_factor[mask] = canonical[mask] / post1[mask]
    raw = wrap_pm_pi(np.angle(desired_factor) - np.angle(canonical_factor))
    raw = smooth_phase(raw, sigma=0.65)
    raw = np.clip(raw, -PHASE_CLIP_RAD, PHASE_CLIP_RAD)
    # Relax the command rather than replacing it in one step.  Later iterations
    # are allowed to move more strongly once the multi-plane phase has settled.
    eta = (0.34, 0.42, 0.50, 0.56)[min(int(iteration)//6, 3)]
    z = (1.0-eta)*np.exp(1j*np.asarray(current,float)) + eta*np.exp(1j*raw)
    updated = np.angle(z)
    updated = smooth_phase(updated, sigma=0.45)
    return np.clip(updated, -PHASE_CLIP_RAD, PHASE_CLIP_RAD)


def optimise_multiplane(config, z_abs: np.ndarray, target_amp: np.ndarray, pcoef: np.ndarray, acoef: np.ndarray, out: Path):
    # v9 is a strong single-plane initialiser and is cheap on the 512 relay grid.
    init_dir = out / "v9_initialiser"
    init_dir.mkdir(exist_ok=True)
    static0, v9_history, _, = v9.ifta_solve(config, pcoef, acoef, init_dir)
    command = np.asarray(static0, float)

    # Canonical relay is used only for the phase-only projection reference.
    canonical = build_system_route(f"V{Q}", grid_n=RELAY_N, config=config, window_m=FIT_WINDOW_M)
    nominal, _, fwd, prop_meta = nominal_target(config, z_abs)
    _, adj, _ = fixed_propagation_operators(nominal["post_axicon"], nominal["grid"], z_abs)
    control_mask = np.asarray(nominal["grid"]["R"], float) <= CONTROL_RADIUS_M
    target_display_fields = [target_amp[i] + 0j for i in range(len(z_abs))]
    target_display = sample_display_stack(target_display_fields, nominal["grid"])

    best = {"score": float("inf"), "command": command.copy(), "iteration": 0, "metrics": None}
    history = []
    for it in range(ITERATIONS):
        route = slm2_command_route(config, OPT_N, command, pcoef, acoef)
        current_fields = [fwd(route["post_axicon"], float(z)) for z in z_abs]
        # Multi-plane amplitude constraints on train planes only.
        returned = []
        beta = BETA_SCHEDULE[min(it//6, len(BETA_SCHEDULE)-1)]
        for iz in TRAIN_PLANES:
            u = current_fields[int(iz)]
            amp = target_amp[int(iz)]
            # Match target power in the controlled Bessel region; do not ask the
            # phase-only actuator to create/remove arbitrary total power.
            current_power = float(np.sum(np.abs(u[control_mask])**2))
            target_power = float(np.sum(amp[control_mask]**2))
            scale = math.sqrt(current_power / max(target_power, EPS))
            target_complex = scale * amp * np.exp(1j*np.angle(u))
            constrained = np.asarray(u, complex).copy()
            constrained[control_mask] = (
                (1.0-float(beta))*u[control_mask]
                + float(beta)*target_complex[control_mask]
            )
            returned.append(adj(constrained, float(z_abs[int(iz)])))
        back_post_axicon = np.mean(np.stack(returned), axis=0)
        # Adjoint of post-axicon multiplication by residual and axicon phase.
        back_selected_fine = back_post_axicon * np.conj(route["residual_multiplier"] * route["axicon_transmission"])
        back_selected = resample_adjoint(back_selected_fine, RELAY_N)
        back_slm = v9.relay_adjoint(back_selected, route["relay_route"]["grid"], config)
        proposal = command_from_backprop(back_slm, canonical, command, it)

        # Validate the proposal on inner even planes before accepting it.
        trial = slm2_command_route(config, OPT_N, proposal, pcoef, acoef)
        trial_fields = [fwd(trial["post_axicon"], float(z)) for z in z_abs]
        trial_display = sample_display_stack(trial_fields, trial["grid"])
        val_score, val_multi = multiring_objective(trial_display, target_display, VALID_PLANES)
        current_display = sample_display_stack(current_fields, route["grid"])
        cur_score, cur_multi = multiring_objective(current_display, target_display, VALID_PLANES)
        accepted = bool(val_score <= cur_score * 1.002)
        if accepted:
            command = proposal
            kept_score, kept_multi = val_score, val_multi
        else:
            kept_score, kept_multi = cur_score, cur_multi
        if kept_score < best["score"]:
            best = {
                "score": float(kept_score), "command": command.copy(),
                "iteration": int(it+1), "metrics": kept_multi,
            }
        row = {
            "iteration": int(it+1), "beta": float(beta), "accepted": accepted,
            "validation_score_before": float(cur_score), "validation_score_proposal": float(val_score),
            "validation_score_kept": float(kept_score),
            "validation_multiring_kept": kept_multi,
        }
        history.append(row)
        print(json.dumps(row, indent=2))

    np.save(out/"multiplane_v10_best_command_phase_rad.npy", np.asarray(best["command"],np.float32))
    return np.asarray(best["command"],float), history, v9_history, target_display, prop_meta, best


def evaluation_target(config, z_abs: np.ndarray):
    nominal = build_multirate_system_route(
        f"V{Q}",relay_grid_n=RELAY_N,propagation_grid_n=OPT_N,
        window_m=FIT_WINDOW_M,config=config,
    )
    fields = v5.field_stack(nominal, z_abs) if hasattr(v5, "field_stack") else None
    if fields is None:
        wl=float(hardware_value(canonical_hardware_manifest(),"wavelength_m"))
        p=build_fixed_support_spectrum(nominal["post_axicon"],nominal["grid"],wavelength_m=wl,z_max_m=max(np.max(np.abs(z_abs))+0.002,0.002),minimum_retained_spectral_power=0.98)
        fields=[ifft2c(p.spectrum*np.exp(1j*p.kz_m_inv*float(z))) for z in z_abs]
    optical=np.stack([radial_symmetrise_native(np.abs(u)**2,nominal["grid"]) for u in fields])
    shown,_=sample_camera_response(optical,np.asarray(nominal["grid"]["x"],float),v5.AXIS_UM*1e-6,pixel_pitch_m=PIXEL_M,quadrature_n=3)
    detector=plane_normalise(shown)
    return detector


def production(config, z_abs, z_rel, target_det, command, pcoef, acoef, out):
    positive = v5.build_route(config, PROD_N, slm2_phase=None, pcoef=pcoef, acoef=acoef)
    # command-route correction at production resolution
    corrected = slm2_command_route(config, PROD_N, command, pcoef, acoef)
    pdet=v5.detector_stack(positive,z_abs); cdet=v5.detector_stack(corrected,z_abs)
    popt=v5.optical_stack(positive,z_abs); copt=v5.optical_stack(corrected,z_abs)
    wp,_=topology(positive); wc,top_ok=topology(corrected)
    groups={"train":TRAIN_PLANES,"inner_validation":VALID_PLANES,"legacy_heldout":LEGACY_HELD,"all_planes":np.arange(len(z_rel),dtype=int)}
    metrics={}
    for name,ids in groups.items():
        pscore,pmulti=multiring_objective(pdet,target_det,ids); cscore,cmulti=multiring_objective(cdet,target_det,ids)
        metrics[name]={
            "detector_positive_multiring_score":pscore,"detector_corrected_multiring_score":cscore,
            "detector_positive_multiring":pmulti,"detector_corrected_multiring":cmulti,
            "detector_positive":v5.concentric_metrics(pdet,target_det,ids),
            "detector_corrected":v5.concentric_metrics(cdet,target_det,ids),
            "optical_positive":v5.concentric_metrics(popt,target_det,ids),
            "optical_corrected":v5.concentric_metrics(copt,target_det,ids),
        }
    np.save(out/"model_space_slm2_command_phase_multiplane_v10_rad.npy",np.asarray(command,np.float32))
    np.savez_compressed(out/"multiplane_v10_4096_display_arrays.npz",axis_um=v5.AXIS_UM,z_relative_mm=z_rel,concentric_target=target_det.astype(np.float32),detector_positive=pdet.astype(np.float32),detector_corrected=cdet.astype(np.float32),optical_positive=popt.astype(np.float32),optical_corrected=copt.astype(np.float32))

    ext=[v5.AXIS_UM[0],v5.AXIS_UM[-1],v5.AXIS_UM[0],v5.AXIS_UM[-1]]; ids=[1,5,9,13,17]
    fig,axs=plt.subplots(3,len(ids),figsize=(15.5,8.8),constrained_layout=True)
    for col,iz in enumerate(ids):
        for row,(stack,label) in enumerate(((pdet,"diagnosed model"),(target_det,"concentric target"),(cdet,"v10 corrected"))):
            axs[row,col].imshow(stack[iz],origin="lower",extent=ext,cmap=THERMAL,vmin=0,vmax=1,interpolation="nearest"); axs[row,col].set_aspect("equal"); axs[row,col].set_xticks([]); axs[row,col].set_yticks([])
            if row==0: axs[row,col].set_title(f"z = {z_rel[iz]:.0f} mm")
            if col==0: axs[row,col].set_ylabel(label,fontweight="bold")
    fig.suptitle("q=20 v10: multi-plane phase-only SLM2 correction of the full ring train",fontsize=14,fontweight="bold"); savefig(fig,out/"poster_multiplane_v10_detector")

    ids2=[5,11,17]; fig,axs=plt.subplots(2,3,figsize=(11.2,7.0),constrained_layout=True)
    for col,iz in enumerate(ids2):
        for row,stack in enumerate((popt,copt)):
            axs[row,col].imshow(stack[iz],origin="lower",extent=ext,cmap=THERMAL,vmin=0,vmax=1,interpolation="nearest"); axs[row,col].set_aspect("equal"); axs[row,col].set_xticks([]); axs[row,col].set_yticks([])
            if row==0: axs[row,col].set_title(f"z = {z_rel[iz]:.0f} mm")
            if col==0: axs[row,col].set_ylabel("diagnosed" if row==0 else "v10 corrected",fontweight="bold")
    fig.suptitle("4096-grid optical field: multi-plane correction before/after",fontsize=13,fontweight="bold"); savefig(fig,out/"poster_multiplane_v10_optical")
    return {"production_grid_n":PROD_N,"topology_q20_all_contours":bool(top_ok),"winding_positive":wp,"winding_corrected":wc,"metrics":metrics}


def run(source_dir:Path,candidate_json:Path,residual_json:Path,out:Path)->dict:
    source_dir=Path(source_dir); out=Path(out); out.mkdir(parents=True,exist_ok=True)
    config,candidate=v9.config_from_files(candidate_json,source_dir); pcoef,acoef,frozen=v9.frozen_residual(residual_json)
    summary0=json.loads((source_dir/"run_summary.json").read_text(encoding="utf-8")); z_rel=np.asarray(summary0["data"]["z_relative_mm"],float); z0=float(candidate["physical_nuisance"]["selected_z0_mm"]); z_abs=(z0+z_rel)*1e-3
    nominal,target_amp,_,_ = nominal_target(config,z_abs)
    command,history,v9_history,target_display,prop_meta,best=optimise_multiplane(config,z_abs,target_amp,pcoef,acoef,out)
    target_det=evaluation_target(config,z_abs)

    # Strength regularisation is selected only on inner even validation planes.
    sweep=[]
    for s in COMMAND_STRENGTHS:
        route=slm2_command_route(config,OPT_N,float(s)*command,pcoef,acoef)
        det=v5.detector_stack(route,z_abs)
        wd,ok=topology(route)
        score,multi=multiring_objective(det,target_det,VALID_PLANES)
        sweep.append({"strength":float(s),"topology_q20_all_contours":bool(ok),"winding":wd,"validation_multiring_score":score,"validation_multiring":multi})
        print(json.dumps(sweep[-1],indent=2))
    (out/"multiplane_v10_strength_sweep.json").write_text(json.dumps(sweep,indent=2)+"\n",encoding="utf-8")
    passing=[x for x in sweep if x["topology_q20_all_contours"]]
    selected=min(passing,key=lambda x:x["validation_multiring_score"]) if passing else min(sweep,key=lambda x:x["validation_multiring_score"])
    final_command=float(selected["strength"])*command
    prod=production(config,z_abs,z_rel,target_det,final_command,pcoef,acoef,out)
    held=prod["metrics"]["legacy_heldout"]; pm=held["detector_positive_multiring"]; cm=held["detector_corrected_multiring"]
    reductions={
        "ring_intensity_cv":float(1.0-cm["mean_ring_intensity_cv"]/max(pm["mean_ring_intensity_cv"],EPS)),
        "angular_harmonic_energy":float(1.0-cm["mean_angular_harmonic_energy"]/max(pm["mean_angular_harmonic_energy"],EPS)),
        "ring_radius_std":float(1.0-cm["mean_ring_radius_std_um"]/max(pm["mean_ring_radius_std_um"],EPS)),
        "ring_radius_peak_to_peak":float(1.0-cm["mean_ring_radius_peak_to_peak_um"]/max(pm["mean_ring_radius_peak_to_peak_um"],EPS)),
        "multiring_score":float(1.0-held["detector_corrected_multiring_score"]/max(held["detector_positive_multiring_score"],EPS)),
    }
    acceptance={
        "q20_topology_preserved":bool(prod["topology_q20_all_contours"]),
        "legacy_heldout_reductions":reductions,
        "legacy_heldout_corrected_multiring":cm,
        "passes_concentricity_gate":bool(
            prod["topology_q20_all_contours"]
            and reductions["multiring_score"]>=0.28
            and reductions["ring_intensity_cv"]>=0.22
            and reductions["angular_harmonic_energy"]>=0.22
            and reductions["ring_radius_std"]>=0.12
        ),
    }
    result={
        "status":"q20_multiplane_circular_slm2_command_candidate_v10",
        "frozen_residual_source":Path(residual_json).name,
        "train_planes":TRAIN_PLANES.tolist(),"inner_validation_planes":VALID_PLANES.tolist(),"legacy_heldout_planes":LEGACY_HELD.tolist(),
        "best_iteration":best,"selected_strength":selected,"production_validation":prod,"acceptance":acceptance,
        "propagation_support":prop_meta,
        "target_policy":"radially projected nominal optical amplitude enforced at multiple propagated train planes; measured angular structure is never a correction target",
        "slm_policy":"correction added to SLM2 command before physical pixelation and phase quantisation",
        "hardware_ready":False,
        "evidence_boundary":"corrected fields are numerical model-space predictions only; no corrected BeamGage frame is experimental evidence",
        "literature_basis":[
            "Gerchberg-Saxton alternating projections / multi-plane phase retrieval",
            "Miao et al., Opt. Express 30, 11360-11371 (2022), doi:10.1364/OE.454796"
        ],
        "optimisation_history":history,"v9_initialiser_history":v9_history,
    }
    (out/"multiplane_v10_summary.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8"); print(json.dumps(result,indent=2)); return result


def main()->None:
    p=argparse.ArgumentParser(); p.add_argument("--source-dir",type=Path,default=EXP/"outputs"/"digital_twin_correction"); p.add_argument("--candidate-json",type=Path,default=EXP/"candidates"/"q20_detector_aware_model_v3_candidate.json"); p.add_argument("--residual-json",type=Path,default=EXP/"candidates"/"q20_miao_initialized_complex_residual_v1.json"); p.add_argument("--out",type=Path,default=ROOT/"outputs"/"validation"/"q20_slm2_multiplane_circular_v10"); a=p.parse_args(); run(a.source_dir,a.candidate_json,a.residual_json,a.out)

if __name__=="__main__": main()
