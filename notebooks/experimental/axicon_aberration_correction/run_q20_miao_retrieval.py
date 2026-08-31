"""Run the corrected q=20 Miao-style retrieval on the full 18x4 BMG scan.

The runner preserves genuine plane-to-plane beam motion, optimizes k_perp at
each z plane, increases aberration modal order until convergence, reconstructs
the radial phase from k_perp(z), and blocks SLM output until the required bench
calibrations and conjugate-branch check are present.

Camera-stage runout is handled separately from optical beam walk: the preferred
calibration is one raw-sensor optical-axis [y,x] position for every z stage
position, measured with an aligned reference beam.  Only repeat-to-repeat jitter
within one z plane is registered away.

A geometric input-plane -> SLM2 phase remap is permitted only when the two
planes have been experimentally confirmed to be conjugate.  Otherwise the
complex field must be propagated/back-propagated through the measured relay; a
simple scale/rotation/parity transform is deliberately blocked.
"""
from __future__ import annotations

from pathlib import Path
import csv
import json

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from modal_vortex_bessel import read_bmg, preprocess, find_dark_core_center, estimate_global_kr
from miao_full_retrieval import (
    fit_plane_adaptive, fit_plane_adaptive_at_k_perp,
    compute_k_perp_cost_curve, find_k_perp_candidate_minima,
    choose_k_perp_regularisation, refine_continuous_k_perp_path,
    assess_k_perp_path_stability, assemble_full_aperture,
    interpolate_to_cartesian, map_input_phase_to_slm2, correction_manifest,
)
from miao_correction_visualization import write_model_comparison


def load_scan_preserve_plane_shift(data_dir, roi_size=768,
                                   expected_planes=18, expected_repeats=4):
    """Average repeat jitter only; never recenter one z plane onto another."""
    folder = Path(data_dir)
    groups = {}
    for p in folder.glob("z*_*.bmg"):
        try:
            zi = int(p.stem.split("_")[0][1:])
        except Exception:
            continue
        groups.setdefault(zi, []).append(p)
    keys = sorted(groups)
    if len(keys) != expected_planes:
        raise ValueError(f"Expected {expected_planes} z planes, found {len(keys)}")
    for zi in keys:
        groups[zi] = sorted(groups[zi])
        if len(groups[zi]) != expected_repeats:
            raise ValueError(f"z{zi}: expected {expected_repeats} repeats, found {len(groups[zi])}")

    frame_centres, all_centres = {}, []
    sensor_shape = None
    for zi in keys:
        rows = []
        for p in groups[zi]:
            a = preprocess(read_bmg(p))
            if sensor_shape is None:
                sensor_shape = tuple(map(int, a.shape))
            elif tuple(a.shape) != sensor_shape:
                raise ValueError(f"Inconsistent BMG sensor shape: {a.shape} != {sensor_shape}")
            cy, cx, score = find_dark_core_center(a)
            rows.append((cy, cx, score))
            all_centres.append((cy, cx))
        frame_centres[zi] = rows

    if sensor_shape is None:
        raise FileNotFoundError("No readable BMG frames found")
    sy, sx = sensor_shape
    roi_size = int(min(roi_size, sy, sx))
    global_cy, global_cx = np.median(np.asarray(all_centres, float), axis=0)
    h = roi_size//2
    y0 = max(0, min(sy-roi_size, int(round(global_cy))-h))
    x0 = max(0, min(sx-roi_size, int(round(global_cx))-h))
    estimated_axis_crop_yx = (float(global_cy-y0), float(global_cx-x0))

    images, qc, mean_shift_by_z = [], [], []
    for zi in keys:
        centres = np.asarray([(c[0], c[1]) for c in frame_centres[zi]], float)
        target = np.median(centres, axis=0)
        repeats, shifts = [], []
        for p, (cy, cx, score) in zip(groups[zi], frame_centres[zi]):
            a = preprocess(read_bmg(p))
            shift = (float(target[0]-cy), float(target[1]-cx))
            shifts.append(shift)
            # Remove repeat-to-repeat acquisition jitter within this z only.
            # The z-plane target itself is never shifted to another z-plane target.
            a = ndimage.shift(a, shift, order=1, mode="constant", cval=0)
            repeats.append(a[y0:y0+roi_size, x0:x0+roi_size])
            qc.append({"z_index": zi, "file": p.name,
                       "core_y_raw_px": float(cy), "core_x_raw_px": float(cx),
                       "core_score": float(score),
                       "repeat_registration_shift_y_px": shift[0],
                       "repeat_registration_shift_x_px": shift[1],
                       "crop_origin_y_px": int(y0), "crop_origin_x_px": int(x0)})
        images.append(np.mean(np.stack(repeats), axis=0))
        mean_shift_by_z.append(np.mean(np.asarray(shifts, float), axis=0))
    crop_origin = (int(y0), int(x0))
    return (images, np.asarray(keys), estimated_axis_crop_yx, crop_origin,
            np.asarray(mean_shift_by_z, float), sensor_shape, qc)


def calibrated_axes_in_crop(calibration, estimated_axis_crop_yx, crop_origin_yx,
                            mean_registration_shift_by_z, n_planes):
    """Return one optical-axis coordinate per z plane in the averaged ROI.

    Preferred input is `camera_optical_axis_yx_px_by_z`, measured in raw camera
    pixels at the same stage positions.  The small mean within-plane registration
    shift is applied because the BMG repeats were shifted before averaging.
    """
    y0, x0 = crop_origin_yx
    shifts = np.asarray(mean_registration_shift_by_z, float)
    if shifts.shape != (int(n_planes), 2):
        raise ValueError("mean_registration_shift_by_z must have shape (n_planes, 2)")

    by_z = calibration.get("camera_optical_axis_yx_px_by_z")
    if by_z is not None:
        raw = np.asarray(by_z, float)
        if raw.shape != (int(n_planes), 2):
            raise ValueError(
                f"camera_optical_axis_yx_px_by_z must have shape ({n_planes}, 2)")
        axes = raw + shifts - np.asarray([y0, x0], float)
        return axes, True, "per-z measured reference axis"

    single = calibration.get("camera_optical_axis_yx_px")
    single_valid = bool(calibration.get("camera_optical_axis_single_value_valid_for_all_z", False))
    if single is not None and single_valid:
        raw = np.asarray(single, float)
        if raw.shape != (2,):
            raise ValueError("camera_optical_axis_yx_px must be [y, x]")
        axes = raw[None, :] + shifts - np.asarray([y0, x0], float)
        axes = np.repeat(raw[None, :], int(n_planes), axis=0) + shifts - np.asarray([y0, x0], float)
        return axes, True, "single measured axis; constant-with-z independently verified"

    estimated = np.repeat(np.asarray(estimated_axis_crop_yx, float)[None, :],
                          int(n_planes), axis=0)
    return estimated, False, "median observed beam-core diagnostic axis"


def _reference_rows_in_rho_order(reference, retrievals, z_abs_m, wavelength_m):
    """Reference rows are supplied in acquisition-z order; match the rho sort."""
    if reference is None:
        return None
    ref = np.asarray(reference, float)
    if ref.shape[0] != len(retrievals):
        raise ValueError("input reference must contain one angular row per z plane")
    k = 2*np.pi/float(wavelength_m)
    kp = np.asarray([r.k_perp_m_inv for r in retrievals], float)
    order = np.argsort(np.asarray(z_abs_m, float)*kp/k)
    return ref[order]


def _write_retrieval_rows(path, retrievals, z_relative_mm, axis_calibrated,
                          axis_source):
    rows = [{
        "z_index": r.z_index, "z_relative_mm": r.z_relative_m*1e3,
        "retrieval_axis_y_px": r.center_y_px, "retrieval_axis_x_px": r.center_x_px,
        "camera_optical_axis_calibrated": axis_calibrated,
        "camera_axis_source": axis_source,
        "k_perp_opt_m_inv": r.k_perp_m_inv,
        "aberration_order_max": r.aberration_order_max,
        "fit_cost": r.fit_cost, "fit_corr": r.fit_corr, "fit_nrmse": r.fit_nrmse,
    } for r in retrievals]
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    return rows


def _save_retrieval_coefficients(path, retrievals, max_order):
    m_grid = np.arange(-int(max_order), int(max_order)+1, dtype=int)
    coeffs = np.zeros((len(retrievals), len(m_grid)), complex)
    active = np.zeros(coeffs.shape, bool)
    for row, retrieval in enumerate(retrievals):
        idx = np.searchsorted(m_grid, retrieval.m_values)
        coeffs[row, idx] = retrieval.coeffs
        active[row, idx] = True
    np.savez_compressed(path, m_values=m_grid, coeffs_real=coeffs.real,
                        coeffs_imag=coeffs.imag, active=active)


def _plot_global_branch_diagnostic(out, z_mm, k_grid, curves, candidates,
                                   independent_k, global_k, stability, path_rows):
    normalized = ((curves-curves.min(axis=1, keepdims=True)) /
                  np.maximum(np.ptp(curves, axis=1, keepdims=True), 1e-12))
    fig, axes = plt.subplots(3, 1, figsize=(15, 13), constrained_layout=True,
                             gridspec_kw={"height_ratios": [2.3, 1, 1]})
    ax = axes[0]
    mesh = ax.pcolormesh(z_mm, k_grid/1e3, normalized.T, shading="nearest",
                         cmap="viridis_r", vmin=0, vmax=1)
    for j, candidate in enumerate(candidates):
        ax.scatter(np.full(len(candidate["k_perp_m_inv"]), z_mm[j]),
                   candidate["k_perp_m_inv"]/1e3, s=18, facecolors="none",
                   edgecolors="white", linewidths=.65)
    ax.plot(z_mm, independent_k/1e3, "x--", color="#D55E00", lw=1.2,
            ms=7, label="independent optimum")
    ax.errorbar(z_mm, global_k/1e3, yerr=stability["std_m_inv"]/1e3,
                fmt="o-", color="cyan", ecolor="cyan", capsize=2, lw=2,
                label="global continuous path ± stability SD")
    ax.set(title="Full radial cost landscapes and selected branch",
           xlabel="relative z (mm)", ylabel=r"$k_\perp$ (krad/m)")
    ax.legend(fontsize=8); fig.colorbar(mesh, ax=ax, label="plane-normalized radial cost")

    global_jump = np.abs(np.diff(global_k)/global_k[:-1])
    independent_jump = np.abs(np.diff(independent_k)/independent_k[:-1])
    axes[1].plot(z_mm[1:], independent_jump*100, "x--", color="#D55E00",
                 label="independent")
    axes[1].plot(z_mm[1:], global_jump*100, "o-", color="#0072B2",
                 label="global")
    axes[1].axhline(8, color="0.4", ls=":", label="8% reliability gate")
    axes[1].set(title="Adjacent fractional transverse-wavenumber change",
                xlabel="relative z (mm)", ylabel="absolute change (%)")
    axes[1].grid(alpha=.25); axes[1].legend(fontsize=8)

    independent_cost = np.asarray([row["independent_radial_cost"] for row in path_rows])
    global_cost = np.asarray([row["global_radial_cost"] for row in path_rows])
    axes[2].plot(z_mm, independent_cost, "x--", color="#D55E00",
                 label="independent radial cost")
    axes[2].plot(z_mm, global_cost, "o-", color="#0072B2",
                 label="global radial cost")
    axes[2].set(title="Data-fit price of branch continuity", xlabel="relative z (mm)",
                ylabel="ideal-mode radial cost")
    axes[2].grid(alpha=.25); axes[2].legend(fontsize=8)
    fig.suptitle("q=20 continuity-aware k-perp branch selection", fontsize=15)
    fig.savefig(Path(out)/"k_perp_global_branch_diagnostic.png", dpi=400,
                bbox_inches="tight")
    fig.savefig(Path(out)/"k_perp_global_branch_diagnostic.pdf", dpi=400,
                bbox_inches="tight")
    plt.close(fig)


def run(data_dir, output_dir, *, calibration_json=None,
        z_relative_mm=np.arange(-17.0, 1.0), wavelength_m=1030e-9,
        pixel_pitch_m=5.5e-6, q=20, max_aberration_order=30):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    calibration = {}
    if calibration_json:
        calibration = json.loads(Path(calibration_json).read_text(encoding="utf-8"))

    (images, z_keys, estimated_axis, crop_origin, mean_shifts,
     sensor_shape, qc) = load_scan_preserve_plane_shift(data_dir)
    z_relative_mm = np.asarray(z_relative_mm, float)
    if len(images) != len(z_relative_mm):
        raise ValueError("z_relative_mm must contain one position per measured plane")
    axes_by_z, axis_calibrated, axis_source = calibrated_axes_in_crop(
        calibration, estimated_axis, crop_origin, mean_shifts, len(images))

    # The seed defines only the common sampled radial search range.  Production
    # retrieval selects one stack-aware branch through all measured landscapes.
    seed_kp, _ = estimate_global_kr(images, pixel_pitch_m, q, .55)
    independent_retrievals = [fit_plane_adaptive(
        image, i, zmm*1e-3, axes_by_z[i], pixel_pitch_m, q, seed_kp,
        max_aberration_order=max_aberration_order)
        for i, (image, zmm) in enumerate(zip(images, z_relative_mm))]
    independent_k = np.asarray([r.k_perp_m_inv for r in independent_retrievals])

    landscapes = [compute_k_perp_cost_curve(
        image, axes_by_z[i], pixel_pitch_m, q, seed_kp, n_samples=321)
        for i, image in enumerate(images)]
    k_grid = landscapes[0][0]
    if not all(np.array_equal(k_grid, item[0]) for item in landscapes):
        raise RuntimeError("k_perp landscapes do not share an identical grid")
    cost_curves = np.stack([item[1] for item in landscapes])
    candidates = [find_k_perp_candidate_minima(k_grid, cost)
                  for cost in cost_curves]
    chosen, regularisation_trials, selection_status = choose_k_perp_regularisation(
        z_relative_mm, candidates, k_perp_seed_m_inv=seed_kp)
    discrete_k = np.asarray(chosen["k_perp_m_inv"], float)
    global_k, refinement = refine_continuous_k_perp_path(
        k_grid, cost_curves, discrete_k, k_perp_seed_m_inv=seed_kp,
        lambda_first=chosen["lambda_first"],
        lambda_second=chosen["lambda_second"])
    stability = assess_k_perp_path_stability(
        z_relative_mm, k_grid, cost_curves, global_k,
        k_perp_seed_m_inv=seed_kp,
        lambda_first=chosen["lambda_first"],
        lambda_second=chosen["lambda_second"])

    retrievals = [fit_plane_adaptive_at_k_perp(
        image, i, zmm*1e-3, axes_by_z[i], pixel_pitch_m, q, global_k[i],
        max_aberration_order=max_aberration_order)
        for i, (image, zmm) in enumerate(zip(images, z_relative_mm))]

    rows = _write_retrieval_rows(out/"per_plane_retrieval.csv", retrievals,
                                 z_relative_mm, axis_calibrated, axis_source)
    independent_rows = _write_retrieval_rows(
        out/"per_plane_independent_retrieval.csv", independent_retrievals,
        z_relative_mm, axis_calibrated, axis_source)
    _save_retrieval_coefficients(out/"global_modal_coefficients.npz", retrievals,
                                 max_aberration_order)
    np.savez_compressed(out/"k_perp_cost_landscapes.npz", z_relative_mm=z_relative_mm,
                        k_perp_grid_m_inv=k_grid, cost=cost_curves)

    candidate_rows = []
    for j, candidate in enumerate(candidates):
        for index, kp, cost, rank in zip(candidate["indices"],
                                         candidate["k_perp_m_inv"],
                                         candidate["cost"], candidate["rank"]):
            candidate_rows.append({"z_index": j, "z_relative_mm": z_relative_mm[j],
                                   "grid_index": int(index), "k_perp_m_inv": float(kp),
                                   "radial_cost": float(cost), "cost_rank": int(rank)})
    with (out/"k_perp_candidate_minima.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(candidate_rows[0]))
        writer.writeheader(); writer.writerows(candidate_rows)

    independent_radial_cost = np.asarray([
        np.interp(independent_k[j], k_grid, cost_curves[j]) for j in range(len(images))])
    global_radial_cost = np.asarray([
        np.interp(global_k[j], k_grid, cost_curves[j]) for j in range(len(images))])
    curve_scale = np.maximum(np.ptp(cost_curves, axis=1), 1e-12)
    radial_cost_increase = float(np.mean(
        (global_radial_cost-independent_radial_cost)/curve_scale))
    global_jump = np.abs(np.diff(global_k)/global_k[:-1])
    independent_jump = np.abs(np.diff(independent_k)/independent_k[:-1])
    path_rows = []
    for j in range(len(images)):
        nearest = int(np.argmin(np.abs(candidates[j]["k_perp_m_inv"]-global_k[j])))
        path_rows.append({
            "z_index": j, "z_relative_mm": float(z_relative_mm[j]),
            "independent_k_perp_m_inv": float(independent_k[j]),
            "global_k_perp_m_inv": float(global_k[j]),
            "independent_radial_cost": float(independent_radial_cost[j]),
            "global_radial_cost": float(global_radial_cost[j]),
            "fractional_change_from_previous": (np.nan if j == 0 else
                float((global_k[j]-global_k[j-1])/global_k[j-1])),
            "candidate_branch_rank": int(candidates[j]["rank"][nearest]),
            "k_perp_global_std_m_inv": float(stability["std_m_inv"][j]),
            "branch_selection_fraction": float(stability["branch_selection_fraction"][j]),
            "independent_fit_corr": float(independent_retrievals[j].fit_corr),
            "global_fit_corr": float(retrievals[j].fit_corr),
            "independent_fit_nrmse": float(independent_retrievals[j].fit_nrmse),
            "global_fit_nrmse": float(retrievals[j].fit_nrmse),
            "independent_modal_order": int(independent_retrievals[j].aberration_order_max),
            "global_modal_order": int(retrievals[j].aberration_order_max),
        })
    with (out/"k_perp_global_path.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(path_rows[0]))
        writer.writeheader(); writer.writerows(path_rows)
    with (out/"k_perp_path_stability.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=("z_index", "z_relative_mm",
            "k_perp_global_std_m_inv", "branch_selection_fraction"))
        writer.writeheader()
        for row in path_rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})
    np.savez_compressed(out/"k_perp_stability_trials.npz",
                        trial_paths_m_inv=stability["paths_m_inv"])

    mean_independent_corr = float(np.mean([r.fit_corr for r in independent_retrievals]))
    mean_global_corr = float(np.mean([r.fit_corr for r in retrievals]))
    mean_independent_nrmse = float(np.mean([r.fit_nrmse for r in independent_retrievals]))
    mean_global_nrmse = float(np.mean([r.fit_nrmse for r in retrievals]))
    independent_max_order = int(np.count_nonzero(
        [r.aberration_order_max == max_aberration_order for r in independent_retrievals]))
    global_max_order = int(np.count_nonzero(
        [r.aberration_order_max == max_aberration_order for r in retrievals]))
    path_reliability_checks = {
        "continuous_refinement_converged": bool(refinement.success),
        "max_adjacent_jump_le_8pct": bool(np.max(global_jump) <= .08),
        "normalized_radial_cost_increase_le_5pct": bool(radial_cost_increase <= .05),
        "median_branch_stability_ge_70pct": bool(
            stability["median_branch_selection_fraction"] >= .70),
        "all_planes_branch_stability_ge_50pct": bool(
            stability["min_branch_selection_fraction"] >= .50),
        "mean_modal_corr_not_degraded_gt_0p015": bool(
            mean_global_corr >= mean_independent_corr-.015),
        "mean_modal_nrmse_not_degraded_gt_0p025": bool(
            mean_global_nrmse <= mean_independent_nrmse+.025),
        "max_order_saturation_not_increased": bool(global_max_order <= independent_max_order),
    }
    k_perp_path_reliable = bool(all(path_reliability_checks.values()))
    selection_record = {
        "selection_status": selection_status,
        "lambda_first": float(chosen["lambda_first"]),
        "lambda_second": float(chosen["lambda_second"]),
        "discrete_path_diagnostics": {key: value for key, value in chosen.items()
            if key not in ("indices", "k_perp_m_inv")},
        "continuous_refinement_success": bool(refinement.success),
        "continuous_refinement_message": str(refinement.message),
        "normalized_global_vs_independent_radial_cost_increase": radial_cost_increase,
        "max_independent_adjacent_fractional_jump": float(np.max(independent_jump)),
        "max_global_adjacent_fractional_jump": float(np.max(global_jump)),
        "median_global_adjacent_fractional_jump": float(np.median(global_jump)),
        "stability": {key: value for key, value in stability.items()
                      if key not in ("paths_m_inv", "std_m_inv", "branch_selection_fraction")},
        "path_reliability_checks": path_reliability_checks,
        "k_perp_path_reliable": k_perp_path_reliable,
        "tradeoff_scan": regularisation_trials,
    }
    (out/"k_perp_regularisation_selection.json").write_text(
        json.dumps(selection_record, indent=2), encoding="utf-8")
    _plot_global_branch_diagnostic(out, z_relative_mm, k_grid, cost_curves,
                                   candidates, independent_k, global_k,
                                   stability, path_rows)

    # Always render the diagnostic model comparison, even when the calibration
    # or path-quality gates correctly prevent physical correction-map assembly.
    model_comparison = write_model_comparison(
        out/"model_comparison", images, retrievals, z_relative_mm,
        pixel_pitch_m, q)

    with (out/"frame_qc_preserved_coordinates.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(qc[0])); w.writeheader(); w.writerows(qc)

    z0 = calibration.get("z_at_relative_zero_from_axicon_mm")
    nominal_kp = calibration.get("k_perp_nominal_m_inv")
    conjugate_plane = calibration.get("slm2_is_conjugate_to_input_plane")
    base = {
        "method": "Miao-style stack-aware k_perp branch + adaptive complex Bessel modal retrieval",
        "planes": len(images), "repeats_per_plane": 4,
        "sensor_shape_yx": list(map(int, sensor_shape)),
        "programmed_q": int(q), "programmed_vortex_in_correction": False,
        "plane_to_plane_beam_recentering": False,
        "within_plane_repeat_jitter_registration": True,
        "camera_optical_axis_calibrated": axis_calibrated,
        "camera_axis_source": axis_source,
        "retrieval_axis_yx_px_by_z_in_crop": axes_by_z.tolist(),
        "global_k_perp_used_only_as_seed_m_inv": float(seed_kp),
        "k_perp_path_reliable": k_perp_path_reliable,
        "max_independent_adjacent_k_perp_fractional_jump": float(np.max(independent_jump)),
        "max_adjacent_k_perp_fractional_jump": float(np.max(global_jump)),
        "median_adjacent_k_perp_fractional_jump": float(np.median(global_jump)),
        "global_vs_independent_radial_cost_increase": radial_cost_increase,
        "mean_independent_fit_correlation": mean_independent_corr,
        "mean_global_fit_correlation": mean_global_corr,
        "mean_independent_fit_nrmse": mean_independent_nrmse,
        "mean_global_fit_nrmse": mean_global_nrmse,
        "independent_planes_at_max_modal_order": independent_max_order,
        "global_planes_at_max_modal_order": global_max_order,
        "k_perp_path_stability": stability["median_branch_selection_fraction"],
        "k_perp_path_min_branch_stability": stability["min_branch_selection_fraction"],
        "k_perp_regularisation": {"lambda_first": float(chosen["lambda_first"]),
                                  "lambda_second": float(chosen["lambda_second"]),
                                  "selection_status": selection_status},
        "nominal_k_perp_calibrated": nominal_kp is not None,
        "absolute_z_calibrated": z0 is not None,
        "slm2_conjugacy_confirmed": conjugate_plane is True,
        "hardware_ready": False,
        "model_comparison": model_comparison,
    }
    early_blockers = []
    if z0 is None:
        early_blockers.append("measure z_at_relative_zero_from_axicon_mm before radial phase assembly")
    if nominal_kp is None:
        early_blockers.append("supply the intended/calibrated k_perp_nominal_m_inv before a correction trial")
    if not axis_calibrated:
        early_blockers.append(
            "measure camera optical axis versus z-stage position; median beam core is diagnostic only")
    if not k_perp_path_reliable:
        early_blockers.append(
            "global k_perp(z) branch failed numerical fit/stability reliability gates")
    if z0 is None or not k_perp_path_reliable:
        base.update({
            "status": ("LOCAL_RETRIEVAL_ONLY" if k_perp_path_reliable
                       else "GLOBAL_K_PERP_PATH_UNRELIABLE"),
            "hardware_blockers": early_blockers + [
                "resolve the conjugate/180-degree branch with an independent reference",
                "confirm whether SLM2 is conjugate to the reconstructed input plane",
                "calibrate the applicable input-plane/SLM2 transform and 1030-nm LUT",
            ],
        })
        (out/"correction_manifest.json").write_text(json.dumps(base, indent=2), encoding="utf-8")
        return base

    z_abs_m = (float(z0)+z_relative_mm)*1e-3
    if np.any(z_abs_m <= 0):
        raise ValueError("absolute z calibration places at least one plane at z<=0")

    reference = None
    ref_path = calibration.get("input_reference_annular_intensity_npy")
    if ref_path:
        reference = np.load(ref_path)
        reference = _reference_rows_in_rho_order(reference, retrievals, z_abs_m, wavelength_m)
    full = assemble_full_aperture(retrievals, z_abs_m, wavelength_m,
                                  k_perp_nominal_m_inv=nominal_kp,
                                  reference_intensity_rows=reference)
    cart = interpolate_to_cartesian(full, grid_size=768)
    np.save(out/"retrieved_full_residual_phase_input_plane_rad.npy",
            cart["residual_phase_rad"].astype(np.float32))
    np.save(out/"conjugate_correction_input_plane_rad.npy",
            cart["conjugate_correction_phase_rad"].astype(np.float32))
    np.save(out/"rho_sampled_m.npy", full.rho_m)
    np.save(out/"radial_phase_rad.npy", full.radial_phase_rad)
    np.save(out/"radial_phase_gradient_rad_per_m.npy", full.radial_phase_gradient_rad_per_m)
    np.save(out/"angular_phase_rows_rad.npy", full.angular_phase_rows_rad.astype(np.float32))

    transform_keys = ("slm2_shape", "input_plane_m_per_slm2_pixel",
                      "slm2_center_yx_px", "slm2_rotation_deg",
                      "slm2_parity_x", "slm2_parity_y")
    transform_values_ready = all(calibration.get(k) is not None for k in transform_keys)
    geometric_mapping_ready = bool(conjugate_plane is True and transform_values_ready and axis_calibrated)
    lut_ready = bool(calibration.get("slm2_phase_lut_1030nm_calibrated", False))
    nominal_ready = nominal_kp is not None

    slm_written = False
    if full.branch != "unresolved" and geometric_mapping_ready and nominal_ready:
        slm_map = map_input_phase_to_slm2(
            cart, tuple(calibration["slm2_shape"]),
            float(calibration["input_plane_m_per_slm2_pixel"]),
            tuple(calibration["slm2_center_yx_px"]),
            float(calibration["slm2_rotation_deg"]),
            int(calibration["slm2_parity_x"]), int(calibration["slm2_parity_y"]))
        np.save(out/"slm2_correction_phase_rad.npy", slm_map.astype(np.float32))
        slm_written = True

    mapping_for_manifest = bool(geometric_mapping_ready and nominal_ready)
    manifest = correction_manifest(full, True, mapping_for_manifest, lut_ready, False)
    extra_pretrial = []
    if not nominal_ready:
        extra_pretrial.append("intended/calibrated nominal k_perp is missing")
    if not axis_calibrated:
        extra_pretrial.append("camera optical axis/stage runout is not independently calibrated")
    if conjugate_plane is not True:
        if conjugate_plane is False:
            extra_pretrial.append(
                "SLM2 is not conjugate to the reconstructed input plane; implement measured relay complex-field back-propagation before mapping to SLM2")
        else:
            extra_pretrial.append("SLM2/input-plane conjugacy has not been established")
    for item in extra_pretrial:
        if item not in manifest["pretrial_blockers"]:
            manifest["pretrial_blockers"].append(item)
        if item not in manifest["hardware_blockers"]:
            manifest["hardware_blockers"].insert(0, item)

    manifest.update(base)
    manifest.update({
        "status": "FULL_RETRIEVAL_COMPLETE",
        "branch": full.branch,
        "branch_score_direct": full.branch_score_direct,
        "branch_score_conjugate": full.branch_score_conjugate,
        "absolute_z_at_relative_zero_from_axicon_mm": float(z0),
        "k_perp_nominal_m_inv": None if nominal_kp is None else float(nominal_kp),
        "slm2_is_conjugate_to_input_plane": conjugate_plane,
        "input_plane_to_slm2_geometric_mapping_calibrated": geometric_mapping_ready,
        "slm2_phase_lut_1030nm_calibrated": lut_ready,
        "slm2_phase_map_written": slm_written,
        "slm2_shape_yx": (list(map(int, calibration["slm2_shape"]))
                          if slm_written else None),
        "nonconjugate_relay_backpropagation_implemented": False,
    })
    manifest["application_ready_for_low_gain_trial"] = len(manifest["pretrial_blockers"]) == 0
    manifest["hardware_ready"] = len(manifest["hardware_blockers"]) == 0
    (out/"correction_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    import argparse

    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=here/"z-scan 2 1010")
    parser.add_argument("--output-dir", type=Path, default=here/"outputs"/"miao_full_q20")
    parser.add_argument("--calibration-json", type=Path)
    args = parser.parse_args()
    cal = args.calibration_json
    if cal is None and (here/"q20_hardware_calibration.json").exists():
        cal = here/"q20_hardware_calibration.json"
    result = run(args.data_dir, args.output_dir,
                 calibration_json=cal)
    print(json.dumps(result, indent=2))
