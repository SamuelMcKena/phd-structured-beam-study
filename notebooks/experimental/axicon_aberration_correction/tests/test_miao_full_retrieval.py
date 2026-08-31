from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import pytest
from scipy import special

ROOT = Path(__file__).resolve().parents[1]
MOD = ROOT
if not (MOD / "miao_full_retrieval.py").is_file():
    MOD = ROOT / "notebooks" / "experimental" / "axicon_aberration_correction"
sys.path.insert(0, str(MOD))

from miao_full_retrieval import (
    PlaneRetrieval, angular_field_from_coefficients, optimise_k_perp_ideal_mode,
    assemble_full_aperture, interpolate_to_cartesian, resolve_conjugate_branch,
    correction_manifest, find_k_perp_candidate_minima,
    select_continuous_k_perp_path, refine_continuous_k_perp_path,
    assess_k_perp_path_stability,
)
from iterative_correction_controller_v2 import _candidate_mask, evaluate_experimental_update
from q20_experimental_acceptance_metrics import compute_plane_metrics
from run_q20_miao_retrieval import calibrated_axes_in_crop


def fake_plane(i, z, kp, theta, phase):
    return PlaneRetrieval(i, z, 64.0, 64.0, kp, 4, 0.01, 0.99, 0.02,
                          np.arange(-4, 5), np.zeros(9, complex), theta,
                          np.exp(1j*phase))


def test_per_plane_kperp_optimizer_recovers_ideal_ring():
    q = 3
    pixel = 5.5e-6
    kp_true = 4.7e5
    n = 128
    yy, xx = np.indices((n, n), dtype=float)
    r = np.hypot(xx-(n-1)/2, yy-(n-1)/2)*pixel
    image = special.jv(q, kp_true*r)**2
    kp = optimise_k_perp_ideal_mode(image, ((n-1)/2, (n-1)/2), pixel, q,
                                    kp_true*1.06, search_fraction=0.15,
                                    rmax_um=220, n_r=36, n_theta=72)
    assert abs(kp-kp_true)/kp_true < 0.02


def test_stationary_phase_radius_and_radial_gradient_are_recovered():
    lam = 1030e-9
    k = 2*np.pi/lam
    nominal = 4.9e5
    z = np.linspace(0.020, 0.035, 6)
    theta = np.linspace(0, 2*np.pi, 720, endpoint=False)
    kp = nominal + np.asarray([0, -800, -1200, -400, 500, 900], float)
    planes = [fake_plane(i, z[i], kp[i], theta, 0.2*np.cos(2*theta))
              for i in range(len(z))]
    full = assemble_full_aperture(planes, z, lam, k_perp_nominal_m_inv=nominal)
    order = np.argsort(z*kp/k)
    assert np.allclose(full.rho_m, (z*kp/k)[order])
    assert np.allclose(full.radial_phase_gradient_rad_per_m, (nominal-kp)[order])
    assert np.isfinite(full.radial_phase_rad).all()


def test_global_branch_rejects_isolated_deeper_false_minimum():
    z = np.arange(9, dtype=float)
    seed = 4.9e5
    smooth = seed*(1 + .012*np.sin(np.linspace(0, 1.4*np.pi, len(z))))
    candidate_k, candidate_cost = [], []
    for j, expected in enumerate(smooth):
        false = expected*(1.16 if j == 4 else 1.002)
        candidate_k.append(np.asarray([expected, false]))
        candidate_cost.append(np.asarray([.02, 0.0 if j == 4 else .55]))
    independent = np.asarray([candidate_k[j][np.argmin(candidate_cost[j])]
                              for j in range(len(z))])
    indices, _ = select_continuous_k_perp_path(
        z, candidate_k, candidate_cost, k_perp_seed_m_inv=seed,
        lambda_first=30, lambda_second=100)
    selected = np.asarray([candidate_k[j][indices[j]] for j in range(len(z))])
    assert np.max(np.abs(np.diff(independent))/independent[:-1]) > .12
    assert np.allclose(selected, smooth)


def test_global_branch_preserves_smooth_nonmonotonic_path():
    z = np.arange(11, dtype=float)
    seed = 4.8e5
    target = seed*(1 + .025*np.sin(np.linspace(0, 2*np.pi, len(z))))
    candidate_k = [np.asarray([value, seed*1.10]) for value in target]
    candidate_cost = [np.asarray([.01, .20]) for _ in target]
    indices, _ = select_continuous_k_perp_path(
        z, candidate_k, candidate_cost, k_perp_seed_m_inv=seed,
        lambda_first=10, lambda_second=30)
    selected = np.asarray([candidate_k[j][indices[j]] for j in range(len(z))])
    assert np.allclose(selected, target)
    assert np.ptp(selected) > .04*seed
    assert np.any(np.diff(selected) > 0) and np.any(np.diff(selected) < 0)


def test_continuous_refinement_respects_sampled_bounds_and_curvature():
    z = np.arange(8, dtype=float)
    seed = 4.9e5
    grid = np.linspace(.85*seed, 1.15*seed, 151)
    target = seed*(1 + .02*np.sin(np.linspace(0, 1.5*np.pi, len(z))))
    curves = np.stack([((grid-value)/(.02*seed))**2 for value in target])
    initial = target + np.asarray([0, 500, -700, 400, -300, 600, -400, 0])
    refined, result = refine_continuous_k_perp_path(
        grid, curves, initial, k_perp_seed_m_inv=seed,
        lambda_first=3, lambda_second=10)
    assert result.success
    assert np.all(refined >= grid[0]) and np.all(refined <= grid[-1])
    assert np.ptp(refined) > .02*seed
    assert np.sqrt(np.mean(np.diff(np.log(refined), n=2)**2)) < .03


def test_candidate_detection_and_stability_outputs_are_bounded():
    z = np.arange(6, dtype=float)
    seed = 5e5
    grid = np.linspace(.9*seed, 1.1*seed, 121)
    target = seed*(1+.01*np.sin(np.linspace(0, np.pi, len(z))))
    curves = np.stack([np.minimum(((grid-value)/5000)**2,
                                  .03+((grid-1.07*seed)/5000)**2)
                       for value in target])
    candidates = [find_k_perp_candidate_minima(grid, row) for row in curves]
    indices, _ = select_continuous_k_perp_path(
        z, [c["k_perp_m_inv"] for c in candidates], [c["cost"] for c in candidates],
        k_perp_seed_m_inv=seed, lambda_first=10, lambda_second=30)
    selected = np.asarray([candidates[j]["k_perp_m_inv"][indices[j]]
                           for j in range(len(z))])
    stability = assess_k_perp_path_stability(
        z, grid, curves, selected, k_perp_seed_m_inv=seed,
        lambda_first=10, lambda_second=30, n_trials=9, random_seed=4)
    assert stability["paths_m_inv"].shape == (9, len(z))
    assert np.all((stability["branch_selection_fraction"] >= 0) &
                  (stability["branch_selection_fraction"] <= 1))


def test_programmed_vortex_is_not_reinserted_in_angular_residual():
    theta = np.linspace(0, 2*np.pi, 720, endpoint=False)
    m = np.asarray([-2, -1, 0, 1, 2])
    c = np.asarray([0, 0.1j, 1.0, 0.2, 0], complex)
    g1 = angular_field_from_coefficients(c, m, theta)
    g2 = angular_field_from_coefficients(c, m, theta)
    assert np.allclose(g1, g2)
    assert np.max(np.abs(np.unwrap(np.angle(g1)))) < 2*np.pi


def test_conjugate_branch_resolution_uses_independent_input_intensity():
    theta = np.linspace(0, 2*np.pi, 720, endpoint=False)
    amp = 1.0 + 0.35*np.cos(theta) + 0.12*np.sin(2*theta)
    g = np.stack([amp*np.exp(1j*0.2*np.cos(3*theta)) for _ in range(4)])
    branch, sd, sc = resolve_conjugate_branch(g, np.abs(g)**2)
    assert branch == "direct"
    assert sd > sc
    rotated = np.roll(np.abs(g)**2, g.shape[1]//2, axis=1)
    branch2, sd2, sc2 = resolve_conjugate_branch(g, rotated)
    assert branch2 == "conjugate"
    assert sc2 > sd2


def test_unresolved_branch_blocks_low_gain_trial():
    lam = 1030e-9
    theta = np.linspace(0, 2*np.pi, 720, endpoint=False)
    z = np.asarray([0.02, 0.025, 0.03])
    planes = [fake_plane(i, z[i], 4.9e5, theta, 0.1*np.cos(2*theta)) for i in range(3)]
    full = assemble_full_aperture(planes, z, lam)
    manifest = correction_manifest(full, True, True, True, False)
    assert full.branch == "unresolved"
    assert not manifest["application_ready_for_low_gain_trial"]
    assert not manifest["hardware_ready"]


def test_wrapped_phase_interpolation_is_finite_inside_sampled_annulus():
    lam = 1030e-9
    theta = np.linspace(0, 2*np.pi, 720, endpoint=False)
    z = np.asarray([0.02, 0.025, 0.03])
    phases = [3.05*np.cos(theta), -3.05*np.cos(theta), 3.0*np.sin(2*theta)]
    planes = [fake_plane(i, z[i], 4.9e5, theta, phases[i]) for i in range(3)]
    full = assemble_full_aperture(planes, z, lam)
    cart = interpolate_to_cartesian(full, grid_size=96)
    assert np.isfinite(cart["residual_phase_rad"][cart["valid"]]).all()


def test_native_slm_candidate_keeps_shape_and_signed_phase():
    residual = np.full((37, 53), np.nan)
    residual[5:30, 7:45] = 0.8
    accepted = np.zeros_like(residual)
    accepted[~np.isfinite(residual)] = np.nan
    candidate = _candidate_mask(residual, accepted, 0.05)
    assert candidate.shape == residual.shape
    assert np.isnan(candidate[0, 0])
    assert np.allclose(candidate[5:30, 7:45], 0.04)


def test_per_z_camera_axis_calibration_tracks_stage_runout():
    n = 4
    cal = {"camera_optical_axis_yx_px_by_z":
           [[100.0, 200.0], [100.4, 199.8], [100.9, 199.5], [101.2, 199.3]]}
    shifts = np.asarray([[0.1, -0.2], [0.0, 0.1], [-0.1, 0.0], [0.2, 0.2]])
    axes, ready, source = calibrated_axes_in_crop(
        cal, (10.0, 20.0), (50, 150), shifts, n)
    expected = np.asarray(cal["camera_optical_axis_yx_px_by_z"]) + shifts - np.asarray([50, 150])
    assert ready
    assert "per-z" in source
    assert np.allclose(axes, expected)


def test_independent_acceptance_metric_is_near_ideal_for_synthetic_bessel():
    q = 5
    kp = 4.4e5
    pixel = 2.0e-6
    n = 192
    cy = cx = (n-1)/2
    yy, xx = np.indices((n, n), dtype=float)
    r = np.hypot(yy-cy, xx-cx)*pixel
    image = special.jv(q, kp*r)**2
    result = compute_plane_metrics(
        image, (cy, cx), pixel_pitch_m=pixel, q=q,
        k_perp_nominal_m_inv=kp, roi_radius_um=150)
    assert result["measured_vs_ideal_corr"] > 0.999999
    assert result["measured_vs_ideal_rmse"] < 1e-8
    assert result["measured_ring_cv"] < 0.01
    assert result["measured_dark_core_ratio"] < 0.2


def _acceptance_frame(digest, corr=0.5, rmse=0.3, ring_cv=0.4, dark=0.1):
    return pd.DataFrame({
        "z_mm": [-1.0, 0.0],
        "measured_vs_ideal_corr": [corr, corr],
        "measured_vs_ideal_rmse": [rmse, rmse],
        "measured_ring_cv": [ring_cv, ring_cv],
        "measured_dark_core_ratio": [dark, dark],
        "dataset_sha256": [digest, digest],
        "q_target": [20, 20],
        "k_perp_nominal_m_inv": [4.9e5, 4.9e5],
        "wavelength_m": [1030e-9, 1030e-9],
        "pixel_pitch_m": [5.5e-6, 5.5e-6],
        "roi_radius_um": [160.0, 160.0],
        "camera_axis_source": ["per-z measured reference axis"]*2,
    })


def test_experimental_acceptance_rejects_same_dataset_hash(tmp_path):
    before = tmp_path/"before.csv"
    after = tmp_path/"after.csv"
    _acceptance_frame("same").to_csv(before, index=False)
    _acceptance_frame("same", corr=0.6).to_csv(after, index=False)
    state = tmp_path/"closed_loop_state.json"
    state.write_text(json.dumps({"status": "AWAITING_EXPERIMENTAL_MEASUREMENT",
                                 "iteration": 0, "candidate_phase_path": "unused.npy"}),
                     encoding="utf-8")
    with pytest.raises(ValueError, match="same BMG dataset"):
        evaluate_experimental_update(before, after, state)


def test_experimental_acceptance_checks_matching_target_provenance(tmp_path):
    before = tmp_path/"before.csv"
    after = tmp_path/"after.csv"
    a = _acceptance_frame("before")
    b = _acceptance_frame("after", corr=0.4)  # worsening => no candidate-file access
    b["k_perp_nominal_m_inv"] = 5.0e5
    a.to_csv(before, index=False)
    b.to_csv(after, index=False)
    state = tmp_path/"closed_loop_state.json"
    state.write_text(json.dumps({"status": "AWAITING_EXPERIMENTAL_MEASUREMENT",
                                 "iteration": 0, "candidate_phase_path": "unused.npy"}),
                     encoding="utf-8")
    with pytest.raises(ValueError, match="k_perp_nominal_m_inv differs"):
        evaluate_experimental_update(before, after, state)


def test_legacy_normalized_z_map_and_second_remap_are_not_consumed_by_v3_controller():
    text = (MOD/"iterative_correction_controller_v2.py").read_text(encoding="utf-8")
    assert "UNCALIBRATED_DO_NOT_APPLY_q20_modal_correction.npy" not in text
    assert "slm2_correction_phase_rad.npy" in text
    assert "build_slm2_complete_preview" not in text
    assert "second_coordinate_mapping_applied" in text


def test_runner_blocks_geometric_slm_mapping_without_conjugacy_and_axis_calibration():
    text = (MOD/"run_q20_miao_retrieval.py").read_text(encoding="utf-8")
    assert "slm2_is_conjugate_to_input_plane" in text
    assert "camera_optical_axis_yx_px_by_z" in text
    assert "nonconjugate_relay_backpropagation_implemented" in text
    assert "k_perp_nominal_m_inv" in text
    assert '"all_planes_branch_stability_ge_50pct"' in text
    assert "if z0 is None or not k_perp_path_reliable" in text


def test_controller_acceptance_contract_matches_metric_generator():
    controller = (MOD/"iterative_correction_controller_v2.py").read_text(encoding="utf-8")
    metrics = (MOD/"q20_experimental_acceptance_metrics.py").read_text(encoding="utf-8")
    for name in ("measured_vs_ideal_corr", "measured_vs_ideal_rmse",
                 "measured_ring_cv", "measured_dark_core_ratio",
                 "dataset_sha256", "q_target", "k_perp_nominal_m_inv"):
        assert name in controller
        assert name in metrics
