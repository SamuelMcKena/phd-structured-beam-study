import inspect

import numpy as np
import pytest
from scipy import special

from vbb_study.correction.miao_forward_model import (
    MiaoSyntheticAberration, apply_phase_error, classify_error,
    primary_edge_mode_scope, published_synthetic_terms,
)
from vbb_study.correction.miao_diagnostics import (
    first_bright_ring_radius, hollow_core_metrics, hollow_core_residual_metrics,
    oracle_relative_field_error,
)
from vbb_study.correction.miao_retrieval import fit_coefficients, fit_plane_adaptive
from vbb_study.digital_twin.phase2a_contracts import error_injection_registry_rows
from vbb_study.digital_twin.vortex_wavefront_errors import ZERNIKE_NAMES, unit_rms_zernike
from vbb_study.digital_twin.legacy_error_policy import (
    assert_correction_evidence_allowed, module_policy,
)


def test_miao_published_synthetic_includes_spherical():
    assert published_synthetic_terms() == ("astigmatism", "trefoil", "quadrafoil", "spherical")
    rho = np.array([0.0])
    th = np.array([0.0])
    assert np.allclose(MiaoSyntheticAberration().phase(rho, th), np.sqrt(5.0))


def test_phase_only_operator_preserves_amplitude():
    rng = np.random.default_rng(4)
    field = rng.normal(size=(17, 13)) + 1j * rng.normal(size=(17, 13))
    phase = rng.normal(size=field.shape)
    assert np.allclose(np.abs(apply_phase_error(field, phase)), np.abs(field))


def test_miao_classification_does_not_promise_success():
    assert classify_error(changes_complex_phase=True) == "miao_phase_retrieval_candidate"
    assert classify_error(changes_complex_phase=True, changes_amplitude=True) == "amplitude_or_clipping"
    assert classify_error(changes_complex_phase=True, applied_after_intensity=True) == "detector_or_post_intensity"


def test_primary_m_gt_1_scope_is_experiment_specific_helper():
    assert primary_edge_mode_scope([2, -3, 4])
    assert not primary_edge_mode_scope([0, 2])
    assert not primary_edge_mode_scope([1, 3])


def test_oracle_phase_correction_is_exact_to_roundoff():
    rng = np.random.default_rng(5)
    field = rng.normal(size=(31, 29)) + 1j * rng.normal(size=(31, 29))
    phase = rng.normal(size=field.shape)
    assert oracle_relative_field_error(field, phase) < 2e-15


def test_first_bright_ring_uses_derivative_zero():
    q = 20; kp = 1.2
    expected = special.jnp_zeros(q, 1)[0] / kp
    assert np.isclose(first_bright_ring_radius(q, kp), expected)


def test_hollow_core_metric_detects_added_centre_light():
    q = 8; kp = 1.1
    x = np.linspace(-20.0, 20.0, 401); y = x.copy()
    X, Y = np.meshgrid(x, y, indexing="xy")
    R = np.hypot(X, Y)
    ideal = special.jv(q, kp * R) ** 2
    bright = ideal + 0.03 * np.exp(-(R / 2.0) ** 2)
    m0 = hollow_core_metrics(ideal, x, y, q=q, k_perp=kp)
    m1 = hollow_core_metrics(bright, x, y, q=q, k_perp=kp)
    residual = hollow_core_residual_metrics(ideal, bright, x, y, q=q, k_perp=kp)
    assert m1["centre_intensity_over_peak"] > m0["centre_intensity_over_peak"] + 0.02
    assert m1["inner_core_mean_over_peak"] > m0["inner_core_mean_over_peak"]
    assert residual["inner_core_max_abs_residual_over_reference_peak"] > 0.02


def test_default_miao_modal_fit_has_no_silent_regularisation():
    fit_sig = inspect.signature(fit_coefficients)
    plane_sig = inspect.signature(fit_plane_adaptive)
    assert fit_sig.parameters["reg"].default == 0.0
    assert plane_sig.parameters["coefficient_regularization"].default == 0.0


def test_generic_zernike_basis_now_contains_miao_azimuthal_orders():
    for name in ("trefoil_x", "trefoil_y", "quadrafoil_x", "quadrafoil_y"):
        assert name in ZERNIKE_NAMES
    x = np.linspace(-1.0, 1.0, 101)
    X, Y = np.meshgrid(x, x, indexing="xy")
    grid = {"X": X, "Y": Y}
    z = unit_rms_zernike("quadrafoil_x", grid, pupil_radius_m=1.0)
    mask = np.hypot(X, Y) <= 1.0
    assert np.isclose(np.sqrt(np.mean(z[mask] ** 2)), 1.0, atol=2e-12)


def test_post_engine_diagnostic_is_blocked_from_correction_evidence():
    row = module_policy("vbb_study.digital_twin.lab_perturbations")
    assert row.status == "diagnostic_only"
    with pytest.raises(RuntimeError):
        assert_correction_evidence_allowed("vbb_study.digital_twin.lab_perturbations")


def test_canonical_system_route_is_allowed():
    assert_correction_evidence_allowed("vbb_study.digital_twin.vortex_system_route")


def test_legacy_physical_error_api_is_wrapper_not_evidence_source():
    row = module_policy("vbb_study.digital_twin.vortex_physical_errors")
    assert row.status == "compatibility_restricted"
    assert row.physical_claim_allowed
    assert not row.correction_evidence_allowed
    with pytest.raises(RuntimeError):
        assert_correction_evidence_allowed("vbb_study.digital_twin.vortex_physical_errors")


def test_component_plane_legacy_fill_factor_route_is_restricted():
    row = module_policy("vbb_study.digital_twin.component_plane_pipeline")
    assert row.status == "compatibility_restricted"
    assert not row.correction_evidence_allowed


def test_phase2a_registry_no_longer_models_rigid_axicon_tilt_as_linear_ramp():
    rows = {row["error_id"]: row for row in error_injection_registry_rows()}
    tilt = rows["axicon_tilt"]
    assert "vortex_rotated_plane" in tilt["module"]
    assert "rotate" in tilt["mathematical_operator"].lower()
    assert "linear phase ramp" not in tilt["mathematical_operator"].lower()
    assert tilt["implementation_status"] == "calibration_limited"


def test_phase2a_registry_keeps_sample_tilt_out_of_post_engine_diagnostics():
    rows = {row["error_id"]: row for row in error_injection_registry_rows()}
    sample = rows["sample_interface_tilt"]
    assert "lab_perturbations" not in sample["module"]
    assert "vector" in sample["supported_routes"].lower()
