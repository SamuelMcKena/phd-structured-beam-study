import numpy as np
import pytest
from scipy import special

from vbb_study.correction.miao_forward_model import (
    MiaoSyntheticAberration, apply_phase_error, classify_error,
    primary_edge_mode_scope, published_synthetic_terms,
)
from vbb_study.correction.miao_diagnostics import (
    first_bright_ring_radius, hollow_core_metrics, oracle_relative_field_error,
)
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
    assert m1["centre_intensity_over_peak"] > m0["centre_intensity_over_peak"] + 0.02
    assert m1["inner_core_mean_over_peak"] > m0["inner_core_mean_over_peak"]


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


def test_component_plane_legacy_fill_factor_route_is_restricted():
    row = module_policy("vbb_study.digital_twin.component_plane_pipeline")
    assert row.status == "compatibility_restricted"
    assert not row.correction_evidence_allowed
