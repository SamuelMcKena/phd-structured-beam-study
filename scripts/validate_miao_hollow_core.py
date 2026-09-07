"""Synthetic q=20 regression for faint hollow-core residuals.

This is a matched-model numerical validation, not a laboratory validation.  It
uses the published Miao synthetic angular aberration at one representative
annulus, retrieves complex Bessel-mode coefficients from intensity, resolves the
intensity-only direct/conjugate branch using the known synthetic truth, applies
the retrieved conjugate phase, and checks the corrected core against the ideal
J_20 field.

The known truth is used only because this is a regression test. Hardware use
still requires independent branch resolution and the calibration gates in
``miao_retrieval.correction_manifest``.
"""
from __future__ import annotations

import json

import numpy as np

from vbb_study.correction.miao_diagnostics import (
    hollow_core_metrics,
    hollow_core_residual_metrics,
)
from vbb_study.correction.miao_forward_model import (
    MiaoSyntheticAberration,
    miao_modal_basis,
    sampling_window_report,
    synthesize_miao_focal_field,
)
from vbb_study.correction.miao_retrieval import (
    angular_field_from_coefficients,
    fit_coefficients,
)

TWOPI = 2.0 * np.pi
EPS = 1e-15


def _circular_phase_rms(candidate: np.ndarray, truth: np.ndarray) -> float:
    delta = np.angle(np.exp(1j * (np.asarray(candidate) - np.asarray(truth))))
    piston = np.angle(np.mean(np.exp(1j * delta)))
    delta = np.angle(np.exp(1j * (delta - piston)))
    return float(np.sqrt(np.mean(delta * delta)))


def _align_piston(candidate: np.ndarray, truth: np.ndarray) -> np.ndarray:
    delta = np.angle(np.mean(np.exp(1j * (np.asarray(candidate) - np.asarray(truth)))))
    return np.angle(np.exp(1j * (np.asarray(candidate) - delta)))


def _periodic_phase_interpolator(theta: np.ndarray, phase: np.ndarray):
    theta = np.asarray(theta, float)
    unit = np.exp(1j * np.asarray(phase, float))
    xp = np.concatenate([theta, [TWOPI]])
    re = np.concatenate([unit.real, [unit.real[0]]])
    im = np.concatenate([unit.imag, [unit.imag[0]]])

    def evaluate(query: np.ndarray) -> np.ndarray:
        q = np.mod(np.asarray(query, float), TWOPI)
        zr = np.interp(q, xp, re)
        zi = np.interp(q, xp, im)
        return np.angle(zr + 1j * zi)

    return evaluate


def _global_nrmse(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = np.asarray(reference, float)
    cand = np.asarray(candidate, float)
    ref = ref / max(float(np.max(ref)), EPS)
    cand = cand / max(float(np.max(cand)), EPS)
    return float(
        np.sqrt(np.mean((cand - ref) ** 2))
        / max(float(np.sqrt(np.mean(ref ** 2))), EPS)
    )


def run_validation() -> dict[str, object]:
    q = 20
    k_perp = 1.2e6  # m^-1; synthetic numerical scale, not a bench calibration
    x = np.linspace(-40e-6, 40e-6, 121)
    y = x.copy()
    pixel_pitch = float(x[1] - x[0])
    model = MiaoSyntheticAberration(scale=1.0)
    rho_hat = 0.82

    sampling = sampling_window_report(x, y, q=q, k_perp=k_perp)
    if not sampling["adequate"]:
        raise AssertionError(f"q20 validation field of view is inadequate: {sampling}")

    def true_phase(theta: np.ndarray) -> np.ndarray:
        return model.angular_phase(theta, rho_hat=rho_hat)

    ideal_field = synthesize_miao_focal_field(
        q=q, k_perp=k_perp, x=x, y=y, phase_function=None,
        m_max=40, n_theta=4096,
    )
    aberrated_field = synthesize_miao_focal_field(
        q=q, k_perp=k_perp, x=x, y=y, phase_function=true_phase,
        m_max=40, n_theta=4096,
    )
    ideal_intensity = np.abs(ideal_field) ** 2
    aberrated_intensity = np.abs(aberrated_field) ** 2

    X, Y = np.meshgrid(x, y, indexing="xy")
    R = np.hypot(X, Y)
    PHI = np.arctan2(Y, X)
    fit_mask = (R >= 1.5e-6) & (R <= 38e-6)
    measured = aberrated_intensity[fit_mask]
    r = R[fit_mask]
    phi = PHI[fit_mask]
    weights = r / max(float(np.max(r)), EPS)
    weights *= 0.25 + 0.75 * np.sqrt(
        np.clip(measured / max(float(np.max(measured)), EPS), 0.0, 1.0)
    )

    m_values = np.arange(-30, 31, dtype=int)
    B = miao_modal_basis(q, m_values, k_perp, r, phi)
    coeffs, fit_cost = fit_coefficients(
        B, measured, weights, m_values, maxiter=500, reg=0.0
    )

    theta = np.linspace(0.0, TWOPI, 720, endpoint=False)
    truth = true_phase(theta)
    retrieved_field = angular_field_from_coefficients(coeffs, m_values, theta)
    direct = np.angle(retrieved_field)
    conjugate = -np.roll(direct, direct.size // 2)
    rms_direct = _circular_phase_rms(direct, truth)
    rms_conjugate = _circular_phase_rms(conjugate, truth)
    branch = "direct" if rms_direct <= rms_conjugate else "conjugate"
    retrieved_phase = direct if branch == "direct" else conjugate
    retrieved_phase = _align_piston(retrieved_phase, truth)
    phase_rms = _circular_phase_rms(retrieved_phase, truth)
    retrieved_phase_at = _periodic_phase_interpolator(theta, retrieved_phase)

    def residual_phase(query: np.ndarray) -> np.ndarray:
        return np.angle(
            np.exp(1j * (true_phase(query) - retrieved_phase_at(query)))
        )

    corrected_field = synthesize_miao_focal_field(
        q=q, k_perp=k_perp, x=x, y=y, phase_function=residual_phase,
        m_max=40, n_theta=4096,
    )
    corrected_intensity = np.abs(corrected_field) ** 2

    ideal_core = hollow_core_metrics(
        ideal_intensity, x, y, q=q, k_perp=k_perp, inner_fraction=0.65
    )
    corrected_core = hollow_core_metrics(
        corrected_intensity, x, y, q=q, k_perp=k_perp, inner_fraction=0.65
    )
    core_residual = hollow_core_residual_metrics(
        ideal_intensity, corrected_intensity, x, y,
        q=q, k_perp=k_perp, inner_fraction=0.65,
    )
    metrics = {
        "status": "synthetic_matched_model_known_truth_branch_selection",
        "q": q,
        "k_perp_m_inv": k_perp,
        "rho_hat": rho_hat,
        "grid_n": int(x.size),
        "pixel_pitch_m": pixel_pitch,
        "sampling": sampling,
        "fit_cost": float(fit_cost),
        "retrieval_branch_selected_from_known_truth": branch,
        "retrieved_phase_rms_rad": float(phase_rms),
        "ideal_to_aberrated_global_nrmse": _global_nrmse(ideal_intensity, aberrated_intensity),
        "ideal_to_corrected_global_nrmse": _global_nrmse(ideal_intensity, corrected_intensity),
        "ideal_core": ideal_core,
        "corrected_core": corrected_core,
        "core_residual": core_residual,
        "acceptance": {
            "retrieved_phase_rms_rad_max": 5e-3,
            "ideal_to_corrected_global_nrmse_max": 1e-2,
            "centre_intensity_over_peak_max": 1e-5,
            "inner_core_max_abs_residual_over_reference_peak_max": 1e-4,
        },
    }

    acceptance = metrics["acceptance"]
    failures: list[str] = []
    if phase_rms > acceptance["retrieved_phase_rms_rad_max"]:
        failures.append("retrieved phase RMS exceeds acceptance")
    if metrics["ideal_to_corrected_global_nrmse"] > acceptance["ideal_to_corrected_global_nrmse_max"]:
        failures.append("corrected global NRMSE exceeds acceptance")
    if corrected_core["centre_intensity_over_peak"] > acceptance["centre_intensity_over_peak_max"]:
        failures.append("corrected centre is too bright")
    if (
        core_residual["inner_core_max_abs_residual_over_reference_peak"]
        > acceptance["inner_core_max_abs_residual_over_reference_peak_max"]
    ):
        failures.append("corrected inner-core residual exceeds acceptance")
    metrics["pass"] = not failures
    metrics["failures"] = failures
    return metrics


def main() -> None:
    metrics = run_validation()
    print(json.dumps(metrics, indent=2))
    if not metrics["pass"]:
        raise SystemExit("Miao q20 hollow-core regression failed: " + "; ".join(metrics["failures"]))


if __name__ == "__main__":
    main()
