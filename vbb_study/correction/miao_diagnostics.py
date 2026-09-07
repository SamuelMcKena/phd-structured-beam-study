"""Core-sensitive validation for high-order Bessel/vortex correction.

Global image NRMSE can hide weak light inside a nominally hollow core. These
metrics therefore evaluate the centre and an inner-core disk separately from the
bright annulus. The natural radius scale is the first positive maximum of
J_q(k_perp r)^2, i.e. the first positive zero of J_q'(x).
"""
from __future__ import annotations

import numpy as np
from scipy import special

EPS = 1e-15


def first_bright_ring_radius(q: int, k_perp: float) -> float:
    q = abs(int(q)); kp = abs(float(k_perp))
    if q == 0:
        return 0.0
    if kp <= 0.0:
        raise ValueError("k_perp must be positive")
    return float(special.jnp_zeros(q, 1)[0] / kp)


def _core_mask(x: np.ndarray, y: np.ndarray, *, q: int, k_perp: float,
               inner_fraction: float) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    x = np.asarray(x, float); y = np.asarray(y, float)
    if not (0.0 < float(inner_fraction) < 1.0):
        raise ValueError("inner_fraction must lie between 0 and 1")
    X, Y = np.meshgrid(x, y, indexing="xy")
    ring_r = first_bright_ring_radius(q, k_perp)
    if ring_r <= 0.0:
        raise ValueError("hollow-core metric requires q != 0")
    radius = float(inner_fraction) * ring_r
    return np.hypot(X, Y) <= radius, radius, X, Y


def hollow_core_metrics(
    intensity: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    q: int,
    k_perp: float,
    inner_fraction: float = 0.65,
) -> dict[str, float]:
    """Measure intensity inside a conservative fraction of the first bright ring.

    Values are normalized to the global peak intensity. For q>0 the analytic
    ideal Bessel mode satisfies J_q(0)=0; the disk is not expected to be exactly
    zero because J_q rises continuously toward its first bright annulus.
    """
    I = np.asarray(intensity, float)
    x = np.asarray(x, float); y = np.asarray(y, float)
    if I.shape != (y.size, x.size):
        raise ValueError("intensity shape must be (len(y), len(x))")
    core, radius, _, _ = _core_mask(
        x, y, q=q, k_perp=k_perp, inner_fraction=inner_fraction
    )
    peak = max(float(np.max(I)), EPS)
    iy = int(np.argmin(np.abs(y))); ix = int(np.argmin(np.abs(x)))
    return {
        "first_bright_ring_radius": first_bright_ring_radius(q, k_perp),
        "inner_core_radius": radius,
        "centre_intensity_over_peak": float(I[iy, ix] / peak),
        "inner_core_mean_over_peak": float(np.mean(I[core]) / peak),
        "inner_core_max_over_peak": float(np.max(I[core]) / peak),
    }


def hollow_core_residual_metrics(
    reference_intensity: np.ndarray,
    candidate_intensity: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    q: int,
    k_perp: float,
    inner_fraction: float = 0.65,
) -> dict[str, float]:
    """Compare candidate and ideal/reference intensities inside the hollow core.

    Residuals are normalized to the *reference global peak*, so an apparently
    clean candidate cannot hide error by renormalising its own peak. This is the
    appropriate regression metric for faint rings superimposed on an ideal
    high-order Bessel core.
    """
    ref = np.asarray(reference_intensity, float)
    cand = np.asarray(candidate_intensity, float)
    x = np.asarray(x, float); y = np.asarray(y, float)
    expected_shape = (y.size, x.size)
    if ref.shape != expected_shape or cand.shape != expected_shape:
        raise ValueError("reference and candidate intensities must match the x/y grid")
    core, radius, _, _ = _core_mask(
        x, y, q=q, k_perp=k_perp, inner_fraction=inner_fraction
    )
    peak = max(float(np.max(ref)), EPS)
    residual = cand - ref
    abs_residual = np.abs(residual[core])
    return {
        "inner_core_radius": radius,
        "inner_core_mean_abs_residual_over_reference_peak": float(np.mean(abs_residual) / peak),
        "inner_core_rms_residual_over_reference_peak": float(np.sqrt(np.mean(residual[core] ** 2)) / peak),
        "inner_core_max_abs_residual_over_reference_peak": float(np.max(abs_residual) / peak),
    }


def oracle_phase_correction(field: np.ndarray, aberration_phase_rad: np.ndarray) -> np.ndarray:
    """Exact conjugate-phase oracle used to test the forward-model contract."""
    E = np.asarray(field, complex); phi = np.asarray(aberration_phase_rad, float)
    if E.shape != phi.shape:
        raise ValueError("field and phase must have the same shape")
    return E * np.exp(1j * phi) * np.exp(-1j * phi)


def oracle_relative_field_error(field: np.ndarray, aberration_phase_rad: np.ndarray) -> float:
    E = np.asarray(field, complex)
    recovered = oracle_phase_correction(E, aberration_phase_rad)
    return float(np.linalg.norm(recovered - E) / max(np.linalg.norm(E), EPS))


__all__ = [
    "first_bright_ring_radius", "hollow_core_metrics", "hollow_core_residual_metrics",
    "oracle_phase_correction", "oracle_relative_field_error",
]
