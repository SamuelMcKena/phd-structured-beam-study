"""Core-sensitive validation for high-order Bessel/vortex correction.

Global image NRMSE can hide weak light inside a nominally hollow core.  These
metrics therefore evaluate the centre and an inner-core disk separately from the
bright annulus.  The natural radius scale is the first positive maximum of
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


def hollow_core_metrics(
    intensity: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    q: int,
    k_perp: float,
    inner_fraction: float = 0.65,
) -> dict[str, float]:
    """Measure leakage inside a conservative fraction of the first bright ring.

    Values are normalized to the global peak intensity.  For q>0 the analytic
    ideal Bessel mode satisfies J_q(0)=0; the disk is not expected to be exactly
    zero because J_q rises continuously toward its first bright annulus.
    """
    I = np.asarray(intensity, float)
    x = np.asarray(x, float); y = np.asarray(y, float)
    if I.shape != (y.size, x.size):
        raise ValueError("intensity shape must be (len(y), len(x))")
    if not (0.0 < float(inner_fraction) < 1.0):
        raise ValueError("inner_fraction must lie between 0 and 1")
    peak = max(float(np.max(I)), EPS)
    X, Y = np.meshgrid(x, y, indexing="xy")
    r = np.hypot(X, Y)
    ring_r = first_bright_ring_radius(q, k_perp)
    if ring_r <= 0.0:
        raise ValueError("hollow-core metric requires q != 0")
    core = r <= float(inner_fraction) * ring_r
    iy = int(np.argmin(np.abs(y))); ix = int(np.argmin(np.abs(x)))
    return {
        "first_bright_ring_radius": ring_r,
        "inner_core_radius": float(inner_fraction) * ring_r,
        "centre_intensity_over_peak": float(I[iy, ix] / peak),
        "inner_core_mean_over_peak": float(np.mean(I[core]) / peak),
        "inner_core_max_over_peak": float(np.max(I[core]) / peak),
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


__all__ = ["first_bright_ring_radius", "hollow_core_metrics",
           "oracle_phase_correction", "oracle_relative_field_error"]
