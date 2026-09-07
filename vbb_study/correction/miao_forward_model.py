"""Forward models and provenance rules for Miao-style Bessel-beam validation.

The synthetic benchmark follows Miao et al., Optics Express 30, 11360-11371
(2022), doi:10.1364/OE.454796.  Their Eq. (7) validation phase contains
astigmatism, trefoil, quadrafoil *and spherical aberration*.  Do not delete the
spherical term from anything labelled the published synthetic benchmark.

Important scope distinction
---------------------------
The retrieval formalism reconstructs radial and non-cylindrically-symmetric
phase terms.  For the particular reflective axicons used experimentally, Miao
et al. found that the dominant surface error near the aperture edge could be
described by primary rho**m cos(m theta)/sin(m theta), m>1 terms.  That
experiment-specific simplification must not be mislabelled as a universal
restriction of the retrieval method.

Detector overlays, post-intensity edits, amplitude clipping/vignetting and
incoherent backgrounds remain separate error classes and cannot be presented as
phase-only correction targets.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Sequence

import numpy as np
from scipy import special

EPS = 1e-12
TWOPI = 2.0 * np.pi

ErrorClass = Literal[
    "miao_phase_retrieval_candidate",
    "coherent_parasitic_field",
    "amplitude_or_clipping",
    "detector_or_post_intensity",
]

# Backward-compatible name retained, but the value is deliberately more precise.
MIAO_CORRECTABLE_ERROR_CLASSES: tuple[str, ...] = ("miao_phase_retrieval_candidate",)
NON_MIAO_ERROR_CLASSES: tuple[str, ...] = (
    "coherent_parasitic_field",
    "amplitude_or_clipping",
    "detector_or_post_intensity",
)


@dataclass(frozen=True)
class MiaoSyntheticAberration:
    """Published synthetic phase mixture from Miao et al. Eq. (7), scaled as a whole."""

    scale: float = 1.0
    astigmatism_rad: float = 5.0
    trefoil_rad: float = 3.0
    quadrafoil_rad: float = 2.0
    spherical_rad: float = float(np.sqrt(5.0))

    def phase(self, rho_hat: np.ndarray, theta_rad: np.ndarray) -> np.ndarray:
        rho = np.asarray(rho_hat, dtype=float)
        theta = np.asarray(theta_rad, dtype=float)
        return float(self.scale) * (
            float(self.astigmatism_rad) * rho**2 * np.cos(2.0 * theta + 2.0 * np.pi / 3.0)
            + float(self.trefoil_rad) * rho**3 * np.sin(3.0 * theta - np.pi / 6.0)
            + float(self.quadrafoil_rad) * rho**4 * np.cos(4.0 * theta - np.pi / 4.0)
            + float(self.spherical_rad) * (6.0 * rho**4 - 6.0 * rho**2 + 1.0)
        )

    def angular_phase(self, theta_rad: np.ndarray, *, rho_hat: float = 0.82) -> np.ndarray:
        theta = np.asarray(theta_rad, dtype=float)
        rho = np.full_like(theta, float(rho_hat), dtype=float)
        phase = self.phase(rho, theta)
        unit = np.exp(1j * phase)
        return np.angle(unit * np.exp(-1j * np.angle(np.mean(unit))))


def published_synthetic_terms() -> tuple[str, ...]:
    """Return the four aberration families explicitly present in Miao Eq. (7)."""
    return ("astigmatism", "trefoil", "quadrafoil", "spherical")


def primary_edge_mode_scope(m_values: Sequence[int]) -> bool:
    """Whether harmonics match the experiment-specific primary m>1 simplification.

    This is *not* a universal correctability test.  It only records the primary
    azimuthal family used to describe the particular reflective-axicon surface
    errors discussed by Miao et al.
    """
    m = np.asarray(tuple(m_values), dtype=int)
    return bool(m.size and np.all(np.abs(m) > 1))


def apply_phase_error(field: np.ndarray, phase_rad: np.ndarray) -> np.ndarray:
    """Apply a phase-only aberration without changing field amplitude."""
    e = np.asarray(field, dtype=np.complex128)
    phase = np.asarray(phase_rad, dtype=float)
    if e.shape != phase.shape:
        raise ValueError("field and phase_rad must have the same shape")
    return e * np.exp(1j * phase)


def phase_on_normalised_pupil(
    x: np.ndarray,
    y: np.ndarray,
    pupil_radius: float,
    aberration: MiaoSyntheticAberration | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if float(pupil_radius) <= 0:
        raise ValueError("pupil_radius must be positive")
    X, Y = np.meshgrid(np.asarray(x, float), np.asarray(y, float), indexing="xy")
    rho = np.hypot(X, Y) / float(pupil_radius)
    theta = np.arctan2(Y, X)
    support = rho <= 1.0
    model = aberration or MiaoSyntheticAberration()
    phase = np.zeros_like(rho)
    phase[support] = model.phase(rho[support], theta[support])
    return phase, support


def fourier_coefficients_from_annular_phase(
    phase_rad: np.ndarray,
    theta_rad: np.ndarray,
    m_values: np.ndarray,
) -> np.ndarray:
    phase = np.asarray(phase_rad, float)
    theta = np.asarray(theta_rad, float)
    if phase.shape != theta.shape:
        raise ValueError("phase_rad and theta_rad must have the same shape")
    f = np.exp(1j * phase)
    return np.asarray(
        [np.mean(f * np.exp(1j * int(m) * theta)) for m in np.asarray(m_values, int)],
        dtype=np.complex128,
    )


def miao_modal_basis(
    q: int,
    m_values: np.ndarray,
    k_perp: float,
    r: np.ndarray,
    phi: np.ndarray,
) -> np.ndarray:
    r = np.asarray(r, float)
    phi = np.asarray(phi, float)
    if r.shape != phi.shape:
        raise ValueError("r and phi must have the same shape")
    cols = []
    for m in np.asarray(m_values, int):
        n = int(m) - int(q)
        cols.append(((-1j) ** n) * special.jv(n, float(k_perp) * r) * np.exp(-1j * n * phi))
    return np.column_stack(cols)


def synthesize_miao_focal_field(
    *, q: int, k_perp: float, x: np.ndarray, y: np.ndarray,
    phase_function: Callable[[np.ndarray], np.ndarray] | None = None,
    m_max: int = 36, n_theta: int = 4096,
) -> np.ndarray:
    """Synthesize a focal Bessel field from a coherent annular phase error."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    X, Y = np.meshgrid(x, y, indexing="xy")
    r = np.hypot(X, Y); phi = np.arctan2(Y, X)
    m_values = np.arange(-int(m_max), int(m_max) + 1, dtype=int)
    theta = np.linspace(0.0, TWOPI, int(n_theta), endpoint=False)
    phase = np.zeros_like(theta) if phase_function is None else np.asarray(phase_function(theta), float)
    coeffs = fourier_coefficients_from_annular_phase(phase, theta, m_values)
    basis = miao_modal_basis(q, m_values, k_perp, r.ravel(), phi.ravel())
    return (basis @ coeffs).reshape(r.shape)


def add_detector_ring_contamination(
    intensity: np.ndarray, x: np.ndarray, y: np.ndarray, *,
    period: float = 3.6, envelope_radius: float = 16.0, amplitude: float = 0.28,
) -> np.ndarray:
    """Unsupported post-intensity ring overlay for a deliberately negative control."""
    I = np.asarray(intensity, float)
    X, Y = np.meshgrid(np.asarray(x, float), np.asarray(y, float), indexing="xy")
    r = np.hypot(X, Y)
    rings = float(amplitude) * (0.5 + 0.5 * np.cos(TWOPI * r / float(period)))
    rings *= np.exp(-(r / float(envelope_radius)) ** 2)
    return I + rings


def recommended_radial_extent(q: int, k_perp: float, *, margin_orders: float = 8.0) -> float:
    """Conservative half-width guard for observing a high-order Bessel mode."""
    kp = abs(float(k_perp))
    if kp <= 0:
        raise ValueError("k_perp must be non-zero")
    return (abs(int(q)) + float(margin_orders)) / kp


def sampling_window_report(x: np.ndarray, y: np.ndarray, *, q: int, k_perp: float,
                           margin_orders: float = 8.0) -> dict:
    x = np.asarray(x, float); y = np.asarray(y, float)
    available = min(float(np.max(np.abs(x))), float(np.max(np.abs(y))))
    recommended = recommended_radial_extent(q, k_perp, margin_orders=margin_orders)
    return {
        "available_half_width": available,
        "recommended_half_width": recommended,
        "adequate": bool(available >= recommended),
        "q": int(q), "k_perp": float(k_perp),
        "criterion": "half-width >= (|q| + margin_orders)/|k_perp|",
    }


def classify_error(*, changes_complex_phase: bool, changes_amplitude: bool = False,
                   coherent_extra_field: bool = False, applied_after_intensity: bool = False) -> ErrorClass:
    """Classify by physical operator; do not infer success from the word 'phase'."""
    if applied_after_intensity:
        return "detector_or_post_intensity"
    if coherent_extra_field:
        return "coherent_parasitic_field"
    if changes_amplitude:
        return "amplitude_or_clipping"
    if changes_complex_phase:
        return "miao_phase_retrieval_candidate"
    raise ValueError("error description does not identify a supported class")


__all__ = [
    "ErrorClass", "MIAO_CORRECTABLE_ERROR_CLASSES", "NON_MIAO_ERROR_CLASSES",
    "MiaoSyntheticAberration", "published_synthetic_terms", "primary_edge_mode_scope",
    "apply_phase_error", "phase_on_normalised_pupil", "fourier_coefficients_from_annular_phase",
    "miao_modal_basis", "synthesize_miao_focal_field", "add_detector_ring_contamination",
    "recommended_radial_extent", "sampling_window_report", "classify_error",
]
