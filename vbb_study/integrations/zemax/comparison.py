"""Numerical comparison metrics with explicit transverse/ZBF governance."""
from __future__ import annotations

import numpy as np

from .alignment import align_field_planes
from .models import CrossValidationResult, FieldPlane

EPS = np.finfo(float).tiny


def _normalise_energy(intensity: np.ndarray) -> np.ndarray:
    arr = np.maximum(np.asarray(intensity, dtype=float), 0.0)
    return arr / max(float(np.sum(arr)), EPS)


def normalized_intensity_correlation(first: np.ndarray, second: np.ndarray) -> float:
    a = _normalise_energy(first).ravel()
    b = _normalise_energy(second).ravel()
    return float(np.vdot(a, b).real / max(float(np.linalg.norm(a) * np.linalg.norm(b)), EPS))


def normalized_intensity_l2_error(first: np.ndarray, second: np.ndarray) -> float:
    a = _normalise_energy(first)
    b = _normalise_energy(second)
    return float(np.linalg.norm(a - b) / max(float(np.linalg.norm(a)), EPS))


def centroid(intensity: np.ndarray, x_m: np.ndarray, y_m: np.ndarray) -> tuple[float, float]:
    values = np.maximum(np.asarray(intensity, dtype=float), 0.0)
    total = max(float(np.sum(values)), EPS)
    xx, yy = np.meshgrid(x_m, y_m, indexing="xy")
    return float(np.sum(values * xx) / total), float(np.sum(values * yy) / total)


def complex_vector_overlap(ex_a: np.ndarray, ey_a: np.ndarray, ex_b: np.ndarray, ey_b: np.ndarray) -> float:
    numerator = np.sum(np.conj(ex_a) * ex_b + np.conj(ey_a) * ey_b)
    norm_a = np.sum(np.abs(ex_a) ** 2 + np.abs(ey_a) ** 2)
    norm_b = np.sum(np.abs(ex_b) ** 2 + np.abs(ey_b) ** 2)
    return float(np.abs(numerator) ** 2 / max(float(norm_a * norm_b), EPS))


def phase_rms_after_piston(first_phase: np.ndarray, second_phase: np.ndarray) -> float:
    delta = np.angle(np.exp(1j * (np.asarray(second_phase) - np.asarray(first_phase))))
    piston = np.angle(np.mean(np.exp(1j * delta)))
    residual = np.angle(np.exp(1j * (delta - piston)))
    return float(np.sqrt(np.mean(residual**2)))


def radial_profile(intensity: np.ndarray, x_m: np.ndarray, y_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.maximum(np.asarray(intensity, dtype=float), 0.0)
    xx, yy = np.meshgrid(x_m, y_m, indexing="xy")
    radius = np.hypot(xx, yy)
    spacing = min(float(np.median(np.diff(x_m))), float(np.median(np.diff(y_m))))
    edges = np.arange(0.0, float(radius.max()) + spacing, spacing)
    if edges.size < 4:
        edges = np.linspace(0.0, float(radius.max()), 16)
    idx = np.clip(np.digitize(radius.ravel(), edges) - 1, 0, edges.size - 2)
    sums = np.bincount(idx, weights=values.ravel(), minlength=edges.size - 1)
    counts = np.bincount(idx, minlength=edges.size - 1)
    profile = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    return 0.5 * (edges[:-1] + edges[1:]), profile


def ring_radius(intensity: np.ndarray, x_m: np.ndarray, y_m: np.ndarray) -> float:
    radius, profile = radial_profile(intensity, x_m, y_m)
    if radius.size == 0:
        return float("nan")
    spacing = min(float(np.median(np.diff(x_m))), float(np.median(np.diff(y_m))))
    eligible = radius >= 2.0 * abs(spacing)
    if not np.any(eligible):
        return float(radius[int(np.argmax(profile))])
    indices = np.flatnonzero(eligible)
    return float(radius[indices[int(np.argmax(profile[eligible]))]])


def _central_null_metric(intensity: np.ndarray) -> float:
    arr = np.asarray(intensity, dtype=float)
    cy, cx = arr.shape[0] // 2, arr.shape[1] // 2
    return float(arr[cy, cx] / max(float(np.max(arr)), EPS))


def compare_field_planes(python_plane: FieldPlane, zemax_plane: FieldPlane, *, case_id: str = "G0", comparison_grid: str = "python", provenance: dict[str, object] | None = None) -> CrossValidationResult:
    py, ze, alignment_meta = align_field_planes(python_plane, zemax_plane, comparison_grid=comparison_grid)  # type: ignore[arg-type]
    py_i = py.transverse_intensity
    ze_i = ze.transverse_intensity
    py_centroid = centroid(py_i, py.x_m, py.y_m)
    ze_centroid = centroid(ze_i, ze.x_m, ze.y_m)
    metrics: dict[str, object] = {"intensity_scope": "transverse", "normalized_intensity_correlation": normalized_intensity_correlation(py_i, ze_i), "normalized_intensity_l2_error": normalized_intensity_l2_error(py_i, ze_i), "python_centroid_x_m": py_centroid[0], "python_centroid_y_m": py_centroid[1], "zemax_centroid_x_m": ze_centroid[0], "zemax_centroid_y_m": ze_centroid[1], "centroid_x_error_m": ze_centroid[0] - py_centroid[0], "centroid_y_error_m": ze_centroid[1] - py_centroid[1], "python_peak_position_index": [int(v) for v in np.unravel_index(np.argmax(py_i), py_i.shape)], "zemax_peak_position_index": [int(v) for v in np.unravel_index(np.argmax(ze_i), ze_i.shape)], "relative_discrete_power_error": float((np.sum(ze_i) - np.sum(py_i)) / max(float(np.sum(py_i)), EPS)), **alignment_meta}
    if case_id.upper() in {"B0", "V1", "V3"}:
        metrics["python_ring_radius_m"] = ring_radius(py_i, py.x_m, py.y_m)
        metrics["zemax_ring_radius_m"] = ring_radius(ze_i, ze.x_m, ze.y_m)
    if case_id.upper() in {"V1", "V3"}:
        metrics["python_dark_core_null_metric"] = _central_null_metric(py_i)
        metrics["zemax_dark_core_null_metric"] = _central_null_metric(ze_i)
        metrics["topological_charge_claim"] = "not established by irradiance morphology alone"
    if all(value is not None for value in (py.Ex, py.Ey, ze.Ex, ze.Ey)):
        metrics["complex_transverse_field_overlap"] = complex_vector_overlap(py.Ex, py.Ey, ze.Ex, ze.Ey)  # type: ignore[arg-type]
    if py.phase is not None and ze.phase is not None:
        metrics["phase_rms_after_piston_rad"] = phase_rms_after_piston(py.phase, ze.phase)
    flags: dict[str, bool | str] = {"metrics_computed": True, "longitudinal_component_validated": False, "experimental_validation": False, "registered_comparison": False}
    return CrossValidationResult(python_plane=python_plane, zemax_plane=zemax_plane, aligned_python_plane=py, aligned_zemax_plane=ze, metrics=metrics, pass_fail_flags=flags, provenance=dict(provenance or {}), warnings=["ZBF/POP transverse comparison does not validate Ez."] if python_plane.Ez is not None else [])
