from __future__ import annotations

"""Multi-plane experiment-to-simulation matching.

Ranking is performed on regenerated complex-field simulations, never rendered
atlas images.  The default objective fits one non-negative *global* radiometric
gain over the entire z stack and then computes a global energy-normalised RMS
residual.  There is no per-plane gain, recentering or coordinate warping.

The current matcher is intentionally a single-mechanism grid screen.  A low
score identifies a useful physical hypothesis, not a unique laboratory truth;
combined-error optimisation belongs to a later identifiability stage.
"""

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence
import math

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from vbb_study.digital_twin.phase2a_contracts import canonical_hardware_manifest, hardware_value
from vbb_study.digital_twin.vortex_system_route import build_system_route
from vbb_study.equations.fields import make_xy_grid
from vbb_study.equations.propagation import angular_spectrum_propagate_bl
from .candidate_models import (
    DEFAULT_MATCH_FAMILIES,
    FAMILY_REGISTRY,
    build_candidate_route_arguments,
    family_spec,
)
from .experimental_stack import IntensityStack


EPS = np.finfo(float).tiny
DEFAULT_WINDOW_M = 10.0e-3


@dataclass(frozen=True)
class CandidateScore:
    family: str
    value: float
    score: float
    global_gain: float
    global_correlation: float
    valid_fraction: float
    mean_plane_nrmse: float
    max_plane_nrmse: float
    mean_shape_nrmse: float
    per_plane: tuple[Mapping[str, float], ...]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class MatchResult:
    case_id: str
    scores: tuple[CandidateScore, ...]
    objective: str
    model_z_offset_m: float
    metadata: Mapping[str, Any]

    @property
    def best(self) -> CandidateScore:
        if not self.scores:
            raise RuntimeError("match result contains no candidate scores")
        return self.scores[0]


def simulate_candidate_stack(
    *,
    case_id: str,
    family: str,
    value: float,
    z_m: Sequence[float],
    grid_n: int,
    window_m: float = DEFAULT_WINDOW_M,
) -> IntensityStack:
    """Regenerate one physical candidate through the canonical optical route."""
    z = np.asarray(z_m, dtype=float)
    if z.ndim != 1 or z.size < 1 or not np.all(np.isfinite(z)):
        raise ValueError("z_m must be a finite one-dimensional sequence")
    if np.any(z < 0.0):
        raise ValueError("model propagation distances from the post-axicon plane cannot be negative")
    if int(grid_n) < 32:
        raise ValueError("grid_n must be >=32")
    if float(window_m) <= 0.0:
        raise ValueError("window_m must be positive")

    manifest = canonical_hardware_manifest()
    wavelength = float(hardware_value(manifest, "wavelength_m"))
    gamma = math.radians(float(hardware_value(manifest, "axicon_base_angle_deg")))
    map_grid = make_xy_grid(int(grid_n), float(window_m) / int(grid_n))
    config, maps = build_candidate_route_arguments(
        family,
        float(value),
        grid=map_grid,
        wavelength_m=wavelength,
        axicon_base_angle_rad=gamma,
    )
    route = build_system_route(
        case_id,
        grid_n=int(grid_n),
        window_m=float(window_m),
        config=config,
        **maps,
    )
    grid = dict(route["grid"])
    source = np.asarray(route["post_axicon"], dtype=np.complex128)
    planes = []
    for zz in z:
        if float(zz) == 0.0:
            field = source
        else:
            field = angular_spectrum_propagate_bl(
                source,
                grid,
                wavelength,
                float(zz),
                bandlimit=True,
            )
        planes.append(np.abs(field) ** 2)
    x = np.asarray(grid["x"], dtype=float)
    return IntensityStack(
        z_m=z,
        x_m=x,
        y_m=x.copy(),
        intensity=np.stack(planes, axis=0),
        metadata={
            "source": "canonical_physical_route_simulation",
            "case_id": str(case_id),
            "family": str(family),
            "value": float(value),
            "grid_n": int(grid_n),
            "window_m": float(window_m),
            "wavelength_m": wavelength,
            "route_id": route["metadata"]["route_id"],
            "candidate_provenance": family_spec(family).provenance,
        },
    )


def _measurement_mask(stack: IntensityStack) -> np.ndarray:
    values = np.asarray(stack.intensity, dtype=float)
    mask = np.isfinite(values)
    if stack.valid_mask is not None:
        supplied = np.asarray(stack.valid_mask, dtype=bool)
        if supplied.ndim == 2:
            supplied = np.broadcast_to(supplied[None, :, :], values.shape)
        mask &= supplied
    return mask


def resample_simulation_to_measurement(
    simulation: IntensityStack,
    measurement: IntensityStack,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate simulated intensities onto measured x/y coordinates.

    Only intensity is interpolated at this comparison boundary.  The simulation
    itself is propagated on its native complex-field grid.  Points outside the
    simulated physical window become invalid rather than being extrapolated.
    """
    if simulation.z_m.shape != measurement.z_m.shape or not np.allclose(
        simulation.z_m, measurement.z_m, rtol=0.0, atol=1e-12
    ):
        raise ValueError("simulation and measurement z coordinates must already match")
    Y, X = np.meshgrid(measurement.y_m, measurement.x_m, indexing="ij")
    query = np.column_stack([Y.ravel(), X.ravel()])
    result = np.empty_like(measurement.intensity, dtype=float)
    valid = np.ones_like(result, dtype=bool)
    for iz, plane in enumerate(np.asarray(simulation.intensity, dtype=float)):
        interp = RegularGridInterpolator(
            (np.asarray(simulation.y_m, float), np.asarray(simulation.x_m, float)),
            plane,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        sampled = interp(query).reshape(X.shape)
        result[iz] = sampled
        valid[iz] = np.isfinite(sampled)
    return result, valid


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    av = np.asarray(a, dtype=float).ravel()
    bv = np.asarray(b, dtype=float).ravel()
    if av.size < 2 or float(np.std(av)) <= EPS or float(np.std(bv)) <= EPS:
        return float("nan")
    return float(np.corrcoef(av, bv)[0, 1])


def score_simulation_stack(
    measurement: IntensityStack,
    simulation: IntensityStack,
    *,
    maximum_invalid_fraction: float = 0.02,
) -> tuple[float, float, float, float, float, float, tuple[Mapping[str, float], ...]]:
    """Fit one global gain and score a matched z stack.

    Returns ``score, gain, correlation, valid_fraction, mean_plane_nrmse,
    max_plane_nrmse, per_plane``.  The rank score is
    ``sqrt(sum((E-gS)^2)/sum(E^2))`` over every valid pixel in every plane.
    """
    sampled, sim_valid = resample_simulation_to_measurement(simulation, measurement)
    exp = np.asarray(measurement.intensity, dtype=float)
    mask = _measurement_mask(measurement) & sim_valid
    valid_fraction = float(np.count_nonzero(mask) / mask.size)
    if 1.0 - valid_fraction > float(maximum_invalid_fraction):
        raise RuntimeError(
            "measured field of view is not adequately covered by simulation: "
            f"invalid fraction={1.0-valid_fraction:.6g} > {float(maximum_invalid_fraction):.6g}"
        )
    e = exp[mask]
    s = sampled[mask]
    denominator = float(np.dot(s, s))
    gain = max(0.0, float(np.dot(e, s) / max(denominator, EPS)))
    prediction = gain * sampled
    residual = exp - prediction
    score = float(np.sqrt(np.sum(residual[mask] ** 2) / max(float(np.sum(e**2)), EPS)))
    corr = _correlation(e, prediction[mask])

    per_plane: list[Mapping[str, float]] = []
    plane_nrmse: list[float] = []
    shape_nrmse: list[float] = []
    for iz, zz in enumerate(measurement.z_m):
        m = mask[iz]
        ee = exp[iz][m]
        pp = prediction[iz][m]
        rr = ee - pp
        nrmse = float(np.sqrt(np.sum(rr**2) / max(float(np.sum(ee**2)), EPS)))
        plane_nrmse.append(nrmse)
        e_peak = max(float(np.max(ee)), EPS)
        p_peak = max(float(np.max(pp)), EPS)
        shape = float(
            np.sqrt(np.mean((ee / e_peak - pp / p_peak) ** 2))
        )
        shape_nrmse.append(shape)
        per_plane.append({
            "z_m": float(zz),
            "nrmse": nrmse,
            "correlation": _correlation(ee, pp),
            "shape_nrmse_own_peak_diagnostic": shape,
            "measured_power_sum": float(np.sum(ee)),
            "predicted_power_sum_after_global_gain": float(np.sum(pp)),
            "valid_fraction": float(np.count_nonzero(m) / m.size),
        })
    return (
        score,
        gain,
        corr,
        valid_fraction,
        float(np.mean(plane_nrmse)),
        float(np.max(plane_nrmse)),
        tuple(per_plane),
    )


def _resolve_model_z(measurement: IntensityStack, model_z_offset_m: float | None) -> tuple[np.ndarray, float]:
    reference = str(measurement.metadata.get("z_reference", ""))
    if model_z_offset_m is None:
        if reference != "post_axicon_plane":
            raise ValueError(
                "measurement z_reference is not 'post_axicon_plane'; provide explicit model_z_offset_m "
                "instead of silently aligning the z stack"
            )
        offset = 0.0
    else:
        offset = float(model_z_offset_m)
    model_z = np.asarray(measurement.z_m, dtype=float) + offset
    if np.any(model_z < 0.0):
        raise ValueError("model_z_offset_m places at least one plane before the post-axicon source plane")
    return model_z, offset


def match_error_candidates(
    measurement: IntensityStack,
    *,
    case_id: str,
    grid_n: int,
    families: Sequence[str] = DEFAULT_MATCH_FAMILIES,
    candidate_values: Mapping[str, Sequence[float]] | None = None,
    model_z_offset_m: float | None = None,
    window_m: float = DEFAULT_WINDOW_M,
    maximum_invalid_fraction: float = 0.02,
) -> MatchResult:
    """Rank single physical error hypotheses against the entire measured stack."""
    model_z, resolved_offset = _resolve_model_z(measurement, model_z_offset_m)
    values_override = dict(candidate_values or {})
    scores: list[CandidateScore] = []
    for family in families:
        spec = family_spec(family)
        values: Iterable[float] = values_override.get(spec.key, spec.default_values)
        values = tuple(float(v) for v in values)
        if not values:
            raise ValueError(f"candidate family {spec.key!r} has no values")
        for value in values:
            simulation = simulate_candidate_stack(
                case_id=case_id,
                family=spec.key,
                value=value,
                z_m=model_z,
                grid_n=int(grid_n),
                window_m=float(window_m),
            )
            # Coordinate comparison uses the model z distances, but the measured
            # stack retains its laboratory z labels.  Give the resampler a view
            # with the same numerical z coordinates only; no plane reordering or
            # interpolation in z is performed.
            measurement_for_score = IntensityStack(
                z_m=model_z,
                x_m=measurement.x_m,
                y_m=measurement.y_m,
                intensity=measurement.intensity,
                valid_mask=measurement.valid_mask,
                metadata=measurement.metadata,
            )
            score, gain, corr, valid_fraction, mean_nrmse, max_nrmse, per_plane_model = (
                score_simulation_stack(
                    measurement_for_score,
                    simulation,
                    maximum_invalid_fraction=float(maximum_invalid_fraction),
                )
            )
            # Report the original measured z labels alongside model distances.
            per_plane = tuple(
                {
                    **dict(record),
                    "measurement_z_m": float(measurement.z_m[i]),
                    "model_z_m": float(model_z[i]),
                }
                for i, record in enumerate(per_plane_model)
            )
            scores.append(CandidateScore(
                family=spec.key,
                value=value,
                score=score,
                global_gain=gain,
                global_correlation=corr,
                valid_fraction=valid_fraction,
                mean_plane_nrmse=mean_nrmse,
                max_plane_nrmse=max_nrmse,
                mean_shape_nrmse=float(np.mean([float(p["shape_nrmse_own_peak_diagnostic"]) for p in per_plane])),
                per_plane=per_plane,
                metadata={
                    "title": spec.title,
                    "parameter_unit": spec.parameter_unit,
                    "operator_class": spec.operator_class,
                    "physical_plane": spec.physical_plane,
                    "candidate_provenance": spec.provenance,
                    "miao_handoff_policy": spec.miao_handoff_policy,
                },
            ))
    scores.sort(key=lambda item: item.score)
    return MatchResult(
        case_id=str(case_id),
        scores=tuple(scores),
        objective="global_nonnegative_gain_then_energy_normalised_stack_RMS",
        model_z_offset_m=resolved_offset,
        metadata={
            "inference_scope": "single_mechanism_grid_screening_not_unique_inverse_solution",
            "coordinate_policy": "fixed_measured_lab_coordinates_no_hidden_xy_recentering",
            "radiometry_policy": "one_nonnegative_gain_for_entire_z_stack_no_per_plane_gain",
            "z_policy": "no_z_interpolation; explicit_model_z_offset_only",
            "simulation_source": "canonical_complex_field_physical_route_not_rendered_atlas",
            "grid_n": int(grid_n),
            "window_m": float(window_m),
            "maximum_invalid_fraction": float(maximum_invalid_fraction),
            "families": [str(f) for f in families],
        },
    )


__all__ = [
    "CandidateScore",
    "MatchResult",
    "simulate_candidate_stack",
    "resample_simulation_to_measurement",
    "score_simulation_stack",
    "match_error_candidates",
]
