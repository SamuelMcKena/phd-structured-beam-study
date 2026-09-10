"""Full-field uncertainty propagation for calibrated optical simulations.

Unlike ``calibration.uncertainty``, which propagates supplied uncertainty
through reduced analytical metrics, this module reruns a user-supplied optical
solver for each sampled parameter set and accumulates uncertainty on the
physical intensity field itself.

The simulation callback must return either a 2-D intensity array or an object
with ``intensity``, ``x_m`` and ``y_m`` attributes.  Every sample must remain on
the same physical grid; hidden re-registration is forbidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np


@dataclass(frozen=True)
class ParameterDistribution:
    name: str
    nominal: float
    standard_uncertainty: float = 0.0
    distribution: str = "normal"
    lower_bound: float | None = None
    upper_bound: float | None = None

    def validate(self) -> None:
        if not np.isfinite(self.nominal):
            raise ValueError(f"{self.name}: nominal must be finite")
        if not np.isfinite(self.standard_uncertainty) or self.standard_uncertainty < 0.0:
            raise ValueError(f"{self.name}: standard_uncertainty must be finite and non-negative")
        if self.distribution not in {"normal", "uniform", "fixed"}:
            raise ValueError(f"{self.name}: unsupported distribution {self.distribution!r}")
        if self.lower_bound is not None and self.upper_bound is not None and self.lower_bound >= self.upper_bound:
            raise ValueError(f"{self.name}: lower_bound must be below upper_bound")

    def draw(self, rng: np.random.Generator, size: int) -> np.ndarray:
        self.validate()
        if self.distribution == "fixed" or self.standard_uncertainty == 0.0:
            values = np.full(int(size), float(self.nominal), dtype=float)
        elif self.distribution == "normal":
            values = rng.normal(float(self.nominal), float(self.standard_uncertainty), size=int(size))
        else:
            # For a uniform distribution, the supplied quantity is interpreted
            # as a standard uncertainty: half-width = sqrt(3) * u.
            half_width = np.sqrt(3.0) * float(self.standard_uncertainty)
            values = rng.uniform(float(self.nominal) - half_width, float(self.nominal) + half_width, size=int(size))
        if self.lower_bound is not None:
            values = np.maximum(values, float(self.lower_bound))
        if self.upper_bound is not None:
            values = np.minimum(values, float(self.upper_bound))
        return np.asarray(values, dtype=float)


@dataclass(frozen=True)
class FullFieldUncertaintyConfig:
    samples: int = 200
    random_seed: int = 12345
    confidence_level: float = 0.95
    keep_sample_fields: bool = False

    def validate(self) -> None:
        if int(self.samples) < 2:
            raise ValueError("full-field uncertainty requires at least two samples")
        if not 0.0 < float(self.confidence_level) < 1.0:
            raise ValueError("confidence_level must lie strictly between zero and one")


@dataclass(frozen=True)
class FullFieldUncertaintyResult:
    nominal_parameters: Mapping[str, float]
    mean_intensity: np.ndarray
    standard_deviation_intensity: np.ndarray
    lower_intensity: np.ndarray
    upper_intensity: np.ndarray
    relative_standard_deviation: np.ndarray
    x_m: np.ndarray | None
    y_m: np.ndarray | None
    metric_samples: Mapping[str, np.ndarray]
    metric_summary: Mapping[str, Mapping[str, float]]
    failed_samples: int
    sample_parameter_table: Mapping[str, np.ndarray]
    sample_fields: np.ndarray | None
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _extract_intensity_and_axes(result: Any) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    if isinstance(result, np.ndarray):
        intensity = np.asarray(result, dtype=float)
        x = y = None
    else:
        if not hasattr(result, "intensity"):
            raise TypeError("simulation callback must return ndarray or object with .intensity")
        intensity = np.asarray(result.intensity, dtype=float)
        x = np.asarray(result.x_m, dtype=float) if hasattr(result, "x_m") else None
        y = np.asarray(result.y_m, dtype=float) if hasattr(result, "y_m") else None
    if intensity.ndim != 2 or intensity.size == 0 or np.any(~np.isfinite(intensity)) or np.any(intensity < 0.0):
        raise ValueError("simulation intensity must be a finite non-negative 2-D array")
    if (x is None) != (y is None):
        raise ValueError("simulation must provide both x_m and y_m or neither")
    if x is not None:
        if intensity.shape != (y.size, x.size):
            raise ValueError("simulation field does not match its physical axes")
        if np.any(np.diff(x) <= 0.0) or np.any(np.diff(y) <= 0.0):
            raise ValueError("simulation axes must be strictly increasing")
    return intensity, x, y


def _same_axes(reference: np.ndarray | None, candidate: np.ndarray | None) -> bool:
    if reference is None or candidate is None:
        return reference is None and candidate is None
    return reference.shape == candidate.shape and bool(np.allclose(reference, candidate, rtol=0.0, atol=1e-15))


def propagate_full_field_uncertainty(
    parameters: tuple[ParameterDistribution, ...] | list[ParameterDistribution],
    simulate: Callable[[Mapping[str, float]], Any],
    *,
    config: FullFieldUncertaintyConfig = FullFieldUncertaintyConfig(),
    metric_functions: Mapping[str, Callable[[Any], float]] | None = None,
) -> FullFieldUncertaintyResult:
    """Monte-Carlo rerun of the optical model with no hidden image registration."""

    config.validate()
    specs = tuple(parameters)
    if not specs:
        raise ValueError("at least one uncertain/fixed parameter must be declared")
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError("parameter names must be unique")
    for spec in specs:
        spec.validate()

    nominal_parameters = {spec.name: float(spec.nominal) for spec in specs}
    nominal_result = simulate(nominal_parameters)
    nominal_intensity, reference_x, reference_y = _extract_intensity_and_axes(nominal_result)

    rng = np.random.default_rng(int(config.random_seed))
    draws = {spec.name: spec.draw(rng, int(config.samples)) for spec in specs}
    accepted_fields: list[np.ndarray] = []
    metric_values: dict[str, list[float]] = {name: [] for name in (metric_functions or {})}
    failed = 0

    for index in range(int(config.samples)):
        sample_parameters = {name: float(values[index]) for name, values in draws.items()}
        try:
            sample_result = simulate(sample_parameters)
            intensity, x, y = _extract_intensity_and_axes(sample_result)
            if intensity.shape != nominal_intensity.shape:
                raise ValueError("sample changed field shape")
            if not _same_axes(reference_x, x) or not _same_axes(reference_y, y):
                raise ValueError("sample changed the physical output grid")
            sample_metrics: dict[str, float] = {}
            for name, function in (metric_functions or {}).items():
                value = float(function(sample_result))
                if not np.isfinite(value):
                    raise ValueError(f"metric {name} returned non-finite value")
                sample_metrics[name] = value
        except (ArithmeticError, FloatingPointError, TypeError, ValueError):
            failed += 1
            continue
        accepted_fields.append(intensity)
        for name, value in sample_metrics.items():
            metric_values[name].append(value)

    if len(accepted_fields) < 2:
        raise RuntimeError("fewer than two uncertainty samples produced valid fields")
    stack = np.stack(accepted_fields, axis=0)
    alpha = (1.0 - float(config.confidence_level)) / 2.0
    mean = np.mean(stack, axis=0)
    std = np.std(stack, axis=0, ddof=1)
    lower = np.quantile(stack, alpha, axis=0)
    upper = np.quantile(stack, 1.0 - alpha, axis=0)
    relative = np.divide(std, mean, out=np.zeros_like(std), where=mean > np.finfo(float).tiny)

    metric_arrays = {name: np.asarray(values, dtype=float) for name, values in metric_values.items()}
    metric_summary: dict[str, dict[str, float]] = {}
    for name, values in metric_arrays.items():
        if values.size == 0:
            continue
        metric_summary[name] = {
            "mean": float(np.mean(values)),
            "standard_deviation": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
            "lower": float(np.quantile(values, alpha)),
            "upper": float(np.quantile(values, 1.0 - alpha)),
            "accepted_samples": int(values.size),
        }

    return FullFieldUncertaintyResult(
        nominal_parameters=nominal_parameters,
        mean_intensity=np.asarray(mean, dtype=float),
        standard_deviation_intensity=np.asarray(std, dtype=float),
        lower_intensity=np.asarray(lower, dtype=float),
        upper_intensity=np.asarray(upper, dtype=float),
        relative_standard_deviation=np.asarray(relative, dtype=float),
        x_m=None if reference_x is None else np.asarray(reference_x, dtype=float),
        y_m=None if reference_y is None else np.asarray(reference_y, dtype=float),
        metric_samples=metric_arrays,
        metric_summary=metric_summary,
        failed_samples=int(failed),
        sample_parameter_table={name: np.asarray(values, dtype=float) for name, values in draws.items()},
        sample_fields=np.asarray(stack, dtype=float) if config.keep_sample_fields else None,
        metadata={
            "method": "full_field_monte_carlo_rerun",
            "requested_samples": int(config.samples),
            "accepted_samples": int(stack.shape[0]),
            "confidence_level": float(config.confidence_level),
            "automatic_registration": False,
            "calibration_uncertainty_only": True,
        },
    )


__all__ = [
    "FullFieldUncertaintyConfig",
    "FullFieldUncertaintyResult",
    "ParameterDistribution",
    "propagate_full_field_uncertainty",
]
