"""Native-array metrics for the governed Phase 2E source-scale propagation route."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.signal import find_peaks, periodogram


EPS = np.finfo(float).tiny


@dataclass(frozen=True)
class FeatureMeasurement:
    case_id: str
    primary_observable_raw: float
    feature_radius_m: float
    feature_width_m: float
    dark_core_radius_m: float
    valid: bool
    invalid_reason: str


@dataclass(frozen=True)
class FixedRegion:
    case_id: str
    inner_radius_m: float
    outer_radius_m: float
    provenance: str


def on_axis_intensity(intensity: np.ndarray) -> float:
    """Bilinear centre estimate for the even-sized, cell-centred source grids."""

    image = np.asarray(intensity, dtype=float)
    cy = image.shape[0] // 2
    cx = image.shape[1] // 2
    return float(np.mean(image[cy - 1 : cy + 1, cx - 1 : cx + 1]))


def radial_profile(
    intensity: np.ndarray,
    grid: Mapping[str, Any],
    *,
    maximum_radius_m: float = 0.5e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a native annular average without interpolating the metric data."""

    x = np.asarray(grid["x"], dtype=float)
    dx = float(grid["dx"])
    keep = np.flatnonzero(np.abs(x) <= float(maximum_radius_m))
    if keep.size < 12:
        raise ValueError("radial analysis crop contains too few native samples")
    lo, hi = int(keep[0]), int(keep[-1]) + 1
    image = np.asarray(intensity[lo:hi, lo:hi], dtype=float)
    X = np.asarray(grid["X"][lo:hi, lo:hi], dtype=float)
    Y = np.asarray(grid["Y"][lo:hi, lo:hi], dtype=float)
    radius = np.hypot(X, Y)
    bins = np.floor(radius / dx).astype(np.int32)
    n_bins = int(np.floor(float(maximum_radius_m) / dx)) + 1
    valid = bins < n_bins
    sums = np.bincount(bins[valid].ravel(), weights=image[valid].ravel(), minlength=n_bins)
    counts = np.bincount(bins[valid].ravel(), minlength=n_bins)
    populated = counts > 0
    radii = (np.arange(n_bins, dtype=float) + 0.5) * dx
    profile = np.full(n_bins, np.nan, dtype=float)
    profile[populated] = sums[populated] / counts[populated]
    return radii[populated], profile[populated], counts[populated]


def _outward_crossing(
    radius_m: np.ndarray,
    values: np.ndarray,
    start_index: int,
    threshold: float,
) -> float:
    for index in range(int(start_index) + 1, values.size):
        if values[index] <= threshold < values[index - 1]:
            x0, x1 = radius_m[index - 1 : index + 1]
            y0, y1 = values[index - 1 : index + 1]
            denominator = float(y1 - y0)
            if abs(denominator) <= EPS:
                continue
            fraction = float(np.clip((threshold - y0) / denominator, 0.0, 1.0))
            return float(x0 + fraction * (x1 - x0))
    return float("nan")


def _inward_crossing(
    radius_m: np.ndarray,
    values: np.ndarray,
    start_index: int,
    threshold: float,
) -> float:
    for index in range(int(start_index) - 1, -1, -1):
        if values[index] <= threshold < values[index + 1]:
            x0, x1 = radius_m[index : index + 2]
            y0, y1 = values[index : index + 2]
            denominator = float(y1 - y0)
            if abs(denominator) <= EPS:
                continue
            fraction = float(np.clip((threshold - y0) / denominator, 0.0, 1.0))
            return float(x0 + fraction * (x1 - x0))
    return float("nan")


def measure_feature(
    case_id: str,
    intensity: np.ndarray,
    grid: Mapping[str, Any],
    *,
    formation_threshold_raw: float = 0.0,
    maximum_radius_m: float = 0.5e-3,
) -> FeatureMeasurement:
    """Measure the B0 HWHM core or the V1/V3 ring on native samples."""

    radius, profile, _ = radial_profile(
        intensity, grid, maximum_radius_m=maximum_radius_m
    )
    dx = float(grid["dx"])
    axis = on_axis_intensity(intensity)
    case = str(case_id).upper()
    if case == "B0":
        local_count = min(3, profile.size)
        peak_index = int(np.nanargmax(profile[:local_count]))
        peak = float(profile[peak_index])
        if axis < float(formation_threshold_raw):
            return FeatureMeasurement(case, axis, np.nan, np.nan, np.nan, False, "below_formation_threshold")
        if peak < float(np.nanmax(profile[: min(profile.size, peak_index + 6)])) * (1.0 - 1e-9):
            return FeatureMeasurement(case, axis, np.nan, np.nan, np.nan, False, "central_feature_not_local_maximum")
        crossing = _outward_crossing(radius, profile, peak_index, 0.5 * peak)
        width = 2.0 * crossing
        if not np.isfinite(crossing):
            return FeatureMeasurement(case, axis, np.nan, np.nan, np.nan, False, "no_outward_half_maximum_crossing")
        if crossing >= 0.95 * maximum_radius_m:
            return FeatureMeasurement(case, axis, np.nan, np.nan, np.nan, False, "crossing_reaches_analysis_boundary")
        if width / dx < 6.0:
            return FeatureMeasurement(case, axis, np.nan, np.nan, np.nan, False, "fewer_than_six_native_samples_across_width")
        return FeatureMeasurement(case, axis, crossing, width, np.nan, True, "")

    if case not in {"V1", "V3"}:
        raise ValueError(f"unsupported final source case {case_id!r}")
    minimum_index = int(np.searchsorted(radius, max(2.0 * dx, 4.0e-6)))
    maximum_index = int(np.searchsorted(radius, min(maximum_radius_m * 0.8, 0.3e-3)))
    candidate = profile[minimum_index:maximum_index]
    peaks, _ = find_peaks(candidate)
    if peaks.size:
        peak_index = minimum_index + int(peaks[np.nanargmax(candidate[peaks])])
    else:
        peak_index = minimum_index + int(np.nanargmax(candidate))
    peak = float(profile[peak_index])
    peak_radius = float(radius[peak_index])
    if 0 < peak_index < profile.size - 1:
        left = float(profile[peak_index - 1])
        right = float(profile[peak_index + 1])
        curvature = left - 2.0 * peak + right
        if abs(curvature) > EPS:
            offset_bins = float(np.clip(0.5 * (left - right) / curvature, -1.0, 1.0))
            peak_radius += offset_bins * float(radius[peak_index + 1] - radius[peak_index])
    primary = peak
    if primary < float(formation_threshold_raw):
        return FeatureMeasurement(case, primary, np.nan, np.nan, np.nan, False, "below_formation_threshold")
    half = 0.5 * peak
    inner = _inward_crossing(radius, profile, peak_index, half)
    outer = _outward_crossing(radius, profile, peak_index, half)
    if not np.isfinite(inner) or not np.isfinite(outer):
        return FeatureMeasurement(case, primary, np.nan, np.nan, np.nan, False, "missing_ring_half_maximum_crossing")
    width = outer - inner
    if outer >= 0.95 * maximum_radius_m:
        return FeatureMeasurement(case, primary, np.nan, np.nan, np.nan, False, "crossing_reaches_analysis_boundary")
    if width / dx < 4.0:
        return FeatureMeasurement(
            case,
            primary,
            np.nan,
            np.nan,
            np.nan,
            False,
            "fewer_than_four_native_samples_across_ring_width",
        )
    return FeatureMeasurement(
        case,
        primary,
        peak_radius,
        float(width),
        float(inner),
        True,
        "",
    )


def fixed_region_from_reference(measurement: FeatureMeasurement) -> FixedRegion:
    if not measurement.valid:
        raise ValueError("reference feature must be valid before defining a fixed region")
    if measurement.case_id == "B0":
        return FixedRegion(
            "B0",
            0.0,
            2.0 * float(measurement.feature_radius_m),
            "N3072 z60 native B0 HWHM; fixed bucket outer radius = 2x HWHM",
        )
    half_width = 0.5 * float(measurement.feature_width_m)
    return FixedRegion(
        measurement.case_id,
        max(0.0, float(measurement.feature_radius_m) - half_width),
        float(measurement.feature_radius_m) + half_width,
        "N3072 z60 native ring FWHM annulus; fixed across grid and z",
    )


def fixed_region_power(
    intensity: np.ndarray,
    radius_m: np.ndarray,
    region: FixedRegion,
    dx_m: float,
) -> float:
    mask = (radius_m >= region.inner_radius_m) & (radius_m <= region.outer_radius_m)
    return float(np.sum(np.asarray(intensity, dtype=float)[mask]) * float(dx_m) ** 2)


def edge_energy_fraction(
    intensity: np.ndarray,
    grid: Mapping[str, Any],
    *,
    edge_fraction: float = 0.05,
) -> float:
    image = np.asarray(intensity, dtype=float)
    n = int(grid["N"])
    border = max(1, int(np.ceil(float(edge_fraction) * n)))
    edge_sum = (
        np.sum(image[:border, :])
        + np.sum(image[-border:, :])
        + np.sum(image[border:-border, :border])
        + np.sum(image[border:-border, -border:])
    )
    return float(edge_sum / max(float(np.sum(image)), EPS))


def longest_true_interval(z_m: np.ndarray, mask: np.ndarray) -> tuple[float, float] | None:
    valid = np.asarray(mask, dtype=bool)
    if not np.any(valid):
        return None
    padded = np.r_[False, valid, False]
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    starts, stops = changes[0::2], changes[1::2]
    choice = int(np.argmax(stops - starts))
    return float(z_m[starts[choice]]), float(z_m[stops[choice] - 1])


def zone_summary(
    case_id: str,
    z_m: np.ndarray,
    primary_raw: np.ndarray,
    fixed_power_raw: np.ndarray,
    feature_radius_m: np.ndarray,
    feature_valid: np.ndarray,
) -> dict[str, Any]:
    primary = np.asarray(primary_raw, dtype=float)
    bucket = np.asarray(fixed_power_raw, dtype=float)
    valid = np.asarray(feature_valid, dtype=bool) & np.isfinite(feature_radius_m)
    fwhm_mask = primary >= 0.5 * max(float(np.nanmax(primary)), EPS)
    reference_indices = np.flatnonzero(valid & fwhm_mask)
    reference_radius = (
        float(np.nanmedian(np.asarray(feature_radius_m)[reference_indices]))
        if reference_indices.size
        else float("nan")
    )
    radius_ok = valid & (
        np.abs(np.asarray(feature_radius_m) - reference_radius)
        <= 0.1 * max(reference_radius, EPS)
    )
    strict_mask = (
        fwhm_mask
        & (bucket >= 0.5 * max(float(np.nanmax(bucket)), EPS))
        & radius_ok
        & valid
    )
    return {
        "case_id": str(case_id),
        "configured_nominal_interval_m": [0.020, 0.060],
        "measured_FWHM_axial_zone_m": longest_true_interval(z_m, fwhm_mask),
        "measured_strict_useful_region_m": longest_true_interval(z_m, strict_mask),
        "reference_feature_radius_m": reference_radius,
        "configured_interval_is_measured_zone": False,
    }


def normalised_l2(reference: Sequence[float], candidate: Sequence[float]) -> float:
    ref = np.asarray(reference, dtype=float)
    cand = np.asarray(candidate, dtype=float)
    return float(np.linalg.norm(cand - ref) / max(float(np.linalg.norm(ref)), EPS))


def dominant_ripple_period_m(z_m: Sequence[float], trace: Sequence[float]) -> float:
    z = np.asarray(z_m, dtype=float)
    values = np.asarray(trace, dtype=float)
    if z.size < 8:
        return float("nan")
    detrended = values - np.polyval(np.polyfit(z, values, deg=2), z)
    frequency, power = periodogram(detrended, fs=1.0 / float(np.median(np.diff(z))))
    usable = frequency > 0.0
    if not np.any(usable) or float(np.max(power[usable])) <= EPS:
        return float("nan")
    peak_frequency = float(frequency[usable][np.argmax(power[usable])])
    return 1.0 / peak_frequency
