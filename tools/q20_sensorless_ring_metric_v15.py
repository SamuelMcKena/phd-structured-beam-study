"""Sensorless ring-quality utilities for high-order vortex/Bessel correction.

This module deliberately does *not* infer a unique system wavefront from intensity.
It provides the experimentally observable part of a camera-in-the-loop correction:

    SLM trial phase -> camera intensity -> ring-quality score.

The score rewards concentric annuli by measuring angular coefficient of variation,
azimuthal Fourier content (m=2,3,4 by default), and radial peak wobble.  Trial
phase maps use a small, interpretable annular harmonic basis.  These utilities
are intended to complement the detector-aware digital twin; they are not a
claim of experimental post-correction closure.

References motivating the method:
- Miao et al., Opt. Express 30, 11360-11371 (2022), DOI 10.1364/OE.454796.
- Kim et al., Opt. Express 33, 680-693 (2025), DOI 10.1364/OE.541033.
- Luo et al., Opt. Express 34, 26872-26882 (2026), DOI 10.1364/OE.606056.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


def load_intensity(path: str | Path) -> np.ndarray:
    """Load an intensity image from .npy or a conventional image file."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
    else:
        arr = np.asarray(Image.open(path).convert("F"), dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2-D intensity array, got shape {arr.shape}")
    arr = np.asarray(arr, dtype=np.float64)
    arr -= np.nanmin(arr)
    peak = float(np.nanmax(arr))
    if not np.isfinite(peak) or peak <= 0:
        raise ValueError("Intensity image has no positive finite dynamic range")
    return arr / peak


def intensity_centroid(image: np.ndarray, power: float = 1.0) -> tuple[float, float]:
    """Return (cx, cy) in pixel coordinates using a positive intensity centroid."""
    w = np.clip(np.asarray(image, dtype=np.float64), 0.0, None) ** power
    yy, xx = np.indices(w.shape, dtype=np.float64)
    s = float(w.sum())
    if s <= 0:
        raise ValueError("Cannot centroid an empty image")
    return float((w * xx).sum() / s), float((w * yy).sum() / s)


def bilinear_sample(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vectorised bilinear sampling; out-of-frame samples become NaN."""
    h, w = image.shape
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1
    valid = (x0 >= 0) & (y0 >= 0) & (x1 < w) & (y1 < h)
    out = np.full(np.broadcast(x, y).shape, np.nan, dtype=np.float64)
    if not np.any(valid):
        return out
    xv, yv = x[valid], y[valid]
    x0v, x1v, y0v, y1v = x0[valid], x1[valid], y0[valid], y1[valid]
    dx, dy = xv - x0v, yv - y0v
    out[valid] = (
        image[y0v, x0v] * (1 - dx) * (1 - dy)
        + image[y0v, x1v] * dx * (1 - dy)
        + image[y1v, x0v] * (1 - dx) * dy
        + image[y1v, x1v] * dx * dy
    )
    return out


def polar_intensity(
    image: np.ndarray,
    centre: tuple[float, float] | None = None,
    n_r: int = 256,
    n_theta: int = 720,
    r_max_px: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample image on a polar grid and return radii, angles, intensity[r,theta]."""
    h, w = image.shape
    cx, cy = centre if centre is not None else intensity_centroid(image, power=1.5)
    if r_max_px is None:
        r_max_px = 0.47 * min(h, w)
    radii = np.linspace(1.0, float(r_max_px), n_r)
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    rr, tt = np.meshgrid(radii, theta, indexing="ij")
    x = cx + rr * np.cos(tt)
    y = cy + rr * np.sin(tt)
    return radii, theta, bilinear_sample(image, x, y)


def select_significant_rings(
    polar: np.ndarray,
    radii: np.ndarray,
    max_rings: int = 5,
    min_relative_peak: float = 0.12,
    min_separation_px: float = 4.0,
) -> list[int]:
    """Select strong radial maxima from the azimuthal mean without scipy."""
    profile = np.nanmean(polar, axis=1)
    if profile.size < 3:
        return []
    cand = np.flatnonzero((profile[1:-1] > profile[:-2]) & (profile[1:-1] >= profile[2:])) + 1
    if cand.size == 0:
        cand = np.array([int(np.nanargmax(profile))])
    threshold = min_relative_peak * float(np.nanmax(profile))
    cand = [int(i) for i in cand if profile[i] >= threshold]
    cand.sort(key=lambda i: float(profile[i]), reverse=True)
    chosen: list[int] = []
    for idx in cand:
        if all(abs(float(radii[idx] - radii[j])) >= min_separation_px for j in chosen):
            chosen.append(idx)
        if len(chosen) >= max_rings:
            break
    return sorted(chosen, key=lambda i: float(radii[i]))


def ring_metrics(
    image: np.ndarray,
    centre: tuple[float, float] | None = None,
    harmonic_orders: Iterable[int] = (2, 3, 4),
    max_rings: int = 5,
) -> dict:
    """Compute concentricity metrics directly from an intensity image.

    Lower values are better for all distortion metrics and for ``score``.
    The radial wobble is evaluated by following the local radial maximum around
    each selected ring, rather than assuming a fixed-radius contour is physical.
    """
    image = np.asarray(image, dtype=np.float64)
    radii, theta, polar = polar_intensity(image, centre=centre)
    ring_idx = select_significant_rings(polar, radii, max_rings=max_rings)
    if not ring_idx:
        raise ValueError("No significant annular maxima found")

    harmonic_orders = tuple(int(m) for m in harmonic_orders)
    per_ring = []
    for idx in ring_idx:
        # Fixed-radius angular uniformity on the actual bright ring.
        trace = np.nan_to_num(polar[idx], nan=0.0)
        mean = float(np.mean(trace))
        cv = float(np.std(trace) / (mean + 1e-12))
        fft = np.fft.rfft(trace - mean)
        dc = float(np.sum(np.abs(trace)) + 1e-12)
        harmonics = {
            f"m{m}": float((2.0 * np.abs(fft[m]) / dc) ** 2) if m < fft.size else 0.0
            for m in harmonic_orders
        }

        # Follow the local ring maximum in a radial neighbourhood for each angle.
        half_window = max(2, int(round(0.02 * len(radii))))
        lo = max(0, idx - half_window)
        hi = min(len(radii), idx + half_window + 1)
        local = np.nan_to_num(polar[lo:hi], nan=-np.inf)
        peak_local = np.argmax(local, axis=0) + lo
        peak_r = radii[peak_local]
        r_mean = float(np.mean(peak_r))
        r_std_fraction = float(np.std(peak_r) / (r_mean + 1e-12))
        r_p2p_fraction = float(np.ptp(peak_r) / (r_mean + 1e-12))

        per_ring.append(
            {
                "radius_px": float(radii[idx]),
                "mean_intensity": mean,
                "angular_cv": cv,
                "radial_wobble_std_fraction": r_std_fraction,
                "radial_wobble_p2p_fraction": r_p2p_fraction,
                **harmonics,
            }
        )

    weights = np.array([r["mean_intensity"] for r in per_ring], dtype=float)
    weights /= weights.sum() + 1e-12
    agg = {
        "angular_cv": float(np.sum(weights * [r["angular_cv"] for r in per_ring])),
        "radial_wobble_std_fraction": float(
            np.sum(weights * [r["radial_wobble_std_fraction"] for r in per_ring])
        ),
        "radial_wobble_p2p_fraction": float(
            np.sum(weights * [r["radial_wobble_p2p_fraction"] for r in per_ring])
        ),
    }
    for m in harmonic_orders:
        agg[f"m{m}"] = float(np.sum(weights * [r[f"m{m}"] for r in per_ring]))

    # Balanced dimensionless objective.  No topology claim is made here; topology
    # must be checked independently at a physically justified high-support contour.
    score = (
        agg["angular_cv"]
        + 1.5 * agg["radial_wobble_std_fraction"]
        + 0.5 * agg["radial_wobble_p2p_fraction"]
        + sum(2.0 * agg[f"m{m}"] for m in harmonic_orders)
    )
    return {
        "centre_px": list(centre if centre is not None else intensity_centroid(image, power=1.5)),
        "n_rings": len(per_ring),
        "rings": per_ring,
        "aggregate": agg,
        "score": float(score),
        "score_direction": "lower_is_better",
    }


def annular_harmonic_phase(
    shape: tuple[int, int],
    order_m: int,
    amplitude_rad: float,
    r_inner_fraction: float = 0.18,
    r_outer_fraction: float = 0.46,
    orientation_rad: float = 0.0,
    sine_component: bool = False,
) -> np.ndarray:
    """Generate one smoothly windowed annular harmonic SLM trial phase.

    Radii are fractions of the smaller image/SLM dimension, making the function
    independent of a still-uncalibrated SLM2-to-axicon conjugate mapping.
    """
    h, w = shape
    yy, xx = np.indices(shape, dtype=np.float64)
    cx, cy = 0.5 * (w - 1), 0.5 * (h - 1)
    x, y = xx - cx, yy - cy
    r = np.hypot(x, y) / min(h, w)
    theta = np.arctan2(y, x) - orientation_rad
    if not (0 <= r_inner_fraction < r_outer_fraction <= 0.5):
        raise ValueError("Require 0 <= r_inner < r_outer <= 0.5")

    # Raised-cosine edges occupy 15% of the annulus width.
    width = r_outer_fraction - r_inner_fraction
    edge = max(1e-6, 0.15 * width)
    win = np.ones_like(r)
    win[r < r_inner_fraction] = 0.0
    win[r > r_outer_fraction] = 0.0
    a = (r >= r_inner_fraction) & (r < r_inner_fraction + edge)
    b = (r > r_outer_fraction - edge) & (r <= r_outer_fraction)
    win[a] = 0.5 - 0.5 * np.cos(np.pi * (r[a] - r_inner_fraction) / edge)
    win[b] = 0.5 - 0.5 * np.cos(np.pi * (r_outer_fraction - r[b]) / edge)
    angular = np.sin(order_m * theta) if sine_component else np.cos(order_m * theta)
    return float(amplitude_rad) * win * angular


def azimuthal_reference(image: np.ndarray, centre: tuple[float, float] | None = None) -> np.ndarray:
    """Construct the closest purely radial intensity reference from the image itself.

    This intentionally preserves the measured radial structure while removing
    angular distortion.  It is a *target/reference*, never an experimental after-image.
    """
    image = np.asarray(image, dtype=np.float64)
    h, w = image.shape
    cx, cy = centre if centre is not None else intensity_centroid(image, power=1.5)
    yy, xx = np.indices(image.shape, dtype=np.float64)
    rr = np.hypot(xx - cx, yy - cy)
    rbin = np.floor(rr).astype(int)
    n = int(rbin.max()) + 1
    sums = np.bincount(rbin.ravel(), weights=image.ravel(), minlength=n)
    counts = np.bincount(rbin.ravel(), minlength=n)
    radial = sums / np.maximum(counts, 1)
    return radial[np.clip(rbin, 0, n - 1)]


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Intensity image (.npy or image file)")
    parser.add_argument("--json", dest="json_path", help="Write metrics JSON")
    parser.add_argument("--reference", help="Write azimuthally averaged target as .npy")
    args = parser.parse_args()
    image = load_intensity(args.image)
    metrics = ring_metrics(image)
    text = json.dumps(metrics, indent=2)
    print(text)
    if args.json_path:
        Path(args.json_path).write_text(text + "\n")
    if args.reference:
        np.save(args.reference, azimuthal_reference(image))


if __name__ == "__main__":
    _main()
