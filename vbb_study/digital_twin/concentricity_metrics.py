"""Metrics for *visible* concentricity of annular/Bessel intensity fields.

A single principal-ring CV can miss the fan/cross morphology that is obvious in
outer Bessel sidelobes.  These helpers measure several target radial maxima and
score both (i) peak intensity variation around each ring and (ii) radial motion
of the local peak as azimuth changes.  They are detector/image metrics only;
they make no claim about optical phase or hardware calibration.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

EPS = np.finfo(float).tiny


def radial_profile(image: np.ndarray, axis_um: np.ndarray, *, dr_um: float = 0.5, rmax_um: float = 145.0) -> tuple[np.ndarray, np.ndarray]:
    axis = np.asarray(axis_um, float)
    img = np.asarray(image, float)
    X, Y = np.meshgrid(axis, axis, indexing="xy")
    R = np.hypot(X, Y)
    edges = np.arange(0.0, float(rmax_um) + float(dr_um), float(dr_um))
    idx = np.digitize(R.ravel(), edges) - 1
    good = (idx >= 0) & (idx < len(edges) - 1)
    sums = np.bincount(idx[good], weights=img.ravel()[good], minlength=len(edges)-1)
    num = np.bincount(idx[good], minlength=len(edges)-1)
    return 0.5*(edges[:-1] + edges[1:]), sums/np.maximum(num, 1)


def target_ring_radii(
    target: np.ndarray,
    axis_um: np.ndarray,
    *,
    rmin_um: float = 20.0,
    rmax_um: float = 135.0,
    minimum_relative_peak: float = 0.015,
    minimum_separation_um: float = 4.0,
    max_rings: int = 10,
) -> np.ndarray:
    rr, p = radial_profile(target, axis_um, dr_um=0.5, rmax_um=rmax_um+2.0)
    local = np.flatnonzero((p[1:-1] > p[:-2]) & (p[1:-1] >= p[2:])) + 1
    local = local[(rr[local] >= rmin_um) & (rr[local] <= rmax_um)]
    if local.size == 0:
        return np.asarray([], float)
    keep = local[p[local] >= float(minimum_relative_peak)*max(float(np.max(p)), EPS)]
    # Greedy by peak height, then return in radial order.  This prevents many
    # adjacent samples of a broad annulus from dominating the metric.
    chosen: list[int] = []
    for i in keep[np.argsort(p[keep])[::-1]]:
        if all(abs(float(rr[i]-rr[j])) >= float(minimum_separation_um) for j in chosen):
            chosen.append(int(i))
        if len(chosen) >= int(max_rings):
            break
    return np.sort(rr[np.asarray(chosen, int)])


def _sample_polar(image: np.ndarray, axis_um: np.ndarray, radii_um: np.ndarray, theta_rad: np.ndarray) -> np.ndarray:
    axis = np.asarray(axis_um, float)
    radii = np.asarray(radii_um, float)[:, None]
    theta = np.asarray(theta_rad, float)[None, :]
    xs = radii*np.cos(theta); ys = radii*np.sin(theta)
    ix = np.interp(xs, axis, np.arange(axis.size, dtype=float))
    iy = np.interp(ys, axis, np.arange(axis.size, dtype=float))
    return map_coordinates(np.asarray(image, float), [iy, ix], order=1, mode="nearest")


def ring_metrics(
    image: np.ndarray,
    target: np.ndarray,
    axis_um: np.ndarray,
    *,
    ntheta: int = 360,
    radial_search_halfwidth_um: float = 2.75,
    radial_search_samples: int = 13,
    max_harmonic: int = 8,
) -> dict:
    rings = target_ring_radii(target, axis_um)
    if rings.size == 0:
        return {
            "ring_radii_um": [],
            "mean_ring_intensity_cv": float("nan"),
            "mean_ring_radius_std_um": float("nan"),
            "mean_ring_radius_peak_to_peak_um": float("nan"),
            "mean_angular_harmonic_energy": float("nan"),
            "per_ring": [],
        }
    theta = np.linspace(0.0, 2.0*np.pi, int(ntheta), endpoint=False)
    offsets = np.linspace(-float(radial_search_halfwidth_um), float(radial_search_halfwidth_um), int(radial_search_samples))
    per = []
    for r0 in rings:
        rr = r0 + offsets
        samples = _sample_polar(image, axis_um, rr, theta)
        imax = np.argmax(samples, axis=0)
        peak_i = samples[imax, np.arange(theta.size)]
        peak_r = rr[imax]
        mean_i = max(float(np.mean(peak_i)), EPS)
        norm_i = peak_i/mean_i
        harmonics = []
        for m in range(1, int(max_harmonic)+1):
            c = np.mean(norm_i*np.exp(-1j*float(m)*theta))
            harmonics.append(float(2.0*abs(c)))
        per.append({
            "target_radius_um": float(r0),
            "intensity_cv": float(np.std(peak_i)/mean_i),
            "radius_std_um": float(np.std(peak_r)),
            "radius_peak_to_peak_um": float(np.ptp(peak_r)),
            "angular_harmonic_amplitudes_m1_to_mmax": harmonics,
            "angular_harmonic_energy": float(np.sqrt(np.sum(np.square(harmonics)))),
        })
    return {
        "ring_radii_um": [float(x) for x in rings],
        "mean_ring_intensity_cv": float(np.mean([r["intensity_cv"] for r in per])),
        "mean_ring_radius_std_um": float(np.mean([r["radius_std_um"] for r in per])),
        "mean_ring_radius_peak_to_peak_um": float(np.mean([r["radius_peak_to_peak_um"] for r in per])),
        "mean_angular_harmonic_energy": float(np.mean([r["angular_harmonic_energy"] for r in per])),
        "per_ring": per,
    }


def stack_metrics(stack: np.ndarray, target_stack: np.ndarray, axis_um: np.ndarray, indices: np.ndarray | list[int]) -> dict:
    rows = [ring_metrics(np.asarray(stack)[int(i)], np.asarray(target_stack)[int(i)], axis_um) for i in np.asarray(indices, int)]
    keys = ("mean_ring_intensity_cv", "mean_ring_radius_std_um", "mean_ring_radius_peak_to_peak_um", "mean_angular_harmonic_energy")
    out = {k: float(np.nanmean([r[k] for r in rows])) for k in keys}
    out["per_plane"] = rows
    return out
