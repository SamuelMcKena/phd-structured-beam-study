"""Diagnostics for the sub-pixel SLM registration study.

The study compares fields, not pictures, so the primary quantities are complex
field comparisons against a declared reference plus vortex-specific structural
measures.  Two reference conventions are used throughout and must never be
conflated:

``continuous``
    the same route with the pixel lattice bypassed entirely.  Differences
    against it measure the *absolute penalty of pixelation* at a given
    registration.

``registered``
    the same route pixelated at zero registration offset.  Differences against
    it, and more usefully the spread across the offset sweep, measure the
    *registration-induced variability* -- the part an experimenter could change
    by translating a panel.

Axial profiling uses the peak transverse intensity rather than the on-axis
value.  For any ``ell > 0`` the on-axis intensity is zero by construction, so an
on-axis axial profile would be meaningless for exactly the cases this study
cares most about.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, Mapping, Sequence

import numpy as np

EPS = np.finfo(float).tiny
TWOPI = 2.0 * np.pi


# --------------------------------------------------------------------------
# field comparisons
# --------------------------------------------------------------------------

def complex_fidelity(field: np.ndarray, reference: np.ndarray) -> float:
    """Normalised overlap ``|<E,Eref>|^2 / (<E,E><Eref,Eref>)``.

    Invariant to a global complex scale, so it isolates genuine structural
    change from an overall piston phase or throughput difference.  1.0 means
    the two fields are identical up to that global factor.
    """

    a = np.asarray(field, dtype=np.complex128).ravel()
    b = np.asarray(reference, dtype=np.complex128).ravel()
    num = abs(complex(np.vdot(b, a))) ** 2
    den = float(np.vdot(a, a).real) * float(np.vdot(b, b).real)
    return float(np.clip(num / max(den, EPS), 0.0, 1.0))


def relative_l2_phase_aligned(field: np.ndarray, reference: np.ndarray) -> float:
    """``||E e^{i phi} - Eref|| / ||Eref||`` minimised over a global piston phase.

    Amplitude is *not* rescaled, so a genuine throughput change still shows up;
    only the physically meaningless global phase is removed.
    """

    a = np.asarray(field, dtype=np.complex128)
    b = np.asarray(reference, dtype=np.complex128)
    inner = complex(np.vdot(b.ravel(), a.ravel()))
    phase = 1.0 if abs(inner) < 1e-300 else np.conj(inner) / abs(inner)
    return float(
        np.linalg.norm((a * phase - b).ravel())
        / max(float(np.linalg.norm(b.ravel())), 1e-300)
    )


def field_power(field: np.ndarray, grid: Mapping[str, Any]) -> float:
    return float(np.sum(np.abs(np.asarray(field)) ** 2) * float(grid["dx"]) ** 2)


def _intensity_centroid_m(
    field: np.ndarray, grid: Mapping[str, Any]
) -> tuple[float, float]:
    """Return the intensity centroid in laboratory x/y coordinates."""

    intensity = np.abs(np.asarray(field, dtype=np.complex128)) ** 2
    total = float(np.sum(intensity))
    X = np.asarray(grid["X"], dtype=float)
    Y = np.asarray(grid["Y"], dtype=float)
    return (
        float(np.sum(intensity * X) / max(total, EPS)),
        float(np.sum(intensity * Y) / max(total, EPS)),
    )


def fourier_translate(
    field: np.ndarray,
    grid: Mapping[str, Any],
    *,
    shift_x_m: float,
    shift_y_m: float,
) -> np.ndarray:
    """Translate a complex field without interpolation using the Fourier shift theorem.

    Positive shifts move the field towards positive laboratory coordinates.
    The shifts are tiny compared with the computational window, so periodic
    wrap-around is negligible for the registration study.
    """

    source = np.asarray(field, dtype=np.complex128)
    if source.ndim != 2 or source.shape[0] != source.shape[1]:
        raise ValueError("field must be a square 2D array")
    n = int(source.shape[0])
    dx = float(grid["dx"])
    fx = np.fft.fftfreq(n, d=dx)
    FY, FX = np.meshgrid(fx, fx, indexing="ij")
    phase = np.exp(-1j * TWOPI * (FX * float(shift_x_m) + FY * float(shift_y_m)))
    return np.fft.ifft2(np.fft.fft2(source) * phase)


def translation_registered_fidelity(
    field: np.ndarray,
    reference: np.ndarray,
    grid: Mapping[str, Any],
) -> dict[str, float]:
    """Compare fields after removing only their transverse centroid separation.

    Raw complex-field fidelity is laboratory-frame sensitive: a physically
    identical beam translated by a few microns has non-zero infidelity. That is
    useful for beam-walk diagnostics but is not by itself a morphology change.
    This metric recentres the field onto the reference centroid with an exact
    Fourier translation. Phase tilt, deformation and higher-order changes remain.
    """

    cx, cy = _intensity_centroid_m(field, grid)
    rx, ry = _intensity_centroid_m(reference, grid)
    shift_x = rx - cx
    shift_y = ry - cy
    aligned = fourier_translate(field, grid, shift_x_m=shift_x, shift_y_m=shift_y)
    fidelity = complex_fidelity(aligned, reference)
    return {
        "fidelity": float(fidelity),
        "infidelity": float(1.0 - fidelity),
        "alignment_shift_x_m": float(shift_x),
        "alignment_shift_y_m": float(shift_y),
        "centroid_separation_m": float(math.hypot(cx - rx, cy - ry)),
    }


# --------------------------------------------------------------------------
# radial / azimuthal structure
# --------------------------------------------------------------------------

@lru_cache(maxsize=16)
def _radial_binning(
    n: int, dx: float, r_max: float, n_bins: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cached radial bin assignment for one grid geometry.

    An axial scan evaluates the radial profile at every z on the same grid, so
    recomputing ``hypot`` and ``digitize`` over the whole grid each time
    dominated the runtime.  The geometry depends only on (N, dx, r_max, n_bins),
    so it is built once and reused.  Returns the flat indices of the samples
    inside ``r_max``, their bin numbers, the per-bin counts and the bin centres.
    """

    x = (np.arange(n) - n / 2 + 0.5) * dx
    X, Y = np.meshgrid(x, x, indexing="xy")
    r = np.hypot(X, Y).ravel()
    edges = np.linspace(0.0, r_max, int(n_bins) + 1)
    inside = np.flatnonzero(r <= r_max)
    idx = np.clip(np.digitize(r[inside], edges) - 1, 0, int(n_bins) - 1)
    count = np.bincount(idx, minlength=int(n_bins))
    centres = 0.5 * (edges[:-1] + edges[1:])
    return inside, idx, count, centres


def radial_profile(
    intensity: np.ndarray, grid: Mapping[str, Any], *, n_bins: int = 400,
    r_max_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Azimuthally averaged intensity on a uniform radial grid."""

    I = np.maximum(np.asarray(intensity, dtype=float), 0.0)
    r_max = float(r_max_m) if r_max_m is not None else float(np.max(np.abs(grid["x"])))
    inside, idx, count, centres = _radial_binning(
        int(grid["N"]), float(grid["dx"]), r_max, int(n_bins)
    )
    total = np.bincount(idx, weights=I.ravel()[inside], minlength=int(n_bins))
    prof = np.zeros(int(n_bins), dtype=float)
    ok = count > 0
    prof[ok] = total[ok] / count[ok]
    return centres, prof


def _sample_ring(
    field: np.ndarray, grid: Mapping[str, Any], radius_m: float, n_theta: int
) -> np.ndarray:
    """Bilinearly sample a complex field around a circle of given radius."""

    n = int(grid["N"])
    dx = float(grid["dx"])
    theta = np.arange(int(n_theta)) * TWOPI / int(n_theta)
    xs = float(radius_m) * np.cos(theta)
    ys = float(radius_m) * np.sin(theta)
    # cell-centred grid: x_i = (i - n/2 + 0.5) dx  ->  i = x/dx + n/2 - 0.5
    fi = xs / dx + n / 2 - 0.5
    fj = ys / dx + n / 2 - 0.5
    i0 = np.floor(fi).astype(int)
    j0 = np.floor(fj).astype(int)
    if i0.min() < 0 or j0.min() < 0 or i0.max() + 1 >= n or j0.max() + 1 >= n:
        raise ValueError("ring radius falls outside the computational window")
    tx = fi - i0
    ty = fj - j0
    f = np.asarray(field, dtype=np.complex128)
    return (
        f[j0, i0] * (1 - tx) * (1 - ty)
        + f[j0, i0 + 1] * tx * (1 - ty)
        + f[j0 + 1, i0] * (1 - tx) * ty
        + f[j0 + 1, i0 + 1] * tx * ty
    )


def azimuthal_spectrum(
    field: np.ndarray, grid: Mapping[str, Any], radius_m: float, *, n_theta: int = 256
) -> np.ndarray:
    """Power in each azimuthal order on a ring, normalised to unit total."""

    ring = _sample_ring(field, grid, radius_m, n_theta)
    spec = np.abs(np.fft.fft(ring)) ** 2
    return spec / max(float(np.sum(spec)), EPS)


def vortex_structure(
    field: np.ndarray,
    grid: Mapping[str, Any],
    *,
    charge: int,
    r_max_m: float | None = None,
    core_fraction: float = 0.3,
    n_theta: int = 256,
) -> dict[str, float]:
    """Ring radius, core darkness, azimuthal purity and ring asymmetry."""

    I = np.abs(np.asarray(field)) ** 2
    r, prof = radial_profile(I, grid, r_max_m=r_max_m)
    if not np.any(prof > 0):
        return {
            "ring_radius_m": float("nan"),
            "ring_peak_intensity": 0.0,
            "core_darkness": float("nan"),
            "azimuthal_purity": float("nan"),
            "ring_asymmetry_rms": float("nan"),
        }
    k = int(np.argmax(prof))
    r_ring = float(r[k])
    ring_peak = float(prof[k])

    core_r = float(core_fraction) * r_ring
    core_mask = r <= max(core_r, float(r[0]))
    core_mean = float(np.mean(prof[core_mask])) if np.any(core_mask) else float("nan")

    purity = float("nan")
    asym = float("nan")
    if r_ring > 2.0 * float(grid["dx"]):
        try:
            spec = azimuthal_spectrum(field, grid, r_ring, n_theta=n_theta)
            purity = float(spec[int(charge) % int(n_theta)])
            ring = _sample_ring(field, grid, r_ring, n_theta)
            ring_i = np.abs(ring) ** 2
            asym = float(np.std(ring_i) / max(np.mean(ring_i), EPS))
        except ValueError:
            pass

    return {
        "ring_radius_m": r_ring,
        "ring_peak_intensity": ring_peak,
        "core_darkness": float(core_mean / max(ring_peak, EPS)),
        "azimuthal_purity": purity,
        "ring_asymmetry_rms": asym,
    }


def plane_metrics(
    field: np.ndarray,
    grid: Mapping[str, Any],
    *,
    charge: int,
    reference: np.ndarray | None = None,
    r_max_m: float | None = None,
) -> dict[str, float]:
    """Full metric set for one transverse plane."""

    I = np.abs(np.asarray(field)) ** 2
    total = float(np.sum(I))
    X = np.asarray(grid["X"], dtype=float)
    Y = np.asarray(grid["Y"], dtype=float)
    out: dict[str, float] = {
        "power": field_power(field, grid),
        "peak_intensity": float(np.max(I)),
        "centroid_x_m": float(np.sum(I * X) / max(total, EPS)),
        "centroid_y_m": float(np.sum(I * Y) / max(total, EPS)),
    }
    out.update(vortex_structure(field, grid, charge=charge, r_max_m=r_max_m))
    if reference is not None:
        out["fidelity"] = complex_fidelity(field, reference)
        out["infidelity"] = 1.0 - out["fidelity"]
        out["relative_l2"] = relative_l2_phase_aligned(field, reference)
        registered = translation_registered_fidelity(field, reference, grid)
        out["translation_registered_fidelity"] = registered["fidelity"]
        out["translation_registered_infidelity"] = registered["infidelity"]
        out["translation_alignment_shift_x_m"] = registered["alignment_shift_x_m"]
        out["translation_alignment_shift_y_m"] = registered["alignment_shift_y_m"]
        out["centroid_separation_m"] = registered["centroid_separation_m"]
        ref_power = field_power(reference, grid)
        out["power_ratio"] = float(out["power"] / max(ref_power, EPS))
    return out


# --------------------------------------------------------------------------
# axial profiling
# --------------------------------------------------------------------------

def axial_profile(
    post_axicon: np.ndarray,
    grid: Mapping[str, Any],
    *,
    wavelength_m: float,
    z_values_m: Sequence[float],
    charge: int,
    r_max_m: float | None = None,
    keep_planes_m: Sequence[float] = (),
) -> dict[str, Any]:
    """Propagate behind the axicon and profile the Bessel region.

    Returns the peak transverse intensity and core darkness against ``z``, plus
    the complex field at any requested planes.  The peak transverse intensity is
    used in place of the on-axis value because a charge ``ell > 0`` beam is dark
    on axis by construction.
    """

    from vbb_study.digital_twin.slm_pixel_registration import lean_asm_propagator

    z_values = [float(z) for z in z_values_m]
    keep = {float(z) for z in keep_planes_m}
    peak = np.zeros(len(z_values), dtype=float)
    core = np.zeros(len(z_values), dtype=float)
    ring = np.zeros(len(z_values), dtype=float)
    planes: dict[float, np.ndarray] = {}

    propagate = lean_asm_propagator(post_axicon, grid, wavelength_m)
    for i, z in enumerate(z_values):
        u = propagate(z)
        s = vortex_structure(u, grid, charge=charge, r_max_m=r_max_m)
        peak[i] = s["ring_peak_intensity"]
        core[i] = s["core_darkness"]
        ring[i] = s["ring_radius_m"]
        if z in keep:
            planes[z] = u
        else:
            del u

    k = int(np.argmax(peak)) if peak.size else 0
    half = peak >= 0.5 * peak[k] if peak.size else np.zeros(0, dtype=bool)
    if np.any(half):
        idx = np.where(half)[0]
        zone = float(z_values[idx[-1]] - z_values[idx[0]])
    else:
        zone = float("nan")

    return {
        "z_m": np.asarray(z_values, dtype=float),
        "peak_intensity_z": peak,
        "core_darkness_z": core,
        "ring_radius_z": ring,
        "z_at_peak_m": float(z_values[k]) if z_values else float("nan"),
        "peak_intensity": float(peak[k]) if peak.size else float("nan"),
        "bessel_zone_fwhm_m": zone,
        "planes": planes,
    }


def sweep_spread(values: Sequence[float]) -> dict[str, float]:
    """Spread of a metric across a registration sweep.

    The registration-induced variability is the physically meaningful quantity:
    it is what an experimenter can change by translating a panel, whereas the
    mean offset from the continuous reference is a fixed pixelation penalty.
    """

    arr = np.asarray([float(v) for v in values], dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"min": float("nan"), "max": float("nan"), "mean": float("nan"),
                "peak_to_peak": float("nan"), "relative_peak_to_peak": float("nan")}
    mean = float(np.mean(finite))
    ptp = float(np.max(finite) - np.min(finite))
    return {
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": mean,
        "peak_to_peak": ptp,
        "relative_peak_to_peak": float(ptp / max(abs(mean), EPS)),
    }
