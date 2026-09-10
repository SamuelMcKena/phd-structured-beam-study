"""Broadband linear-optics orchestration for the bench-calibrated digital twin.

This module does not invent a new propagation solver. It repeatedly calls a
user-supplied single-wavelength field solver and combines the returned fields
with an explicitly supplied optical spectrum. Slow camera/beam-profiler
observables are formed as the incoherent spectral sum of intensities. Coherent
time-domain reconstruction is available only when complex spectral phase is
supplied or deliberately declared zero.

The intended use is:

    measured spectrum -> one-wavelength canonical optical route -> spectral
    field stack -> detector-integrated intensity and optional temporal field.

All wavelength values are SI metres. ``Spectrum.energy_weights`` are integrated
energy fractions assigned to the discrete spectral samples, not raw spectral-
density point values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


EPS = np.finfo(float).tiny
C0 = 299_792_458.0
TWOPI = 2.0 * np.pi


def _quadrature_cell_widths(axis: np.ndarray) -> np.ndarray:
    """Return trapezoidal point weights for a strictly increasing 1-D axis."""

    values = np.asarray(axis, dtype=float)
    if values.ndim != 1 or values.size < 2 or np.any(~np.isfinite(values)) or np.any(np.diff(values) <= 0.0):
        raise ValueError("quadrature axis must be finite, 1-D and strictly increasing")
    widths = np.empty_like(values)
    widths[0] = 0.5 * (values[1] - values[0])
    widths[-1] = 0.5 * (values[-1] - values[-2])
    if values.size > 2:
        widths[1:-1] = 0.5 * (values[2:] - values[:-2])
    return widths


@dataclass(frozen=True)
class Spectrum:
    wavelengths_m: np.ndarray
    energy_weights: np.ndarray
    spectral_phase_rad: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validated(self) -> "Spectrum":
        wl = np.asarray(self.wavelengths_m, dtype=float)
        weights = np.asarray(self.energy_weights, dtype=float)
        if wl.ndim != 1 or weights.ndim != 1 or wl.size != weights.size or wl.size < 1:
            raise ValueError("spectrum wavelengths and weights must be matching 1-D arrays")
        if np.any(~np.isfinite(wl)) or np.any(wl <= 0.0) or np.any(np.diff(wl) <= 0.0):
            raise ValueError("wavelengths_m must be finite, positive and strictly increasing")
        if np.any(~np.isfinite(weights)) or np.any(weights < 0.0) or float(np.sum(weights)) <= 0.0:
            raise ValueError("energy_weights must be finite, non-negative and non-zero")
        phase = None
        if self.spectral_phase_rad is not None:
            phase = np.asarray(self.spectral_phase_rad, dtype=float)
            if phase.shape != wl.shape or np.any(~np.isfinite(phase)):
                raise ValueError("spectral_phase_rad must match wavelengths_m")
        return Spectrum(
            wavelengths_m=wl.copy(),
            energy_weights=weights / float(np.sum(weights)),
            spectral_phase_rad=None if phase is None else phase.copy(),
            metadata=dict(self.metadata),
        )

    @property
    def central_wavelength_m(self) -> float:
        s = self.validated()
        return float(np.sum(s.wavelengths_m * s.energy_weights))


@dataclass(frozen=True)
class SpectralFieldPlane:
    wavelength_m: float
    x_m: np.ndarray
    y_m: np.ndarray
    Ex: np.ndarray
    Ey: np.ndarray | None = None
    Ez: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validated(self) -> "SpectralFieldPlane":
        x = np.asarray(self.x_m, dtype=float)
        y = np.asarray(self.y_m, dtype=float)
        ex = np.asarray(self.Ex, dtype=np.complex128)
        if x.ndim != 1 or y.ndim != 1 or np.any(np.diff(x) <= 0.0) or np.any(np.diff(y) <= 0.0):
            raise ValueError("field axes must be strictly increasing one-dimensional arrays")
        if ex.shape != (y.size, x.size) or np.any(~np.isfinite(ex)):
            raise ValueError("Ex does not match physical axes or contains non-finite values")
        components: list[np.ndarray | None] = [ex]
        for component, name in ((self.Ey, "Ey"), (self.Ez, "Ez")):
            if component is None:
                components.append(None)
                continue
            arr = np.asarray(component, dtype=np.complex128)
            if arr.shape != ex.shape or np.any(~np.isfinite(arr)):
                raise ValueError(f"{name} does not match Ex")
            components.append(arr)
        if not np.isfinite(self.wavelength_m) or self.wavelength_m <= 0.0:
            raise ValueError("field wavelength_m must be finite and positive")
        return SpectralFieldPlane(
            wavelength_m=float(self.wavelength_m),
            x_m=x.copy(),
            y_m=y.copy(),
            Ex=components[0].copy(),
            Ey=None if components[1] is None else components[1].copy(),
            Ez=None if components[2] is None else components[2].copy(),
            metadata=dict(self.metadata),
        )

    @property
    def intensity(self) -> np.ndarray:
        total = np.abs(np.asarray(self.Ex)) ** 2
        if self.Ey is not None:
            total = total + np.abs(np.asarray(self.Ey)) ** 2
        if self.Ez is not None:
            total = total + np.abs(np.asarray(self.Ez)) ** 2
        return np.asarray(total, dtype=float)


@dataclass(frozen=True)
class BroadbandResult:
    spectrum: Spectrum
    fields: tuple[SpectralFieldPlane, ...]
    detector_integrated_intensity: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    spectral_centroid_x_m: np.ndarray
    spectral_centroid_y_m: np.ndarray
    metadata: Mapping[str, Any]


def spectrum_from_arrays(
    wavelengths_m: Sequence[float],
    spectral_energy: Sequence[float],
    *,
    spectral_phase_rad: Sequence[float] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Spectrum:
    """Build a spectrum from already-integrated discrete energy weights."""

    return Spectrum(
        wavelengths_m=np.asarray(wavelengths_m, dtype=float),
        energy_weights=np.asarray(spectral_energy, dtype=float),
        spectral_phase_rad=None if spectral_phase_rad is None else np.asarray(spectral_phase_rad, dtype=float),
        metadata={"spectral_value_interpretation": "integrated_discrete_energy_weights", **dict(metadata or {})},
    ).validated()


def spectrum_from_wavelength_density(
    wavelengths_m: Sequence[float],
    spectral_density: Sequence[float],
    *,
    spectral_phase_rad: Sequence[float] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Spectrum:
    """Integrate a sampled spectral density over wavelength before normalising.

    The absolute density unit may be per metre, per nanometre, detector counts
    per nanometre, or another constant-scaled wavelength-density unit because
    only relative pulse-energy fractions are formed. Non-uniform wavelength
    spacing is handled by trapezoidal quadrature point weights.
    """

    wl = np.asarray(wavelengths_m, dtype=float)
    density = np.asarray(spectral_density, dtype=float)
    if wl.ndim != 1 or density.shape != wl.shape or wl.size < 2:
        raise ValueError("spectral density requires matching 1-D arrays with at least two samples")
    if np.any(~np.isfinite(density)) or np.any(density < 0.0):
        raise ValueError("spectral density must be finite and non-negative")
    widths = _quadrature_cell_widths(wl)
    integrated = density * widths
    return spectrum_from_arrays(
        wl,
        integrated,
        spectral_phase_rad=spectral_phase_rad,
        metadata={
            "spectral_value_interpretation": "wavelength_density_integrated_by_trapezoidal_point_weights",
            "wavelength_grid_nonuniform": bool(not np.allclose(np.diff(wl), np.diff(wl)[0], rtol=1e-9, atol=0.0)),
            **dict(metadata or {}),
        },
    )


def load_spectrum_csv(path: str | Path) -> Spectrum:
    """Load measured spectrum CSV with strict, physically distinct columns.

    Accepted wavelength columns are ``wavelength_m`` or ``wavelength_nm``.
    ``energy_weight`` means an already-integrated per-sample energy weight.
    ``spectral_intensity`` means sampled spectral density versus wavelength and
    is integrated with wavelength-cell widths before normalisation. Optional
    ``spectral_phase_rad`` is preserved. No wavelength or phase is guessed from
    filename metadata.
    """

    array = np.genfromtxt(Path(path), delimiter=",", names=True, dtype=float, encoding="utf-8-sig")
    if array.size == 0 or array.dtype.names is None:
        raise ValueError("spectrum CSV is empty or has no header")
    names = set(array.dtype.names)
    if "wavelength_m" in names:
        wl = np.atleast_1d(np.asarray(array["wavelength_m"], dtype=float))
    elif "wavelength_nm" in names:
        wl = np.atleast_1d(np.asarray(array["wavelength_nm"], dtype=float)) * 1.0e-9
    else:
        raise ValueError("spectrum CSV requires wavelength_m or wavelength_nm")
    phase = np.atleast_1d(np.asarray(array["spectral_phase_rad"], dtype=float)) if "spectral_phase_rad" in names else None
    order = np.argsort(wl)
    wl = wl[order]
    phase = None if phase is None else phase[order]
    source_meta = {"source": str(Path(path)), "data_classification": "supplied_spectrum"}
    if "energy_weight" in names:
        weight = np.atleast_1d(np.asarray(array["energy_weight"], dtype=float))[order]
        return spectrum_from_arrays(wl, weight, spectral_phase_rad=phase, metadata=source_meta)
    if "spectral_intensity" in names:
        density = np.atleast_1d(np.asarray(array["spectral_intensity"], dtype=float))[order]
        return spectrum_from_wavelength_density(wl, density, spectral_phase_rad=phase, metadata=source_meta)
    raise ValueError("spectrum CSV requires energy_weight or spectral_intensity")


def gaussian_transform_limited_spectrum(
    central_wavelength_m: float,
    intensity_fwhm_s: float,
    *,
    samples: int = 21,
    sigma_span: float = 3.5,
) -> Spectrum:
    """Return a transform-limited Gaussian *planning* spectrum.

    This is provided for numerical controls, not as a substitute for the
    measured PHAROS spectrum. The Gaussian time-bandwidth product 0.441 is used
    for intensity FWHM in frequency.
    """

    if central_wavelength_m <= 0.0 or intensity_fwhm_s <= 0.0:
        raise ValueError("central wavelength and pulse duration must be positive")
    if int(samples) < 3 or int(samples) % 2 == 0:
        raise ValueError("samples must be an odd integer >=3")
    nu0 = C0 / float(central_wavelength_m)
    fwhm_nu = 0.441 / float(intensity_fwhm_s)
    sigma_nu = fwhm_nu / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    nu = np.linspace(nu0 - sigma_span * sigma_nu, nu0 + sigma_span * sigma_nu, int(samples))
    if np.any(nu <= 0.0):
        raise ValueError("requested Gaussian spectrum reaches non-positive optical frequency")
    density_nu = np.exp(-0.5 * ((nu - nu0) / sigma_nu) ** 2)
    # nu is uniform, therefore each point's trapezoidal integration weight is
    # proportional to density_nu except the physically half-weighted endpoints.
    nu_widths = _quadrature_cell_widths(nu)
    integrated_energy = density_nu * nu_widths
    wl = C0 / nu
    order = np.argsort(wl)
    return spectrum_from_arrays(
        wl[order],
        integrated_energy[order],
        spectral_phase_rad=np.zeros_like(wl)[order],
        metadata={
            "source": "transform_limited_gaussian_control",
            "measured_spectrum": False,
            "time_bandwidth_product": 0.441,
            "spectral_density_domain": "frequency",
        },
    )


def _centroid(intensity: np.ndarray, x_m: np.ndarray, y_m: np.ndarray) -> tuple[float, float]:
    X, Y = np.meshgrid(x_m, y_m, indexing="xy")
    I = np.maximum(np.asarray(intensity, dtype=float), 0.0)
    total = max(float(np.sum(I)), EPS)
    return float(np.sum(I * X) / total), float(np.sum(I * Y) / total)


def propagate_broadband(
    spectrum: Spectrum,
    propagate_one_wavelength: Callable[[float], SpectralFieldPlane],
    *,
    wavelength_tolerance_m: float = 1.0e-15,
) -> BroadbandResult:
    """Execute the same physical optical model independently at every lambda."""

    spec = spectrum.validated()
    fields: list[SpectralFieldPlane] = []
    reference_x: np.ndarray | None = None
    reference_y: np.ndarray | None = None
    detector: np.ndarray | None = None
    cx: list[float] = []
    cy: list[float] = []
    for wavelength, weight in zip(spec.wavelengths_m, spec.energy_weights):
        field = propagate_one_wavelength(float(wavelength)).validated()
        if abs(field.wavelength_m - float(wavelength)) > wavelength_tolerance_m:
            raise ValueError("single-wavelength solver returned a mismatched wavelength")
        if reference_x is None:
            reference_x = field.x_m
            reference_y = field.y_m
            detector = np.zeros_like(field.intensity, dtype=float)
        elif (
            field.x_m.shape != reference_x.shape
            or field.y_m.shape != reference_y.shape
            or not np.allclose(field.x_m, reference_x, rtol=0.0, atol=1e-15)
            or not np.allclose(field.y_m, reference_y, rtol=0.0, atol=1e-15)
        ):
            raise ValueError(
                "broadband fields must share one declared physical output grid; "
                "resample explicitly before spectral integration"
            )
        detector += float(weight) * field.intensity
        x0, y0 = _centroid(field.intensity, field.x_m, field.y_m)
        cx.append(x0)
        cy.append(y0)
        fields.append(field)
    assert reference_x is not None and reference_y is not None and detector is not None
    return BroadbandResult(
        spectrum=spec,
        fields=tuple(fields),
        detector_integrated_intensity=np.asarray(detector, dtype=float),
        x_m=np.asarray(reference_x, dtype=float),
        y_m=np.asarray(reference_y, dtype=float),
        spectral_centroid_x_m=np.asarray(cx, dtype=float),
        spectral_centroid_y_m=np.asarray(cy, dtype=float),
        metadata={
            "combination_rule": "incoherent_integrated_energy_weighted_intensity_sum_for_slow_detector",
            "spectral_samples": int(spec.wavelengths_m.size),
            "central_wavelength_m": spec.central_wavelength_m,
            "temporal_material_response_modelled": False,
            "nonlinear_propagation_modelled": False,
        },
    )


def apply_polynomial_spectral_phase(
    spectrum: Spectrum,
    *,
    gdd_s2: float = 0.0,
    tod_s3: float = 0.0,
    phase_offset_rad: float = 0.0,
) -> Spectrum:
    """Add phase phi=phi0+GDD/2*domega^2+TOD/6*domega^3 about weighted omega."""

    spec = spectrum.validated()
    omega = TWOPI * C0 / spec.wavelengths_m
    omega0 = float(np.sum(omega * spec.energy_weights))
    dw = omega - omega0
    base = np.zeros_like(dw) if spec.spectral_phase_rad is None else spec.spectral_phase_rad
    phase = base + float(phase_offset_rad) + 0.5 * float(gdd_s2) * dw**2 + (float(tod_s3) / 6.0) * dw**3
    return Spectrum(
        wavelengths_m=spec.wavelengths_m,
        energy_weights=spec.energy_weights,
        spectral_phase_rad=phase,
        metadata={**dict(spec.metadata), "gdd_s2_added": float(gdd_s2), "tod_s3_added": float(tod_s3)},
    ).validated()


def _angular_frequency_widths(wavelengths_m: np.ndarray) -> np.ndarray:
    omega = TWOPI * C0 / np.asarray(wavelengths_m, dtype=float)
    order = np.argsort(omega)
    widths_sorted = _quadrature_cell_widths(omega[order])
    widths = np.empty_like(widths_sorted)
    widths[order] = widths_sorted
    return widths


def temporal_field_at_pixel(
    result: BroadbandResult,
    *,
    iy: int,
    ix: int,
    time_s: np.ndarray,
    component: str = "Ex",
    require_measured_or_declared_phase: bool = True,
) -> np.ndarray:
    """Reconstruct a relative coherent temporal field at one spatial sample.

    Discrete energy fractions are converted back to an angular-frequency
    spectral amplitude using the local frequency-cell widths. The quadrature
    coefficient is therefore proportional to ``sqrt(E_i * delta_omega_i)``,
    not simply ``sqrt(E_i)``. A global normalisation is applied because this is
    a relative waveform. Absolute electric-field calibration requires measured
    pulse energy, spatial normalisation and a solver with an absolute field
    prefactor.
    """

    spec = result.spectrum.validated()
    if require_measured_or_declared_phase and spec.spectral_phase_rad is None:
        raise ValueError("temporal reconstruction requires supplied/declared spectral phase")
    phase = np.zeros_like(spec.wavelengths_m) if spec.spectral_phase_rad is None else spec.spectral_phase_rad
    t = np.asarray(time_s, dtype=float)
    if t.ndim != 1 or np.any(~np.isfinite(t)):
        raise ValueError("time_s must be a finite 1-D axis")
    if not (0 <= int(iy) < result.y_m.size and 0 <= int(ix) < result.x_m.size):
        raise IndexError("requested temporal-field pixel is outside the broadband grid")
    amplitudes: list[complex] = []
    for field in result.fields:
        if component == "Ex":
            value = field.Ex[iy, ix]
        elif component == "Ey" and field.Ey is not None:
            value = field.Ey[iy, ix]
        elif component == "Ez" and field.Ez is not None:
            value = field.Ez[iy, ix]
        else:
            raise ValueError(f"component {component!r} is unavailable")
        amplitudes.append(complex(value))
    delta_omega = _angular_frequency_widths(spec.wavelengths_m)
    quadrature_amplitude = np.sqrt(spec.energy_weights * delta_omega)
    quadrature_amplitude /= max(float(np.linalg.norm(quadrature_amplitude)), EPS)
    a = np.asarray(amplitudes, dtype=np.complex128) * quadrature_amplitude * np.exp(1j * phase)
    omega = TWOPI * C0 / spec.wavelengths_m
    return np.sum(a[:, None] * np.exp(-1j * omega[:, None] * t[None, :]), axis=0)


__all__ = [
    "BroadbandResult",
    "SpectralFieldPlane",
    "Spectrum",
    "apply_polynomial_spectral_phase",
    "gaussian_transform_limited_spectrum",
    "load_spectrum_csv",
    "propagate_broadband",
    "spectrum_from_arrays",
    "spectrum_from_wavelength_density",
    "temporal_field_at_pixel",
]
