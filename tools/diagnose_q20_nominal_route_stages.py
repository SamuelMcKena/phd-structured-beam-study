"""Diagnose the zero-error q=20 source route before inverse fitting.

This deliberately separates three questions which had been conflated in the
first real-BMG bridge:

1. Does the explicit carrier/4F route remove the carrier and preserve winding?
2. Does the canonical scalar ``V20`` surrogate remain rotationally symmetric?
3. Does a finite Gaussian with the calibrated conical phase form an annulus?

The real bench convention is ell_SLM1=+10, ell_SLM2=-10, effective q=20.  A
single scalar V20 field is therefore only an effective-channel surrogate, not
a Jones model of the two polarization-selective SLM interactions.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vbb_study.digital_twin.vortex_continuous_propagation import (
    build_fixed_support_spectrum,
    native_field_at_z,
)
from vbb_study.digital_twin.vortex_system_route import (
    AxiconError,
    SystemErrorConfig,
    build_system_route,
)
from vbb_study.equations.fields import make_xy_grid
from vbb_study.viz_fields import phase_winding


EPS = np.finfo(float).tiny
TWOPI = 2.0 * np.pi


def _normalised_intensity(field: np.ndarray) -> np.ndarray:
    intensity = np.abs(np.asarray(field, dtype=np.complex128)) ** 2
    return intensity / max(float(np.max(intensity)), EPS)


def _intensity_correlation(a: np.ndarray, b: np.ndarray) -> float:
    aa = _normalised_intensity(a).ravel()
    bb = _normalised_intensity(b).ravel()
    aa -= float(np.mean(aa))
    bb -= float(np.mean(bb))
    return float(np.dot(aa, bb) / max(np.linalg.norm(aa) * np.linalg.norm(bb), EPS))


def _spectral_centroid_cpm(field: np.ndarray, dx_m: float) -> tuple[float, float]:
    power = np.abs(np.fft.fftshift(np.fft.fft2(np.asarray(field, complex)))) ** 2
    freq = np.fft.fftshift(np.fft.fftfreq(power.shape[0], d=float(dx_m)))
    fy, fx = np.meshgrid(freq, freq, indexing="ij")
    total = max(float(np.sum(power)), EPS)
    return float(np.sum(power * fx) / total), float(np.sum(power * fy) / total)


def _anisotropy(field: np.ndarray) -> float:
    intensity = _normalised_intensity(field)
    return float(1.0 - _intensity_correlation(intensity, np.rot90(intensity)))


def _image_correlation(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, float).ravel().copy()
    bb = np.asarray(b, float).ravel().copy()
    aa -= float(np.mean(aa)); bb -= float(np.mean(bb))
    return float(np.dot(aa, bb) / max(np.linalg.norm(aa) * np.linalg.norm(bb), EPS))


def _image_anisotropy(image: np.ndarray) -> float:
    return float(1.0 - _image_correlation(image, np.rot90(image)))


def _propagate(source: np.ndarray, grid: dict, wavelength_m: float, z_m: np.ndarray) -> list[np.ndarray]:
    support = build_fixed_support_spectrum(
        np.asarray(source, complex), grid, wavelength_m=float(wavelength_m),
        z_max_m=float(np.max(np.abs(z_m))), minimum_retained_spectral_power=0.98,
    )
    images = []
    for z in z_m:
        field = np.asarray(native_field_at_z(support, float(z)), complex)
        images.append(_normalised_intensity(field).astype(np.float32))
    return images


def _fourier_resample(field: np.ndarray, output_n: int) -> np.ndarray:
    """Band-limited periodic resampling for a fixed physical source window."""
    source = np.asarray(field, complex)
    input_n = int(source.shape[0])
    if source.shape != (input_n, input_n) or output_n < input_n:
        raise ValueError("expected a square field and output_n >= input_n")
    if output_n == input_n:
        return source.copy()
    spectrum = np.fft.fftshift(np.fft.fft2(source))
    padded = np.zeros((output_n, output_n), dtype=np.complex128)
    start = (output_n - input_n) // 2
    padded[start:start + input_n, start:start + input_n] = spectrum
    return np.fft.ifft2(np.fft.ifftshift(padded)) * (float(output_n) / input_n) ** 2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-n", type=int, default=512)
    parser.add_argument("--propagation-n", type=int, default=None)
    parser.add_argument("--window-mm", type=float, default=5.632)
    parser.add_argument("--axicon-scale", type=float, default=5.32935)
    parser.add_argument("--out", type=Path, default=Path("outputs/validation/q20_nominal_route"))
    args = parser.parse_args()

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg = SystemErrorConfig(axicon=AxiconError(base_angle_scale=float(args.axicon_scale)))
    route_v20 = build_system_route(
        "V20", grid_n=int(args.grid_n), window_m=float(args.window_mm) * 1e-3, config=cfg,
    )
    route_b0 = build_system_route(
        "B0", grid_n=int(args.grid_n), window_m=float(args.window_mm) * 1e-3, config=cfg,
    )
    grid = route_v20["grid"]
    theta = np.arctan2(np.asarray(grid["Y"], float), np.asarray(grid["X"], float))
    propagation_n = int(args.propagation_n or args.grid_n)
    fine_grid = (
        grid if propagation_n == int(args.grid_n)
        else make_xy_grid(propagation_n, float(args.window_mm) * 1e-3 / propagation_n)
    )
    fine_theta = np.arctan2(np.asarray(fine_grid["Y"], float), np.asarray(fine_grid["X"], float))
    fine_r = np.hypot(np.asarray(fine_grid["X"], float), np.asarray(fine_grid["Y"], float))
    kp = float(route_v20["metadata"]["axicon"]["exact_kr_m_inv"])
    axicon_t = np.exp(-1j * kp * fine_r)
    v20_pre_axicon = _fourier_resample(route_v20["field_on_axicon_plane"], propagation_n)
    b0_pre_axicon = _fourier_resample(route_b0["field_on_axicon_plane"], propagation_n)
    input_beam_fine = _fourier_resample(route_b0["input_beam"], propagation_n)
    effective_pre_axicon = b0_pre_axicon * np.exp(1j * 20.0 * fine_theta)
    effective_post_axicon = effective_pre_axicon * axicon_t
    direct_post_axicon = input_beam_fine * np.exp(1j * 20.0 * fine_theta) * axicon_t
    scalar_v20_post_axicon = v20_pre_axicon * axicon_t

    radii_m = (0.35e-3, 0.70e-3, 1.05e-3)
    stage_names = (
        "input_beam", "post_slm1", "post_slm2", "fourier_plane_before_iris",
        "post_4f_selected_order", "field_on_axicon_plane", "post_axicon",
    )
    rows: list[dict] = []
    for name in stage_names:
        field = np.asarray(route_v20[name], complex)
        fx, fy = _spectral_centroid_cpm(field, float(grid["dx"]))
        rows.append({
            "stage": name,
            "spectral_centroid_fx_cpm": fx,
            "spectral_centroid_fy_cpm": fy,
            "rot90_intensity_anisotropy": _anisotropy(field),
            "winding_r0p35mm": phase_winding(field, grid, radii_m[0], n_phi=720),
            "winding_r0p70mm": phase_winding(field, grid, radii_m[1], n_phi=720),
            "winding_r1p05mm": phase_winding(field, grid, radii_m[2], n_phi=720),
        })

    z_mm = np.asarray([2.0, 5.0, 10.0, 15.0, 18.0, 20.0, 25.0, 30.0])
    propagated = {
        "scalar_V20_before_4F": _propagate(scalar_v20_post_axicon, fine_grid, route_v20["metadata"]["wavelength_m"], z_mm * 1e-3),
        "effective_q20_at_axicon": _propagate(effective_post_axicon, fine_grid, route_v20["metadata"]["wavelength_m"], z_mm * 1e-3),
        "direct_gaussian_q20": _propagate(direct_post_axicon, fine_grid, route_v20["metadata"]["wavelength_m"], z_mm * 1e-3),
    }
    propagation_rows = []
    for iz, z in enumerate(z_mm):
        reference = propagated["direct_gaussian_q20"][iz]
        for label, fields in propagated.items():
            propagation_rows.append({
                "route": label,
                "z_mm": float(z),
                "rot90_intensity_anisotropy": _image_anisotropy(fields[iz]),
                "intensity_correlation_to_direct_gaussian_q20": _image_correlation(fields[iz], reference),
            })

    report = {
        "status": "diagnostic_only",
        "real_bench_charge_convention": {"ell_slm1": 10, "ell_slm2": -10, "effective_q": 20},
        "scalar_route_scope": "effective-channel surrogate; not the sequential two-polarisation Jones bench",
        "grid_n": int(args.grid_n),
        "propagation_n": propagation_n,
        "window_mm": float(args.window_mm),
        "dx_um": float(grid["dx"]) * 1e6,
        "axicon_scale": float(args.axicon_scale),
        "selected_order_fraction_V20": float(route_v20["metadata"]["fourf"]["iris_selected_power_fraction"]),
        "selected_order_fraction_B0": float(route_b0["metadata"]["fourf"]["iris_selected_power_fraction"]),
        "stage_diagnostics": rows,
        "propagation_diagnostics": propagation_rows,
    }
    (out / "nominal_route_diagnostics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    extent_mm = np.asarray([grid["x"][0], grid["x"][-1], grid["x"][0], grid["x"][-1]]) * 1e3
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    camera_preview = propagated["scalar_V20_before_4F"][4]
    if camera_preview.shape != route_v20["input_beam"].shape:
        stride = max(1, camera_preview.shape[0] // route_v20["input_beam"].shape[0])
        camera_preview = camera_preview[::stride, ::stride]
    plane_fields = [route_v20[name] for name in stage_names] + [camera_preview]
    plane_titles = list(stage_names) + ["camera z=18 mm"]
    for ax, field, title in zip(axes.ravel(), plane_fields, plane_titles):
        ax.imshow(_normalised_intensity(field), origin="lower", extent=extent_mm, cmap="inferno", vmin=0, vmax=1)
        ax.set(title=title, xlabel="x (mm)", ylabel="y (mm)")
    fig.suptitle("Zero-error scalar V20 route, plane by plane (diagnostic only)")
    fig.savefig(out / "nominal_route_plane_by_plane.png", dpi=400)
    fig.savefig(out / "nominal_route_plane_by_plane.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(len(propagated), len(z_mm), figsize=(24, 9), constrained_layout=True)
    for iy, (label, fields) in enumerate(propagated.items()):
        for ix, (z, field) in enumerate(zip(z_mm, fields)):
            axes[iy, ix].imshow(field, origin="lower", extent=extent_mm, cmap="inferno", vmin=0, vmax=1)
            axes[iy, ix].set_title(f"{z:g} mm")
            if ix == 0:
                axes[iy, ix].set_ylabel(label.replace("_", " "))
            axes[iy, ix].set_xticks([]); axes[iy, ix].set_yticks([])
    fig.suptitle("Finite-energy q=20 propagation: route surrogates before inverse fitting")
    fig.savefig(out / "nominal_q20_route_comparison.png", dpi=400)
    fig.savefig(out / "nominal_q20_route_comparison.pdf")
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
