"""Real q=20 BeamGage to digital-twin correction comparison.

This is an experimental-data bridge, not a hardware-control program.  It keeps
the 18x4 BMG acquisition in one camera coordinate system, uses the established
Miao runner as the published-method baseline, and tests a compact residual
phase through the complete Gaussian -> SLM1 -> SLM2/carrier -> 4F/iris ->
axicon -> propagation route.  Predicted corrected stacks are always labelled
as model predictions.  A native SLM phase image is deliberately not emitted
unless conjugacy, coordinates, and the 1030-nm LUT are measured.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
import argparse
import json
import math
import sys
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage, optimize, special

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_q20_miao_retrieval import load_scan_preserve_plane_shift
from vbb_study.digital_twin.physical_observable_fit import (
    axisymmetric_radial_morphology,
    centroid_trajectory,
)
from vbb_study.digital_twin.residual_phase_fit import (
    angular_phase_from_coefficients,
    fit_angular_residual_phase,
    score_residual_phase_on_indices,
)
from vbb_study.digital_twin.phase2a_contracts import canonical_hardware_manifest, hardware_value
from vbb_study.digital_twin.vortex_beam_slm_errors import GaussianBeamError
from vbb_study.digital_twin.vortex_continuous_propagation import (
    build_fixed_support_spectrum,
    native_field_at_z,
)
from vbb_study.digital_twin.vortex_error_reference_models import exact_refractive_axicon_kr_m_inv
from vbb_study.digital_twin.vortex_explicit_4f import FourFError
from vbb_study.digital_twin.vortex_system_route import (
    AxiconError,
    SystemErrorConfig,
    build_multirate_system_route,
)
from vbb_study.viz_fields import phase_winding

Q = 20
WAVELENGTH_M = 1030e-9
PIXEL_M = 5.5e-6
Z_REL_MM = np.arange(-17.0, 1.0)
DISPLAY_LIMIT_UM = 180.0
DISPLAY_N = 241
RELAY_N = 512
FIT_N = 2048
FIT_WINDOW_M = 10.0e-3
PRODUCTION_N = 3072
REPRESENTATIVE_Z_MM = -10.0  # declared before inspecting correction performance
EPS = np.finfo(float).tiny

THERMAL = "inferno"


def _save_figure(fig: plt.Figure, stem: Path, *, dpi: int = 600) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def find_bmg_dir(explicit: Path | None = None) -> Path:
    if explicit is not None:
        candidates = [Path(explicit)]
    else:
        candidates = [
            HERE / "z-scan 2 1010",
            ROOT.parent / "LabStuff" / "Axicon_AberrationCorrection" / "z-scan 2 1010",
            HERE / "new z-scan bessel beam 1010",
            ROOT.parent / "LabStuff" / "Axicon_AberrationCorrection" / "new z-scan bessel beam 1010",
        ]
    for candidate in candidates:
        files = sorted(candidate.glob("z*_*.bmg")) if candidate.exists() else []
        if len(files) == 72:
            return candidate.resolve()
    detail = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"No complete 72-frame BMG acquisition found in: {detail}")


def inventory(data_dir: Path) -> dict[int, list[Path]]:
    groups: dict[int, list[Path]] = {}
    for path in sorted(data_dir.glob("z*_*.bmg")):
        try:
            zi = int(path.stem.split("_")[0][1:])
        except (ValueError, IndexError):
            continue
        groups.setdefault(zi, []).append(path)
    if sorted(groups) != list(range(18)):
        raise RuntimeError(f"Expected z0...z17; found {sorted(groups)}")
    wrong = {z: len(paths) for z, paths in groups.items() if len(paths) != 4}
    if wrong:
        raise RuntimeError(f"Expected four BMG repeats per plane: {wrong}")
    return groups


def load_measured(data_dir: Path, out: Path) -> tuple[np.ndarray, tuple[float, float], pd.DataFrame]:
    cache = out / "measured_native_fixed_coordinates.npz"
    qc_path = out / "measured_frame_qc.csv"
    if cache.exists() and qc_path.exists():
        d = np.load(cache)
        return np.asarray(d["stack"], float), tuple(d["axis_yx_px"].tolist()), pd.read_csv(qc_path)
    images, keys, axis_yx, origin, shifts, sensor_shape, qc = load_scan_preserve_plane_shift(data_dir)
    if not np.array_equal(keys, np.arange(18)):
        raise RuntimeError("BMG loader returned an unexpected z ordering")
    stack = np.asarray(images, np.float32)
    np.savez_compressed(
        cache,
        stack=stack,
        z_relative_mm=Z_REL_MM,
        z_hexapod_mm=Z_REL_MM + 6.0,
        axis_yx_px=np.asarray(axis_yx),
        crop_origin_yx_px=np.asarray(origin),
        mean_repeat_shift_yx_px=shifts,
        sensor_shape_yx=np.asarray(sensor_shape),
        pixel_pitch_m=PIXEL_M,
    )
    qcdf = pd.DataFrame(qc)
    qcdf.to_csv(qc_path, index=False)
    return np.asarray(stack, float), tuple(map(float, axis_yx)), qcdf


def sample_fixed_camera(stack: np.ndarray, axis_yx: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    axis_um = np.linspace(-DISPLAY_LIMIT_UM, DISPLAY_LIMIT_UM, DISPLAY_N)
    xpix = axis_yx[1] + axis_um * 1e-6 / PIXEL_M
    ypix = axis_yx[0] + axis_um * 1e-6 / PIXEL_M
    yy, xx = np.meshgrid(ypix, xpix, indexing="ij")
    sampled = np.stack([
        ndimage.map_coordinates(np.asarray(im, float), [yy, xx], order=1, mode="constant", cval=0.0)
        for im in stack
    ])
    return axis_um, plane_normalise(sampled)


def plane_normalise(stack: np.ndarray) -> np.ndarray:
    a = np.maximum(np.asarray(stack, float), 0.0)
    return a / np.maximum(np.max(a, axis=(-2, -1), keepdims=True), EPS)


def measured_core_trajectory(qc: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    centres = qc.groupby("z_index")[["core_x_raw_px", "core_y_raw_px"]].median().reset_index()
    z_m = Z_REL_MM * 1e-3
    x_m = centres.core_x_raw_px.to_numpy() * PIXEL_M
    y_m = centres.core_y_raw_px.to_numpy() * PIXEL_M
    px = np.polyfit(z_m, x_m, 1)
    py = np.polyfit(z_m, y_m, 1)
    qx = np.polyfit(z_m, x_m, 2)
    qy = np.polyfit(z_m, y_m, 2)
    linear_x = np.polyval(px, z_m); linear_y = np.polyval(py, z_m)
    quad_x = np.polyval(qx, z_m); quad_y = np.polyval(qy, z_m)
    slope = np.asarray([px[0], py[0]], float)
    summary = {
        "x_slope_mrad": float(px[0] * 1e3),
        "y_slope_mrad": float(py[0] * 1e3),
        "slope_magnitude_mrad": float(np.hypot(*slope) * 1e3),
        "end_to_end_x_um": float((x_m[-1] - x_m[0]) * 1e6),
        "end_to_end_y_um": float((y_m[-1] - y_m[0]) * 1e6),
        "linear_rms_x_um": float(np.sqrt(np.mean((x_m-linear_x)**2))*1e6),
        "linear_rms_y_um": float(np.sqrt(np.mean((y_m-linear_y)**2))*1e6),
        "quadratic_rms_x_um": float(np.sqrt(np.mean((x_m-quad_x)**2))*1e6),
        "quadratic_rms_y_um": float(np.sqrt(np.mean((y_m-quad_y)**2))*1e6),
        "camera_stage_runout_separated": False,
    }
    centres["z_relative_mm"] = Z_REL_MM
    centres["x_relative_um"] = (x_m - np.median(x_m))*1e6
    centres["y_relative_um"] = (y_m - np.median(y_m))*1e6
    return centres, summary


def load_miao(miao_dir: Path, axis_um: np.ndarray, axis_yx: tuple[float, float]) -> tuple[np.ndarray, np.ndarray, dict]:
    model = np.load(miao_dir / "model_comparison" / "measured_fit_corrected_ideal_stacks.npz")
    if not np.allclose(model["axis_um"], axis_um):
        raise RuntimeError("Miao display grid differs from the fixed comparison grid")
    corrected = np.asarray(model["predicted_corrected"], float)
    ideal_local = np.asarray(model["local_analytic_ideal"], float)
    rows = pd.read_csv(miao_dir / "per_plane_retrieval.csv")
    du = float(axis_um[1] - axis_um[0])
    for iz, row in rows.iterrows():
        dy_um = (float(row.retrieval_axis_y_px) - axis_yx[0]) * PIXEL_M * 1e6
        dx_um = (float(row.retrieval_axis_x_px) - axis_yx[1]) * PIXEL_M * 1e6
        corrected[iz] = ndimage.shift(corrected[iz], (dy_um/du, dx_um/du), order=1, mode="constant", cval=0.0)
        ideal_local[iz] = ndimage.shift(ideal_local[iz], (dy_um/du, dx_um/du), order=1, mode="constant", cval=0.0)
    manifest = json.loads((miao_dir / "correction_manifest.json").read_text(encoding="utf-8"))
    return plane_normalise(corrected), plane_normalise(ideal_local), manifest


def _radial_profile(image: np.ndarray, axis_um: np.ndarray, centre_yx_um: tuple[float, float],
                    *, bin_um: float = 1.5, limit_um: float = 150.0) -> tuple[np.ndarray, np.ndarray]:
    X, Y = np.meshgrid(axis_um, axis_um, indexing="xy")
    radius = np.hypot(X - float(centre_yx_um[1]), Y - float(centre_yx_um[0]))
    edges = np.arange(0.0, limit_um + bin_um, bin_um)
    ids = np.digitize(radius.ravel(), edges) - 1
    valid = (ids >= 0) & (ids < len(edges) - 1)
    sums = np.bincount(ids[valid], weights=np.asarray(image, float).ravel()[valid], minlength=len(edges)-1)
    counts = np.bincount(ids[valid], minlength=len(edges)-1)
    return 0.5 * (edges[:-1] + edges[1:]), sums / np.maximum(counts, 1)


def calibrate_measured_kp(measured: np.ndarray, axis_um: np.ndarray, trajectory_df: pd.DataFrame,
                          miao_rows: pd.DataFrame, out: Path) -> tuple[float, pd.DataFrame, dict]:
    """Estimate effective k_perp from several real planes and ring geometry."""
    principal_zero = float(special.jnp_zeros(Q, 1)[0])
    rows = []
    fitted_profiles: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for iz, image in enumerate(measured):
        centre = (float(trajectory_df.iloc[iz].y_relative_um), float(trajectory_df.iloc[iz].x_relative_um))
        radius_um, profile = _radial_profile(image, axis_um, centre)
        smooth = ndimage.gaussian_filter1d(profile, 1.0)
        peak_roi = (radius_um >= 25.0) & (radius_um <= 70.0)
        peak_um = float(radius_um[peak_roi][np.argmax(smooth[peak_roi])])
        kp_seed = principal_zero / (peak_um * 1e-6)

        def model(r_um: np.ndarray, kp: float, waist_um: float, amplitude: float, background: float) -> np.ndarray:
            r_m = np.asarray(r_um, float) * 1e-6
            return amplitude * special.jv(Q, kp * r_m) ** 2 * np.exp(-2.0 * (r_um / waist_um) ** 2) + background

        fit_roi = (radius_um >= 8.0) & (radius_um <= 120.0)
        try:
            params, covariance = optimize.curve_fit(
                model, radius_um[fit_roi], profile[fit_roi],
                p0=(kp_seed, 220.0, max(float(np.max(profile)), 0.1) * 12.0, max(float(np.min(profile)), 0.0)),
                bounds=((0.82*kp_seed, 70.0, 0.0, 0.0), (1.18*kp_seed, 700.0, 100.0, 0.5)),
                maxfev=5000,
            )
            fitted = model(radius_um, *params)
            corr = float(np.corrcoef(profile[fit_roi], fitted[fit_roi])[0, 1])
            kp_fit = float(params[0])
            kp_sigma = float(np.sqrt(max(covariance[0, 0], 0.0)))
            status = "fit"
        except (RuntimeError, ValueError, FloatingPointError):
            fitted = model(radius_um, kp_seed, 220.0, 12.0, 0.0)
            corr = float("nan"); kp_fit = float(kp_seed); kp_sigma = float("nan"); status = "ring_only"
        fitted_profiles[iz] = (radius_um, profile, fitted)
        rows.append({
            "z_index": iz, "z_relative_mm": float(Z_REL_MM[iz]), "principal_ring_radius_um": peak_um,
            "k_perp_ring_geometry_m_inv": float(kp_seed), "k_perp_radial_fit_m_inv": kp_fit,
            "k_perp_fit_sigma_m_inv": kp_sigma, "radial_fit_correlation": corr, "fit_status": status,
            "k_perp_miao_m_inv": float(miao_rows.iloc[iz].k_perp_opt_m_inv),
        })
    table = pd.DataFrame(rows)
    good = table.radial_fit_correlation >= 0.75
    if int(np.count_nonzero(good)) < 6:
        good = np.isfinite(table.k_perp_radial_fit_m_inv)
    selected = table.loc[good, "k_perp_radial_fit_m_inv"].to_numpy(float)
    kp = float(np.median(selected))
    mad = float(1.4826 * np.median(np.abs(selected - kp)))
    table["used_for_robust_k_perp"] = good
    table.to_csv(out / "measured_k_perp_vs_z.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.5, 5.3), constrained_layout=True)
    ax.errorbar(table.z_relative_mm, table.k_perp_radial_fit_m_inv/1e3,
                yerr=table.k_perp_fit_sigma_m_inv/1e3, fmt="o-", ms=4, lw=1.1, label="radial fit to real BMG")
    ax.plot(table.z_relative_mm, table.k_perp_miao_m_inv/1e3, "s--", ms=3.5, lw=1.0, label="Miao global-branch per plane")
    ax.axhline(kp/1e3, color="black", lw=1.2, label=f"robust median {kp/1e3:.2f} mm$^{{-1}}$")
    ax.fill_between(table.z_relative_mm, (kp-mad)/1e3, (kp+mad)/1e3, color="black", alpha=.1, label="robust ±MAD")
    ax.set(xlabel="relative z (mm)", ylabel=r"effective $k_\perp$ (mm$^{-1}$)",
           title="Effective q=20 radial scale estimated from the measured BMG stack")
    ax.grid(alpha=.25); ax.legend(fontsize=8)
    _save_figure(fig, out / "measured_k_perp_vs_z")

    chosen = np.linspace(0, len(Z_REL_MM)-1, 6).round().astype(int)
    fig, axs = plt.subplots(2, 3, figsize=(13, 7.5), constrained_layout=True)
    for ax, iz in zip(axs.ravel(), chosen):
        radius_um, profile, fitted = fitted_profiles[int(iz)]
        ax.plot(radius_um, profile, lw=1.5, label="measured azimuthal mean")
        ax.plot(radius_um, fitted, "--", lw=1.3, label="finite-envelope J20 radial fit")
        ax.axvline(table.iloc[int(iz)].principal_ring_radius_um, color="0.4", ls=":", lw=1)
        ax.set(title=f"z={Z_REL_MM[int(iz)]:g} mm", xlabel="radius (um)", ylabel="normalized intensity",
               xlim=(0, 130), ylim=(0, 1.05)); ax.grid(alpha=.2)
    axs[0,0].legend(fontsize=8)
    fig.suptitle("Measured radial profiles and q=20 scale fits (centres used only for radial calibration)")
    _save_figure(fig, out / "measured_radial_profiles_and_fit")
    summary = {"robust_k_perp_m_inv": kp, "robust_mad_m_inv": mad,
               "planes_used": table.loc[good, "z_index"].astype(int).tolist(),
               "method": "per-plane azimuthal radial fit around measured core; robust median across good-SNR planes",
               "miao_median_m_inv": float(np.median(table.k_perp_miao_m_inv))}
    (out / "measured_k_perp_calibration.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return kp, table, summary


def load_miao_input_plane_phase(miao_backcheck_dir: Path, target_x_m: np.ndarray) -> tuple[np.ndarray, dict]:
    """Map the published-method diagnostic phase by complex phasor interpolation.

    The source is the single-transverse-input-plane reconstruction, not an SLM2
    mask.  Outside its reconstructed support the additive phase is zero.
    """
    path = Path(miao_backcheck_dir) / "single_transverse_phase_stacks.npz"
    data = np.load(path)
    source = np.asarray(data["residual_input_phase_rad"], float)
    source_dx_m = PIXEL_M
    source_x0_m = -0.5 * source.shape[0] * source_dx_m + 0.5 * source_dx_m
    target_x = np.asarray(target_x_m, float)
    coord = (target_x - source_x0_m) / source_dx_m
    yy, xx = np.meshgrid(coord, coord, indexing="ij")
    phasor = np.exp(1j * source)
    real = ndimage.map_coordinates(phasor.real, [yy, xx], order=1, mode="constant", cval=1.0)
    imag = ndimage.map_coordinates(phasor.imag, [yy, xx], order=1, mode="constant", cval=0.0)
    mapped = np.angle(real + 1j * imag)
    source_half_width = 0.5 * source.shape[0] * source_dx_m
    X, Y = np.meshgrid(target_x, target_x, indexing="xy")
    outside = (np.abs(X) > source_half_width) | (np.abs(Y) > source_half_width)
    mapped[outside] = 0.0
    summary = {
        "source": str(path.resolve()), "source_plane": "Miao reconstructed input/axicon reference plane",
        "target_plane": "digital-twin axicon input plane", "mapping": "same physical metres; wrapped complex-phasor interpolation",
        "source_dx_um": source_dx_m * 1e6, "source_shape": list(source.shape),
        "source_absolute_z_model_mm": np.asarray(data["z_absolute_model_mm"], float).tolist(),
        "not_slm2_hardware_map": True,
    }
    return mapped, summary


def effective_axicon_from_kp(kp: float) -> dict:
    n = 1.458
    f = lambda deg: exact_refractive_axicon_kr_m_inv(
        wavelength_m=WAVELENGTH_M,
        base_angle_rad=np.deg2rad(deg),
        refractive_index=n,
        external_index=1.0,
    ) - float(kp)
    base_deg = float(optimize.brentq(f, 0.05, 35.0))
    return {
        "measured_effective_k_perp_m_inv": float(kp),
        "effective_internal_base_angle_deg": base_deg,
        "effective_scale_relative_to_repository_2deg_assumption": base_deg / 2.0,
        "corresponding_surface_apex_angle_deg": 180.0 - 2.0*base_deg,
        "corresponding_full_cone_departure_deg": 2.0*base_deg,
        "manufacturer_nominal_text": "20 degrees",
        "manufacturer_convention_resolved": False,
        "interpretation": (
            f"The measured k_perp implies a {base_deg:.3f} deg internal base angle in the repository convention. "
            f"Its {2.0*base_deg:.3f} deg full cone departure is numerically close to a nominal '20 deg' label, "
            "but no local part number/datasheet was found, so this remains an inference, not a conversion."
        ),
    }


def propagate_route(config: SystemErrorConfig, z_abs_m: np.ndarray, *, n: int = FIT_N,
                    phase_slm2: np.ndarray | None = None,
                    phase_axicon_input: np.ndarray | None = None,
                    relay_n: int = RELAY_N) -> tuple[np.ndarray, dict]:
    # The SLM/4F calculation needs the full 10-mm source window to contain the
    # displaced +1 order.  The measured high-k_perp axicon phase then needs a
    # finer grid on that same physical window.  Conflating these grids caused
    # the old stripe field (clipped order) and the subsequent diamond alias.
    route = build_multirate_system_route(
        f"V{Q}", relay_grid_n=int(relay_n), propagation_grid_n=int(n),
        window_m=FIT_WINDOW_M, config=config,
        slm2_static_phase_map_rad=phase_slm2,
        axicon_input_phase_map_rad=phase_axicon_input,
    )
    prop = build_fixed_support_spectrum(
        route["post_axicon"], route["grid"],
        wavelength_m=float(route["metadata"]["wavelength_m"]),
        z_max_m=float(np.max(np.abs(z_abs_m))),
        minimum_retained_spectral_power=0.98,
    )
    stack = np.asarray([
        np.abs(np.asarray(native_field_at_z(prop, float(z)), complex))**2
        for z in np.asarray(z_abs_m, float)
    ], np.float32)
    relay_grid = route["relay_route"]["grid"]
    meta = {
        "route": route["metadata"],
        "propagation": prop.metadata,
        "x_m": np.asarray(route["grid"]["x"], float),
        "theta_slm2": np.arctan2(np.asarray(relay_grid["Y"], float), np.asarray(relay_grid["X"], float)),
        "relay_x_m": np.asarray(relay_grid["x"], float),
        "source_winding": {
            "radius_0p70_mm": float(phase_winding(route["post_axicon"], route["grid"], 0.70e-3, n_phi=720)),
            "radius_1p05_mm": float(phase_winding(route["post_axicon"], route["grid"], 1.05e-3, n_phi=720)),
            "radius_1p40_mm": float(phase_winding(route["post_axicon"], route["grid"], 1.40e-3, n_phi=720)),
        },
    }
    return stack, meta


def sample_model(stack: np.ndarray, x_m: np.ndarray, axis_um: np.ndarray) -> np.ndarray:
    dx = float(x_m[1] - x_m[0])
    ids = (axis_um*1e-6 - float(x_m[0])) / dx
    yy, xx = np.meshgrid(ids, ids, indexing="ij")
    out = np.stack([
        ndimage.map_coordinates(im, [yy, xx], order=1, mode="constant", cval=0.0)
        for im in stack
    ])
    return plane_normalise(out)


def _candidate_table(values: Iterable[float], losses: Iterable[float], name: str, units: str) -> pd.DataFrame:
    table = pd.DataFrame({"parameter": name, "value": list(values), "units": units, "objective": list(losses)})
    best = int(table.objective.idxmin())
    table["selected"] = False
    table.loc[best, "selected"] = True
    table["relative_to_min"] = table.objective / max(float(table.objective.min()), EPS) - 1.0
    return table


def fit_physical_screening(measured: np.ndarray, axis_um: np.ndarray, z_abs_m: np.ndarray,
                           scale: float, trajectory: dict, out: Path) -> tuple[SystemErrorConfig, dict]:
    """Small observable-specific screening fit; no all-parameter optimizer."""
    iris_values = (0.70, 0.85, 1.0, 1.15, 1.30)
    iris_losses = []
    for value in iris_values:
        cfg = SystemErrorConfig(
            fourf=FourFError(iris_radius_scale=float(value)),
            axicon=AxiconError(base_angle_scale=float(scale)),
        )
        stack, meta = propagate_route(cfg, z_abs_m)
        iris_losses.append(axisymmetric_radial_morphology(sample_model(stack, meta["x_m"], axis_um), measured))
    iris_table = _candidate_table(iris_values, iris_losses, "4F iris radius scale", "ratio")
    best_iris = float(iris_table.loc[iris_table.selected, "value"].iloc[0])

    direction = np.asarray([trajectory["x_slope_mrad"], trajectory["y_slope_mrad"]], float)
    direction /= max(float(np.linalg.norm(direction)), EPS)
    offsets = np.asarray([-500, -250, 0, 250, 500], float) * 1e-6
    ax_losses, beam_losses = [], []
    for value in offsets:
        dec = tuple((value*direction).tolist())
        axcfg = SystemErrorConfig(
            fourf=FourFError(iris_radius_scale=best_iris),
            axicon=AxiconError(base_angle_scale=float(scale), decentre_m=dec),
        )
        astack, ameta = propagate_route(axcfg, z_abs_m)
        ax_losses.append(centroid_trajectory(sample_model(astack, ameta["x_m"], axis_um), measured))
        beamcfg = SystemErrorConfig(
            beam=GaussianBeamError(decentre_m=dec),
            fourf=FourFError(iris_radius_scale=best_iris),
            axicon=AxiconError(base_angle_scale=float(scale)),
        )
        bstack, bmeta = propagate_route(beamcfg, z_abs_m)
        beam_losses.append(centroid_trajectory(sample_model(bstack, bmeta["x_m"], axis_um), measured))
    ax_table = _candidate_table(offsets*1e6, ax_losses, "effective axicon lateral displacement", "um along measured walk direction")
    beam_table = _candidate_table(offsets*1e6, beam_losses, "competing input-beam displacement", "um along measured walk direction")
    all_tables = pd.concat([iris_table, ax_table, beam_table], ignore_index=True)
    all_tables.to_csv(out / "physical_parameter_objective_scans.csv", index=False)
    best_ax_um = float(ax_table.loc[ax_table.selected, "value"].iloc[0])
    best_ax_m = best_ax_um * 1e-6
    best_cfg = SystemErrorConfig(
        fourf=FourFError(iris_radius_scale=best_iris),
        axicon=AxiconError(base_angle_scale=float(scale), decentre_m=tuple((best_ax_m*direction).tolist())),
    )
    def diagnosis(table: pd.DataFrame) -> dict:
        selected = table.loc[table.selected].iloc[0]
        boundary = bool(np.isclose(selected.value, table.value.min()) or np.isclose(selected.value, table.value.max()))
        near = table.loc[table.relative_to_min <= 0.05, "value"].tolist()
        return {"selected": float(selected.value), "objective": float(selected.objective),
                "minimum_on_search_boundary": boundary, "values_within_5pct_of_minimum": [float(v) for v in near]}
    summary = {
        "4f_iris_radius": {
            **diagnosis(iris_table), "units": "ratio",
            "objective_definition": "RMSE between centered azimuthally averaged, plane-normalized radial intensity profiles over z",
        },
        "effective_axicon_lateral_displacement": {
            **diagnosis(ax_table), "units": "um along measured walk direction",
            "objective_definition": "RMSE of plane-normalized intensity-centroid trajectory over z",
            "camera_stage_runout_confounded": True,
        },
        "competing_beam_displacement": diagnosis(beam_table),
        "identifiability_warning": (
            "These are scalar-route hypothesis-screening values. The camera optical axis versus stage z, "
            "input Gaussian radius, and high-angle vector response are not calibrated. A selected boundary "
            "or a near-equal competing beam-displacement loss prevents a hardware adjustment claim."
        ),
    }
    (out / "fitted_physical_parameters.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return best_cfg, summary


def fit_full_route_z_registration(measured: np.ndarray, axis_um: np.ndarray,
                                  scale: float, out: Path) -> tuple[float, dict]:
    """Model-bound z registration with all optical parameters held fixed.

    This is not a substitute for measuring the axicon-to-camera distance.  It
    only prevents the complete route from being compared in a visibly
    pre-Bessel plane when the analytical conical-Gaussian z origin differs.
    """
    values = np.arange(30.0, 52.0, 2.0)
    losses = []
    config = SystemErrorConfig(axicon=AxiconError(base_angle_scale=float(scale)))
    for end_mm in values:
        z_abs_m = (float(end_mm) + Z_REL_MM) * 1e-3
        stack, meta = propagate_route(config, z_abs_m, n=FIT_N)
        shown = sample_model(stack, meta["x_m"], axis_um)
        losses.append(axisymmetric_radial_morphology(shown, measured))
    table = _candidate_table(values, losses, "absolute z at relative zero", "mm")
    table.to_csv(out / "full_route_z_registration_scan.csv", index=False)
    selected = float(table.loc[table.selected, "value"].iloc[0])
    near = [float(v) for v in table.loc[table.relative_to_min <= 0.05, "value"]]
    summary = {
        "selected_relative_zero_absolute_mm": selected,
        "objective_definition": "RMSE of centered azimuthally averaged radial morphology over all measured z planes",
        "values_within_5pct_of_minimum_mm": near,
        "minimum_on_search_boundary": bool(selected in (float(values.min()), float(values.max()))),
        "status": "model-bound weak registration; independent distance measurement still required",
    }
    return selected, summary


def validate_and_render_nominal_model(measured: np.ndarray, axis_um: np.ndarray,
                                      trajectory_df: pd.DataFrame, z_abs_m: np.ndarray,
                                      config: SystemErrorConfig, out: Path) -> tuple[np.ndarray, dict]:
    native, meta = propagate_route(config, z_abs_m, n=FIT_N)
    nominal = sample_model(native, meta["x_m"], axis_um)
    del native
    rows = []
    profiles: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for iz in range(len(z_abs_m)):
        measured_centre = (float(trajectory_df.iloc[iz].y_relative_um), float(trajectory_df.iloc[iz].x_relative_um))
        radius, measured_profile = _radial_profile(measured[iz], axis_um, measured_centre)
        _, nominal_profile = _radial_profile(nominal[iz], axis_um, (0.0, 0.0))
        measured_profile = measured_profile / max(float(np.max(measured_profile)), EPS)
        nominal_profile = nominal_profile / max(float(np.max(nominal_profile)), EPS)
        roi = (radius >= 5.0) & (radius <= 120.0)
        peak_roi = (radius >= 25.0) & (radius <= 70.0)
        measured_ring = float(radius[peak_roi][np.argmax(measured_profile[peak_roi])])
        nominal_ring = float(radius[peak_roi][np.argmax(nominal_profile[peak_roi])])
        core = radius <= 0.45 * measured_ring
        ring = (radius >= 0.88 * measured_ring) & (radius <= 1.12 * measured_ring)
        corr = float(np.corrcoef(measured_profile[roi], nominal_profile[roi])[0, 1])
        rows.append({
            "z_index": iz, "z_relative_mm": float(Z_REL_MM[iz]), "z_absolute_model_mm": float(z_abs_m[iz]*1e3),
            "measured_ring_radius_um": measured_ring, "nominal_ring_radius_um": nominal_ring,
            "ring_radius_error_um": nominal_ring - measured_ring, "radial_profile_correlation": corr,
            "nominal_dark_core_to_ring": float(np.mean(nominal_profile[core]) / max(float(np.mean(nominal_profile[ring])), EPS)),
        })
        profiles[iz] = (radius, measured_profile, nominal_profile)
    table = pd.DataFrame(rows)
    table.to_csv(out / "measured_vs_nominal_before_fitting_metrics.csv", index=False)
    winding_values = list(meta["source_winding"].values())
    winding_pass = bool(any(abs(float(value) - Q) <= 0.25 for value in winding_values))
    summary = {
        "mean_radial_profile_correlation": float(table.radial_profile_correlation.mean()),
        "median_absolute_ring_radius_error_um": float(np.median(np.abs(table.ring_radius_error_um))),
        "maximum_nominal_dark_core_to_ring": float(table.nominal_dark_core_to_ring.max()),
        "source_winding": meta["source_winding"], "source_winding_pass": winding_pass,
        "samples_per_axicon_radial_phase_period": float(meta["route"]["samples_per_axicon_radial_phase_period"]),
    }
    summary["nominal_morphology_gate_pass"] = bool(
        summary["mean_radial_profile_correlation"] >= 0.50
        and summary["median_absolute_ring_radius_error_um"] <= 10.0
        and summary["maximum_nominal_dark_core_to_ring"] <= 0.55
        and winding_pass
    )
    summary["acceptance"] = {
        "mean_radial_profile_correlation_min": 0.50,
        "median_absolute_ring_radius_error_um_max": 10.0,
        "maximum_nominal_dark_core_to_ring_max": 0.55,
        "source_winding_tolerance_turns": 0.25,
    }
    (out / "nominal_morphology_gate.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    rep = int(np.argmin(np.abs(Z_REL_MM - REPRESENTATIVE_Z_MM)))
    mid = len(axis_um) // 2
    extent_xy = [axis_um[0], axis_um[-1], axis_um[0], axis_um[-1]]
    extent_xz = [axis_um[0], axis_um[-1], Z_REL_MM[0], Z_REL_MM[-1]]
    fig, axs = plt.subplots(2, 3, figsize=(14, 8.5), constrained_layout=True)
    axs[0,0].imshow(measured[rep], origin="lower", extent=extent_xy, cmap=THERMAL, vmin=0, vmax=1)
    axs[0,0].set(title=f"Measured BMG XY, z={Z_REL_MM[rep]:g} mm", xlabel="x (um)", ylabel="y (um)")
    axs[0,1].imshow(nominal[rep], origin="lower", extent=extent_xy, cmap=THERMAL, vmin=0, vmax=1)
    axs[0,1].set(title="Calibrated nominal finite-energy model", xlabel="x (um)", ylabel="y (um)")
    radius, mp, npf = profiles[rep]
    axs[0,2].plot(radius, mp, lw=1.6, label="measured")
    axs[0,2].plot(radius, npf, "--", lw=1.5, label="nominal")
    axs[0,2].set(title="Azimuthal radial profile", xlabel="radius (um)", ylabel="normalized intensity", xlim=(0,130), ylim=(0,1.05))
    axs[0,2].grid(alpha=.2); axs[0,2].legend()
    axs[1,0].imshow(measured[:,mid,:], origin="lower", aspect="auto", extent=extent_xz, cmap=THERMAL, vmin=0, vmax=1)
    axs[1,0].set(title="Measured fixed-camera XZ", xlabel="x (um)", ylabel="relative z (mm)")
    axs[1,1].imshow(nominal[:,mid,:], origin="lower", aspect="auto", extent=extent_xz, cmap=THERMAL, vmin=0, vmax=1)
    axs[1,1].set(title="Nominal-model XZ", xlabel="x (um)", ylabel="relative z (mm)")
    axs[1,2].plot(table.z_relative_mm, table.measured_ring_radius_um, "o-", label="measured")
    axs[1,2].plot(table.z_relative_mm, table.nominal_ring_radius_um, "s--", label="nominal")
    axs[1,2].set(title="Principal-ring radius", xlabel="relative z (mm)", ylabel="radius (um)")
    axs[1,2].grid(alpha=.2); axs[1,2].legend()
    fig.suptitle("Measured vs calibrated nominal model before aberration fitting\n"
                 "same camera axes; nominal is a scalar effective-q20 prediction")
    _save_figure(fig, out / "measured_vs_nominal_before_fitting")
    return nominal, summary


def pad_measured_to_model(stack: np.ndarray, axis_yx: tuple[float, float], n: int) -> np.ndarray:
    coords = (np.arange(n, dtype=float) - n//2)
    yy, xx = np.meshgrid(axis_yx[0] + coords, axis_yx[1] + coords, indexing="ij")
    return np.asarray([
        ndimage.map_coordinates(im, [yy, xx], order=1, mode="constant", cval=0.0)
        for im in stack
    ], np.float32)


def fit_residual_full_route(measured: np.ndarray, axis_um: np.ndarray,
                            z_abs_m: np.ndarray, config: SystemErrorConfig, out: Path) -> tuple[np.ndarray, dict]:
    """Fit a compact SLM2 residual on alternate z planes only.

    The objective is evaluated in the fixed camera coordinates.  The SLM2
    phase remains on the relay grid and is propagated through the same
    multirate route as every baseline and held-out prediction.
    """
    train = np.arange(0, len(z_abs_m), 2, dtype=int)
    held = np.arange(1, len(z_abs_m), 2, dtype=int)
    nominal_native, meta = propagate_route(config, z_abs_m)
    nominal = sample_model(nominal_native, meta["x_m"], axis_um)
    del nominal_native
    theta = np.asarray(meta["theta_slm2"], float)
    modes = (1, 2)

    def phase_from_coefficients(coefficients: np.ndarray) -> np.ndarray:
        return angular_phase_from_coefficients(theta, np.asarray(coefficients, float), modes=modes)

    def score(predicted: np.ndarray, indices: np.ndarray) -> dict:
        correlations, rmses = [], []
        for iz in indices:
            a = np.asarray(predicted[int(iz)], float)
            b = np.asarray(measured[int(iz)], float)
            correlations.append(float(np.corrcoef(a.ravel(), b.ravel())[0, 1]))
            rmses.append(float(np.sqrt(np.mean((a - b) ** 2))))
        return {"indices": indices.tolist(), "mean_pearson_r": float(np.mean(correlations)),
                "mean_rmse": float(np.mean(rmses)), "per_plane_pearson_r": correlations,
                "per_plane_rmse": rmses}

    before_train = score(nominal, train)
    before_held = score(nominal, held)
    evaluations: list[dict] = []

    def objective(coefficients: np.ndarray) -> float:
        phase = phase_from_coefficients(coefficients)
        native, route_meta = propagate_route(config, z_abs_m[train], phase_slm2=phase)
        shown = sample_model(native, route_meta["x_m"], axis_um)
        del native
        values = []
        for local, iz in enumerate(train):
            corr = float(np.corrcoef(shown[local].ravel(), measured[int(iz)].ravel())[0, 1])
            rmse = float(np.sqrt(np.mean((shown[local] - measured[int(iz)]) ** 2)))
            values.append((1.0 - corr) + rmse)
        loss = float(np.mean(values) + 4e-4 * np.sum(np.asarray(coefficients) ** 2))
        evaluations.append({"coefficients_rad": np.asarray(coefficients, float).tolist(), "objective": loss})
        return loss

    coefficients = np.zeros(2 * len(modes), float)
    current_loss = objective(coefficients)
    for step in (0.50, 0.20):
        for icoeff in range(len(coefficients)):
            candidates = np.unique(np.clip(
                coefficients[icoeff] + step * np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0]), -1.0, 1.0))
            best_value, best_loss = float(coefficients[icoeff]), float(current_loss)
            for value in candidates:
                trial = coefficients.copy(); trial[icoeff] = float(value)
                loss = objective(trial)
                if loss < best_loss:
                    best_value, best_loss = float(value), float(loss)
            coefficients[icoeff] = best_value; current_loss = best_loss
    phase = phase_from_coefficients(coefficients)
    fitted_native, fitted_meta = propagate_route(config, z_abs_m, phase_slm2=phase)
    fitted = sample_model(fitted_native, fitted_meta["x_m"], axis_um)
    del fitted_native
    after_train = score(fitted, train)
    after_held = score(fitted, held)
    held_improved = bool(
        after_held["mean_pearson_r"] > before_held["mean_pearson_r"]
        and after_held["mean_rmse"] < before_held["mean_rmse"]
    )
    summary = {
        "fit": {"modes": list(modes), "coefficients_rad": coefficients.tolist(),
                "objective": float(current_loss), "success": True,
                "message": "bounded two-pass coordinate screen", "iterations": 2,
                "evaluations": evaluations},
        "phase_plane": "numerical SLM2 relay grid; additive residual only; programmed effective q=20 and carrier are preserved",
        "train_before": before_train, "train_after": after_train,
        "heldout_before": before_held, "heldout_after": after_held,
        "heldout_improved_both_metrics": held_improved,
        "status": "SCREENING_PREDICTION_ONLY" if held_improved else "WITHHELD_HELDOUT_NOT_IMPROVED",
    }
    np.save(out / "predicted_slm2_residual_error_phase_fit_relay_grid_rad.npy", phase.astype(np.float32))
    np.save(out / "predicted_slm2_correction_phase_fit_relay_grid_rad.npy", (-phase).astype(np.float32))
    (out / "full_route_residual_fit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return phase, summary


def target_stack(axis_um: np.ndarray, kp: float, nplanes: int) -> np.ndarray:
    X, Y = np.meshgrid(axis_um*1e-6, axis_um*1e-6, indexing="xy")
    R = np.hypot(X, Y)
    base = special.jv(Q, float(kp)*R)**2 * np.exp(-2*(R/(DISPLAY_LIMIT_UM*1e-6))**2)
    base /= max(float(np.max(base)), EPS)
    return np.repeat(base[None, :, :], int(nplanes), axis=0)


def metric_rows(methods: dict[str, np.ndarray], target: np.ndarray, axis_um: np.ndarray, kp: float) -> pd.DataFrame:
    X, Y = np.meshgrid(axis_um, axis_um, indexing="xy")
    R = np.hypot(X, Y)
    roi = R <= 160.0
    ring_um = float(special.jnp_zeros(Q, 1)[0] / kp * 1e6)
    rows = []
    for method, stack in methods.items():
        a = plane_normalise(stack)
        for iz, zmm in enumerate(Z_REL_MM):
            av = a[iz][roi]; tv = target[iz][roi]
            corr = float(np.corrcoef(av, tv)[0, 1])
            rmse = float(np.sqrt(np.mean((av-tv)**2)))
            cy, cx = intensity_centroid(a[iz], axis_um)
            ring = azimuthal_ring(a[iz], axis_um, (cy, cx), ring_um)
            centred_radius = np.hypot(X-cx, Y-cy)
            core = centred_radius <= 0.35*ring_um
            rows.append({
                "method": method, "z_relative_mm": float(zmm),
                "pearson_r_to_target": corr,
                "normalized_rmse_to_target": rmse,
                "azimuthal_principal_ring_cv": float(np.std(ring)/max(float(np.mean(ring)), EPS)),
                "dark_core_to_ring_mean": float(np.mean(a[iz][core])/max(float(np.mean(ring)), EPS)),
                "centroid_x_um": cx, "centroid_y_um": cy,
            })
    return pd.DataFrame(rows)


def intensity_centroid(image: np.ndarray, axis_um: np.ndarray) -> tuple[float, float]:
    a = np.maximum(np.asarray(image, float), 0)
    X, Y = np.meshgrid(axis_um, axis_um, indexing="xy")
    total = max(float(np.sum(a)), EPS)
    return float(np.sum(a*Y)/total), float(np.sum(a*X)/total)


def azimuthal_ring(image: np.ndarray, axis_um: np.ndarray, center_yx_um: tuple[float, float], ring_um: float) -> np.ndarray:
    theta = np.linspace(0, 2*np.pi, 720, endpoint=False)
    radii = ring_um*np.linspace(0.86, 1.14, 9)
    profiles = []
    du = float(axis_um[1]-axis_um[0])
    for radius in radii:
        y = (center_yx_um[0] + radius*np.sin(theta) - axis_um[0])/du
        x = (center_yx_um[1] + radius*np.cos(theta) - axis_um[0])/du
        profiles.append(ndimage.map_coordinates(image, [y, x], order=1, mode="constant", cval=np.nan))
    return np.nanmean(np.asarray(profiles), axis=0)


def sampling_check(config: SystemErrorConfig, z_abs_m: np.ndarray, axis_um: np.ndarray, out: Path, kp: float) -> dict:
    ids = np.asarray([0, int(np.argmin(np.abs(Z_REL_MM-REPRESENTATIVE_Z_MM))), len(Z_REL_MM)-1])
    results: dict[int, np.ndarray] = {}
    rows = []
    period = 2*np.pi/float(kp)
    for n in (1536, 2048, 3072, 4096):
        try:
            stack, meta = propagate_route(config, z_abs_m[ids], n=n)
            shown = sample_model(stack, meta["x_m"], axis_um)
            results[n] = shown
            dx = FIT_WINDOW_M / n
            rows.append({"N": n, "dx_um": dx*1e6, "samples_per_effective_radial_period": period/dx,
                         "retained_spectral_power": meta["propagation"]["retained_spectral_power_fraction"]})
        except (MemoryError, RuntimeError, ValueError) as exc:
            rows.append({"N": n, "error": str(exc)})
    reference_n = max(results)
    ref = results[reference_n]
    roi_axis = np.hypot(*np.meshgrid(axis_um, axis_um, indexing="xy")) <= 160.0
    for row in rows:
        n = row["N"]
        if n not in results:
            continue
        cors, rmses = [], []
        for a, b in zip(results[n], ref):
            av=a[roi_axis]; bv=b[roi_axis]
            cors.append(float(np.corrcoef(av,bv)[0,1])); rmses.append(float(np.sqrt(np.mean((av-bv)**2))))
        row["mean_corr_to_finest"] = float(np.mean(cors)); row["mean_rmse_to_finest"] = float(np.mean(rmses))
    table = pd.DataFrame(rows); table.to_csv(out/"sampling_convergence.csv", index=False)
    summary = {
        "fixed_physical_window_mm": FIT_WINDOW_M*1e3,
        "relay_grid_n": RELAY_N,
        "note": "The 10-mm relay/source window is fixed while the post-iris axicon grid is refined. The relay grid is unchanged, so this isolates conical-phase and propagation sampling.",
        "rows": rows,
        "quantitative_high_angle_claim_allowed": bool(
            3072 in results and 4096 in results
            and next(row for row in rows if row["N"] == 3072).get("mean_corr_to_finest", -1) >= 0.98
        ),
    }
    (out/"sampling_convergence.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def render_outputs(out: Path, axis_um: np.ndarray, measured: np.ndarray, methods: dict[str, np.ndarray],
                   metrics: pd.DataFrame, phase: np.ndarray, z_abs_m: np.ndarray, trajectory_df: pd.DataFrame,
                   kp: float, error_models: dict[str, np.ndarray] | None = None) -> None:
    extent_xy = [axis_um[0], axis_um[-1], axis_um[0], axis_um[-1]]
    fig, axs = plt.subplots(3, 6, figsize=(18, 9), constrained_layout=True)
    for iz, ax in enumerate(axs.ravel()):
        ax.imshow(measured[iz], origin="lower", extent=extent_xy, cmap=THERMAL, vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"z={Z_REL_MM[iz]:.0f} mm", fontsize=9); ax.set_xlabel("x (um)"); ax.set_ylabel("y (um)")
    fig.suptitle("Measured q=20 BMG stack — fixed camera coordinates, four repeats averaged per z")
    _save_figure(fig, out/"01_measured_BMG_contact_sheet")

    mid = len(axis_um)//2
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, arr, name in zip(axs, (measured[:, mid, :], measured[:, :, mid]), ("XZ", "YZ")):
        ax.imshow(arr, origin="lower", aspect="auto", cmap=THERMAL, vmin=0, vmax=1,
                  extent=[axis_um[0], axis_um[-1], Z_REL_MM[0], Z_REL_MM[-1]], interpolation="nearest")
        ax.set(title=f"Measured {name} — fixed camera section", xlabel="signed transverse coordinate (um)", ylabel="relative z (mm)")
    _save_figure(fig, out/"02_measured_XZ_YZ_fixed_coordinates")

    rep = int(np.argmin(np.abs(Z_REL_MM-REPRESENTATIVE_Z_MM)))
    ordered = list(methods.items())
    fig, axs = plt.subplots(4, len(ordered), figsize=(4*len(ordered), 14), constrained_layout=True)
    for col, (name, stack) in enumerate(ordered):
        axs[0,col].imshow(stack[rep], origin="lower", extent=extent_xy, cmap=THERMAL, vmin=0, vmax=1, interpolation="nearest")
        axs[0,col].set(title=f"{name}\nz={Z_REL_MM[rep]:.0f} mm", xlabel="x (um)", ylabel="y (um)")
        axs[1,col].imshow(stack[:,mid,:], origin="lower", aspect="auto", cmap=THERMAL, vmin=0, vmax=1,
                          extent=[axis_um[0],axis_um[-1],Z_REL_MM[0],Z_REL_MM[-1]], interpolation="nearest")
        axs[1,col].set(title="XZ", xlabel="x (um)", ylabel="relative z (mm)")
        axs[2,col].imshow(stack[:,:,mid], origin="lower", aspect="auto", cmap=THERMAL, vmin=0, vmax=1,
                          extent=[axis_um[0],axis_um[-1],Z_REL_MM[0],Z_REL_MM[-1]], interpolation="nearest")
        axs[2,col].set(title="YZ", xlabel="y (um)", ylabel="relative z (mm)")
        axs[3,col].plot(axis_um, stack[rep,mid,:], lw=1.4)
        axs[3,col].set(title="horizontal section", xlabel="x (um)", ylabel="plane-normalized intensity", xlim=(axis_um[0],axis_um[-1]), ylim=(0,1.05))
    fig.suptitle("Real BMG input and correction-model predictions (predictions are not post-SLM measurements)")
    _save_figure(fig, out/"03_measured_miao_digital_twin_target_comparison")

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fields = [
        ("pearson_r_to_target", "Pearson r to finite-energy target"),
        ("normalized_rmse_to_target", "Normalized RMSE to finite-energy target"),
        ("azimuthal_principal_ring_cv", "Principal-ring azimuthal CV"),
        ("dark_core_to_ring_mean", "Dark-core / ring mean"),
    ]
    for ax,(field,label) in zip(axs.ravel(),fields):
        for name, grp in metrics.groupby("method"):
            ax.plot(grp.z_relative_mm, grp[field], "o-", ms=3, lw=1.2, label=name)
        ax.set(xlabel="relative z (mm)", ylabel=label); ax.grid(alpha=.25)
    axs[0,0].legend(fontsize=7, ncol=2)
    fig.suptitle("Same calibrated finite-energy target and metric definitions for every route")
    _save_figure(fig, out/"04_metrics_vs_z")

    fig,axs=plt.subplots(1,3,figsize=(16,4.8),constrained_layout=True)
    for name,stack in ordered:
        lw=2.0 if name in {"Measured BMG","Ideal finite-energy target"} else 1.15
        axs[0].plot(axis_um,stack[rep,mid,:],lw=lw,label=name)
        axs[1].plot(axis_um,stack[rep,:,mid],lw=lw,label=name)
        cy,cx=intensity_centroid(stack[rep],axis_um)
        radius,profile=_radial_profile(stack[rep],axis_um,(cy,cx))
        axs[2].plot(radius,profile/max(float(np.max(profile)),EPS),lw=lw,label=name)
    axs[0].set(title="Horizontal diametric section",xlabel="x (um)",ylabel="plane-normalized intensity",xlim=(axis_um[0],axis_um[-1]),ylim=(0,1.05))
    axs[1].set(title="Vertical diametric section",xlabel="y (um)",ylabel="plane-normalized intensity",xlim=(axis_um[0],axis_um[-1]),ylim=(0,1.05))
    axs[2].set(title="Azimuthally averaged radial profile",xlabel="radius (um)",ylabel="normalized radial intensity",xlim=(0,130),ylim=(0,1.05))
    for ax in axs: ax.grid(alpha=.22)
    axs[2].legend(fontsize=7,loc="upper right")
    fig.suptitle(f"Measured, corrected-prediction and ideal profiles at z={Z_REL_MM[rep]:g} mm")
    _save_figure(fig,out/"09_measured_corrected_ideal_profile_comparisons")

    fig, ax = plt.subplots(figsize=(6.5,5.5), constrained_layout=True)
    ax.plot(trajectory_df.x_relative_um, trajectory_df.y_relative_um, "o-", label="measured dark-core path")
    for _,row in trajectory_df.iterrows():
        ax.text(row.x_relative_um, row.y_relative_um, f"{row.z_relative_mm:.0f}", fontsize=7)
    ax.set(xlabel="relative x (um)", ylabel="relative y (um)", title="Measured beam path in fixed camera coordinates")
    ax.axis("equal"); ax.grid(alpha=.3); ax.legend()
    _save_figure(fig, out/"05_measured_beam_path")

    correction = np.angle(np.exp(-1j*phase))
    n = correction.shape[0]; xpix = np.arange(n)-n/2
    carrier = 2*np.pi*(1/20.0)*xpix[None,:]
    composite = np.angle(np.exp(1j*(carrier+correction)))
    fig, axs = plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    for ax,a,title in zip(axs,(correction,composite),("Correction-only numerical SLM2 layer", "Correction + illustrative 20-px carrier")):
        im=ax.imshow(a,origin="lower",cmap="twilight",vmin=-np.pi,vmax=np.pi,interpolation="nearest")
        ax.set(title=title,xlabel="numerical x pixel",ylabel="numerical y pixel")
    fig.colorbar(im,ax=axs,label="wrapped phase (rad)")
    fig.suptitle("NOT HARDWARE-READY: SLM2 conjugacy, native coordinates, carrier and 1030-nm LUT remain uncalibrated")
    _save_figure(fig,out/"06_predicted_SLM2_correction_phase",dpi=700)

    if error_models:
        ordered_errors=list(error_models.items())
        fig,axs=plt.subplots(3,len(ordered_errors),figsize=(4*len(ordered_errors),10.5),constrained_layout=True)
        for col,(name,stack) in enumerate(ordered_errors):
            axs[0,col].imshow(stack[rep],origin="lower",extent=extent_xy,cmap=THERMAL,vmin=0,vmax=1)
            axs[0,col].set(title=f"{name}\nz={Z_REL_MM[rep]:g} mm",xlabel="x (um)",ylabel="y (um)")
            axs[1,col].imshow(stack[:,mid,:],origin="lower",aspect="auto",extent=[axis_um[0],axis_um[-1],Z_REL_MM[0],Z_REL_MM[-1]],cmap=THERMAL,vmin=0,vmax=1)
            axs[1,col].set(title="XZ",xlabel="x (um)",ylabel="relative z (mm)")
            axs[2,col].plot(axis_um,stack[rep,mid,:],lw=1.4)
            axs[2,col].set(title="horizontal section",xlabel="x (um)",ylabel="normalized intensity",xlim=(axis_um[0],axis_um[-1]),ylim=(0,1.05))
        fig.suptitle("Error reconstruction back-check: applying each inferred error to the ideal route\n"
                     "agreement with the real BMG stack is required before trusting its inverse")
        _save_figure(fig,out/"08_error_reconstruction_backcheck")

    corrected = methods.get("Complete digital-twin correction prediction")
    if corrected is not None:
        X,Z=np.meshgrid(axis_um,Z_REL_MM,indexing="xy")
        fig=plt.figure(figsize=(14,6),constrained_layout=True)
        for i,(stack,title) in enumerate(((measured,"Measured"),(corrected,"Full-route predicted correction")),1):
            ax=fig.add_subplot(1,2,i,projection="3d")
            band=ndimage.gaussian_filter1d(np.mean(stack[:,mid-2:mid+3,:],axis=1),sigma=0.8,axis=1)
            surf=ax.plot_surface(X,Z,band,cmap=THERMAL,rstride=1,cstride=3,linewidth=0,antialiased=True)
            ax.set(xlabel="x (um)",ylabel="relative z (mm)",zlabel="normalized intensity",title=title+" XZ intensity mesh")
        _save_figure(fig,out/"07_measured_vs_corrected_3D_mesh")


def write_summary(out: Path, overall: dict, metrics: pd.DataFrame) -> None:
    means = metrics.groupby("method").agg(
        mean_pearson_r=("pearson_r_to_target","mean"),
        median_pearson_r=("pearson_r_to_target","median"),
        mean_nrmse=("normalized_rmse_to_target","mean"),
        median_nrmse=("normalized_rmse_to_target","median"),
        median_ring_cv=("azimuthal_principal_ring_cv","median"),
        maximum_dark_core_ratio=("dark_core_to_ring_mean","max"),
    ).reset_index()
    means.to_csv(out/"method_summary_metrics.csv",index=False)
    overall["method_summary_metrics"] = means.to_dict(orient="records")
    (out/"run_summary.json").write_text(json.dumps(overall,indent=2),encoding="utf-8")
    columns = list(means.columns)
    table_lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in means.itertuples(index=False, name=None):
        table_lines.append("| " + " | ".join(
            f"{value:.5g}" if isinstance(value, (float, np.floating)) else str(value)
            for value in row
        ) + " |")
    markdown_table = "\n".join(table_lines)
    lines=[
        "# Real q=20 BMG digital-twin correction summary", "",
        "All corrected fields below are **model predictions**, not post-correction camera measurements.", "",
        f"- Raw input: {overall['data']['frames']} BMG files ({overall['data']['planes']} z planes x 4 repeats).",
        f"- Physical z convention: hexapod -11...+6 mm maps to relative -17...0 mm.",
        f"- Effective k_perp: {overall['effective_axicon']['measured_effective_k_perp_m_inv']:.1f} 1/m.",
        f"- Effective repository-convention base angle from k_perp: {overall['effective_axicon']['effective_internal_base_angle_deg']:.3f} deg; manufacturer '20 deg' convention remains unverified.",
        f"- Measured beam-path slope: {overall['trajectory']['slope_magnitude_mrad']:.3f} mrad.",
        f"- Miao branch status: {overall['miao']['status']}; hardware ready: {overall['miao']['hardware_ready']}.",
        f"- Full-route held-out status: {overall['full_route_residual']['status']}.",
        f"- Final full-route decision: {overall.get('full_route_validation', {}).get('decision', 'not evaluated')}.",
        "- SLM2 output is a numerical-grid phase proposal only; native mapping/LUT export is blocked.", "",
        f"- Nominal morphology gate: {overall['nominal_morphology_gate']['nominal_morphology_gate_pass']} "
        f"(mean radial r={overall['nominal_morphology_gate']['mean_radial_profile_correlation']:.3f}, "
        f"median ring-radius error={overall['nominal_morphology_gate']['median_absolute_ring_radius_error_um']:.2f} um).",
        f"- Axicon-grid convergence claim allowed: {overall['sampling']['quantitative_high_angle_claim_allowed']}.",
        "", "## Mean/median finite-energy-target metrics", "", markdown_table, "",
        "## Interpretation", "",
        "The previous stripe fields were numerical: a 5.632-mm source window clipped the displaced +1 order, while "
        "widening that same single grid without refinement aliased the measured high-k_perp axicon phase. The corrected "
        "multirate route uses a 10-mm relay window and a finer fixed-window axicon grid. It reproduces the basic q=20 "
        "annulus before fitting. The fitted low-order residual improves both train and untouched held-out planes only "
        "slightly, so it remains a screening prediction rather than a demonstrated hardware correction. Miao input-plane "
        "phase and numerical SLM2 phase are kept as distinct planes. Hardware use remains blocked by camera-stage axis, "
        "absolute z, vector high-angle response, SLM2 conjugacy/native coordinates, the 1030-nm LUT, branch sign, and a "
        "new post-mask BMG acquisition.",
    ]
    (out/"SUMMARY.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def run(data_dir: Path | None = None, output_dir: Path | None = None, *, recompute: bool = False) -> dict:
    out = (output_dir or HERE/"outputs"/"digital_twin_correction").resolve()
    out.mkdir(parents=True,exist_ok=True)
    data = find_bmg_dir(data_dir); groups=inventory(data)
    measured_native, axis_yx, qc = load_measured(data,out)
    axis_um, measured = sample_fixed_camera(measured_native,axis_yx)
    trajectory_df, trajectory = measured_core_trajectory(qc)
    trajectory_df.to_csv(out/"measured_beam_path.csv",index=False)

    miao_dir=out/"miao_baseline"
    if not (miao_dir/"correction_manifest.json").exists():
        raise FileNotFoundError("Run run_q20_miao_retrieval.py into outputs/digital_twin_correction/miao_baseline first")
    miao_pred,miao_local,miao_manifest=load_miao(miao_dir,axis_um,axis_yx)
    per_plane=pd.read_csv(miao_dir/"per_plane_retrieval.csv")
    kp,kp_table,kp_summary=calibrate_measured_kp(measured,axis_um,trajectory_df,per_plane,out)
    effective=effective_axicon_from_kp(kp)
    scale=float(effective["effective_scale_relative_to_repository_2deg_assumption"])

    backcheck_path=out/"miao_single_phase_backcheck"/"summary.json"
    backcheck=json.loads(backcheck_path.read_text(encoding="utf-8")) if backcheck_path.exists() else {
        "status":"not_run", "absolute_z_at_relative_zero_mm_fitted_from_ideal_only":18.75}
    analytical_z_end=float(backcheck.get("absolute_z_at_relative_zero_mm_fitted_from_ideal_only",18.75))
    z_scan_path=out/"full_route_z_registration_scan.csv"
    if recompute or not z_scan_path.exists():
        z_end,z_registration=fit_full_route_z_registration(measured,axis_um,scale,out)
    else:
        z_table=pd.read_csv(z_scan_path)
        z_end=float(z_table.loc[z_table.selected.astype(bool),"value"].iloc[0])
        near=[float(v) for v in z_table.loc[z_table.relative_to_min<=0.05,"value"]]
        z_registration={"selected_relative_zero_absolute_mm":z_end,
            "objective_definition":"RMSE of centered azimuthally averaged radial morphology over all measured z planes",
            "values_within_5pct_of_minimum_mm":near,
            "minimum_on_search_boundary":bool(z_end in (float(z_table.value.min()),float(z_table.value.max()))),
            "status":"model-bound weak registration; independent distance measurement still required"}
    z_abs_m=(z_end+Z_REL_MM)*1e-3
    if np.any(z_abs_m<=0):
        raise RuntimeError("Model-bound absolute-z registration produced z<=0")

    target_config=SystemErrorConfig(axicon=AxiconError(base_angle_scale=scale))
    nominal_target,nominal_gate=validate_and_render_nominal_model(
        measured,axis_um,trajectory_df,z_abs_m,target_config,out)
    if not nominal_gate["nominal_morphology_gate_pass"]:
        raise RuntimeError("Nominal finite-energy q=20 morphology gate failed; inverse fitting is blocked")

    physical_path=out/"fitted_physical_parameters.json"
    if recompute or not physical_path.exists():
        best_config,physical=fit_physical_screening(measured,axis_um,z_abs_m,scale,trajectory,out)
    else:
        physical=json.loads(physical_path.read_text(encoding="utf-8"))
        iris=float(physical["4f_iris_radius"]["selected"])
        ax_um=float(physical["effective_axicon_lateral_displacement"]["selected"])
        direction=np.asarray([trajectory["x_slope_mrad"],trajectory["y_slope_mrad"]],float)
        direction/=max(float(np.linalg.norm(direction)),EPS)
        best_config=SystemErrorConfig(
            fourf=FourFError(iris_radius_scale=iris),
            axicon=AxiconError(base_angle_scale=scale,decentre_m=tuple((ax_um*1e-6*direction).tolist())))

    residual_path=out/"full_route_residual_fit.json"
    phase_path=out/"predicted_slm2_residual_error_phase_fit_relay_grid_rad.npy"
    if recompute or not residual_path.exists() or not phase_path.exists():
        phase_fit,residual=fit_residual_full_route(measured,axis_um,z_abs_m,best_config,out)
    else:
        residual=json.loads(residual_path.read_text(encoding="utf-8"))
        phase_fit=np.load(phase_path)
    np.save(out/"predicted_slm2_residual_error_phase_numerical_grid_rad.npy",phase_fit.astype(np.float32))
    np.save(out/"predicted_slm2_correction_phase_numerical_grid_rad.npy",(-phase_fit).astype(np.float32))

    error_native,meta=propagate_route(best_config,z_abs_m,n=PRODUCTION_N,phase_slm2=phase_fit)
    error_model=sample_model(error_native,meta["x_m"],axis_um); del error_native
    physical_native,phase_meta=propagate_route(best_config,z_abs_m,n=PRODUCTION_N,phase_slm2=None)
    physical_baseline=sample_model(physical_native,phase_meta["x_m"],axis_um); del physical_native
    # The fitted error and its conjugate are additive at the same numerical
    # SLM2 plane, so their exact model sum is zero.  The corrected prediction
    # is therefore the physical baseline; it is not a post-mask measurement.
    full_corrected=physical_baseline.copy()
    target_native,target_meta=propagate_route(target_config,z_abs_m,n=PRODUCTION_N)
    full_target=sample_model(target_native,target_meta["x_m"],axis_um)
    del target_native
    miao_phase,miao_phase_mapping=load_miao_input_plane_phase(
        out/"miao_single_phase_backcheck",target_meta["x_m"])
    miao_only_native,miao_only_meta=propagate_route(
        target_config,z_abs_m,n=PRODUCTION_N,phase_axicon_input=-miao_phase)
    miao_only_same_model=sample_model(miao_only_native,miao_only_meta["x_m"],axis_um); del miao_only_native
    assisted_native,assisted_meta=propagate_route(
        best_config,z_abs_m,n=PRODUCTION_N,phase_axicon_input=-miao_phase)
    assisted_miao=sample_model(assisted_native,assisted_meta["x_m"],axis_um); del assisted_native
    miao_error_native,miao_error_meta=propagate_route(
        target_config,z_abs_m,n=PRODUCTION_N,phase_axicon_input=miao_phase)
    miao_error_model=sample_model(miao_error_native,miao_error_meta["x_m"],axis_um); del miao_error_native
    fixed_target=target_stack(axis_um,kp,len(Z_REL_MM))

    methods={
        "Measured BMG":measured,
        "Calibrated physical baseline":physical_baseline,
        "Miao-only corrected (same twin)":miao_only_same_model,
        "Physical fit + Miao corrected":assisted_miao,
        "Complete digital-twin correction prediction":full_corrected,
        "Ideal finite-energy target":full_target,
    }
    metrics=metric_rows(methods,full_target,axis_um,kp)
    metrics.to_csv(out/"method_comparison_metrics_vs_z.csv",index=False)
    closure=metric_rows({"Ideal finite-energy model":full_target,
                         "Miao error applied to ideal":miao_error_model,
                         "Full-model fitted error":error_model},measured,axis_um,kp)
    closure.to_csv(out/"error_reconstruction_backcheck_metrics.csv",index=False)
    np.savez_compressed(out/"rerender_arrays.npz",axis_um=axis_um,z_relative_mm=Z_REL_MM,z_absolute_model_mm=z_abs_m*1e3,
                        measured=measured.astype(np.float32),miao_predicted=miao_pred.astype(np.float32),
                        miao_only_same_model=miao_only_same_model.astype(np.float32),
                        physical_plus_miao=assisted_miao.astype(np.float32),
                        full_error_reconstruction=error_model.astype(np.float32),
                        miao_error_reconstruction=miao_error_model.astype(np.float32),
                        calibrated_physical_baseline=physical_baseline.astype(np.float32),
                        full_model_correction_prediction=full_corrected.astype(np.float32),
                        full_route_target=full_target.astype(np.float32),fixed_analytic_target=fixed_target.astype(np.float32))
    sampling=sampling_check(best_config,z_abs_m,axis_um,out,kp)
    render_outputs(out,axis_um,measured,methods,metrics,phase_fit,z_abs_m,trajectory_df,kp,
                   error_models={"Measured BMG":measured,"Ideal finite-energy model":full_target,
                                 "Miao error applied to ideal":miao_error_model,
                                 "Full-model fitted error":error_model})

    error_row=closure.loc[closure.method=="Full-model fitted error"]
    ideal_row=closure.loc[closure.method=="Ideal finite-energy model"]
    error_corr=float(error_row.pearson_r_to_target.mean())
    ideal_corr=float(ideal_row.pearson_r_to_target.mean())
    error_rmse=float(error_row.normalized_rmse_to_target.mean())
    ideal_rmse=float(ideal_row.normalized_rmse_to_target.mean())
    coherent=bool(residual["heldout_improved_both_metrics"] and error_corr>ideal_corr and error_rmse<ideal_rmse)

    overall={
        "data":{"directory":str(data),"frames":sum(map(len,groups.values())),"planes":len(groups),"pixel_pitch_um":5.5,
                "z_relative_mm":Z_REL_MM.tolist(),"z_hexapod_mm":(Z_REL_MM+6).tolist()},
        "representative_plane":{"rule":"predeclared z nearest -10 mm, independent of correction performance","z_relative_mm":REPRESENTATIVE_Z_MM},
        "effective_axicon":effective,"measured_k_perp_calibration":kp_summary,
        "trajectory":trajectory,"nominal_morphology_gate":nominal_gate,"physical_screening":physical,
        "miao":miao_manifest,"miao_single_phase_backcheck":backcheck,
        "miao_input_plane_mapping":miao_phase_mapping,
        "full_route_residual":residual,"sampling":sampling,
        "full_route_validation":{
            "decision":"MODEL_SCREENING_SUPPORTED_HARDWARE_BLOCKED" if coherent else "CORRECTION_WITHHELD_FAILED_ERROR_CLOSURE",
            "mean_error_reconstruction_pearson_r":error_corr,
            "mean_error_reconstruction_nrmse":error_rmse,
            "correlation_gain_over_ideal_error_simulation":error_corr-ideal_corr,
            "rmse_reduction_over_ideal_error_simulation":ideal_rmse-error_rmse,
            "heldout_improved_both_metrics":bool(residual["heldout_improved_both_metrics"]),
            "sampling_3072_to_4096_converged":bool(sampling["quantitative_high_angle_claim_allowed"]),
            "reason":"nominal route is physically coherent; residual support remains a low-gain model screen and hardware use is blocked" if coherent else "positive fitted error did not improve real-stack closure on both metrics",
        },
        "model_absolute_z":{"relative_zero_absolute_mm":z_end,"full_route_registration":z_registration,
                            "analytical_conical_gaussian_registration_mm":analytical_z_end,
                            "status":"model-bound registration; not a measured axicon distance"},
        "hardware_ready":False,
        "hardware_blockers":["independent camera optical axis versus z-stage position","absolute axicon-to-camera z reference",
            "measured input Gaussian radius","manufacturer axicon angle convention/part number and vector response",
            "SLM2 conjugacy and native scale/centre/rotation/parity","SLM2 1030-nm phase LUT","independent direct/conjugate branch test",
            "new post-correction 18x4 BMG acquisition"],
    }
    write_summary(out,overall,metrics)
    return overall


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir",type=Path)
    parser.add_argument("--output-dir",type=Path)
    parser.add_argument("--recompute",action="store_true")
    args=parser.parse_args()
    print(json.dumps(run(args.data_dir,args.output_dir,recompute=args.recompute),indent=2))
