"""Diagnose camera-frame walk before fitting q=20 optical aberrations.

This operates entirely on committed rerender arrays from the real 72-frame BMG
run.  It asks a deliberately narrow question: how much of the fixed-camera XZ
hourglass / YZ diagonal is explained by the nearly affine relative beam-camera
trajectory alone?

The affine transform is a nuisance coordinate model, not an optical diagnosis.
Without a reference-beam camera-axis calibration it cannot be assigned uniquely
to stage runout, residual carrier, beam pointing, or optic tilt and must never be
converted into an SLM correction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vbb_study.digital_twin.observation_frame import (
    fit_affine_trajectory,
    ring_chord_branches,
    shift_stack_by_trajectory,
)

EPS = np.finfo(float).tiny
THERMAL = "inferno"


def savefig(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def normalise(stack: np.ndarray) -> np.ndarray:
    a = np.maximum(np.asarray(stack, float), 0.0)
    peak = np.maximum(a.reshape(a.shape[0], -1).max(axis=1), EPS)
    return a / peak[:, None, None]


def radial_profile(image: np.ndarray, axis_um: np.ndarray, *, bin_um: float = 1.5) -> tuple[np.ndarray, np.ndarray]:
    X, Y = np.meshgrid(axis_um, axis_um, indexing="xy")
    R = np.hypot(X, Y)
    edges = np.arange(0.0, min(abs(axis_um[0]), abs(axis_um[-1])) + bin_um, bin_um)
    ids = np.digitize(R.ravel(), edges) - 1
    valid = (ids >= 0) & (ids < len(edges) - 1)
    sums = np.bincount(ids[valid], weights=np.asarray(image, float).ravel()[valid], minlength=len(edges)-1)
    counts = np.bincount(ids[valid], minlength=len(edges)-1)
    return 0.5 * (edges[:-1] + edges[1:]), sums / np.maximum(counts, 1)


def principal_ring_radius(stack: np.ndarray, axis_um: np.ndarray) -> np.ndarray:
    radii = []
    for image in stack:
        r, p = radial_profile(image, axis_um)
        p = ndimage.gaussian_filter1d(p, 0.8)
        roi = (r >= 25.0) & (r <= 70.0)
        radii.append(float(r[roi][np.argmax(p[roi])]))
    return np.asarray(radii, float)


def corr_rmse(a: np.ndarray, b: np.ndarray, axis_um: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X, Y = np.meshgrid(axis_um, axis_um, indexing="xy")
    roi = np.hypot(X, Y) <= 160.0
    corrs, rmses = [], []
    for aa, bb in zip(normalise(a), normalise(b)):
        av, bv = aa[roi], bb[roi]
        corrs.append(float(np.corrcoef(av, bv)[0, 1]))
        rmses.append(float(np.sqrt(np.mean((av - bv) ** 2))))
    return np.asarray(corrs), np.asarray(rmses)


def nearest_local_peak(profile: np.ndarray, axis_um: np.ndarray, predicted_um: float, *, half_window_um: float = 14.0) -> float:
    if not np.isfinite(predicted_um):
        return float("nan")
    mask = np.abs(axis_um - float(predicted_um)) <= float(half_window_um)
    if not np.any(mask):
        return float("nan")
    values = np.asarray(profile, float)[mask]
    coords = np.asarray(axis_um, float)[mask]
    return float(coords[int(np.argmax(values))])


def azimuthal_harmonics(stack: np.ndarray, axis_um: np.ndarray, ring_radius_um: np.ndarray, max_m: int = 6) -> pd.DataFrame:
    du = float(axis_um[1] - axis_um[0])
    theta = np.linspace(0.0, 2*np.pi, 720, endpoint=False)
    rows = []
    for iz, (image, radius) in enumerate(zip(stack, ring_radius_um)):
        radial_samples = []
        for scale in np.linspace(0.90, 1.10, 9):
            r = float(radius) * float(scale)
            y = (r*np.sin(theta) - axis_um[0]) / du
            x = (r*np.cos(theta) - axis_um[0]) / du
            radial_samples.append(ndimage.map_coordinates(image, [y, x], order=1, mode="constant", cval=0.0))
        angular = np.mean(np.asarray(radial_samples), axis=0)
        c0 = max(abs(np.mean(angular)), EPS)
        row = {"z_index": iz, "ring_radius_um": float(radius)}
        for m in range(1, int(max_m)+1):
            cm = np.mean(angular * np.exp(-1j*m*theta))
            row[f"m{m}_relative_amplitude"] = float(abs(cm) / c0)
        rows.append(row)
    return pd.DataFrame(rows)


def run(source_dir: Path, out: Path) -> dict:
    source_dir = Path(source_dir)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    arrays = np.load(source_dir / "rerender_arrays.npz")
    axis_um = np.asarray(arrays["axis_um"], float)
    z_mm = np.asarray(arrays["z_relative_mm"], float)
    measured = normalise(np.asarray(arrays["measured"], float))
    baseline = normalise(np.asarray(arrays["calibrated_physical_baseline"], float))
    target = normalise(np.asarray(arrays["full_route_target"], float))

    path = pd.read_csv(source_dir / "measured_beam_path.csv")
    measured_yx = path[["y_relative_um", "x_relative_um"]].to_numpy(float)
    affine = fit_affine_trajectory(z_mm, measured_yx, centre_fit=True)
    fitted_yx = affine.fitted_yx

    measured_beam_frame = shift_stack_by_trajectory(measured, axis_um, fitted_yx, inverse=True)
    baseline_camera = shift_stack_by_trajectory(baseline, axis_um, fitted_yx)
    target_camera = shift_stack_by_trajectory(target, axis_um, fitted_yx)

    baseline_fixed_r, baseline_fixed_e = corr_rmse(baseline, measured, axis_um)
    baseline_camera_r, baseline_camera_e = corr_rmse(baseline_camera, measured, axis_um)
    baseline_beam_r, baseline_beam_e = corr_rmse(baseline, measured_beam_frame, axis_um)
    target_fixed_r, target_fixed_e = corr_rmse(target, measured, axis_um)
    target_camera_r, target_camera_e = corr_rmse(target_camera, measured, axis_um)
    target_beam_r, target_beam_e = corr_rmse(target, measured_beam_frame, axis_um)

    metric_rows = []
    for name, rr, ee in (
        ("baseline_vs_measured_fixed_camera", baseline_fixed_r, baseline_fixed_e),
        ("baseline_shifted_by_affine_walk_vs_measured", baseline_camera_r, baseline_camera_e),
        ("baseline_vs_affine_detrended_measured", baseline_beam_r, baseline_beam_e),
        ("target_vs_measured_fixed_camera", target_fixed_r, target_fixed_e),
        ("target_shifted_by_affine_walk_vs_measured", target_camera_r, target_camera_e),
        ("target_vs_affine_detrended_measured", target_beam_r, target_beam_e),
    ):
        for iz in range(len(z_mm)):
            metric_rows.append({"comparison": name, "z_index": iz, "z_relative_mm": float(z_mm[iz]),
                                "pearson_r": float(rr[iz]), "nrmse": float(ee[iz])})
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out / "observation_frame_metrics.csv", index=False)

    # Quantify whether the fixed-axis hourglass is simply the chord geometry of
    # a translated, nearly constant-radius annulus.
    ring_radius = principal_ring_radius(target, axis_um)
    xbranches, ybranches = ring_chord_branches(ring_radius, fitted_yx)
    mid = len(axis_um) // 2
    xz = measured[:, mid, :]
    yz = measured[:, :, mid]
    chord_rows = []
    for iz in range(len(z_mm)):
        obs_x = [nearest_local_peak(xz[iz], axis_um, value) for value in xbranches[iz]]
        obs_y = [nearest_local_peak(yz[iz], axis_um, value) for value in ybranches[iz]]
        chord_rows.append({
            "z_index": iz, "z_relative_mm": float(z_mm[iz]), "ring_radius_um": float(ring_radius[iz]),
            "affine_centre_y_um": float(fitted_yx[iz,0]), "affine_centre_x_um": float(fitted_yx[iz,1]),
            "predicted_x_negative_um": float(xbranches[iz,0]), "predicted_x_positive_um": float(xbranches[iz,1]),
            "observed_x_negative_um": obs_x[0], "observed_x_positive_um": obs_x[1],
            "predicted_y_negative_um": float(ybranches[iz,0]), "predicted_y_positive_um": float(ybranches[iz,1]),
            "observed_y_negative_um": obs_y[0], "observed_y_positive_um": obs_y[1],
        })
    chord = pd.DataFrame(chord_rows)
    chord.to_csv(out / "hourglass_chord_test.csv", index=False)
    xerr = np.concatenate([
        np.abs(chord.observed_x_negative_um - chord.predicted_x_negative_um).to_numpy(float),
        np.abs(chord.observed_x_positive_um - chord.predicted_x_positive_um).to_numpy(float),
    ])
    yerr = np.concatenate([
        np.abs(chord.observed_y_negative_um - chord.predicted_y_negative_um).to_numpy(float),
        np.abs(chord.observed_y_positive_um - chord.predicted_y_positive_um).to_numpy(float),
    ])
    x_mae = float(np.nanmean(xerr)); y_mae = float(np.nanmean(yerr))

    # Residual azimuthal content after removing only the affine observation walk.
    harmonics = azimuthal_harmonics(measured_beam_frame, axis_um, ring_radius)
    harmonics["z_relative_mm"] = z_mm
    harmonics.to_csv(out / "beam_frame_azimuthal_harmonics.csv", index=False)

    # Figure 1: measured trajectory and the part deliberately classified as an
    # unresolved observation-frame nuisance.
    fig, axs = plt.subplots(2, 2, figsize=(12, 7.5), constrained_layout=True)
    labels = ((0, "y"), (1, "x"))
    for col, label in labels:
        axs[0,col].plot(z_mm, measured_yx[:,col], "o", label="measured relative core")
        axs[0,col].plot(z_mm, fitted_yx[:,col], "-", lw=2, label="affine component")
        axs[0,col].set(xlabel="relative z (mm)", ylabel=f"{label} position (um)", title=f"{label}(z) trajectory")
        axs[0,col].grid(alpha=.25); axs[0,col].legend(fontsize=8)
        axs[1,col].plot(z_mm, measured_yx[:,col]-fitted_yx[:,col], "o-")
        axs[1,col].axhline(0, color="0.3", lw=1)
        axs[1,col].set(xlabel="relative z (mm)", ylabel=f"residual {label} (um)", title=f"non-affine residual; RMS={affine.rms_residual_yx[col]:.2f} um")
        axs[1,col].grid(alpha=.25)
    fig.suptitle("Measured q=20 relative beam/camera trajectory\nAffine part is not assigned to stage or optics without a reference-axis scan")
    savefig(fig, out / "01_measured_affine_trajectory")

    # Figure 2: direct geometric test of the hourglass/diagonal morphology.
    extent = [axis_um[0], axis_um[-1], z_mm[0], z_mm[-1]]
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.8), constrained_layout=True, sharey=True)
    axs[0].imshow(xz, origin="lower", aspect="auto", extent=extent, cmap=THERMAL, vmin=0, vmax=1)
    axs[0].plot(xbranches[:,0], z_mm, "c--", lw=1.8, label="constant-ring chord prediction")
    axs[0].plot(xbranches[:,1], z_mm, "c--", lw=1.8)
    axs[0].set(title=f"Measured fixed-camera XZ | branch MAE {x_mae:.1f} um", xlabel="x (um)", ylabel="relative z (mm)")
    axs[0].legend(fontsize=8)
    axs[1].imshow(yz, origin="lower", aspect="auto", extent=extent, cmap=THERMAL, vmin=0, vmax=1)
    axs[1].plot(ybranches[:,0], z_mm, "c--", lw=1.8, label="constant-ring chord prediction")
    axs[1].plot(ybranches[:,1], z_mm, "c--", lw=1.8)
    axs[1].set(title=f"Measured fixed-camera YZ | branch MAE {y_mae:.1f} um", xlabel="y (um)")
    axs[1].legend(fontsize=8)
    fig.suptitle("The apparent hourglass is tested as fixed-axis slicing of a translated q=20 annulus")
    savefig(fig, out / "02_hourglass_chord_geometry")

    # Figure 3: show what remains once only the affine coordinate drift is removed.
    bxz, byz = measured_beam_frame[:,mid,:], measured_beam_frame[:,:,mid]
    txz, tyz = target[:,mid,:], target[:,:,mid]
    fig, axs = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True, sharey=True)
    for row, (raw, detrended, model, lab) in enumerate(((xz, bxz, txz, "XZ"), (yz, byz, tyz, "YZ"))):
        for col, (arr, title) in enumerate(((raw, "measured fixed camera"), (detrended, "measured after affine walk removal"), (model, "calibrated finite-energy target"))):
            axs[row,col].imshow(arr, origin="lower", aspect="auto", extent=extent, cmap=THERMAL, vmin=0, vmax=1)
            axs[row,col].set(title=f"{lab}: {title}", xlabel=("x (um)" if lab=="XZ" else "y (um)"), ylabel="relative z (mm)")
    fig.suptitle("Separate observation-frame walk from residual optical morphology")
    savefig(fig, out / "03_fixed_camera_vs_beam_frame")

    # Figure 4: quantitative effect of the observation transform.
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    axs[0].plot(z_mm, baseline_fixed_r, "o-", label="baseline vs measured, fixed camera")
    axs[0].plot(z_mm, baseline_camera_r, "o-", label="baseline + affine walk vs measured")
    axs[0].plot(z_mm, baseline_beam_r, "s--", label="baseline vs detrended measured")
    axs[0].set(xlabel="relative z (mm)", ylabel="Pearson r", title="Morphology agreement")
    axs[0].grid(alpha=.25); axs[0].legend(fontsize=8)
    axs[1].plot(z_mm, baseline_fixed_e, "o-", label="fixed camera")
    axs[1].plot(z_mm, baseline_camera_e, "o-", label="affine observation transform")
    axs[1].plot(z_mm, baseline_beam_e, "s--", label="detrended measured")
    axs[1].set(xlabel="relative z (mm)", ylabel="normalized RMSE", title="Intensity error")
    axs[1].grid(alpha=.25); axs[1].legend(fontsize=8)
    fig.suptitle("Do not ask the optical residual phase to fit unresolved camera/beam walk")
    savefig(fig, out / "04_observation_frame_metric_effect")

    # Figure 5: residual angular structure in the approximately beam-following frame.
    fig, ax = plt.subplots(figsize=(9.5, 5.5), constrained_layout=True)
    for m in range(1, 7):
        ax.plot(z_mm, harmonics[f"m{m}_relative_amplitude"], "o-", ms=3, label=f"m={m}")
    ax.set(xlabel="relative z (mm)", ylabel="|angular Fourier coefficient| / DC",
           title="Measured principal-ring azimuthal content after affine walk removal")
    ax.grid(alpha=.25); ax.legend(ncol=3, fontsize=8)
    savefig(fig, out / "05_beam_frame_azimuthal_harmonics")

    def mean(values): return float(np.nanmean(np.asarray(values, float)))
    y_meas = measured_yx[:,0]
    x_meas = measured_yx[:,1]
    yfit = fitted_yx[:,0]
    xfit = fitted_yx[:,1]
    def r2(obs, pred):
        return float(1.0 - np.sum((obs-pred)**2) / max(np.sum((obs-np.mean(obs))**2), EPS))
    summary = {
        "status": "OBSERVATION_FRAME_DIAGNOSTIC_NOT_OPTICAL_DIAGNOSIS",
        "affine_trajectory": {
            "y_slope_um_per_mm": float(affine.slope_yx_per_z[0]),
            "x_slope_um_per_mm": float(affine.slope_yx_per_z[1]),
            "y_rms_non_affine_um": float(affine.rms_residual_yx[0]),
            "x_rms_non_affine_um": float(affine.rms_residual_yx[1]),
            "y_variance_explained_r2": r2(y_meas, yfit),
            "x_variance_explained_r2": r2(x_meas, xfit),
            "ownership": "unresolved: camera/stage runout versus optical beam walk versus mixture",
        },
        "hourglass_chord_test": {
            "x_branch_mean_absolute_error_um": x_mae,
            "y_branch_mean_absolute_error_um": y_mae,
            "interpretation": "fixed-axis XZ/YZ morphology is largely predicted by translating a near-constant-radius annulus along the affine measured trajectory" if max(x_mae, y_mae) <= 10.0 else "affine translated-ring geometry alone is insufficient",
        },
        "baseline_agreement": {
            "fixed_camera_mean_r": mean(baseline_fixed_r),
            "affine_shifted_model_mean_r": mean(baseline_camera_r),
            "affine_detrended_measured_mean_r": mean(baseline_beam_r),
            "fixed_camera_mean_nrmse": mean(baseline_fixed_e),
            "affine_shifted_model_mean_nrmse": mean(baseline_camera_e),
            "affine_detrended_measured_mean_nrmse": mean(baseline_beam_e),
        },
        "target_agreement": {
            "fixed_camera_mean_r": mean(target_fixed_r),
            "affine_shifted_target_mean_r": mean(target_camera_r),
            "affine_detrended_measured_mean_r": mean(target_beam_r),
        },
        "next_inverse_contract": (
            "Fit optical/system morphology in the affine-detrended beam-following frame; keep the affine trajectory as a separate nuisance parameter until a reference-beam camera-axis-vs-z calibration resolves its physical ownership."
        ),
    }
    (out / "observation_frame_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path,
                        default=ROOT / "notebooks" / "experimental" / "axicon_aberration_correction" / "outputs" / "digital_twin_correction")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "outputs" / "validation" / "q20_observation_frame")
    args = parser.parse_args()
    run(args.source_dir, args.out)


if __name__ == "__main__":
    main()
