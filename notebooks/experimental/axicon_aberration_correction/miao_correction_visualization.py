"""Honest high-resolution visual diagnostics for the Miao retrieval.

The corrected stack is the intensity predicted after removing the retrieved
angular phase from each plane.  It is not a propagated post-SLM measurement.
The ideal is the local analytic q-order Bessel-Gauss mode evaluated at the
globally selected k_perp for that plane; no unmeasured nominal k_perp is used.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from miao_full_retrieval import modal_basis, phase_only_corrected_coefficients


EPS = 1e-12


def _normalise(a):
    a = np.asarray(a, float)
    return a / max(float(np.max(a)), EPS)


def _similarity(a, b, mask):
    av, bv = np.asarray(a)[mask], np.asarray(b)[mask]
    corr = float(np.corrcoef(av, bv)[0, 1])
    rmse = float(np.sqrt(np.mean((av-bv)**2)))
    return corr, rmse


def _make_stacks(images, retrievals, pixel_pitch_m, q, *, limit_um=180., size=241):
    axis_um = np.linspace(-limit_um, limit_um, int(size))
    X_um, Y_um = np.meshgrid(axis_um, axis_um, indexing="xy")
    X, Y = X_um*1e-6, Y_um*1e-6
    R, phi = np.hypot(X, Y), np.arctan2(Y, X)
    envelope = np.exp(-(R/(limit_um*1e-6))**2)
    measured, fitted, corrected, ideal = [], [], [], []
    for image, retrieval in zip(images, retrievals):
        yy = retrieval.center_y_px + Y/pixel_pitch_m
        xx = retrieval.center_x_px + X/pixel_pitch_m
        measured.append(_normalise(ndimage.map_coordinates(
            np.asarray(image, float), [yy, xx], order=1,
            mode="constant", cval=0.0)))
        basis = modal_basis(q, retrieval.m_values, retrieval.k_perp_m_inv,
                            R.ravel(), phi.ravel())
        fit_field = (basis @ retrieval.coeffs).reshape(R.shape)*envelope
        corrected_coeffs = phase_only_corrected_coefficients(
            retrieval.coeffs, retrieval.m_values)
        corrected_field = (basis @ corrected_coeffs).reshape(R.shape)*envelope
        ideal_basis = modal_basis(q, np.asarray([0]), retrieval.k_perp_m_inv,
                                  R.ravel(), phi.ravel())[:, 0].reshape(R.shape)
        fitted.append(_normalise(np.abs(fit_field)**2))
        corrected.append(_normalise(np.abs(corrected_field)**2))
        ideal.append(_normalise(np.abs(ideal_basis*envelope)**2))
    return (axis_um, np.stack(measured), np.stack(fitted),
            np.stack(corrected), np.stack(ideal))


def _write_pages(out, axis, z_mm, stacks, titles):
    extent = [axis[0], axis[-1], axis[0], axis[-1]]
    for page, start in enumerate(range(0, len(z_mm), 6), 1):
        stop = min(start+6, len(z_mm))
        fig, axs = plt.subplots(stop-start, 4, figsize=(15, 3.35*(stop-start)),
                                constrained_layout=True, squeeze=False)
        for row, iz in enumerate(range(start, stop)):
            for col, (stack, title) in enumerate(zip(stacks, titles)):
                im = axs[row, col].imshow(stack[iz], origin="lower", extent=extent,
                    cmap="inferno", vmin=0, vmax=1, interpolation="nearest")
                axs[row, col].axhline(0, color="cyan", lw=.35, alpha=.45)
                axs[row, col].axvline(0, color="cyan", lw=.35, alpha=.45)
                axs[row, col].set(title=title, xlabel="signed x (um)",
                                  ylabel="signed y (um)", aspect="equal")
                axs[row, col].text(.02, .97, f"z={z_mm[iz]:g} mm",
                    transform=axs[row, col].transAxes, va="top", color="white",
                    bbox=dict(facecolor="black", alpha=.55, edgecolor="none", pad=2))
        fig.colorbar(im, ax=axs, label="plane-normalized intensity", shrink=.7)
        fig.suptitle("Global-branch q=20 retrieval: measured, fit, predicted correction, and local ideal\n"
                     "corrected is a phase-only model prediction, not a post-SLM measurement",
                     fontsize=14)
        fig.savefig(out/f"measured_fit_corrected_ideal_page_{page}.png", dpi=400,
                    bbox_inches="tight")
        fig.savefig(out/f"measured_fit_corrected_ideal_page_{page}.pdf",
                    bbox_inches="tight")
        plt.close(fig)


def _write_representative(out, axis, z_mm, stacks, titles):
    iz = int(np.argmin(np.abs(np.asarray(z_mm)+10.0)))
    extent = [axis[0], axis[-1], axis[0], axis[-1]]
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.6), constrained_layout=True)
    for ax, stack, title in zip(axs, stacks, titles):
        im = ax.imshow(stack[iz], origin="lower", extent=extent, cmap="inferno",
                       vmin=0, vmax=1, interpolation="nearest")
        ax.axhline(0, color="cyan", lw=.4, alpha=.5)
        ax.axvline(0, color="cyan", lw=.4, alpha=.5)
        ax.set(title=title, xlabel="signed x (um)", ylabel="signed y (um)", aspect="equal")
        ax.title.set_fontsize(10)
    fig.colorbar(im, ax=axs, label="plane-normalized intensity", shrink=.84)
    fig.suptitle(f"Representative q=20 plane at z={z_mm[iz]:g} mm")
    fig.savefig(out/"representative_measured_fit_corrected_ideal.png", dpi=500,
                bbox_inches="tight")
    fig.savefig(out/"representative_measured_fit_corrected_ideal.pdf",
                bbox_inches="tight")
    plt.close(fig)


def _write_propagation(out, axis, z_mm, measured, corrected, ideal):
    mid = len(axis)//2
    sections = [
        (measured[:, mid, :], corrected[:, mid, :], ideal[:, mid, :], "x", "y=0"),
        (measured[:, :, mid], corrected[:, :, mid], ideal[:, :, mid], "y", "x=0"),
    ]
    fig, axs = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    for row, (msec, csec, isec, coord, fixed) in enumerate(sections):
        for col, (arr, title) in enumerate(((msec, "LAB MEASURED"),
                (csec, "PREDICTED PHASE-ONLY CORRECTED"),
                (isec, "LOCAL ANALYTIC q=20 IDEAL"))):
            im = axs[row, col].imshow(arr, origin="lower", aspect="auto",
                cmap="inferno", vmin=0, vmax=1,
                extent=[axis[0], axis[-1], z_mm[0], z_mm[-1]],
                interpolation="nearest")
            axs[row, col].axvline(0, color="cyan", lw=.6, alpha=.7)
            axs[row, col].set(title=f"{coord}-z ({fixed}) | {title}",
                              xlabel=f"signed {coord} (um)", ylabel="relative z (mm)")
    fig.colorbar(im, ax=axs, label="plane-normalized intensity", shrink=.82)
    fig.suptitle("Full signed propagation region; corrected is a per-plane model prediction")
    fig.savefig(out/"signed_xz_yz_measured_corrected_ideal.png", dpi=500,
                bbox_inches="tight")
    fig.savefig(out/"signed_xz_yz_measured_corrected_ideal.pdf", bbox_inches="tight")
    plt.close(fig)


def _write_cross_sections(out, axis, z_mm, measured, corrected, ideal):
    mid = len(axis)//2
    for page, start in enumerate(range(0, len(z_mm), 6), 1):
        stop = min(start+6, len(z_mm))
        fig, axs = plt.subplots(stop-start, 2, figsize=(14, 2.8*(stop-start)),
                                constrained_layout=True, squeeze=False)
        for row, iz in enumerate(range(start, stop)):
            cuts = ((measured[iz, mid], corrected[iz, mid], ideal[iz, mid], "x", "y=0"),
                    (measured[iz, :, mid], corrected[iz, :, mid], ideal[iz, :, mid], "y", "x=0"))
            for col, (mcut, ccut, icut, coord, fixed) in enumerate(cuts):
                ax = axs[row, col]
                ax.plot(axis, mcut, color="black", lw=1.5, label="measured")
                ax.plot(axis, ccut, color="#0072B2", lw=1.5, label="predicted corrected")
                ax.plot(axis, icut, color="#D55E00", lw=1.3, ls="--", label="local ideal")
                ax.axvline(0, color=".55", lw=.6)
                ax.set(title=f"z={z_mm[iz]:g} mm | {coord} cut ({fixed})",
                       xlabel=f"signed {coord} (um)", ylabel="normalized intensity",
                       xlim=(axis[0], axis[-1]), ylim=(0, 1.08))
                ax.grid(alpha=.2)
        axs[0, 0].legend(ncol=3, fontsize=8)
        fig.suptitle("Measured / predicted-corrected / local-ideal linear sections")
        fig.savefig(out/f"linear_cross_sections_page_{page}.png", dpi=400,
                    bbox_inches="tight")
        fig.savefig(out/f"linear_cross_sections_page_{page}.pdf", bbox_inches="tight")
        plt.close(fig)


def _write_3d(out, axis, z_mm, measured, corrected):
    # Do not imply extra measurements: only XY is decimated for rendering and the
    # 18 acquired z planes are retained exactly.
    a = axis[::4]
    volumes = (measured[:, ::4, ::4], corrected[:, ::4, ::4])
    np.savez_compressed(out/"measured_corrected_3d_intensity_stacks.npz",
                        x_um=a, y_um=a, z_relative_mm=z_mm,
                        measured=volumes[0].astype(np.float32),
                        predicted_corrected=volumes[1].astype(np.float32))
    fig = plt.figure(figsize=(14, 7))
    for panel, (volume, title) in enumerate(zip(volumes,
            ("LAB MEASURED", "PREDICTED PHASE-ONLY CORRECTED")), 1):
        ax = fig.add_subplot(1, 2, panel, projection="3d")
        iz, iy, ix = np.where(volume >= .30)
        val = volume[iz, iy, ix]
        keep = np.arange(len(val))[::max(1, len(val)//20000)]
        ax.scatter(a[ix[keep]], a[iy[keep]], np.asarray(z_mm)[iz[keep]],
                   c=val[keep], cmap="inferno", vmin=.3, vmax=1,
                   s=2, alpha=.30, linewidths=0, rasterized=True)
        ax.set(xlabel="x (um)", ylabel="y (um)", zlabel="z (mm)", title=title,
               xlim=(a[0], a[-1]), ylim=(a[0], a[-1]))
        ax.set_box_aspect((1, 1, 2.2)); ax.view_init(elev=23, azim=-58)
        ax.tick_params(labelsize=8)
    fig.subplots_adjust(left=.01, right=.99, bottom=.05, top=.89, wspace=-.12)
    fig.suptitle("q=20 3D intensity morphology (I >= 0.30; each plane normalized)")
    fig.savefig(out/"measured_vs_predicted_corrected_3d.png", dpi=500,
                bbox_inches="tight")
    fig.savefig(out/"measured_vs_predicted_corrected_3d.pdf", bbox_inches="tight")
    plt.close(fig)


def _write_metric_plot(out, rows):
    z = np.asarray([r["z_relative_mm"] for r in rows], float)
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for key, label in (("measured_vs_local_ideal_corr", "measured vs local ideal"),
            ("modal_fit_vs_measured_corr", "modal fit vs measured"),
            ("predicted_corrected_vs_local_ideal_corr", "predicted corrected vs local ideal")):
        axs[0].plot(z, [r[key] for r in rows], "o-", label=label)
    for key, label in (("measured_vs_local_ideal_rmse", "measured vs local ideal"),
            ("modal_fit_vs_measured_rmse", "modal fit vs measured"),
            ("predicted_corrected_vs_local_ideal_rmse", "predicted corrected vs local ideal")):
        axs[1].plot(z, [r[key] for r in rows], "o-", label=label)
    axs[0].set(ylabel="Pearson correlation", ylim=(-.05, 1.05),
               title="Plane-normalized intensity correlation")
    axs[1].set(ylabel="normalized RMSE", title="Plane-normalized intensity error")
    for ax in axs:
        ax.set_xlabel("relative z (mm)"); ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.suptitle("Global-branch model comparison metrics\n"
                 "local ideal uses the selected k_perp; corrected is not a post-SLM measurement")
    fig.savefig(out/"measured_corrected_ideal_metrics_vs_z.png", dpi=400,
                bbox_inches="tight")
    fig.savefig(out/"measured_corrected_ideal_metrics_vs_z.pdf", bbox_inches="tight")
    plt.close(fig)


def write_model_comparison(output_dir, images, retrievals, z_relative_mm,
                           pixel_pitch_m, q):
    """Write the measured/fit/corrected/ideal diagnostic suite and metrics."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    z_mm = np.asarray(z_relative_mm, float)
    axis, measured, fitted, corrected, ideal = _make_stacks(
        images, retrievals, pixel_pitch_m, q)
    stacks = (measured, fitted, corrected, ideal)
    titles = ("LAB MEASURED", "MODAL FIT",
              "PREDICTED CORRECTED", "LOCAL q=20 IDEAL")
    mask = np.hypot(*np.meshgrid(axis, axis, indexing="ij")) <= 160
    rows = []
    for iz, retrieval in enumerate(retrievals):
        mi_c, mi_e = _similarity(measured[iz], ideal[iz], mask)
        fm_c, fm_e = _similarity(fitted[iz], measured[iz], mask)
        ci_c, ci_e = _similarity(corrected[iz], ideal[iz], mask)
        rows.append({"z_index": iz, "z_relative_mm": float(z_mm[iz]),
            "selected_k_perp_m_inv": float(retrieval.k_perp_m_inv),
            "measured_vs_local_ideal_corr": mi_c,
            "measured_vs_local_ideal_rmse": mi_e,
            "modal_fit_vs_measured_corr": fm_c,
            "modal_fit_vs_measured_rmse": fm_e,
            "predicted_corrected_vs_local_ideal_corr": ci_c,
            "predicted_corrected_vs_local_ideal_rmse": ci_e})
    with (out/"measured_corrected_ideal_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(out/"measured_fit_corrected_ideal_stacks.npz",
                        axis_um=axis, z_relative_mm=z_mm,
                        measured=measured.astype(np.float32), fitted=fitted.astype(np.float32),
                        predicted_corrected=corrected.astype(np.float32),
                        local_analytic_ideal=ideal.astype(np.float32))
    _write_pages(out, axis, z_mm, stacks, titles)
    _write_representative(out, axis, z_mm, stacks, titles)
    _write_propagation(out, axis, z_mm, measured, corrected, ideal)
    _write_cross_sections(out, axis, z_mm, measured, corrected, ideal)
    _write_3d(out, axis, z_mm, measured, corrected)
    _write_metric_plot(out, rows)
    summary = {
        "interpretation": "phase-only corrected intensity is a per-plane retrieval prediction, not a post-SLM measurement",
        "ideal_definition": "local analytic q=20 Bessel-Gauss intensity at each globally selected k_perp; not a calibrated nominal propagation target",
        "normalization": "each plane and each model independently normalized to peak intensity",
        "mean_measured_vs_local_ideal_corr": float(np.mean([r["measured_vs_local_ideal_corr"] for r in rows])),
        "mean_predicted_corrected_vs_local_ideal_corr": float(np.mean([r["predicted_corrected_vs_local_ideal_corr"] for r in rows])),
        "mean_modal_fit_vs_measured_corr": float(np.mean([r["modal_fit_vs_measured_corr"] for r in rows])),
    }
    (out/"model_comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
