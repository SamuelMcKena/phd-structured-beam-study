"""Build additional dense A0-portrait poster panels from accepted q=20 inverse artifacts.

Run this after presentation/build_q20_poster_assets.py.  This script creates the
multi-plane measured-vs-fit strip, held-out per-z validation, selected residual
coefficient chart, and representative inverse comparison used by the portrait
poster.  It does not invent or relabel any post-correction camera measurement.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def build(detector_dir: Path, out: Path) -> None:
    detector_dir = Path(detector_dir)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    arrays = np.load(detector_dir / "detector_aware_residual_stacks.npz")
    measured = np.asarray(arrays["measured_beam_frame"], float)
    nominal = np.asarray(arrays["detector_nominal"], float)
    fitted = np.asarray(arrays["axicon_input_positive_error"], float)
    axis = np.asarray(arrays["axis_um"], float)
    z = np.asarray(arrays["z_relative_mm"], float)
    coeff = np.asarray(arrays["axicon_input_coefficients_rad"], float)
    summary = json.loads((detector_dir / "detector_aware_residual_summary.json").read_text())
    extent = [axis[0], axis[-1], axis[0], axis[-1]]

    ids = [0, 4, 8, 12, 17]
    fig, axs = plt.subplots(2, 5, figsize=(13.0, 5.25), constrained_layout=True)
    for col, idx in enumerate(ids):
        for row, stack in enumerate((measured, fitted)):
            ax = axs[row, col]
            ax.imshow(stack[idx], origin="lower", extent=extent, cmap="inferno", vmin=0, vmax=1, interpolation="nearest")
            ax.set_aspect("equal")
            ax.set_xticks([-100, 0, 100]); ax.set_yticks([-100, 0, 100])
            ax.tick_params(labelsize=7, length=2)
            if row == 0:
                ax.set_title(f"z = {z[idx]:.0f} mm", fontsize=9, pad=3)
            if col == 0:
                ax.set_ylabel("measured" if row == 0 else "fitted model", fontsize=9)
            else:
                ax.set_yticklabels([])
            if row == 1:
                ax.set_xlabel("x (µm)", fontsize=8)
            else:
                ax.set_xticklabels([])
    fig.suptitle("Multi-plane morphology: real BeamGage stack versus detector-aware fitted model", fontsize=12, fontweight="bold")
    fig.savefig(out / "zstack_measured_fit_strip.png", dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    held = np.arange(1, len(z), 2)
    nominal_rows = summary["detector_aware_nominal"]["heldout"]["per_plane"]
    fitted_rows = summary["phase_models"]["axicon_input"]["heldout_after"]["per_plane"]
    r0 = np.array([row["pearson_r"] for row in nominal_rows])
    r1 = np.array([row["pearson_r"] for row in fitted_rows])
    e0 = np.array([row["nrmse"] for row in nominal_rows])
    e1 = np.array([row["nrmse"] for row in fitted_rows])

    fig, axs = plt.subplots(2, 1, figsize=(7.2, 5.4), sharex=True, constrained_layout=True)
    axs[0].plot(z[held], r0, "o-", lw=1.7, label="nominal")
    axs[0].plot(z[held], r1, "o-", lw=1.7, label="fitted residual")
    axs[0].axhline(r0.mean(), ls="--", lw=1, alpha=.5)
    axs[0].axhline(r1.mean(), ls="--", lw=1, alpha=.5)
    axs[0].set_ylabel("Pearson r"); axs[0].set_ylim(.30, 1.0); axs[0].grid(alpha=.22)
    axs[0].legend(frameon=False, ncol=2, fontsize=8)
    axs[0].set_title("Held-out planes were never used by the inverse fit", fontsize=11, fontweight="bold")
    axs[1].plot(z[held], e0, "o-", lw=1.7, label="nominal")
    axs[1].plot(z[held], e1, "o-", lw=1.7, label="fitted residual")
    axs[1].set_ylabel("NRMSE"); axs[1].set_xlabel("relative z (mm)"); axs[1].grid(alpha=.22)
    axs[1].text(.01, .92, f"mean r: {r0.mean():.3f} → {r1.mean():.3f}\nmean NRMSE: {e0.mean():.3f} → {e1.mean():.3f}", transform=axs[1].transAxes, va="top", fontsize=9)
    fig.savefig(out / "heldout_validation_vs_z.png", dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    labels = ["m1 cos", "m1 sin", "m2 cos", "m2 sin", "m3 cos", "m3 sin"]
    fig, ax = plt.subplots(figsize=(7.4, 3.2), constrained_layout=True)
    ax.bar(np.arange(6), coeff)
    ax.axhline(0, lw=1)
    ax.set_xticks(np.arange(6), labels, fontsize=8)
    ax.set_ylabel("phase coefficient (rad)")
    ax.set_title("Selected axicon-input residual coefficients", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=.2)
    for i, value in enumerate(coeff):
        ax.text(i, value + (.035 if value >= 0 else -.055), f"{value:+.2f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=8)
    fig.savefig(out / "residual_coefficients.png", dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    rep = int(np.argmin(abs(z + 10)))
    fig, axs = plt.subplots(1, 4, figsize=(12.4, 3.4), constrained_layout=True)
    items = [
        (measured[rep], "Measured", "inferno", 0, 1),
        (nominal[rep], "Nominal model", "inferno", 0, 1),
        (fitted[rep], "Fitted residual", "inferno", 0, 1),
        (fitted[rep] - measured[rep], "Fit - measured", "coolwarm", -.28, .28),
    ]
    for ax, (image, title, cmap, vmin, vmax) in zip(axs, items):
        ax.imshow(image, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title, fontsize=9, fontweight="bold"); ax.set_aspect("equal")
        ax.set_xticks([-100, 0, 100]); ax.set_yticks([-100, 0, 100]); ax.tick_params(labelsize=7, length=2)
    axs[0].set_ylabel(f"z = {z[rep]:.0f} mm\ny (µm)", fontsize=8)
    for ax in axs[1:]:
        ax.set_yticklabels([])
    for ax in axs:
        ax.set_xlabel("x (µm)", fontsize=8)
    fig.savefig(out / "inverse_representative_row.png", dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("presentation/generated/q20_poster_portrait_assets"))
    args = parser.parse_args()
    build(args.detector_dir, args.out)


if __name__ == "__main__":
    main()
