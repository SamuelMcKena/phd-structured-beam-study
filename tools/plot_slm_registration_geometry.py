#!/usr/bin/env python3
"""Generate schematic figures for the two-blaze SLM registration study.

These are deterministic technical diagrams (not AI artwork).  They are intended
for the academic report and show:
1) the two phase-allocation architectures with blaze/carrier on BOTH SLMs;
2) representative sub-pixel registration positions inside one 8 um pixel cell;
3) a beam footprint over a local SLM lattice for the 0 -> 0.5 pixel x sweep.

The drawings are schematic: they explain the coordinate convention and do not
claim to reproduce the physical size of the full 1920 x 1080 panel.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs" / "figures" / "slm_registration_architecture" / "geometry"


def _rounded_box(ax, xy, w, h, text, fontsize=10):
    box = FancyBboxPatch(
        xy, w, h,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=1.2,
        facecolor="white",
        edgecolor="black",
    )
    ax.add_patch(box)
    ax.text(xy[0] + w/2, xy[1] + h/2, text, ha="center", va="center",
            fontsize=fontsize)
    return box


def plot_architectures(outdir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5, 0.965,
        "Dual-SLM architectures used in the registration study",
        ha="center", va="top", fontsize=16, fontweight="bold",
    )
    ax.text(
        0.5, 0.915,
        "Both SLM1 and SLM2 carry a 20-pixel blaze/carrier; only vortex/correction ownership changes",
        ha="center", va="top", fontsize=10,
    )

    rows = [
        (
            0.61,
            "Architecture A: upstream vortex",
            ["Gaussian input", "SLM1\nvortex + blaze", "SLM2\ncorrection + blaze",
             "4F + selected order", "physical axicon", "propagated field"],
        ),
        (
            0.20,
            "Architecture B: downstream vortex",
            ["Gaussian input", "SLM1\ncorrection + blaze", "SLM2\nvortex + blaze",
             "4F + selected order", "physical axicon", "propagated field"],
        ),
    ]

    x0 = [0.04, 0.20, 0.38, 0.57, 0.73, 0.87]
    widths = [0.11, 0.13, 0.14, 0.12, 0.10, 0.10]
    h = 0.14

    for y, title, labels in rows:
        ax.text(0.035, y + 0.205, title, fontsize=12.5, fontweight="bold",
                ha="left", va="center")
        for i, (x, w, label) in enumerate(zip(x0, widths, labels)):
            _rounded_box(ax, (x, y), w, h, label, fontsize=9.5)
            if i < len(labels)-1:
                ax.annotate(
                    "",
                    xy=(x0[i+1]-0.008, y+h/2),
                    xytext=(x+w+0.008, y+h/2),
                    arrowprops=dict(arrowstyle="->", linewidth=1.4),
                )

    ax.text(
        0.5, 0.035,
        "Flat-correction effective-channel comparison: without measured SLM1->SLM2 transfer, A/B are panel-label symmetric.",
        ha="center", va="bottom", fontsize=9,
    )

    path = outdir / "00_architecture_schematic.png"
    fig.savefig(path, dpi=320, bbox_inches="tight")
    plt.close(fig)
    return path


def _pixel_panel(ax, dx_px, dy_px, title, *, show_reference=True):
    ax.set_aspect("equal")
    ax.set_xlim(-2.7, 2.7)
    ax.set_ylim(-2.7, 2.7)
    ax.set_xticks([])
    ax.set_yticks([])

    # Fixed beam/hologram centre.
    beam = Circle((0, 0), 1.75, facecolor="none", edgecolor="0.25",
                  linewidth=1.3, linestyle="--")
    ax.add_patch(beam)
    ax.scatter([0], [0], s=42, facecolor="white", edgecolor="black", zorder=5)

    # Reference lattice (faint).
    if show_reference:
        for q in np.arange(-3, 4, 1):
            ax.axvline(q, color="0.82", linewidth=0.8)
            ax.axhline(q, color="0.82", linewidth=0.8)

    # Shifted physical lattice.  We draw boundaries at integer+offset.
    for q in np.arange(-4, 5, 1):
        ax.axvline(q + dx_px, color="black", linewidth=1.0)
        ax.axhline(q + dy_px, color="black", linewidth=1.0)

    # Highlight the cell containing the fixed singularity.
    # Find nearest lower boundary.
    left = np.floor(-dx_px) + dx_px
    bottom = np.floor(-dy_px) + dy_px
    rect = Rectangle(
        (left, bottom), 1, 1,
        facecolor="none", edgecolor="tab:blue", linewidth=2.0
    )
    ax.add_patch(rect)

    ax.set_title(title, fontsize=10)
    ax.text(-2.55, -2.42, f"lattice shift = ({dx_px:.3f}, {dy_px:.3f}) px",
            fontsize=7.5, ha="left")


def plot_registration_positions(outdir: Path) -> Path:
    positions = [
        (0.000, 0.000, "0 px"),
        (0.125, 0.000, "1/8 px in x"),
        (0.250, 0.000, "1/4 px in x"),
        (0.375, 0.000, "3/8 px in x"),
        (0.500, 0.000, "1/2 px in x"),
        (0.500, 0.500, "1/2 px in x and y"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(10.8, 7.2), constrained_layout=True)
    for ax, (dx, dy, title) in zip(axes.ravel(), positions):
        _pixel_panel(ax, dx, dy, title)

    fig.suptitle(
        "Representative sub-pixel registration positions on the SLM",
        fontsize=15, fontweight="bold",
    )
    fig.text(
        0.5, 0.01,
        "Beam and hologram centre remain fixed; the physical 8 um pixel lattice moves underneath.",
        ha="center", fontsize=9.5,
    )
    path = outdir / "01_registration_positions.png"
    fig.savefig(path, dpi=320, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_x_sweep(outdir: Path) -> Path:
    offsets = [0.0, 0.125, 0.25, 0.375, 0.5]
    fig, axes = plt.subplots(1, len(offsets), figsize=(15, 3.4), constrained_layout=True)
    for ax, d in zip(axes, offsets):
        _pixel_panel(ax, d, 0.0, f"{d:.3f} px")
    fig.suptitle(
        "The x-registration sweep used for representative XY profiles",
        fontsize=14, fontweight="bold",
    )
    path = outdir / "02_registration_x_sweep.png"
    fig.savefig(path, dpi=320, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default=str(DEFAULT_OUT))
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for path in (
        plot_architectures(outdir),
        plot_registration_positions(outdir),
        plot_x_sweep(outdir),
    ):
        print(path)


if __name__ == "__main__":
    main()
