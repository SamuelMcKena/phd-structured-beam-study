"""Shared Phase 2J presentation-rendering style.

This module changes presentation rendering only. It does not change optical
fields, propagation, calibration assumptions, or scientific claim boundaries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

EPS = np.finfo(float).tiny

FIG_BG = "#080b0f"
AX_BG = "#0b0d10"
TEXT = "#f1f3f4"
MUTED = "#c6cdd4"
GRID = "#39434d"
RED = "#ff3b30"
GREEN = "#39d6ad"
GOLD = "#f2c14e"
BORDER = "#334252"
CMAP = "inferno"
SAVE_DPI = 420
DISPLAY_INTERPOLATION = "lanczos"


def normalise(values: np.ndarray, peak: float | None = None) -> np.ndarray:
    arr = np.maximum(np.asarray(values, dtype=float), 0.0)
    scale = float(np.max(arr)) if peak is None else float(peak)
    return arr / max(scale, EPS)


def style_fig(fig: plt.Figure) -> None:
    fig.patch.set_facecolor(FIG_BG)


def style_ax(ax: plt.Axes) -> None:
    ax.set_facecolor(AX_BG)
    for spine in ax.spines.values():
        spine.set_color("#b8c1c8")
        spine.set_linewidth(0.75)
    ax.tick_params(colors=MUTED, labelsize=9, length=2.5)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT)
    ax.grid(color=GRID, alpha=0.34, linewidth=0.5)


def save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=SAVE_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def draw_xy(
    ax: plt.Axes,
    values: np.ndarray,
    extent_mm: Sequence[float],
    title: str,
    *,
    peak: float | None = None,
    ylabel: bool = False,
    show_y: bool | None = None,
) -> None:
    if show_y is not None:
        ylabel = bool(show_y)
    style_ax(ax)
    ax.imshow(
        normalise(values, peak),
        origin="lower",
        extent=list(map(float, extent_mm)),
        cmap=CMAP,
        vmin=0.0,
        vmax=1.0,
        interpolation=DISPLAY_INTERPOLATION,
        aspect="equal",
    )
    ax.set_title(title, fontsize=12.5, weight="bold", pad=7)
    ax.set_xlabel("x (mm)", fontsize=9)
    if ylabel:
        ax.set_ylabel("y (mm)", fontsize=9)
    else:
        ax.tick_params(labelleft=False)
    ax.axhline(0.0, color="white", alpha=0.15, linewidth=0.45)
    ax.axvline(0.0, color="white", alpha=0.15, linewidth=0.45)


def draw_xz(
    ax: plt.Axes,
    values: np.ndarray,
    coord_m: np.ndarray,
    z_values_m: np.ndarray,
    *,
    peak: float | None = None,
    ylabel: bool = True,
    show_y: bool | None = None,
    z_ref_m: float = 60e-3,
) -> None:
    if show_y is not None:
        ylabel = bool(show_y)
    style_ax(ax)
    ax.imshow(
        normalise(np.asarray(values, dtype=float).T, peak),
        origin="lower",
        extent=[
            float(z_values_m[0]) * 1e3,
            float(z_values_m[-1]) * 1e3,
            float(coord_m[0]) * 1e3,
            float(coord_m[-1]) * 1e3,
        ],
        cmap=CMAP,
        vmin=0.0,
        vmax=1.0,
        interpolation=DISPLAY_INTERPOLATION,
        aspect="auto",
    )
    ax.set_xlabel("z from axicon (mm)", fontsize=9)
    if ylabel:
        ax.set_ylabel("x at fixed y=0 (mm)", fontsize=9)
    else:
        ax.tick_params(labelleft=False)
    ax.axhline(0.0, color="white", alpha=0.15, linewidth=0.45)
    ax.axvline(float(z_ref_m) * 1e3, color="white", alpha=0.30, linestyle="--", linewidth=0.7)


def presentation_crop_halfwidth(requested_halfwidth_m: float) -> float:
    """Map legacy presentation crops to the Phase 2J tighter framing.

    The mapping is visual only. It never changes the simulated field.
    """
    h = float(requested_halfwidth_m)
    if h >= 2.0e-3:
        return h  # axicon-plane illumination / tip-loading context needs wide field
    if h >= 0.8e-3:
        return 0.65e-3  # decentre / error fingerprints
    if h >= 0.34e-3:
        return 0.26e-3  # non-ideal tip
    if h >= 0.30e-3:
        return 0.20e-3  # ideal output / route output
    return h
