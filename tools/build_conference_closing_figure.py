"""Build the clean closing simulation/experiment/calibration workshop graphic."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def _box(
    ax: plt.Axes,
    *,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.015,rounding_size=0.02",
            linewidth=1.3,
            edgecolor="0.22",
            facecolor="0.975",
        )
    )
    ax.text(
        x + w / 2,
        y + 0.67 * h,
        title,
        ha="center",
        va="center",
        fontsize=15,
        weight="bold",
    )
    ax.text(
        x + w / 2,
        y + 0.35 * h,
        body,
        ha="center",
        va="center",
        fontsize=10,
        color="0.28",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/figures/conference_workshop/06_simulation_experiment_loop.png"),
    )
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(11.8, 4.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _box(
        ax,
        x=0.055,
        y=0.55,
        w=0.28,
        h=0.30,
        title="Simulation",
        body="predict beam structure\nand error signatures",
    )
    _box(
        ax,
        x=0.665,
        y=0.55,
        w=0.28,
        h=0.30,
        title="Experiment",
        body="measure the real beam\nand system response",
    )
    _box(
        ax,
        x=0.36,
        y=0.08,
        w=0.28,
        h=0.25,
        title="Calibration / validation",
        body="use measured parameters\nto update the model",
    )

    arrow_kw = dict(
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.4,
        color="0.30",
    )
    ax.add_patch(FancyArrowPatch((0.34, 0.70), (0.66, 0.70), **arrow_kw))
    ax.add_patch(FancyArrowPatch((0.79, 0.54), (0.61, 0.33), **arrow_kw))
    ax.add_patch(FancyArrowPatch((0.39, 0.33), (0.21, 0.54), **arrow_kw))

    ax.text(
        0.50,
        0.745,
        "predict what to look for",
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="0.28",
    )
    ax.text(
        0.79,
        0.39,
        "measure system\nparameters",
        ha="center",
        va="center",
        fontsize=9.0,
        color="0.28",
    )
    ax.text(
        0.21,
        0.39,
        "refine and\ncompare",
        ha="center",
        va="center",
        fontsize=9.0,
        color="0.28",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
