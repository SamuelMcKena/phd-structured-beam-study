"""Phase 2J presentation visual refinement for the ideal B0/V1/V3 figure.

This is a presentation-only renderer built on the current validated optical route.
It does not replace the underlying numerical evidence. The changes are visual:

- inferno colormap (black -> red -> orange -> yellow),
- tighter transverse and longitudinal framing,
- higher native simulation grid for the presentation render,
- denser z sampling and higher export DPI,
- display interpolation only; all fields are regenerated from the current model.

The figure therefore stays tied to the same dual-SLM -> explicit 4F -> physical
axicon model while being easier to read on a presentation slide.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np

from vbb_study.digital_twin.vortex_continuous_propagation import (
    build_fixed_plane_longitudinal_map,
    build_fixed_support_spectrum,
    native_field_at_z,
)
from vbb_study.digital_twin.vortex_system_route import build_system_route


EPS = np.finfo(float).tiny
Z_REF_M = 60.0e-3
Z_VALUES_M = np.linspace(5.0e-3, 120.0e-3, 64)
TRANSVERSE_HALFWIDTH_M = 0.20e-3
LONGITUDINAL_COORD_M = np.linspace(-0.22e-3, 0.22e-3, 441)

FIG_BG = "#080b0f"
AX_BG = "#0b0d10"
TEXT = "#f1f3f4"
MUTED = "#c6cdd4"
GRID = "#39434d"
CMAP = "inferno"


def _normalise(values: np.ndarray) -> np.ndarray:
    arr = np.maximum(np.asarray(values, dtype=float), 0.0)
    return arr / max(float(np.max(arr)), EPS)


def _style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(AX_BG)
    for spine in ax.spines.values():
        spine.set_color("#b8c1c8")
        spine.set_linewidth(0.75)
    ax.tick_params(colors=MUTED, labelsize=9, length=2.5)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT)
    ax.grid(color=GRID, alpha=0.38, linewidth=0.55)


def _propagator(route: Mapping[str, Any]):
    return build_fixed_support_spectrum(
        np.asarray(route["post_axicon"], dtype=np.complex128),
        dict(route["grid"]),
        wavelength_m=float(route["metadata"]["wavelength_m"]),
        z_max_m=float(Z_VALUES_M[-1]),
        minimum_retained_spectral_power=0.995,
    )


def _xy_at_z(route: Mapping[str, Any], z_m: float = Z_REF_M):
    prop = _propagator(route)
    field = native_field_at_z(prop, float(z_m))
    return np.asarray(field, dtype=np.complex128), prop


def _longitudinal(route: Mapping[str, Any], label: str):
    prop = _propagator(route)
    mapped = build_fixed_plane_longitudinal_map(
        prop,
        z_values_m=Z_VALUES_M,
        x_coordinates_m=LONGITUDINAL_COORD_M,
        y_coordinates_m=LONGITUDINAL_COORD_M,
        fixed_x_m=0.0,
        fixed_y_m=0.0,
        source_label=label,
    )
    return mapped, prop


def _fixed_lab_crop(intensity: np.ndarray, grid: Mapping[str, Any]):
    x = np.asarray(grid["x"], dtype=float)
    ids = np.flatnonzero(np.abs(x) <= TRANSVERSE_HALFWIDTH_M)
    if ids.size < 40:
        raise RuntimeError(
            f"Phase 2J transverse crop has only {ids.size} native samples; "
            "increase --grid-n rather than faking resolution."
        )
    crop = np.asarray(intensity)[np.ix_(ids, ids)]
    extent = [
        x[ids[0]] * 1e3,
        x[ids[-1]] * 1e3,
        x[ids[0]] * 1e3,
        x[ids[-1]] * 1e3,
    ]
    return crop, extent, int(ids.size)


def build_figure(output_dir: Path, grid_n: int) -> Path:
    cases = (
        ("B0", "B0 — bright-core Bessel"),
        ("V1", "V1 — vortex-Bessel ℓ=1"),
        ("V3", "V3 — vortex-Bessel ℓ=3"),
    )

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.25), constrained_layout=True)
    fig.patch.set_facecolor(FIG_BG)

    retained = []
    native_crop_samples = []

    for col, (case_id, title) in enumerate(cases):
        route = build_system_route(case_id, grid_n=int(grid_n))
        field, prop_xy = _xy_at_z(route)
        intensity = np.abs(field) ** 2
        crop, extent, nsamp = _fixed_lab_crop(intensity, route["grid"])
        native_crop_samples.append(nsamp)

        ax = axes[0, col]
        _style_axis(ax)
        ax.imshow(
            _normalise(crop),
            origin="lower",
            extent=extent,
            cmap=CMAP,
            vmin=0.0,
            vmax=1.0,
            interpolation="lanczos",  # display only; numerical field is unchanged
            aspect="equal",
        )
        ax.set_title(title, fontsize=13.5, weight="bold", pad=7)
        ax.set_xlabel("x (mm)", fontsize=9)
        if col == 0:
            ax.set_ylabel("y (mm)", fontsize=9)
        else:
            ax.tick_params(labelleft=False)
        ax.axhline(0.0, color="white", alpha=0.16, linewidth=0.45)
        ax.axvline(0.0, color="white", alpha=0.16, linewidth=0.45)

        mapped, prop = _longitudinal(route, f"phase2j-ideal-{case_id}")
        retained.append(float(prop.retained_spectral_power_fraction))
        ax = axes[1, col]
        _style_axis(ax)
        ax.imshow(
            _normalise(np.asarray(mapped.xz_intensity, dtype=float).T),
            origin="lower",
            extent=[
                Z_VALUES_M[0] * 1e3,
                Z_VALUES_M[-1] * 1e3,
                LONGITUDINAL_COORD_M[0] * 1e3,
                LONGITUDINAL_COORD_M[-1] * 1e3,
            ],
            cmap=CMAP,
            vmin=0.0,
            vmax=1.0,
            interpolation="lanczos",  # display only
            aspect="auto",
        )
        ax.set_title(f"{case_id} — fixed-lab x–z", fontsize=13.5, weight="bold", pad=7)
        ax.set_xlabel("z from axicon (mm)", fontsize=9)
        if col == 0:
            ax.set_ylabel("x at fixed y=0 (mm)", fontsize=9)
        else:
            ax.tick_params(labelleft=False)
        ax.axhline(0.0, color="white", alpha=0.16, linewidth=0.45)
        ax.axvline(Z_REF_M * 1e3, color="white", alpha=0.32, linestyle="--", linewidth=0.7)

    fig.suptitle(
        "Beam profile shaping — ideal simulated outputs",
        color=TEXT,
        fontsize=18,
        weight="bold",
        y=1.025,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "02_beam_profile_shaping_B0_V1_V3_inferno_tight.png"
    fig.savefig(out, dpi=420, bbox_inches="tight", facecolor=FIG_BG)
    plt.close(fig)

    manifest = output_dir / "02_beam_profile_shaping_B0_V1_V3_inferno_tight.txt"
    manifest.write_text(
        "\n".join(
            [
                "PHASE2J-PRESENTATION-VISUAL-REFINEMENT",
                f"grid_n={grid_n}",
                f"colormap={CMAP}",
                f"transverse_halfwidth_mm={TRANSVERSE_HALFWIDTH_M*1e3:.6f}",
                f"longitudinal_halfwidth_mm={abs(LONGITUDINAL_COORD_M[0])*1e3:.6f}",
                f"z_samples={len(Z_VALUES_M)}",
                f"native_crop_samples={native_crop_samples}",
                f"fixed_support_retained_power_fraction={retained}",
                "display_interpolation=lanczos_only_for_rendering",
                "longitudinal_coordinates=fixed_lab_no_per_z_recentering",
                "normalisation=per_case_peak_for_morphology_only",
            ]
        ) + "\n",
        encoding="utf-8",
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-n", type=int, default=1536)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/figures/presentation_phase2j"),
    )
    args = parser.parse_args()
    path = build_figure(args.output_dir, args.grid_n)
    print(path)


if __name__ == "__main__":
    main()
