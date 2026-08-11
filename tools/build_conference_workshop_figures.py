"""Build clean presentation-only figures from the validated vortex system model.

These figures are intentionally *not* diagnostic plots.  They are compact,
consistent, 16:9-friendly visualisations for the first-year PhD workshop talk.
All numerical fields are regenerated from the validated physical-system route;
no raster crops of old diagnostic figures are used.

Presentation outputs
--------------------
01_simulation_pipeline.png
    V1 computational route: input -> SLM1 -> SLM2/4F -> axicon -> propagated field.
02_ideal_beam_family.png
    B0/V1/V3 transverse and longitudinal fields with common physical axes.
03_realistic_error_scope.png
    Compact diagram of the controlled error families represented by the model.
04_v1_axicon_decentre.png
    V1 lateral axicon decentre at -500/0/+500 um, centred on the topological core.
05_v1_rounded_tip.png
    V1 sharp/200/800 um hyperbolic tip-radius sensitivity, resolution checked.

The rigid-tilt surrogate is deliberately absent because it is rejected by the
axicon-physics-v2 validation suite.  Lab-specific refractive tilt remains a
separate reference problem until the real axicon geometry is supplied.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

from vbb_study.digital_twin.phase2a_contracts import (
    canonical_hardware_manifest,
    hardware_value,
)
from vbb_study.digital_twin.phase2e_spectral_propagation import (
    build_dense_spectral_propagation,
)
from vbb_study.digital_twin.vortex_axicon_tip_reference import tip_resolution
from vbb_study.digital_twin.vortex_following_propagation import (
    build_beam_following_propagation,
    transverse_morphology_axis,
)
from vbb_study.digital_twin.vortex_morphology_tracking import (
    track_bessel_feature_axis,
)
from vbb_study.digital_twin.vortex_system_route import (
    AxiconError,
    SystemErrorConfig,
    build_system_route,
)
from vbb_study.equations.propagation import angular_spectrum_propagate_bl


EPS = np.finfo(float).tiny
XY_Z_M = 60e-3
Z_VALUES_M = np.arange(5e-3, 140e-3 + 2e-3, 2e-3)
MORPHOLOGY_OFFSETS_M = np.linspace(-220e-6, 220e-6, 401)
WIDE_COORDINATE_M = np.linspace(-1.2e-3, 1.2e-3, 481)


def _ell(case_id: str) -> int:
    return {"B0": 0, "V1": 1, "V3": 3}[case_id]


def _normalise(intensity: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(intensity, dtype=float), 0.0)
    return values / max(float(np.max(values)), EPS)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _propagate_xy(route: Mapping[str, Any], z_m: float = XY_Z_M) -> np.ndarray:
    grid = dict(route["grid"])
    wavelength = float(route["metadata"]["wavelength_m"])
    field = angular_spectrum_propagate_bl(
        route["post_axicon"],
        grid,
        wavelength,
        float(z_m),
        n_medium=1.0,
        bandlimit=True,
        include_evanescent=True,
    )
    return np.asarray(field, dtype=np.complex128)


def _crop_about(
    data: np.ndarray,
    grid: Mapping[str, Any],
    *,
    centre_x_m: float,
    centre_y_m: float,
    halfwidth_m: float,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    x = np.asarray(grid["x"], dtype=float)
    ix = np.flatnonzero(np.abs(x - float(centre_x_m)) <= float(halfwidth_m))
    iy = np.flatnonzero(np.abs(x - float(centre_y_m)) <= float(halfwidth_m))
    if ix.size < 24 or iy.size < 24:
        raise RuntimeError("presentation ROI is under-sampled")
    crop = np.asarray(data)[np.ix_(iy, ix)]
    extent = (
        (x[ix[0]] - centre_x_m) * 1e6,
        (x[ix[-1]] - centre_x_m) * 1e6,
        (x[iy[0]] - centre_y_m) * 1e6,
        (x[iy[-1]] - centre_y_m) * 1e6,
    )
    return crop, extent


def _axisymmetric_xz(route: Mapping[str, Any]) -> np.ndarray:
    following = build_beam_following_propagation(
        grid=dict(route["grid"]),
        wavelength_m=float(route["metadata"]["wavelength_m"]),
        z_values_m=Z_VALUES_M,
        transverse_offsets_m=MORPHOLOGY_OFFSETS_M,
        scalar_field=np.asarray(route["post_axicon"]),
        x_axis_m=0.0,
        y_axis_m=0.0,
        source_label="conference-axisymmetric",
    )
    return np.asarray(following.xz_intensity, dtype=float)


def _tracked_xz_for_x_decentre(
    route: Mapping[str, Any],
    *,
    case_id: str,
    seed_x_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    grid = dict(route["grid"])
    wavelength = float(route["metadata"]["wavelength_m"])
    fixed = build_dense_spectral_propagation(
        grid=grid,
        wavelength_m=wavelength,
        z_values_m=Z_VALUES_M,
        transverse_coordinates_m=WIDE_COORDINATE_M,
        scalar_field=np.asarray(route["post_axicon"]),
        source_label=f"conference-wide-{case_id}-{seed_x_m:g}",
    )
    track = track_bessel_feature_axis(
        fixed.xz_intensity,
        WIDE_COORDINATE_M,
        vortex_charge=_ell(case_id),
        seed_coordinate_m=float(seed_x_m),
        search_halfwidth_m=0.24e-3,
        maximum_step_m=55e-6,
    )
    if track.detected_fraction < 0.95:
        raise RuntimeError(
            f"presentation core track resolved only {track.detected_fraction:.1%} of active planes"
        )
    following = build_beam_following_propagation(
        grid=grid,
        wavelength_m=wavelength,
        z_values_m=Z_VALUES_M,
        transverse_offsets_m=MORPHOLOGY_OFFSETS_M,
        scalar_field=np.asarray(route["post_axicon"]),
        x_axis_m=track.coordinate_m,
        y_axis_m=0.0,
        source_label=f"conference-following-{case_id}-{seed_x_m:g}",
    )
    return np.asarray(following.xz_intensity, dtype=float), np.asarray(track.coordinate_m)


def _draw_intensity(
    ax: plt.Axes,
    values: np.ndarray,
    *,
    extent: Sequence[float],
    xlabel: str | None,
    ylabel: str | None,
    title: str | None = None,
    aspect: str = "equal",
) -> None:
    ax.imshow(
        _normalise(values),
        origin="lower",
        extent=list(map(float, extent)),
        cmap="inferno",
        vmin=0.0,
        vmax=1.0,
        aspect=aspect,
        interpolation="nearest",
    )
    if title:
        ax.set_title(title, fontsize=13, pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10)
    else:
        ax.set_xticklabels([])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    else:
        ax.set_yticklabels([])
    ax.tick_params(labelsize=8, length=3)


def build_simulation_pipeline(output_dir: Path, grid_n: int) -> Path:
    """Presentation schematic populated with fields from the actual route."""

    route = build_system_route("V1", grid_n=int(grid_n))
    grid = dict(route["grid"])
    x = np.asarray(grid["x"], dtype=float)
    X = np.asarray(grid["X"], dtype=float)
    Y = np.asarray(grid["Y"], dtype=float)

    input_i = np.abs(route["input_beam"]) ** 2
    slm1_phase = np.angle(
        np.asarray(route["post_slm1"]) * np.conj(np.asarray(route["input_beam"]))
    )
    slm2_phase = np.angle(
        np.asarray(route["post_slm2"]) * np.conj(np.asarray(route["post_slm1"]))
    )
    fourier_i = np.abs(route["fourier_plane_before_iris"]) ** 2
    output_field = _propagate_xy(route)
    output_i = np.abs(output_field) ** 2

    # The first four panels show the full optical aperture / Fourier plane.  The
    # final panel is a deliberately tighter physical ROI around the generated
    # vortex-Bessel structure.
    full_half = 2.35e-3
    full_mask = np.flatnonzero(np.abs(x) <= full_half)
    full_extent_mm = [
        x[full_mask[0]] * 1e3,
        x[full_mask[-1]] * 1e3,
        x[full_mask[0]] * 1e3,
        x[full_mask[-1]] * 1e3,
    ]
    output_axis = transverse_morphology_axis(
        output_field,
        grid,
        vortex_charge=1,
        seed_x_m=0.0,
        seed_y_m=0.0,
    )
    output_crop, output_extent = _crop_about(
        output_i,
        grid,
        centre_x_m=output_axis.x_m,
        centre_y_m=output_axis.y_m,
        halfwidth_m=190e-6,
    )

    fig = plt.figure(figsize=(14.2, 3.4))
    gs = fig.add_gridspec(1, 5, left=0.035, right=0.985, bottom=0.18, top=0.82, wspace=0.32)
    axes = [fig.add_subplot(gs[0, i]) for i in range(5)]

    arrays = [input_i, slm1_phase, slm2_phase, fourier_i]
    titles = ["Input beam", "SLM 1\nvortex phase", "SLM 2\ncarrier", "4F Fourier\nfiltering"]
    cmaps = ["inferno", "twilight", "twilight", "inferno"]
    for ax, values, title, cmap in zip(axes[:4], arrays, titles, cmaps):
        sub = np.asarray(values)[np.ix_(full_mask, full_mask)]
        if "phase" in title.lower() or "carrier" in title.lower():
            ax.imshow(
                sub,
                origin="lower",
                extent=full_extent_mm,
                cmap=cmap,
                vmin=-np.pi,
                vmax=np.pi,
                interpolation="nearest",
            )
        else:
            ax.imshow(
                _normalise(sub),
                origin="lower",
                extent=full_extent_mm,
                cmap=cmap,
                vmin=0,
                vmax=1,
                interpolation="nearest",
            )
        ax.set_title(title, fontsize=12, pad=7)
        ax.set_xlabel("x (mm)", fontsize=8)
        if ax is axes[0]:
            ax.set_ylabel("y (mm)", fontsize=8)
        else:
            ax.set_yticklabels([])
        ax.tick_params(labelsize=7, length=2)

    _draw_intensity(
        axes[4],
        output_crop,
        extent=output_extent,
        xlabel="x (µm)",
        ylabel=None,
        title="Propagated\nvortex-Bessel field",
    )

    for left, right in zip(axes[:-1], axes[1:]):
        box_l = left.get_position()
        box_r = right.get_position()
        arrow = FancyArrowPatch(
            (box_l.x1 + 0.006, 0.50),
            (box_r.x0 - 0.006, 0.50),
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.1,
            color="0.3",
        )
        fig.add_artist(arrow)

    fig.text(
        0.5,
        0.06,
        "Numerical route mirrors the experimental optical train",
        ha="center",
        fontsize=11,
        color="0.25",
    )
    path = output_dir / "01_simulation_pipeline.png"
    _save(fig, path)
    return path


def build_ideal_family(output_dir: Path, grid_n: int) -> Path:
    cases = ("B0", "V1", "V3")
    labels = ("B0   (ℓ = 0)", "V1   (ℓ = 1)", "V3   (ℓ = 3)")
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.0), constrained_layout=True)

    for col, (case_id, label) in enumerate(zip(cases, labels)):
        route = build_system_route(case_id, grid_n=int(grid_n))
        field_xy = _propagate_xy(route)
        axis = transverse_morphology_axis(
            field_xy,
            dict(route["grid"]),
            vortex_charge=_ell(case_id),
            seed_x_m=0.0,
            seed_y_m=0.0,
            search_radius_m=0.8e-3,
        )
        halfwidth = 250e-6
        crop, extent = _crop_about(
            np.abs(field_xy) ** 2,
            dict(route["grid"]),
            centre_x_m=axis.x_m,
            centre_y_m=axis.y_m,
            halfwidth_m=halfwidth,
        )
        _draw_intensity(
            axes[0, col],
            crop,
            extent=extent,
            xlabel=None,
            ylabel="y (µm)" if col == 0 else None,
            title=label,
        )

        xz = _axisymmetric_xz(route)
        _draw_intensity(
            axes[1, col],
            xz.T,
            extent=[
                Z_VALUES_M[0] * 1e3,
                Z_VALUES_M[-1] * 1e3,
                MORPHOLOGY_OFFSETS_M[0] * 1e6,
                MORPHOLOGY_OFFSETS_M[-1] * 1e6,
            ],
            xlabel="z from axicon (mm)",
            ylabel="x (µm)" if col == 0 else None,
            aspect="auto",
        )
        axes[0, col].axhline(0, color="white", alpha=0.25, linewidth=0.5)
        axes[0, col].axvline(0, color="white", alpha=0.25, linewidth=0.5)

    fig.text(0.5, 0.995, "Ideal simulated beam family", ha="center", va="top", fontsize=16)
    fig.text(
        0.99,
        0.01,
        "Normalised intensity • identical physical axes",
        ha="right",
        va="bottom",
        fontsize=8,
        color="0.35",
    )
    path = output_dir / "02_ideal_beam_family.png"
    _save(fig, path)
    return path


def build_error_scope(output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(13.3, 4.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    blocks = [
        (0.03, "Input beam", "size • offset\nellipticity"),
        (0.22, "SLM 1 + SLM 2", "phase response • registration\npixel/fill-factor effects"),
        (0.46, "4F filtering", "iris position/size\nlens alignment"),
        (0.68, "Axicon", "lateral decentre • tip shape\nangle / refractive index"),
        (0.88, "Output", "transverse structure\npropagation signature"),
    ]
    widths = [0.14, 0.18, 0.15, 0.16, 0.10]
    for (x0, title, body), width in zip(blocks, widths):
        patch = FancyBboxPatch(
            (x0, 0.32),
            width,
            0.38,
            boxstyle="round,pad=0.012,rounding_size=0.015",
            linewidth=1.25,
            edgecolor="0.2",
            facecolor="0.97",
        )
        ax.add_patch(patch)
        ax.text(x0 + width / 2, 0.59, title, ha="center", va="center", fontsize=12, weight="bold")
        ax.text(x0 + width / 2, 0.43, body, ha="center", va="center", fontsize=9, color="0.25")

    for i in range(len(blocks) - 1):
        x_start = blocks[i][0] + widths[i] + 0.008
        x_end = blocks[i + 1][0] - 0.008
        ax.annotate(
            "",
            xy=(x_end, 0.51),
            xytext=(x_start, 0.51),
            arrowprops=dict(arrowstyle="-|>", lw=1.25, color="0.35"),
        )

    ax.text(
        0.5,
        0.84,
        "Moving from an ideal beam model towards the real optical system",
        ha="center",
        fontsize=15,
        weight="bold",
    )
    ax.text(
        0.5,
        0.12,
        "Change one physical parameter → predict the observable signature",
        ha="center",
        fontsize=11,
        color="0.25",
    )
    path = output_dir / "03_realistic_error_scope.png"
    _save(fig, path)
    return path


def build_v1_decentre(output_dir: Path, grid_n: int) -> Path:
    values = (-500e-6, 0.0, 500e-6)
    labels = ("−500 µm", "Aligned", "+500 µm")
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.0), constrained_layout=True)
    rows: list[dict[str, float]] = []

    for col, (value, label) in enumerate(zip(values, labels)):
        route = build_system_route(
            "V1",
            grid_n=int(grid_n),
            config=SystemErrorConfig(axicon=AxiconError(decentre_m=(float(value), 0.0))),
        )
        field_xy = _propagate_xy(route)
        axis = transverse_morphology_axis(
            field_xy,
            dict(route["grid"]),
            vortex_charge=1,
            seed_x_m=float(value),
            seed_y_m=0.0,
        )
        crop, extent = _crop_about(
            np.abs(field_xy) ** 2,
            dict(route["grid"]),
            centre_x_m=axis.x_m,
            centre_y_m=axis.y_m,
            halfwidth_m=190e-6,
        )
        _draw_intensity(
            axes[0, col],
            crop,
            extent=extent,
            xlabel=None,
            ylabel="Δy from vortex core (µm)" if col == 0 else None,
            title=label,
        )
        axes[0, col].axhline(0, color="white", alpha=0.30, linewidth=0.55)
        axes[0, col].axvline(0, color="white", alpha=0.30, linewidth=0.55)

        xz, track = _tracked_xz_for_x_decentre(
            route,
            case_id="V1",
            seed_x_m=float(value),
        )
        _draw_intensity(
            axes[1, col],
            xz.T,
            extent=[
                Z_VALUES_M[0] * 1e3,
                Z_VALUES_M[-1] * 1e3,
                MORPHOLOGY_OFFSETS_M[0] * 1e6,
                MORPHOLOGY_OFFSETS_M[-1] * 1e6,
            ],
            xlabel="z from axicon (mm)",
            ylabel="Δx from tracked core (µm)" if col == 0 else None,
            aspect="auto",
        )
        rows.append(
            {
                "decentre_m": float(value),
                "xy_topological_axis_x_m": float(axis.x_m),
                "xy_topological_axis_y_m": float(axis.y_m),
                "mean_longitudinal_axis_x_m": float(np.mean(track)),
                "longitudinal_axis_span_m": float(np.ptp(track)),
            }
        )

    fig.text(0.5, 0.995, "V1 sensitivity to lateral axicon decentre", ha="center", va="top", fontsize=16)
    fig.text(
        0.99,
        0.01,
        "Normalised intensity • morphology shown in the vortex-core frame",
        ha="right",
        va="bottom",
        fontsize=8,
        color="0.35",
    )
    path = output_dir / "04_v1_axicon_decentre.png"
    _save(fig, path)
    (output_dir / "04_v1_axicon_decentre_metrics.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    return path


def build_v1_rounded_tip(output_dir: Path, grid_n: int) -> Path:
    manifest = canonical_hardware_manifest()
    gamma = math.radians(float(hardware_value(manifest, "axicon_base_angle_deg")))
    radii = (0.0, 200e-6, 800e-6)
    labels = ("Ideal sharp tip", "200 µm rounding", "800 µm rounding")
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.0), constrained_layout=True)
    rows: list[dict[str, float | bool]] = []

    for col, (radius, label) in enumerate(zip(radii, labels)):
        if radius == 0.0:
            error = AxiconError(tip_model="sharp")
        else:
            error = AxiconError(
                tip_model="hyperboloidal_round",
                rounding_parameter_m=float(radius) * math.tan(gamma),
            )
        route = build_system_route(
            "V1",
            grid_n=int(grid_n),
            config=SystemErrorConfig(axicon=error),
        )
        resolution = tip_resolution(radius, float(route["grid"]["dx"]), minimum_pixels=12.0)
        if radius != 0.0 and not resolution.resolved:
            raise RuntimeError(
                f"rounded-tip presentation case {radius:g} m is only {resolution.radius_pixels:.2f} pixels"
            )

        field_xy = _propagate_xy(route)
        axis = transverse_morphology_axis(
            field_xy,
            dict(route["grid"]),
            vortex_charge=1,
            seed_x_m=0.0,
            seed_y_m=0.0,
        )
        crop, extent = _crop_about(
            np.abs(field_xy) ** 2,
            dict(route["grid"]),
            centre_x_m=axis.x_m,
            centre_y_m=axis.y_m,
            halfwidth_m=190e-6,
        )
        _draw_intensity(
            axes[0, col],
            crop,
            extent=extent,
            xlabel=None,
            ylabel="y (µm)" if col == 0 else None,
            title=label,
        )
        axes[0, col].axhline(0, color="white", alpha=0.30, linewidth=0.55)
        axes[0, col].axvline(0, color="white", alpha=0.30, linewidth=0.55)

        xz = _axisymmetric_xz(route)
        _draw_intensity(
            axes[1, col],
            xz.T,
            extent=[
                Z_VALUES_M[0] * 1e3,
                Z_VALUES_M[-1] * 1e3,
                MORPHOLOGY_OFFSETS_M[0] * 1e6,
                MORPHOLOGY_OFFSETS_M[-1] * 1e6,
            ],
            xlabel="z from axicon (mm)",
            ylabel="x (µm)" if col == 0 else None,
            aspect="auto",
        )
        rows.append(
            {
                "radial_rounding_scale_m": float(radius),
                "native_radius_pixels": float(resolution.radius_pixels),
                "passes_12_pixel_gate": bool(resolution.resolved),
            }
        )

    fig.text(0.5, 0.995, "Effect of a non-ideal rounded axicon tip", ha="center", va="top", fontsize=16)
    fig.text(
        0.99,
        0.01,
        "Normalised intensity • rounded-tip values are resolved sensitivity cases, not measured hardware defects",
        ha="right",
        va="bottom",
        fontsize=8,
        color="0.35",
    )
    path = output_dir / "05_v1_rounded_tip.png"
    _save(fig, path)
    (output_dir / "05_v1_rounded_tip_metrics.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build clean workshop-presentation figures.")
    parser.add_argument("--grid-n", type=int, default=1536)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/figures/conference_workshop"),
    )
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    generated = [
        build_simulation_pipeline(output, int(args.grid_n)),
        build_ideal_family(output, int(args.grid_n)),
        build_error_scope(output),
        build_v1_decentre(output, int(args.grid_n)),
        build_v1_rounded_tip(output, int(args.grid_n)),
    ]
    manifest = {
        "outcome": "CONFERENCE-WORKSHOP-PRESENTATION-FIGURES",
        "grid_n": int(args.grid_n),
        "generated": [path.name for path in generated],
        "physics_source": "validated phase2e refractive-axicon-physics branch",
        "presentation_only": True,
        "report_figures_authorised": False,
        "rigid_axicon_tilt_included": False,
        "notes": [
            "All intensity panels are regenerated from complex fields, not raster crops.",
            "V1 decentre transverse ROI is centred on the phase singularity and x-z follows the tracked vortex core.",
            "Rounded-tip nonzero radii must pass the 12-native-pixel resolution gate.",
            "The rounded-tip magnitudes are controlled sensitivity cases until physical profilometry is supplied.",
        ],
    }
    (output / "presentation_figure_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
