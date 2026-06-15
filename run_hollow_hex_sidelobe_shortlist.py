"""Run a hollow-hexagon side-lobe suppression shortlist.

This is the follow-on to the broad "hexagon" sweep: it focuses on the
fabrication-facing target, namely one hollow hexagonal ring with a dark core
and weak side lobes. The comparison deliberately includes both ideal
continuous-ring fields and the current lab-realistic phase-only SLM encoding.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import bessel_twin_core as bt
from vbb_study import setup_study, vbb_polygonal, vbb_style


def _out_tree(paths: dict[str, Path]) -> dict[str, Path]:
    base = paths["outputs"]
    out = {
        "figures": base / "figures" / "polygonal_hex_ring",
        "csv": base / "csv" / "polygonal_hex_ring",
    }
    for path in out.values():
        path.mkdir(parents=True, exist_ok=True)
    return out


def _config(corner_sharpness: float, k_width_fraction: float, ring_width_um: float) -> vbb_polygonal.PolygonalVortexConfig:
    return vbb_polygonal.PolygonalVortexConfig(
        N=6,
        flat_radius_m=7.0 * bt.um,
        ring_width_m=float(ring_width_um) * bt.um,
        kr_m_inv=1.2 / bt.um,
        charge=2,
        orientation_rad=0.0,
        corner_sharpness=float(corner_sharpness),
        k_ring_width_fraction=float(k_width_fraction),
        refinement_iterations=60,
        angular_samples=2048,
    )


def _row(
    candidate: str,
    path: str,
    phase_source: str,
    config: vbb_polygonal.PolygonalVortexConfig,
    case: dict[str, Any],
) -> dict[str, Any]:
    metrics = case["metrics"]
    side = vbb_polygonal.hollow_hex_side_lobe_metrics(case["field"], case["grid"], config)
    return {
        "candidate": candidate,
        "path": path,
        "phase_source": phase_source,
        "corner_sharpness": config.corner_sharpness,
        "k_ring_width_fraction": config.k_ring_width_fraction,
        "ring_width_um": config.ring_width_m / bt.um,
        **side,
        "contour_order_fidelity": metrics["contour_order_fidelity"],
        "edge_uniformity": metrics["edge_uniformity"],
        "core_null_depth": metrics["core_null_depth"],
        "corner_to_edge_fluence_ratio": metrics["corner_to_edge_fluence_ratio"],
        "sixfold_ring_pass": metrics["sixfold_ring_pass"],
        "single_closed_contour_pass": metrics["single_closed_contour_pass"],
        "encoded_power_fraction": float(case.get("encoded_power_fraction", np.nan)),
    }


def _plot_shortlist(
    images: list[tuple[str, str, str, np.ndarray, dict[str, Any], dict[str, float], dict[str, Any]]],
    candidates: list[tuple[str, float, float, float]],
    output_path: str | Path,
) -> Path:
    vbb_style.apply_style()
    fig, axes = plt.subplots(len(candidates), 3, figsize=(10.8, 12.4), constrained_layout=True)
    for row, (candidate, *_unused) in enumerate(candidates):
        ordered = []
        for wanted in (("ideal", "refined"), ("lab", "analytic"), ("lab", "refined")):
            for item in images:
                if item[0] == candidate and item[1] == wanted[0] and item[2] == wanted[1]:
                    ordered.append(item)
                    break
        for col, (label, path, phase_source, intensity, grid, side, metrics) in enumerate(ordered):
            ax = axes[row, col]
            x_um = np.asarray(grid["x"], dtype=float) / bt.um
            ax.imshow(
                vbb_style.display_scale(intensity, gamma=0.55),
                origin="lower",
                extent=[float(x_um[0]), float(x_um[-1]), float(x_um[0]), float(x_um[-1])],
                cmap=vbb_style.INTENSITY_CMAP,
                vmin=0.0,
                vmax=1.0,
            )
            ax.set_xlim(-22.0, 22.0)
            ax.set_ylim(-22.0, 22.0)
            ax.set_aspect("equal")
            ax.set_xlabel("x [um]")
            ax.set_ylabel("y [um]")
            ax.set_title(
                (
                    f"{label}: {path} {phase_source}\n"
                    f"side={side['side_lobe_peak_ratio']:.2f}, "
                    f"core={side['core_peak_ratio']:.2f}, "
                    f"hex={metrics['contour_order_fidelity']:.2f}"
                ),
                fontsize=9,
            )
    fig.suptitle("Hollow hexagon side-lobe shortlist: ideal vs lab-realistic encoding", fontsize=14)
    caption = (
        "Shortlist comparing hollow hexagonal vortex-ring candidates. "
        "Ideal smoothing lowers side lobes at the cost of hexagon fidelity; "
        "the current lab-realistic phase-only encoding mostly returns a rounded annulus."
    )
    out = vbb_style.save_figure(
        fig,
        output_path,
        caption,
        metadata={"figure": "hollow_hex_sidelobe_lab_shortlist"},
    )
    plt.close(fig)
    return out


def run_shortlist() -> dict[str, Any]:
    paths = setup_study.bootstrap(Path(__file__))
    out = _out_tree(paths)
    twin = bt.default_config("fast")
    twin = replace(
        twin,
        grid=replace(twin.grid, N=512, crop_pixels=192, device_downsample=4),
        target=replace(twin.target, target_bessel_length_m=150.0 * bt.um),
        apply_interface=True,
        correct_interface=True,
    )
    grid = bt.make_xy_grid(384, 0.20 * bt.um)
    z0 = np.asarray([0.0])
    candidates = [
        ("checkpoint", 0.75, 0.08, 1.00),
        ("shape_pass", 0.30, 0.12, 1.30),
        ("balanced_low_halo", 0.75, 0.17, 1.65),
        ("smooth_lowest_halo", 0.30, 0.23, 1.65),
    ]

    rows: list[dict[str, Any]] = []
    images: list[tuple[str, str, str, np.ndarray, dict[str, Any], dict[str, float], dict[str, Any]]] = []
    for candidate, corner_sharpness, k_width_fraction, ring_width_um in candidates:
        config = _config(corner_sharpness, k_width_fraction, ring_width_um)
        ideal = vbb_polygonal.build_ideal_case(
            grid,
            config,
            wavelength_m=twin.laser.wavelength_m,
            n_medium=twin.material.refractive_index,
            z_values_m=z0,
            crop_pixels=256,
        )
        rows.append(_row(candidate, "ideal", "refined_field", config, ideal))
        ideal_side = vbb_polygonal.hollow_hex_side_lobe_metrics(ideal["field"], ideal["grid"], config)
        images.append(
            (
                candidate,
                "ideal",
                "refined",
                np.abs(ideal["field"]) ** 2,
                ideal["grid"],
                ideal_side,
                ideal["metrics"],
            )
        )
        for phase_source, phase_table in (
            ("analytic", ideal["analytic_phase_table"]),
            ("refined", ideal["refined_phase_table"]),
        ):
            lab = vbb_polygonal.build_lab_realistic_case(
                twin,
                config,
                phase_table,
                z_values_m=z0,
                crop_pixels=192,
            )
            rows.append(_row(candidate, "lab_realistic", phase_source, config, lab))
            lab_side = vbb_polygonal.hollow_hex_side_lobe_metrics(lab["field"], lab["grid"], config)
            images.append(
                (
                    candidate,
                    "lab",
                    phase_source,
                    np.abs(lab["field"]) ** 2,
                    lab["grid"],
                    lab_side,
                    lab["metrics"],
                )
            )

    df = pd.DataFrame(rows)
    df["hollow_hex_score"] = (
        df["ring_energy_fraction"].astype(float)
        * df["contour_order_fidelity"].astype(float)
        * df["edge_uniformity"].astype(float)
        / ((0.04 + df["side_lobe_peak_ratio"].astype(float)) * (0.03 + df["core_peak_ratio"].astype(float)))
    )
    csv_path = out["csv"] / "12_hollow_hex_sidelobe_lab_shortlist.csv"
    df.to_csv(csv_path, index=False)
    figure_path = _plot_shortlist(images, candidates, out["figures"] / "12_hollow_hex_sidelobe_lab_shortlist.png")
    return {"csv": csv_path, "figure": figure_path, "shortlist": df}


if __name__ == "__main__":
    bundle = run_shortlist()
    columns = [
        "candidate",
        "path",
        "phase_source",
        "side_lobe_peak_ratio",
        "core_peak_ratio",
        "ring_energy_fraction",
        "contour_order_fidelity",
        "edge_uniformity",
        "sixfold_ring_pass",
        "hollow_hex_score",
    ]
    print(bundle["shortlist"].sort_values("hollow_hex_score", ascending=False)[columns].to_string(index=False))
    print(f"CSV: {bundle['csv']}")
    print(f"Figure: {bundle['figure']}")
