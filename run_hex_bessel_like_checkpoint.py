"""Multi-plane hollow hexagon checkpoint for Bessel-like writing."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import bessel_twin_core as bt
from vbb_study import setup_study, vbb_hex_outline, vbb_style

try:
    from run_hex_outline_hybrid_checkpoint import _build_transient_seed
except ModuleNotFoundError:  # package import path
    from Publication_Study.run_hex_outline_hybrid_checkpoint import _build_transient_seed


def _out_tree(paths: dict[str, Path]) -> dict[str, Path]:
    base = paths["outputs"]
    out = {
        "figures": base / "figures" / "hex_outline",
        "csv": base / "csv" / "hex_outline",
        "json": base / "json" / "hex_outline",
        "holograms": base / "holograms" / "hex_outline",
    }
    for path in out.values():
        path.mkdir(parents=True, exist_ok=True)
    return out


def _row_from_summary(candidate: str, path: str, config: vbb_hex_outline.HexOutlineConfig, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate": candidate,
        "path": path,
        "line_fwhm_um": 2.355 * config.line_sigma_m / bt.um,
        "accepted_depth_um": summary["accepted_depth_um"],
        "accepted_plane_count": summary["accepted_plane_count"],
        "mean_outline_f1": summary["mean_outline_f1"],
        "min_outline_f1": summary["min_outline_f1"],
        "max_core_peak_ratio": summary["max_core_peak_ratio"],
        "max_side_lobe_peak_ratio": summary["max_side_lobe_peak_ratio"],
        "accepted_any": summary["accepted_any"],
        "accepted_z_start_um": summary["accepted_z_start_um"],
        "accepted_z_end_um": summary["accepted_z_end_um"],
    }


def _plot_stack(
    cases: list[dict[str, Any]],
    z_plot_um: tuple[float, ...],
    output_path: Path,
) -> Path:
    vbb_style.apply_style()
    fig, axes = plt.subplots(len(cases), len(z_plot_um), figsize=(3.0 * len(z_plot_um), 3.2 * len(cases)), constrained_layout=True)
    if len(cases) == 1:
        axes = axes[None, :]
    for row, case in enumerate(cases):
        grid = case["lab"]["grid"]
        x_um = np.asarray(grid["x"], dtype=float) / bt.um
        stack = np.asarray(case["lab_z"]["intensity_stack"], dtype=float)
        z_values_um = np.asarray(case["lab_z"]["z_values_m"], dtype=float) / bt.um
        cfg = case["config"]
        for col, z_um in enumerate(z_plot_um):
            iz = int(np.argmin(np.abs(z_values_um - float(z_um))))
            img = stack[iz]
            mask = np.abs(x_um) <= 18.0
            peak = float(np.max(img[np.ix_(mask, mask)])) + bt.EPS
            ax = axes[row, col]
            ax.imshow(
                vbb_style.display_scale(img / peak, gamma=0.55, normalise=False),
                origin="lower",
                extent=[float(x_um[0]), float(x_um[-1]), float(x_um[0]), float(x_um[-1])],
                cmap=vbb_style.INTENSITY_CMAP,
                vmin=0.0,
                vmax=1.0,
            )
            verts = vbb_hex_outline.hex_vertices(cfg) / bt.um
            closed = np.vstack([verts, verts[0]])
            ax.plot(closed[:, 0], closed[:, 1], color="white", lw=0.8, alpha=0.85)
            metric = case["lab_z"]["metrics_z"]["rows"][iz]
            ax.set_title(
                f"{case['label']}\nz={z_values_um[iz]:.0f} um, F1={metric['outline_f1']:.2f}",
                fontsize=8,
            )
            ax.set_xlim(-18.0, 18.0)
            ax.set_ylim(-18.0, 18.0)
            ax.set_aspect("equal")
            ax.set_xlabel("x [um]")
            ax.set_ylabel("y [um]")
    fig.suptitle("Lab-realistic multi-plane hollow hexagon outline survival", fontsize=14)
    caption = (
        "Multi-plane checkpoint for a Bessel-like hollow hexagonal writing beam. "
        "The same outline is scored at multiple forward z planes after quantized/interface-corrected lab propagation."
    )
    out = vbb_style.save_figure(
        fig,
        output_path,
        caption,
        metadata={"figure": "hex_bessel_like_multiplane_survival"},
    )
    plt.close(fig)
    return out


def run_checkpoint() -> dict[str, Any]:
    paths = setup_study.bootstrap(Path(__file__))
    out = _out_tree(paths)
    twin = bt.default_config("fast")
    twin = replace(
        twin,
        grid=replace(twin.grid, N=512, crop_pixels=192, device_downsample=4),
        apply_interface=True,
        correct_interface=True,
    )
    pupil_grid, pupil_amp, _ = vbb_hex_outline.lab_pupil_amplitude(twin)
    target_grid = vbb_hex_outline.focus_grid_from_pupil(twin, pupil_grid)
    transient_seed = _build_transient_seed(target_grid)
    z_design = np.linspace(0.0, 70.0 * bt.um, 8)
    z_eval = np.linspace(0.0, 90.0 * bt.um, 19)

    cases: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    z_rows: list[dict[str, Any]] = []
    specs = [
        ("wide_single_plane_seeded", 0.90, False),
        ("wide_multiplane_seeded", 0.90, True),
        ("balanced_multiplane_seeded", 0.65, True),
    ]
    for idx, (label, sigma_um, use_multiplane) in enumerate(specs):
        config = vbb_hex_outline.HexOutlineConfig(
            flat_radius_m=7.0 * bt.um,
            line_sigma_m=sigma_um * bt.um,
            retrieval_iterations=120 if use_multiplane else 260,
            roi_margin_m=18.0 * bt.um,
            random_seed=22340 + idx,
        )
        target = vbb_hex_outline.hex_outline_target(target_grid, config)
        focus_seed = target["target_amplitude"] * transient_seed["unit_phase"]
        if use_multiplane:
            ideal = vbb_hex_outline.phase_retrieve_outline_multiplane(
                pupil_amp,
                config,
                target_grid=target_grid,
                z_values_m=z_design,
                wavelength_m=twin.laser.wavelength_m,
                n_medium=twin.material.refractive_index,
                initial_focus_field=focus_seed,
            )
        else:
            ideal = vbb_hex_outline.phase_retrieve_outline(
                pupil_amp,
                config,
                target_grid=target_grid,
                initial_focus_field=focus_seed,
            )
        lab = vbb_hex_outline.build_lab_outline_case(
            twin,
            config,
            ideal["pupil_phase"],
            correct_interface=True,
            quantize=True,
            include_interface=True,
        )
        lab_z = vbb_hex_outline.propagate_outline_case_z(
            lab["field"],
            lab["grid"],
            config,
            z_eval,
            wavelength_m=twin.laser.wavelength_m,
            n_medium=twin.material.refractive_index,
        )
        cases.append({"label": label, "config": config, "ideal": ideal, "lab": lab, "lab_z": lab_z})
        summary_rows.append(_row_from_summary(label, "lab_corrected_quantized", config, lab_z["metrics_z"]))
        for row in lab_z["metrics_z"]["rows"]:
            z_rows.append({"candidate": label, **row})

    summary = pd.DataFrame(summary_rows)
    summary_csv = out["csv"] / "16_hex_bessel_like_summary.csv"
    z_csv = out["csv"] / "16_hex_bessel_like_z_profile.csv"
    summary.to_csv(summary_csv, index=False)
    pd.DataFrame(z_rows).to_csv(z_csv, index=False)
    fig_path = _plot_stack(cases, (0.0, 20.0, 40.0, 60.0, 80.0), out["figures"] / "16_hex_bessel_like_multiplane_survival.png")

    best = summary.sort_values(["accepted_depth_um", "mean_outline_f1"], ascending=False).iloc[0]
    best_case = next(case for case in cases if case["label"] == best["candidate"])
    cgh_paths = vbb_hex_outline.export_outline_hologram(
        best_case["lab"],
        out["holograms"],
        label=f"bessel_like_{best['candidate']}_hollow_hex_outline",
    )
    manifest = setup_study.write_run_manifest(
        out["json"] / "16_hex_bessel_like_run_manifest.json",
        config={
            "twin": twin,
            "z_design_um": (z_design / bt.um).tolist(),
            "z_eval_um": (z_eval / bt.um).tolist(),
            "transient_seed": {
                "component": transient_seed["component"],
                "component_intensity_correlation": transient_seed["component_intensity_correlation"],
                "z_um": transient_seed["z_um"],
                "order6_over_order0": transient_seed["order6_over_order0"],
            },
            "best_candidate": str(best["candidate"]),
        },
        paths={"summary_csv": summary_csv, "z_csv": z_csv, "figure": fig_path, "cgh": cgh_paths},
        root=paths["root"],
    )
    return {
        "summary": summary,
        "summary_csv": summary_csv,
        "z_csv": z_csv,
        "figure": fig_path,
        "hologram": cgh_paths,
        "manifest": manifest,
    }


if __name__ == "__main__":
    bundle = run_checkpoint()
    print(bundle["summary"].sort_values(["accepted_depth_um", "mean_outline_f1"], ascending=False).to_string(index=False))
    print(f"Summary: {bundle['summary_csv']}")
    print(f"Z profile: {bundle['z_csv']}")
    print(f"Figure: {bundle['figure']}")
    print(f"Hologram: {bundle['hologram']['phase_png']}")
