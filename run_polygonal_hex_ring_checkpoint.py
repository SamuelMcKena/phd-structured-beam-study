"""Generate the localized hexagonal vortex-ring checkpoint outputs."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vbb_study import setup_study, vbb_discrete, vbb_polygonal, vbb_style, vbb_materials

import bessel_twin_core as bt
from Publication_Study.finalize_publication_outputs import finalize_outputs


def _out_tree(paths: dict[str, Path]) -> dict[str, Path]:
    base = paths["outputs"]
    out = {
        "figures": base / "figures" / "polygonal_hex_ring",
        "csv": base / "csv" / "polygonal_hex_ring",
        "json": base / "json" / "polygonal_hex_ring",
        "holograms": base / "holograms" / "polygonal_hex_ring",
    }
    for path in out.values():
        path.mkdir(parents=True, exist_ok=True)
    return out


def _row_from_case(label: str, case: dict[str, Any], *, include_path: str) -> dict[str, Any]:
    metrics = case["metrics"]
    return {
        "case": label,
        "path": include_path,
        "component_count": metrics["component_count"],
        "single_closed_contour_pass": metrics["single_closed_contour_pass"],
        "localization_fraction": metrics["localization_fraction"],
        "localization_pass": metrics["localization_pass"],
        "lattice_autocorr_side_peak": metrics["lattice_autocorr_side_peak"],
        "no_lattice_periodicity_pass": metrics["no_lattice_periodicity_pass"],
        "core_null_depth": metrics["core_null_depth"],
        "dark_core_pass": metrics["dark_core_pass"],
        "phase_winding": metrics["phase_winding"],
        "oam_pass": metrics["oam_pass"],
        "contour_order_fidelity": metrics["contour_order_fidelity"],
        "sixfold_ring_pass": metrics["sixfold_ring_pass"],
        "edge_uniformity": metrics["edge_uniformity"],
        "corner_to_edge_fluence_ratio": metrics["corner_to_edge_fluence_ratio"],
        "encoded_power_fraction": float(case.get("encoded_power_fraction", np.nan)),
    }


def plot_lattice_negative_control(
    grid: dict[str, Any],
    lattice_field: np.ndarray,
    polygon_case: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Save the old lattice next to the new localized ring."""

    vbb_style.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.0), constrained_layout=True)
    images = [np.abs(lattice_field) ** 2, np.abs(polygon_case["field"]) ** 2]
    titles = ["old discrete lattice", "new localized hex ring"]
    vmax = max(float(np.max(img)) for img in images)
    artist = None
    for ax, img, title in zip(axes, images, titles):
        x_um = np.asarray(grid["x"], dtype=float) / bt.um
        artist = ax.imshow(
            vbb_style.display_scale(img / (vmax + bt.EPS), gamma=0.45, normalise=False),
            origin="lower",
            extent=[float(x_um[0]), float(x_um[-1]), float(x_um[0]), float(x_um[-1])],
            cmap=vbb_style.INTENSITY_CMAP,
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_title(title)
        ax.set_xlabel("x [um, sample plane]")
        ax.set_ylabel("y [um, sample plane]")
    cb = fig.colorbar(artist, ax=axes, shrink=0.92)
    cb.set_label("matched display intensity, gamma=0.45 [a.u.]")
    caption = (
        "Negative control distinguishing the old finite-plane-wave kaleidoscope lattice from the new continuous-ring localized hexagonal vortex beam. "
        "Both carry sixfold structure, but only the new field is one closed ring with intensity decaying outside it."
    )
    out = vbb_style.save_figure(fig, output_path, caption, metadata={"figure": "polygonal_lattice_negative_control"})
    plt.close(fig)
    return out


def run_checkpoint() -> dict[str, Any]:
    paths = setup_study.bootstrap(Path(__file__))
    out = _out_tree(paths)
    base = bt.default_config("fast")
    base = replace(
        base,
        grid=replace(base.grid, N=512, crop_pixels=192, device_downsample=4),
        target=replace(base.target, target_bessel_length_m=150.0 * bt.um),
        apply_interface=True,
        correct_interface=True,
    )
    config = vbb_polygonal.PolygonalVortexConfig(
        N=6,
        flat_radius_m=7.0 * bt.um,
        ring_width_m=1.0 * bt.um,
        kr_m_inv=1.2 / bt.um,
        charge=2,
        orientation_rad=0.0,
        corner_sharpness=0.75,
        k_ring_width_fraction=0.08,
        refinement_iterations=60,
        angular_samples=2048,
    )
    grid = bt.make_xy_grid(384, 0.20 * bt.um)
    z_values = np.linspace(0.0, 100.0 * bt.um, 41)
    ideal = vbb_polygonal.build_ideal_case(
        grid,
        config,
        wavelength_m=base.laser.wavelength_m,
        n_medium=base.material.refractive_index,
        z_values_m=z_values,
        crop_pixels=256,
    )
    lab = vbb_polygonal.build_lab_realistic_case(
        base,
        config,
        ideal["analytic_phase_table"],
        z_values_m=z_values,
        crop_pixels=192,
    )
    pattern = vbb_discrete.pattern_preset("hexagonal_vortex", vortex_charge=config.charge)
    lattice_field = vbb_discrete.n_wave_complex_field(grid, config.kr_m_inv, pattern, waist_m=35.0 * bt.um)
    lattice_metrics = vbb_polygonal.polygonal_ring_metrics(lattice_field, grid, config)
    lattice_case = {"metrics": lattice_metrics, "encoded_power_fraction": 1.0}

    rows = [
        _row_from_case("localized_hex_ring", ideal, include_path="ideal"),
        _row_from_case("localized_hex_ring", lab, include_path="lab_realistic"),
        _row_from_case("discrete_lattice_negative_control", lattice_case, include_path="ideal_lattice_reference"),
    ]
    acceptance = pd.DataFrame(rows)
    acceptance["all_required_pass"] = (
        acceptance["single_closed_contour_pass"]
        & acceptance["localization_pass"]
        & acceptance["no_lattice_periodicity_pass"]
        & acceptance["dark_core_pass"]
        & acceptance["oam_pass"]
        & acceptance["sixfold_ring_pass"]
    )
    acceptance.loc[acceptance["case"].str.contains("negative"), "all_required_pass"] = False
    acceptance_csv = out["csv"] / vbb_style.csv_name(11, "polygonal_hex_ring", "acceptance_metrics")
    acceptance.to_csv(acceptance_csv, index=False)

    stability_rows = []
    for label, case in (("ideal", ideal), ("lab_realistic", lab)):
        df = vbb_polygonal.propagation_stability(case, config)
        df.insert(0, "path", label)
        stability_rows.append(df)
    stability = pd.concat(stability_rows, ignore_index=True)
    stability_csv = out["csv"] / vbb_style.csv_name(11, "polygonal_hex_ring", "z_stability")
    stability.to_csv(stability_csv, index=False)

    threshold = vbb_materials.incubated_threshold_J_cm2(base.material, base.laser.rep_rate_Hz)
    mat_rows = []
    for label, case in (("ideal", ideal), ("lab_realistic", lab)):
        mat = vbb_polygonal.material_hex_ring_metrics(case, base.energy.pulse_energy_at_sample_J, threshold)
        mat_rows.append(
            {
                "path": label,
                "threshold_area_um2": mat["threshold_area_um2"],
                "edge_fluence_uniformity": mat["edge_fluence_uniformity"],
                "corner_to_edge_fluence_ratio": mat["corner_to_edge_fluence_ratio"],
                "dark_core_radius_um": mat["dark_core_radius_um"],
            }
        )
    material = pd.DataFrame(mat_rows)
    material_csv = out["csv"] / vbb_style.csv_name(11, "polygonal_hex_ring", "materials_proxy")
    material.to_csv(material_csv, index=False)

    checkpoint_fig = vbb_polygonal.plot_hex_ring_checkpoint(
        ideal,
        lab,
        out["figures"] / vbb_style.figure_name(11, "polygonal_hex_ring", "ideal_lab_xy_xz"),
    )
    negative_fig = plot_lattice_negative_control(
        grid,
        lattice_field,
        ideal,
        out["figures"] / vbb_style.figure_name(11, "polygonal_hex_ring", "lattice_negative_control"),
    )
    cgh_paths = vbb_polygonal.export_lab_cgh(lab, out["holograms"], label="hexagonal_vortex_ring")
    manifest = setup_study.write_run_manifest(
        out["json"] / "11_polygonal_hex_ring_run_manifest.json",
        config={"polygonal": config, "twin": base},
        paths={
            "acceptance_csv": acceptance_csv,
            "stability_csv": stability_csv,
            "material_csv": material_csv,
            "checkpoint_fig": checkpoint_fig,
            "negative_fig": negative_fig,
            "cgh": cgh_paths,
        },
        extra={"old_discrete_label": "discrete lattice (kaleidoscope) -- not a localized ring"},
        root=paths["root"],
    )
    finalize_outputs(paths["outputs"])
    return {
        "ideal": ideal,
        "lab": lab,
        "lattice_metrics": lattice_metrics,
        "acceptance": acceptance,
        "stability": stability,
        "material": material,
        "acceptance_csv": acceptance_csv,
        "stability_csv": stability_csv,
        "material_csv": material_csv,
        "checkpoint_fig": checkpoint_fig,
        "negative_fig": negative_fig,
        "cgh_paths": cgh_paths,
        "manifest": manifest,
    }


if __name__ == "__main__":
    bundle = run_checkpoint()
    print(f"Acceptance: {bundle['acceptance_csv']}")
    print(f"Stability: {bundle['stability_csv']}")
    print(f"Materials: {bundle['material_csv']}")
    print(f"Figure: {bundle['checkpoint_fig']}")
    print(f"Negative control: {bundle['negative_fig']}")
    print(f"CGH: {bundle['cgh_paths']['phase_png']}")
