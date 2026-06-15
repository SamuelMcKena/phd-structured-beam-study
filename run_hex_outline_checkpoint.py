"""Generate the direct hollow-hexagon outline checkpoint outputs."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import bessel_twin_core as bt
from vbb_study import setup_study, vbb_hex_outline


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


def _row(label: str, path: str, config: vbb_hex_outline.HexOutlineConfig, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate": label,
        "path": path,
        "flat_radius_um": config.flat_radius_m / bt.um,
        "line_sigma_um": config.line_sigma_m / bt.um,
        "line_fwhm_um": 2.355 * config.line_sigma_m / bt.um,
        "outline_f1": metrics["outline_f1"],
        "outline_precision": metrics["outline_precision"],
        "outline_recall": metrics["outline_recall"],
        "core_peak_ratio": metrics["core_peak_ratio"],
        "side_lobe_peak_ratio": metrics["side_lobe_peak_ratio"],
        "outline_energy_fraction": metrics["outline_energy_fraction"],
        "edge_uniformity": metrics["edge_uniformity"],
        "side_balance": metrics["side_balance"],
        "component_count": metrics["component_count"],
        "single_outline_component_pass": metrics["single_outline_component_pass"],
        "outline_f1_pass": metrics["outline_f1_pass"],
        "dark_core_pass": metrics["dark_core_pass"],
        "side_lobe_pass": metrics["side_lobe_pass"],
        "outline_energy_pass": metrics["outline_energy_pass"],
        "edge_uniformity_pass": metrics["edge_uniformity_pass"],
        "side_balance_pass": metrics["side_balance_pass"],
    }


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
    pupil_grid, pupil_amp, _meta = vbb_hex_outline.lab_pupil_amplitude(twin)
    target_grid = vbb_hex_outline.focus_grid_from_pupil(twin, pupil_grid)
    candidates = [
        ("thin_outline", 0.45),
        ("balanced_outline", 0.65),
        ("wide_outline", 0.90),
    ]

    rows: list[dict[str, Any]] = []
    bundles: list[dict[str, Any]] = []
    for idx, (label, sigma_um) in enumerate(candidates):
        config = vbb_hex_outline.HexOutlineConfig(
            flat_radius_m=7.0 * bt.um,
            line_sigma_m=sigma_um * bt.um,
            retrieval_iterations=240,
            roi_margin_m=10.0 * bt.um,
            random_seed=12345 + idx,
        )
        target = vbb_hex_outline.hex_outline_target(target_grid, config)
        target_metrics = vbb_hex_outline.hex_outline_metrics(target["target_amplitude"] ** 2, target_grid, config)
        ideal = vbb_hex_outline.phase_retrieve_outline(
            pupil_amp,
            config,
            target_grid=target_grid,
        )
        lab = vbb_hex_outline.build_lab_outline_case(
            twin,
            config,
            ideal["pupil_phase"],
            correct_interface=True,
            quantize=True,
            include_interface=True,
        )
        rows.append(_row(label, "target_mask", config, target_metrics))
        rows.append(_row(label, "phase_only_ideal", config, ideal["metrics"]))
        rows.append(_row(label, "lab_corrected_quantized", config, lab["metrics"]))
        bundles.append({"label": label, "config": config, "target_grid": target_grid, "ideal": ideal, "lab": lab})

    df = pd.DataFrame(rows)
    df["outline_score"] = (
        df["outline_f1"].astype(float)
        * df["outline_energy_fraction"].astype(float)
        * df["edge_uniformity"].astype(float)
        / ((0.03 + df["core_peak_ratio"].astype(float)) * (0.05 + df["side_lobe_peak_ratio"].astype(float)))
    )
    csv_path = out["csv"] / "13_hollow_hex_outline_checkpoint.csv"
    df.to_csv(csv_path, index=False)

    fig_path = vbb_hex_outline.plot_outline_checkpoint(
        bundles,
        out["figures"] / "13_hollow_hex_outline_checkpoint.png",
    )
    lab_rows = df[df["path"] == "lab_corrected_quantized"].sort_values("outline_score", ascending=False)
    best_label = str(lab_rows.iloc[0]["candidate"])
    best_bundle = next(bundle for bundle in bundles if bundle["label"] == best_label)
    cgh_paths = vbb_hex_outline.export_outline_hologram(
        best_bundle["lab"],
        out["holograms"],
        label=f"{best_label}_hollow_hex_outline",
    )
    manifest = setup_study.write_run_manifest(
        out["json"] / "13_hollow_hex_outline_run_manifest.json",
        config={"twin": twin, "candidates": [bundle["config"] for bundle in bundles]},
        paths={"csv": csv_path, "figure": fig_path, "cgh": cgh_paths},
        root=paths["root"],
    )
    return {
        "csv": csv_path,
        "figure": fig_path,
        "hologram": cgh_paths,
        "manifest": manifest,
        "metrics": df,
    }


if __name__ == "__main__":
    bundle = run_checkpoint()
    cols = [
        "candidate",
        "path",
        "line_fwhm_um",
        "outline_f1",
        "core_peak_ratio",
        "side_lobe_peak_ratio",
        "outline_energy_fraction",
        "edge_uniformity",
        "outline_score",
    ]
    print(bundle["metrics"].sort_values("outline_score", ascending=False)[cols].to_string(index=False))
    print(f"CSV: {bundle['csv']}")
    print(f"Figure: {bundle['figure']}")
    print(f"Hologram: {bundle['hologram']['phase_png']}")
