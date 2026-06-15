"""Hybrid hollow-hexagon outline checkpoint using a transient sixfold seed."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import bessel_twin_core as bt
from vbb_study import (
    setup_study,
    vbb_hex_outline,
    vbb_hexagon_metrics,
    vbb_hexagon_study,
    vbb_polarized_train,
    vbb_style,
)


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


def _interp_image_to_grid(image: np.ndarray, src_grid: Mapping[str, Any], dst_grid: Mapping[str, Any]) -> np.ndarray:
    src_x = np.asarray(src_grid["x"], dtype=float)
    dst_x = np.asarray(dst_grid["x"], dtype=float)
    arr = np.asarray(image)
    rows = np.empty((arr.shape[0], len(dst_x)), dtype=arr.dtype)
    for iy in range(arr.shape[0]):
        if np.iscomplexobj(arr):
            rows[iy] = np.interp(dst_x, src_x, arr[iy].real, left=0.0, right=0.0) + 1j * np.interp(
                dst_x, src_x, arr[iy].imag, left=0.0, right=0.0
            )
        else:
            rows[iy] = np.interp(dst_x, src_x, arr[iy], left=0.0, right=0.0)
    out = np.empty((len(dst_x), len(dst_x)), dtype=arr.dtype)
    for ix in range(len(dst_x)):
        if np.iscomplexobj(rows):
            out[:, ix] = np.interp(dst_x, src_x, rows[:, ix].real, left=0.0, right=0.0) + 1j * np.interp(
                dst_x, src_x, rows[:, ix].imag, left=0.0, right=0.0
            )
        else:
            out[:, ix] = np.interp(dst_x, src_x, rows[:, ix], left=0.0, right=0.0)
    return out


def _find_ring_radius_m(intensity: np.ndarray, grid: Mapping[str, Any]) -> float:
    R = np.asarray(grid["R"], dtype=float)
    bins = np.linspace(0.0, float(np.max(R)), 260)
    centres = 0.5 * (bins[:-1] + bins[1:])
    idx = np.clip(np.digitize(R.ravel(), bins) - 1, 0, bins.size - 2)
    radial = np.zeros_like(centres)
    counts = np.zeros_like(centres)
    np.add.at(radial, idx, np.asarray(intensity, dtype=float).ravel())
    np.add.at(counts, idx, 1.0)
    radial /= np.maximum(counts, 1.0)
    search = centres > 0.10 * float(np.max(R))
    return float(centres[search][np.argmax(radial[search])]) if np.any(search) else float(centres[np.argmax(radial)])


def _build_transient_seed(target_grid: Mapping[str, Any]) -> dict[str, Any]:
    """Return a screenshot-like transient caustic phase seed on the target grid."""

    config = vbb_polarized_train.PolarizedTrainConfig(
        N=320,
        dx_m=0.18 * bt.um,
        n_axicon=1.46,
        axicon_base_angle_deg=32.0,
        segment_count=12,
        z_max_m=110.0 * bt.um,
        z_points=111,
    )
    transient = vbb_hexagon_study.run_segmented_vector_hexagon(config, realism="ideal")
    state = transient["state"]
    z_values = np.linspace(0.0, 110.0 * bt.um, 111)
    prop = vbb_polarized_train.vector_angular_spectrum_propagate(
        state.Ex,
        state.Ey,
        state.grid,
        Ez=state.Ez,
        wavelength_m=config.wavelength_m,
        n_medium=config.n_medium,
        z_values_m=z_values,
    )
    z_index = int(np.argmin(np.abs(z_values / bt.um - 81.0)))
    field = prop["fields"][z_index]
    intensity = vbb_polarized_train.total_intensity_3d(field)
    central = np.asarray(state.grid["R"], dtype=float) <= 18.0 * bt.um
    components = {
        "Ex": field.Ex,
        "Ey": field.Ey,
        "Ez": field.Ez,
        "Ex_plus_Ey": field.Ex + field.Ey,
        "Ex_iEy": field.Ex + 1j * field.Ey,
    }
    best_name = "Ex"
    best_corr = -np.inf
    for name, component in components.items():
        a = (np.abs(component) ** 2)[central].ravel()
        b = intensity[central].ravel()
        corr = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0.0 and np.std(b) > 0.0 else -np.inf
        if corr > best_corr:
            best_name = name
            best_corr = corr

    unit_src = np.exp(1j * np.angle(components[best_name]))
    unit = _interp_image_to_grid(unit_src, state.grid, target_grid)
    unit = unit / (np.abs(unit) + bt.EPS)
    intensity_norm = intensity / (float(np.max(intensity[central])) + bt.EPS)
    intensity_target_grid = _interp_image_to_grid(intensity_norm, state.grid, target_grid)
    ring_r = _find_ring_radius_m(intensity, state.grid)
    six = vbb_hexagon_metrics.sixfold_from_intensity(
        intensity,
        np.asarray(state.grid["R"], dtype=float),
        np.asarray(state.grid["PHI"], dtype=float),
        ring_r,
    )
    return {
        "unit_phase": unit,
        "intensity": intensity_target_grid,
        "source_intensity": intensity,
        "source_grid": state.grid,
        "component": best_name,
        "component_intensity_correlation": best_corr,
        "z_um": float(z_values[z_index] / bt.um),
        "order6_over_order0": float(six["order6_over_order0"]),
    }


def _score(df: pd.DataFrame) -> pd.Series:
    return (
        df["outline_f1"].astype(float)
        * df["outline_energy_fraction"].astype(float)
        * df["edge_uniformity"].astype(float)
        / ((0.03 + df["core_peak_ratio"].astype(float)) * (0.05 + df["side_lobe_peak_ratio"].astype(float)))
    )


def _plot(
    cases: list[dict[str, Any]],
    seed: Mapping[str, Any],
    target_grid: Mapping[str, Any],
    output_path: Path,
) -> Path:
    vbb_style.apply_style()
    fig, axes = plt.subplots(2, 4, figsize=(14.0, 7.2), constrained_layout=True)
    for row, line_family in enumerate(("balanced", "wide")):
        cfg_ref = next(case["config"] for case in cases if case["line_family"] == line_family)
        target = vbb_hex_outline.hex_outline_target(target_grid, cfg_ref)
        panels = [
            ("transient seed intensity", seed["intensity"], target_grid, None),
            ("literal target", target["target_amplitude"] ** 2, target_grid, cfg_ref),
        ]
        for seed_label in ("random", "transient_phase_seed"):
            case = next(case for case in cases if case["line_family"] == line_family and case["seed"] == seed_label)
            m = case["lab"]["metrics"]
            panels.append(
                (
                    f"{line_family} {seed_label}\nF1={m['outline_f1']:.2f}, side={m['side_lobe_peak_ratio']:.2f}",
                    np.abs(case["lab"]["field"]) ** 2,
                    case["lab"]["grid"],
                    case["config"],
                )
            )
        for col, (title, image, grid, cfg_plot) in enumerate(panels):
            ax = axes[row, col]
            x_um = np.asarray(grid["x"], dtype=float) / bt.um
            arr = np.asarray(image, dtype=float)
            mask = np.abs(x_um) <= 18.0
            peak = float(np.max(arr[np.ix_(mask, mask)])) + bt.EPS if np.any(mask) else float(np.max(arr)) + bt.EPS
            ax.imshow(
                vbb_style.display_scale(arr / peak, gamma=0.55, normalise=False),
                origin="lower",
                extent=[float(x_um[0]), float(x_um[-1]), float(x_um[0]), float(x_um[-1])],
                cmap=vbb_style.INTENSITY_CMAP,
                vmin=0.0,
                vmax=1.0,
            )
            if cfg_plot is not None:
                verts = vbb_hex_outline.hex_vertices(cfg_plot) / bt.um
                closed = np.vstack([verts, verts[0]])
                ax.plot(closed[:, 0], closed[:, 1], color="white", lw=0.9, alpha=0.9)
            ax.set_xlim(-18.0, 18.0)
            ax.set_ylim(-18.0, 18.0)
            ax.set_aspect("equal")
            ax.set_xlabel("x [um]")
            ax.set_ylabel("y [um]")
            ax.set_title(title, fontsize=9)
    fig.suptitle("Lab-realistic hollow hex outline: random seed vs transient-caustic seed", fontsize=14)
    caption = (
        "Hybrid checkpoint: transient sixfold caustic used as the focal-plane phase seed for a hollow regular-hexagon outline, "
        "then passed through quantized/interface-corrected lab propagation."
    )
    out = vbb_style.save_figure(
        fig,
        output_path,
        caption,
        metadata={"figure": "hybrid_transient_seed_lab_gate"},
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
    seed = _build_transient_seed(target_grid)

    rows: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    for line_family, sigma_um, random_seed in (("balanced", 0.65, 12346), ("wide", 0.90, 12347)):
        config = vbb_hex_outline.HexOutlineConfig(
            flat_radius_m=7.0 * bt.um,
            line_sigma_m=sigma_um * bt.um,
            retrieval_iterations=260,
            roi_margin_m=16.0 * bt.um,
            random_seed=random_seed,
        )
        target = vbb_hex_outline.hex_outline_target(target_grid, config)
        target_amp = target["target_amplitude"]
        seeded_focus = target_amp * seed["unit_phase"]
        hybrid_amp = target_amp * (0.80 + 0.20 * np.sqrt(np.clip(seed["intensity"], 0.0, 1.0)))
        seeds = [
            ("random", None),
            ("transient_phase_seed", seeded_focus),
            ("transient_phase_amp_seed", hybrid_amp * seed["unit_phase"]),
        ]
        for seed_label, focus_seed in seeds:
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
            row = {
                "candidate": f"{line_family}_{seed_label}",
                "line_family": line_family,
                "seed": seed_label,
                "line_fwhm_um": 2.355 * sigma_um,
                "transient_seed_component": seed["component"],
                "transient_seed_component_corr": seed["component_intensity_correlation"],
                "transient_seed_z_um": seed["z_um"],
                "transient_seed_order6_over_order0": seed["order6_over_order0"],
                **lab["metrics"],
            }
            rows.append(row)
            cases.append({"line_family": line_family, "seed": seed_label, "config": config, "ideal": ideal, "lab": lab})

    df = pd.DataFrame(rows)
    df["outline_score"] = _score(df)
    csv_path = out["csv"] / "15_hybrid_transient_seed_lab_gate.csv"
    df.to_csv(csv_path, index=False)
    fig_path = _plot(cases, seed, target_grid, out["figures"] / "15_hybrid_transient_seed_lab_gate.png")

    best = df.sort_values("outline_score", ascending=False).iloc[0]
    best_case = next(case for case in cases if f"{case['line_family']}_{case['seed']}" == best["candidate"])
    cgh_paths = vbb_hex_outline.export_outline_hologram(
        best_case["lab"],
        out["holograms"],
        label=f"hybrid_{best['candidate']}_hollow_hex_outline",
    )
    manifest = setup_study.write_run_manifest(
        out["json"] / "15_hybrid_transient_seed_run_manifest.json",
        config={
            "twin": twin,
            "best_candidate": str(best["candidate"]),
            "seed": {
                "component": seed["component"],
                "component_intensity_correlation": seed["component_intensity_correlation"],
                "z_um": seed["z_um"],
                "order6_over_order0": seed["order6_over_order0"],
            },
        },
        paths={"csv": csv_path, "figure": fig_path, "cgh": cgh_paths},
        root=paths["root"],
    )
    return {"csv": csv_path, "figure": fig_path, "hologram": cgh_paths, "manifest": manifest, "metrics": df}


if __name__ == "__main__":
    bundle = run_checkpoint()
    cols = [
        "candidate",
        "line_fwhm_um",
        "outline_f1",
        "core_peak_ratio",
        "side_lobe_peak_ratio",
        "outline_energy_fraction",
        "edge_uniformity",
        "side_balance",
        "outline_score",
    ]
    print(bundle["metrics"].sort_values("outline_score", ascending=False)[cols].to_string(index=False))
    print(f"CSV: {bundle['csv']}")
    print(f"Figure: {bundle['figure']}")
    print(f"Hologram: {bundle['hologram']['phase_png']}")
