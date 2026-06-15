"""Zernike-perturbed Bessel cone test for hollow hexagonal writing."""

from __future__ import annotations

from dataclasses import replace
from math import factorial
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.special as sp

try:
    import bessel_twin_core as bt
    from vbb_study import setup_study, vbb_hex_outline, vbb_style
except ModuleNotFoundError:  # package import path
    from Publication_Study import bessel_twin_core as bt
    from Publication_Study.vbb_study import setup_study, vbb_hex_outline, vbb_style


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


def _zernike_radial(n: int, m: int, rho: np.ndarray) -> np.ndarray:
    n = int(n)
    m = abs(int(m))
    if (n - m) % 2:
        return np.zeros_like(rho, dtype=float)
    out = np.zeros_like(rho, dtype=float)
    for k in range((n - m) // 2 + 1):
        c = (-1) ** k * factorial(n - k) / (
            factorial(k) * factorial((n + m) // 2 - k) * factorial((n - m) // 2 - k)
        )
        out = out + c * rho ** (n - 2 * k)
    return out


def _zernike_phase(grid: dict[str, Any], pupil_radius_m: float, *, n: int, m: int, amp_rad: float, orientation_rad: float) -> np.ndarray:
    rho = np.clip(np.asarray(grid["R"], dtype=float) / max(float(pupil_radius_m), bt.EPS), 0.0, 1.0)
    theta = np.asarray(grid["PHI"], dtype=float) - float(orientation_rad)
    radial = _zernike_radial(int(n), int(m), rho)
    angular = np.cos(abs(int(m)) * theta)
    return float(amp_rad) * radial * angular * (rho <= 1.0)


def _lab_zernike_bessel_case(
    twin: bt.TwinConfig,
    *,
    ell: int,
    flat_radius_um: float,
    line_sigma_um: float,
    zernike_n: int,
    zernike_m: int,
    zernike_amp_rad: float,
    orientation_rad: float,
    z_eval_m: np.ndarray,
) -> dict[str, Any]:
    """Build and score one lab-realistic holographic Bessel/Zernike phase."""

    jprime = float(sp.jnp_zeros(abs(int(ell)), 1)[0])
    kr_sample = jprime / (float(flat_radius_um) * bt.um)
    target_core = 2.0 * 2.405 / max(kr_sample, bt.EPS)
    cfg = replace(
        twin,
        target=replace(
            twin.target,
            ell=int(ell),
            target_core_diameter_m=target_core,
            target_bessel_length_m=150.0 * bt.um,
        ),
    )
    design = bt.compute_design_from_targets(cfg.laser, cfg.target, cfg.material)
    pupil_grid, amp, meta = vbb_hex_outline.lab_pupil_amplitude(cfg)
    phase = -float(design.kr_slm_m_inv) * np.asarray(pupil_grid["R"], dtype=float)
    phase = phase + int(ell) * np.asarray(pupil_grid["PHI"], dtype=float)
    phase = phase + _zernike_phase(
        pupil_grid,
        cfg.objective.pupil_radius_m,
        n=int(zernike_n),
        m=int(zernike_m),
        amp_rad=float(zernike_amp_rad),
        orientation_rad=float(orientation_rad),
    )
    phase = phase + bt.interface_correction_phase(pupil_grid, cfg.laser, cfg.objective, cfg.material)
    encoded = bt.quantize_phase(phase, cfg.slm.phase_bits)
    U = amp * np.exp(1j * encoded)
    U = U * np.exp(1j * bt.interface_aberration_pupil(pupil_grid, cfg.laser, cfg.objective, cfg.material))
    focus, focal_grid = bt.focus_to_focal_plane(U, pupil_grid, cfg.laser, cfg.objective)

    outline_cfg = vbb_hex_outline.HexOutlineConfig(
        flat_radius_m=float(flat_radius_um) * bt.um,
        line_sigma_m=float(line_sigma_um) * bt.um,
        roi_margin_m=18.0 * bt.um,
        threshold_fraction=0.35,
    )
    z_case = vbb_hex_outline.propagate_outline_case_z(
        focus,
        focal_grid,
        outline_cfg,
        z_eval_m,
        wavelength_m=cfg.laser.wavelength_m,
        n_medium=cfg.material.refractive_index,
    )
    gray = bt.phase_to_gray(encoded, cfg.slm.phase_bits, invert=cfg.slm.invert_gray)
    rect_grid = meta["rect_grid"]
    ny = int(rect_grid["ny"])
    nx = int(rect_grid["nx"])
    y0 = (int(pupil_grid["N"]) - ny) // 2
    x0 = (int(pupil_grid["N"]) - nx) // 2
    lab = {
        "field": focus,
        "grid": focal_grid,
        "encoded_phase": encoded,
        "gray": gray,
        "gray_rect": gray[y0 : y0 + ny, x0 : x0 + nx],
        "metrics": vbb_hex_outline.hex_outline_metrics(focus, focal_grid, outline_cfg),
        "config": outline_cfg,
    }
    return {
        "label": f"ell{ell}_Z{zernike_n}_{zernike_m}_amp{zernike_amp_rad:+.2f}",
        "ell": int(ell),
        "kr_sample_per_um": float(kr_sample * bt.um),
        "zernike_n": int(zernike_n),
        "zernike_m": int(zernike_m),
        "zernike_amp_rad": float(zernike_amp_rad),
        "outline_config": outline_cfg,
        "lab": lab,
        "z_case": z_case,
        "design": design,
    }


def _select_plot_cases(df: pd.DataFrame, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selectors = [
        ("best z-stack score", int(df["score"].idxmax())),
        ("best focal outline F1", int(df["focus_outline_f1"].idxmax())),
        ("best focal side control", int(df["focus_penalized_score"].idxmax())),
    ]
    selected = []
    seen = set()
    for tag, idx in selectors:
        if idx in seen:
            continue
        tagged = dict(cases[idx])
        tagged["plot_tag"] = tag
        selected.append(tagged)
        seen.add(idx)
    for idx in np.argsort(-df["score"].to_numpy()):
        idx = int(idx)
        if len(selected) >= 3:
            break
        if idx in seen:
            continue
        tagged = dict(cases[idx])
        tagged["plot_tag"] = "next best z-stack score"
        selected.append(tagged)
        seen.add(idx)
    return selected


def _plot_best(cases: list[dict[str, Any]], output_path: Path) -> Path:
    best_cases = cases[: min(3, len(cases))]
    z_plot_um = (0.0, 20.0, 40.0, 60.0, 80.0)
    vbb_style.apply_style()
    fig, axes = plt.subplots(len(best_cases), len(z_plot_um), figsize=(3.0 * len(z_plot_um), 3.2 * len(best_cases)), constrained_layout=True)
    if len(best_cases) == 1:
        axes = axes[None, :]
    for row, case in enumerate(best_cases):
        stack = np.asarray(case["z_case"]["intensity_stack"], dtype=float)
        grid = case["lab"]["grid"]
        x_um = np.asarray(grid["x"], dtype=float) / bt.um
        z_values_um = np.asarray(case["z_case"]["z_values_m"], dtype=float) / bt.um
        cfg = case["outline_config"]
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
            metric = case["z_case"]["metrics_z"]["rows"][iz]
            tag = case.get("plot_tag", "candidate")
            ax.set_title(f"{tag}\n{case['label']}\nz={z_values_um[iz]:.0f} um, F1={metric['outline_f1']:.2f}", fontsize=8)
            ax.set_xlim(-18.0, 18.0)
            ax.set_ylim(-18.0, 18.0)
            ax.set_aspect("equal")
            ax.set_xlabel("x [um]")
            ax.set_ylabel("y [um]")
    fig.suptitle("Zernike-perturbed Bessel cone: lab-realistic z survival", fontsize=14)
    caption = (
        "Lab-realistic holographic axicon/vortex Bessel cone plus sixfold Zernike phase mask. "
        "Rows show the best z-stack score, best single-plane outline F1, and best side-lobe-penalized focal case."
    )
    out = vbb_style.save_figure(
        fig,
        output_path,
        caption,
        metadata={"figure": "zernike_hex_bessel_sweep"},
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
    z_eval = np.linspace(0.0, 100.0 * bt.um, 21)
    cases = []
    rows = []
    for ell in (4, 6, 8):
        for n, m in ((6, 6), (8, 6), (10, 6)):
            for amp in (-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0):
                case = _lab_zernike_bessel_case(
                    twin,
                    ell=ell,
                    flat_radius_um=7.0,
                    line_sigma_um=0.90,
                    zernike_n=n,
                    zernike_m=m,
                    zernike_amp_rad=amp,
                    orientation_rad=0.0,
                    z_eval_m=z_eval,
                )
                summary = case["z_case"]["metrics_z"]
                row = {
                    "label": case["label"],
                    "ell": case["ell"],
                    "kr_sample_per_um": case["kr_sample_per_um"],
                    "zernike_n": case["zernike_n"],
                    "zernike_m": case["zernike_m"],
                    "zernike_amp_rad": case["zernike_amp_rad"],
                    "accepted_depth_um": summary["accepted_depth_um"],
                    "accepted_plane_count": summary["accepted_plane_count"],
                    "mean_outline_f1": summary["mean_outline_f1"],
                    "min_outline_f1": summary["min_outline_f1"],
                    "max_core_peak_ratio": summary["max_core_peak_ratio"],
                    "max_side_lobe_peak_ratio": summary["max_side_lobe_peak_ratio"],
                    "focus_outline_f1": case["lab"]["metrics"]["outline_f1"],
                    "focus_core_peak_ratio": case["lab"]["metrics"]["core_peak_ratio"],
                    "focus_side_lobe_peak_ratio": case["lab"]["metrics"]["side_lobe_peak_ratio"],
                }
                rows.append(row)
                cases.append(case)
    df = pd.DataFrame(rows)
    df["focus_penalized_score"] = (
        df["focus_outline_f1"].astype(float)
        - df["focus_core_peak_ratio"].astype(float)
        - df["focus_side_lobe_peak_ratio"].astype(float)
    )
    df["score"] = (
        df["accepted_depth_um"].astype(float)
        + 10.0 * df["mean_outline_f1"].astype(float)
        - 2.0 * df["max_side_lobe_peak_ratio"].astype(float)
        - 2.0 * df["max_core_peak_ratio"].astype(float)
    )
    order = np.argsort(-df["score"].to_numpy())
    ordered_cases = [cases[int(i)] for i in order]
    df_sorted = df.iloc[order].reset_index(drop=True)
    csv_path = out["csv"] / "17_zernike_hex_bessel_sweep.csv"
    df_sorted.to_csv(csv_path, index=False)
    fig_path = _plot_best(_select_plot_cases(df, cases), out["figures"] / "17_zernike_hex_bessel_sweep.png")
    best_case = ordered_cases[0]
    cgh_paths = vbb_hex_outline.export_outline_hologram(
        best_case["lab"],
        out["holograms"],
        label=f"zernike_{best_case['label']}_hex_bessel",
    )
    manifest = setup_study.write_run_manifest(
        out["json"] / "17_zernike_hex_bessel_run_manifest.json",
        config={
            "twin": twin,
            "flat_radius_um": 7.0,
            "line_sigma_um": 0.90,
            "z_eval_um": (z_eval / bt.um).tolist(),
            "best": df_sorted.iloc[0].to_dict(),
        },
        paths={"csv": csv_path, "figure": fig_path, "cgh": cgh_paths},
        root=paths["root"],
    )
    return {"summary": df_sorted, "csv": csv_path, "figure": fig_path, "hologram": cgh_paths, "manifest": manifest}


if __name__ == "__main__":
    bundle = run_checkpoint()
    print(bundle["summary"].head(12).to_string(index=False))
    print(f"CSV: {bundle['csv']}")
    print(f"Figure: {bundle['figure']}")
    print(f"Hologram: {bundle['hologram']['phase_png']}")
