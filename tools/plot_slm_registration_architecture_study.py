#!/usr/bin/env python3
"""Plot the two-SLM registration architecture comparison study."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "outputs" / "validation" / "slm_registration_architecture"
DEFAULT_FIG = ROOT / "outputs" / "figures" / "slm_registration_architecture"

ARCH_LABELS = {
    "upstream_vortex": "A: SLM1 vortex; SLM2 correction + carrier/blaze",
    "downstream_vortex": "B: SLM1 correction; SLM2 vortex + carrier/blaze",
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _metric(df: pd.DataFrame) -> str:
    for name in (
        "translation_registered_infidelity",
        "post_iris_translation_registered_infidelity",
        "infidelity",
        "post_iris_infidelity",
    ):
        if name in df.columns:
            return name
    raise KeyError("no registration metric found")


def plot_offset_curves(df: pd.DataFrame, outdir: Path) -> Path:
    metric = _metric(df)
    # Representative but source-grounded: charge 20, w=50 if present.
    charge = 20 if 20 in set(df["charge"]) else int(df["charge"].max())
    radius = 50.0 if np.any(np.isclose(df["beam_radius_px"], 50.0)) else float(df["beam_radius_px"].iloc[0])
    axis = "x" if "x" in set(df["axis"]) else str(df["axis"].iloc[0])

    sub = df[
        (df["charge"] == charge)
        & np.isclose(df["beam_radius_px"], radius)
        & (df["axis"] == axis)
    ].copy()

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    for architecture, group in sub.groupby("architecture"):
        for panel, g in group.groupby("panel_dof"):
            g = g.sort_values("offset_fraction_px")
            axs[0].plot(
                g["offset_fraction_px"],
                g[metric],
                marker="o",
                label=f"{architecture}: {panel}",
            )
    axs[0].set_xlabel("Registration shift / pixel pitch")
    axs[0].set_ylabel(metric.replace("_", " "))
    axs[0].set_title(f"Registration sensitivity: L={charge}, w={radius:g} px, {axis}")
    axs[0].grid(True, alpha=0.25)
    axs[0].legend(fontsize=7)

    # Direct architecture ratio for each panel at maximum tested offset.
    maxoff = sub["offset_fraction_px"].max()
    edge = sub[np.isclose(sub["offset_fraction_px"], maxoff)]
    panels = [p for p in ("slm1", "slm2", "common", "differential") if p in set(edge["panel_dof"])]
    ratios = []
    for panel in panels:
        a = edge[(edge["architecture"] == "upstream_vortex") & (edge["panel_dof"] == panel)]
        b = edge[(edge["architecture"] == "downstream_vortex") & (edge["panel_dof"] == panel)]
        if len(a) and len(b):
            av = float(a.iloc[0][metric])
            bv = float(b.iloc[0][metric])
            ratios.append(bv / max(av, 1e-30))
        else:
            ratios.append(np.nan)
    axs[1].bar(np.arange(len(panels)), ratios)
    axs[1].axhline(1.0, ls="--", lw=1)
    axs[1].set_xticks(np.arange(len(panels)), panels, rotation=25)
    axs[1].set_ylabel("Architecture B / Architecture A sensitivity")
    axs[1].set_title(f"Direct architecture comparison at {maxoff:.3f} pixel")
    axs[1].grid(True, axis="y", alpha=0.25)

    path = outdir / "01_offset_curves_and_architecture_ratio.png"
    fig.savefig(path, dpi=320, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_beam_charge_maps(df: pd.DataFrame, outdir: Path) -> list[Path]:
    metric = _metric(df)
    # Worst morphology change over registration DOFs/axes/offsets at each L,w.
    grouped = (
        df.groupby(["architecture", "charge", "beam_radius_px"], as_index=False)[metric]
        .max()
    )
    paths: list[Path] = []
    arch_mats = {}
    for architecture in sorted(grouped["architecture"].unique()):
        g = grouped[grouped["architecture"] == architecture]
        pivot = g.pivot(index="charge", columns="beam_radius_px", values=metric)
        arch_mats[architecture] = pivot
        fig, ax = plt.subplots(figsize=(8.6, 5.4), constrained_layout=True)
        arr = pivot.to_numpy()
        im = ax.imshow(arr, origin="lower", aspect="auto", cmap="viridis")
        ax.set_xticks(np.arange(pivot.shape[1]), [f"{v:g}" for v in pivot.columns])
        ax.set_yticks(np.arange(pivot.shape[0]), [f"{int(v)}" for v in pivot.index])
        ax.set_xlabel("Beam radius on SLM (pixels)")
        ax.set_ylabel("Vortex charge L")
        ax.set_title(f"Worst-case registration sensitivity\n{ARCH_LABELS.get(architecture, architecture)}")
        plt.colorbar(im, ax=ax, pad=0.01, label=metric.replace("_", " "))
        path = outdir / f"02_beam_charge_map_{architecture}.png"
        fig.savefig(path, dpi=320, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    if {"upstream_vortex", "downstream_vortex"} <= set(arch_mats):
        a, b = arch_mats["upstream_vortex"].align(
            arch_mats["downstream_vortex"], join="inner", axis=None
        )
        ratio = b.to_numpy() / np.maximum(a.to_numpy(), 1e-30)
        fig, ax = plt.subplots(figsize=(8.6, 5.4), constrained_layout=True)
        vmax = max(np.nanmax(ratio), 1.0)
        vmin = min(np.nanmin(ratio), 1.0)
        if vmin < 1.0 < vmax:
            norm = TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)
        else:
            norm = None
        im = ax.imshow(ratio, origin="lower", aspect="auto", cmap="coolwarm", norm=norm)
        ax.set_xticks(np.arange(a.shape[1]), [f"{v:g}" for v in a.columns])
        ax.set_yticks(np.arange(a.shape[0]), [f"{int(v)}" for v in a.index])
        ax.set_xlabel("Beam radius on SLM (pixels)")
        ax.set_ylabel("Vortex charge L")
        ax.set_title("Architecture sensitivity ratio: downstream-vortex / upstream-vortex")
        plt.colorbar(im, ax=ax, pad=0.01, label="B / A")
        path = outdir / "03_architecture_ratio_beam_charge.png"
        fig.savefig(path, dpi=320, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def plot_unit_cell(df: pd.DataFrame, outdir: Path) -> list[Path]:
    if df.empty:
        return []
    metric = _metric(df)
    paths = []
    for (architecture, charge, radius, panel), g in df.groupby(
        ["architecture", "charge", "beam_radius_px", "panel"]
    ):
        pivot = g.pivot(index="dy_px", columns="dx_px", values=metric).sort_index().sort_index(axis=1)
        fig, ax = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
        im = ax.imshow(
            pivot.to_numpy(),
            origin="lower",
            extent=[
                float(pivot.columns.min()),
                float(pivot.columns.max()),
                float(pivot.index.min()),
                float(pivot.index.max()),
            ],
            aspect="equal",
            cmap="viridis",
        )
        ax.set_xlabel("x registration / pixel")
        ax.set_ylabel("y registration / pixel")
        ax.set_title(
            f"Pixel-unit-cell sensitivity: {architecture}, {panel}\n"
            f"L={int(charge)}, w={float(radius):g} px"
        )
        plt.colorbar(im, ax=ax, pad=0.01, label=metric.replace("_", " "))
        path = outdir / (
            f"04_unit_cell_{architecture}_{panel}_L{int(charge)}_w{float(radius):g}.png"
        )
        fig.savefig(path, dpi=320, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def plot_interpanel(df: pd.DataFrame, outdir: Path) -> list[Path]:
    if df.empty:
        return []
    metric = _metric(df)
    paths = []
    for architecture, g in df.groupby("architecture"):
        pivot = (
            g.pivot(
                index="slm2_offset_px",
                columns="slm1_offset_px",
                values=metric,
            )
            .sort_index()
            .sort_index(axis=1)
        )
        fig, ax = plt.subplots(figsize=(6.4, 5.4), constrained_layout=True)
        im = ax.imshow(
            pivot.to_numpy(),
            origin="lower",
            extent=[
                float(pivot.columns.min()),
                float(pivot.columns.max()),
                float(pivot.index.min()),
                float(pivot.index.max()),
            ],
            aspect="equal",
            cmap="viridis",
        )
        ax.set_xlabel("SLM1 registration / pixel")
        ax.set_ylabel("SLM2 registration / pixel")
        ax.set_title(
            f"Relative SLM1-SLM2 registration\n{ARCH_LABELS.get(architecture, architecture)}"
        )
        plt.colorbar(im, ax=ax, pad=0.01, label=metric.replace("_", " "))
        path = outdir / f"05_interpanel_{architecture}.png"
        fig.savefig(path, dpi=320, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def plot_axial(df: pd.DataFrame, outdir: Path) -> Path:
    """Plot propagated Bessel-region sensitivity for a representative case."""

    charge = 20 if 20 in set(df["charge"]) else int(df["charge"].max())
    radius = 50.0 if np.any(np.isclose(df["beam_radius_px"], 50.0)) else float(df["beam_radius_px"].iloc[0])
    sub = df[
        (df["charge"] == charge)
        & np.isclose(df["beam_radius_px"], radius)
    ].copy()

    fig, axs = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)
    for architecture, ga in sub.groupby("architecture"):
        for panel, g in ga.groupby("panel"):
            g = g.sort_values("offset_fraction_px")
            label = f"{architecture}: {panel}"
            axs[0].plot(
                g["offset_fraction_px"],
                g["peak_intensity_ratio"],
                marker="o",
                label=label,
            )
            axs[1].plot(
                g["offset_fraction_px"],
                g["bessel_zone_ratio"],
                marker="o",
                label=label,
            )
            morph = (
                "propagated_translation_registered_infidelity"
                if "propagated_translation_registered_infidelity" in g.columns
                else "propagated_infidelity"
            )
            axs[2].plot(
                g["offset_fraction_px"],
                g[morph],
                marker="o",
                label=label,
            )

    axs[0].axhline(1.0, ls="--", lw=1)
    axs[1].axhline(1.0, ls="--", lw=1)
    axs[0].set_ylabel("Peak intensity / zero-registration")
    axs[1].set_ylabel("Bessel-zone length / zero-registration")
    axs[2].set_ylabel("Propagated morphology infidelity")
    for ax in axs:
        ax.set_xlabel("Registration shift / pixel pitch")
        ax.grid(True, alpha=0.25)
    axs[0].set_title("Peak intensity")
    axs[1].set_title("Bessel-zone length")
    axs[2].set_title("XY field at reference z")
    axs[0].legend(fontsize=7)
    fig.suptitle(f"Post-axicon sensitivity: L={charge}, w={radius:g} px")
    path = outdir / "07_axial_registration_sensitivity.png"
    fig.savefig(path, dpi=320, bbox_inches="tight")
    plt.close(fig)
    return path


def _crop(arr: np.ndarray, x_m: np.ndarray, half_width_um: float) -> tuple[np.ndarray, np.ndarray]:
    mask = np.abs(x_m) <= float(half_width_um) * 1e-6
    idx = np.where(mask)[0]
    return arr[np.ix_(idx, idx)], x_m[idx] * 1e6


def plot_xy(npz_path: Path, outdir: Path, half_width_um: float = 300.0) -> list[Path]:
    if not npz_path.exists():
        return []
    data = np.load(npz_path)
    paths = []
    for architecture in ARCH_LABELS:
        for panel in ("slm1", "slm2"):
            prefix = f"{architecture}__{panel}__"
            key = prefix + "field_pre_axicon"
            if key not in data:
                continue
            offsets = data[prefix + "offset_fraction_px"]
            x_m = data[prefix + "x_m"]
            pre = data[prefix + "field_pre_axicon"]
            post = data[prefix + "field_at_reference_z"]

            for plane_name, stack in (
                ("pre_axicon", pre),
                ("after_axicon_propagated", post),
            ):
                ref = np.abs(stack[0]) ** 2
                scale = float(np.max(ref))
                fig, axs = plt.subplots(
                    2,
                    len(offsets),
                    figsize=(3.0 * len(offsets), 6.0),
                    constrained_layout=True,
                )
                for j, (frac, field) in enumerate(zip(offsets, stack)):
                    I = np.abs(field) ** 2
                    Icrop, xx = _crop(I, x_m, half_width_um)
                    Rcrop, _ = _crop(
                        (I - ref) / max(scale, 1e-30),
                        x_m,
                        half_width_um,
                    )
                    axs[0, j].imshow(
                        Icrop / max(scale, 1e-30),
                        origin="lower",
                        extent=[xx[0], xx[-1], xx[0], xx[-1]],
                        cmap="inferno",
                        vmin=0,
                    )
                    lim = max(float(np.max(np.abs(Rcrop))), 1e-12)
                    axs[1, j].imshow(
                        Rcrop,
                        origin="lower",
                        extent=[xx[0], xx[-1], xx[0], xx[-1]],
                        cmap="coolwarm",
                        norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim),
                    )
                    axs[0, j].set_title(f"{frac:.3f} pixel")
                    axs[1, j].set_title("signed residual")
                    for ax in (axs[0, j], axs[1, j]):
                        ax.set_xlabel("x (um)")
                        ax.set_ylabel("y (um)")
                fig.suptitle(
                    f"{ARCH_LABELS[architecture]} | {panel.upper()} registration | "
                    f"{plane_name.replace('_', ' ')} XY profiles"
                )
                path = outdir / f"06_xy_{architecture}_{panel}_{plane_name}.png"
                fig.savefig(path, dpi=320, bbox_inches="tight")
                plt.close(fig)
                paths.append(path)
    return paths

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="core")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA))
    parser.add_argument("--figure-root", default=str(DEFAULT_FIG))
    parser.add_argument("--xy-half-width-um", type=float, default=300.0)
    args = parser.parse_args()

    datadir = Path(args.data_root) / args.tag
    outdir = Path(args.figure_root) / args.tag
    outdir.mkdir(parents=True, exist_ok=True)

    made: list[Path] = []
    sweep_path = datadir / "architecture_sweep.csv"
    if sweep_path.exists():
        sweep = _read_csv(sweep_path)
        made.append(plot_offset_curves(sweep, outdir))
        made.extend(plot_beam_charge_maps(sweep, outdir))

    unit_path = datadir / "unit_cell_maps.csv"
    if unit_path.exists():
        made.extend(plot_unit_cell(_read_csv(unit_path), outdir))

    axial_path = datadir / "axial_metrics.csv"
    if axial_path.exists():
        made.append(plot_axial(_read_csv(axial_path), outdir))

    inter_path = datadir / "interpanel_registration_map.csv"
    if inter_path.exists():
        made.extend(plot_interpanel(_read_csv(inter_path), outdir))

    made.extend(
        plot_xy(
            datadir / "representative_xy_fields.npz",
            outdir,
            half_width_um=args.xy_half_width_um,
        )
    )

    print("Generated figures:")
    for path in made:
        print(path)


if __name__ == "__main__":
    main()
