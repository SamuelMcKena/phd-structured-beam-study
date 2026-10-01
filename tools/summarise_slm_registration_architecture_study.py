#!/usr/bin/env python3
"""Print compact headline tables from the SLM registration architecture study."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "outputs" / "validation" / "slm_registration_architecture"


def _metric(df: pd.DataFrame) -> str:
    for name in (
        "translation_registered_infidelity",
        "post_iris_translation_registered_infidelity",
        "infidelity",
        "post_iris_infidelity",
    ):
        if name in df.columns:
            return name
    raise KeyError("no supported infidelity metric found")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="core")
    p.add_argument("--data-root", default=str(DEFAULT_DATA))
    args = p.parse_args()

    datadir = Path(args.data_root) / args.tag
    sweep = pd.read_csv(datadir / "architecture_sweep.csv")
    metric = _metric(sweep)

    print("\n=== Study contract ===")
    print("A: SLM1 vortex; SLM2 correction + carrier/blaze")
    print("B: SLM1 correction; SLM2 vortex + carrier/blaze")
    print("SLM2 carrier/blaze present in BOTH architectures")
    print(f"headline metric: {metric}")

    edge = sweep[sweep["offset_fraction_px"] > 0].copy()
    worst = (
        edge.groupby(["architecture", "charge", "beam_radius_px"], as_index=False)[metric]
        .max()
        .sort_values(["beam_radius_px", "charge", "architecture"])
    )
    print("\n=== Worst registration sensitivity by architecture, L and beam radius ===")
    print(worst.to_string(index=False))

    if {"upstream_vortex", "downstream_vortex"} <= set(worst["architecture"]):
        a = worst[worst["architecture"] == "upstream_vortex"].rename(
            columns={metric: "A"}
        )
        b = worst[worst["architecture"] == "downstream_vortex"].rename(
            columns={metric: "B"}
        )
        ratio = a.merge(
            b,
            on=["charge", "beam_radius_px"],
            suffixes=("_A", "_B"),
        )
        ratio["B_over_A"] = ratio["B"] / np.maximum(ratio["A"], 1e-30)
        print("\n=== Direct architecture ratio (B/A) ===")
        print(
            ratio[["charge", "beam_radius_px", "A", "B", "B_over_A"]]
            .sort_values(["beam_radius_px", "charge"])
            .to_string(index=False)
        )

    print("\n=== Panel/axis worst cases ===")
    panel = (
        edge.groupby(["architecture", "panel_dof", "axis"], as_index=False)[metric]
        .max()
        .sort_values(["architecture", metric], ascending=[True, False])
    )
    print(panel.to_string(index=False))

    unit = datadir / "unit_cell_maps.csv"
    if unit.exists():
        u = pd.read_csv(unit)
        umetric = _metric(u)
        uworst = (
            u.groupby(["architecture", "panel", "charge", "beam_radius_px"], as_index=False)
            .agg(
                worst=(umetric, "max"),
                mean=(umetric, "mean"),
                p95=(umetric, lambda s: float(np.nanpercentile(s, 95.0))),
            )
        )
        print("\n=== Pixel-unit-cell sensitivity ===")
        print(uworst.to_string(index=False))

    inter = datadir / "interpanel_registration_map.csv"
    if inter.exists():
        d = pd.read_csv(inter)
        dmetric = _metric(d)
        print("\n=== Relative SLM1-SLM2 registration map extrema ===")
        print(
            d.groupby("architecture", as_index=False)[dmetric]
            .agg(["min", "mean", "max"])
            .to_string()
        )


if __name__ == "__main__":
    main()
