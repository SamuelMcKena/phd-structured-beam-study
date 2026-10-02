#!/usr/bin/env python3
"""Build an academic Markdown report from registration architecture outputs.

The report is deliberately data-driven: it only states quantitative findings
that are present in the generated CSV/NPZ evidence.  Missing stages are reported
as missing rather than inferred.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "outputs" / "validation" / "slm_registration_architecture"
DEFAULT_FIG = ROOT / "outputs" / "figures" / "slm_registration_architecture"
DEFAULT_REPORT = ROOT / "outputs" / "reports" / "slm_registration_architecture"


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


def _frame_block(df: pd.DataFrame, *, index: bool = False) -> str:
    """Render a dataframe without requiring the optional tabulate package."""

    text = df.to_string(index=index)
    return "~~~\n" + text + "\n~~~"


def _fmt(x: float) -> str:
    x = float(x)
    if not np.isfinite(x):
        return "not available"
    if x == 0.0:
        return "0"
    if abs(x) < 1e-3 or abs(x) >= 1e3:
        return f"{x:.3e}"
    return f"{x:.4f}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="full")
    p.add_argument("--data-root", default=str(DEFAULT_DATA))
    p.add_argument("--figure-root", default=str(DEFAULT_FIG))
    p.add_argument("--report-root", default=str(DEFAULT_REPORT))
    args = p.parse_args()

    datadir = Path(args.data_root) / args.tag
    figdir = Path(args.figure_root) / args.tag
    reportdir = Path(args.report_root) / args.tag
    reportdir.mkdir(parents=True, exist_ok=True)

    manifest_path = datadir / "manifest.json"
    sweep_path = datadir / "architecture_sweep.csv"
    if not manifest_path.exists() or not sweep_path.exists():
        raise FileNotFoundError(
            "manifest.json and architecture_sweep.csv are required before a report can be built"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sweep = pd.read_csv(sweep_path)
    metric = _metric(sweep)
    edge = sweep[sweep["offset_fraction_px"] > 0].copy()

    worst = (
        edge.groupby(["architecture", "charge", "beam_radius_px"], as_index=False)[metric]
        .max()
    )
    overall = (
        edge.groupby("architecture", as_index=False)[metric]
        .max()
        .set_index("architecture")[metric]
        .to_dict()
    )

    ratio_text = "A direct architecture ratio could not be formed."
    ratio_table = ""
    if {"upstream_vortex", "downstream_vortex"} <= set(worst["architecture"]):
        a = worst[worst["architecture"] == "upstream_vortex"].rename(columns={metric: "A"})
        b = worst[worst["architecture"] == "downstream_vortex"].rename(columns={metric: "B"})
        joined = a.merge(b, on=["charge", "beam_radius_px"])
        joined["B_over_A"] = joined["B"] / np.maximum(joined["A"], 1e-30)
        finite = joined[np.isfinite(joined["B_over_A"])]
        if not finite.empty:
            row_min = finite.loc[finite["B_over_A"].idxmin()]
            row_max = finite.loc[finite["B_over_A"].idxmax()]
            ratio_text = (
                "Across the generated charge/beam-radius matrix, the downstream-vortex "
                f"architecture had B/A sensitivity ratios ranging from {_fmt(row_min['B_over_A'])} "
                f"(L={int(row_min['charge'])}, w={row_min['beam_radius_px']:g} px) to "
                f"{_fmt(row_max['B_over_A'])} "
                f"(L={int(row_max['charge'])}, w={row_max['beam_radius_px']:g} px)."
            )
        ratio_table = _frame_block(
            joined[["charge", "beam_radius_px", "A", "B", "B_over_A"]]
            .sort_values(["beam_radius_px", "charge"]),
            index=False,
        )

    unit_text = (
        "The two-dimensional pixel-unit-cell stage was not generated for this tag."
    )
    unit_path = datadir / "unit_cell_maps.csv"
    if unit_path.exists():
        unit = pd.read_csv(unit_path)
        umetric = _metric(unit)
        u = (
            unit.groupby(["architecture", "panel"], as_index=False)[umetric]
            .max()
            .sort_values(umetric, ascending=False)
        )
        unit_text = (
            "The generated pixel-unit-cell maps quantify centre/edge/corner sensitivity. "
            "The worst morphology metric by architecture/panel was:\n\n"
            + _frame_block(u, index=False)
        )

    inter_text = "The relative SLM1-SLM2 map was not generated for this tag."
    inter_path = datadir / "interpanel_registration_map.csv"
    if inter_path.exists():
        inter = pd.read_csv(inter_path)
        imetric = _metric(inter)
        table = (
            inter.groupby("architecture", as_index=False)[imetric]
            .agg(["min", "mean", "max"])
        )
        inter_text = (
            "The relative SLM1-SLM2 registration map gives the following extrema:\n\n"
            + _frame_block(table, index=True)
        )

    axial_text = "The post-axicon axial stage was not generated for this tag."
    axial_path = datadir / "axial_metrics.csv"
    if axial_path.exists():
        axial = pd.read_csv(axial_path)
        cols = [
            c
            for c in (
                "architecture",
                "charge",
                "beam_radius_px",
                "panel",
                "offset_fraction_px",
                "peak_intensity_ratio",
                "bessel_zone_ratio",
                "propagated_translation_registered_infidelity",
                "propagated_infidelity",
            )
            if c in axial.columns
        ]
        axial_text = (
            "The generated axial stage quantifies peak intensity, Bessel-zone "
            "length and propagated transverse morphology at the zero-registration "
            "reference-z plane. Representative rows are:\n\n"
            + _frame_block(axial[cols].head(30), index=False)
        )

    symmetry_text = (
        "Architecture label-symmetry evidence was not generated for this tag."
    )
    symmetry_path = datadir / "architecture_label_symmetry.csv"
    if symmetry_path.exists():
        sym = pd.read_csv(symmetry_path)
        max_delta = float(sym["abs_delta"].max()) if not sym.empty else float("nan")
        symmetry_text = (
            "Because both SLMs carry the same blaze and the present effective-channel "
            "model has no measured inter-SLM propagation/relay transform, swapping the "
            "vortex-owning panel is mathematically a panel relabelling in the flat-correction "
            "baseline. The generated symmetry gate gives a maximum A/B paired metric "
            f"difference of {_fmt(max_delta)}. Therefore this model can quantify local "
            "registration sensitivity but cannot, by itself, establish that upstream or "
            "downstream vortex generation is intrinsically more robust. Any architecture "
            "preference requires measured panel-specific geometry, calibration or correction."
        )

    correction_status = manifest.get("correction_map", "unknown")
    spp = manifest.get("sampling", {})
    amax = overall.get("upstream_vortex", float("nan"))
    bmax = overall.get("downstream_vortex", float("nan"))

    figures = []
    if figdir.exists():
        figures = sorted(figdir.glob("*.png"))

    fig_md = "\n".join(
        f"![{path.stem}]({os.path.relpath(path, start=reportdir)})"
        for path in figures
    )

    report = f"""# Sub-pixel SLM registration sensitivity in two dual-SLM vortex architectures

## Abstract

This study compares two phase-allocation strategies for the dual-SLM
vortex-Bessel bench while varying only the sub-pixel registration of the optical
field with respect to the physical SLM pixel lattices. Architecture A generates
the vortex on SLM1 and applies correction plus carrier/blaze on SLM2.
Architecture B applies correction on SLM1 and generates the vortex plus
carrier/blaze on SLM2. SLM2 therefore carries the carrier/blaze in both
architectures. Registration is implemented with compensated physical-panel and
electronic-pattern shifts so that beam-to-hologram centring remains nominal.

The primary morphology metric used in this generated report is
`{metric}`. The largest generated value for Architecture A is {_fmt(amax)};
the largest for Architecture B is {_fmt(bmax)}. The corrected two-blaze model
also enforces an architecture-label symmetry gate in the flat-correction
effective-channel limit. {symmetry_text}

## 1. Study definition

Architecture A:

- SLM1: vortex + carrier/blaze
- SLM2: correction + carrier/blaze

Architecture B:

- SLM1: correction + carrier/blaze
- SLM2: vortex + carrier/blaze

The carrier/blaze term is present on both SLM1 and SLM2 in both cases. The correction status
for this run is: **{correction_status}**.

The registration variables are independent x/y offsets of the SLM1 and SLM2
pixel lattices. For each selected panel, the physical panel is translated by
-d while the displayed pattern is shifted by +d. This changes beam-to-lattice
registration without changing beam-to-hologram centring.

## 2. Numerical method

The study used a fine SLM grid of {spp.get('fine_grid_n', 'unknown')} samples per
side and a relay grid of {spp.get('relay_grid_n', 'unknown')} samples per side
over a {spp.get('window_mm', 'unknown')} mm window. Pixel values used the
{spp.get('pixel_value_model', 'unknown')} convention and the fill-factor model
was {spp.get('fill_factor_model', 'unknown')}.

The downstream optical route is held fixed between architectures. Both SLM
commands contain their carrier/blaze terms; in the scalar effective-channel
model the common 4F selected-order centre is computed from the summed two-panel
carrier, after which the summed deterministic carrier is removed in the image
frame before the physical axicon. No unmeasured
SLM1-to-SLM2 free-space separation is introduced.

The principal comparisons are:

- SLM1-only registration;
- SLM2-only registration;
- common-mode registration;
- differential registration;
- x, y and diagonal directions;
- charge and beam-radius sweeps;
- two-dimensional pixel-unit-cell maps where available;
- direct SLM1-offset x SLM2-offset maps where available;
- representative XY fields before the axicon and after propagation.

## 3. Quantitative results

### 3.1 Architecture-label symmetry sanity check

{symmetry_text}

### 3.2 Worst generated sensitivity

Architecture A maximum: **{_fmt(amax)}**

Architecture B maximum: **{_fmt(bmax)}**

{ratio_text}

### 3.3 Architecture ratio by charge and beam radius

{ratio_table if ratio_table else "Not available."}

### 3.4 Pixel-unit-cell sensitivity

{unit_text}

### 3.5 Relative SLM1-SLM2 registration

{inter_text}

### 3.6 Post-axicon axial sensitivity

{axial_text}

## 4. Qualitative error signatures

The representative XY evidence, when generated, is stored in
`representative_xy_fields.npz` and plotted by
`tools/plot_slm_registration_architecture_study.py`. The report figures compare
the actual transverse intensity and signed residual at the pre-axicon plane and
at a matched propagated plane after the physical axicon.

The intended qualitative test is not whether one beam image looks generally
'worse'. It is whether the residual pattern changes systematically with:

- which panel is registered;
- the x/y direction of registration;
- topological charge;
- beam radius;
- phase-allocation architecture.

## 5. Discussion

The corrected two-blaze effective-channel model is intentionally symmetric
under exchanging SLM1 and SLM2 when correction is flat and no measured
inter-SLM transfer is supplied. Consequently, an apparent A/B winner in that
limit would be a model bug rather than a physical conclusion. The valid result
from this layer is the sensitivity of a vortex+blaze mask to its own pixel
registration. Architecture discrimination must be deferred to a model that
contains measured panel-specific transfer, parity/rotation, calibration and
correction-map information.

The study also separates the physics-isolation run from correction-on runs. A
flat zero correction tests phase ownership only. Any claim about the practical
corrected system must use a supplied correction map with explicit provenance.

## 6. Claim boundary

This report does not invent an SLM1-to-SLM2 physical propagation distance.
The current accepted registration route is the effective-channel explicit-4F
route. The repository's component-owned CSLM separation remains a diagnostic
placeholder rather than measured laboratory geometry.

## 7. Figures

{fig_md if fig_md else "No plotted figures were found for this tag."}
"""

    path = reportdir / "report.md"
    path.write_text(report, encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
