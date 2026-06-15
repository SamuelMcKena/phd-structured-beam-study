"""Generate the upgraded polarized-train mechanism checkpoint outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from vbb_study import setup_study, vbb_polarized_train, vbb_style

import bessel_twin_core as bt
from Publication_Study.finalize_publication_outputs import finalize_outputs


def _out_tree(paths: dict[str, Path]) -> dict[str, Path]:
    base = paths["outputs"]
    out = {
        "figures": base / "figures" / "polarized_train",
        "csv": base / "csv" / "polarized_train",
        "json": base / "json" / "polarized_train",
        "holograms": base / "holograms" / "polarized_train",
    }
    for path in out.values():
        path.mkdir(parents=True, exist_ok=True)
    return out


def run_checkpoint() -> dict[str, Any]:
    paths = setup_study.bootstrap(Path(__file__))
    out = _out_tree(paths)
    config = vbb_polarized_train.PolarizedTrainConfig(
        N=320,
        dx_m=0.18 * bt.um,
        wavelength_m=1029.0 * bt.nm,
        n_medium=1.0,
        n_axicon=1.46,
        axicon_base_angle_deg=32.0,
        ring_radius_m=8.0 * bt.um,
        ring_width_m=0.95 * bt.um,
        vortex_charge=1,
        vector_element="segmented_ra",
        segment_count=12,
        symmetry_order=6,
        z_max_m=110.0 * bt.um,
        z_points=45,
    )

    gate = vbb_polarized_train.run_hexagon_mechanism_gate(config)
    comparison = vbb_polarized_train.ideal_lab_comparison(config)
    fair_suite = vbb_polarized_train.run_fair_comparison_suite(config)

    gate_csv = out["csv"] / vbb_style.csv_name(12, "polarized_train", "mechanism_gate")
    gate["summary"].to_csv(gate_csv, index=False)
    comparison_csv = out["csv"] / vbb_style.csv_name(12, "polarized_train", "ideal_lab_metrics")
    comparison["summary"].to_csv(comparison_csv, index=False)
    fair_delta_csv = out["csv"] / vbb_style.csv_name(12, "polarized_train", "fair_ideal_lab_delta")
    fair_suite["delta_table"].to_csv(fair_delta_csv, index=False)

    mechanism_fig = vbb_polarized_train.plot_mechanism_gate(
        gate,
        out["figures"] / vbb_style.figure_name(12, "polarized_train", "mechanism_gate"),
    )
    ideal_lab_fig = vbb_polarized_train.plot_ideal_lab_comparison(
        comparison,
        out["figures"] / vbb_style.figure_name(12, "polarized_train", "ideal_lab_xy_xz"),
    )
    element_maps = vbb_polarized_train.export_element_maps(
        config,
        out["holograms"],
        label="segmented_ra_physical_axicon_train",
    )
    mechanism_doc = vbb_polarized_train.write_mechanism_doc(
        gate,
        paths["docs"] / "HEXAGON_MECHANISM.md",
    )
    fair_doc = vbb_polarized_train.write_fair_comparison_audit(
        fair_suite,
        paths["docs"] / "FAIR_COMPARISON_AUDIT.md",
    )

    acceptance = pd.DataFrame(
        [
            {
                "check": "matched_index_hexagon_vanishes_at_axicon_exit",
                "pass": bool(gate["vectorial"]),
                "value": gate["vanish_ratio"],
                "tolerance": "< 0.35",
            },
            {
                "check": "full_vector_path_enforces_k_dot_e",
                "pass": bool(comparison["ideal"]["metrics"]["k_dot_e_rms"] < 1.0e-10),
                "value": comparison["ideal"]["metrics"]["k_dot_e_rms"],
                "tolerance": "< 1e-10",
            },
            {
                "check": "ez_retained",
                "pass": bool(comparison["ideal"]["metrics"]["ez_power_fraction"] > 1.0e-3),
                "value": comparison["ideal"]["metrics"]["ez_power_fraction"],
                "tolerance": "> 1e-3",
            },
            {
                "check": "lab_realistic_same_train",
                "pass": comparison["ideal"]["config"] == comparison["lab_realistic"]["config"],
                "value": 1.0,
                "tolerance": "same config object values",
            },
        ]
    )
    acceptance_csv = out["csv"] / vbb_style.csv_name(12, "polarized_train", "acceptance_summary")
    acceptance.to_csv(acceptance_csv, index=False)

    manifest = setup_study.write_run_manifest(
        out["json"] / "12_polarized_train_run_manifest.json",
        config=config,
        paths={
            "gate_csv": gate_csv,
            "comparison_csv": comparison_csv,
            "fair_delta_csv": fair_delta_csv,
            "acceptance_csv": acceptance_csv,
            "mechanism_fig": mechanism_fig,
            "ideal_lab_fig": ideal_lab_fig,
            "element_maps": element_maps,
            "mechanism_doc": mechanism_doc,
            "fair_doc": fair_doc,
        },
        extra={
            "verdict": gate["verdict"],
            "default_path": gate["default_path"],
            "case1_guardrail": "co-aligned phase-only SLMs alone remain non-vector for radial/azimuthal targets",
        },
        root=paths["root"],
    )
    finalize_outputs(paths["outputs"])
    return {
        "config": config,
        "gate": gate,
        "comparison": comparison,
        "fair_suite": fair_suite,
        "acceptance": acceptance,
        "gate_csv": gate_csv,
        "comparison_csv": comparison_csv,
        "fair_delta_csv": fair_delta_csv,
        "acceptance_csv": acceptance_csv,
        "mechanism_fig": mechanism_fig,
        "ideal_lab_fig": ideal_lab_fig,
        "element_maps": element_maps,
        "mechanism_doc": mechanism_doc,
        "fair_doc": fair_doc,
        "manifest": manifest,
    }


if __name__ == "__main__":
    bundle = run_checkpoint()
    print(f"Verdict: {bundle['gate']['verdict']}")
    print(f"Default path: {bundle['gate']['default_path']}")
    print(f"Mechanism gate: {bundle['gate_csv']}")
    print(f"Ideal/lab metrics: {bundle['comparison_csv']}")
    print(f"Fair delta table: {bundle['fair_delta_csv']}")
    print(f"Acceptance: {bundle['acceptance_csv']}")
    print(f"Mechanism figure: {bundle['mechanism_fig']}")
    print(f"Ideal/lab figure: {bundle['ideal_lab_fig']}")
    print(f"Mechanism doc: {bundle['mechanism_doc']}")
    print(f"Fair audit doc: {bundle['fair_doc']}")
