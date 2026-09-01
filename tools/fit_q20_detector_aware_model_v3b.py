"""q=20 detector-aware model v3b: extend the v3 beam-radius search.

V3 selected the largest tested Gaussian radius scale (1.25), so that nuisance
parameter was not demonstrably bracketed.  This follow-up preserves the same
train/held-out protocol but extends the beam-radius range before accepting the
calibration.  It is a separate evidence path and does not overwrite v3.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import fit_q20_detector_aware_model_v2 as v2  # noqa: E402
import fit_q20_detector_aware_model_v3 as v3  # noqa: E402

EXP = ROOT / "notebooks" / "experimental" / "axicon_aberration_correction"


def run(source_dir: Path, out: Path) -> dict:
    source_dir = Path(source_dir); out = Path(out); out.mkdir(parents=True, exist_ok=True)
    base = v2.build_context(source_dir)
    n = len(base["z_rel"])
    train = np.arange(0, n, 2, dtype=int)
    held = np.arange(1, n, 2, dtype=int)
    initial_z0, zwide = v2.scan_absolute_z(base, train, out)
    zwide.to_csv(out / "00_wide_z_registration.csv", index=False)

    base_config = base["config"]
    # V3's objective decreased monotonically through 1.25.  Extend far enough
    # to bracket the minimum while keeping the fitted radius physically modest
    # relative to the ~4 mm-diameter nominal beam assumption.
    beam_values = np.asarray([1.10, 1.20, 1.30, 1.40, 1.50, 1.60], float)
    beam_scale, beam_table = v3.scan_one_parameter(
        base, base_config,
        z0_mm=initial_z0, train=train,
        parameter="beam_radius_scale", values=beam_values,
        fixed_beam_scale=1.0, fixed_iris_scale=1.0,
    )
    beam_table.to_csv(out / "06b_beam_radius_scale_extended.csv", index=False)
    v3.plot_parameter_scan(
        beam_table, "Gaussian 1/e field-radius scale",
        "Extended train-only input-beam radius screening",
        out / "06b_beam_radius_scale_extended",
    )

    iris_values = np.asarray([0.90, 0.975, 1.05, 1.125, 1.20], float)
    iris_scale, iris_table = v3.scan_one_parameter(
        base, base_config,
        z0_mm=initial_z0, train=train,
        parameter="iris_radius_scale", values=iris_values,
        fixed_beam_scale=beam_scale, fixed_iris_scale=1.0,
    )
    iris_table.to_csv(out / "07b_iris_radius_scale_refined.csv", index=False)
    v3.plot_parameter_scan(
        iris_table, "Fourier-iris radius scale",
        "Refined train-only 4F iris screening",
        out / "07b_iris_radius_scale_refined",
    )

    cfg = v3.config_with(base_config, beam_scale=beam_scale, iris_scale=iris_scale)
    context = v3.route_context(base, cfg)
    z0_mm, ztable = v3.scan_z_for_context(context, train, initial_z0)
    ztable.to_csv(out / "08b_absolute_z_refinement.csv", index=False)
    v3.plot_z_scan(ztable, out / "08b_absolute_z_refinement")

    z_abs = (z0_mm + base["z_rel"]) * 1e-3
    nominal = v2.render_baseline(context, z_abs)
    baseline_train = v2.score(nominal[train], base["data"][train], base["axis_um"])
    baseline_held = v2.score(nominal[held], base["data"][held], base["axis_um"])
    fit = v2.fit_complex_residual(context, z_abs, train, held, out)

    selected_is_boundary = bool(
        np.isclose(beam_scale, float(beam_values.min()))
        or np.isclose(beam_scale, float(beam_values.max()))
    )
    summary = {
        "study": "q20 detector-aware model v3b: extended beam-radius bracket + iris/z calibration + compact complex residual",
        "data_split": {
            "train_indices": train.tolist(),
            "heldout_indices": held.tolist(),
            "heldout_used_for_parameter_selection": False,
        },
        "physical_nuisance": {
            "selected_beam_radius_scale": float(beam_scale),
            "selected_model_beam_radius_m": float(2.0e-3 * beam_scale),
            "beam_search_values": beam_values.tolist(),
            "beam_optimum_on_tested_boundary": selected_is_boundary,
            "selected_iris_radius_scale": float(iris_scale),
            "selected_z0_mm": float(z0_mm),
            "interpretation": "model-bound nuisance calibration; beam/iris/z values are not bench measurements",
        },
        "baseline_after_nuisance_train": baseline_train,
        "baseline_after_nuisance_heldout": baseline_held,
        "residual_model": fit["result"],
        "acceptance_note": (
            "Use as poster evidence only if the beam-radius optimum is bracketed and held-out performance equals or exceeds v3 without increasing residual complexity."
        ),
        "hardware_ready": False,
    }
    (out / "model_v3b_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source-dir", type=Path, default=EXP / "outputs" / "digital_twin_correction")
    p.add_argument("--out", type=Path, default=ROOT / "outputs" / "validation" / "q20_detector_aware_model_v3b")
    a = p.parse_args(); run(a.source_dir, a.out)


if __name__ == "__main__":
    main()
