"""q=20 real-data model v3: calibrate model-bound physical nuisance parameters
before fitting residual wavefront structure.

The v2 audit showed that the former absolute-z registration (48 mm at relative
z=0) drove the nominal route into a late-scan collapse that is absent from the
real BeamGage data.  Train-only detector-aware re-registration removed that
failure and raised held-out nominal agreement substantially.

This v3 step asks the next physically necessary question before attributing the
remaining mismatch to aberration: the repository manifest explicitly labels the
2 mm input Gaussian radius and Fourier iris radius as uncalibrated/assumed.  We
therefore fit only three low-dimensional model-bound nuisance quantities on
EVEN z planes:

    isotropic input Gaussian radius scale,
    4F Fourier-iris radius scale,
    absolute z registration.

The measured k_perp/effective axicon angle is held fixed.  Odd z planes remain
untouched until all nuisance parameters and residual coefficients are frozen.
The selected nuisance configuration is then passed to the v2 Miao-informed
smooth phase + low-order amplitude diagnostic residual fit.

No hardware correction mask is emitted.  Fitted beam/iris/z values are model
registration parameters until independently measured on the bench.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import fit_q20_detector_aware_model_v2 as v2  # noqa: E402
from vbb_study.digital_twin.vortex_beam_slm_errors import GaussianBeamError  # noqa: E402
from vbb_study.digital_twin.vortex_continuous_propagation import build_fixed_support_spectrum  # noqa: E402
from vbb_study.digital_twin.vortex_system_route import (  # noqa: E402
    FourFError,
    SystemErrorConfig,
    build_multirate_system_route,
)

EXP = ROOT / "notebooks" / "experimental" / "axicon_aberration_correction"


def config_with(base: SystemErrorConfig, *, beam_scale: float, iris_scale: float) -> SystemErrorConfig:
    return replace(
        base,
        beam=GaussianBeamError(radius_x_scale=float(beam_scale), radius_y_scale=float(beam_scale)),
        fourf=FourFError(iris_radius_scale=float(iris_scale)),
    )


def route_context(base_context: dict, config: SystemErrorConfig) -> dict:
    route = build_multirate_system_route(
        "V20",
        relay_grid_n=v2.RELAY_N,
        propagation_grid_n=v2.FIT_N,
        window_m=v2.FIT_WINDOW_M,
        config=config,
    )
    grid = route["grid"]
    wavelength = float(route["metadata"]["wavelength_m"])
    field = np.asarray(route["post_axicon"], np.complex128)
    prop = build_fixed_support_spectrum(
        field,
        grid,
        wavelength_m=wavelength,
        z_max_m=0.060,
        minimum_retained_spectral_power=0.98,
    )
    return {
        **base_context,
        "config": config,
        "route": route,
        "grid": grid,
        "wavelength": wavelength,
        "baseline_field": field,
        "baseline_prop": prop,
    }


def evaluate_context(context: dict, z0_mm: float, ids: np.ndarray) -> dict:
    z_abs = (float(z0_mm) + context["z_rel"][ids]) * 1e-3
    pred = v2.render_baseline(context, z_abs)
    metrics = v2.score(pred, context["data"][ids], context["axis_um"])
    return {**metrics, "objective": v2.robust_objective(metrics)}


def scan_one_parameter(
    base_context: dict,
    base_config: SystemErrorConfig,
    *,
    z0_mm: float,
    train: np.ndarray,
    parameter: str,
    values: np.ndarray,
    fixed_beam_scale: float,
    fixed_iris_scale: float,
) -> tuple[float, pd.DataFrame]:
    rows = []
    for value in np.asarray(values, float):
        beam = float(value) if parameter == "beam_radius_scale" else float(fixed_beam_scale)
        iris = float(value) if parameter == "iris_radius_scale" else float(fixed_iris_scale)
        config = config_with(base_config, beam_scale=beam, iris_scale=iris)
        context = route_context(base_context, config)
        m = evaluate_context(context, z0_mm, train)
        rows.append({
            "parameter": parameter,
            "value": float(value),
            "mean_pearson_r": m["mean_pearson_r"],
            "mean_nrmse": m["mean_nrmse"],
            "max_nrmse": m["max_nrmse"],
            "objective": m["objective"],
        })
    table = pd.DataFrame(rows)
    best = table.iloc[int(table.objective.argmin())]
    table["selected"] = np.isclose(table.value, float(best.value))
    return float(best.value), table


def scan_z_for_context(context: dict, train: np.ndarray, centre_mm: float) -> tuple[float, pd.DataFrame]:
    rows = []
    # 0.5 mm is below the 1 mm acquisition step while still keeping the scan
    # computationally tractable.  Search is deliberately local after v2's wide
    # 18--50 mm train-only registration.
    for z0 in np.arange(float(centre_mm) - 5.0, float(centre_mm) + 5.01, 0.5):
        m = evaluate_context(context, float(z0), train)
        rows.append({
            "z0_mm": float(z0),
            "mean_pearson_r": m["mean_pearson_r"],
            "mean_nrmse": m["mean_nrmse"],
            "max_nrmse": m["max_nrmse"],
            "objective": m["objective"],
        })
    table = pd.DataFrame(rows)
    best = table.iloc[int(table.objective.argmin())]
    table["selected"] = np.isclose(table.z0_mm, float(best.z0_mm))
    return float(best.z0_mm), table


def plot_parameter_scan(table: pd.DataFrame, xlabel: str, title: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 4.6), constrained_layout=True)
    ax.plot(table.value, table.objective, "o-", lw=1.8)
    selected = table.loc[table.selected].iloc[0]
    ax.axvline(float(selected.value), ls="--", lw=1.4, label=f"selected = {float(selected.value):.2f}")
    ax.set(xlabel=xlabel, ylabel="train-only robust objective", title=title)
    ax.grid(alpha=.22); ax.legend(frameon=False)
    v2.savefig(fig, out)


def plot_z_scan(table: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 4.6), constrained_layout=True)
    ax.plot(table.z0_mm, table.objective, "o-", lw=1.8)
    selected = table.loc[table.selected].iloc[0]
    ax.axvline(float(selected.z0_mm), ls="--", lw=1.4, label=f"selected = {float(selected.z0_mm):.1f} mm")
    ax.set(xlabel="absolute z at relative z=0 (mm)", ylabel="train-only robust objective",
           title="Absolute-z refinement after beam/iris calibration")
    ax.grid(alpha=.22); ax.legend(frameon=False)
    v2.savefig(fig, out)


def run(source_dir: Path, out: Path) -> dict:
    source_dir = Path(source_dir); out = Path(out); out.mkdir(parents=True, exist_ok=True)
    base = v2.build_context(source_dir)
    n = len(base["z_rel"])
    train = np.arange(0, n, 2, dtype=int)
    held = np.arange(1, n, 2, dtype=int)

    # Re-use v2's wide z audit as a reproducible starting point.  It is still
    # train-only; held-out planes never enter this stage.
    initial_z0, initial_zscan = v2.scan_absolute_z(base, train, out)
    initial_zscan.to_csv(out / "00_v2_wide_z_registration.csv", index=False)

    base_config = base["config"]
    beam_values = np.asarray([0.75, 0.85, 0.95, 1.05, 1.15, 1.25], float)
    beam_scale, beam_table = scan_one_parameter(
        base, base_config,
        z0_mm=initial_z0, train=train,
        parameter="beam_radius_scale", values=beam_values,
        fixed_beam_scale=1.0, fixed_iris_scale=1.0,
    )
    beam_table.to_csv(out / "06_beam_radius_scale_scan.csv", index=False)
    plot_parameter_scan(
        beam_table, "Gaussian 1/e field-radius scale", "Train-only input-beam radius screening",
        out / "06_beam_radius_scale_scan",
    )

    iris_values = np.asarray([0.85, 0.95, 1.05, 1.15, 1.25], float)
    iris_scale, iris_table = scan_one_parameter(
        base, base_config,
        z0_mm=initial_z0, train=train,
        parameter="iris_radius_scale", values=iris_values,
        fixed_beam_scale=beam_scale, fixed_iris_scale=1.0,
    )
    iris_table.to_csv(out / "07_iris_radius_scale_scan.csv", index=False)
    plot_parameter_scan(
        iris_table, "Fourier-iris radius scale", "Train-only 4F iris screening",
        out / "07_iris_radius_scale_scan",
    )

    selected_config = config_with(base_config, beam_scale=beam_scale, iris_scale=iris_scale)
    selected_context = route_context(base, selected_config)
    z0_mm, ztable = scan_z_for_context(selected_context, train, initial_z0)
    ztable.to_csv(out / "08_absolute_z_refinement.csv", index=False)
    plot_z_scan(ztable, out / "08_absolute_z_refinement")

    z_abs = (z0_mm + base["z_rel"]) * 1e-3
    baseline_all = v2.render_baseline(selected_context, z_abs)
    baseline_train = v2.score(baseline_all[train], base["data"][train], base["axis_um"])
    baseline_held = v2.score(baseline_all[held], base["data"][held], base["axis_um"])

    # Fit the already-audited Miao-informed compact residual only after the
    # physical nuisance parameters have been frozen.
    fit = v2.fit_complex_residual(selected_context, z_abs, train, held, out)
    residual = fit["result"]

    summary = {
        "study": "q20 detector-aware model v3: train-only beam/iris/z nuisance calibration + compact complex residual",
        "data_split": {
            "train_indices": train.tolist(),
            "heldout_indices": held.tolist(),
            "heldout_used_for_parameter_selection": False,
        },
        "physical_nuisance": {
            "initial_z0_mm_from_v2_wide_train_scan": float(initial_z0),
            "selected_beam_radius_scale": float(beam_scale),
            "canonical_beam_radius_m": 2.0e-3,
            "selected_model_beam_radius_m": float(2.0e-3 * beam_scale),
            "selected_iris_radius_scale": float(iris_scale),
            "selected_z0_mm": float(z0_mm),
            "interpretation": (
                "model-bound nuisance calibration only: canonical beam radius and Fourier iris are marked assumed/calibration_required in the hardware manifest; values must be independently measured before hardware claims"
            ),
        },
        "baseline_after_nuisance_train": baseline_train,
        "baseline_after_nuisance_heldout": baseline_held,
        "residual_model": residual,
        "miao_connection": {
            "paper": "B. Miao et al., Opt. Express 30, 11360-11371 (2022), doi:10.1364/OE.454796",
            "use_here": "Miao motivates compact physically interpretable low-order pupil aberration coordinates; this workflow keeps explicit bench propagation and held-out z validation",
        },
        "hardware_ready": False,
        "hardware_boundary": (
            "No post-correction BeamGage evidence is generated. The complex residual is diagnostic. Amplitude nuisance is not a phase-only SLM correction, and the phase component must still be mapped upstream using calibrated SLM2-to-axicon coordinates and the 1030 nm LUT."
        ),
    }
    (out / "model_v3_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source-dir", type=Path, default=EXP / "outputs" / "digital_twin_correction")
    p.add_argument("--out", type=Path, default=ROOT / "outputs" / "validation" / "q20_detector_aware_model_v3")
    args = p.parse_args()
    run(args.source_dir, args.out)


if __name__ == "__main__":
    main()
