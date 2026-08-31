"""Render a high-resolution QC dashboard for a local q=20 Miao retrieval."""
from __future__ import annotations

from pathlib import Path
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def render(output_dir: Path, *, pixel_pitch_um: float = 5.5) -> dict:
    output_dir = Path(output_dir)
    fit = pd.read_csv(output_dir / "per_plane_retrieval.csv").sort_values("z_relative_mm")
    independent_path = output_dir / "per_plane_independent_retrieval.csv"
    independent = (pd.read_csv(independent_path).sort_values("z_relative_mm")
                   if independent_path.exists() else None)
    qc = pd.read_csv(output_dir / "frame_qc_preserved_coordinates.csv")
    qc["repeat_shift_px"] = np.hypot(qc.repeat_registration_shift_x_px,
                                      qc.repeat_registration_shift_y_px)
    plane_qc = qc.groupby("z_index", as_index=False).agg(
        core_y_raw_px=("core_y_raw_px", "mean"),
        core_x_raw_px=("core_x_raw_px", "mean"),
        core_score=("core_score", "mean"),
        max_repeat_shift_px=("repeat_shift_px", "max"),
    )
    plane_qc = plane_qc.sort_values("z_index")
    z = fit.z_relative_mm.to_numpy(float)
    kp = fit.k_perp_opt_m_inv.to_numpy(float)
    order = fit.aberration_order_max.to_numpy(int)
    x_um = (plane_qc.core_x_raw_px.to_numpy(float) - plane_qc.core_x_raw_px.iloc[0]) * pixel_pitch_um
    y_um = (plane_qc.core_y_raw_px.to_numpy(float) - plane_qc.core_y_raw_px.iloc[0]) * pixel_pitch_um
    repeat_shift = plane_qc.max_repeat_shift_px.to_numpy(float)

    fig, axes = plt.subplots(2, 2, figsize=(17, 10), constrained_layout=True)
    ax = axes[0, 0]
    if independent is not None:
        ax.plot(z, independent.k_perp_opt_m_inv.to_numpy(float) / 1e3,
                "x--", lw=1.3, color="#D55E00", label="independent old solution")
    ax.plot(z, kp / 1e3, "o-", lw=1.9, color="#0072B2",
            label="global continuous branch")
    ax.axhline(np.median(kp) / 1e3, color="0.35", ls="--",
               label=f"median {np.median(kp)/1e3:.1f} krad/m")
    ax.set(title="Stack-aware transverse-wavenumber retrieval",
           xlabel="relative z (mm)", ylabel=r"$k_\perp$ (krad/m)")
    ax.grid(alpha=.25); ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(z, fit.fit_corr, "o-", label="global correlation")
    ax.plot(z, fit.fit_nrmse, "s-", label="global normalized RMSE")
    if independent is not None:
        ax.plot(z, independent.fit_corr, "o--", alpha=.65,
                label="independent correlation")
        ax.plot(z, independent.fit_nrmse, "s--", alpha=.65,
                label="independent normalized RMSE")
    ax.set(title="Per-plane modal-fit quality", xlabel="relative z (mm)",
           ylabel="metric", ylim=(0, 1))
    ax.grid(alpha=.25); ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(z, fit.fit_cost, "o-", color="#0072B2", label="fit cost")
    ax.set(title="Adaptive fit stopping behaviour", xlabel="relative z (mm)",
           ylabel="fit cost", ylim=(0, max(.3, 1.08 * float(fit.fit_cost.max()))))
    ax.grid(alpha=.25)
    twin = ax.twinx()
    twin.step(z, order, where="mid", color="#D55E00", lw=1.8,
              label="maximum fitted |m|")
    twin.axhline(30, color="#D55E00", ls=":", alpha=.55)
    twin.set(ylabel="adaptive aberration order", ylim=(0, 32))
    lines = ax.lines + twin.lines[:1]
    ax.legend(lines, [line.get_label() for line in lines], fontsize=8)

    ax = axes[1, 1]
    ax.plot(z, x_um, "o-", label="dark-core x")
    ax.plot(z, y_um, "o-", label="dark-core y")
    ax.fill_between(z, -repeat_shift * pixel_pitch_um, repeat_shift * pixel_pitch_um,
                    color="0.6", alpha=.2, label="max within-plane repeat shift")
    ax.set(title="Raw-sensor beam walk retained by retrieval",
           xlabel="relative z (mm)", ylabel="displacement from first plane (um)")
    ax.grid(alpha=.25); ax.legend(fontsize=8)

    fig.suptitle("q=20 stack-aware Miao retrieval — diagnostic only; hardware correction blocked",
                 fontsize=15)
    png = output_dir / "miao_local_retrieval_diagnostics.png"
    pdf = output_dir / "miao_local_retrieval_diagnostics.pdf"
    fig.savefig(png, dpi=400, bbox_inches="tight")
    fig.savefig(pdf, dpi=400, bbox_inches="tight")
    plt.close(fig)

    result = {
        "planes": int(len(fit)),
        "k_perp_median_m_inv": float(np.median(kp)),
        "k_perp_min_m_inv": float(np.min(kp)),
        "k_perp_max_m_inv": float(np.max(kp)),
        "max_adjacent_k_perp_jump_fraction": float(np.max(np.abs(np.diff(kp)/kp[:-1]))),
        "mean_fit_corr": float(fit.fit_corr.mean()),
        "min_fit_corr": float(fit.fit_corr.min()),
        "mean_fit_nrmse": float(fit.fit_nrmse.mean()),
        "planes_at_max_order_30": int(np.count_nonzero(order == 30)),
        "raw_core_x_range_um": float(np.ptp(plane_qc.core_x_raw_px) * pixel_pitch_um),
        "raw_core_y_range_um": float(np.ptp(plane_qc.core_y_raw_px) * pixel_pitch_um),
        "max_repeat_registration_shift_px": float(np.max(repeat_shift)),
        "interpretation": "Global k_perp branch with fixed-k modal fits; no calibrated full-aperture or SLM2 phase was produced.",
    }
    if independent is not None:
        ikp = independent.k_perp_opt_m_inv.to_numpy(float)
        result.update({
            "independent_max_adjacent_k_perp_jump_fraction": float(
                np.max(np.abs(np.diff(ikp)/ikp[:-1]))),
            "independent_mean_fit_corr": float(independent.fit_corr.mean()),
            "independent_mean_fit_nrmse": float(independent.fit_nrmse.mean()),
            "independent_planes_at_max_order_30": int(np.count_nonzero(
                independent.aberration_order_max.to_numpy(int) == 30)),
        })
    (output_dir / "miao_local_retrieval_diagnostic_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--pixel-pitch-um", type=float, default=5.5)
    args = parser.parse_args()
    print(json.dumps(render(args.output_dir, pixel_pitch_um=args.pixel_pitch_um), indent=2))
