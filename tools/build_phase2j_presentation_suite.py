"""Regenerate the current presentation figure set under one Phase 2J visual standard.

Physics is inherited from the existing validated Phase 2I/2J forward-model
renderers.  This module deliberately changes presentation rendering only:

* inferno intensity maps (black -> red -> orange -> yellow),
* high-resolution export,
* tighter transverse / longitudinal framing,
* denser z sampling for slide-facing longitudinal maps,
* smooth display interpolation only after the numerical field has been computed.

The underlying complex fields, physical error injection planes, fixed-laboratory
coordinate convention, common nominal normalisation, tip sampling gate, and
synthetic/experimental claim boundaries are preserved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import build_phase2i_presentation_figures as core
import build_phase2j_ideal_beam_profile_figure as ideal
import build_presentation_extended_evidence as ext
import build_presentation_extended_evidence_v2 as extv2
import presentation_phase2j_style as style


Z_VALUES_M = np.linspace(5.0e-3, 120.0e-3, 64)
IDEAL_COORD_M = np.linspace(-0.22e-3, 0.22e-3, 441)
DECENTRE_COORD_M = np.linspace(-0.70e-3, 0.70e-3, 561)
TIP_COORD_M = np.linspace(-0.26e-3, 0.26e-3, 521)
ERROR_COORD_M = np.linspace(-0.70e-3, 0.70e-3, 561)


def _patch_core_renderer() -> None:
    """Apply Phase 2J rendering policy to the existing core presentation logic."""
    core.FIG_BG = style.FIG_BG
    core.AX_BG = style.AX_BG
    core.TEXT = style.TEXT
    core.MUTED = style.MUTED
    core.RED = style.RED
    core.GREEN = style.GREEN
    core.BORDER = style.BORDER
    core.CMAP = style.CMAP
    core.Z_VALUES_M = Z_VALUES_M
    core.IDEAL_COORD_M = IDEAL_COORD_M
    core.DECENTRE_COORD_M = DECENTRE_COORD_M
    core.TIP_COORD_M = TIP_COORD_M
    core._normalise = style.normalise
    core._style_figure = style.style_fig
    core._style_axis = style.style_ax
    core._save = style.save
    core._draw_xy = style.draw_xy

    def draw_xz(ax, values, coordinate_m, *, peak=None, show_y=False):
        return style.draw_xz(
            ax,
            values,
            np.asarray(coordinate_m, dtype=float),
            np.asarray(core.Z_VALUES_M, dtype=float),
            peak=peak,
            show_y=show_y,
            z_ref_m=float(core.Z_REF_M),
        )

    def crop(intensity: np.ndarray, grid: Mapping[str, Any], halfwidth_m: float):
        halfwidth = style.presentation_crop_halfwidth(float(halfwidth_m))
        x = np.asarray(grid["x"], dtype=float)
        ids = np.flatnonzero(np.abs(x) <= halfwidth)
        if ids.size < 40:
            raise RuntimeError(
                f"Phase 2J crop has only {ids.size} native samples; "
                "increase grid_n instead of inventing display resolution."
            )
        crop_values = np.asarray(intensity)[np.ix_(ids, ids)]
        extent = [
            x[ids[0]] * 1e3,
            x[ids[-1]] * 1e3,
            x[ids[0]] * 1e3,
            x[ids[-1]] * 1e3,
        ]
        return crop_values, extent

    core._draw_xz = draw_xz
    core._fixed_lab_crop = crop


def _patch_extended_renderer() -> None:
    """Apply the same visual policy to error, tip-avoidance and inverse evidence."""
    ext.FIG_BG = style.FIG_BG
    ext.AX_BG = style.AX_BG
    ext.TEXT = style.TEXT
    ext.MUTED = style.MUTED
    ext.RED = style.RED
    ext.GREEN = style.GREEN
    ext.BORDER = style.BORDER
    ext.CMAP = style.CMAP
    ext.Z_VALUES_M = Z_VALUES_M
    ext.IDEAL_COORD_M = IDEAL_COORD_M
    ext.ERROR_COORD_M = ERROR_COORD_M
    ext._style_fig = style.style_fig
    ext._style_ax = style.style_ax
    ext._save = style.save
    ext._norm = style.normalise
    ext._draw_xy = style.draw_xy
    ext._draw_xz = style.draw_xz

    def crop(values: np.ndarray, grid: Mapping[str, Any], halfwidth_m: float):
        halfwidth = style.presentation_crop_halfwidth(float(halfwidth_m))
        x = np.asarray(grid["x"], dtype=float)
        ids = np.flatnonzero(np.abs(x) <= halfwidth)
        if ids.size < 40:
            raise RuntimeError(
                f"Phase 2J crop has only {ids.size} native samples; "
                "increase grid_n instead of inventing display resolution."
            )
        return np.asarray(values)[np.ix_(ids, ids)], [
            x[ids[0]] * 1e3,
            x[ids[-1]] * 1e3,
            x[ids[0]] * 1e3,
            x[ids[-1]] * 1e3,
        ]

    ext._crop = crop


def _rename(path: Path, new_name: str) -> Path:
    new_path = path.with_name(new_name)
    if new_path.exists():
        new_path.unlink()
    path.replace(new_path)
    return new_path


def build_suite(output_dir: Path, grid_n: int, inverse_grid_n: int) -> dict[str, Any]:
    if grid_n < 1536:
        raise ValueError("Phase 2J full presentation suite requires grid_n >= 1536")
    if inverse_grid_n < 512:
        raise ValueError("Phase 2J inverse proof requires inverse_grid_n >= 512")

    output_dir.mkdir(parents=True, exist_ok=True)
    _patch_core_renderer()
    _patch_extended_renderer()

    # 1 — actual numerical route. Intensity panels inherit inferno; phase panels
    # remain cyclic because phase is not an intensity quantity.
    p1 = core.build_computational_route(output_dir, grid_n)
    p1 = _rename(p1, "01_computational_route_phase2j.png")

    # 2 — dedicated high-resolution ideal family renderer already established in
    # the first Phase 2J pass.
    p2 = ideal.build_figure(output_dir, grid_n)

    # 4/5 — concrete axicon examples keep the original physical perturbations and
    # common nominal normalisation; only their presentation crop/render changes.
    p4, m4 = core.build_v1_decentre(output_dir, grid_n)
    p4 = _rename(p4, "04_V1_axicon_decentre_fixed_lab_inferno_tight.png")

    p5, m5 = core.build_v1_tip(output_dir, grid_n)
    p5 = _rename(p5, "05_V1_nonideal_tip_fixed_lab_inferno_tight.png")

    # 6 — loop schematic uses the same Phase 2J typography/background treatment.
    p6 = core.build_closure(output_dir)
    p6 = _rename(p6, "06_simulation_experiment_closure_phase2j.png")

    # 8/9/10 — diagnostic fingerprints, tip-avoidance planning proxy and
    # synthetic inverse proof retain their existing scientific claim boundaries.
    p8, m8 = ext.build_real_error_fingerprints(output_dir, grid_n)
    p8 = _rename(p8, "08_V1_real_error_fingerprints_inferno_tight.png")

    p9, m9 = ext.build_tip_avoidance_proxy(output_dir, grid_n)
    p9 = _rename(p9, "09_tip_avoidance_planning_proxy_inferno.png")

    p10, m10 = extv2.build_synthetic_inverse_recovery(output_dir, inverse_grid_n)
    p10 = _rename(p10, "10_synthetic_zstack_inverse_recovery_inferno.png")

    figures = [p1, p2, p4, p5, p6, p8, p9, p10]
    manifest: dict[str, Any] = {
        "outcome": "PHASE2J-PRESENTATION-VISUAL-STANDARDISATION",
        "physics_source": "current repository dual-SLM -> explicit 4F -> physical axicon model",
        "grid_n": int(grid_n),
        "inverse_grid_n": int(inverse_grid_n),
        "visual_policy": {
            "intensity_colormap": style.CMAP,
            "palette": "black -> red -> orange -> yellow",
            "save_dpi": style.SAVE_DPI,
            "display_interpolation": style.DISPLAY_INTERPOLATION,
            "display_interpolation_changes_numerical_field": False,
            "z_samples_for_main_longitudinal_maps": int(len(Z_VALUES_M)),
            "fixed_lab_longitudinal": True,
            "per_z_recentering": False,
            "ideal_transverse_halfwidth_m": 0.20e-3,
            "tip_transverse_halfwidth_m": 0.26e-3,
            "decentre_and_error_transverse_halfwidth_m": 0.65e-3,
        },
        "figures": [str(path) for path in figures],
        "axicon_decentre_metrics": m4,
        "axicon_tip_metrics": m5,
        "real_error_fingerprints": m8,
        "tip_avoidance_planning_proxy": m9,
        "synthetic_inverse_recovery": m10,
        "claim_boundaries": [
            "presentation rendering changes do not alter optical physics",
            "axicon perturbation magnitudes remain sensitivity examples unless independently calibrated",
            "tip-avoidance image remains a planning proxy rather than calibrated SLM efficiency evidence",
            "inverse-recovery image remains synthetic model-to-model validation until calibrated camera data are fitted",
        ],
    }
    manifest_path = output_dir / "presentation_phase2j_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-n", type=int, default=1536)
    parser.add_argument("--inverse-grid-n", type=int, default=512)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/figures/presentation_phase2j"),
    )
    args = parser.parse_args()
    build_suite(args.output_dir, args.grid_n, args.inverse_grid_n)


if __name__ == "__main__":
    main()
