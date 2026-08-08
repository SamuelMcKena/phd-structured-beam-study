"""Report-oriented figure generation for the repaired-route vortex visual atlas."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from vbb_study.digital_twin.vortex_visual_atlas import (
    DEFAULT_FIGURE_ROOT,
    DEFAULT_REPORT_N,
    ZERNIKE_REGISTRY,
    alignment_registry,
    aberration_registry,
    build_atlas_source,
    parameter_registry,
    propagate_selected_planes,
)


EPS = np.finfo(float).tiny


def _crop(intensity: np.ndarray, grid: Mapping[str, Any], halfwidth_m: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(grid["x"], dtype=float)
    mask = np.abs(x) <= float(halfwidth_m)
    return x[mask], np.asarray(intensity)[np.ix_(mask, mask)]


def _normalise(intensity: np.ndarray, reference: float | None = None) -> tuple[np.ndarray, float]:
    arr = np.maximum(np.asarray(intensity, dtype=float), 0.0)
    scale = max(float(np.max(arr)) if reference is None else float(reference), EPS)
    return arr / scale, scale


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")


def canonical_plane_grid(
    *,
    figure_root: Path = DEFAULT_FIGURE_ROOT,
    grid_n: int = DEFAULT_REPORT_N,
    cases: Sequence[str] = ("B0", "V1", "V3"),
    z_values_m: Sequence[float] = (20e-3, 40e-3, 60e-3, 80e-3, 100e-3),
    halfwidth_m: float = 0.35e-3,
) -> list[str]:
    import matplotlib.pyplot as plt

    outputs: list[str] = []
    for case_id in cases:
        source, grid, meta = build_atlas_source(case_id, grid_n=grid_n)
        planes = propagate_selected_planes(source, grid, float(meta["wavelength_m"]), z_values_m)
        common_scale = max(float(np.max(p)) for p in planes)
        fig, axes = plt.subplots(1, len(planes), figsize=(3.0 * len(planes), 3.0), constrained_layout=True)
        for ax, z_m, plane in zip(np.atleast_1d(axes), z_values_m, planes):
            x, cropped = _crop(plane, grid, halfwidth_m)
            shown, _ = _normalise(cropped, common_scale)
            im = ax.imshow(shown, origin="lower", extent=[x[0]*1e6, x[-1]*1e6, x[0]*1e6, x[-1]*1e6], vmin=0, vmax=1, cmap="inferno")
            ax.set_title(f"z = {z_m*1e3:.0f} mm")
            ax.set_xlabel("x (µm)")
            ax.set_ylabel("y (µm)")
        fig.colorbar(im, ax=np.atleast_1d(axes).tolist(), label="I / global max")
        fig.suptitle(f"{case_id} — repaired source-scale propagation")
        path = figure_root / "01_canonical_beams" / f"{case_id.lower()}_selected_xy_planes"
        _save(fig, path)
        plt.close(fig)
        outputs.extend([str(path.with_suffix(".png")), str(path.with_suffix(".pdf"))])
    return outputs


def _sweep_values(family: str, parameter: str):
    if family == "parameter":
        return parameter_registry()[parameter]
    if family == "aberration":
        return aberration_registry()[parameter]
    if family == "alignment":
        return alignment_registry()[parameter]
    raise ValueError(family)


def _source_kwargs(family: str, parameter: str, value: Any) -> dict[str, Any]:
    if family == "parameter":
        return {parameter: value}
    if family == "aberration":
        return {"zernike_name": parameter, "zernike_waves_rms": float(value)}
    if family == "alignment":
        if parameter == "axicon_decentre_x_m":
            return {"axicon_decentre_m": (float(value), 0.0)}
        if parameter == "axicon_tilt_x_rad":
            return {"axicon_tilt_rad": (float(value), 0.0)}
    raise ValueError((family, parameter))


def sweep_transverse_grid(
    case_id: str,
    *,
    family: str,
    parameter: str,
    figure_root: Path = DEFAULT_FIGURE_ROOT,
    grid_n: int = DEFAULT_REPORT_N,
    z_m: float = 60e-3,
    halfwidth_m: float = 0.35e-3,
) -> list[str]:
    import matplotlib.pyplot as plt

    values = tuple(_sweep_values(family, parameter))
    records: list[tuple[Any, np.ndarray, Mapping[str, Any]]] = []
    scales: list[float] = []
    for value in values:
        source, grid, meta = build_atlas_source(case_id, grid_n=grid_n, **_source_kwargs(family, parameter, value))
        plane = propagate_selected_planes(source, grid, float(meta["wavelength_m"]), (z_m,))[0]
        records.append((value, plane, grid))
        scales.append(float(np.max(plane)))
    common = max(scales)
    fig, axes = plt.subplots(1, len(values), figsize=(3.0 * len(values), 3.1), constrained_layout=True)
    for ax, (value, plane, grid) in zip(np.atleast_1d(axes), records):
        x, cropped = _crop(plane, grid, halfwidth_m)
        shown, _ = _normalise(cropped, common)
        im = ax.imshow(shown, origin="lower", extent=[x[0]*1e6, x[-1]*1e6, x[0]*1e6, x[-1]*1e6], vmin=0, vmax=1, cmap="inferno")
        if family == "aberration":
            title = f"{float(value):+.2f} waves RMS"
        elif "decentre" in parameter:
            title = f"{float(value)*1e6:+.0f} µm"
        elif "tilt" in parameter:
            title = f"{float(value)*1e3:+.2f} mrad"
        else:
            title = str(value)
        ax.set_title(title)
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")
    fig.colorbar(im, ax=np.atleast_1d(axes).tolist(), label="I / sweep-global max")
    fig.suptitle(f"{case_id} — {parameter} sweep at z={z_m*1e3:.0f} mm")
    sub = "02_parameter_sweeps" if family == "parameter" else "03_aberration_catalogue" if family == "aberration" else "04_alignment_errors"
    path = figure_root / sub / f"{case_id.lower()}_{parameter}_xy_sweep"
    _save(fig, path)
    plt.close(fig)
    return [str(path.with_suffix(".png")), str(path.with_suffix(".pdf"))]


def run_visual_atlas_figures(
    *,
    figure_root: Path = DEFAULT_FIGURE_ROOT,
    grid_n: int = DEFAULT_REPORT_N,
    cases: Sequence[str] = ("B0", "V1", "V3"),
) -> dict[str, Any]:
    outputs: list[str] = []
    outputs.extend(canonical_plane_grid(figure_root=figure_root, grid_n=grid_n, cases=cases))
    for case_id in cases:
        for parameter in ("beam_radius_scale", "axicon_angle_scale", "aperture_model"):
            outputs.extend(sweep_transverse_grid(case_id, family="parameter", parameter=parameter, figure_root=figure_root, grid_n=grid_n))
        for parameter in ZERNIKE_REGISTRY:
            outputs.extend(sweep_transverse_grid(case_id, family="aberration", parameter=parameter, figure_root=figure_root, grid_n=grid_n))
        for parameter in alignment_registry():
            outputs.extend(sweep_transverse_grid(case_id, family="alignment", parameter=parameter, figure_root=figure_root, grid_n=grid_n))
    manifest = {
        "outcome": "VORTEX-VISUAL-ATLAS-FIGURES",
        "grid_n": int(grid_n),
        "route": "repaired Phase 2E source-scale",
        "cases": list(cases),
        "normalisation": "shared within each sweep; canonical selected-z panels share one case-global scale",
        "figure_count": len(outputs),
        "files": outputs,
    }
    figure_root.mkdir(parents=True, exist_ok=True)
    (figure_root / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
