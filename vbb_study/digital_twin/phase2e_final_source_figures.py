"""Report-grade figures and provenance for final Phase 2E source propagation."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.ndimage import map_coordinates, zoom
from scipy.signal import savgol_filter

from vbb_study.digital_twin.phase2e_final_figure_style import (
    FINAL_FIGURE_STYLE,
    FinalFigureStyle,
)
from vbb_study.digital_twin.phase2e_final_source_propagation import (
    _make_threaded_bl_asm_propagator,
    build_final_source_field,
    CASE_CHARGES,
    FIGURE_ROOT,
    VALIDATION_ROOT,
    FinalSourcePropagationResult,
    load_final_source_result,
)
from vbb_study.digital_twin.phase2e_final_source_metrics import on_axis_intensity
from vbb_study.digital_twin.phase2e_source_sampling_repair import sampling_diagnostic


EPS = np.finfo(float).tiny
CASE_TITLES = {
    "B0": "B0 bright-core Bessel",
    "V1": "V1 charge-1 vortex Bessel",
    "V3": "V3 charge-3 vortex Bessel",
}


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2) + "\n", encoding="utf-8")
    return path


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    materialised = [dict(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in materialised:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialised)
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mpl(style: FinalFigureStyle) -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": style.font_family,
        "font.size": style.base_font_size,
        "axes.titlesize": style.title_font_size,
        "axes.labelsize": style.base_font_size,
        "legend.fontsize": style.base_font_size - 1,
        "lines.linewidth": style.line_width,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
    })
    return plt


def _save(fig: Any, stem: Path, style: FinalFigureStyle) -> tuple[Path, Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    fig.savefig(png, dpi=style.output_dpi)
    fig.savefig(pdf)
    return png, pdf


def _panel_label(axis: Any, label: str, style: FinalFigureStyle) -> None:
    axis.text(
        0.015, 0.985, label, transform=axis.transAxes, ha="left", va="top",
        fontsize=style.panel_label_font_size, fontweight="bold", color="white",
        bbox={"facecolor": "black", "alpha": 0.70, "edgecolor": "none", "pad": 2.0},
    )


def _map(
    axis: Any,
    values: np.ndarray,
    transverse_m: np.ndarray,
    z_m: np.ndarray,
    *,
    vmax: float,
    title: str,
    style: FinalFigureStyle,
    halfwidth_m: float | None = None,
) -> Any:
    keep = np.ones(transverse_m.size, dtype=bool)
    if halfwidth_m is not None:
        keep = np.abs(transverse_m) <= float(halfwidth_m)
    image = np.asarray(values[:, keep], dtype=float).T / max(float(vmax), EPS)
    shown = axis.imshow(
        image,
        origin="lower",
        aspect="auto",
        extent=[z_m[0] * 1e3, z_m[-1] * 1e3, transverse_m[keep][0] * 1e3, transverse_m[keep][-1] * 1e3],
        vmin=0.0,
        vmax=1.0,
        cmap=style.intensity_colormap,
        interpolation=style.display_interpolation,
    )
    axis.set_title(title)
    axis.set_xlabel("z (mm)")
    axis.set_ylabel("transverse position (mm)")
    return shown


def _record(
    manifest: list[dict[str, Any]],
    *,
    figure_id: str,
    case_id: str,
    route_id: str,
    result: FinalSourcePropagationResult,
    paths: tuple[Path, Path],
    style: FinalFigureStyle,
    report_role: str,
    notes: str,
    x_limits: Sequence[float],
    y_limits: Sequence[float],
    colour_limits: Sequence[float] = (0.0, 1.0),
) -> None:
    png, pdf = paths
    manifest.append({
        "figure_id": figure_id,
        "case_id": case_id,
        "route_id": route_id,
        "aperture_model": result.metadata["aperture_model"],
        "source_grid_n": result.metadata["source_grid_n"],
        "source_dx_m": result.metadata["dx_m"],
        "samples_per_radial_period": result.metadata["samples_per_axicon_radial_period"],
        "z_step_m": result.metadata["z_step_m"],
        "x_limits": json.dumps(list(x_limits)),
        "y_limits": json.dumps(list(y_limits)),
        "z_limits": json.dumps(list(style.z_limits_m)),
        "normalisation_policy": style.primary_normalisation,
        "colour_limits": json.dumps(list(colour_limits)),
        "colourmap": style.intensity_colormap,
        "metric_bearing": True,
        "metrics_computed_on_native_arrays": True,
        "display_interpolation": style.display_interpolation,
        "display_interpolation_used_for_metrics": False,
        "report_role": report_role,
        "calibration_required": result.metadata["calibration_required"],
        "png_path": png.as_posix(),
        "pdf_path": pdf.as_posix(),
        "sha256": _sha256(png),
        "notes": notes,
    })


def render_primary(
    result: FinalSourcePropagationResult,
    manifest: list[dict[str, Any]],
    *,
    output_root: Path = FIGURE_ROOT,
    style: FinalFigureStyle = FINAL_FIGURE_STYLE,
) -> tuple[Path, Path]:
    plt = _mpl(style)
    case_id = str(result.metadata["case_id"])
    vmax = max(float(np.max(result.xz_intensity)), float(np.max(result.yz_intensity)), EPS)
    fig, axes = plt.subplots(2, 4, figsize=style.primary_figsize_inches, constrained_layout=True)
    maps = (
        (axes[0, 0], result.xz_intensity, "full-field x-z", None),
        (axes[0, 1], result.yz_intensity, "full-field y-z", None),
        (axes[0, 2], result.xz_intensity, "fixed-detail x-z", style.detail_halfwidth_m),
        (axes[0, 3], result.yz_intensity, "fixed-detail y-z", style.detail_halfwidth_m),
    )
    image = None
    for index, (axis, values, title, crop) in enumerate(maps):
        image = _map(axis, values, result.x_m, result.z_m, vmax=vmax, title=title, style=style, halfwidth_m=crop)
        _panel_label(axis, f"({chr(97 + index)})", style)
    assert image is not None
    fig.colorbar(image, ax=list(axes[0]), shrink=0.86, label="I / global Imax (linear)")

    primary = result.axial_trace_raw / max(float(np.max(result.axial_trace_raw)), EPS)
    axes[1, 0].plot(result.z_m * 1e3, primary, color="#0072B2")
    axes[1, 0].set(title="raw axis/ring observable", xlabel="z (mm)", ylabel="normalised raw observable")
    bucket = result.fixed_bucket_power_raw / max(float(np.max(result.fixed_bucket_power_raw)), EPS)
    axes[1, 1].plot(result.z_m * 1e3, bucket, color="#009E73")
    axes[1, 1].set(title="fixed core/annulus power", xlabel="z (mm)", ylabel="normalised fixed power")
    axes[1, 2].plot(result.z_m * 1e3, result.feature_radius_m * 1e6, color="#D55E00")
    axes[1, 2].set(title="valid core/ring radius", xlabel="z (mm)", ylabel="radius (um)")
    power = result.total_plane_power / max(float(result.metadata["source_power_raw"]), EPS)
    power_loss_ppm = (1.0 - power) * 1e6
    edge_ppm = result.edge_energy_fraction * 1e6
    axes[1, 3].plot(result.z_m * 1e3, power_loss_ppm, color="#4D4D4D", label="power loss")
    axes[1, 3].plot(result.z_m * 1e3, edge_ppm, color="#CC79A7", label="edge fraction")
    axes[1, 3].set(title="power-loss and edge audit", xlabel="z (mm)", ylabel="fraction (ppm)")
    axes[1, 3].legend(frameon=False, fontsize=8)
    for index, axis in enumerate(axes[1]):
        axis.grid(alpha=0.22)
        axis.text(0.015, 0.985, f"({chr(101 + index)})", transform=axis.transAxes, ha="left", va="top", fontweight="bold")
    fig.suptitle(f"{CASE_TITLES[case_id]} | final source-scale propagation | global linear intensity")
    stem = output_root / "01_primary_propagation" / f"{case_id.lower()}_source_scale_propagation_primary"
    paths = _save(fig, stem, style)
    plt.close(fig)
    _record(
        manifest, figure_id=stem.name, case_id=case_id,
        route_id=str(result.metadata["route_id"]), result=result, paths=paths, style=style,
        report_role="primary", notes="Eight-panel native-metric propagation summary; no log, gamma, dB or per-z normalisation.",
        x_limits=(float(result.x_m[0]), float(result.x_m[-1])), y_limits=(-style.detail_halfwidth_m, style.detail_halfwidth_m),
    )
    return paths


def _snapshot_nearest_strict_centre(result: FinalSourcePropagationResult) -> tuple[float, np.ndarray]:
    zone = result.metadata["zones"]["measured_strict_useful_region_m"]
    target = 0.5 * (float(zone[0]) + float(zone[1]))
    z_value = min(result.snapshot_fields, key=lambda value: abs(value - target))
    return float(z_value), result.snapshot_fields[z_value]


def measured_winding(result: FinalSourcePropagationResult) -> float:
    if str(result.metadata["case_id"]) == "B0":
        return 0.0
    _, field = _snapshot_nearest_strict_centre(result)
    x = np.asarray(result.metadata["snapshot_x_m"], dtype=float)
    radius = float(result.metadata["zones"]["reference_feature_radius_m"])
    theta = np.linspace(0.0, 2.0 * np.pi, 721)
    xq = radius * np.cos(theta)
    yq = radius * np.sin(theta)
    ix = (xq - x[0]) / float(np.median(np.diff(x)))
    iy = (yq - x[0]) / float(np.median(np.diff(x)))
    sampled = map_coordinates(field.real, [iy, ix], order=1, mode="nearest") + 1j * map_coordinates(field.imag, [iy, ix], order=1, mode="nearest")
    phase = np.unwrap(np.angle(sampled))
    return float((phase[-1] - phase[0]) / (2.0 * np.pi))


def render_snapshots(
    result: FinalSourcePropagationResult,
    manifest: list[dict[str, Any]],
    *,
    output_root: Path = FIGURE_ROOT,
    style: FinalFigureStyle = FINAL_FIGURE_STYLE,
) -> tuple[Path, Path]:
    plt = _mpl(style)
    case_id = str(result.metadata["case_id"])
    x = np.asarray(result.metadata["snapshot_x_m"], dtype=float)
    keep = np.abs(x) <= style.snapshot_halfwidth_m
    snapshots = [(z, np.abs(field[np.ix_(keep, keep)]) ** 2) for z, field in result.snapshot_fields.items()]
    vmax = max(float(np.max(image)) for _, image in snapshots)
    fig, axes = plt.subplots(2, 3, figsize=style.snapshot_figsize_inches, constrained_layout=True)
    image_artist = None
    for index, (axis, (z_value, image)) in enumerate(zip(axes.flat, snapshots)):
        image_artist = axis.imshow(
            image / max(vmax, EPS), origin="lower",
            extent=[x[keep][0] * 1e3, x[keep][-1] * 1e3, x[keep][0] * 1e3, x[keep][-1] * 1e3],
            vmin=0.0, vmax=1.0, cmap=style.intensity_colormap,
            interpolation=style.display_interpolation,
        )
        axis.set(title=f"z = {z_value * 1e3:.2f} mm", xlabel="x (mm)", ylabel="y (mm)")
        _panel_label(axis, f"({chr(97 + index)})", style)
    assert image_artist is not None
    fig.colorbar(image_artist, ax=list(axes.flat), shrink=0.86, label="I / matched snapshot Imax (linear)")
    fig.suptitle(f"{CASE_TITLES[case_id]} | matched native transverse snapshots")
    stem = output_root / "04_transverse_snapshots" / f"{case_id.lower()}_transverse_snapshots"
    paths = _save(fig, stem, style)
    plt.close(fig)
    _record(
        manifest, figure_id=stem.name, case_id=case_id, route_id=str(result.metadata["route_id"]),
        result=result, paths=paths, style=style, report_role="primary",
        notes="Six semantic z snapshots with one crop and one matched linear colour scale within the case.",
        x_limits=(-style.snapshot_halfwidth_m, style.snapshot_halfwidth_m),
        y_limits=(-style.snapshot_halfwidth_m, style.snapshot_halfwidth_m),
    )
    return paths


def render_surface(
    result: FinalSourcePropagationResult,
    manifest: list[dict[str, Any]],
    *,
    output_root: Path = FIGURE_ROOT,
    style: FinalFigureStyle = FINAL_FIGURE_STYLE,
) -> tuple[Path, Path]:
    plt = _mpl(style)
    case_id = str(result.metadata["case_id"])
    z_value, field = _snapshot_nearest_strict_centre(result)
    x = np.asarray(result.metadata["snapshot_x_m"], dtype=float)
    keep = np.abs(x) <= style.surface_halfwidth_m
    coordinate = x[keep] * 1e6
    intensity = np.abs(field[np.ix_(keep, keep)]) ** 2
    intensity /= max(float(np.max(intensity)), EPS)
    factor = int(style.surface_display_upsampling)
    display = zoom(intensity, factor, order=3) if factor > 1 else intensity
    display_x = np.linspace(coordinate[0], coordinate[-1], display.shape[0])
    X, Y = np.meshgrid(display_x, display_x, indexing="xy")
    fig = plt.figure(figsize=(14.0, style.surface_figsize_inches[1]), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=(1.0, 0.12, 1.0))
    oblique = fig.add_subplot(grid[0, 0], projection="3d")
    top = fig.add_subplot(grid[0, 2])
    oblique.plot_surface(X, Y, display, cmap=style.intensity_colormap, vmin=0.0, vmax=1.0, linewidth=0, antialiased=True)
    oblique.set(xlabel="x (um)", ylabel="y (um)", zlim=style.surface_intensity_limits)
    oblique.set_zlabel("I / Imax", labelpad=2)
    oblique.view_init(elev=style.surface_elevation_deg, azim=style.surface_azimuth_deg)
    oblique.set_title("oblique intensity surface")
    shown = top.imshow(display, origin="lower", extent=[display_x[0], display_x[-1], display_x[0], display_x[-1]], vmin=0.0, vmax=1.0, cmap=style.intensity_colormap, interpolation="none")
    top.set(title="top-down parity view", xlabel="x (um)", ylabel="y (um)")
    fig.colorbar(shown, ax=[oblique, top], shrink=0.72, label="I / Imax (linear)")
    fig.suptitle(f"{CASE_TITLES[case_id]} | transverse intensity at z={z_value * 1e3:.2f} mm")
    stem = output_root / "06_3d_surfaces" / f"{case_id.lower()}_transverse_intensity_surface"
    paths = _save(fig, stem, style)
    plt.close(fig)
    _record(
        manifest, figure_id=stem.name, case_id=case_id, route_id=str(result.metadata["route_id"]), result=result,
        paths=paths, style=style, report_role="primary", notes=f"Display-only cubic upsampling factor {factor}; top-down uses the identical surface array.",
        x_limits=(-style.surface_halfwidth_m, style.surface_halfwidth_m), y_limits=(-style.surface_halfwidth_m, style.surface_halfwidth_m),
    )
    return paths


def render_aperture_comparison(
    results: Mapping[str, FinalSourcePropagationResult],
    manifest: list[dict[str, Any]],
    *,
    output_root: Path = FIGURE_ROOT,
    style: FinalFigureStyle = FINAL_FIGURE_STYLE,
) -> tuple[Path, Path]:
    plt = _mpl(style)
    route_order = (
        "nominal_no_additional_aperture",
        "soft_aperture_sensitivity",
        "hard_aperture_diagnostic",
    )
    nominal = results[route_order[0]]
    case_id = str(nominal.metadata["case_id"])
    keep = np.abs(nominal.x_m) <= style.detail_halfwidth_m
    vmax = max(
        max(float(np.max(results[route].xz_intensity)), float(np.max(results[route].yz_intensity)))
        for route in route_order
    )
    difference_max = max(
        float(np.max(np.abs(results[route].xz_intensity[:, keep] - nominal.xz_intensity[:, keep])))
        for route in route_order[1:]
    )
    difference_max = max(
        difference_max,
        max(float(np.max(np.abs(results[route].yz_intensity[:, keep] - nominal.yz_intensity[:, keep]))) for route in route_order[1:]),
        EPS,
    )
    fig, axes = plt.subplots(5, 3, figsize=(16.0, 17.0), constrained_layout=True)
    extent = [nominal.z_m[0] * 1e3, nominal.z_m[-1] * 1e3, nominal.x_m[keep][0] * 1e3, nominal.x_m[keep][-1] * 1e3]
    labels = ("nominal: no additional aperture", "soft assumed truncation", "hard diagnostic truncation")
    for column, (route, label) in enumerate(zip(route_order, labels)):
        result = results[route]
        axes[0, column].imshow(result.xz_intensity[:, keep].T / max(vmax, EPS), origin="lower", aspect="auto", extent=extent, vmin=0, vmax=1, cmap=style.intensity_colormap, interpolation=style.display_interpolation)
        axes[1, column].imshow(result.yz_intensity[:, keep].T / max(vmax, EPS), origin="lower", aspect="auto", extent=extent, vmin=0, vmax=1, cmap=style.intensity_colormap, interpolation=style.display_interpolation)
        dxz = np.abs(result.xz_intensity[:, keep] - nominal.xz_intensity[:, keep]).T / difference_max
        dyz = np.abs(result.yz_intensity[:, keep] - nominal.yz_intensity[:, keep]).T / difference_max
        axes[2, column].imshow(dxz, origin="lower", aspect="auto", extent=extent, vmin=0, vmax=1, cmap="viridis", interpolation=style.display_interpolation)
        axes[3, column].imshow(dyz, origin="lower", aspect="auto", extent=extent, vmin=0, vmax=1, cmap="viridis", interpolation=style.display_interpolation)
        axes[0, column].set_title(label)
        for row in range(4):
            axes[row, column].set_xlabel("z (mm)")
            axes[row, column].set_ylabel("position (mm)")
    axes[0, 0].set_ylabel("x (mm)\nmatched x-z")
    axes[1, 0].set_ylabel("y (mm)\nmatched y-z")
    axes[2, 0].set_ylabel("x (mm)\nabsolute x-z difference")
    axes[3, 0].set_ylabel("y (mm)\nabsolute y-z difference")
    colours = ("#0072B2", "#009E73", "#D55E00")
    for route, label, colour in zip(route_order, labels, colours):
        result = results[route]
        axes[4, 0].plot(result.z_m * 1e3, result.axial_trace_raw / max(float(np.max(result.axial_trace_raw)), EPS), label=label, color=colour)
        axes[4, 1].plot(result.z_m * 1e3, result.fixed_bucket_power_raw / max(float(np.max(result.fixed_bucket_power_raw)), EPS), label=label, color=colour)
        axes[4, 2].plot(result.z_m * 1e3, result.feature_radius_m * 1e6, label=label, color=colour)
    axes[4, 0].set(title="raw observable", xlabel="z (mm)", ylabel="normalised raw value")
    axes[4, 1].set(title="fixed bucket/annulus", xlabel="z (mm)", ylabel="normalised power")
    axes[4, 2].set(title="valid feature radius", xlabel="z (mm)", ylabel="radius (um)")
    axes[4, 0].legend(frameon=False, fontsize=8)
    for axis in axes[4]:
        axis.grid(alpha=0.22)
    fig.suptitle(f"{CASE_TITLES[case_id]} | aperture-route comparison\nhard route is diagnostic only, not a nominal experimental prediction")
    stem = output_root / "02_aperture_comparison" / f"{case_id.lower()}_aperture_route_comparison"
    paths = _save(fig, stem, style)
    plt.close(fig)
    _record(
        manifest, figure_id=stem.name, case_id=case_id, route_id="three_route_aperture_comparison",
        result=nominal, paths=paths, style=style, report_role="sensitivity_comparison",
        notes="Matched axes, z grid, raw linear normalisation and colour limits. Hard route diagnostic only.",
        x_limits=(-style.detail_halfwidth_m, style.detail_halfwidth_m),
        y_limits=(-style.detail_halfwidth_m, style.detail_halfwidth_m),
    )
    return paths


def render_profiles_and_metrics(
    results: Mapping[str, FinalSourcePropagationResult],
    manifest: list[dict[str, Any]],
    *,
    output_root: Path = FIGURE_ROOT,
    style: FinalFigureStyle = FINAL_FIGURE_STYLE,
) -> tuple[Path, Path]:
    plt = _mpl(style)
    fig, axes = plt.subplots(3, 3, figsize=(15.0, 11.0), constrained_layout=True, sharex="col")
    for row, case_id in enumerate(CASE_CHARGES):
        result = results[case_id]
        axes[row, 0].plot(result.z_m * 1e3, result.axial_trace_raw / max(float(np.max(result.axial_trace_raw)), EPS), color="#0072B2")
        axes[row, 1].plot(result.z_m * 1e3, result.feature_radius_m * 1e6, color="#D55E00", label="feature radius")
        if case_id != "B0":
            axes[row, 1].plot(result.z_m * 1e3, result.dark_core_radius_m * 1e6, color="#CC79A7", label="dark-core radius")
        axes[row, 2].plot(result.z_m * 1e3, result.feature_width_m * 1e6, color="#009E73")
        axes[row, 0].set_ylabel(f"{case_id}\nnormalised raw")
        axes[row, 1].set_ylabel("radius (um)")
        axes[row, 2].set_ylabel("width (um)")
        for axis in axes[row]:
            axis.grid(alpha=0.22)
        if case_id != "B0":
            axes[row, 1].legend(frameon=False, fontsize=8)
    axes[0, 0].set_title("axis/ring observable")
    axes[0, 1].set_title("feature and dark-core radii")
    axes[0, 2].set_title("core/ring FWHM width")
    for axis in axes[-1]:
        axis.set_xlabel("z (mm)")
    fig.suptitle("Final source-scale native profiles and validity-aware feature metrics")
    stem = output_root / "05_profiles_and_metrics" / "bessel_family_profiles_and_metrics"
    paths = _save(fig, stem, style)
    plt.close(fig)
    nominal = results["B0"]
    _record(
        manifest, figure_id=stem.name, case_id="B0,V1,V3", route_id="nominal_no_additional_aperture",
        result=nominal, paths=paths, style=style, report_role="primary_metrics",
        notes="NaN invalid radii remain disconnected; no zero substitution.",
        x_limits=(0.0, 0.180), y_limits=(0.0, 0.180),
    )
    return paths


def render_hero_family(
    results: Mapping[str, FinalSourcePropagationResult],
    manifest: list[dict[str, Any]],
    *,
    output_root: Path = FIGURE_ROOT,
    style: FinalFigureStyle = FINAL_FIGURE_STYLE,
) -> tuple[Path, Path]:
    plt = _mpl(style)
    vmax = max(max(float(np.max(result.xz_intensity)), float(np.max(result.yz_intensity))) for result in results.values())
    fig, axes = plt.subplots(3, 3, figsize=(14.0, 12.0), constrained_layout=True)
    for row, case_id in enumerate(CASE_CHARGES):
        result = results[case_id]
        keep = np.abs(result.x_m) <= style.detail_halfwidth_m
        extent = [result.z_m[0] * 1e3, result.z_m[-1] * 1e3, result.x_m[keep][0] * 1e3, result.x_m[keep][-1] * 1e3]
        axes[row, 0].imshow(result.xz_intensity[:, keep].T / max(vmax, EPS), origin="lower", aspect="auto", extent=extent, vmin=0, vmax=1, cmap=style.intensity_colormap, interpolation=style.display_interpolation)
        z_value, field = _snapshot_nearest_strict_centre(result)
        sx = np.asarray(result.metadata["snapshot_x_m"], dtype=float)
        sk = np.abs(sx) <= style.snapshot_halfwidth_m
        snapshot = np.abs(field[np.ix_(sk, sk)]) ** 2
        axes[row, 1].imshow(snapshot / max(float(np.max(snapshot)), EPS), origin="lower", extent=[sx[sk][0] * 1e3, sx[sk][-1] * 1e3, sx[sk][0] * 1e3, sx[sk][-1] * 1e3], vmin=0, vmax=1, cmap=style.intensity_colormap, interpolation=style.display_interpolation)
        axes[row, 2].plot(result.z_m * 1e3, result.axial_trace_raw / max(float(np.max(result.axial_trace_raw)), EPS), color="#0072B2")
        axes[row, 0].set_ylabel(f"{case_id}\nposition (mm)")
        axes[row, 1].set_ylabel("y (mm)")
        axes[row, 1].set_title(f"z={z_value * 1e3:.2f} mm")
        axes[row, 2].set_ylabel("normalised raw observable")
        axes[row, 2].grid(alpha=0.22)
    axes[0, 0].set_title("matched detail x-z | shared raw global Imax")
    axes[0, 1].set_title("strict-region transverse morphology")
    axes[0, 2].set_title("case-aware axial/ring trace")
    for row in range(3):
        axes[row, 0].set_xlabel("z (mm)")
        axes[row, 1].set_xlabel("x (mm)")
        axes[row, 2].set_xlabel("z (mm)")
    fig.suptitle("Source-scale Bessel family propagation | N=3072, dz=0.25 mm")
    stem = output_root / "07_report_hero_figures" / "hero_bessel_family_source_propagation"
    paths = _save(fig, stem, style)
    plt.close(fig)
    _record(
        manifest, figure_id=stem.name, case_id="B0,V1,V3", route_id="nominal_no_additional_aperture",
        result=results["B0"], paths=paths, style=style, report_role="hero",
        notes="Matched family detail maps, strict-region snapshots and case-aware raw traces.",
        x_limits=(-style.detail_halfwidth_m, style.detail_halfwidth_m), y_limits=(-style.snapshot_halfwidth_m, style.snapshot_halfwidth_m),
    )
    return paths


def render_hero_aperture(
    all_results: Mapping[str, Mapping[str, FinalSourcePropagationResult]],
    manifest: list[dict[str, Any]],
    *,
    output_root: Path = FIGURE_ROOT,
    style: FinalFigureStyle = FINAL_FIGURE_STYLE,
) -> tuple[Path, Path]:
    plt = _mpl(style)
    routes = (
        "nominal_no_additional_aperture",
        "soft_aperture_sensitivity",
        "hard_aperture_diagnostic",
    )
    labels = ("no additional aperture", "soft assumed truncation", "hard diagnostic truncation")
    fig, axes = plt.subplots(3, 4, figsize=(17.0, 12.0), constrained_layout=True)
    for row, case_id in enumerate(CASE_CHARGES):
        case_results = all_results[case_id]
        nominal = case_results[routes[0]]
        keep = np.abs(nominal.x_m) <= style.detail_halfwidth_m
        vmax = max(float(np.max(case_results[route].xz_intensity[:, keep])) for route in routes)
        extent = [nominal.z_m[0] * 1e3, nominal.z_m[-1] * 1e3, nominal.x_m[keep][0] * 1e3, nominal.x_m[keep][-1] * 1e3]
        for column, (route, label) in enumerate(zip(routes, labels)):
            result = case_results[route]
            axes[row, column].imshow(result.xz_intensity[:, keep].T / max(vmax, EPS), origin="lower", aspect="auto", extent=extent, vmin=0, vmax=1, cmap=style.intensity_colormap, interpolation=style.display_interpolation)
            axes[row, column].set(title=label if row == 0 else "", xlabel="z (mm)", ylabel=f"{case_id}\nx (mm)" if column == 0 else "x (mm)")
            axes[row, 3].plot(result.z_m * 1e3, result.axial_trace_raw / max(float(np.max(result.axial_trace_raw)), EPS), label=label)
        axes[row, 3].set(title="raw trace overlay" if row == 0 else "", xlabel="z (mm)", ylabel="normalised raw observable")
        axes[row, 3].grid(alpha=0.22)
    axes[0, 3].legend(frameon=False, fontsize=8)
    fig.suptitle("Aperture effect on the source-scale Bessel family\nhard truncation is diagnostic only")
    stem = output_root / "07_report_hero_figures" / "hero_aperture_effect_on_bessel_zone"
    paths = _save(fig, stem, style)
    plt.close(fig)
    _record(
        manifest, figure_id=stem.name, case_id="B0,V1,V3", route_id="three_route_aperture_comparison",
        result=all_results["B0"][routes[0]], paths=paths, style=style, report_role="hero",
        notes="Matched route maps and ring-aware/raw trace overlays; hard route diagnostic only.",
        x_limits=(-style.detail_halfwidth_m, style.detail_halfwidth_m), y_limits=(-style.detail_halfwidth_m, style.detail_halfwidth_m),
    )
    return paths


def render_hero_surfaces(
    results: Mapping[str, FinalSourcePropagationResult],
    manifest: list[dict[str, Any]],
    *,
    output_root: Path = FIGURE_ROOT,
    style: FinalFigureStyle = FINAL_FIGURE_STYLE,
) -> tuple[Path, Path]:
    plt = _mpl(style)
    fig = plt.figure(figsize=(18.0, 6.0), constrained_layout=True)
    for column, case_id in enumerate(CASE_CHARGES, start=1):
        result = results[case_id]
        z_value, field = _snapshot_nearest_strict_centre(result)
        x = np.asarray(result.metadata["snapshot_x_m"], dtype=float)
        keep = np.abs(x) <= style.surface_halfwidth_m
        coordinate = x[keep] * 1e6
        intensity = np.abs(field[np.ix_(keep, keep)]) ** 2
        intensity /= max(float(np.max(intensity)), EPS)
        display = zoom(intensity, int(style.surface_display_upsampling), order=3)
        display_x = np.linspace(coordinate[0], coordinate[-1], display.shape[0])
        X, Y = np.meshgrid(display_x, display_x, indexing="xy")
        axis = fig.add_subplot(1, 3, column, projection="3d")
        axis.plot_surface(X, Y, display, cmap=style.intensity_colormap, vmin=0, vmax=1, linewidth=0, antialiased=True)
        axis.set(xlabel="x (um)", ylabel="y (um)", zlim=style.surface_intensity_limits, title=f"{case_id} | z={z_value * 1e3:.2f} mm")
        axis.set_zlabel("I / Imax", labelpad=-1)
        axis.view_init(elev=style.surface_elevation_deg, azim=style.surface_azimuth_deg)
    fig.suptitle("Matched B0, V1 and V3 transverse intensity surfaces")
    stem = output_root / "07_report_hero_figures" / "hero_b0_v1_v3_3d_intensity"
    paths = _save(fig, stem, style)
    plt.close(fig)
    _record(
        manifest, figure_id=stem.name, case_id="B0,V1,V3", route_id="nominal_no_additional_aperture",
        result=results["B0"], paths=paths, style=style, report_role="hero",
        notes=f"Identical x/y/z limits, camera and display-only cubic upsampling factor {style.surface_display_upsampling}.",
        x_limits=(-style.surface_halfwidth_m, style.surface_halfwidth_m), y_limits=(-style.surface_halfwidth_m, style.surface_halfwidth_m),
    )
    return paths


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sampling_rows() -> list[dict[str, Any]]:
    n_values = (512, 1024, 1536, 2048, 2560, 3072)
    gate_rows = _read_csv(VALIDATION_ROOT / "final_resolution_gate.csv")
    gate_b0 = {
        int(row["grid_n"]): row
        for row in gate_rows
        if row["case_id"] == "B0" and abs(float(row["z_m"]) - 0.060) < 1e-12
    }
    rows: list[dict[str, Any]] = []
    for n in n_values:
        diagnostic = sampling_diagnostic(n)
        if n in gate_b0:
            intensity = float(gate_b0[n]["on_axis_intensity_raw"])
        else:
            source, grid, metadata = build_final_source_field("B0", grid_n=n)
            propagate = _make_threaded_bl_asm_propagator(source, grid, float(metadata["wavelength_m"]))
            intensity = on_axis_intensity(np.abs(propagate(0.060)) ** 2)
            del propagate, source, grid
        rows.append({
            **diagnostic.__dict__,
            "case_id": "B0",
            "route_id": "nominal_no_additional_aperture",
            "selected_plane_z_m": 0.060,
            "on_axis_intensity_raw": intensity,
        })
    reference = rows[-1]["on_axis_intensity_raw"]
    for row in rows:
        row["on_axis_relative_difference_to_n3072"] = abs(row["on_axis_intensity_raw"] - reference) / max(abs(reference), EPS)
        row["report_category"] = {
            "invalid": "invalid",
            "marginal": "marginal",
            "acceptable_for_screening": "screening",
            "quantitative_reference": "quantitative-reference candidate",
        }[row["sampling_class"]]
    return rows


def _n512_xz_map(z_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source, grid, metadata = build_final_source_field("B0", grid_n=512)
    propagate = _make_threaded_bl_asm_propagator(source, grid, float(metadata["wavelength_m"]))
    centre = 256
    values = np.empty((z_m.size, 512), dtype=np.float32)
    for index, z_value in enumerate(z_m):
        field = source if float(z_value) == 0.0 else propagate(float(z_value))
        values[index] = np.abs(field[centre, :]) ** 2
    return np.asarray(grid["x"], dtype=float), values


def render_sampling_convergence(
    final_b0: FinalSourcePropagationResult,
    manifest: list[dict[str, Any]],
    *,
    output_root: Path = FIGURE_ROOT,
    style: FinalFigureStyle = FINAL_FIGURE_STYLE,
) -> tuple[tuple[Path, Path], list[dict[str, Any]]]:
    plt = _mpl(style)
    rows = _sampling_rows()
    _write_csv(VALIDATION_ROOT / "final_sampling_convergence.csv", rows)
    kr = float(rows[-1]["kr_m_inv"])
    period = float(rows[-1]["radial_period_m"])
    phase_images = []
    for n in (512, 3072):
        dx = 10.0e-3 / n
        coordinate = np.arange(-3.0 * period, 3.0 * period + dx, dx)
        X, Y = np.meshgrid(coordinate, coordinate, indexing="xy")
        phase_images.append(np.mod(-kr * np.hypot(X, Y) + np.pi, 2.0 * np.pi) - np.pi)
    previous_x, previous_xz = _n512_xz_map(final_b0.z_m)
    keep_final = np.abs(final_b0.x_m) <= style.detail_halfwidth_m
    final_detail = final_b0.xz_intensity[:, keep_final]
    previous_on_final = np.empty_like(final_detail, dtype=float)
    for index in range(final_b0.z_m.size):
        previous_on_final[index] = np.interp(final_b0.x_m[keep_final], previous_x, previous_xz[index])
    previous_norm = previous_on_final / max(float(np.max(previous_on_final)), EPS)
    final_norm = final_detail / max(float(np.max(final_detail)), EPS)
    difference = final_norm - previous_norm
    extent = [final_b0.z_m[0] * 1e3, final_b0.z_m[-1] * 1e3, final_b0.x_m[keep_final][0] * 1e3, final_b0.x_m[keep_final][-1] * 1e3]

    fig, axes = plt.subplots(2, 4, figsize=(17.0, 9.5), constrained_layout=True)
    phase_display = [
        zoom(image, (192.0 / image.shape[0], 192.0 / image.shape[1]), order=0)
        for image in phase_images
    ]
    combined_phase = np.hstack(phase_display)
    axes[0, 0].imshow(combined_phase, origin="lower", cmap="twilight", vmin=-np.pi, vmax=np.pi, interpolation="nearest")
    axes[0, 0].axvline(phase_display[0].shape[1] - 0.5, color="white", linewidth=1.0)
    axes[0, 0].set_title("wrapped axicon phase | N512 left, N3072 right")
    radial = np.linspace(0.0, 4.0 * period, 400)
    for n, colour in ((512, "#D55E00"), (3072, "#0072B2")):
        dx = 10.0e-3 / n
        samples = np.arange(0.0, 4.0 * period + dx, dx)
        axes[0, 1].plot(samples * 1e6, np.mod(-kr * samples + np.pi, 2 * np.pi) - np.pi, marker="o", markersize=2.5, label=f"N={n}", color=colour)
    axes[0, 1].set(title="native radial phase samples", xlabel="radius (um)", ylabel="wrapped phase (rad)")
    axes[0, 1].legend(frameon=False)
    n_axis = [row["grid_n"] for row in rows]
    axes[0, 2].plot(n_axis, [row["samples_per_radial_period"] for row in rows], marker="o")
    axes[0, 2].set(title="samples per radial period", xlabel="source grid N", ylabel="samples / period")
    axes[0, 3].plot(n_axis, [row["adjacent_radial_phase_increment_rad"] for row in rows], marker="o", color="#CC79A7")
    axes[0, 3].set(title="adjacent radial phase increment", xlabel="source grid N", ylabel="phase step (rad)")
    axes[1, 0].plot(n_axis, [row["on_axis_intensity_raw"] for row in rows], marker="o", color="#009E73")
    axes[1, 0].set(title="B0 raw on-axis intensity at z=60 mm", xlabel="source grid N", ylabel="raw intensity")
    axes[1, 1].semilogy(n_axis, np.maximum([row["on_axis_relative_difference_to_n3072"] for row in rows], 1e-8), marker="o")
    axes[1, 1].axhline(0.01, color="black", linestyle="--", linewidth=1.0)
    axes[1, 1].set(title="relative difference to N3072", xlabel="source grid N", ylabel="relative difference")
    joined = np.hstack((previous_norm.T, final_norm.T))
    axes[1, 2].imshow(joined, origin="lower", aspect="auto", vmin=0, vmax=1, cmap=style.intensity_colormap, interpolation=style.display_interpolation)
    axes[1, 2].axvline(previous_norm.shape[0] - 0.5, color="white", linewidth=1.0)
    axes[1, 2].set_title("N512 left versus N3072 right | matched detail")
    diff_limit = max(float(np.max(np.abs(difference))), EPS)
    axes[1, 3].imshow(difference.T, origin="lower", aspect="auto", extent=extent, vmin=-diff_limit, vmax=diff_limit, cmap=style.difference_colormap, interpolation=style.display_interpolation)
    axes[1, 3].set(title="N3072 - interpolated N512", xlabel="z (mm)", ylabel="x (mm)")
    for axis in axes.flat:
        axis.grid(alpha=0.14)
    fig.suptitle("Final source-sampling convergence | thresholds are project-specific candidates, not universal laws")
    stem = output_root / "03_sampling_convergence" / "source_sampling_convergence"
    paths = _save(fig, stem, style)
    plt.close(fig)
    _record(
        manifest, figure_id=stem.name, case_id="B0", route_id="nominal_no_additional_aperture",
        result=final_b0, paths=paths, style=style, report_role="sampling_validation",
        notes="N512 comparison is display-resampled only; every gate metric uses native arrays.",
        x_limits=(-style.detail_halfwidth_m, style.detail_halfwidth_m), y_limits=(-style.detail_halfwidth_m, style.detail_halfwidth_m),
    )
    hero_stem = output_root / "07_report_hero_figures" / "hero_source_sampling_convergence"
    hero_stem.parent.mkdir(parents=True, exist_ok=True)
    hero_paths = (hero_stem.with_suffix(".png"), hero_stem.with_suffix(".pdf"))
    shutil.copy2(paths[0], hero_paths[0])
    shutil.copy2(paths[1], hero_paths[1])
    _record(
        manifest, figure_id=hero_stem.name, case_id="B0", route_id="nominal_no_additional_aperture",
        result=final_b0, paths=hero_paths, style=style, report_role="hero",
        notes="Hero copy of the governed source-sampling convergence figure.",
        x_limits=(-style.detail_halfwidth_m, style.detail_halfwidth_m), y_limits=(-style.detail_halfwidth_m, style.detail_halfwidth_m),
    )
    return paths, rows


def _interval_text(interval: Sequence[float] | None) -> tuple[float | None, float | None]:
    if interval is None:
        return None, None
    return float(interval[0]), float(interval[1])


def _case_and_zone_rows(
    nominal: Mapping[str, FinalSourcePropagationResult],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    case_rows: list[dict[str, Any]] = []
    zone_rows: list[dict[str, Any]] = []
    for case_id, result in nominal.items():
        zones = result.metadata["zones"]
        strict = zones["measured_strict_useful_region_m"]
        strict_mask = (result.z_m >= float(strict[0])) & (result.z_m <= float(strict[1])) & result.feature_valid
        case_rows.append({
            "case_id": case_id,
            "route_id": result.metadata["route_id"],
            "grid_n": result.metadata["source_grid_n"],
            "dx_m": result.metadata["dx_m"],
            "z_step_m": result.metadata["z_step_m"],
            "reference_feature_radius_m": zones["reference_feature_radius_m"],
            "median_strict_feature_width_m": float(np.nanmedian(result.feature_width_m[strict_mask])),
            "median_strict_dark_core_radius_m": float(np.nanmedian(result.dark_core_radius_m[strict_mask])) if case_id != "B0" else None,
            "requested_winding": CASE_CHARGES[case_id],
            "measured_winding": measured_winding(result),
            "maximum_edge_energy_fraction": result.metadata["maximum_edge_energy_fraction"],
            "maximum_propagation_power_drift_fraction": result.metadata["maximum_propagation_power_drift_fraction"],
            "runtime_seconds": result.metadata["runtime_seconds"],
        })
        definitions = (
            ("configured_nominal_interval", zones["configured_nominal_interval_m"], "configuration_reference_only"),
            ("geometric_zone_estimate", zones["geometric_zone_estimate_m"], "beam_radius_and_axicon_geometry"),
            ("measured_FWHM_axial_zone", zones["measured_FWHM_axial_zone_m"], "raw_case_aware_observable"),
            ("measured_strict_useful_region", zones["measured_strict_useful_region_m"], "raw_plus_fixed_power_plus_radius_validity"),
        )
        for definition, interval, provenance in definitions:
            start, stop = _interval_text(interval)
            zone_rows.append({
                "case_id": case_id,
                "route_id": result.metadata["route_id"],
                "zone_definition": definition,
                "start_m": start,
                "stop_m": stop,
                "provenance": provenance,
                "is_measured": definition.startswith("measured_"),
            })
    return case_rows, zone_rows


def _aperture_rows(
    all_results: Mapping[str, Mapping[str, FinalSourcePropagationResult]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case_id, route_results in all_results.items():
        for route_id, result in route_results.items():
            common = (result.z_m >= 0.020) & (result.z_m <= 0.120)
            trace = result.axial_trace_raw[common]
            ripple_region = (result.z_m >= 0.025) & (result.z_m <= 0.090)
            ripple_trace = result.axial_trace_raw[ripple_region]
            smooth = savgol_filter(ripple_trace, 81, 3)
            residual = ripple_trace - smooth
            rows.append({
                "case_id": case_id,
                "route_id": route_id,
                "aperture_model": result.metadata["aperture_model"],
                "aperture_radius_m": result.metadata["aperture_radius_m"],
                "aperture_provenance": result.metadata["aperture_provenance"],
                "report_eligibility": result.metadata["report_eligibility"],
                "calibration_required": result.metadata["calibration_required"],
                "raw_trace_coefficient_of_variation_20_120mm": float(np.std(trace) / max(float(np.mean(trace)), EPS)),
                "raw_trace_peak_to_valley_over_mean_20_120mm": float((np.max(trace) - np.min(trace)) / max(float(np.mean(trace)), EPS)),
                "detrended_ripple_rms_fraction_25_90mm": float(np.std(residual) / max(float(np.mean(smooth)), EPS)),
                "detrended_ripple_peak_to_valley_fraction_25_90mm": float((np.max(residual) - np.min(residual)) / max(float(np.mean(smooth)), EPS)),
                "measured_FWHM_axial_zone_m": json.dumps(result.metadata["zones"]["measured_FWHM_axial_zone_m"]),
                "measured_strict_useful_region_m": json.dumps(result.metadata["zones"]["measured_strict_useful_region_m"]),
                "maximum_edge_energy_fraction": result.metadata["maximum_edge_energy_fraction"],
                "maximum_propagation_power_drift_fraction": result.metadata["maximum_propagation_power_drift_fraction"],
            })
    return rows


def _claim_rows() -> list[dict[str, Any]]:
    return [
        {
            "claim_id": "PHASE2E_FINAL_CLAIM_001",
            "previous_claim": "Broad Bessel and vortex morphology exists on the source-scale branch.",
            "previous_source_grid": 512,
            "previous_aperture_model": "historical hard 1.8 mm mask",
            "new_source_grid": 3072,
            "new_aperture_model": "nominal_no_additional_aperture",
            "remains_valid": True,
            "narrowed": False,
            "superseded": False,
            "replacement_evidence": "final_case_summary.csv and final primary figures",
            "notes": "Morphology survives the repaired source route.",
        },
        {
            "claim_id": "PHASE2E_FINAL_CLAIM_002",
            "previous_claim": "N=512 source propagation supports quantitative axial detail.",
            "previous_source_grid": 512,
            "previous_aperture_model": "historical hard 1.8 mm mask",
            "new_source_grid": 3072,
            "new_aperture_model": "nominal_no_additional_aperture",
            "remains_valid": False,
            "narrowed": False,
            "superseded": True,
            "replacement_evidence": "final_resolution_gate.csv and final_sampling_convergence.csv",
            "notes": "N=512 axial-detail claims are not authorised.",
        },
        {
            "claim_id": "PHASE2E_FINAL_CLAIM_003",
            "previous_claim": "Hard-aperture axial beading is the nominal laboratory prediction.",
            "previous_source_grid": 512,
            "previous_aperture_model": "hard 1.8 mm",
            "new_source_grid": 3072,
            "new_aperture_model": "hard_aperture_diagnostic",
            "remains_valid": False,
            "narrowed": True,
            "superseded": True,
            "replacement_evidence": "final_aperture_comparison.csv",
            "notes": "Hard truncation is diagnostic only unless a physical stop is measured at this plane.",
        },
        {
            "claim_id": "PHASE2E_FINAL_CLAIM_004",
            "previous_claim": "Vortex winding is preserved.",
            "previous_source_grid": 512,
            "previous_aperture_model": "accepted Phase 2B/2C evidence",
            "new_source_grid": 3072,
            "new_aperture_model": "nominal_no_additional_aperture",
            "remains_valid": True,
            "narrowed": False,
            "superseded": False,
            "replacement_evidence": "final_case_summary.csv; existing Phase 2B/2C winding contract remains authoritative",
            "notes": "Source-scale snapshot winding is a consistency check, not a replacement for Phase 2C focal claims.",
        },
        {
            "claim_id": "PHASE2E_FINAL_CLAIM_005",
            "previous_claim": "Exact experimental Bessel-zone length is known.",
            "previous_source_grid": 512,
            "previous_aperture_model": "historical route",
            "new_source_grid": 3072,
            "new_aperture_model": "nominal_no_additional_aperture",
            "remains_valid": False,
            "narrowed": True,
            "superseded": True,
            "replacement_evidence": "final_zone_summary.csv",
            "notes": "Simulated source-scale zones are reported; exact experimental length remains calibration-required.",
        },
        {
            "claim_id": "PHASE2E_FINAL_CLAIM_006",
            "previous_claim": "Phase 2C vector Debye focal conclusions.",
            "previous_source_grid": "Phase2C pupil quadrature",
            "previous_aperture_model": "objective entrance pupil",
            "new_source_grid": 3072,
            "new_aperture_model": "source-scale no-additional-aperture",
            "remains_valid": True,
            "narrowed": False,
            "superseded": False,
            "replacement_evidence": "none; Phase 2C remains frozen",
            "notes": "Source-scale axial propagation and objective/sample-scale focal propagation remain separate.",
        },
    ]


def _upstream_hashes() -> dict[str, str]:
    roots = (
        Path("outputs/validation/phase2a"),
        Path("outputs/validation/phase2b"),
        Path("outputs/validation/phase2c"),
        Path("outputs/figures/phase2b_visual_diagnostics"),
        Path("outputs/figures/phase2c"),
    )
    return {
        path.as_posix(): _sha256(path)
        for root in roots if root.exists()
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def build_final_figure_pack(
    *,
    output_root: Path = FIGURE_ROOT,
    validation_root: Path = VALIDATION_ROOT,
    style: FinalFigureStyle = FINAL_FIGURE_STYLE,
) -> dict[str, Any]:
    z_gate = json.loads((validation_root / "z_step_convergence.json").read_text(encoding="utf-8"))
    if z_gate.get("status") != "passed" or not z_gate.get("report_figures_authorised"):
        raise RuntimeError("final report figures remain unauthorised until the complete z gate passes")
    upstream_before = _upstream_hashes()
    for directory in (
        "00_manifest", "01_primary_propagation", "02_aperture_comparison",
        "03_sampling_convergence", "04_transverse_snapshots", "05_profiles_and_metrics",
        "06_3d_surfaces", "07_report_hero_figures",
    ):
        (output_root / directory).mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "00_manifest" / "final_figure_style.json", style.as_dict())
    manifest: list[dict[str, Any]] = []
    routes = (
        "nominal_no_additional_aperture",
        "soft_aperture_sensitivity",
        "hard_aperture_diagnostic",
    )
    all_results = {
        case_id: {route: load_final_source_result(case_id, route, output_root=validation_root) for route in routes}
        for case_id in CASE_CHARGES
    }
    nominal = {case_id: all_results[case_id][routes[0]] for case_id in CASE_CHARGES}
    for result in nominal.values():
        render_primary(result, manifest, output_root=output_root, style=style)
        render_snapshots(result, manifest, output_root=output_root, style=style)
        render_surface(result, manifest, output_root=output_root, style=style)
    for case_id in CASE_CHARGES:
        render_aperture_comparison(all_results[case_id], manifest, output_root=output_root, style=style)
    render_profiles_and_metrics(nominal, manifest, output_root=output_root, style=style)
    render_hero_family(nominal, manifest, output_root=output_root, style=style)
    render_hero_aperture(all_results, manifest, output_root=output_root, style=style)
    render_hero_surfaces(nominal, manifest, output_root=output_root, style=style)
    _, sampling_rows = render_sampling_convergence(nominal["B0"], manifest, output_root=output_root, style=style)

    case_rows, zone_rows = _case_and_zone_rows(nominal)
    aperture_rows = _aperture_rows(all_results)
    claim_rows = _claim_rows()
    _write_csv(validation_root / "final_case_summary.csv", case_rows)
    _write_csv(validation_root / "final_zone_summary.csv", zone_rows)
    _write_csv(validation_root / "final_aperture_comparison.csv", aperture_rows)
    _write_csv(validation_root / "final_claim_impact_registry.csv", claim_rows)
    _write_json(output_root / "00_manifest" / "final_figure_manifest.json", manifest)
    artifact_rows: list[dict[str, Any]] = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "final_artifact_manifest.json":
            artifact_rows.append({"path": path.as_posix(), "bytes": path.stat().st_size, "sha256": _sha256(path)})
    _write_json(output_root / "00_manifest" / "final_artifact_manifest.json", artifact_rows)
    upstream_after = _upstream_hashes()
    upstream_status = {
        "status": "unchanged" if upstream_before == upstream_after else "changed",
        "unchanged": upstream_before == upstream_after,
        "file_count": len(upstream_before),
        "before": upstream_before,
        "after": upstream_after,
    }
    _write_json(validation_root / "upstream_hash_status.json", upstream_status)
    outcome = {
        "outcome": "PHASE2E-FINAL-A",
        "report_figures_authorised": True,
        "selected_production_grid_n": 3072,
        "selected_z_step_m": 0.25e-3,
        "nominal_aperture_route": "nominal_no_additional_aperture",
        "soft_aperture_role": "assumed_soft_truncation_sensitivity",
        "hard_aperture_role": "diagnostic_only; calibration_required",
        "figure_count": len(manifest),
        "figure_manifest": (output_root / "00_manifest" / "final_figure_manifest.json").as_posix(),
        "case_summary": (validation_root / "final_case_summary.csv").as_posix(),
        "zone_summary": (validation_root / "final_zone_summary.csv").as_posix(),
        "aperture_comparison": (validation_root / "final_aperture_comparison.csv").as_posix(),
        "sampling_convergence": (validation_root / "final_sampling_convergence.csv").as_posix(),
        "claim_registry": (validation_root / "final_claim_impact_registry.csv").as_posix(),
        "upstream_hash_status": upstream_status["status"],
        "remaining_calibration_blockers": [
            "physical stop/aperture presence and radius at the reconstructed-field plane",
            "SLM phase LUT/stroke",
            "exact 4F iris centre and radius",
            "camera scale and z-stage",
            "beam and axicon centring",
        ],
    }
    _write_json(validation_root / "final_outcome_report.json", outcome)
    return outcome
