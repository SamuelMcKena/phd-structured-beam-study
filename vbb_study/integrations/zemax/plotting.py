"""Publication-clean Phase 3 comparison figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .comparison import radial_profile
from .models import CrossValidationResult

EPS = np.finfo(float).tiny


def _save_image(path: Path, image: np.ndarray, extent: list[float], title: str, cbar: str) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.2), constrained_layout=True)
    im = ax.imshow(image, origin="lower", extent=extent, aspect="equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=cbar)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_cross_validation_figures(result: CrossValidationResult, output_directory: str | Path) -> list[Path]:
    out = Path(output_directory)
    out.mkdir(parents=True, exist_ok=True)
    py = result.aligned_python_plane
    ze = result.aligned_zemax_plane
    py_i = py.transverse_intensity
    ze_i = ze.transverse_intensity
    scale = max(float(np.max(py_i)), float(np.max(ze_i)), EPS)
    py_n, ze_n = py_i / scale, ze_i / scale
    extent = [float(py.x_m[0] * 1e3), float(py.x_m[-1] * 1e3), float(py.y_m[0] * 1e3), float(py.y_m[-1] * 1e3)]
    files: list[Path] = []
    for number, name, image, title, cbar in ((1, "python_intensity", py_n, "Python transverse intensity", "shared-normalised intensity"), (2, "zemax_intensity", ze_n, "Zemax POP transverse irradiance", "shared-normalised irradiance"), (3, "intensity_difference", ze_n - py_n, "Zemax - Python", "shared-normalised difference")):
        path = out / f"{number:02d}_{name}.png"; _save_image(path, image, extent, title, cbar); files.append(path)
    r_py, p_py = radial_profile(py_i, py.x_m, py.y_m); r_ze, p_ze = radial_profile(ze_i, ze.x_m, ze.y_m)
    path = out / "04_radial_profiles.png"; fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True); ax.plot(r_py * 1e3, p_py / max(float(np.max(p_py)), EPS), label="Python"); ax.plot(r_ze * 1e3, p_ze / max(float(np.max(p_ze)), EPS), label="Zemax POP"); ax.set_xlabel("radius (mm)"); ax.set_ylabel("peak-normalised radial intensity"); ax.legend(); fig.savefig(path, dpi=220); plt.close(fig); files.append(path)
    cy, cx = py_i.shape[0] // 2, py_i.shape[1] // 2
    for number, name, axis_mm, p_line, z_line, xlabel in ((5, "x_profile", py.x_m * 1e3, py_i[cy, :], ze_i[cy, :], "x (mm)"), (6, "y_profile", py.y_m * 1e3, py_i[:, cx], ze_i[:, cx], "y (mm)")):
        path = out / f"{number:02d}_{name}.png"; fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True); ax.plot(axis_mm, p_line / max(float(np.max(p_line)), EPS), label="Python"); ax.plot(axis_mm, z_line / max(float(np.max(z_line)), EPS), label="Zemax POP"); ax.set_xlabel(xlabel); ax.set_ylabel("peak-normalised intensity"); ax.legend(); fig.savefig(path, dpi=220); plt.close(fig); files.append(path)
    if py.phase is not None and ze.phase is not None:
        for number, name, image, title in ((7, "python_phase", py.phase, "Python phase"), (8, "zemax_phase", ze.phase, "Zemax phase"), (9, "phase_difference", np.angle(np.exp(1j * (ze.phase - py.phase))), "Wrapped phase difference")):
            path = out / f"{number:02d}_{name}.png"; _save_image(path, image, extent, title, "phase (rad)"); files.append(path)
    if "complex_transverse_field_overlap" in result.metrics:
        path = out / "10_complex_overlap_summary.png"; fig, ax = plt.subplots(figsize=(6.4, 3.2), constrained_layout=True); ax.axis("off"); ax.text(0.03, 0.70, "Complex transverse-field overlap", fontsize=14, weight="bold"); ax.text(0.03, 0.42, f"η = {float(result.metrics['complex_transverse_field_overlap']):.6f}", fontsize=16); ax.text(0.03, 0.15, "Ex and Ey only; Ez is not validated through transverse ZBF exchange.", fontsize=9); fig.savefig(path, dpi=220); plt.close(fig); files.append(path)
    path = out / "11_cross_validation_summary.png"; fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.4), constrained_layout=True)
    for ax, image, title in zip(axes, (py_n, ze_n, ze_n - py_n), ("Python", "Zemax POP", "Difference"), strict=True):
        im = ax.imshow(image, origin="lower", extent=extent, aspect="equal"); ax.set_title(title); ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)"); fig.colorbar(im, ax=ax, shrink=0.82)
    corr = result.metrics.get("normalized_intensity_correlation", float("nan")); fig.suptitle(f"Phase 3 independent POP comparison — intensity correlation {float(corr):.5f}\nNo experimental or Ez-validation claim"); fig.savefig(path, dpi=220); plt.close(fig); files.append(path)
    return files
