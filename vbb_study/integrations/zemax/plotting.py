"""Publication-clean Phase 3 comparison figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .models import CrossValidationResult


def save_cross_validation_figures(result: CrossValidationResult, output_directory: str | Path) -> list[Path]:
    out = Path(output_directory)
    out.mkdir(parents=True, exist_ok=True)
    py = result.aligned_python_plane
    ze = result.aligned_zemax_plane
    py_i = py.transverse_intensity
    ze_i = ze.transverse_intensity
    peak = max(float(np.max(py_i)), float(np.max(ze_i)), np.finfo(float).tiny)
    extent = [py.x_m[0] * 1e3, py.x_m[-1] * 1e3, py.y_m[0] * 1e3, py.y_m[-1] * 1e3]
    files: list[Path] = []
    for number, name, image, title in ((1, "python_intensity", py_i / peak, "Python transverse intensity"), (2, "zemax_intensity", ze_i / peak, "Zemax POP transverse irradiance"), (3, "intensity_difference", (ze_i - py_i) / peak, "Zemax - Python (shared peak scale)")):
        path = out / f"{number:02d}_{name}.png"
        fig, ax = plt.subplots(figsize=(6.4, 5.2), constrained_layout=True)
        im = ax.imshow(image, origin="lower", extent=extent, aspect="equal")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
        fig.savefig(path, dpi=220)
        plt.close(fig)
        files.append(path)
    return files
