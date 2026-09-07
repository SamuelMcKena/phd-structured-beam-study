from __future__ import annotations

"""Measured intensity-stack contract and manifest-driven ingestion.

The loader is intentionally strict about physical coordinates.  Pixel pitch,
array-centre position and z reference are measurement/calibration inputs; they
are never guessed from a camera model or from the simulated beam.

No per-plane centring, peak normalisation, background subtraction or clipping is
performed by default.  Those operations can erase exactly the displacement,
throughput and background signatures needed for physical error inference.
"""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping
import json

import numpy as np
from PIL import Image


SUPPORTED_PLANE_SUFFIXES = {".npy", ".npz", ".txt", ".csv", ".bmp", ".png", ".tif", ".tiff"}


@dataclass(frozen=True)
class IntensityStack:
    z_m: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    intensity: np.ndarray
    metadata: Mapping[str, Any]
    valid_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        z = np.asarray(self.z_m, dtype=float)
        x = np.asarray(self.x_m, dtype=float)
        y = np.asarray(self.y_m, dtype=float)
        values = np.asarray(self.intensity, dtype=float)
        if z.ndim != 1 or x.ndim != 1 or y.ndim != 1:
            raise ValueError("z_m, x_m and y_m must be one-dimensional")
        if z.size < 1 or x.size < 2 or y.size < 2:
            raise ValueError("intensity stack needs >=1 z plane and >=2 samples per transverse axis")
        if values.shape != (z.size, y.size, x.size):
            raise ValueError(
                "intensity shape must be (nz, ny, nx); "
                f"got {values.shape}, expected {(z.size, y.size, x.size)}"
            )
        if not np.all(np.isfinite(z)) or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("physical coordinates must be finite")
        if not np.all(np.diff(z) > 0.0):
            raise ValueError("z_m must be strictly increasing")
        if not np.all(np.diff(x) > 0.0) or not np.all(np.diff(y) > 0.0):
            raise ValueError("x_m and y_m must be strictly increasing")
        if self.valid_mask is not None:
            mask = np.asarray(self.valid_mask, dtype=bool)
            if mask.shape not in {values.shape, values.shape[1:]}:
                raise ValueError("valid_mask must be (nz, ny, nx) or (ny, nx)")

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(np.asarray(self.intensity).shape)  # type: ignore[return-value]


def _load_numeric_plane(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_PLANE_SUFFIXES:
        raise ValueError(f"unsupported intensity-plane format {suffix!r}: {path}")
    if suffix == ".npy":
        arr = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            if "intensity" in data.files:
                arr = data["intensity"]
            elif len(data.files) == 1:
                arr = data[data.files[0]]
            else:
                raise ValueError(f"NPZ {path} must contain 'intensity' or exactly one array")
    elif suffix == ".txt":
        arr = np.loadtxt(path)
    elif suffix == ".csv":
        arr = np.loadtxt(path, delimiter=",")
    else:
        with Image.open(path) as image:
            arr = np.asarray(image)
        if arr.ndim == 3:
            raise ValueError(
                f"colour image {path} is ambiguous as radiometric intensity; export a scalar camera plane"
            )
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"intensity plane must be 2-D, got {arr.shape} from {path}")
    return arr


def _pair(value: Any, *, name: str, positive: bool = False) -> tuple[float, float]:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        pair = (float(arr), float(arr))
    elif arr.shape == (2,):
        pair = (float(arr[0]), float(arr[1]))
    else:
        raise ValueError(f"{name} must be a scalar or [x, y] pair")
    if positive and (pair[0] <= 0.0 or pair[1] <= 0.0):
        raise ValueError(f"{name} must be positive")
    return pair


def _orient(array: np.ndarray, orientation: Mapping[str, Any]) -> np.ndarray:
    result = np.asarray(array)
    if bool(orientation.get("transpose", False)):
        result = result.T
    if bool(orientation.get("flip_x", False)):
        result = result[:, ::-1]
    if bool(orientation.get("flip_y", False)):
        result = result[::-1, :]
    return np.asarray(result, dtype=float)


def load_experimental_stack(manifest_path: str | Path) -> IntensityStack:
    """Load a measured stack from an explicit physical-coordinate manifest.

    Required manifest fields::

        {
          "schema": "vbb-experimental-intensity-stack-v1",
          "case_id": "V3",
          "z_reference": "post_axicon_plane",
          "coordinate_calibration": {
            "pixel_pitch_m": [dx, dy],
            "array_centre_m": [x0, y0],
            "orientation": {"transpose": false, "flip_x": false, "flip_y": false}
          },
          "planes": [
            {"z_m": 0.040, "path": "z040.npy"},
            {"z_m": 0.060, "path": "z060.npy"}
          ]
        }

    ``z_reference`` may use another laboratory origin, but the matcher then
    requires an explicit model-z offset.  This makes the coordinate ambiguity a
    visible nuisance parameter rather than an invisible recentering step.
    """
    manifest_file = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("schema") != "vbb-experimental-intensity-stack-v1":
        raise ValueError("manifest schema must be 'vbb-experimental-intensity-stack-v1'")
    if not manifest.get("z_reference"):
        raise ValueError("manifest must declare z_reference")
    calibration = manifest.get("coordinate_calibration")
    if not isinstance(calibration, Mapping):
        raise ValueError("manifest must declare coordinate_calibration")
    if "pixel_pitch_m" not in calibration or "array_centre_m" not in calibration:
        raise ValueError("coordinate_calibration requires pixel_pitch_m and array_centre_m")
    dx, dy = _pair(calibration["pixel_pitch_m"], name="pixel_pitch_m", positive=True)
    x0, y0 = _pair(calibration["array_centre_m"], name="array_centre_m")
    orientation = calibration.get("orientation", {})
    if not isinstance(orientation, Mapping):
        raise ValueError("coordinate_calibration.orientation must be an object")

    planes = manifest.get("planes")
    if not isinstance(planes, list) or not planes:
        raise ValueError("manifest planes must be a non-empty list")

    loaded: list[tuple[float, np.ndarray, str]] = []
    for record in planes:
        if not isinstance(record, Mapping) or "z_m" not in record or "path" not in record:
            raise ValueError("every plane needs z_m and path")
        z = float(record["z_m"])
        path = (manifest_file.parent / str(record["path"])).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        loaded.append((z, _orient(_load_numeric_plane(path), orientation), str(record["path"])))
    loaded.sort(key=lambda item: item[0])

    shape = loaded[0][1].shape
    if any(item[1].shape != shape for item in loaded):
        raise ValueError("all measured intensity planes must have the same oriented shape")
    ny, nx = shape
    x = (np.arange(nx, dtype=float) - 0.5 * (nx - 1)) * dx + x0
    y = (np.arange(ny, dtype=float) - 0.5 * (ny - 1)) * dy + y0
    values = np.stack([item[1] for item in loaded], axis=0)

    valid = np.isfinite(values)
    metadata = {
        "source": "experimental_manifest",
        "manifest_path": str(manifest_file),
        "schema": manifest["schema"],
        "case_id": manifest.get("case_id"),
        "z_reference": str(manifest["z_reference"]),
        "pixel_pitch_m": [dx, dy],
        "array_centre_m": [x0, y0],
        "orientation": dict(orientation),
        "plane_paths": [item[2] for item in loaded],
        "input_dtype_policy": "loaded_as_float64_without_radiometric_rescaling",
        "preprocessing": [],
        "user_metadata": manifest.get("metadata", {}),
    }
    return IntensityStack(
        z_m=np.asarray([item[0] for item in loaded], dtype=float),
        x_m=x,
        y_m=y,
        intensity=values,
        valid_mask=valid,
        metadata=metadata,
    )


def preprocess_stack(
    stack: IntensityStack,
    *,
    dark_frame: np.ndarray | None = None,
    scalar_background: float | None = None,
    clip_negative: bool = False,
) -> IntensityStack:
    """Apply only explicitly requested radiometric preprocessing.

    A 2-D dark frame is broadcast to all z planes.  A 3-D dark stack must match
    the measurement stack exactly.  No normalisation or spatial registration is
    offered here by design.
    """
    values = np.asarray(stack.intensity, dtype=float).copy()
    steps = list(stack.metadata.get("preprocessing", []))
    if dark_frame is not None:
        dark = np.asarray(dark_frame, dtype=float)
        if dark.shape == values.shape[1:]:
            values -= dark[None, :, :]
        elif dark.shape == values.shape:
            values -= dark
        else:
            raise ValueError("dark_frame must match (ny,nx) or full stack shape")
        steps.append("explicit_dark_frame_subtraction")
    if scalar_background is not None:
        values -= float(scalar_background)
        steps.append(f"explicit_scalar_background_subtraction:{float(scalar_background):.12g}")
    if clip_negative:
        values = np.maximum(values, 0.0)
        steps.append("explicit_clip_negative_to_zero")

    metadata = dict(stack.metadata)
    metadata["preprocessing"] = steps
    return replace(stack, intensity=values, metadata=metadata)


__all__ = [
    "IntensityStack",
    "SUPPORTED_PLANE_SUFFIXES",
    "load_experimental_stack",
    "preprocess_stack",
]
