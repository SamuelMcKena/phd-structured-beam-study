"""Data contracts for the optional OpticStudio cross-validation backend."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np


@dataclass(frozen=True)
class ZemaxAvailability:
    platform: str
    python_version: str
    python_executable: str
    zospy_installed: bool
    zospy_version: str | None = None
    pythonnet_loadable: bool = False
    opticstudio_detected: bool = False
    zosapi_loadable: bool = False
    opticstudio_version: str | None = None
    license_available: bool = False
    standalone_connection_available: bool = False
    interactive_connection_available: bool = False
    diagnostic_messages: tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        return bool(self.zospy_installed and self.zosapi_loadable and self.license_available and self.standalone_connection_available)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ZemaxModelMetadata:
    source_path: str
    source_sha256: str
    zemax_filename: str
    sequential_or_nonsequential: str
    system_units: str
    wavelength_table: tuple[dict[str, Any], ...] = ()
    field_table: tuple[dict[str, Any], ...] = ()
    surface_count: int = 0
    surface_summary: tuple[dict[str, Any], ...] = ()
    aperture_summary: tuple[dict[str, Any], ...] = ()
    model_modified: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PopRequest:
    model_path: str | None
    wavelength_index: int = 1
    field_index: int = 1
    start_surface: int | str = 1
    end_surface: int | str = "Image"
    x_sampling: int | str = 128
    y_sampling: int | str = 128
    x_width_m: float = 5e-3
    y_width_m: float = 5e-3
    polarization: bool = False
    beam_type: str = "GaussianWaist"
    beam_file: str = ""
    beam_parameters_m: dict[str, float] | None = None
    save_output_beam: bool = False
    output_beam_file: str = ""
    save_beam_at_all_surfaces: bool = False
    auto_calculate_beam_sampling: bool = False
    use_total_power: bool = True
    total_power: float = 1.0
    use_peak_irradiance: bool = False
    peak_irradiance: float = 1.0
    data_type: str = "Irradiance"

    def validate(self) -> None:
        if int(self.wavelength_index) < 1 or int(self.field_index) < 1:
            raise ValueError("wavelength_index and field_index are 1-based positive integers")
        if self.x_width_m <= 0 or self.y_width_m <= 0:
            raise ValueError("POP physical widths must be positive")
        if self.total_power < 0 or self.peak_irradiance < 0:
            raise ValueError("power and irradiance settings cannot be negative")
        if self.beam_type.lower() == "file" and not self.beam_file:
            raise ValueError("beam_type='File' requires beam_file")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PopResult:
    x_m: np.ndarray
    y_m: np.ndarray
    irradiance: np.ndarray
    phase_if_available: np.ndarray | None = None
    Ex_if_available: np.ndarray | None = None
    Ey_if_available: np.ndarray | None = None
    total_power_if_available: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.x_m = np.asarray(self.x_m, dtype=float)
        self.y_m = np.asarray(self.y_m, dtype=float)
        self.irradiance = np.asarray(self.irradiance, dtype=float)
        if self.irradiance.shape != (self.y_m.size, self.x_m.size):
            raise ValueError("POP irradiance shape must match y/x axes")
        if not np.all(np.isfinite(self.irradiance)) or np.any(self.irradiance < 0):
            raise ValueError("POP irradiance must be finite and non-negative")


@dataclass
class FieldPlane:
    x_m: np.ndarray
    y_m: np.ndarray
    wavelength_m: float
    Ex: np.ndarray | None = None
    Ey: np.ndarray | None = None
    Ez: np.ndarray | None = None
    intensity: np.ndarray | None = None
    phase: np.ndarray | None = None
    plane_name: str = "unknown"
    source_solver: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.x_m = np.asarray(self.x_m, dtype=float)
        self.y_m = np.asarray(self.y_m, dtype=float)
        if self.x_m.ndim != 1 or self.y_m.ndim != 1 or self.x_m.size < 2 or self.y_m.size < 2:
            raise ValueError("FieldPlane axes must be one-dimensional with at least two samples")
        if not np.all(np.diff(self.x_m) > 0) or not np.all(np.diff(self.y_m) > 0):
            raise ValueError("FieldPlane axes must be strictly increasing")
        if self.wavelength_m <= 0:
            raise ValueError("wavelength_m must be positive")
        shape = (self.y_m.size, self.x_m.size)
        for name in ("Ex", "Ey", "Ez"):
            value = getattr(self, name)
            if value is not None:
                arr = np.asarray(value, dtype=np.complex128)
                if arr.shape != shape:
                    raise ValueError(f"{name} shape must match y/x axes")
                setattr(self, name, arr)
        if self.intensity is not None:
            self.intensity = np.asarray(self.intensity, dtype=float)
            if self.intensity.shape != shape:
                raise ValueError("intensity shape must match y/x axes")
        if self.phase is not None:
            self.phase = np.asarray(self.phase, dtype=float)
            if self.phase.shape != shape:
                raise ValueError("phase shape must match y/x axes")

    @property
    def transverse_intensity(self) -> np.ndarray:
        if self.Ex is not None or self.Ey is not None:
            ex = 0.0 if self.Ex is None else np.abs(self.Ex) ** 2
            ey = 0.0 if self.Ey is None else np.abs(self.Ey) ** 2
            return np.asarray(ex + ey, dtype=float)
        if self.intensity is None:
            raise ValueError("no transverse field or irradiance is available")
        return np.asarray(self.intensity, dtype=float)

    @property
    def total_intensity(self) -> np.ndarray:
        if self.Ex is not None or self.Ey is not None or self.Ez is not None:
            total = np.zeros((self.y_m.size, self.x_m.size), dtype=float)
            for component in (self.Ex, self.Ey, self.Ez):
                if component is not None:
                    total += np.abs(component) ** 2
            return total
        if self.intensity is None:
            raise ValueError("no field or irradiance is available")
        return np.asarray(self.intensity, dtype=float)

    def copy_with(self, **changes: Any) -> "FieldPlane":
        values = {"x_m": self.x_m, "y_m": self.y_m, "wavelength_m": self.wavelength_m, "Ex": self.Ex, "Ey": self.Ey, "Ez": self.Ez, "intensity": self.intensity, "phase": self.phase, "plane_name": self.plane_name, "source_solver": self.source_solver, "metadata": dict(self.metadata)}
        values.update(changes)
        return FieldPlane(**values)


@dataclass
class CrossValidationResult:
    python_plane: FieldPlane
    zemax_plane: FieldPlane
    aligned_python_plane: FieldPlane
    aligned_zemax_plane: FieldPlane
    metrics: dict[str, Any]
    pass_fail_flags: dict[str, bool | str]
    provenance: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


ValidationBackend = Literal["none", "zemax_pop"]


def model_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()
