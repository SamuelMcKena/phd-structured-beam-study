"""Schema objects and null-preserving canonical calibration template."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping


CALIBRATION_SCHEMA_VERSION = "1.0"
DATA_CLASSIFICATIONS = {"laboratory_measurement", "synthetic_not_experimental", "template_unpopulated"}


@dataclass(frozen=True)
class CalibrationBundle:
    data: dict[str, Any]

    @property
    def schema_version(self) -> str:
        return str(self.data.get("schema_version", ""))

    @property
    def calibration_id(self) -> str:
        return str(self.data.get("calibration_id", ""))

    @property
    def data_classification(self) -> str:
        return str(self.data.get("data_classification", "template_unpopulated"))

    @property
    def is_synthetic(self) -> bool:
        return self.data_classification == "synthetic_not_experimental"

    def copy_data(self) -> dict[str, Any]:
        return deepcopy(self.data)


def measurement(value: Any, uncertainty: Any, source: str, unit: str, **extra: Any) -> dict[str, Any]:
    row = {"value": value, "uncertainty": uncertainty, "source": source, "unit": unit}
    row.update(extra)
    return row


def canonical_calibration_template() -> dict[str, Any]:
    """Return an editable bundle; nulls are deliberate and never auto-filled."""

    return {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "calibration_id": "lab_YYYYMMDD",
        "created_utc": None,
        "operator": "",
        "data_classification": "template_unpopulated",
        "laser": {
            "wavelength_m": measurement(1.029e-6, None, "manufacturer", "m"),
            "pulse_energy_J": measurement(None, None, "calibration_required", "J"),
            "beam_radius_on_slm_m": measurement(
                None,
                None,
                "calibration_required",
                "m",
                definition="1/e^2 intensity radius",
            ),
        },
        "slm": {
            "model": "HOLOEYE PLUTO-2.1 NIR-149",
            "phase_lut_path": None,
            "phase_stroke_rad": None,
            "phase_stroke_uncertainty_rad": None,
            "panel_orientation_verified": False,
            "calibration_date": None,
            "active_resolution": [1920, 1080],
            "pixel_pitch_m": 8e-6,
        },
        "fourier_filter": {
            "focal_length_m": measurement(0.300, None, "assumed", "m"),
            "iris_radius_m": measurement(None, None, "calibration_required", "m"),
            "plus_one_position_m": measurement(None, None, "calibration_required", "m"),
        },
        "objective": {
            "numerical_aperture": measurement(0.45, None, "manufacturer", "1"),
            "focal_length_m": measurement(None, None, "calibration_required", "m"),
            "effective_pupil_radius_m": measurement(None, None, "calibration_required", "m"),
            "pupil_fill_fraction": measurement(None, None, "calibration_required", "1"),
        },
        "relay": {
            "magnification": measurement(None, None, "calibration_required", "1"),
        },
        "axicon": {
            "base_angle_deg": measurement(2.0, None, "assumed", "deg"),
            "clear_aperture_m": measurement(None, None, "calibration_required", "m"),
            "refractive_index": measurement(1.458, None, "assumed", "1"),
        },
        "camera": {
            "pixel_pitch_m": measurement(None, None, "calibration_required", "m"),
            "magnification": measurement(None, None, "calibration_required", "1"),
            "object_plane_scale_m_per_pixel": measurement(None, None, "calibration_required", "m/pixel"),
            "rotation_deg": measurement(None, None, "calibration_required", "deg"),
            "centre_pixel": None,
        },
        "transmissions": {
            name: measurement(None, None, "calibration_required", "1")
            for name in ("slm1", "slm2", "four_f", "objective", "interface")
        },
        "material": {
            "name": "Cr:ZnSe",
            "refractive_index": measurement(2.44, None, "placeholder", "1"),
            "coating_state": "unknown",
            "surface_orientation_verified": False,
        },
        "experimental_comparison": {
            "performed": False,
            "evidence_path": None,
            "acceptance_passed": False,
        },
    }


def value_at(bundle: CalibrationBundle | Mapping[str, Any], path: str) -> Any:
    data: Mapping[str, Any] = bundle.data if isinstance(bundle, CalibrationBundle) else bundle
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    if isinstance(current, Mapping) and "value" in current:
        return current.get("value")
    return current


def uncertainty_at(bundle: CalibrationBundle | Mapping[str, Any], path: str) -> float | None:
    data: Mapping[str, Any] = bundle.data if isinstance(bundle, CalibrationBundle) else bundle
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    if isinstance(current, Mapping):
        value = current.get("uncertainty")
        return None if value is None else float(value)
    return None


def source_at(bundle: CalibrationBundle | Mapping[str, Any], path: str) -> str:
    data: Mapping[str, Any] = bundle.data if isinstance(bundle, CalibrationBundle) else bundle
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return "missing"
        current = current[part]
    if isinstance(current, Mapping):
        return str(current.get("source", "unspecified"))
    if current is None:
        return "missing"
    return "supplied"
