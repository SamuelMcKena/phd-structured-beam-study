"""Measured-objective pupil layer around the established vector Debye solver.

The accepted Debye implementation remains untouched.  This module only modifies
its *input pupil field* using explicitly supplied calibration maps before calling
``debye_focus_plane``.  No map is resized, guessed, or fitted automatically.

Transmission maps are amplitude transmission, not intensity transmission.  OPD
maps are metres of optical path difference and therefore produce wavelength-
dependent phase ``exp(i 2*pi*OPD/lambda)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from vbb_study.equations.vector_debye import DebyeConfig, VectorFieldPlane, debye_focus_plane


@dataclass(frozen=True)
class ObjectivePupilCalibration:
    amplitude_transmission: np.ndarray | None = None
    opd_map_m: np.ndarray | None = None
    valid_mask: np.ndarray | None = None
    source: str = "user_supplied"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self, shape: tuple[int, int]) -> None:
        for values, name in (
            (self.amplitude_transmission, "amplitude_transmission"),
            (self.opd_map_m, "opd_map_m"),
            (self.valid_mask, "valid_mask"),
        ):
            if values is None:
                continue
            arr = np.asarray(values)
            if arr.shape != shape:
                raise ValueError(f"{name} must exactly match the pupil field shape")
            if name != "valid_mask" and np.any(~np.isfinite(arr.astype(float))):
                raise ValueError(f"{name} contains non-finite values")
        if self.amplitude_transmission is not None:
            t = np.asarray(self.amplitude_transmission, dtype=float)
            if np.any(t < 0.0):
                raise ValueError("amplitude_transmission cannot be negative")


def apply_objective_pupil_calibration(
    Ex_pupil: np.ndarray,
    Ey_pupil: np.ndarray,
    *,
    wavelength_m: float,
    calibration: ObjectivePupilCalibration,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply measured amplitude/OPD/mask data to a transverse pupil field."""

    ex = np.asarray(Ex_pupil, dtype=np.complex128)
    ey = np.asarray(Ey_pupil, dtype=np.complex128)
    if ex.ndim != 2 or ex.shape != ey.shape:
        raise ValueError("Ex_pupil and Ey_pupil must be same-shaped 2-D arrays")
    if not np.isfinite(wavelength_m) or wavelength_m <= 0.0:
        raise ValueError("wavelength_m must be finite and positive")
    calibration.validate(ex.shape)

    transfer = np.ones(ex.shape, dtype=np.complex128)
    applied: list[str] = []
    if calibration.amplitude_transmission is not None:
        transfer *= np.asarray(calibration.amplitude_transmission, dtype=float)
        applied.append("measured_amplitude_transmission")
    if calibration.opd_map_m is not None:
        opd = np.asarray(calibration.opd_map_m, dtype=float)
        transfer *= np.exp(1j * (2.0 * np.pi / float(wavelength_m)) * opd)
        applied.append("measured_or_supplied_opd")
    if calibration.valid_mask is not None:
        transfer *= np.asarray(calibration.valid_mask, dtype=bool)
        applied.append("measured_valid_pupil_mask")

    input_power = float(np.sum(np.abs(ex) ** 2 + np.abs(ey) ** 2))
    out_ex = ex * transfer
    out_ey = ey * transfer
    output_power = float(np.sum(np.abs(out_ex) ** 2 + np.abs(out_ey) ** 2))
    metadata = {
        "source": calibration.source,
        "applied_calibrations": applied,
        "pupil_transfer_applied": bool(applied),
        "relative_transverse_power_after_pupil": output_power / max(input_power, np.finfo(float).tiny),
        "automatic_map_registration": False,
        "automatic_map_resizing": False,
        **dict(calibration.metadata),
    }
    return out_ex, out_ey, metadata


def calibrated_debye_focus_plane(
    Ex_pupil: np.ndarray,
    Ey_pupil: np.ndarray,
    pupil_x_m: np.ndarray,
    pupil_y_m: np.ndarray,
    output_x_m: np.ndarray,
    output_y_m: np.ndarray,
    z_m: float,
    config: DebyeConfig,
    *,
    pupil_calibration: ObjectivePupilCalibration | None = None,
) -> VectorFieldPlane:
    """Call the canonical Debye solver after optional measured-pupil transfer."""

    if pupil_calibration is None:
        ex = np.asarray(Ex_pupil, dtype=np.complex128)
        ey = np.asarray(Ey_pupil, dtype=np.complex128)
        pupil_meta = {
            "pupil_transfer_applied": False,
            "applied_calibrations": [],
            "source": "ideal_objective_no_measured_pupil_transfer",
        }
    else:
        ex, ey, pupil_meta = apply_objective_pupil_calibration(
            Ex_pupil,
            Ey_pupil,
            wavelength_m=config.wavelength_m,
            calibration=pupil_calibration,
        )

    result = debye_focus_plane(
        ex,
        ey,
        pupil_x_m,
        pupil_y_m,
        output_x_m,
        output_y_m,
        z_m,
        config,
    )
    result.metadata = {
        **dict(result.metadata),
        "objective_pupil_calibration": pupil_meta,
        "objective_model": (
            "measured_pupil_plus_aplanatic_vector_debye"
            if pupil_calibration is not None
            else "ideal_aplanatic_vector_debye"
        ),
    }
    return result


__all__ = [
    "ObjectivePupilCalibration",
    "apply_objective_pupil_calibration",
    "calibrated_debye_focus_plane",
]
