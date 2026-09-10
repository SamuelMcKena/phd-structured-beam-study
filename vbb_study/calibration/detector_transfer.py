"""Measured detector-response operators for simulation-to-camera prediction.

This module keeps three detector stages separate:

1. ``optical_response_intensity``: optical intensity after measured PSF and
   relative-response effects, before additive background or saturation. This is
   the correct simulated quantity to compare against a background-subtracted
   camera image while masking saturated measured pixels.
2. ``raw_unsaturated_intensity``: predicted raw detector signal after supplied
   additive background but before saturation.
3. ``intensity``: final predicted raw detector signal after explicit saturation.

Nothing is inferred by optimising against the target image.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from scipy.signal import fftconvolve


@dataclass(frozen=True)
class DetectorTransferCalibration:
    psf: np.ndarray | None = None
    relative_response: np.ndarray | None = None
    additive_background: np.ndarray | float | None = None
    saturation_level: float | None = None
    source: str = "user_supplied"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self, shape: tuple[int, int]) -> None:
        if self.psf is not None:
            psf = np.asarray(self.psf, dtype=float)
            if psf.ndim != 2 or psf.size == 0 or np.any(~np.isfinite(psf)) or np.any(psf < 0.0):
                raise ValueError("detector PSF must be a finite non-negative 2-D array")
            if float(np.sum(psf)) <= 0.0:
                raise ValueError("detector PSF must have non-zero integral")
        if self.relative_response is not None:
            response = np.asarray(self.relative_response, dtype=float)
            if response.shape != shape or np.any(~np.isfinite(response)) or np.any(response < 0.0):
                raise ValueError("relative_response must match the simulation plane and be non-negative")
        if isinstance(self.additive_background, np.ndarray):
            bg = np.asarray(self.additive_background, dtype=float)
            if bg.shape != shape or np.any(~np.isfinite(bg)):
                raise ValueError("additive background array must match the simulation plane")
        elif self.additive_background is not None and not np.isfinite(float(self.additive_background)):
            raise ValueError("additive background scalar must be finite")
        if self.saturation_level is not None and (
            not np.isfinite(self.saturation_level) or self.saturation_level <= 0.0
        ):
            raise ValueError("saturation_level must be finite and positive")


@dataclass(frozen=True)
class DetectorTransferResult:
    intensity: np.ndarray
    optical_response_intensity: np.ndarray
    raw_unsaturated_intensity: np.ndarray
    saturated_mask: np.ndarray
    metadata: Mapping[str, Any]

    @property
    def unsaturated_intensity(self) -> np.ndarray:
        """Backward-compatible alias for the raw pre-saturation detector signal."""

        return self.raw_unsaturated_intensity


def apply_detector_transfer(
    optical_intensity: np.ndarray,
    calibration: DetectorTransferCalibration,
) -> DetectorTransferResult:
    """Apply measured detector response without fitting to the observed image."""

    intensity = np.asarray(optical_intensity, dtype=float)
    if intensity.ndim != 2 or intensity.size == 0 or np.any(~np.isfinite(intensity)) or np.any(intensity < 0.0):
        raise ValueError("optical_intensity must be a finite non-negative 2-D array")
    calibration.validate(intensity.shape)
    out = intensity.copy()
    applied: list[str] = []

    if calibration.psf is not None:
        psf = np.asarray(calibration.psf, dtype=float)
        psf = psf / float(np.sum(psf))
        out = fftconvolve(out, psf, mode="same")
        out = np.maximum(out, 0.0)
        applied.append("measured_psf_convolution")

    if calibration.relative_response is not None:
        out *= np.asarray(calibration.relative_response, dtype=float)
        applied.append("measured_relative_response_map")

    optical_response = np.asarray(out, dtype=float).copy()

    if calibration.additive_background is not None:
        out = out + np.asarray(calibration.additive_background, dtype=float)
        applied.append("supplied_additive_background")

    raw_unsaturated = np.asarray(out, dtype=float).copy()
    if calibration.saturation_level is None:
        saturated = np.zeros_like(out, dtype=bool)
    else:
        threshold = float(calibration.saturation_level)
        saturated = out >= threshold
        out = np.minimum(out, threshold)
        applied.append("explicit_saturation_clip")

    return DetectorTransferResult(
        intensity=np.asarray(out, dtype=float),
        optical_response_intensity=optical_response,
        raw_unsaturated_intensity=raw_unsaturated,
        saturated_mask=saturated,
        metadata={
            "source": calibration.source,
            "applied_calibrations": applied,
            "automatic_fit_to_measurement": False,
            "comparison_stage": "optical_response_intensity_before_background_and_saturation",
            "raw_prediction_stage": "intensity_after_background_and_saturation",
            "linear_detector_model_before_saturation": True,
            "saturated_fraction": float(np.mean(saturated)),
            **dict(calibration.metadata),
        },
    )


__all__ = [
    "DetectorTransferCalibration",
    "DetectorTransferResult",
    "apply_detector_transfer",
]
