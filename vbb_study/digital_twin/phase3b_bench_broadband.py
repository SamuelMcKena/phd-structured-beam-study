"""Phase 3B bench-calibrated broadband digital-twin orchestration.

This layer binds measured calibration assets to the new broadband framework
without modifying the frozen Phase 1-2C evidence routes.  It is intentionally
strict: a measured-spectrum run is blocked when the spectrum is absent, an
unknown material never silently inherits fused-silica dispersion, and pupil or
detector maps are never resized to make them fit.

The actual one-wavelength optical calculation is injected as a callback so the
existing canonical scalar/vector/4F/Debye solvers remain the source of optical
physics.  Phase 3B provides wavelength-dependent material context and measured
assets to that callback, then performs spectral integration and detector
transfer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from vbb_study.calibration.detector_transfer import (
    DetectorTransferCalibration,
    DetectorTransferResult,
    apply_detector_transfer,
)
from vbb_study.calibration.schema import CalibrationBundle, value_at
from vbb_study.digital_twin.broadband_propagation import (
    BroadbandResult,
    SpectralFieldPlane,
    Spectrum,
    apply_polynomial_spectral_phase,
    gaussian_transform_limited_spectrum,
    load_spectrum_csv,
    propagate_broadband,
)
from vbb_study.equations.calibrated_objective import ObjectivePupilCalibration
from vbb_study.equations.dispersion import ConstantIndexMaterial, FUSED_SILICA_MALITSON, SellmeierMaterial


MaterialModel = SellmeierMaterial | ConstantIndexMaterial


@dataclass(frozen=True)
class Phase3BAssetPaths:
    objective_amplitude_transmission_path: str | None = None
    objective_opd_map_m_path: str | None = None
    objective_valid_mask_path: str | None = None
    detector_psf_path: str | None = None
    detector_relative_response_path: str | None = None
    detector_background_path: str | None = None


@dataclass(frozen=True)
class Phase3BConfig:
    require_measured_spectrum: bool = True
    allow_transform_limited_control: bool = False
    transform_limited_samples: int = 21
    apply_bundle_gdd_tod: bool = True
    sample_material_name: str | None = None
    axicon_material_name: str | None = None
    assets: Phase3BAssetPaths = field(default_factory=Phase3BAssetPaths)


@dataclass(frozen=True)
class Phase3BWavelengthContext:
    wavelength_m: float
    sample_refractive_index: float | None
    axicon_refractive_index: float | None
    objective_pupil_calibration: ObjectivePupilCalibration | None
    calibration_id: str
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class Phase3BResult:
    broadband: BroadbandResult
    detector_transfer: DetectorTransferResult | None
    spectrum_status: str
    sample_material_status: str
    axicon_material_status: str
    blockers: tuple[str, ...]
    metadata: Mapping[str, Any]


def _load_numeric_array(path: str | None, *, key: str | None = None) -> np.ndarray | None:
    if path in (None, ""):
        return None
    source = Path(str(path))
    if not source.is_file():
        raise FileNotFoundError(source)
    suffix = source.suffix.lower()
    if suffix == ".npy":
        values = np.load(source, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(source, allow_pickle=False) as data:
            if key is not None:
                if key not in data:
                    raise KeyError(f"{source} has no array {key!r}")
                values = data[key]
            elif len(data.files) == 1:
                values = data[data.files[0]]
            else:
                raise ValueError(f"{source} contains multiple arrays; an explicit key is required")
    elif suffix in {".csv", ".txt"}:
        values = np.loadtxt(source, delimiter="," if suffix == ".csv" else None)
    else:
        raise ValueError(f"unsupported calibration-array format: {suffix}")
    array = np.asarray(values)
    if array.size == 0 or np.any(~np.isfinite(array.astype(float))):
        raise ValueError(f"calibration array {source} is empty or non-finite")
    return array


def _normalise_material_name(name: str | None) -> str:
    return "" if name is None else "".join(ch for ch in str(name).lower() if ch.isalnum())


def material_model_from_calibration(
    bundle: CalibrationBundle,
    *,
    role: str,
    explicit_name: str | None = None,
) -> tuple[MaterialModel | None, str]:
    """Resolve a material without silently assigning a glass type.

    Fused silica gets the named Malitson dispersion law.  Any other material
    may use an explicitly measured constant index as a temporary monochromatic
    model, but is labelled ``constant_index_no_dispersion``.  Unknown material
    with no index is blocked.
    """

    if role not in {"sample", "axicon"}:
        raise ValueError("role must be 'sample' or 'axicon'")
    if role == "sample":
        stored_name = bundle.data.get("material", {}).get("name")
        index_path = "material.refractive_index"
    else:
        axicon = bundle.data.get("axicon", {})
        stored_name = axicon.get("material_name") or axicon.get("glass")
        index_path = "axicon.refractive_index"
    name = explicit_name or stored_name
    normalised = _normalise_material_name(name)
    if normalised in {"fusedsilica", "silica", "sio2", "fusedsilicamalitson"}:
        return FUSED_SILICA_MALITSON, "sellmeier_fused_silica"
    index = value_at(bundle, index_path)
    if index not in (None, ""):
        label = str(name or f"{role}_unknown_glass")
        return ConstantIndexMaterial(label, float(index), source=f"calibration:{index_path}"), "constant_index_no_dispersion"
    return None, "blocked_unknown_material_or_index"


def _load_separate_spectral_phase(path: str, target_wavelengths_m: np.ndarray) -> tuple[np.ndarray, str]:
    table = np.genfromtxt(Path(path), delimiter=",", names=True, dtype=float, encoding="utf-8-sig")
    if table.size == 0 or table.dtype.names is None:
        raise ValueError("spectral phase CSV is empty or lacks a header")
    names = set(table.dtype.names)
    if "wavelength_m" in names:
        wl = np.asarray(table["wavelength_m"], dtype=float)
    elif "wavelength_nm" in names:
        wl = np.asarray(table["wavelength_nm"], dtype=float) * 1.0e-9
    else:
        raise ValueError("spectral phase CSV requires wavelength_m or wavelength_nm")
    if "spectral_phase_rad" not in names:
        raise ValueError("spectral phase CSV requires spectral_phase_rad")
    phase = np.asarray(table["spectral_phase_rad"], dtype=float)
    order = np.argsort(wl)
    wl = wl[order]
    phase = phase[order]
    if np.any(np.diff(wl) <= 0.0) or target_wavelengths_m[0] < wl[0] or target_wavelengths_m[-1] > wl[-1]:
        raise ValueError("spectral phase wavelength range must span the measured spectrum")
    if wl.shape == target_wavelengths_m.shape and np.allclose(wl, target_wavelengths_m, rtol=0.0, atol=1e-15):
        return phase, "exact_grid"
    return np.interp(target_wavelengths_m, wl, phase), "linear_phase_interpolation_onto_spectrum_grid"


def spectrum_from_calibration(bundle: CalibrationBundle, config: Phase3BConfig) -> tuple[Spectrum, str]:
    temporal = bundle.data.get("temporal", {})
    spectrum_path = temporal.get("spectrum_path")
    if spectrum_path:
        spectrum = load_spectrum_csv(spectrum_path)
        status = "measured_or_supplied_spectrum"
    else:
        if config.require_measured_spectrum and not config.allow_transform_limited_control:
            raise ValueError("Phase 3B measured-spectrum run blocked: temporal.spectrum_path is missing")
        if not config.allow_transform_limited_control:
            raise ValueError("no spectrum supplied and transform-limited control is disabled")
        wavelength = value_at(bundle, "laser.wavelength_m")
        duration = value_at(bundle, "laser.pulse_duration_s")
        if wavelength in (None, "") or duration in (None, ""):
            raise ValueError("transform-limited control requires laser wavelength and pulse duration")
        spectrum = gaussian_transform_limited_spectrum(
            float(wavelength),
            float(duration),
            samples=int(config.transform_limited_samples),
        )
        status = "synthetic_transform_limited_control_not_measurement"

    phase_path = temporal.get("spectral_phase_path")
    if phase_path:
        phase, interpolation = _load_separate_spectral_phase(str(phase_path), spectrum.wavelengths_m)
        spectrum = Spectrum(
            wavelengths_m=spectrum.wavelengths_m,
            energy_weights=spectrum.energy_weights,
            spectral_phase_rad=phase,
            metadata={**dict(spectrum.metadata), "spectral_phase_source": str(phase_path), "spectral_phase_grid": interpolation},
        ).validated()
        status += "+measured_or_supplied_phase"

    if config.apply_bundle_gdd_tod:
        gdd = value_at(bundle, "temporal.gdd_s2")
        tod = value_at(bundle, "temporal.tod_s3")
        if gdd not in (None, "") or tod not in (None, ""):
            spectrum = apply_polynomial_spectral_phase(
                spectrum,
                gdd_s2=0.0 if gdd in (None, "") else float(gdd),
                tod_s3=0.0 if tod in (None, "") else float(tod),
            )
            status += "+bundle_dispersion_phase"
    return spectrum, status


def objective_pupil_from_assets(assets: Phase3BAssetPaths) -> ObjectivePupilCalibration | None:
    amplitude = _load_numeric_array(assets.objective_amplitude_transmission_path)
    opd = _load_numeric_array(assets.objective_opd_map_m_path)
    valid = _load_numeric_array(assets.objective_valid_mask_path)
    if amplitude is None and opd is None and valid is None:
        return None
    return ObjectivePupilCalibration(
        amplitude_transmission=None if amplitude is None else np.asarray(amplitude, dtype=float),
        opd_map_m=None if opd is None else np.asarray(opd, dtype=float),
        valid_mask=None if valid is None else np.asarray(valid, dtype=bool),
        source="phase3b_asset_paths",
    )


def detector_from_assets(bundle: CalibrationBundle, assets: Phase3BAssetPaths) -> DetectorTransferCalibration | None:
    psf = _load_numeric_array(assets.detector_psf_path)
    response = _load_numeric_array(assets.detector_relative_response_path)
    background = _load_numeric_array(assets.detector_background_path)
    saturation = bundle.data.get("camera", {}).get("saturation_level")
    if psf is None and response is None and background is None and saturation in (None, ""):
        return None
    return DetectorTransferCalibration(
        psf=None if psf is None else np.asarray(psf, dtype=float),
        relative_response=None if response is None else np.asarray(response, dtype=float),
        additive_background=None if background is None else np.asarray(background, dtype=float),
        saturation_level=None if saturation in (None, "") else float(saturation),
        source="phase3b_calibration_assets",
    )


def run_phase3b_broadband(
    bundle: CalibrationBundle,
    propagate_one_context: Callable[[Phase3BWavelengthContext], SpectralFieldPlane],
    *,
    config: Phase3BConfig = Phase3BConfig(),
) -> Phase3BResult:
    """Run calibrated spectral integration around an existing optical solver."""

    spectrum, spectrum_status = spectrum_from_calibration(bundle, config)
    sample_material, sample_status = material_model_from_calibration(
        bundle, role="sample", explicit_name=config.sample_material_name
    )
    axicon_material, axicon_status = material_model_from_calibration(
        bundle, role="axicon", explicit_name=config.axicon_material_name
    )
    objective_calibration = objective_pupil_from_assets(config.assets)
    detector_calibration = detector_from_assets(bundle, config.assets)

    blockers: list[str] = []
    if sample_material is None:
        blockers.append("sample material/index unresolved")
    if axicon_material is None:
        blockers.append("axicon material/index unresolved")

    def one(wavelength_m: float) -> SpectralFieldPlane:
        sample_n = None if sample_material is None else float(sample_material.refractive_index(wavelength_m))
        axicon_n = None if axicon_material is None else float(axicon_material.refractive_index(wavelength_m))
        context = Phase3BWavelengthContext(
            wavelength_m=float(wavelength_m),
            sample_refractive_index=sample_n,
            axicon_refractive_index=axicon_n,
            objective_pupil_calibration=objective_calibration,
            calibration_id=bundle.calibration_id,
            metadata={
                "sample_material_status": sample_status,
                "axicon_material_status": axicon_status,
                "spectrum_status": spectrum_status,
            },
        )
        return propagate_one_context(context)

    broadband = propagate_broadband(spectrum, one)
    detector_result = None
    if detector_calibration is not None:
        detector_result = apply_detector_transfer(
            broadband.detector_integrated_intensity,
            detector_calibration,
        )
    return Phase3BResult(
        broadband=broadband,
        detector_transfer=detector_result,
        spectrum_status=spectrum_status,
        sample_material_status=sample_status,
        axicon_material_status=axicon_status,
        blockers=tuple(blockers),
        metadata={
            "phase": "PHASE 3B BENCH-CALIBRATED BROADBAND DIGITAL TWIN",
            "calibration_id": bundle.calibration_id,
            "data_classification": bundle.data_classification,
            "measured_objective_pupil_applied": objective_calibration is not None,
            "detector_transfer_applied": detector_result is not None,
            "surface_by_surface_boundary_primitive_available": True,
            "full_curved_surface_wave_remapping_available": False,
            "nonlinear_material_propagation_available": False,
            "experimental_validation_claimed": False,
        },
    )


__all__ = [
    "Phase3BAssetPaths",
    "Phase3BConfig",
    "Phase3BResult",
    "Phase3BWavelengthContext",
    "detector_from_assets",
    "material_model_from_calibration",
    "objective_pupil_from_assets",
    "run_phase3b_broadband",
    "spectrum_from_calibration",
]
