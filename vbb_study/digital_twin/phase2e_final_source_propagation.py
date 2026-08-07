"""Governed final source-scale Bessel propagation for Phase 2E.

The nominal route ends at the selected-order 4F reconstruction, applies no
additional real-space stop, applies the physical axicon exactly once, and then
propagates in air.  Soft and hard truncations are explicit sensitivity routes.
"""

from __future__ import annotations

import csv
import gc
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
from scipy import fft as scipy_fft

from vbb_study.digital_twin.phase2a_canonical import _axicon_phase, _variant_settings
from vbb_study.digital_twin.phase2a_contracts import canonical_hardware_manifest, hardware_value
from vbb_study.digital_twin.phase2e_final_source_metrics import (
    dominant_ripple_period_m,
    FixedRegion,
    edge_energy_fraction,
    fixed_region_from_reference,
    fixed_region_power,
    measure_feature,
    normalised_l2,
    on_axis_intensity,
    zone_summary,
)
from vbb_study.digital_twin.phase2e_propagation_repair import (
    build_scalar_reconstructed_4f_field,
)
from vbb_study.digital_twin.phase2e_source_sampling_repair import sampling_diagnostic
from vbb_study.equations.propagation import (
    _transfer_function_medium,
    bandlimit_mask_matsushima,
    make_bl_asm_propagator,
)


EPS = np.finfo(float).tiny
VALIDATION_ROOT = Path("outputs/validation/phase2e_final_propagation")
FIGURE_ROOT = Path("outputs/figures/phase2e_final_source_propagation")
CASE_CHARGES = {"B0": 0, "V1": 1, "V3": 3}
FINAL_RESOLUTION_N = (2048, 2560, 3072)
FINAL_RESOLUTION_Z_M = (0.020, 0.040, 0.060, 0.080, 0.100)

ApertureRoute = Literal[
    "nominal_no_additional_aperture",
    "soft_aperture_sensitivity",
    "hard_aperture_diagnostic",
]


@dataclass(frozen=True)
class FinalSourcePropagationConfig:
    case_id: str
    grid_n: int
    window_m: float = 10.0e-3
    z_min_m: float = 0.0
    z_max_m: float = 180.0e-3
    z_step_m: float = 0.25e-3
    aperture_route: ApertureRoute = "nominal_no_additional_aperture"
    detail_halfwidth_m: float = 0.5e-3
    output_root: Path = FIGURE_ROOT
    snapshot_z_m: tuple[float, ...] = ()
    fixed_region: FixedRegion | None = None
    progress_interval_planes: int = 100


@dataclass
class FinalSourcePropagationResult:
    x_m: np.ndarray
    y_m: np.ndarray
    z_m: np.ndarray
    xz_intensity: np.ndarray
    yz_intensity: np.ndarray
    axial_trace_raw: np.ndarray
    fixed_bucket_power_raw: np.ndarray
    feature_radius_m: np.ndarray
    feature_width_m: np.ndarray
    dark_core_radius_m: np.ndarray
    feature_valid: np.ndarray
    total_plane_power: np.ndarray
    edge_energy_fraction: np.ndarray
    snapshot_fields: dict[float, np.ndarray]
    metadata: dict[str, Any]
    fixed_region: FixedRegion


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, FixedRegion):
        return value.__dict__.copy()
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


def source_scale_route_contract() -> dict[str, Any]:
    manifest = canonical_hardware_manifest()
    radius = float(hardware_value(manifest, "objective_pupil_radius_m"))
    common = {
        "source_plane": "physical_axicon_output_plane",
        "physical_sequence": [
            "SLM phase-only modulation",
            "common 4F selected-order filtering",
            "carrier removal and reconstructed field",
            "declared aperture variant before axicon",
            "physical axicon phase",
            "band-limited angular-spectrum propagation in air",
        ],
        "objective_transform_application_count": 0,
        "axicon_application_count": 1,
        "propagation_medium_index": 1.0,
        "z_origin": "physical_axicon_output_plane",
        "mapping_mode": "fixed_physical_source_scale",
        "not_objective_focused": True,
    }
    return {
        "nominal_no_additional_aperture": {
            **common,
            "aperture_model": "none_additional; finite Gaussian and SLM/numerical support remain",
            "aperture_radius_m": None,
            "aperture_provenance": "nominal source-scale route contract",
            "aperture_application_count": 0,
            "report_eligibility": "primary",
            "calibration_required": False,
        },
        "soft_aperture_sensitivity": {
            **common,
            "aperture_model": "exp[-(r/r_a)^8]",
            "aperture_radius_m": radius,
            "aperture_provenance": "assumed_soft_truncation_sensitivity",
            "aperture_application_count": 1,
            "report_eligibility": "sensitivity_only",
            "calibration_required": True,
        },
        "hard_aperture_diagnostic": {
            **common,
            "aperture_model": "unit disk hard truncation",
            "aperture_radius_m": radius,
            "aperture_provenance": "nominal_hard_aperture_placeholder",
            "aperture_application_count": 1,
            "report_eligibility": "diagnostic_only",
            "calibration_required": True,
            "required_label": "diagnostic hard truncation; not nominal experimental prediction",
        },
    }


def write_source_scale_route_contract(output_root: Path = VALIDATION_ROOT) -> Path:
    return _write_json(output_root / "source_scale_route_contract.json", source_scale_route_contract())


def build_final_source_field(
    case_id: str,
    *,
    grid_n: int,
    window_m: float = 10.0e-3,
    aperture_route: ApertureRoute = "nominal_no_additional_aperture",
) -> tuple[np.ndarray, Mapping[str, Any], dict[str, Any]]:
    """Build one source field at the physical axicon output plane."""

    case = str(case_id).upper()
    if case not in CASE_CHARGES:
        raise ValueError(f"unsupported final source case {case_id!r}")
    contract = source_scale_route_contract()
    if aperture_route not in contract:
        raise ValueError(f"unknown aperture route {aperture_route!r}")
    reconstructed, grid, upstream = build_scalar_reconstructed_4f_field(
        case,
        CASE_CHARGES[case],
        grid_n=int(grid_n),
        window_m=float(window_m),
        variant="realistic_fixed_bench_route",
    )
    route = contract[aperture_route]
    radius = np.asarray(grid["R"], dtype=float)
    aperture_radius = route["aperture_radius_m"]
    if aperture_route == "nominal_no_additional_aperture":
        pre_axicon = reconstructed
    elif aperture_route == "soft_aperture_sensitivity":
        pre_axicon = reconstructed * np.exp(-((radius / float(aperture_radius)) ** 8))
    else:
        pre_axicon = np.where(radius <= float(aperture_radius), reconstructed, 0.0)
    manifest = canonical_hardware_manifest()
    axicon, kr = _axicon_phase(grid, manifest, _variant_settings("realistic_fixed_bench_route"))
    source = np.asarray(pre_axicon * axicon, dtype=np.complex128)
    diagnostic = sampling_diagnostic(int(grid_n), float(window_m))
    metadata = {
        **upstream,
        **route,
        "route_id": aperture_route,
        "case_id": case,
        "requested_winding": CASE_CHARGES[case],
        "source_grid_n": int(grid_n),
        "dx_m": float(grid["dx"]),
        "z_step_m": None,
        "radial_wavevector_m_inv": float(kr),
        "samples_per_axicon_radial_period": diagnostic.samples_per_radial_period,
        "adjacent_radial_phase_increment_rad": diagnostic.adjacent_radial_phase_increment_rad,
        "samples_per_carrier_period": diagnostic.samples_per_carrier_period,
        "axicon_application_count": 1,
        "objective_transform_application_count": 0,
        "first_order_filter_application_count": upstream["first_order_filter_application_count"],
        "carrier_removal_application_count": upstream["carrier_removal_application_count"],
    }
    return source, grid, metadata


def _array_bytes(value: Any) -> int:
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, Mapping):
        return sum(_array_bytes(item) for item in value.values())
    return 0


def _make_threaded_bl_asm_propagator(
    source: np.ndarray,
    grid: Mapping[str, Any],
    wavelength_m: float,
    *,
    workers: int = -1,
) -> Any:
    """Memory-safe single-precision BL-ASM closure for validated production use."""

    spectrum = scipy_fft.fftshift(
        scipy_fft.fft2(
            scipy_fft.ifftshift(np.asarray(source, dtype=np.complex64)),
            workers=workers,
        )
    )
    wavelength = float(wavelength_m)
    k = 2.0 * np.pi / wavelength
    kx = 2.0 * np.pi * np.asarray(grid["FX"], dtype=float)
    ky = 2.0 * np.pi * np.asarray(grid["FY"], dtype=float)
    delta_kz = (
        np.sqrt(np.maximum(k * k - kx * kx - ky * ky, 0.0)) - k
    ).astype(np.float32)
    frequency = np.asarray(grid["FX"][0], dtype=float)
    n = int(grid["N"])
    dx = float(grid["dx"])

    def propagate(z_m: float) -> np.ndarray:
        if float(z_m) == 0.0:
            return np.asarray(source, dtype=np.complex64).copy()
        z_value = float(z_m)
        # The omitted exp(i*k*z) is a spatially uniform phase and cannot alter
        # intensity or winding. Removing it prevents large-phase float32 loss.
        transfer = np.exp(np.complex64(1j * z_value) * delta_kz)
        du = 1.0 / (n * dx)
        limit = 1.0 / (wavelength * math.sqrt((2.0 * du * abs(z_value)) ** 2 + 1.0))
        retained = np.flatnonzero(np.abs(frequency) <= limit)
        if retained.size:
            lo, hi = int(retained[0]), int(retained[-1]) + 1
            transfer[:lo, :] = 0.0
            transfer[hi:, :] = 0.0
            transfer[:, :lo] = 0.0
            transfer[:, hi:] = 0.0
        else:
            transfer.fill(0.0)
        return scipy_fft.fftshift(
            scipy_fft.ifft2(
                scipy_fft.ifftshift(spectrum * transfer), workers=workers
            )
        )

    return propagate


def _snapshot_schedule(z_m: np.ndarray, zones: Mapping[str, Any]) -> tuple[float, ...]:
    measured = zones.get("measured_strict_useful_region_m")
    fwhm = zones.get("measured_FWHM_axial_zone_m")
    if measured is not None:
        start, stop = (float(measured[0]), float(measured[1]))
    elif fwhm is not None:
        start, stop = (float(fwhm[0]), float(fwhm[1]))
    else:
        start, stop = (0.020, 0.100)
    requested = (
        max(float(z_m[0]), start - 10.0e-3),
        start,
        0.5 * (start + stop),
        stop,
        min(float(z_m[-1]), stop + 5.0e-3),
        float(z_m[-1]),
    )
    return tuple(
        float(z_m[int(np.argmin(np.abs(z_m - value)))]) for value in requested
    )


def _geometric_zone_estimate_m() -> tuple[float, float]:
    manifest = canonical_hardware_manifest()
    beam_radius = float(hardware_value(manifest, "beam_radius_on_slm_m"))
    n_axicon = float(hardware_value(manifest, "axicon_refractive_index"))
    n_medium = float(hardware_value(manifest, "axicon_external_medium_index"))
    base = math.radians(float(hardware_value(manifest, "axicon_base_angle_deg")))
    cone_slope = (n_axicon / n_medium - 1.0) * math.tan(base)
    return 0.0, beam_radius / max(cone_slope, EPS)


def validate_production_backend(
    *,
    output_root: Path = VALIDATION_ROOT,
    z_values_m: Sequence[float] = (0.020, 0.060, 0.100),
) -> dict[str, Any]:
    """Validate the production complex64 backend against complex128 BL-ASM."""

    rows: list[dict[str, Any]] = []
    for case_id in CASE_CHARGES:
        source, grid, metadata = build_final_source_field(case_id, grid_n=3072)
        reference_prop = make_bl_asm_propagator(
            source,
            dict(grid),
            float(metadata["wavelength_m"]),
            n_medium=1.0,
            bandlimit=True,
            include_evanescent=True,
        )
        production_prop = _make_threaded_bl_asm_propagator(
            source, grid, float(metadata["wavelength_m"])
        )
        reference_at_60 = reference_prop(0.060)
        reference_measurement = measure_feature(
            case_id, np.abs(reference_at_60) ** 2, grid
        )
        fixed_region = fixed_region_from_reference(reference_measurement)
        del reference_at_60
        radius = np.asarray(grid["R"], dtype=float)
        for z_m in z_values_m:
            reference = reference_prop(float(z_m))
            production = production_prop(float(z_m))
            reference_intensity = np.abs(reference) ** 2
            production_intensity = np.abs(production) ** 2
            ref_feature = measure_feature(case_id, reference_intensity, grid)
            prod_feature = measure_feature(case_id, production_intensity, grid)
            ref_primary = float(ref_feature.primary_observable_raw)
            prod_primary = float(prod_feature.primary_observable_raw)
            ref_bucket = fixed_region_power(reference_intensity, radius, fixed_region, float(grid["dx"]))
            prod_bucket = fixed_region_power(production_intensity, radius, fixed_region, float(grid["dx"]))
            ref_total = float(np.sum(reference_intensity))
            prod_total = float(np.sum(production_intensity))
            production128 = production.astype(np.complex128)
            phase_alignment = np.vdot(production128, reference)
            if abs(phase_alignment) > EPS:
                production128 *= np.exp(1j * np.angle(phase_alignment))
            field_l2 = float(
                np.linalg.norm(production128 - reference)
                / max(float(np.linalg.norm(reference)), EPS)
            )
            intensity_l2 = float(
                np.linalg.norm(production_intensity - reference_intensity)
                / max(float(np.linalg.norm(reference_intensity)), EPS)
            )
            radius_relative = (
                abs(prod_feature.feature_radius_m - ref_feature.feature_radius_m)
                / max(abs(ref_feature.feature_radius_m), EPS)
                if ref_feature.valid and prod_feature.valid
                else float("nan")
            )
            row = {
                "case_id": case_id,
                "z_m": float(z_m),
                "reference_dtype": "complex128",
                "production_dtype": "complex64",
                "phase_aligned_field_normalised_l2_difference": field_l2,
                "intensity_normalised_l2_difference": intensity_l2,
                "primary_relative_difference": abs(prod_primary - ref_primary) / max(abs(ref_primary), EPS),
                "fixed_bucket_relative_difference": abs(prod_bucket - ref_bucket) / max(abs(ref_bucket), EPS),
                "feature_radius_relative_difference": radius_relative,
                "total_power_relative_difference": abs(prod_total - ref_total) / max(abs(ref_total), EPS),
                "reference_feature_valid": ref_feature.valid,
                "production_feature_valid": prod_feature.valid,
            }
            row["pass"] = (
                row["phase_aligned_field_normalised_l2_difference"] <= 1.0e-4
                and row["intensity_normalised_l2_difference"] <= 1.0e-4
                and row["primary_relative_difference"] <= 1.0e-4
                and row["fixed_bucket_relative_difference"] <= 1.0e-4
                and row["total_power_relative_difference"] <= 1.0e-4
                and (
                    not np.isfinite(row["feature_radius_relative_difference"])
                    or row["feature_radius_relative_difference"] <= 1.0e-4
                )
            )
            rows.append(row)
            del reference, production, reference_intensity, production_intensity
        del reference_prop, production_prop, source, grid, radius
        gc.collect()
    status = all(bool(row["pass"]) for row in rows)
    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(output_root / "production_backend_validation.csv", rows)
    result = {
        "status": "passed" if status else "failed",
        "reference_backend": "complex128_numpy_BL_ASM",
        "production_backend": "complex64_scipy_threaded_BL_ASM",
        "threshold": 1.0e-4,
        "rows": rows,
    }
    _write_json(output_root / "production_backend_validation.json", result)
    return result


def _selected_plane_rows(
    case_id: str,
    grid_n: int,
    fixed_region: FixedRegion,
    *,
    z_values_m: Sequence[float],
    window_m: float,
) -> tuple[list[dict[str, Any]], int, float]:
    started = time.perf_counter()
    source, grid, metadata = build_final_source_field(
        case_id,
        grid_n=int(grid_n),
        window_m=float(window_m),
        aperture_route="nominal_no_additional_aperture",
    )
    source_power = float(np.sum(np.abs(source) ** 2) * float(grid["dx"]) ** 2)
    propagate = make_bl_asm_propagator(
        source,
        dict(grid),
        float(metadata["wavelength_m"]),
        n_medium=1.0,
        bandlimit=True,
        include_evanescent=True,
    )
    estimated_peak = _array_bytes(grid) + int(source.nbytes) * 4
    rows: list[dict[str, Any]] = []
    for z_m in z_values_m:
        field = propagate(float(z_m))
        intensity = np.abs(field) ** 2
        feature = measure_feature(case_id, intensity, grid)
        total_power = float(np.sum(intensity) * float(grid["dx"]) ** 2)
        rows.append({
            "case_id": str(case_id),
            "route_id": "nominal_no_additional_aperture",
            "grid_n": int(grid_n),
            "window_m": float(window_m),
            "dx_m": float(grid["dx"]),
            "samples_per_axicon_radial_period": metadata["samples_per_axicon_radial_period"],
            "adjacent_radial_phase_increment_rad": metadata["adjacent_radial_phase_increment_rad"],
            "samples_per_carrier_period": metadata["samples_per_carrier_period"],
            "z_m": float(z_m),
            "on_axis_intensity_raw": on_axis_intensity(intensity),
            "primary_observable_raw": float(feature.primary_observable_raw),
            "total_power_raw": total_power,
            "fixed_core_or_annulus_power_raw": fixed_region_power(
                intensity, np.asarray(grid["R"], dtype=float), fixed_region, float(grid["dx"])
            ),
            "feature_radius_or_ring_radius_m": float(feature.feature_radius_m),
            "feature_width_m": float(feature.feature_width_m),
            "feature_valid": bool(feature.valid),
            "feature_invalid_reason": feature.invalid_reason,
            "edge_energy_fraction": edge_energy_fraction(intensity, grid),
            "propagation_power_drift_fraction": abs(total_power - source_power) / max(source_power, EPS),
            "fixed_region_inner_radius_m": fixed_region.inner_radius_m,
            "fixed_region_outer_radius_m": fixed_region.outer_radius_m,
        })
        del field, intensity
    elapsed = time.perf_counter() - started
    del propagate, source, grid
    gc.collect()
    return rows, estimated_peak, elapsed


def _reference_fixed_region(
    case_id: str,
    *,
    grid_n: int = 3072,
    window_m: float = 10.0e-3,
    z_m: float = 0.060,
) -> tuple[FixedRegion, int, float]:
    started = time.perf_counter()
    source, grid, metadata = build_final_source_field(
        case_id,
        grid_n=grid_n,
        window_m=window_m,
        aperture_route="nominal_no_additional_aperture",
    )
    propagate = make_bl_asm_propagator(
        source, dict(grid), float(metadata["wavelength_m"]), n_medium=1.0,
        bandlimit=True, include_evanescent=True,
    )
    plane = propagate(float(z_m))
    measurement = measure_feature(case_id, np.abs(plane) ** 2, grid)
    region = fixed_region_from_reference(measurement)
    estimated_peak = _array_bytes(grid) + int(source.nbytes) * 4
    del plane, propagate, source, grid
    gc.collect()
    return region, estimated_peak, time.perf_counter() - started


def run_final_resolution_gate(
    *,
    output_root: Path = VALIDATION_ROOT,
    n_values: Sequence[int] = FINAL_RESOLUTION_N,
    z_values_m: Sequence[float] = FINAL_RESOLUTION_Z_M,
    window_m: float = 10.0e-3,
) -> dict[str, Any]:
    """Run the mandatory selected-plane gate before any final report rendering."""

    if tuple(int(n) for n in n_values) != FINAL_RESOLUTION_N:
        raise ValueError(f"final gate requires N={FINAL_RESOLUTION_N}")
    if not math.isclose(float(window_m), 10.0e-3, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("final source-resolution gate requires the fixed 10 mm window")
    output_root.mkdir(parents=True, exist_ok=True)
    write_source_scale_route_contract(output_root)
    rows: list[dict[str, Any]] = []
    peak_memory = 0
    runtime = 0.0
    fixed_regions: dict[str, FixedRegion] = {}
    for case_id in CASE_CHARGES:
        region, memory_bytes, elapsed = _reference_fixed_region(case_id, window_m=window_m)
        fixed_regions[case_id] = region
        peak_memory = max(peak_memory, memory_bytes)
        runtime += elapsed
        for grid_n in n_values:
            case_rows, memory_bytes, elapsed = _selected_plane_rows(
                case_id,
                int(grid_n),
                region,
                z_values_m=z_values_m,
                window_m=window_m,
            )
            rows.extend(case_rows)
            peak_memory = max(peak_memory, memory_bytes)
            runtime += elapsed

    by_key = {(row["case_id"], row["grid_n"], row["z_m"]): row for row in rows}
    for row in rows:
        reference = by_key[(row["case_id"], 3072, row["z_m"])]
        primary_rel = abs(row["primary_observable_raw"] - reference["primary_observable_raw"]) / max(
            abs(reference["primary_observable_raw"]), EPS
        )
        bucket_rel = abs(
            row["fixed_core_or_annulus_power_raw"] - reference["fixed_core_or_annulus_power_raw"]
        ) / max(abs(reference["fixed_core_or_annulus_power_raw"]), EPS)
        total_rel = abs(row["total_power_raw"] - reference["total_power_raw"]) / max(
            abs(reference["total_power_raw"]), EPS
        )
        if reference["feature_valid"]:
            if row["feature_valid"]:
                radius_rel = abs(
                    row["feature_radius_or_ring_radius_m"]
                    - reference["feature_radius_or_ring_radius_m"]
                ) / max(abs(reference["feature_radius_or_ring_radius_m"]), EPS)
                radius_applicable = True
            else:
                radius_rel = float("inf")
                radius_applicable = True
        else:
            radius_rel = float("nan")
            radius_applicable = False
        row.update({
            "reference_grid_n": 3072,
            "primary_observable_relative_difference_to_n3072": primary_rel,
            "fixed_bucket_relative_difference_to_n3072": bucket_rel,
            "feature_radius_relative_difference_to_n3072": radius_rel,
            "feature_radius_gate_applicable": radius_applicable,
            "total_power_relative_difference_to_n3072": total_rel,
            "primary_gate_pass": primary_rel <= 0.01,
            "fixed_bucket_gate_pass": bucket_rel <= 0.01,
            "feature_radius_gate_pass": (not radius_applicable) or radius_rel <= 0.01,
            "total_power_gate_pass": total_rel <= 1.0e-3,
            "edge_energy_gate_pass": row["edge_energy_fraction"] <= 0.01,
            "power_drift_gate_pass": row["propagation_power_drift_fraction"] <= 1.0e-3,
        })
        row["row_gate_pass"] = all(
            row[name] for name in (
                "primary_gate_pass",
                "fixed_bucket_gate_pass",
                "feature_radius_gate_pass",
                "total_power_gate_pass",
                "edge_energy_gate_pass",
                "power_drift_gate_pass",
            )
        )

    grid_pass = {
        n: all(row["row_gate_pass"] for row in rows if row["grid_n"] == n)
        for n in n_values
    }
    n3072_internal = bool(grid_pass[3072])
    selected_n = 2560 if grid_pass[2560] else (3072 if n3072_internal else None)
    authorised = selected_n is not None
    outcome = "PHASE2E-FINAL-C" if not authorised else "PENDING_Z_GATE_AND_FIGURES"
    rows.sort(key=lambda row: (row["case_id"], row["grid_n"], row["z_m"]))
    _write_csv(output_root / "final_resolution_gate.csv", rows)
    summary = {
        "gate_id": "phase2e_final_source_resolution_gate",
        "status": "passed" if authorised else "failed",
        "outcome": outcome,
        "report_figures_authorised": False,
        "report_figures_pending_z_gate": bool(authorised),
        "nominal_route": "nominal_no_additional_aperture",
        "window_m": float(window_m),
        "n_values": list(n_values),
        "z_values_m": list(z_values_m),
        "reference_grid_n": 3072,
        "grid_pass": grid_pass,
        "selected_production_grid_n": selected_n,
        "thresholds": {
            "primary_observable_relative_difference": 0.01,
            "fixed_bucket_relative_difference": 0.01,
            "valid_feature_radius_relative_difference": 0.01,
            "total_power_relative_difference": 1.0e-3,
            "maximum_edge_energy_fraction": 0.01,
            "maximum_propagation_power_drift_fraction": 1.0e-3,
        },
        "fixed_regions": fixed_regions,
        "peak_estimated_memory_bytes": int(peak_memory),
        "runtime_seconds": float(runtime),
        "rows": rows,
    }
    _write_json(output_root / "final_resolution_gate.json", summary)
    return summary


def run_final_source_propagation(
    config: FinalSourcePropagationConfig,
) -> FinalSourcePropagationResult:
    """Sequentially propagate one final source case and retain only governed products."""

    if int(config.grid_n) != 3072:
        raise ValueError("the completed final resolution gate selected N=3072")
    if config.z_step_m <= 0.0 or config.z_max_m < config.z_min_m:
        raise ValueError("invalid propagation z domain")
    started = time.perf_counter()
    source, grid, metadata = build_final_source_field(
        config.case_id,
        grid_n=int(config.grid_n),
        window_m=float(config.window_m),
        aperture_route=config.aperture_route,
    )
    propagate = _make_threaded_bl_asm_propagator(
        source, grid, float(metadata["wavelength_m"])
    )
    z_m = np.arange(
        float(config.z_min_m),
        float(config.z_max_m) + 0.5 * float(config.z_step_m),
        float(config.z_step_m),
        dtype=float,
    )
    n = int(config.grid_n)
    centre = n // 2
    dx = float(grid["dx"])
    source_power = float(np.sum(np.abs(source) ** 2) * dx**2)

    fixed_region = config.fixed_region
    if fixed_region is None:
        reference_z = float(np.clip(0.060, z_m[0], z_m[-1]))
        reference_plane = propagate(reference_z)
        reference_feature = measure_feature(
            config.case_id, np.abs(reference_plane) ** 2, grid
        )
        fixed_region = fixed_region_from_reference(reference_feature)
        del reference_plane

    xz = np.empty((z_m.size, n), dtype=np.float32)
    yz = np.empty((z_m.size, n), dtype=np.float32)
    primary = np.empty(z_m.size, dtype=float)
    fixed_power = np.empty(z_m.size, dtype=float)
    feature_radius = np.full(z_m.size, np.nan, dtype=float)
    feature_width = np.full(z_m.size, np.nan, dtype=float)
    dark_core_radius = np.full(z_m.size, np.nan, dtype=float)
    feature_valid = np.zeros(z_m.size, dtype=bool)
    feature_reason = np.empty(z_m.size, dtype=object)
    total_power = np.empty(z_m.size, dtype=float)
    edge_fraction = np.empty(z_m.size, dtype=float)

    radius = np.asarray(grid["R"], dtype=float)
    fixed_mask = (
        (radius >= fixed_region.inner_radius_m)
        & (radius <= fixed_region.outer_radius_m)
    )
    for index, z_value in enumerate(z_m):
        field = source if float(z_value) == 0.0 else propagate(float(z_value))
        intensity = np.abs(field) ** 2
        xz[index] = np.asarray(intensity[centre, :], dtype=np.float32)
        yz[index] = np.asarray(intensity[:, centre], dtype=np.float32)
        measurement = measure_feature(config.case_id, intensity, grid)
        primary[index] = measurement.primary_observable_raw
        fixed_power[index] = float(np.sum(intensity[fixed_mask]) * dx**2)
        feature_radius[index] = measurement.feature_radius_m
        feature_width[index] = measurement.feature_width_m
        dark_core_radius[index] = measurement.dark_core_radius_m
        feature_valid[index] = measurement.valid
        feature_reason[index] = measurement.invalid_reason
        total_power[index] = float(np.sum(intensity) * dx**2)
        edge_fraction[index] = edge_energy_fraction(intensity, grid)
        if field is not source:
            del field
        del intensity
        if (
            config.progress_interval_planes > 0
            and (index + 1) % int(config.progress_interval_planes) == 0
        ):
            print(json.dumps({
                "progress": "phase2e_final_source_propagation",
                "case_id": config.case_id,
                "route_id": config.aperture_route,
                "completed_planes": index + 1,
                "total_planes": int(z_m.size),
                "elapsed_seconds": time.perf_counter() - started,
            }), flush=True)

    formation_threshold = 0.05 * max(float(np.nanmax(primary)), EPS)
    below = primary < formation_threshold
    feature_valid[below] = False
    feature_radius[below] = np.nan
    feature_width[below] = np.nan
    dark_core_radius[below] = np.nan
    feature_reason[below] = "below_declared_five_percent_formation_threshold"
    invalid = ~feature_valid
    feature_radius[invalid] = np.nan
    feature_width[invalid] = np.nan
    dark_core_radius[invalid] = np.nan

    zones = zone_summary(
        config.case_id,
        z_m,
        primary,
        fixed_power,
        feature_radius,
        feature_valid,
    )
    zones["geometric_zone_estimate_m"] = _geometric_zone_estimate_m()
    schedule = tuple(config.snapshot_z_m) or _snapshot_schedule(z_m, zones)
    snapshots: dict[float, np.ndarray] = {}
    snapshot_indices = np.flatnonzero(
        np.abs(np.asarray(grid["x"], dtype=float)) <= float(config.detail_halfwidth_m)
    )
    snapshot_slice = slice(int(snapshot_indices[0]), int(snapshot_indices[-1]) + 1)
    for requested_z in schedule:
        actual_z = float(z_m[int(np.argmin(np.abs(z_m - requested_z)))])
        field = source if actual_z == 0.0 else propagate(actual_z)
        snapshots[actual_z] = np.asarray(
            field[snapshot_slice, snapshot_slice], dtype=np.complex64
        )
        if field is not source:
            del field

    runtime = time.perf_counter() - started
    estimated_peak = (
        _array_bytes(grid)
        + int(source.nbytes) * 5
        + int(xz.nbytes + yz.nbytes)
        + sum(value.nbytes for value in snapshots.values())
    )
    metadata.update({
        "z_step_m": float(config.z_step_m),
        "z_min_m": float(z_m[0]),
        "z_max_m": float(z_m[-1]),
        "detail_halfwidth_m": float(config.detail_halfwidth_m),
        "fixed_region": fixed_region,
        "formation_threshold_policy": "five_percent_of_case_route_raw_primary_maximum",
        "formation_threshold_raw": float(formation_threshold),
        "feature_invalid_reasons": [str(value) for value in feature_reason],
        "zones": zones,
        "propagation_normalisation": "raw_native_intensity; no per-z normalisation",
        "propagation_scaling": "linear",
        "maximum_edge_energy_fraction": float(np.max(edge_fraction)),
        "maximum_propagation_power_drift_fraction": float(
            np.max(np.abs(total_power - source_power)) / max(source_power, EPS)
        ),
        "source_power_raw": source_power,
        "snapshot_z_m": list(snapshots),
        "snapshot_crop_halfwidth_m": float(config.detail_halfwidth_m),
        "snapshot_x_m": np.asarray(grid["x"], dtype=float)[snapshot_slice],
        "peak_estimated_memory_bytes": int(estimated_peak),
        "runtime_seconds": float(runtime),
        "metrics_computed_on_native_arrays": True,
        "display_interpolation_used_for_metrics": False,
    })
    x = np.asarray(grid["x"], dtype=float).copy()
    del propagate, source, grid, radius, fixed_mask
    gc.collect()
    return FinalSourcePropagationResult(
        x_m=x,
        y_m=x.copy(),
        z_m=z_m,
        xz_intensity=xz,
        yz_intensity=yz,
        axial_trace_raw=primary,
        fixed_bucket_power_raw=fixed_power,
        feature_radius_m=feature_radius,
        feature_width_m=feature_width,
        dark_core_radius_m=dark_core_radius,
        feature_valid=feature_valid,
        total_plane_power=total_power,
        edge_energy_fraction=edge_fraction,
        snapshot_fields=snapshots,
        metadata=metadata,
        fixed_region=fixed_region,
    )


def result_cache_stem(
    case_id: str,
    aperture_route: ApertureRoute,
    *,
    output_root: Path = VALIDATION_ROOT,
) -> Path:
    return output_root / "production_cache" / f"{case_id.lower()}_{aperture_route}"


def _maximum_interval_boundary_shift_m(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> float:
    shifts: list[float] = []
    for key in ("measured_FWHM_axial_zone_m", "measured_strict_useful_region_m"):
        a = first.get(key)
        b = second.get(key)
        if a is None or b is None:
            return float("inf")
        shifts.extend((abs(float(a[0]) - float(b[0])), abs(float(a[1]) - float(b[1]))))
    return max(shifts)


def run_z_step_convergence(
    *,
    output_root: Path = VALIDATION_ROOT,
    fine_min_m: float = 0.0,
    fine_max_m: float = 0.140,
    case_ids: Sequence[str] = tuple(CASE_CHARGES),
) -> dict[str, Any]:
    """Compare dz=0.25 mm production traces with native dz=0.125 mm midpoints."""

    rows: list[dict[str, Any]] = []
    total_runtime = 0.0
    peak_memory = 0
    requested_cases = tuple(str(case_id).upper() for case_id in case_ids)
    if not requested_cases or any(case_id not in CASE_CHARGES for case_id in requested_cases):
        raise ValueError("z convergence cases must be a non-empty subset of B0/V1/V3")
    for case_id in requested_cases:
        coarse = load_final_source_result(case_id, "nominal_no_additional_aperture", output_root=output_root)
        select = (coarse.z_m >= fine_min_m - 1e-15) & (coarse.z_m <= fine_max_m + 1e-15)
        coarse_z = coarse.z_m[select]
        coarse_primary = coarse.axial_trace_raw[select]
        coarse_bucket = coarse.fixed_bucket_power_raw[select]
        coarse_radius = coarse.feature_radius_m[select]
        coarse_valid = coarse.feature_valid[select]
        fine_z = np.arange(fine_min_m, fine_max_m + 0.0625e-3, 0.125e-3)
        fine_primary = np.interp(fine_z, coarse_z, coarse_primary)
        fine_bucket = np.interp(fine_z, coarse_z, coarse_bucket)
        fine_radius = np.full(fine_z.size, np.nan, dtype=float)
        finite_coarse_radius = coarse_valid & np.isfinite(coarse_radius)
        if np.count_nonzero(finite_coarse_radius) >= 2:
            fine_radius[:] = np.interp(
                fine_z,
                coarse_z[finite_coarse_radius],
                coarse_radius[finite_coarse_radius],
                left=np.nan,
                right=np.nan,
            )
        fine_valid = np.isfinite(fine_radius)

        started = time.perf_counter()
        source, grid, metadata = build_final_source_field(case_id, grid_n=3072)
        propagate = _make_threaded_bl_asm_propagator(
            source, grid, float(metadata["wavelength_m"])
        )
        radius_grid = np.asarray(grid["R"], dtype=float)
        dx = float(grid["dx"])
        midpoint = np.isclose(
            np.mod(np.round(fine_z * 1.0e6).astype(int), 250),
            125,
        )
        for index in np.flatnonzero(midpoint):
            field = propagate(float(fine_z[index]))
            intensity = np.abs(field) ** 2
            measurement = measure_feature(case_id, intensity, grid)
            fine_primary[index] = measurement.primary_observable_raw
            fine_bucket[index] = fixed_region_power(
                intensity, radius_grid, coarse.fixed_region, dx
            )
            if (
                measurement.valid
                and measurement.primary_observable_raw
                >= float(coarse.metadata["formation_threshold_raw"])
            ):
                fine_radius[index] = measurement.feature_radius_m
                fine_valid[index] = True
            else:
                fine_radius[index] = np.nan
                fine_valid[index] = False
            del field, intensity
        runtime = time.perf_counter() - started
        total_runtime += runtime
        peak_memory = max(peak_memory, _array_bytes(grid) + int(source.nbytes) * 5)
        del propagate, source, grid, radius_grid
        gc.collect()

        interpolated_primary = np.interp(fine_z, coarse_z, coarse_primary)
        interpolated_bucket = np.interp(fine_z, coarse_z, coarse_bucket)
        primary_l2 = normalised_l2(fine_primary, interpolated_primary)
        bucket_l2 = normalised_l2(fine_bucket, interpolated_bucket)
        radius_compare = fine_valid & np.isfinite(fine_radius)
        if np.count_nonzero(finite_coarse_radius) >= 2:
            coarse_radius_on_fine = np.interp(
                fine_z,
                coarse_z[finite_coarse_radius],
                coarse_radius[finite_coarse_radius],
                left=np.nan,
                right=np.nan,
            )
            radius_compare &= np.isfinite(coarse_radius_on_fine)
        else:
            coarse_radius_on_fine = np.full_like(fine_z, np.nan)
        radius_relative = (
            float(np.nanmax(
                np.abs(fine_radius[radius_compare] - coarse_radius_on_fine[radius_compare])
                / np.maximum(np.abs(fine_radius[radius_compare]), EPS)
            ))
            if np.any(radius_compare)
            else float("inf")
        )
        coarse_zones = zone_summary(
            case_id, coarse_z, coarse_primary, coarse_bucket, coarse_radius, coarse_valid
        )
        fine_zones = zone_summary(
            case_id, fine_z, fine_primary, fine_bucket, fine_radius, fine_valid
        )
        boundary_shift = _maximum_interval_boundary_shift_m(coarse_zones, fine_zones)
        coarse_period = dominant_ripple_period_m(coarse_z, coarse_primary)
        fine_period = dominant_ripple_period_m(fine_z, fine_primary)
        period_relative = (
            abs(coarse_period - fine_period) / max(abs(fine_period), EPS)
            if np.isfinite(coarse_period) and np.isfinite(fine_period)
            else float("inf")
        )
        row = {
            "case_id": case_id,
            "coarse_z_step_m": 0.25e-3,
            "fine_z_step_m": 0.125e-3,
            "validation_z_min_m": float(fine_min_m),
            "validation_z_max_m": float(fine_max_m),
            "raw_primary_normalised_l2_difference": primary_l2,
            "fixed_bucket_normalised_l2_difference": bucket_l2,
            "feature_radius_maximum_relative_difference": radius_relative,
            "useful_zone_maximum_boundary_shift_m": boundary_shift,
            "coarse_dominant_ripple_period_m": coarse_period,
            "fine_dominant_ripple_period_m": fine_period,
            "dominant_ripple_period_relative_difference": period_relative,
            "coarse_zones": json.dumps(_json_ready(coarse_zones), sort_keys=True),
            "fine_zones": json.dumps(_json_ready(fine_zones), sort_keys=True),
            "runtime_seconds": runtime,
        }
        row.update({
            "raw_primary_gate_pass": primary_l2 <= 0.01,
            "fixed_bucket_gate_pass": bucket_l2 <= 0.01,
            "feature_radius_gate_pass": radius_relative <= 0.01,
            "zone_boundary_gate_pass": boundary_shift <= 0.5e-3,
            "ripple_period_gate_pass": period_relative <= 0.01,
        })
        row["case_gate_pass"] = all(
            row[key] for key in (
                "raw_primary_gate_pass",
                "fixed_bucket_gate_pass",
                "feature_radius_gate_pass",
                "zone_boundary_gate_pass",
                "ripple_period_gate_pass",
            )
        )
        rows.append(row)
    passed = all(bool(row["case_gate_pass"]) for row in rows)
    complete = set(requested_cases) == set(CASE_CHARGES)
    suffix = "" if complete else "_" + "_".join(case_id.lower() for case_id in requested_cases)
    _write_csv(output_root / f"z_step_convergence{suffix}.csv", rows)
    result = {
        "status": "passed" if passed else "failed",
        "outcome": "PENDING_FIGURE_PACK" if passed else "PHASE2E-FINAL-C",
        "report_figures_authorised": bool(passed and complete),
        "selected_z_step_m": 0.25e-3 if passed else None,
        "reference_z_step_m": 0.125e-3,
        "representative_range_m": [fine_min_m, fine_max_m],
        "thresholds": {
            "raw_primary_normalised_l2_difference": 0.01,
            "fixed_bucket_normalised_l2_difference": 0.01,
            "feature_radius_relative_difference": 0.01,
            "useful_zone_boundary_shift_m": 0.5e-3,
            "dominant_ripple_period_relative_difference": 0.01,
        },
        "runtime_seconds": total_runtime,
        "peak_estimated_memory_bytes": peak_memory,
        "rows": rows,
        "case_ids": list(requested_cases),
        "complete_case_set": complete,
    }
    _write_json(output_root / f"z_step_convergence{suffix}.json", result)
    return result


def finalize_z_step_convergence(
    *,
    output_root: Path = VALIDATION_ROOT,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    total_runtime = 0.0
    peak_memory = 0
    for case_id in CASE_CHARGES:
        path = output_root / f"z_step_convergence_{case_id.lower()}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("case_ids") != [case_id]:
            raise RuntimeError(f"unexpected z-convergence case payload in {path}")
        rows.extend(payload["rows"])
        total_runtime += float(payload["runtime_seconds"])
        peak_memory = max(peak_memory, int(payload["peak_estimated_memory_bytes"]))
    passed = all(bool(row["case_gate_pass"]) for row in rows)
    _write_csv(output_root / "z_step_convergence.csv", rows)
    result = {
        "status": "passed" if passed else "failed",
        "outcome": "PENDING_FIGURE_PACK" if passed else "PHASE2E-FINAL-C",
        "report_figures_authorised": bool(passed),
        "selected_z_step_m": 0.25e-3 if passed else None,
        "reference_z_step_m": 0.125e-3,
        "representative_range_m": [0.0, 0.140],
        "thresholds": {
            "raw_primary_normalised_l2_difference": 0.01,
            "fixed_bucket_normalised_l2_difference": 0.01,
            "feature_radius_relative_difference": 0.01,
            "useful_zone_boundary_shift_m": 0.5e-3,
            "dominant_ripple_period_relative_difference": 0.01,
        },
        "runtime_seconds": total_runtime,
        "peak_estimated_memory_bytes": peak_memory,
        "rows": rows,
        "case_ids": list(CASE_CHARGES),
        "complete_case_set": True,
    }
    _write_json(output_root / "z_step_convergence.json", result)
    return result


def save_final_source_result(
    result: FinalSourcePropagationResult,
    *,
    output_root: Path = VALIDATION_ROOT,
) -> tuple[Path, Path]:
    route = str(result.metadata["route_id"])
    stem = result_cache_stem(str(result.metadata["case_id"]), route, output_root=output_root)
    stem.parent.mkdir(parents=True, exist_ok=True)
    snapshot_z = np.asarray(list(result.snapshot_fields), dtype=float)
    snapshot_stack = np.asarray(
        [result.snapshot_fields[value] for value in snapshot_z], dtype=np.complex64
    )
    array_path = stem.with_suffix(".npz")
    np.savez_compressed(
        array_path,
        x_m=result.x_m,
        y_m=result.y_m,
        z_m=result.z_m,
        xz_intensity=result.xz_intensity,
        yz_intensity=result.yz_intensity,
        axial_trace_raw=result.axial_trace_raw,
        fixed_bucket_power_raw=result.fixed_bucket_power_raw,
        feature_radius_m=result.feature_radius_m,
        feature_width_m=result.feature_width_m,
        dark_core_radius_m=result.dark_core_radius_m,
        feature_valid=result.feature_valid,
        total_plane_power=result.total_plane_power,
        edge_energy_fraction=result.edge_energy_fraction,
        snapshot_z_m=snapshot_z,
        snapshot_fields=snapshot_stack,
    )
    metadata_path = stem.with_suffix(".json")
    _write_json(metadata_path, result.metadata)
    return array_path, metadata_path


def load_final_source_result(
    case_id: str,
    aperture_route: ApertureRoute,
    *,
    output_root: Path = VALIDATION_ROOT,
) -> FinalSourcePropagationResult:
    stem = result_cache_stem(case_id, aperture_route, output_root=output_root)
    with np.load(stem.with_suffix(".npz"), allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    metadata = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
    region_data = metadata["fixed_region"]
    region = FixedRegion(
        case_id=str(region_data["case_id"]),
        inner_radius_m=float(region_data["inner_radius_m"]),
        outer_radius_m=float(region_data["outer_radius_m"]),
        provenance=str(region_data["provenance"]),
    )
    snapshot_fields = {
        float(z): np.asarray(field, dtype=np.complex64)
        for z, field in zip(arrays.pop("snapshot_z_m"), arrays.pop("snapshot_fields"))
    }
    return FinalSourcePropagationResult(
        x_m=arrays["x_m"],
        y_m=arrays["y_m"],
        z_m=arrays["z_m"],
        xz_intensity=arrays["xz_intensity"],
        yz_intensity=arrays["yz_intensity"],
        axial_trace_raw=arrays["axial_trace_raw"],
        fixed_bucket_power_raw=arrays["fixed_bucket_power_raw"],
        feature_radius_m=arrays["feature_radius_m"],
        feature_width_m=arrays["feature_width_m"],
        dark_core_radius_m=arrays["dark_core_radius_m"],
        feature_valid=arrays["feature_valid"].astype(bool),
        total_plane_power=arrays["total_plane_power"],
        edge_energy_fraction=arrays["edge_energy_fraction"],
        snapshot_fields=snapshot_fields,
        metadata=metadata,
        fixed_region=region,
    )
