"""Stage 9B.0 nominal F300 scalar 4F virtual bench.

This module is an opt-in nominal forward model for the declared F300 relay:

    SLM2 plane -> 300 mm -> Lens 1 f=300 mm -> 300 mm
    -> Fourier/pinhole plane -> 300 mm -> Lens 2 f=300 mm
    -> 300 mm -> nominal relay-output plane

It is not bench calibrated. It does not change the existing active CSLM route,
does not mark physical 4F readiness ready, and does not implement a camera
model, inverse correction, AI, nonlinear propagation, or material response.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from vbb_study.equations.fields import make_xy_grid
from vbb_study.equations.holography import (
    TWOPI,
    phase_to_gray,
    quantize_phase_rad,
    spp_phase_rad,
    wrap_phase_rad,
)
from vbb_study.equations.propagation import angular_spectrum_propagate_bl, discrete_power


STAGE = "9B.0"
PROFILE_PATH = Path("configs/hardware/cslm_f300_nominal_4f_profile.json")
MODEL_LABEL = "Nominal F300 4F scenario - not bench calibrated"
CLAIM_BOUNDARY_LABELS = (
    "nominal_4f_forward_model",
    "not_bench_calibrated",
    "not_physical_4f_readiness_ready",
    "not_camera_modelled",
    "not_material_modelled",
)
FINAL_EXPORT_ALLOWED = False
EPS = 1e-30

NOMINAL_COMPONENT_SEQUENCE = (
    "SLM2_phase_plane",
    "SLM2_to_lens1_propagation",
    "lens1_thin_phase_and_pupil",
    "lens1_to_fourier_plane_propagation",
    "fourier_plane_field_pre_stop",
    "fourier_stop_pinhole",
    "fourier_plane_field_post_stop",
    "fourier_plane_to_lens2_propagation",
    "lens2_thin_phase_and_pupil",
    "lens2_to_nominal_relay_output_propagation",
    "nominal_relay_output_plane",
)


@dataclass(frozen=True)
class NominalF300Config:
    """Editable nominal F300 4F numerical scenario."""

    simulation_plane_width_m: float = 0.008
    simulation_grid_size: int = 128
    wavelength_m: float = 1.03e-6
    n_medium: float = 1.0
    input_beam_radius_m: float = 0.0008
    input_beam_decentre_x_m: float = 0.0
    input_beam_decentre_y_m: float = 0.0
    input_beam_ellipticity: float = 1.0
    input_beam_rotation_deg: float = 0.0
    lens_clear_radius_m: float = 0.0032
    pinhole_radius_m: float = 0.00018
    pinhole_offset_x_m: float = 0.00031
    pinhole_offset_y_m: float = 0.0
    command_domain_carrier_cycles_x: float = 8.0
    command_domain_carrier_cycles_y: float = 0.0
    numerical_model_carrier_cycles_x: float = 8.0
    numerical_model_carrier_cycles_y: float = 0.0
    slm2_to_lens1_m: float = 0.300
    lens1_focal_length_m: float = 0.300
    lens1_to_fourier_plane_m: float = 0.300
    fourier_plane_to_lens2_m: float = 0.300
    lens2_focal_length_m: float = 0.300
    lens2_to_nominal_relay_output_m: float = 0.300
    bandlimit: bool = True
    slm_phase_quantisation_levels: int = 256
    relay_output_to_axicon_mode: str = "unknown_not_simulated"
    relay_output_to_axicon_distance_m: float | None = None

    @classmethod
    def fast(cls, **overrides: Any) -> "NominalF300Config":
        base = {
            "simulation_grid_size": 96,
            "simulation_plane_width_m": 0.008,
            "input_beam_radius_m": 0.00075,
            "lens_clear_radius_m": 0.0030,
        }
        base.update(overrides)
        return cls(**base)

    @property
    def dx_m(self) -> float:
        return float(self.simulation_plane_width_m) / int(self.simulation_grid_size)


@dataclass(frozen=True)
class NominalF300Run:
    config: NominalF300Config
    grid: Mapping[str, Any]
    slm1_phase_rad: np.ndarray
    slm2_phase_rad: np.ndarray
    slm2_input_field: np.ndarray
    post_lens1_field: np.ndarray
    fourier_plane_field_pre_stop: np.ndarray
    fourier_stop_transmission: np.ndarray
    fourier_plane_field_post_stop: np.ndarray
    post_lens2_field: np.ndarray
    nominal_relay_output_field: np.ndarray
    component_energy_ledger: tuple[Mapping[str, Any], ...]
    component_manifest: tuple[Mapping[str, Any], ...]
    warnings: tuple[str, ...]
    diagnostics: Mapping[str, Any]
    carrier_coordinate_status: str = "nominal_model_not_bench_calibrated"
    nominal_4f_forward_model: bool = True
    bench_calibrated: bool = False
    physical_4f_readiness: str = "blocked"
    final_export_allowed: bool = FINAL_EXPORT_ALLOWED

    @property
    def outputs(self) -> dict[str, Any]:
        return {
            "slm2_input_field": self.slm2_input_field,
            "post_lens1_field": self.post_lens1_field,
            "fourier_plane_field_pre_stop": self.fourier_plane_field_pre_stop,
            "fourier_stop_transmission": self.fourier_stop_transmission,
            "fourier_plane_field_post_stop": self.fourier_plane_field_post_stop,
            "post_lens2_field": self.post_lens2_field,
            "nominal_relay_output_field": self.nominal_relay_output_field,
            "component_energy_ledger": list(self.component_energy_ledger),
            "component_manifest": list(self.component_manifest),
        }


def load_nominal_f300_profile(path: str | Path = PROFILE_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def config_from_profile(profile: Mapping[str, Any] | None = None) -> NominalF300Config:
    profile = profile or load_nominal_f300_profile()
    geom = profile["known_nominal_geometry"]
    conv = profile["model_convention"]

    def _v(section: Mapping[str, Any], key: str) -> Any:
        return section[key]["value"]

    dec = _v(conv, "input_beam_decentre_m")
    return NominalF300Config(
        simulation_plane_width_m=float(_v(conv, "simulation_plane_width_m")),
        simulation_grid_size=int(_v(conv, "simulation_grid_size")),
        wavelength_m=float(_v(conv, "wavelength_m")),
        input_beam_radius_m=float(_v(conv, "input_beam_radius_m")),
        input_beam_decentre_x_m=float(dec[0]),
        input_beam_decentre_y_m=float(dec[1]),
        input_beam_ellipticity=float(_v(conv, "input_beam_ellipticity")),
        lens_clear_radius_m=float(_v(conv, "lens_clear_radius_m")),
        pinhole_radius_m=float(_v(conv, "pinhole_radius_m")),
        pinhole_offset_x_m=float(_v(conv, "pinhole_offset_x_m")),
        pinhole_offset_y_m=float(_v(conv, "pinhole_offset_y_m")),
        slm2_to_lens1_m=float(_v(geom, "slm2_to_lens1_m")),
        lens1_focal_length_m=float(_v(geom, "lens1_focal_length_m")),
        lens1_to_fourier_plane_m=float(_v(geom, "lens1_to_fourier_plane_m")),
        fourier_plane_to_lens2_m=float(_v(geom, "fourier_plane_to_lens2_m")),
        lens2_focal_length_m=float(_v(geom, "lens2_focal_length_m")),
        lens2_to_nominal_relay_output_m=float(_v(geom, "lens2_to_nominal_relay_output_m")),
    )


def replace_config(config: NominalF300Config, **updates: Any) -> NominalF300Config:
    return replace(config, **updates)


def _grid(config: NominalF300Config) -> dict[str, Any]:
    return make_xy_grid(int(config.simulation_grid_size), config.dx_m)


def _power(field: np.ndarray, config: NominalF300Config) -> float:
    return discrete_power(field, config.dx_m)


def _gaussian_input(grid: Mapping[str, Any], config: NominalF300Config) -> np.ndarray:
    x0 = float(config.input_beam_decentre_x_m)
    y0 = float(config.input_beam_decentre_y_m)
    wx = max(float(config.input_beam_radius_m), EPS)
    wy = wx * max(float(config.input_beam_ellipticity), 0.05)
    rot = np.deg2rad(float(config.input_beam_rotation_deg))
    X = np.asarray(grid["X"], float) - x0
    Y = np.asarray(grid["Y"], float) - y0
    Xr = X * np.cos(rot) + Y * np.sin(rot)
    Yr = -X * np.sin(rot) + Y * np.cos(rot)
    return np.exp(-((Xr / wx) ** 2 + (Yr / wy) ** 2)).astype(complex)


def vortex_phase(grid: Mapping[str, Any], ell: int) -> np.ndarray:
    if int(ell) == 0:
        return np.zeros_like(np.asarray(grid["X"], float))
    return wrap_phase_rad(spp_phase_rad(grid["PHI"], int(ell)))


def carrier_phase(grid: Mapping[str, Any], config: NominalF300Config) -> np.ndarray:
    width = float(config.simulation_plane_width_m)
    X = np.asarray(grid["X"], float)
    Y = np.asarray(grid["Y"], float)
    nx = float(config.numerical_model_carrier_cycles_x)
    ny = float(config.numerical_model_carrier_cycles_y)
    return wrap_phase_rad(TWOPI * (nx * X / width + ny * Y / width))


def thin_lens_phase(grid: Mapping[str, Any], wavelength_m: float, focal_length_m: float) -> np.ndarray:
    k = TWOPI / float(wavelength_m)
    R2 = np.asarray(grid["X"], float) ** 2 + np.asarray(grid["Y"], float) ** 2
    return np.exp(-1j * k * R2 / (2.0 * float(focal_length_m)))


def circular_amplitude(grid: Mapping[str, Any], radius_m: float, x0_m: float = 0.0, y0_m: float = 0.0) -> np.ndarray:
    R = np.hypot(np.asarray(grid["X"], float) - float(x0_m), np.asarray(grid["Y"], float) - float(y0_m))
    return (R <= float(radius_m)).astype(float)


def _prop(field: np.ndarray, grid: Mapping[str, Any], config: NominalF300Config, distance_m: float) -> np.ndarray:
    return angular_spectrum_propagate_bl(
        np.asarray(field, complex),
        dict(grid),
        float(config.wavelength_m),
        float(distance_m),
        n_medium=float(config.n_medium),
        bandlimit=bool(config.bandlimit),
    )


def _field_metrics(field: np.ndarray, config: NominalF300Config) -> dict[str, Any]:
    I = np.abs(np.asarray(field, complex)) ** 2
    grid = _grid(config)
    X = np.asarray(grid["X"], float)
    Y = np.asarray(grid["Y"], float)
    total = float(np.sum(I))
    if total <= EPS:
        cx = cy = sx = sy = 0.0
    else:
        cx = float(np.sum(I * X) / total)
        cy = float(np.sum(I * Y) / total)
        sx = float(np.sqrt(max(np.sum(I * (X - cx) ** 2) / total, 0.0)))
        sy = float(np.sqrt(max(np.sum(I * (Y - cy) ** 2) / total, 0.0)))
    half = 0.5 * float(config.simulation_plane_width_m)
    margin = min(half - abs(cx), half - abs(cy))
    border = _border_power_fraction(I)
    return {
        "total_power_arb": float(total * config.dx_m * config.dx_m),
        "centroid_x_m": cx,
        "centroid_y_m": cy,
        "second_moment_width_x_m": sx,
        "second_moment_width_y_m": sy,
        "field_of_view_margin_m": float(margin),
        "border_power_fraction": float(border),
    }


def _border_power_fraction(intensity: np.ndarray, border_px: int = 2) -> float:
    I = np.asarray(intensity, float)
    total = float(np.sum(I))
    if total <= EPS:
        return 0.0
    b = max(1, int(border_px))
    edge = np.zeros_like(I, dtype=bool)
    edge[:b, :] = True
    edge[-b:, :] = True
    edge[:, :b] = True
    edge[:, -b:] = True
    return float(np.sum(I[edge]) / total)


def _ledger_row(component_id: str, component_type: str, before: float, after: float, note: str) -> dict[str, Any]:
    return {
        "component_id": component_id,
        "component_type": component_type,
        "energy_before_arb_m2": float(before),
        "energy_after_arb_m2": float(after),
        "transmitted_fraction": float(after / max(before, EPS)),
        "note": note,
    }


def _manifest_row(
    component_id: str,
    component_type: str,
    transform_applied: bool,
    distance_m: float | None,
    status: str,
    note: str,
) -> dict[str, Any]:
    return {
        "component_id": component_id,
        "component_type": component_type,
        "transform_applied": bool(transform_applied),
        "distance_m": None if distance_m is None else float(distance_m),
        "model_status": status,
        "downstream_components_affected": list(NOMINAL_COMPONENT_SEQUENCE[
            NOMINAL_COMPONENT_SEQUENCE.index(component_id) + 1 :
        ]) if component_id in NOMINAL_COMPONENT_SEQUENCE else [],
        "note": note,
    }


def run_nominal_f300_4f(
    config: NominalF300Config | None = None,
    *,
    slm1_phase_rad: np.ndarray | None = None,
    input_field: np.ndarray | None = None,
) -> NominalF300Run:
    """Run the nominal F300 scalar 4F model."""
    config = config or NominalF300Config()
    grid = _grid(config)
    slm1_phase = np.zeros((config.simulation_grid_size, config.simulation_grid_size), dtype=float)
    if slm1_phase_rad is not None:
        slm1_phase = wrap_phase_rad(np.asarray(slm1_phase_rad, dtype=float))
    if input_field is None:
        base = _gaussian_input(grid, config)
    else:
        base = np.asarray(input_field, dtype=complex)
    if base.shape != slm1_phase.shape:
        raise ValueError("input_field/slm1_phase shape must match the configured square grid.")

    slm2_input = base * np.exp(1j * slm1_phase)
    slm2_phase = carrier_phase(grid, config)
    field = slm2_input * np.exp(1j * slm2_phase)
    lens1_field = _prop(field, grid, config, config.slm2_to_lens1_m)

    lens_pupil = circular_amplitude(grid, config.lens_clear_radius_m)
    lens1_before = _power(lens1_field, config)
    lens1_after_field = lens1_field * thin_lens_phase(grid, config.wavelength_m, config.lens1_focal_length_m) * lens_pupil
    lens1_after = _power(lens1_after_field, config)
    fourier_pre = _prop(lens1_after_field, grid, config, config.lens1_to_fourier_plane_m)

    stop = circular_amplitude(
        grid,
        config.pinhole_radius_m,
        config.pinhole_offset_x_m,
        config.pinhole_offset_y_m,
    )
    stop_before = _power(fourier_pre, config)
    fourier_post = fourier_pre * stop
    stop_after = _power(fourier_post, config)

    lens2_input = _prop(fourier_post, grid, config, config.fourier_plane_to_lens2_m)
    lens2_before = _power(lens2_input, config)
    lens2_after_field = lens2_input * thin_lens_phase(grid, config.wavelength_m, config.lens2_focal_length_m) * lens_pupil
    lens2_after = _power(lens2_after_field, config)
    relay_output = _prop(lens2_after_field, grid, config, config.lens2_to_nominal_relay_output_m)

    source_power = _power(slm2_input, config)
    after_slm2 = _power(field, config)
    after_prop_lens1 = _power(lens1_field, config)
    fourier_power = _power(fourier_pre, config)
    after_prop_lens2 = _power(lens2_input, config)
    output_power = _power(relay_output, config)
    ledger = (
        _ledger_row("SLM2_phase_plane", "phase_plane", source_power, after_slm2, "carrier phase only; no axicon on SLM2"),
        _ledger_row("SLM2_to_lens1_propagation", "free_space_propagation_segment", after_slm2, after_prop_lens1, "300 mm nominal propagation"),
        _ledger_row("lens1_thin_phase_and_pupil", "thin_lens_with_finite_pupil", lens1_before, lens1_after, "scalar thin-lens phase plus editable nominal pupil"),
        _ledger_row("lens1_to_fourier_plane_propagation", "free_space_propagation_segment", lens1_after, fourier_power, "300 mm nominal propagation"),
        _ledger_row("fourier_stop_pinhole", "fourier_plane_circular_amplitude_stop", stop_before, stop_after, "nominal_stop_parameter; not_measured_bench_geometry"),
        _ledger_row("fourier_plane_to_lens2_propagation", "free_space_propagation_segment", stop_after, after_prop_lens2, "300 mm nominal propagation"),
        _ledger_row("lens2_thin_phase_and_pupil", "thin_lens_with_finite_pupil", lens2_before, lens2_after, "scalar thin-lens phase plus editable nominal pupil"),
        _ledger_row("lens2_to_nominal_relay_output_propagation", "free_space_propagation_segment", lens2_after, output_power, "300 mm nominal propagation"),
    )
    manifest = (
        _manifest_row("SLM2_phase_plane", "phase_plane", True, None, "nominal_4f_forward_model", "SLM2 owns carrier only"),
        _manifest_row("SLM2_to_lens1_propagation", "free_space_propagation_segment", True, config.slm2_to_lens1_m, "nominal_4f_forward_model", "actual distance is nominal bench description"),
        _manifest_row("lens1_thin_phase_and_pupil", "thin_lens_with_finite_pupil", True, None, "nominal_4f_forward_model", "f=300 mm thin scalar lens"),
        _manifest_row("lens1_to_fourier_plane_propagation", "free_space_propagation_segment", True, config.lens1_to_fourier_plane_m, "nominal_4f_forward_model", "propagate to Fourier plane"),
        _manifest_row("fourier_plane_field_pre_stop", "diagnostic_field_plane", False, None, "diagnostic_boundary", "complex field before stop"),
        _manifest_row("fourier_stop_pinhole", "fourier_plane_circular_amplitude_stop", True, None, "nominal_stop_parameter", "applied at Fourier plane, not output crop"),
        _manifest_row("fourier_plane_field_post_stop", "diagnostic_field_plane", False, None, "diagnostic_boundary", "complex field after stop"),
        _manifest_row("fourier_plane_to_lens2_propagation", "free_space_propagation_segment", True, config.fourier_plane_to_lens2_m, "nominal_4f_forward_model", "propagate from Fourier plane to lens 2"),
        _manifest_row("lens2_thin_phase_and_pupil", "thin_lens_with_finite_pupil", True, None, "nominal_4f_forward_model", "f=300 mm thin scalar lens"),
        _manifest_row("lens2_to_nominal_relay_output_propagation", "free_space_propagation_segment", True, config.lens2_to_nominal_relay_output_m, "nominal_4f_forward_model", "propagate to nominal relay output"),
        _manifest_row("nominal_relay_output_plane", "diagnostic_field_plane", False, None, "diagnostic_boundary", "simulation stops here unless explicit axicon handoff scenario is enabled"),
    )
    warnings = tuple(nominal_4f_warnings_from_fields(config, ledger, lens1_field, lens2_input, stop, relay_output))
    diagnostics = {
        "slm2_input_metrics": _field_metrics(slm2_input, config),
        "fourier_plane_pre_stop_metrics": _field_metrics(fourier_pre, config),
        "nominal_relay_output_metrics": _field_metrics(relay_output, config),
        "pinhole_transmitted_fraction": float(stop_after / max(stop_before, EPS)),
        "lens1_pupil_transmitted_fraction": float(lens1_after / max(lens1_before, EPS)),
        "lens2_pupil_transmitted_fraction": float(lens2_after / max(lens2_before, EPS)),
    }
    return NominalF300Run(
        config=config,
        grid=grid,
        slm1_phase_rad=slm1_phase,
        slm2_phase_rad=slm2_phase,
        slm2_input_field=slm2_input,
        post_lens1_field=lens1_after_field,
        fourier_plane_field_pre_stop=fourier_pre,
        fourier_stop_transmission=stop,
        fourier_plane_field_post_stop=fourier_post,
        post_lens2_field=lens2_after_field,
        nominal_relay_output_field=relay_output,
        component_energy_ledger=ledger,
        component_manifest=manifest,
        warnings=warnings,
        diagnostics=diagnostics,
    )


def nominal_4f_warnings_from_fields(
    config: NominalF300Config,
    ledger: Sequence[Mapping[str, Any]],
    lens1_input_field: np.ndarray,
    lens2_input_field: np.ndarray,
    stop: np.ndarray,
    relay_output: np.ndarray,
) -> list[str]:
    warnings: list[str] = []
    carrier_cycles = max(abs(config.numerical_model_carrier_cycles_x), abs(config.numerical_model_carrier_cycles_y))
    pixels_per_cycle = float("inf") if carrier_cycles <= 0 else int(config.simulation_grid_size) / carrier_cycles
    if pixels_per_cycle < 8.0:
        warnings.append("carrier undersampling: fewer than 8 pixels per numerical carrier cycle")
    for label, field in (("lens1", lens1_input_field), ("lens2", lens2_input_field)):
        before = _power(field, config)
        after = _power(field * circular_amplitude(_grid(config), config.lens_clear_radius_m), config)
        if after / max(before, EPS) < 0.98:
            warnings.append(f"beam clipping at {label} pupil")
    half = 0.5 * config.simulation_plane_width_m
    if abs(config.pinhole_offset_x_m) + config.pinhole_radius_m > half or abs(config.pinhole_offset_y_m) + config.pinhole_radius_m > half:
        warnings.append("pinhole lying partly outside the simulation FOV")
    if _border_power_fraction(np.abs(relay_output) ** 2) > 0.02:
        warnings.append("output clipping: relay output has nontrivial border power")
    for row in ledger:
        ctype = str(row["component_type"])
        if ("pupil" in ctype or "stop" in ctype) and float(row["energy_after_arb_m2"]) > float(row["energy_before_arb_m2"]) * (1.0 + 1e-9):
            warnings.append(f"power increase after passive stop/pupil at {row['component_id']}")
    dx = config.dx_m
    if config.pinhole_radius_m < 2.0 * dx:
        warnings.append("insufficient Fourier-plane sampling: pinhole radius spans fewer than two samples")
    if not np.any(stop > 0):
        warnings.append("pinhole transmits no sampled pixels")
    return warnings


def nominal_4f_sanity_report(run: NominalF300Run) -> dict[str, Any]:
    cfg = run.config
    carrier_cycles = max(abs(cfg.numerical_model_carrier_cycles_x), abs(cfg.numerical_model_carrier_cycles_y))
    pixels_per_cycle = float("inf") if carrier_cycles <= 0 else int(cfg.simulation_grid_size) / carrier_cycles
    input_i = np.abs(run.slm2_input_field) ** 2
    output_i = np.abs(run.nominal_relay_output_field) ** 2
    flipped_input = np.flipud(np.fliplr(input_i))
    denom = np.linalg.norm(flipped_input.ravel()) * np.linalg.norm(output_i.ravel()) + EPS
    relay_corr = float(np.dot(flipped_input.ravel(), output_i.ravel()) / denom)
    return {
        "grid_size": int(cfg.simulation_grid_size),
        "simulation_plane_width_m": float(cfg.simulation_plane_width_m),
        "sampling_pitch_m": float(cfg.dx_m),
        "wavelength_m": float(cfg.wavelength_m),
        "maximum_spatial_frequency_content_cpm": float(carrier_cycles / max(cfg.simulation_plane_width_m, EPS)),
        "carrier_cycles_across_model_width": [float(cfg.numerical_model_carrier_cycles_x), float(cfg.numerical_model_carrier_cycles_y)],
        "pixels_per_carrier_cycle": float(pixels_per_cycle),
        "lens_pupil_radius_m": float(cfg.lens_clear_radius_m),
        "pinhole_radius_m": float(cfg.pinhole_radius_m),
        "propagation_distances_m": {
            "slm2_to_lens1_m": float(cfg.slm2_to_lens1_m),
            "lens1_to_fourier_plane_m": float(cfg.lens1_to_fourier_plane_m),
            "fourier_plane_to_lens2_m": float(cfg.fourier_plane_to_lens2_m),
            "lens2_to_nominal_relay_output_m": float(cfg.lens2_to_nominal_relay_output_m),
        },
        "energy_ledger": list(run.component_energy_ledger),
        "warnings": list(run.warnings),
        "open_stop_relay_intensity_correlation_with_inverted_input": relay_corr,
        "carrier_coordinate_status": run.carrier_coordinate_status,
        "final_export_allowed": bool(run.final_export_allowed),
    }


def fourier_plane_centroid_m(run: NominalF300Run) -> tuple[float, float]:
    m = _field_metrics(run.fourier_plane_field_pre_stop, run.config)
    return float(m["centroid_x_m"]), float(m["centroid_y_m"])


def phase_export_payload(phase_rad: np.ndarray, *, mask_id: str, slm_id: str, config: NominalF300Config) -> dict[str, Any]:
    bits = max(1, int(round(np.log2(int(config.slm_phase_quantisation_levels)))))
    wrapped = wrap_phase_rad(phase_rad)
    gray = phase_to_gray(wrapped, bits=bits)
    return {
        "phase_rad": wrapped,
        "quantised_rad": quantize_phase_rad(wrapped, bits),
        "gray": gray,
        "metadata": {
            "mask_id": mask_id,
            "slm_id": slm_id,
            "stage": STAGE,
            "simulation_status": "nominal_unvalidated",
            "physical_4f_readiness": "blocked",
            "carrier_coordinate_status": "nominal_model_not_bench_calibrated",
            "contains_axicon": False,
            "final_export_allowed": False,
        },
    }


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})
    return out


def write_json(path: str | Path, obj: Any) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")
    return out


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def run_to_manifest(run: NominalF300Run) -> dict[str, Any]:
    return {
        "stage": STAGE,
        "model_label": MODEL_LABEL,
        "claim_boundary_labels": list(CLAIM_BOUNDARY_LABELS),
        "simulation_status": "nominal_unvalidated",
        "physical_4f_readiness": run.physical_4f_readiness,
        "camera_validation": "absent",
        "material_prediction": "absent",
        "nominal_4f_forward_model": run.nominal_4f_forward_model,
        "bench_calibrated": run.bench_calibrated,
        "carrier_coordinate_status": run.carrier_coordinate_status,
        "component_sequence": list(NOMINAL_COMPONENT_SEQUENCE),
        "component_manifest": list(run.component_manifest),
        "component_energy_ledger": list(run.component_energy_ledger),
        "warnings": list(run.warnings),
        "config": asdict(run.config),
        "final_export_allowed": bool(run.final_export_allowed),
    }


def plot_component_sequence(
    run: NominalF300Run | None = None,
    output_path: str | Path = "outputs/figures/digital_twin/stage9b0_nominal_f300_4f_component_sequence.png",
):
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    run = run or run_nominal_f300_4f(NominalF300Config.fast())
    x_mm = np.asarray(run.grid["x"], float) * 1e3
    extent = (float(x_mm.min()), float(x_mm.max()), float(x_mm.min()), float(x_mm.max()))
    panels = [
        (np.abs(run.slm2_input_field) ** 2, "SLM2 input intensity"),
        (np.abs(run.post_lens1_field) ** 2, "post Lens 1 pupil"),
        (np.abs(run.fourier_plane_field_pre_stop) ** 2, "Fourier plane pre-stop"),
        (run.fourier_stop_transmission, "Fourier stop transmission"),
        (np.abs(run.fourier_plane_field_post_stop) ** 2, "Fourier plane post-stop"),
        (np.abs(run.post_lens2_field) ** 2, "post Lens 2 pupil"),
        (np.abs(run.nominal_relay_output_field) ** 2, "nominal relay output"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(14.5, 11.0), facecolor="white")
    fig.suptitle(f"{MODEL_LABEL}\ncomponent-owned scalar propagation, lenses, Fourier stop, relay output",
                 x=0.04, y=0.98, ha="left", va="top", fontsize=14, fontweight="bold")
    for ax, (arr, title) in zip(axes.ravel(), panels):
        im = ax.imshow(arr, origin="lower", extent=extent, cmap="viridis")
        ax.set_title(title, fontsize=9.5, fontweight="bold")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        fig.colorbar(im, ax=ax, fraction=0.046)
    axes.ravel()[7].axis("off")
    lines = [
        "Energy ledger",
        *[
            f"{row['component_id']}: {row['transmitted_fraction']:.3f}"
            for row in run.component_energy_ledger
        ],
    ]
    axes.ravel()[7].text(0.0, 1.0, "\n".join(lines), va="top", family="monospace", fontsize=7.2)
    axes.ravel()[8].axis("off")
    axes.ravel()[8].text(
        0.0,
        1.0,
        "Boundary\n"
        "nominal_4f_forward_model=True\n"
        "bench_calibrated=False\n"
        "physical_4f_readiness=blocked\n"
        "camera/material models absent\n"
        "final_export_allowed=False",
        va="top",
        family="monospace",
        fontsize=8.8,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=165, bbox_inches="tight", metadata={"Title": MODEL_LABEL, "final_export_allowed": "False"})
    plt.close(fig)
    return out


__all__ = [
    "CLAIM_BOUNDARY_LABELS",
    "FINAL_EXPORT_ALLOWED",
    "MODEL_LABEL",
    "NOMINAL_COMPONENT_SEQUENCE",
    "NominalF300Config",
    "NominalF300Run",
    "carrier_phase",
    "circular_amplitude",
    "config_from_profile",
    "fourier_plane_centroid_m",
    "load_nominal_f300_profile",
    "nominal_4f_sanity_report",
    "phase_export_payload",
    "plot_component_sequence",
    "replace_config",
    "run_nominal_f300_4f",
    "run_to_manifest",
    "thin_lens_phase",
    "vortex_phase",
    "write_csv",
    "write_json",
]


if __name__ == "__main__":
    plot_component_sequence()
