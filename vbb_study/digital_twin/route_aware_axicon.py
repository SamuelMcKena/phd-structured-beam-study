"""Stage 8C.3R.3 route-aware physical-axicon alignment diagnostics.

This module keeps the Stage 8C.3R study in free space (``n = 1.0``) while
making perturbation placement explicit.  The physical-axicon route is represented
as an ordered graph of named planes, propagation segments, and thin elements.
The same perturbation type can be injected at different represented planes, and
therefore passes through different downstream segments/elements before reaching
the free-space reference plane.

All default distances are diagnostic demo geometry, not measured bench
distances.  Outputs are optical/fluence diagnostics only;
``final_export_allowed=False``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
from scipy.ndimage import shift as nd_shift

import matplotlib
matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt  # noqa: E402

from vbb_study.equations.fields import fft2c, make_xy_grid
from vbb_study.equations.propagation import make_bl_asm_propagator
from vbb_study.digital_twin.component_plane_states import (
    ComponentPlaneState,
    PropagatedFieldStack,
    field_power,
)
from vbb_study.digital_twin.annular_axis_tracking import estimate_annular_axis
from vbb_study.digital_twin.component_plane_metrics import (
    DIAGNOSTIC_SWEEP_LABEL,
    classify_translation_vs_deformation,
    compute_axis_tracking,
    compute_energy_throughput,
    stack_to_fluence,
)

_UM = 1e-6
_MM = 1e-3

FINAL_EXPORT_ALLOWED = False
MODEL_STATUS = "optical_prediction"
DIAGNOSTIC_GEOMETRY_NOTE = (
    "diagnostic demo geometry only; no actual laboratory bench distance is assumed"
)

PHYSICAL_LOCATIONS: tuple[str, ...] = (
    "source_plane",
    "after_beam_conditioning",
    "before_SLM1",
    "after_SLM1",
    "before_SLM2",
    "after_SLM2",
    "before_physical_axicon",
    "physical_axicon_plane",
    "after_physical_axicon",
    "fourier_plane",
    "after_fourier_filter",
    "relay_plane",
    "objective_pupil_plane",
    "after_objective",
    "free_space_reference_plane",
    "diagnostic_only",
    "not_represented_by_current_engine",
)

REPRESENTED_PHYSICAL_AXICON_LOCATIONS: tuple[str, ...] = (
    "source_plane",
    "after_beam_conditioning",
    "before_physical_axicon",
    "physical_axicon_plane",
    "after_physical_axicon",
    "after_objective",
    "free_space_reference_plane",
)


@dataclass(frozen=True)
class RouteGraphNode:
    name: str
    kind: str  # plane | propagation_segment | element | reference
    physical_location: str
    distance_mm: float = 0.0
    represented: bool = True
    note: str = ""


@dataclass(frozen=True)
class RoutePerturbationRecord:
    parameter_name: str
    perturbation_type: str
    magnitude: float | str
    units: str
    enabled: bool
    injection_location: str
    route: str
    implementation_plane: str
    downstream_elements_affected: tuple[str, ...]
    status: str
    note: str = ""

    @property
    def physical_location(self) -> str:
        return self.injection_location

    @property
    def classification(self) -> str:
        return self.status

    def as_dict(self) -> dict[str, Any]:
        return {
            "parameter_name": self.parameter_name,
            "perturbation_type": self.perturbation_type,
            "magnitude": self.magnitude,
            "units": self.units,
            "enabled": bool(self.enabled),
            "injection_location": self.injection_location,
            "route": self.route,
            "implementation_plane": self.implementation_plane,
            "downstream_elements_affected": list(self.downstream_elements_affected),
            "active / warning-only / future status": self.status,
            "classification": self.status,
            "note": self.note,
        }


@dataclass(frozen=True)
class RouteAwareAxiconConfig:
    """Editable route-aware physical-axicon diagnostic geometry."""

    route_mode: str = "physical_axicon"
    wavelength_nm: float = 1030.0
    n_medium: float = 1.0
    grid_N: int = 224
    dx_um: float = 0.5
    n_z: int = 36
    input_pulse_energy_uJ: float = 95.76

    # Diagnostic demo geometry; not actual bench distances.
    pre_axicon_distance_mm: float = 0.80
    post_axicon_distance_mm: float = 0.225
    axicon_to_objective_distance_mm: float = 0.150
    objective_to_reference_distance_mm: float = 0.075

    input_beam_radius_um: float = 22.0
    input_beam_ellipticity: float = 1.0
    input_beam_rotation_deg: float = 0.0
    input_aperture_radius_um: float = 48.0

    # Correct C3R.3 perturbation controls: type + magnitude + injection location.
    field_tilt_x_mrad: float = 0.0
    field_tilt_y_mrad: float = 0.0
    field_tilt_location: str = "source_plane"
    beam_decentre_x_um: float = 0.0
    beam_decentre_y_um: float = 0.0
    beam_decentre_location: str = "source_plane"

    # Compatibility aliases from the first C3R.3 draft / older C3R pipeline.
    # These are normalised into the generic location-aware controls.
    input_beam_decentre_x_um: float = 0.0
    input_beam_decentre_y_um: float = 0.0
    input_beam_tilt_x_mrad: float = 0.0
    input_beam_tilt_y_mrad: float = 0.0

    # Explicit decentre at the axicon input plane, distinct from source decentre.
    axicon_input_beam_decentre_x_um: float = 0.0
    axicon_input_beam_decentre_y_um: float = 0.0

    physical_axicon_centre_x_um: float = 0.0
    physical_axicon_centre_y_um: float = 0.0
    physical_axicon_clear_aperture_radius_um: float = 42.0
    axicon_cone_parameter: float = 1.05  # rad / um
    physical_axicon_mechanical_tilt_x_mrad: float = 0.0
    physical_axicon_mechanical_tilt_y_mrad: float = 0.0

    enable_post_axicon_steering_test: bool = False
    post_axicon_steering_x_mrad: float = 0.0
    post_axicon_steering_y_mrad: float = 0.0
    bandlimit: bool = True

    @classmethod
    def fast(cls, **overrides: Any) -> "RouteAwareAxiconConfig":
        base = dict(grid_N=160, dx_um=0.6, n_z=24, input_beam_radius_um=20.0)
        base.update(overrides)
        return cls(**base)

    @property
    def wavelength_m(self) -> float:
        return float(self.wavelength_nm) * 1e-9

    @property
    def dx_m(self) -> float:
        return float(self.dx_um) * _UM

    @property
    def k_medium_rad_per_m(self) -> float:
        return 2.0 * np.pi * float(self.n_medium) / self.wavelength_m

    @property
    def pre_axicon_distance_um(self) -> float:
        return float(self.pre_axicon_distance_mm) * 1000.0

    @property
    def post_axicon_distance_um(self) -> float:
        return float(self.post_axicon_distance_mm) * 1000.0

    @property
    def axicon_to_objective_distance_um(self) -> float:
        return float(self.axicon_to_objective_distance_mm) * 1000.0

    @property
    def objective_to_reference_distance_um(self) -> float:
        return float(self.objective_to_reference_distance_mm) * 1000.0


@dataclass(frozen=True)
class RouteAwareAxiconRun:
    config: RouteAwareAxiconConfig
    route_mode: str
    source_state: ComponentPlaneState
    axicon_incident_state: ComponentPlaneState
    physical_axicon_state: ComponentPlaneState
    reference_plane_state: ComponentPlaneState
    propagated_stack: PropagatedFieldStack
    axicon_incidence_metrics: Mapping[str, Any]
    perturbation_records: tuple[RoutePerturbationRecord, ...]
    warnings: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    final_export_allowed: bool = FINAL_EXPORT_ALLOWED
    model_status: str = MODEL_STATUS


def _with_overrides(config: RouteAwareAxiconConfig, controls: Mapping[str, Any] | None) -> RouteAwareAxiconConfig:
    controls = dict(controls or {})
    if not controls:
        return _normalise_location_controls(config, {})
    valid = {f.name for f in RouteAwareAxiconConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    updates = {k: v for k, v in controls.items() if k in valid}
    return _normalise_location_controls(replace(config, **updates), controls)


def _valid_location(location: str) -> str:
    loc = str(location)
    if loc == "pre_physical_axicon":
        loc = "before_physical_axicon"
    if loc == "post_physical_axicon":
        loc = "after_physical_axicon"
    if loc == "post_objective_reference_plane":
        loc = "free_space_reference_plane"
    if loc not in PHYSICAL_LOCATIONS:
        raise ValueError(f"Unsupported perturbation injection location: {location!r}.")
    return loc


def _normalise_location_controls(
    config: RouteAwareAxiconConfig,
    controls: Mapping[str, Any],
) -> RouteAwareAxiconConfig:
    updates: dict[str, Any] = {
        "field_tilt_location": _valid_location(config.field_tilt_location),
        "beam_decentre_location": _valid_location(config.beam_decentre_location),
    }
    # Keep the editable total post-axicon distance coherent with the represented
    # downstream split unless the caller explicitly supplied segment distances.
    if "post_axicon_distance_mm" in controls and "axicon_to_objective_distance_mm" not in controls and "objective_to_reference_distance_mm" not in controls:
        post = float(config.post_axicon_distance_mm)
        updates["axicon_to_objective_distance_mm"] = 2.0 * post / 3.0
        updates["objective_to_reference_distance_mm"] = post / 3.0
    else:
        post = float(config.axicon_to_objective_distance_mm) + float(config.objective_to_reference_distance_mm)
        updates["post_axicon_distance_mm"] = post

    generic_tilt_supplied = (
        "field_tilt_x_mrad" in controls or "field_tilt_y_mrad" in controls or "field_tilt_location" in controls
    )
    if not generic_tilt_supplied:
        if bool(config.enable_post_axicon_steering_test):
            updates["field_tilt_x_mrad"] = float(config.post_axicon_steering_x_mrad)
            updates["field_tilt_y_mrad"] = float(config.post_axicon_steering_y_mrad)
            updates["field_tilt_location"] = "after_physical_axicon"
        elif abs(config.input_beam_tilt_x_mrad) > 0 or abs(config.input_beam_tilt_y_mrad) > 0:
            updates["field_tilt_x_mrad"] = float(config.input_beam_tilt_x_mrad)
            updates["field_tilt_y_mrad"] = float(config.input_beam_tilt_y_mrad)
            updates["field_tilt_location"] = "source_plane"

    generic_decentre_supplied = (
        "beam_decentre_x_um" in controls or "beam_decentre_y_um" in controls or "beam_decentre_location" in controls
    )
    if not generic_decentre_supplied:
        if abs(config.axicon_input_beam_decentre_x_um) > 0 or abs(config.axicon_input_beam_decentre_y_um) > 0:
            updates["beam_decentre_x_um"] = float(config.axicon_input_beam_decentre_x_um)
            updates["beam_decentre_y_um"] = float(config.axicon_input_beam_decentre_y_um)
            updates["beam_decentre_location"] = "before_physical_axicon"
        elif abs(config.input_beam_decentre_x_um) > 0 or abs(config.input_beam_decentre_y_um) > 0:
            updates["beam_decentre_x_um"] = float(config.input_beam_decentre_x_um)
            updates["beam_decentre_y_um"] = float(config.input_beam_decentre_y_um)
            updates["beam_decentre_location"] = "source_plane"
    updates["field_tilt_location"] = _valid_location(updates.get("field_tilt_location", config.field_tilt_location))
    updates["beam_decentre_location"] = _valid_location(updates.get("beam_decentre_location", config.beam_decentre_location))
    return replace(config, **updates)


def _grid(config: RouteAwareAxiconConfig) -> dict[str, Any]:
    return make_xy_grid(int(config.grid_N), float(config.dx_m))


def _xy_um(grid: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(grid["x"], float) / _UM
    return x, x.copy()


def physical_axicon_route_graph(config: RouteAwareAxiconConfig) -> tuple[RouteGraphNode, ...]:
    """Represented route graph for the physical-axicon diagnostic path."""
    return (
        RouteGraphNode("source field", "plane", "source_plane", note="complex source/input field"),
        RouteGraphNode("beam conditioning", "element", "after_beam_conditioning",
                       note="ellipticity and source/input aperture"),
        RouteGraphNode("source to axicon", "propagation_segment", "before_physical_axicon",
                       distance_mm=float(config.pre_axicon_distance_mm),
                       note=DIAGNOSTIC_GEOMETRY_NOTE),
        RouteGraphNode("physical axicon", "element", "physical_axicon_plane",
                       note="fixed thin axicon phase and clear aperture"),
        RouteGraphNode("after axicon", "plane", "after_physical_axicon",
                       note="post-axicon field immediately after thin element"),
        RouteGraphNode("axicon to after-objective", "propagation_segment", "after_objective",
                       distance_mm=float(config.axicon_to_objective_distance_mm),
                       note=DIAGNOSTIC_GEOMETRY_NOTE),
        RouteGraphNode("after objective", "plane", "after_objective",
                       note="represented downstream steering plane; no objective model is added"),
        RouteGraphNode("after-objective to reference", "propagation_segment", "free_space_reference_plane",
                       distance_mm=float(config.objective_to_reference_distance_mm),
                       note=DIAGNOSTIC_GEOMETRY_NOTE),
        RouteGraphNode("free-space reference", "reference", "free_space_reference_plane",
                       note="post-objective/reference plane in air; no material model"),
    )


def _route_location_order() -> list[str]:
    return list(REPRESENTED_PHYSICAL_AXICON_LOCATIONS)


def _downstream_elements(location: str) -> tuple[str, ...]:
    loc = _valid_location(location)
    order = _route_location_order()
    if loc not in order:
        return ()
    downstream = order[order.index(loc) + 1:]
    elements = []
    for item in downstream:
        if item in {"physical_axicon_plane", "after_physical_axicon", "after_objective", "free_space_reference_plane"}:
            elements.append(item)
    return tuple(elements)


def _location_is_upstream_of_axicon(location: str) -> bool:
    return _valid_location(location) in {"source_plane", "after_beam_conditioning", "before_physical_axicon"}


def _distance_from_location_to_axicon_um(location: str, config: RouteAwareAxiconConfig) -> float:
    loc = _valid_location(location)
    if loc in {"source_plane", "after_beam_conditioning"}:
        return float(config.pre_axicon_distance_um)
    if loc == "before_physical_axicon":
        return 0.0
    return 0.0


def _phase_ramp(field: np.ndarray, grid: Mapping[str, Any], config: RouteAwareAxiconConfig,
                tx_mrad: float, ty_mrad: float) -> np.ndarray:
    k = config.k_medium_rad_per_m
    kx = k * np.sin(float(tx_mrad) * 1e-3)
    ky = k * np.sin(float(ty_mrad) * 1e-3)
    return field * np.exp(1j * (kx * np.asarray(grid["X"]) + ky * np.asarray(grid["Y"])))


def _circular_mask(grid: Mapping[str, Any], radius_um: float, cx_um: float = 0.0, cy_um: float = 0.0) -> np.ndarray:
    r = np.hypot(np.asarray(grid["X"]) - float(cx_um) * _UM,
                 np.asarray(grid["Y"]) - float(cy_um) * _UM)
    return (r <= float(radius_um) * _UM).astype(float)


def _centroid(field: np.ndarray, x_um: np.ndarray, y_um: np.ndarray) -> tuple[float, float]:
    I = np.abs(field) ** 2
    s = float(np.sum(I))
    if s <= 0:
        return 0.0, 0.0
    X, Y = np.meshgrid(x_um, y_um)
    return float(np.sum(I * X) / s), float(np.sum(I * Y) / s)


def _beam_radii(field: np.ndarray, x_um: np.ndarray, y_um: np.ndarray,
                cx: float, cy: float) -> tuple[float, float, float]:
    I = np.abs(field) ** 2
    s = float(np.sum(I))
    if s <= 0:
        return 0.0, 0.0, 1.0
    X, Y = np.meshgrid(x_um, y_um)
    rx = float(2.0 * np.sqrt(np.sum(I * (X - cx) ** 2) / s))
    ry = float(2.0 * np.sqrt(np.sum(I * (Y - cy) ** 2) / s))
    ell = float(rx / max(ry, 1e-12))
    return rx, ry, ell


def _spectral_angle_mrad(field: np.ndarray, grid: Mapping[str, Any],
                         config: RouteAwareAxiconConfig) -> tuple[float, float]:
    spec = np.abs(fft2c(field)) ** 2
    total = float(np.sum(spec))
    if total <= 0:
        return 0.0, 0.0
    kx = 2.0 * np.pi * np.asarray(grid["FX"])
    ky = 2.0 * np.pi * np.asarray(grid["FY"])
    mx = float(np.sum(spec * kx) / total)
    my = float(np.sum(spec * ky) / total)
    k = config.k_medium_rad_per_m
    kz = np.sqrt(max(k * k - mx * mx - my * my, 1e-30))
    return float(np.arctan2(mx, kz) * 1000.0), float(np.arctan2(my, kz) * 1000.0)


def _field_of_view_margin_um(field: np.ndarray, x_um: np.ndarray, y_um: np.ndarray) -> float:
    cx, cy = _centroid(field, x_um, y_um)
    rx, ry, _ = _beam_radii(field, x_um, y_um, cx, cy)
    mx = min(cx - float(x_um.min()), float(x_um.max()) - cx) - 1.5 * rx
    my = min(cy - float(y_um.min()), float(y_um.max()) - cy) - 1.5 * ry
    return float(min(mx, my))


def _shift_field_um(field: np.ndarray, dx_um: float, dy_um: float, pixel_um: float) -> np.ndarray:
    if abs(dx_um) + abs(dy_um) <= 0:
        return field
    return nd_shift(
        field,
        shift=(float(dy_um) / float(pixel_um), float(dx_um) / float(pixel_um)),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )


def _make_source_field(grid: Mapping[str, Any], config: RouteAwareAxiconConfig) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(grid["X"], float) / _UM
    Y = np.asarray(grid["Y"], float) / _UM
    ell = max(float(config.input_beam_ellipticity), 1e-6)
    wx = float(config.input_beam_radius_um) * np.sqrt(ell)
    wy = float(config.input_beam_radius_um) / np.sqrt(ell)
    rot = np.deg2rad(float(config.input_beam_rotation_deg))
    Xr = X * np.cos(rot) + Y * np.sin(rot)
    Yr = -X * np.sin(rot) + Y * np.cos(rot)
    amp0 = np.exp(-((Xr / max(wx, 1e-9)) ** 2 + (Yr / max(wy, 1e-9)) ** 2))
    aperture = _circular_mask(grid, float(config.input_aperture_radius_um), 0.0, 0.0)
    field = amp0.astype(complex) * aperture
    return field, amp0.astype(complex)


def _apply_location_perturbations(
    field: np.ndarray,
    location: str,
    grid: Mapping[str, Any],
    config: RouteAwareAxiconConfig,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Apply generic field perturbations injected at a represented route location."""
    loc = _valid_location(location)
    out = field
    applied: list[str] = []
    if _valid_location(config.beam_decentre_location) == loc and (
        abs(config.beam_decentre_x_um) > 0 or abs(config.beam_decentre_y_um) > 0
    ):
        out = _shift_field_um(out, config.beam_decentre_x_um, config.beam_decentre_y_um, config.dx_um)
        applied.append(f"beam_decentre_at_{loc}")
    if _valid_location(config.field_tilt_location) == loc and (
        abs(config.field_tilt_x_mrad) > 0 or abs(config.field_tilt_y_mrad) > 0
    ):
        out = _phase_ramp(out, grid, config, config.field_tilt_x_mrad, config.field_tilt_y_mrad)
        label = "post_axicon_steering_test" if loc == "after_physical_axicon" else f"field_tilt_at_{loc}"
        applied.append(label)
    return out, tuple(applied)


def physical_axicon_transmission(grid: Mapping[str, Any], config: RouteAwareAxiconConfig) -> np.ndarray:
    X = np.asarray(grid["X"], float) / _UM
    Y = np.asarray(grid["Y"], float) / _UM
    xa = float(config.physical_axicon_centre_x_um)
    ya = float(config.physical_axicon_centre_y_um)
    r = np.hypot(X - xa, Y - ya)
    aperture = (r <= float(config.physical_axicon_clear_aperture_radius_um)).astype(float)
    phase = -float(config.axicon_cone_parameter) * r
    return aperture * np.exp(1j * phase)


def build_route_perturbation_records(config: RouteAwareAxiconConfig) -> tuple[RoutePerturbationRecord, ...]:
    route = str(config.route_mode)
    tilt_loc = _valid_location(config.field_tilt_location)
    dec_loc = _valid_location(config.beam_decentre_location)
    tilt_status = "post_axicon_steering_test" if tilt_loc == "after_physical_axicon" else "physics_active"
    if tilt_loc == "free_space_reference_plane":
        tilt_status = "diagnostic_only"
    records = [
        RoutePerturbationRecord(
            "field_tilt_x_mrad", "field_tilt", float(config.field_tilt_x_mrad), "mrad",
            abs(config.field_tilt_x_mrad) > 0, tilt_loc, route, tilt_loc,
            _downstream_elements(tilt_loc), tilt_status,
            "generic field tilt injected at the declared represented route location",
        ),
        RoutePerturbationRecord(
            "field_tilt_y_mrad", "field_tilt", float(config.field_tilt_y_mrad), "mrad",
            abs(config.field_tilt_y_mrad) > 0, tilt_loc, route, tilt_loc,
            _downstream_elements(tilt_loc), tilt_status,
            "generic field tilt injected at the declared represented route location",
        ),
        RoutePerturbationRecord(
            "beam_decentre_x_um", "beam_decentre", float(config.beam_decentre_x_um), "um",
            abs(config.beam_decentre_x_um) > 0, dec_loc, route, dec_loc,
            _downstream_elements(dec_loc), "physics_active",
            "generic transverse field shift injected at the declared route location",
        ),
        RoutePerturbationRecord(
            "beam_decentre_y_um", "beam_decentre", float(config.beam_decentre_y_um), "um",
            abs(config.beam_decentre_y_um) > 0, dec_loc, route, dec_loc,
            _downstream_elements(dec_loc), "physics_active",
            "generic transverse field shift injected at the declared route location",
        ),
        RoutePerturbationRecord(
            "input_beam_ellipticity", "beam_shape", float(config.input_beam_ellipticity), "ratio",
            abs(config.input_beam_ellipticity - 1.0) > 1e-12, "source_plane", route,
            "source_plane", _downstream_elements("source_plane"), "physics_active",
            "source-plane beam conditioning",
        ),
        RoutePerturbationRecord(
            "input_aperture_radius_um", "aperture", float(config.input_aperture_radius_um), "um",
            True, "after_beam_conditioning", route, "after_beam_conditioning",
            _downstream_elements("after_beam_conditioning"), "physics_active",
            "source/input aperture; default is diagnostic demo geometry",
        ),
        RoutePerturbationRecord(
            "physical_axicon_centre_x_um", "mechanical_lateral_offset",
            float(config.physical_axicon_centre_x_um), "um",
            abs(config.physical_axicon_centre_x_um) > 0, "physical_axicon_plane", route,
            "physical_axicon_plane", _downstream_elements("physical_axicon_plane"), "physics_active",
            "mechanical lateral axicon displacement",
        ),
        RoutePerturbationRecord(
            "physical_axicon_centre_y_um", "mechanical_lateral_offset",
            float(config.physical_axicon_centre_y_um), "um",
            abs(config.physical_axicon_centre_y_um) > 0, "physical_axicon_plane", route,
            "physical_axicon_plane", _downstream_elements("physical_axicon_plane"), "physics_active",
            "mechanical lateral axicon displacement",
        ),
        RoutePerturbationRecord(
            "physical_axicon_clear_aperture_radius_um", "aperture",
            float(config.physical_axicon_clear_aperture_radius_um), "um",
            True, "physical_axicon_plane", route, "physical_axicon_plane",
            _downstream_elements("physical_axicon_plane"), "physics_active",
            "physical axicon clear aperture",
        ),
        RoutePerturbationRecord(
            "axicon_cone_parameter", "phase_element", float(config.axicon_cone_parameter), "rad/um",
            True, "physical_axicon_plane", route, "physical_axicon_plane",
            _downstream_elements("physical_axicon_plane"), "physics_active",
            "fixed thin physical-axicon phase",
        ),
        RoutePerturbationRecord(
            "physical_axicon_mechanical_tilt_x_mrad", "physical_axicon_mechanical_tilt",
            float(config.physical_axicon_mechanical_tilt_x_mrad), "mrad",
            abs(config.physical_axicon_mechanical_tilt_x_mrad) > 0, "physical_axicon_plane", route,
            "not_represented_by_current_engine", _downstream_elements("physical_axicon_plane"),
            "future_not_implemented",
            "not silently represented as generic field tilt; thin-element approximation not yet implemented",
        ),
        RoutePerturbationRecord(
            "physical_axicon_mechanical_tilt_y_mrad", "physical_axicon_mechanical_tilt",
            float(config.physical_axicon_mechanical_tilt_y_mrad), "mrad",
            abs(config.physical_axicon_mechanical_tilt_y_mrad) > 0, "physical_axicon_plane", route,
            "not_represented_by_current_engine", _downstream_elements("physical_axicon_plane"),
            "future_not_implemented",
            "not silently represented as generic field tilt; thin-element approximation not yet implemented",
        ),
    ]
    return tuple(records)


def holographic_slm_route_declarations() -> tuple[RoutePerturbationRecord, ...]:
    return (
        RoutePerturbationRecord(
            "fourier_plane_filter", "fourier_filter", "not configured", "n/a", False,
            "fourier_plane", "holographic_slm", "not_represented_by_current_engine",
            ("after_fourier_filter", "relay_plane", "objective_pupil_plane"), "warning_only",
            "no explicit 4F Fourier plane in current engine",
        ),
        RoutePerturbationRecord(
            "relay_plane_errors", "relay_error", "not configured", "n/a", False,
            "relay_plane", "holographic_slm", "not_represented_by_current_engine",
            ("objective_pupil_plane",), "warning_only",
            "relay imaging plane not represented by current engine",
        ),
    )


def _axicon_metrics(
    source_field: np.ndarray,
    launch_field: np.ndarray,
    incident: np.ndarray,
    after_axicon: np.ndarray,
    grid: Mapping[str, Any],
    config: RouteAwareAxiconConfig,
    p_source_unapertured: float,
    p_source: float,
) -> dict[str, Any]:
    x_um, y_um = _xy_um(grid)
    sx, sy = _centroid(source_field, x_um, y_um)
    lx, ly = _centroid(launch_field, x_um, y_um)
    cx, cy = _centroid(incident, x_um, y_um)
    rx, ry, ell = _beam_radii(incident, x_um, y_um, cx, cy)
    ax_angle_x, ax_angle_y = _spectral_angle_mrad(incident, grid, config)
    ax_mask = _circular_mask(
        grid,
        config.physical_axicon_clear_aperture_radius_um,
        config.physical_axicon_centre_x_um,
        config.physical_axicon_centre_y_um,
    )
    p_incident = field_power(incident, config.dx_um, config.dx_um)
    p_after = field_power(after_axicon, config.dx_um, config.dx_um)
    overlap = float(np.sum((np.abs(incident) ** 2) * ax_mask) / max(np.sum(np.abs(incident) ** 2), 1e-30))
    tilt_loc = _valid_location(config.field_tilt_location)
    applies_before_axicon = _location_is_upstream_of_axicon(tilt_loc)
    if applies_before_axicon:
        tilt_distance_um = _distance_from_location_to_axicon_um(tilt_loc, config)
        expected_dx = float(tilt_distance_um * np.tan(config.field_tilt_x_mrad * 1e-3))
        expected_dy = float(tilt_distance_um * np.tan(config.field_tilt_y_mrad * 1e-3))
        measured_dx = float(cx - lx)
        measured_dy = float(cy - ly)
        walkoff_error = float(np.hypot(measured_dx - expected_dx, measured_dy - expected_dy))
    else:
        expected_dx = expected_dy = 0.0
        measured_dx = measured_dy = 0.0
        walkoff_error = 0.0
    rel_x = float(cx - config.physical_axicon_centre_x_um)
    rel_y = float(cy - config.physical_axicon_centre_y_um)
    return {
        "field_location": "physical_axicon_plane",
        "field_tilt_injection_location": tilt_loc,
        "beam_decentre_injection_location": _valid_location(config.beam_decentre_location),
        "source_centroid_x_um": sx,
        "source_centroid_y_um": sy,
        "pre_axicon_launch_centroid_x_um": lx,
        "pre_axicon_launch_centroid_y_um": ly,
        "incident_centroid_x_um": cx,
        "incident_centroid_y_um": cy,
        "incident_angle_x_mrad": ax_angle_x,
        "incident_angle_y_mrad": ax_angle_y,
        "incident_beam_radius_x_um": rx,
        "incident_beam_radius_y_um": ry,
        "incident_ellipticity": ell,
        "axicon_centre_x_um": float(config.physical_axicon_centre_x_um),
        "axicon_centre_y_um": float(config.physical_axicon_centre_y_um),
        "relative_beam_to_axicon_offset_x_um": rel_x,
        "relative_beam_to_axicon_offset_y_um": rel_y,
        "relative_beam_to_axicon_offset_um": float(np.hypot(rel_x, rel_y)),
        "axicon_aperture_overlap_fraction": overlap,
        "source_aperture_transmitted_fraction": float(p_source / max(p_source_unapertured, 1e-30)),
        "axicon_transmitted_fraction": float(p_after / max(p_incident, 1e-30)),
        "field_of_view_margin_um": _field_of_view_margin_um(incident, x_um, y_um),
        "upstream_tilt_predicted_walkoff_x_um": expected_dx,
        "upstream_tilt_predicted_walkoff_y_um": expected_dy,
        "upstream_tilt_predicted_walkoff_um": float(np.hypot(expected_dx, expected_dy)),
        "upstream_tilt_measured_walkoff_x_um": measured_dx,
        "upstream_tilt_measured_walkoff_y_um": measured_dy,
        "upstream_tilt_measured_walkoff_um": float(np.hypot(measured_dx, measured_dy)),
        "walkoff_model_error_um": walkoff_error,
        "walkoff_model_applies": bool(applies_before_axicon),
        "diagnostic_geometry_note": DIAGNOSTIC_GEOMETRY_NOTE,
    }


def run_route_aware_axicon_pipeline(
    controls: Mapping[str, Any] | None = None,
    *,
    config: RouteAwareAxiconConfig | None = None,
) -> RouteAwareAxiconRun:
    """Run the route-aware physical-axicon diagnostic pipeline."""
    config = _with_overrides(config or RouteAwareAxiconConfig(), controls)
    if config.route_mode != "physical_axicon":
        warnings = (
            "holographic_slm route declarations are available, but this physical-axicon "
            "pipeline does not fake unrepresented 4F/relay planes.",
        )
        config = replace(config, route_mode="physical_axicon")
    else:
        warnings = ()

    grid = _grid(config)
    x_um, y_um = _xy_um(grid)
    source, source_unapertured = _make_source_field(grid, config)
    applied_source: list[str] = []
    source, a = _apply_location_perturbations(source, "source_plane", grid, config)
    applied_source += list(a)
    source, a = _apply_location_perturbations(source, "after_beam_conditioning", grid, config)
    applied_source += list(a)
    p_unap = field_power(source_unapertured, config.dx_um, config.dx_um)
    p_source = field_power(source, config.dx_um, config.dx_um)
    source_energy = config.input_pulse_energy_uJ * p_source / max(p_unap, 1e-30)
    pre_axicon_launch = source.copy()

    pre_prop = make_bl_asm_propagator(
        source, grid, config.wavelength_m, n_medium=config.n_medium, bandlimit=config.bandlimit
    )
    incident_pre_location = pre_prop(config.pre_axicon_distance_mm * _MM)
    if _valid_location(config.field_tilt_location) in {"source_plane", "after_beam_conditioning"}:
        walkoff_reference = pre_axicon_launch
    else:
        walkoff_reference = incident_pre_location
    incident, applied_before_axicon = _apply_location_perturbations(
        incident_pre_location, "before_physical_axicon", grid, config
    )

    T = physical_axicon_transmission(grid, config)
    after_axicon = incident * T
    after_axicon, applied_after_axicon = _apply_location_perturbations(
        after_axicon, "after_physical_axicon", grid, config
    )
    p_incident = field_power(incident, config.dx_um, config.dx_um)
    p_after = field_power(after_axicon, config.dx_um, config.dx_um)
    axicon_energy = source_energy * p_after / max(p_incident, 1e-30)

    metrics = _axicon_metrics(source, walkoff_reference, incident, after_axicon, grid, config, p_unap, p_source)
    records = build_route_perturbation_records(config)
    warnings = tuple(list(warnings) + [
        r.note for r in records if r.enabled and r.status in {"future_not_implemented", "warning_only"}
    ])

    source_state = ComponentPlaneState(
        plane_name="source_plane",
        field=source,
        x_um=x_um,
        y_um=y_um,
        dx_um=config.dx_um,
        dy_um=config.dx_um,
        pulse_energy_before_uJ=config.input_pulse_energy_uJ,
        pulse_energy_after_uJ=source_energy,
        transmitted_fraction=source_energy / max(config.input_pulse_energy_uJ, 1e-30),
        applied_components=tuple(applied_source),
        metadata={
            "route": "physical_axicon",
            "physical_location": "source_plane",
            "route_graph": [n.__dict__ for n in physical_axicon_route_graph(config)],
        },
    )
    incident_state = ComponentPlaneState(
        plane_name="physical_axicon_incident_field",
        field=incident,
        x_um=x_um,
        y_um=y_um,
        dx_um=config.dx_um,
        dy_um=config.dx_um,
        pulse_energy_before_uJ=source_energy,
        pulse_energy_after_uJ=source_energy,
        transmitted_fraction=1.0,
        applied_components=("pre_axicon_free_space_propagation",) + tuple(applied_before_axicon),
        metadata={"route": "physical_axicon", "physical_location": "before_physical_axicon"},
    )
    axicon_state = ComponentPlaneState(
        plane_name="physical_axicon_plane",
        field=after_axicon,
        x_um=x_um,
        y_um=y_um,
        dx_um=config.dx_um,
        dy_um=config.dx_um,
        pulse_energy_before_uJ=source_energy,
        pulse_energy_after_uJ=axicon_energy,
        transmitted_fraction=axicon_energy / max(source_energy, 1e-30),
        applied_components=("physical_axicon_thin_element",) + tuple(applied_after_axicon),
        metadata={
            "route": "physical_axicon",
            "physical_location": "physical_axicon_plane",
            "axicon_transmission": "aperture * exp(i phi_axicon)",
        },
    )

    post_prop = make_bl_asm_propagator(
        after_axicon, grid, config.wavelength_m, n_medium=config.n_medium, bandlimit=config.bandlimit
    )
    z_um = np.linspace(0.0, config.post_axicon_distance_um, int(config.n_z))
    intensity = np.empty((len(z_um), len(x_um), len(x_um)), dtype=float)
    z_objective_um = float(config.axicon_to_objective_distance_um)
    field_after_objective = post_prop(z_objective_um * _UM)
    field_after_objective, applied_after_objective = _apply_location_perturbations(
        field_after_objective, "after_objective", grid, config
    )
    post_objective_prop = make_bl_asm_propagator(
        field_after_objective, grid, config.wavelength_m, n_medium=config.n_medium, bandlimit=config.bandlimit
    )
    for i, zz in enumerate(z_um):
        if float(zz) <= z_objective_um + 1e-12:
            U = post_prop(float(zz) * _UM)
        else:
            U = post_objective_prop((float(zz) - z_objective_um) * _UM)
        intensity[i] = np.abs(U) ** 2
    reference_field = post_objective_prop(config.objective_to_reference_distance_mm * _MM)
    reference_field, applied_reference = _apply_location_perturbations(
        reference_field, "free_space_reference_plane", grid, config
    )
    intensity[-1] = np.abs(reference_field) ** 2

    reference_state = ComponentPlaneState(
        plane_name="free_space_reference_plane",
        field=reference_field,
        x_um=x_um,
        y_um=y_um,
        dx_um=config.dx_um,
        dy_um=config.dx_um,
        pulse_energy_before_uJ=axicon_energy,
        pulse_energy_after_uJ=axicon_energy,
        transmitted_fraction=1.0,
        applied_components=("post_axicon_free_space_propagation",)
        + tuple(applied_after_objective)
        + tuple(applied_reference),
        metadata={
            "reference_plane": "post-axicon free-space reference plane, n=1.0",
            "no_material_model": True,
            "physical_location": "free_space_reference_plane",
        },
    )

    stack = PropagatedFieldStack(
        intensity_zyx=intensity,
        x_um=x_um,
        y_um=y_um,
        z_um=z_um,
        input_pulse_energy_uJ=config.input_pulse_energy_uJ,
        sample_pulse_energy_uJ=axicon_energy,
        transmitted_fraction=axicon_energy / max(config.input_pulse_energy_uJ, 1e-30),
        plane_states=(source_state, incident_state, axicon_state, reference_state),
        warnings=warnings,
        metadata={
            "stage": "stage8c3r3_route_aware_physical_axicon",
            "route_mode": "physical_axicon",
            "n_medium": float(config.n_medium),
            "diagnostic_geometry_note": DIAGNOSTIC_GEOMETRY_NOTE,
            "route_graph": [n.__dict__ for n in physical_axicon_route_graph(config)],
            "perturbations": [r.as_dict() for r in records],
        },
    )
    return RouteAwareAxiconRun(
        config=config,
        route_mode="physical_axicon",
        source_state=source_state,
        axicon_incident_state=incident_state,
        physical_axicon_state=axicon_state,
        reference_plane_state=reference_state,
        propagated_stack=stack,
        axicon_incidence_metrics=metrics,
        perturbation_records=records,
        warnings=warnings,
        metadata={"stage": "stage8c3r3_route_aware_physical_axicon"},
    )


@dataclass(frozen=True)
class AxiconAlignmentSweepFamily:
    key: str
    title: str
    param_label: str
    param_values: tuple[float, ...]
    control_fn: Callable[[float], Mapping[str, Any]]


@dataclass(frozen=True)
class AxiconAlignmentSweepResult:
    family: AxiconAlignmentSweepFamily
    rows: tuple[Mapping[str, Any], ...]
    label: str = DIAGNOSTIC_SWEEP_LABEL
    final_export_allowed: bool = False


def build_axicon_alignment_sweep_families() -> dict[str, AxiconAlignmentSweepFamily]:
    return {
        "source_tilt": AxiconAlignmentSweepFamily(
            "source_tilt", "Source-plane tilt through fixed axicon", "source tilt (mrad)",
            (0.0, 3.0, 6.0, 9.0, 12.0, 16.0),
            lambda v: {"field_tilt_x_mrad": v, "field_tilt_location": "source_plane", "pre_axicon_distance_mm": 0.80},
        ),
        "axicon_input_decentre": AxiconAlignmentSweepFamily(
            "axicon_input_decentre", "Beam decentre at axicon input", "input decentre at axicon (um)",
            (0.0, 2.0, 4.0, 6.0, 9.0, 12.0),
            lambda v: {"beam_decentre_x_um": v, "beam_decentre_location": "before_physical_axicon"},
        ),
        "axicon_lateral_offset": AxiconAlignmentSweepFamily(
            "axicon_lateral_offset", "Mechanical axicon lateral offset", "axicon offset (um)",
            (0.0, 2.0, 4.0, 6.0, 9.0, 12.0),
            lambda v: {"physical_axicon_centre_x_um": v},
        ),
        "input_radius": AxiconAlignmentSweepFamily(
            "input_radius", "Input beam radius / aperture overlap", "beam radius (um)",
            (12.0, 16.0, 20.0, 24.0, 30.0, 36.0),
            lambda v: {"input_beam_radius_um": v, "input_aperture_radius_um": 44.0},
        ),
        "axicon_aperture": AxiconAlignmentSweepFamily(
            "axicon_aperture", "Physical axicon clear aperture", "clear aperture radius (um)",
            (18.0, 22.0, 26.0, 32.0, 38.0, 44.0),
            lambda v: {"physical_axicon_clear_aperture_radius_um": v},
        ),
        "relative_source_axicon": AxiconAlignmentSweepFamily(
            "relative_source_axicon", "Relative source/axicon alignment", "opposed offset (um)",
            (0.0, 2.0, 4.0, 6.0, 9.0, 12.0),
            lambda v: {"beam_decentre_x_um": v, "beam_decentre_location": "source_plane",
                       "physical_axicon_centre_x_um": -v},
        ),
    }


def _route_reliability(metrics: Mapping[str, Any], axis: Mapping[str, Any]) -> str:
    if float(metrics["field_of_view_margin_um"]) < 0.0 or float(axis["out_of_frame_fraction"]) > 0.02:
        return "invalid_out_of_frame"
    if float(metrics["axicon_aperture_overlap_fraction"]) < 0.85 or float(axis["ring_fit_quality"]) < 0.45:
        return "caution_crop_limited"
    return "numerically_reliable"


def run_axicon_alignment_sweep(
    family: str | AxiconAlignmentSweepFamily,
    *,
    config: RouteAwareAxiconConfig | None = None,
) -> AxiconAlignmentSweepResult:
    if isinstance(family, str):
        family = build_axicon_alignment_sweep_families()[family]
    config = config or RouteAwareAxiconConfig.fast()
    baseline = run_route_aware_axicon_pipeline(config=config)
    rows: list[dict[str, Any]] = []
    for value in family.param_values:
        run = run_route_aware_axicon_pipeline(family.control_fn(value), config=config)
        st = run.propagated_stack
        sel = int(np.argmax(st.intensity_zyx.max(axis=(1, 2))))
        est = estimate_annular_axis(st.intensity_zyx[sel], st.x_um, st.y_um)
        axis = compute_axis_tracking(st, plane_index=sel, core_radius_um=2.0)
        cls = classify_translation_vs_deformation(
            baseline.propagated_stack, st, plane_index=sel, core_radius_um=2.0
        )
        energy = compute_energy_throughput(run)  # type: ignore[arg-type]
        rows.append({
            "param_value": float(value),
            "diagnostic_label": DIAGNOSTIC_SWEEP_LABEL,
            "axicon_plane_walkoff_um": float(run.axicon_incidence_metrics["relative_beam_to_axicon_offset_um"]),
            "axicon_aperture_overlap_fraction": float(run.axicon_incidence_metrics["axicon_aperture_overlap_fraction"]),
            "transmitted_fraction": float(st.transmitted_fraction),
            "reference_plane_pulse_energy_uJ": float(st.reference_plane_pulse_energy_uJ),
            "reference_plane_beam_axis_error_um": float(est["beam_axis_error_um"]),
            "ring_centre_x_um": float(est["ring_centre_x_um"]),
            "core_centre_x_um": float(est["core_centre_x_um"]),
            "azimuthal_uniformity": float(est["azimuthal_uniformity"]),
            "ring_fit_quality": float(est["ring_fit_quality"]),
            "residual_deformation": float(cls["residual_shape_deformation_score"]),
            "peak_fluence_per_transmitted_energy": float(energy["peak_to_reference_energy_ratio"]),
            "numerical_reliability": _route_reliability(run.axicon_incidence_metrics, axis),
        })
    return AxiconAlignmentSweepResult(family=family, rows=tuple(rows))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _extent(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    return float(x.min()), float(x.max()), float(y.min()), float(y.max())


def _save(fig: Any, output_path: str | Path | None, dpi: int, title: str, stage: str) -> None:
    if output_path is None:
        return
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", metadata={
        "Title": title,
        "stage": stage,
        "final_export_allowed": "False",
        "Description": "Stage 8C.3R.3 route-aware physical-axicon free-space diagnostic; no material response.",
    })


def _card(ax: Any, title: str, lines: list[str], face: str = "#fafafa", edge: str = "#455a64",
          fs: float = 7.4) -> None:
    ax.set_axis_off()
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                               facecolor=face, edgecolor=edge, lw=1.2, clip_on=False))
    ax.text(0.04, 0.94, title, transform=ax.transAxes, va="top", ha="left",
            fontsize=fs + 1.5, fontweight="bold", color=edge)
    ax.text(0.04, 0.80, "\n".join(lines), transform=ax.transAxes, va="top", ha="left",
            fontsize=fs, family="monospace", color="#1a1a1a")


def _badges(fig: Any, y: float = 0.92) -> None:
    for x, txt, ec, fc in [
        (0.045, "DIAGNOSTIC ONLY", "#0d47a1", "#e3f2fd"),
        (0.180, "NO MATERIAL RESPONSE", "#4a148c", "#f3e5f5"),
        (0.365, "FREE-SPACE n=1.0", "#1b5e20", "#e8f5e9"),
        (0.505, "ROUTE-AWARE AXICON", "#bf360c", "#fff3e0"),
    ]:
        fig.text(x, y, txt, ha="left", va="center", fontsize=8.8, fontweight="bold",
                 color=ec, bbox=dict(boxstyle="round,pad=0.25", facecolor=fc, edgecolor=ec, lw=1.0))


def plot_route_aware_axicon_pipeline(
    *,
    config: RouteAwareAxiconConfig | None = None,
    controls: Mapping[str, Any] | None = None,
    output_path: str | Path | None = None,
    dpi: int = 155,
) -> "matplotlib.figure.Figure":
    run = run_route_aware_axicon_pipeline(controls, config=config or RouteAwareAxiconConfig())
    st = run.propagated_stack
    fl = stack_to_fluence(st)
    x = np.asarray(st.x_um, float)
    z = np.asarray(st.z_um, float)
    ext = _extent(x, x)
    sel = int(np.argmax(st.intensity_zyx.max(axis=(1, 2))))
    y0 = int(np.argmin(np.abs(x)))
    axm = compute_axis_tracking(st, plane_index=sel, core_radius_um=2.0)
    T = physical_axicon_transmission(_grid(run.config), run.config)

    fig = plt.figure(figsize=(17.0, 12.0), facecolor="white")
    gs = fig.add_gridspec(3, 3, left=0.06, right=0.965, top=0.84, bottom=0.07,
                          hspace=0.42, wspace=0.28)
    fig.suptitle("Stage 8C.3R.3 Route-Aware Physical Axicon Pipeline\n"
                 "Perturbations are injected at declared route locations before downstream propagation/elements",
                 x=0.045, y=0.975, ha="left", va="top", fontsize=15.5, fontweight="bold")
    _badges(fig)

    panels = [
        (np.abs(run.source_state.field), "source/input amplitude", "source_plane", "viridis"),
        (np.angle(run.source_state.field), "source/input phase", "source_plane", "twilight"),
        (np.abs(run.axicon_incident_state.field), "field arriving at axicon", "physical_axicon_plane", "viridis"),
        (np.angle(T), "axicon aperture + phase centre", "physical_axicon_plane", "twilight"),
        (np.abs(run.physical_axicon_state.field), "field immediately after axicon", "physical_axicon_plane", "viridis"),
        (fl.fluence_zyx_j_cm2[-1], "reference-plane XY fluence", "post_objective_reference_plane", "viridis"),
    ]
    for i, (arr, title, loc, cmap) in enumerate(panels):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        im = ax.imshow(arr, origin="lower", extent=ext, cmap=cmap, aspect="equal")
        ax.set_title(f"{title}\n{loc}", fontsize=9.5, fontweight="bold")
        ax.set_xlabel("x (um)"); ax.set_ylabel("y (um)")
        if "axicon" in title:
            ax.scatter([run.config.physical_axicon_centre_x_um], [run.config.physical_axicon_centre_y_um],
                       c="white", marker="+", s=90, linewidths=1.8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    axxz = fig.add_subplot(gs[2, 0])
    axxz.imshow(fl.fluence_zyx_j_cm2[:, y0, :].T, origin="lower",
                extent=(float(z.min()), float(z.max()), float(x.min()), float(x.max())),
                cmap="viridis", aspect="auto")
    axxz.plot(z, axm["axis_x_by_z_um"], color="#00e5ff", lw=1.8, label="fitted axis")
    axxz.set_title("reference-plane XZ fluence + axis", fontsize=9.5, fontweight="bold")
    axxz.set_xlabel("post-axicon z (um)"); axxz.set_ylabel("x (um)")
    axxz.legend(fontsize=7)

    energies = [
        run.source_state.pulse_energy_before_uJ,
        run.source_state.pulse_energy_after_uJ,
        run.axicon_incident_state.pulse_energy_after_uJ,
        run.physical_axicon_state.pulse_energy_after_uJ,
        st.reference_plane_pulse_energy_uJ,
    ]
    axen = fig.add_subplot(gs[2, 1])
    axen.plot(range(len(energies)), energies, "-o", color="#1565c0", lw=2)
    axen.set_xticks(range(len(energies)))
    axen.set_xticklabels(["input", "src ap", "ax in", "ax out", "ref"], rotation=15, fontsize=8)
    axen.set_ylabel("pulse energy (uJ)")
    axen.set_title("energy-throughput waterfall", fontsize=9.5, fontweight="bold")
    axen.grid(alpha=0.28)

    m = run.axicon_incidence_metrics
    _card(fig.add_subplot(gs[2, 2]), "Commanded vs actual axis / incidence", [
        f"pre distance        : {run.config.pre_axicon_distance_mm:.3f} mm",
        f"post distance       : {run.config.post_axicon_distance_mm:.3f} mm",
        f"incident centroid   : ({m['incident_centroid_x_um']:.2f}, {m['incident_centroid_y_um']:.2f}) um",
        f"incident angle      : {m['incident_angle_x_mrad']:.2f}, {m['incident_angle_y_mrad']:.2f} mrad",
        f"beam-axicon offset  : {m['relative_beam_to_axicon_offset_um']:.2f} um",
        f"axicon overlap      : {m['axicon_aperture_overlap_fraction']:.3f}",
        f"axis error @ ref    : {axm['beam_axis_error_um']:.2f} um",
        "geometry: diagnostic demo, not measured bench",
    ], "#e8f5e9", "#1b5e20")

    fig.c3r3_metadata = {  # type: ignore[attr-defined]
        "stage": "stage8c3r3_route_aware_axicon_pipeline",
        "final_export_allowed": False,
        "route_mode": "physical_axicon",
    }
    _save(fig, output_path, dpi, "Stage 8C.3R.3 Route-Aware Axicon Pipeline",
          "stage8c3r3_route_aware_axicon_pipeline")
    return fig


def plot_upstream_vs_post_axicon_tilt_comparison(
    *,
    config: RouteAwareAxiconConfig | None = None,
    tilt_mrad: float = 12.0,
    output_path: str | Path | None = None,
    dpi: int = 155,
) -> "matplotlib.figure.Figure":
    config = config or RouteAwareAxiconConfig()
    runs = [
        (
            "A. before physical axicon path\n(source-plane injection)",
            run_route_aware_axicon_pipeline(
                {"field_tilt_x_mrad": tilt_mrad, "field_tilt_location": "source_plane"},
                config=config,
            ),
            "upstream_field_tilt_location_source_plane",
        ),
        (
            "B. immediately after physical axicon",
            run_route_aware_axicon_pipeline(
                {"field_tilt_x_mrad": tilt_mrad, "field_tilt_location": "after_physical_axicon"},
                config=config,
            ),
            "post_axicon_steering_test",
        ),
        (
            "C. after objective / downstream steering",
            run_route_aware_axicon_pipeline(
                {"field_tilt_x_mrad": tilt_mrad, "field_tilt_location": "after_objective"},
                config=config,
            ),
            "after_objective_downstream_steering",
        ),
    ]
    fig = plt.figure(figsize=(17.0, 13.2), facecolor="white")
    gs = fig.add_gridspec(3, 4, left=0.06, right=0.965, top=0.84, bottom=0.065,
                          hspace=0.45, wspace=0.28)
    fig.suptitle("Stage 8C.3R.3 Field-Tilt Injection Location Sweep\n"
                 "The same tilt produces different downstream behaviour depending on where it enters the route graph",
                 x=0.045, y=0.972, ha="left", va="top", fontsize=15.2, fontweight="bold")
    _badges(fig)
    for row, (label, run, classification) in enumerate(runs):
        st = run.propagated_stack
        fl = stack_to_fluence(st)
        x = np.asarray(st.x_um, float)
        z = np.asarray(st.z_um, float)
        y0 = int(np.argmin(np.abs(x)))
        ext = _extent(x, x)
        axis = compute_axis_tracking(st, plane_index=int(np.argmax(st.intensity_zyx.max(axis=(1, 2)))), core_radius_um=2.0)

        ax0 = fig.add_subplot(gs[row, 0])
        ax0.imshow(np.abs(run.axicon_incident_state.field), origin="lower", extent=ext, cmap="viridis", aspect="equal")
        ax0.scatter([run.config.physical_axicon_centre_x_um], [run.config.physical_axicon_centre_y_um],
                    c="white", marker="+", s=90, linewidths=1.8)
        ax0.set_title(f"{label}\nfield incident on axicon", fontsize=8.6, fontweight="bold")
        ax0.set_xlabel("x (um)"); ax0.set_ylabel("y (um)")

        ax1 = fig.add_subplot(gs[row, 1])
        ax1.imshow(fl.fluence_zyx_j_cm2[-1], origin="lower", extent=ext, cmap="viridis", aspect="equal")
        ax1.set_title("reference-plane XY fluence", fontsize=8.8, fontweight="bold")
        ax1.set_xlabel("x (um)"); ax1.set_ylabel("y (um)")

        ax2 = fig.add_subplot(gs[row, 2])
        ax2.imshow(fl.fluence_zyx_j_cm2[:, y0, :].T, origin="lower",
                   extent=(float(z.min()), float(z.max()), float(x.min()), float(x.max())),
                   cmap="viridis", aspect="auto")
        ax2.plot(z, axis["axis_x_by_z_um"], color="#00e5ff", lw=1.7)
        ax2.set_title("XZ propagation + axis trajectory", fontsize=8.8, fontweight="bold")
        ax2.set_xlabel("post-axicon z (um)"); ax2.set_ylabel("x (um)")

        m = run.axicon_incidence_metrics
        _card(fig.add_subplot(gs[row, 3]), "Classification / overlap / energy", [
            f"tilt location       : {m['field_tilt_injection_location']}",
            f"classification       : {classification}",
            f"incident centroid   : ({m['incident_centroid_x_um']:.2f}, {m['incident_centroid_y_um']:.2f}) um",
            f"pred walkoff        : {m['upstream_tilt_predicted_walkoff_um']:.2f} um",
            f"meas walkoff        : {m['upstream_tilt_measured_walkoff_um']:.2f} um",
            f"walkoff error       : {m['walkoff_model_error_um']:.2f} um",
            f"axicon overlap      : {m['axicon_aperture_overlap_fraction']:.3f}",
            f"transmitted frac    : {st.transmitted_fraction:.3f}",
            f"axis x steering     : {axis['beam_steering_angle_x_mrad']:.2f} mrad",
            "same type, different injection location",
        ], "#fffde7" if row == 0 else ("#e3f2fd" if row == 1 else "#e8f5e9"),
            "#f57f17" if row == 0 else ("#0d47a1" if row == 1 else "#1b5e20"))

    fig.c3r3_metadata = {  # type: ignore[attr-defined]
        "stage": "stage8c3r3_upstream_vs_post_axicon_tilt_comparison",
        "tilt_mrad": float(tilt_mrad),
        "comparison": "field_tilt_injection_location_sweep",
        "final_export_allowed": False,
    }
    _save(fig, output_path, dpi, "Stage 8C.3R.3 Field-Tilt Injection Location Sweep",
          "stage8c3r3_upstream_vs_post_axicon_tilt_comparison")
    return fig


def plot_axicon_alignment_sensitivity_atlas(
    *,
    config: RouteAwareAxiconConfig | None = None,
    sweep_results: Mapping[str, AxiconAlignmentSweepResult] | None = None,
    output_path: str | Path | None = None,
    dpi: int = 150,
) -> "matplotlib.figure.Figure":
    config = config or RouteAwareAxiconConfig.fast()
    families = build_axicon_alignment_sweep_families()
    results = dict(sweep_results or {})
    for key in families:
        if key not in results:
            results[key] = run_axicon_alignment_sweep(key, config=config)

    fig = plt.figure(figsize=(16.0, 11.2), facecolor="white")
    gs = fig.add_gridspec(3, 2, left=0.065, right=0.965, top=0.84, bottom=0.08,
                          hspace=0.55, wspace=0.32)
    fig.suptitle("Stage 8C.3R.3 Physical-Axicon Alignment Sensitivity Atlas\n"
                 + DIAGNOSTIC_SWEEP_LABEL,
                 x=0.045, y=0.972, ha="left", va="top", fontsize=15.2, fontweight="bold")
    _badges(fig)

    for i, key in enumerate(families):
        res = results[key]
        rows = list(res.rows)
        xv = np.array([r["param_value"] for r in rows], float)
        walk = np.array([r["axicon_plane_walkoff_um"] for r in rows], float)
        overlap = np.array([r["axicon_aperture_overlap_fraction"] for r in rows], float)
        trans = np.array([r["transmitted_fraction"] for r in rows], float)
        axis = np.array([r["reference_plane_beam_axis_error_um"] for r in rows], float)
        resid = np.array([r["residual_deformation"] for r in rows], float)
        fitq = np.array([r["ring_fit_quality"] for r in rows], float)
        ax = fig.add_subplot(gs[i // 2, i % 2])
        ax2 = ax.twinx()
        ax.plot(xv, walk, "-o", color="#1565c0", label="axicon walkoff (um)", lw=1.8, ms=4)
        ax.plot(xv, axis, "-s", color="#00838f", label="ref axis error (um)", lw=1.6, ms=3.5)
        ax2.plot(xv, overlap, "-^", color="#2e7d32", label="axicon overlap", lw=1.5, ms=3.5)
        ax2.plot(xv, trans, "-d", color="#ef6c00", label="transmitted fraction", lw=1.4, ms=3.5)
        ax2.plot(xv, resid, "-v", color="#6a1b9a", label="residual deformation", lw=1.3, ms=3.4)
        ax2.plot(xv, fitq, ":", color="#455a64", label="ring fit Q", lw=1.5)
        ax.set_title(res.family.title, fontsize=10.0, fontweight="bold")
        ax.set_xlabel(res.family.param_label, fontsize=8.5)
        ax.set_ylabel("um", fontsize=8.3, color="#1565c0")
        ax2.set_ylabel("fraction / score", fontsize=8.3)
        ax.grid(alpha=0.25)
        ax2.set_ylim(0.0, 1.05)
        lines = ax.get_lines() + ax2.get_lines()
        ax.legend(lines, [ln.get_label() for ln in lines], fontsize=6.8, loc="best", framealpha=0.82)
        ax.text(0.98, 0.04,
                f"last: U {rows[-1]['azimuthal_uniformity']:.2f}, "
                f"peak/E {rows[-1]['peak_fluence_per_transmitted_energy']:.2f}\n"
                f"{rows[-1]['numerical_reliability']}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7.0,
                bbox=dict(facecolor="white", edgecolor="0.72", alpha=0.84, pad=2.5))

    fig.text(0.066, 0.035,
             "All curves are route-aware free-space diagnostics. They are not experimentally measured laboratory tolerances.",
             fontsize=9.0, color="#1b5e20", fontweight="bold")
    fig.c3r3_metadata = {  # type: ignore[attr-defined]
        "stage": "stage8c3r3_axicon_alignment_sensitivity_atlas",
        "families": list(families),
        "diagnostic_label": DIAGNOSTIC_SWEEP_LABEL,
        "final_export_allowed": False,
    }
    _save(fig, output_path, dpi, "Stage 8C.3R.3 Axicon Alignment Sensitivity Atlas",
          "stage8c3r3_axicon_alignment_sensitivity_atlas")
    return fig
