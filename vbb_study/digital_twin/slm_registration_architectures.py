"""SLM registration architecture comparison for the two-SLM vortex-Bessel bench.

This module extends the dedicated sub-pixel registration study without changing
its accepted baseline implementation.  It compares two phase-allocation
architectures while keeping the downstream 4F route and physical axicon fixed.

Architectures
-------------
``upstream_vortex``
    SLM1 creates the vortex.  SLM2 applies the correction term and ALWAYS also
    carries the wrapped/quantised carrier-blaze term.

``downstream_vortex``
    SLM1 applies the correction term.  SLM2 creates the vortex and ALWAYS also
    carries the wrapped/quantised carrier-blaze term.

The user-facing scientific question is therefore whether creating the vortex
locally on SLM2 reduces sensitivity to the relative pixel registration of the
two panels compared with transporting a pixelated vortex from SLM1 into a
second independently registered SLM plane.

Important scope boundary
------------------------
The accepted registration route currently treats the two programmable phase
planes in the effective-channel model used by the explicit 4F bench route; it
does not insert an unmeasured SLM1-to-SLM2 propagation distance.  The separate
component-owned CSLM route contains a placeholder/demo separation, but that is
not measured laboratory geometry and is therefore deliberately not imported
here.  This study quantifies pixel-lattice ownership/registration and the
downstream 4F/axicon consequence without fabricating an inter-SLM distance.

A spatial correction may be supplied as a callable `f(x, y) -> phase_rad`.
With no supplied correction the correction term is exactly zero; this is the
clean physics-isolation baseline and must not be described as a measured
wavefront correction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

from vbb_study.digital_twin.slm_pixel_registration import (
    EPS,
    TWOPI,
    RegistrationSampling,
    _apply_panel,
    band_limited_decimate_fixed_window,
    canonical_registration_hardware,
    lean_asm_propagate,
    lean_xy_grid,
)
from vbb_study.digital_twin.vortex_beam_slm_errors import (
    GaussianBeamError,
    SLMError,
    gaussian_input_field,
)
from vbb_study.digital_twin.vortex_explicit_4f import (
    LensError,
    _apply_lens_plane,
    nominal_order_position_m,
    physical_iris,
)
from vbb_study.digital_twin.vortex_system_route import (
    AxiconError,
    physical_axicon_on_own_plane,
)

ArchitectureName = str
CorrectionCommand = Callable[[np.ndarray, np.ndarray], np.ndarray]

ARCHITECTURES = ("upstream_vortex", "downstream_vortex")


@dataclass(frozen=True)
class RegistrationState:
    """Independent 2-D sub-pixel registration state for both SLMs.

    Values are physical beam/hologram-to-lattice offsets in metres.  The
    intended beam-to-hologram alignment remains nominal because each selected
    panel is physically translated by `-d` and the displayed pattern is shifted
    electronically by `+d`.
    """

    slm1_dx_m: float = 0.0
    slm1_dy_m: float = 0.0
    slm2_dx_m: float = 0.0
    slm2_dy_m: float = 0.0

    def validate(self) -> None:
        vals = (
            self.slm1_dx_m,
            self.slm1_dy_m,
            self.slm2_dx_m,
            self.slm2_dy_m,
        )
        if not all(math.isfinite(float(v)) for v in vals):
            raise ValueError("registration offsets must be finite")

    def as_pixel_fractions(self, pitch_m: float) -> dict[str, float]:
        p = float(pitch_m)
        return {
            "slm1_dx_px": float(self.slm1_dx_m) / p,
            "slm1_dy_px": float(self.slm1_dy_m) / p,
            "slm2_dx_px": float(self.slm2_dx_m) / p,
            "slm2_dy_px": float(self.slm2_dy_m) / p,
        }


def pure_registration_error(dx_m: float, dy_m: float) -> SLMError:
    """Return an SLM error that changes lattice registration but not centring."""

    dx = float(dx_m)
    dy = float(dy_m)
    return SLMError(
        panel_translation_m=(-dx, -dy),
        pattern_offset_m=(+dx, +dy),
    )


def errors_from_registration(state: RegistrationState) -> tuple[SLMError, SLMError]:
    """Convert one two-panel registration state into the two SLM errors."""

    state.validate()
    return (
        pure_registration_error(state.slm1_dx_m, state.slm1_dy_m),
        pure_registration_error(state.slm2_dx_m, state.slm2_dy_m),
    )


def unit_cell_offsets(pitch_m: float, subdivisions: int = 8) -> np.ndarray:
    """Distinct offsets spanning one periodic SLM pixel unit cell.

    The endpoint at one full pitch is omitted because it is physically identical
    to zero registration.
    """

    n = int(subdivisions)
    if n < 2:
        raise ValueError("subdivisions must be >= 2")
    return np.arange(n, dtype=float) * float(pitch_m) / float(n)


def zero_correction(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Flat correction command used for the physics-isolation baseline."""

    return np.zeros(np.broadcast_shapes(np.shape(x), np.shape(y)), dtype=float)


def _validate_architecture(architecture: ArchitectureName) -> str:
    architecture = str(architecture)
    if architecture not in ARCHITECTURES:
        raise ValueError(
            f"architecture must be one of {ARCHITECTURES}; got {architecture!r}"
        )
    return architecture


def _charge_from_case_id(case_id: str) -> int:
    """Parse B0 or arbitrary non-negative V<n> scalar vortex case IDs."""

    cid = str(case_id).strip()
    if cid == "B0":
        return 0
    if cid.startswith("V") and cid[1:].isdigit():
        charge = int(cid[1:])
        if charge >= 0:
            return charge
    raise ValueError(f"unsupported scalar vortex case {case_id!r}")



def architecture_phase_components(
    architecture: ArchitectureName,
    *,
    charge: int,
    carrier_cpm: float,
    correction_command: CorrectionCommand | None = None,
) -> tuple[CorrectionCommand, CorrectionCommand, dict[str, Any]]:
    """Return phase commands for SLM1 and SLM2.

    SLM2 always carries the carrier/blaze ramp in BOTH architectures, as required
    by the actual order-selection strategy.  Wrapping and 8-bit quantisation are
    applied later by the common SLM model.
    """

    architecture = _validate_architecture(architecture)
    ell = int(charge)
    carrier = float(carrier_cpm)
    correction = zero_correction if correction_command is None else correction_command

    def vortex(xp: np.ndarray, yp: np.ndarray) -> np.ndarray:
        return float(ell) * np.arctan2(yp, xp)

    def carrier_blaze(xp: np.ndarray, yp: np.ndarray) -> np.ndarray:
        del yp
        return TWOPI * carrier * xp

    if architecture == "upstream_vortex":
        slm1 = vortex

        def slm2(xp: np.ndarray, yp: np.ndarray) -> np.ndarray:
            return np.asarray(correction(xp, yp), dtype=float) + carrier_blaze(xp, yp)

        roles = {
            "SLM1": ("vortex",),
            "SLM2": ("correction", "carrier_blaze"),
            "vortex_owner": "SLM1",
            "correction_owner": "SLM2",
        }
    else:
        slm1 = correction

        def slm2(xp: np.ndarray, yp: np.ndarray) -> np.ndarray:
            return vortex(xp, yp) + carrier_blaze(xp, yp)

        roles = {
            "SLM1": ("correction",),
            "SLM2": ("vortex", "carrier_blaze"),
            "vortex_owner": "SLM2",
            "correction_owner": "SLM1",
        }

    roles["SLM2_carrier_blaze_present"] = True
    roles["correction_status"] = (
        "flat_zero_physics_isolation"
        if correction_command is None
        else "user_supplied_spatial_correction_command"
    )
    return slm1, slm2, roles


def build_architecture_registration_route(
    case_id: str,
    *,
    architecture: ArchitectureName,
    registration: RegistrationState = RegistrationState(),
    sampling: RegistrationSampling = RegistrationSampling(),
    beam: GaussianBeamError = GaussianBeamError(),
    axicon: AxiconError = AxiconError(),
    beam_radius_m: float | None = None,
    correction_command: CorrectionCommand | None = None,
    pixelate_phase: bool = True,
    fill_factor_model: str = "throughput_only",
    pixel_value_model: str = "area_average",
    allow_unresolved_lattice: bool = False,
    keep_intermediate_fields: bool = False,
    keep_fine_post_iris: bool = True,
) -> dict[str, Any]:
    """Build one architecture-comparison route through the physical axicon.

    The route is intentionally matched between architectures.  Only ownership
    of the vortex and correction terms changes; SLM2 carries the same
    carrier/blaze term in both cases.

    Returned diagnostic planes are sufficient to quantify where the
    registration signature first appears:
      - immediately after SLM1,
      - immediately after SLM2 / before the 4F system,
      - immediately after the selected-order iris (fine grid),
      - immediately before the physical axicon,
      - immediately after the physical axicon.

    The physical axicon is a pure phase element here, so intensity changes only
    after subsequent propagation.
    """

    architecture = _validate_architecture(architecture)
    registration.validate()
    sampling.validate()
    beam.validate()
    axicon.validate()

    hw = canonical_registration_hardware()
    panel_cfg = hw["panel"]
    lam = float(hw["wavelength_m"])
    pitch = float(hw["pixel_pitch_m"])
    carrier = float(hw["carrier_cpm"])
    w0 = (
        float(hw["beam_radius_on_slm_m"])
        if beam_radius_m is None
        else float(beam_radius_m)
    )
    ell = _charge_from_case_id(case_id)

    spp = sampling.samples_per_pixel(pitch)
    if pixelate_phase and spp < 2.0 and not allow_unresolved_lattice:
        raise ValueError(
            "architecture registration study requires a resolved SLM lattice; "
            f"got {spp:.2f} samples per pixel"
        )

    slm1_error, slm2_error = errors_from_registration(registration)
    fine = lean_xy_grid(sampling.fine_grid_n, sampling.fine_dx_m)
    field, beam_meta = gaussian_input_field(
        fine,
        wavelength_m=lam,
        canonical_radius_m=w0,
        error=beam,
    )

    slm1_command, slm2_command, phase_roles = architecture_phase_components(
        architecture,
        charge=ell,
        carrier_cpm=carrier,
        correction_command=correction_command,
    )

    field, slm1_meta = _apply_panel(
        field,
        fine,
        slm1_command,
        panel_cfg=panel_cfg,
        error=slm1_error,
        pixel_pitch_m=pitch,
        pixelate_phase=pixelate_phase,
        fill_factor_model=fill_factor_model,
        pixel_value_model=pixel_value_model,
    )
    after_slm1 = field.copy() if keep_intermediate_fields else None

    # Effective-channel registration route: no fabricated SLM1->SLM2 distance.
    field, slm2_meta = _apply_panel(
        field,
        fine,
        slm2_command,
        panel_cfg=panel_cfg,
        error=slm2_error,
        pixel_pitch_m=pitch,
        pixelate_phase=pixelate_phase,
        fill_factor_model=fill_factor_model,
        pixel_value_model=pixel_value_model,
    )
    after_slm2 = field.copy() if keep_intermediate_fields else None

    # Explicit accepted 4F route.
    f4f = float(hw["fourf_focal_length_m"])
    field = lean_asm_propagate(field, fine, lam, f4f)
    field, lens1_meta = _apply_lens_plane(
        field,
        fine,
        wavelength_m=lam,
        focal_length_m=f4f,
        error=LensError(),
        opd_map_m=None,
    )
    field = lean_asm_propagate(field, fine, lam, f4f)

    centre = nominal_order_position_m(
        wavelength_m=lam,
        focal_length_m=f4f,
        carrier_cpm=carrier,
    )
    iris = physical_iris(
        fine,
        radius_m=float(hw["fourier_iris_radius_m"]),
        centre_m=centre,
    )
    pre_iris_power = float(np.sum(np.abs(field) ** 2))
    field = field * iris
    selected_fraction = float(np.sum(np.abs(field) ** 2)) / max(pre_iris_power, EPS)
    del iris

    fine_post_iris = field.copy() if keep_fine_post_iris else None
    field, decim_meta = band_limited_decimate_fixed_window(
        field,
        sampling.relay_grid_n,
    )
    coarse = lean_xy_grid(
        sampling.relay_grid_n,
        float(sampling.window_m) / sampling.relay_grid_n,
    )

    field = lean_asm_propagate(field, coarse, lam, f4f)
    field, lens2_meta = _apply_lens_plane(
        field,
        coarse,
        wavelength_m=lam,
        focal_length_m=f4f,
        error=LensError(),
        opd_map_m=None,
    )
    field = lean_asm_propagate(field, coarse, lam, f4f)

    # Remove the selected-order carrier in the 4F image frame.  Both
    # architectures use the same SLM2 carrier/blaze, so this operation is common.
    field = field * np.exp(+1j * TWOPI * carrier * coarse["X"])
    field_on_axicon = np.asarray(field, dtype=np.complex128)

    axicon_t, axicon_meta = physical_axicon_on_own_plane(
        coarse,
        wavelength_m=lam,
        base_angle_rad=float(hw["axicon_base_angle_rad"]),
        refractive_index=float(hw["axicon_refractive_index"]),
        external_index=float(hw["axicon_external_index"]),
        error=axicon,
        surface_height_error_m=None,
    )
    post_axicon = field_on_axicon * axicon_t

    blaze_period_px = (
        float("inf")
        if abs(carrier) <= EPS
        else 1.0 / (abs(carrier) * pitch)
    )
    meta: dict[str, Any] = {
        "route_id": "slm_registration_architecture_comparison_v1",
        "case_id": case_id,
        "architecture": architecture,
        "vortex_charge": ell,
        "phase_roles": phase_roles,
        "SLM2_carrier_blaze_present": True,
        "carrier_frequency_cpm": carrier,
        "carrier_blaze_period_px": blaze_period_px,
        "wavelength_m": lam,
        "beam_radius_on_slm_m": w0,
        "beam_radius_in_pixels": w0 / pitch,
        "pixel_pitch_m": pitch,
        "registration_m": {
            "slm1_dx_m": float(registration.slm1_dx_m),
            "slm1_dy_m": float(registration.slm1_dy_m),
            "slm2_dx_m": float(registration.slm2_dx_m),
            "slm2_dy_m": float(registration.slm2_dy_m),
        },
        "registration_px": registration.as_pixel_fractions(pitch),
        "beam_to_hologram_decentre_m": {
            "SLM1": (0.0, 0.0),
            "SLM2": (0.0, 0.0),
        },
        "sampling": {
            "fine_grid_n": int(sampling.fine_grid_n),
            "relay_grid_n": int(sampling.relay_grid_n),
            "window_m": float(sampling.window_m),
            "fine_dx_m": float(sampling.fine_dx_m),
            "samples_per_pixel": float(spp),
            "pixel_value_model": pixel_value_model,
            "fill_factor_model": fill_factor_model,
        },
        "slm1": slm1_meta,
        "slm2": slm2_meta,
        "fourf": {
            "model": "explicit_4f_ASM_nominal_lenses_fixed_plus_one_iris",
            "nominal_focal_length_m": f4f,
            "iris_centre_m": tuple(map(float, centre)),
            "iris_radius_m": float(hw["fourier_iris_radius_m"]),
            "iris_selected_power_fraction": selected_fraction,
            "lens1": lens1_meta,
            "lens2": lens2_meta,
            "selected_order_carrier_removal": "plus_G_after_4F_image_inversion",
        },
        "decimation": decim_meta,
        "axicon": axicon_meta,
        "claim_boundary": (
            "geometric registration sensitivity on the accepted effective-channel "
            "4F route; no measured SLM1-to-SLM2 propagation distance is invented"
        ),
    }

    result: dict[str, Any] = {
        "grid": coarse,
        "field_on_axicon_plane": field_on_axicon,
        "post_axicon": np.asarray(post_axicon, dtype=np.complex128),
        "metadata": meta,
    }
    if keep_fine_post_iris and fine_post_iris is not None:
        result["fine_post_iris"] = np.asarray(fine_post_iris, dtype=np.complex128)
        result["fine_grid"] = fine
    if keep_intermediate_fields:
        result["field_after_slm1"] = np.asarray(after_slm1, dtype=np.complex128)
        result["field_after_slm2"] = np.asarray(after_slm2, dtype=np.complex128)
        result["intermediate_grid"] = fine
    return result


def architecture_registration_states_1d(
    offsets_m: np.ndarray,
    *,
    panel: str,
    axis: str,
) -> list[RegistrationState]:
    """Convenience generator for one-panel x/y/diagonal registration sweeps."""

    if panel not in ("slm1", "slm2", "common", "differential"):
        raise ValueError("panel must be slm1, slm2, common or differential")
    if axis not in ("x", "y", "diagonal"):
        raise ValueError("axis must be x, y or diagonal")

    states: list[RegistrationState] = []
    for d in np.asarray(offsets_m, dtype=float):
        dx = float(d) if axis in ("x", "diagonal") else 0.0
        dy = float(d) if axis in ("y", "diagonal") else 0.0
        if panel == "slm1":
            states.append(RegistrationState(slm1_dx_m=dx, slm1_dy_m=dy))
        elif panel == "slm2":
            states.append(RegistrationState(slm2_dx_m=dx, slm2_dy_m=dy))
        elif panel == "common":
            states.append(
                RegistrationState(
                    slm1_dx_m=dx,
                    slm1_dy_m=dy,
                    slm2_dx_m=dx,
                    slm2_dy_m=dy,
                )
            )
        else:
            states.append(
                RegistrationState(
                    slm1_dx_m=dx,
                    slm1_dy_m=dy,
                    slm2_dx_m=-dx,
                    slm2_dy_m=-dy,
                )
            )
    return states
