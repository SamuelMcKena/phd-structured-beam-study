#!/usr/bin/env python3
"""Run the two-SLM registration architecture comparison study.

The study compares:

A) upstream_vortex
   SLM1 = vortex + carrier/blaze
   SLM2 = correction + carrier/blaze

B) downstream_vortex
   SLM1 = correction + carrier/blaze
   SLM2 = vortex + carrier/blaze

BOTH SLMs carry carrier/blaze ramps in BOTH architectures.  The study changes only
sub-pixel beam/hologram-to-lattice registration; beam-to-hologram centring is
held fixed by the compensated panel-translation/pattern-offset construction.

Outputs are written below
`outputs/validation/slm_registration_architecture/<tag>/`.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, is_dataclass
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from vbb_study.digital_twin.slm_pixel_registration import RegistrationSampling
from vbb_study.digital_twin.slm_registration_architectures import (
    ARCHITECTURES,
    RegistrationState,
    architecture_registration_states_1d,
    build_architecture_registration_route,
    canonical_registration_hardware,
    unit_cell_offsets,
)
from vbb_study.digital_twin.slm_registration_metrics import (
    axial_profile,
    plane_metrics,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "validation" / "slm_registration_architecture"


def _case_id(charge: int) -> str:
    return "B0" if int(charge) == 0 else f"V{int(charge)}"


def _parse_csv_numbers(text: str, cast=float) -> list[Any]:
    return [cast(v.strip()) for v in str(text).split(",") if v.strip()]


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _correction_from_npy(path: str | None, window_m: float) -> Callable | None:
    """Load a square correction map and expose it as a panel-coordinate command.

    The map is assumed to cover the same centred square physical window used by
    the registration simulation.  Bilinear interpolation is used.  No file means
    correction OFF (flat zero), which is the physics-isolation baseline.
    """

    if not path:
        return None
    phase = np.asarray(np.load(path), dtype=float)
    if phase.ndim != 2 or phase.shape[0] != phase.shape[1]:
        raise ValueError("--correction-npy must contain a square 2-D phase map")
    n = int(phase.shape[0])
    dx = float(window_m) / n
    x0 = -0.5 * float(window_m) + 0.5 * dx

    def command(xp: np.ndarray, yp: np.ndarray) -> np.ndarray:
        from scipy.ndimage import map_coordinates

        ix = (np.asarray(xp, dtype=float) - x0) / dx
        iy = (np.asarray(yp, dtype=float) - x0) / dx
        coords = np.vstack((iy.ravel(), ix.ravel()))
        out = map_coordinates(
            phase,
            coords,
            order=1,
            mode="nearest",
            prefilter=False,
        )
        return out.reshape(np.shape(xp))

    return command


def _reference_key(
    architecture: str,
    charge: int,
    beam_radius_px: float,
    panel: str,
    axis: str,
) -> tuple:
    return (architecture, int(charge), float(beam_radius_px), panel, axis)


def _route_metrics(
    route: dict[str, Any],
    *,
    charge: int,
    reference_route: dict[str, Any] | None,
) -> dict[str, Any]:
    ref_axicon = None if reference_route is None else reference_route["field_on_axicon_plane"]
    metrics = plane_metrics(
        route["field_on_axicon_plane"],
        route["grid"],
        charge=int(charge),
        reference=ref_axicon,
    )
    if "fine_post_iris" in route:
        ref_fine = None if reference_route is None else reference_route.get("fine_post_iris")
        fine_metrics = plane_metrics(
            route["fine_post_iris"],
            route["fine_grid"],
            charge=int(charge),
            reference=ref_fine,
        )
        for key, value in fine_metrics.items():
            metrics[f"post_iris_{key}"] = value
    return metrics


def run_sweep(args: argparse.Namespace, outdir: Path) -> Path:
    hw = canonical_registration_hardware()
    pitch = float(hw["pixel_pitch_m"])
    sampling = RegistrationSampling(
        fine_grid_n=int(args.fine_grid_n),
        relay_grid_n=int(args.relay_grid_n),
        window_m=float(args.window_mm) * 1e-3,
    )
    correction = _correction_from_npy(args.correction_npy, sampling.window_m)

    charges = _parse_csv_numbers(args.charges, int)
    radii_px = _parse_csv_numbers(args.beam_radii_px, float)
    fractions = _parse_csv_numbers(args.offset_fractions, float)
    offsets_m = np.asarray(fractions, dtype=float) * pitch
    axes = [v.strip() for v in args.axes.split(",") if v.strip()]
    panels = [v.strip() for v in args.panels.split(",") if v.strip()]
    architectures = [v.strip() for v in args.architectures.split(",") if v.strip()]

    for architecture in architectures:
        if architecture not in ARCHITECTURES:
            raise ValueError(f"unknown architecture {architecture!r}")

    rows: list[dict[str, Any]] = []

    for architecture in architectures:
        for charge in charges:
            case_id = _case_id(charge)
            for radius_px in radii_px:
                radius_m = float(radius_px) * pitch
                for panel in panels:
                    for axis in axes:
                        states = architecture_registration_states_1d(
                            offsets_m,
                            panel=panel,
                            axis=axis,
                        )
                        reference: dict[str, Any] | None = None
                        for frac, state in zip(fractions, states):
                            route = build_architecture_registration_route(
                                case_id,
                                architecture=architecture,
                                registration=state,
                                sampling=sampling,
                                beam_radius_m=radius_m,
                                correction_command=correction,
                                pixelate_phase=True,
                                fill_factor_model=args.fill_factor_model,
                                pixel_value_model=args.pixel_value_model,
                                keep_intermediate_fields=False,
                                keep_fine_post_iris=True,
                            )
                            if float(frac) == 0.0:
                                reference = route
                            metrics = _route_metrics(
                                route,
                                charge=charge,
                                reference_route=reference,
                            )
                            row = {
                                "architecture": architecture,
                                "charge": int(charge),
                                "beam_radius_px": float(radius_px),
                                "panel_dof": panel,
                                "axis": axis,
                                "offset_fraction_px": float(frac),
                                "offset_um": float(frac) * pitch * 1e6,
                                "vortex_owner": route["metadata"]["phase_roles"]["vortex_owner"],
                                "correction_owner": route["metadata"]["phase_roles"]["correction_owner"],
                                "slm1_carrier_blaze_present": True,
                                "slm2_carrier_blaze_present": True,
                                "slm1_carrier_frequency_cpm": route["metadata"]["slm1_carrier_frequency_cpm"],
                                "slm2_carrier_frequency_cpm": route["metadata"]["slm2_carrier_frequency_cpm"],
                                "total_scalar_carrier_frequency_cpm": route["metadata"]["total_scalar_carrier_frequency_cpm"],
                                "correction_status": route["metadata"]["phase_roles"]["correction_status"],
                                "iris_selected_power_fraction": route["metadata"]["fourf"]["iris_selected_power_fraction"],
                            }
                            row.update(metrics)
                            rows.append(row)
                            if float(frac) != 0.0:
                                del route
                        # Release the fine-grid zero-registration reference before
                        # moving to the next charge/radius/panel/axis combination.
                        reference = None

    path = outdir / "architecture_sweep.csv"
    if rows:
        fields = sorted({k for row in rows for k in row})
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    return path


def run_unit_cell(args: argparse.Namespace, outdir: Path) -> Path:
    """2-D within-pixel registration map for representative cases."""

    hw = canonical_registration_hardware()
    pitch = float(hw["pixel_pitch_m"])
    sampling = RegistrationSampling(
        fine_grid_n=int(args.fine_grid_n),
        relay_grid_n=int(args.relay_grid_n),
        window_m=float(args.window_mm) * 1e-3,
    )
    correction = _correction_from_npy(args.correction_npy, sampling.window_m)
    architectures = [v.strip() for v in args.architectures.split(",") if v.strip()]
    panels = [v.strip() for v in args.unit_cell_panels.split(",") if v.strip()]
    charges = _parse_csv_numbers(args.unit_cell_charges, int)
    radii_px = _parse_csv_numbers(args.unit_cell_beam_radii_px, float)
    offsets = unit_cell_offsets(pitch, int(args.unit_cell_subdivisions))

    rows: list[dict[str, Any]] = []

    for architecture in architectures:
        for charge in charges:
            for radius_px in radii_px:
                case_id = _case_id(charge)
                radius_m = radius_px * pitch
                for panel in panels:
                    reference = build_architecture_registration_route(
                        case_id,
                        architecture=architecture,
                        registration=RegistrationState(),
                        sampling=sampling,
                        beam_radius_m=radius_m,
                        correction_command=correction,
                        fill_factor_model=args.fill_factor_model,
                        pixel_value_model=args.pixel_value_model,
                        keep_fine_post_iris=True,
                    )
                    for dx in offsets:
                        for dy in offsets:
                            if panel == "slm1":
                                state = RegistrationState(slm1_dx_m=dx, slm1_dy_m=dy)
                            elif panel == "slm2":
                                state = RegistrationState(slm2_dx_m=dx, slm2_dy_m=dy)
                            else:
                                raise ValueError("unit-cell panels must be slm1 or slm2")
                            route = build_architecture_registration_route(
                                case_id,
                                architecture=architecture,
                                registration=state,
                                sampling=sampling,
                                beam_radius_m=radius_m,
                                correction_command=correction,
                                fill_factor_model=args.fill_factor_model,
                                pixel_value_model=args.pixel_value_model,
                                keep_fine_post_iris=True,
                            )
                            metrics = _route_metrics(
                                route,
                                charge=charge,
                                reference_route=reference,
                            )
                            rows.append(
                                {
                                    "architecture": architecture,
                                    "charge": int(charge),
                                    "beam_radius_px": float(radius_px),
                                    "panel": panel,
                                    "dx_px": float(dx / pitch),
                                    "dy_px": float(dy / pitch),
                                    "dx_um": float(dx * 1e6),
                                    "dy_um": float(dy * 1e6),
                                    **metrics,
                                }
                            )
                            del route

    path = outdir / "unit_cell_maps.csv"
    if rows:
        fields = sorted({k for row in rows for k in row})
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    return path


def run_interpanel(args: argparse.Namespace, outdir: Path) -> Path:
    """SLM1-offset x SLM2-offset map along the chosen registration axis."""

    hw = canonical_registration_hardware()
    pitch = float(hw["pixel_pitch_m"])
    sampling = RegistrationSampling(
        fine_grid_n=int(args.fine_grid_n),
        relay_grid_n=int(args.relay_grid_n),
        window_m=float(args.window_mm) * 1e-3,
    )
    correction = _correction_from_npy(args.correction_npy, sampling.window_m)
    fractions = _parse_csv_numbers(args.interpanel_fractions, float)
    charge = int(args.interpanel_charge)
    radius_px = float(args.interpanel_beam_radius_px)
    axis = str(args.interpanel_axis)
    if axis not in ("x", "y"):
        raise ValueError("--interpanel-axis must be x or y")

    rows: list[dict[str, Any]] = []
    for architecture in [v.strip() for v in args.architectures.split(",") if v.strip()]:
        reference = build_architecture_registration_route(
            _case_id(charge),
            architecture=architecture,
            registration=RegistrationState(),
            sampling=sampling,
            beam_radius_m=radius_px * pitch,
            correction_command=correction,
            fill_factor_model=args.fill_factor_model,
            pixel_value_model=args.pixel_value_model,
            keep_fine_post_iris=True,
        )
        for f1 in fractions:
            for f2 in fractions:
                d1 = float(f1) * pitch
                d2 = float(f2) * pitch
                if axis == "x":
                    state = RegistrationState(slm1_dx_m=d1, slm2_dx_m=d2)
                else:
                    state = RegistrationState(slm1_dy_m=d1, slm2_dy_m=d2)
                route = build_architecture_registration_route(
                    _case_id(charge),
                    architecture=architecture,
                    registration=state,
                    sampling=sampling,
                    beam_radius_m=radius_px * pitch,
                    correction_command=correction,
                    fill_factor_model=args.fill_factor_model,
                    pixel_value_model=args.pixel_value_model,
                    keep_fine_post_iris=True,
                )
                metrics = _route_metrics(
                    route,
                    charge=charge,
                    reference_route=reference,
                )
                rows.append(
                    {
                        "architecture": architecture,
                        "charge": charge,
                        "beam_radius_px": radius_px,
                        "axis": axis,
                        "slm1_offset_px": float(f1),
                        "slm2_offset_px": float(f2),
                        **metrics,
                    }
                )
                del route

    path = outdir / "interpanel_registration_map.csv"
    if rows:
        fields = sorted({k for row in rows for k in row})
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    return path


def run_axial(args: argparse.Namespace, outdir: Path) -> Path:
    """Quantify propagated Bessel-region sensitivity for representative cases."""

    hw = canonical_registration_hardware()
    pitch = float(hw["pixel_pitch_m"])
    lam = float(hw["wavelength_m"])
    sampling = RegistrationSampling(
        fine_grid_n=int(args.fine_grid_n),
        relay_grid_n=int(args.relay_grid_n),
        window_m=float(args.window_mm) * 1e-3,
    )
    correction = _correction_from_npy(args.correction_npy, sampling.window_m)
    charges = _parse_csv_numbers(args.axial_charges, int)
    radii_px = _parse_csv_numbers(args.axial_beam_radii_px, float)
    fractions = _parse_csv_numbers(args.axial_offset_fractions, float)
    panels = [v.strip() for v in args.axial_panels.split(",") if v.strip()]
    axis = str(args.axial_axis)
    if axis not in ("x", "y", "diagonal"):
        raise ValueError("--axial-axis must be x, y or diagonal")
    for panel in panels:
        if panel not in ("slm1", "slm2"):
            raise ValueError("--axial-panels may contain only slm1 and slm2")

    rows: list[dict[str, Any]] = []

    for architecture in [v.strip() for v in args.architectures.split(",") if v.strip()]:
        for charge in charges:
            case_id = _case_id(charge)
            for radius_px in radii_px:
                radius_m = float(radius_px) * pitch
                for panel in panels:
                    states = architecture_registration_states_1d(
                        np.asarray(fractions, dtype=float) * pitch,
                        panel=panel,
                        axis=axis,
                    )
                    routes: list[dict[str, Any]] = []
                    for state in states:
                        routes.append(
                            build_architecture_registration_route(
                                case_id,
                                architecture=architecture,
                                registration=state,
                                sampling=sampling,
                                beam_radius_m=radius_m,
                                correction_command=correction,
                                fill_factor_model=args.fill_factor_model,
                                pixel_value_model=args.pixel_value_model,
                                keep_intermediate_fields=False,
                                keep_fine_post_iris=False,
                            )
                        )

                    ref = routes[0]
                    kr = abs(float(ref["metadata"]["axicon"]["exact_kr_m_inv"]))
                    k0 = 2.0 * math.pi / lam
                    theta = math.asin(min(1.0, kr / k0))
                    zmax = radius_m / max(math.tan(theta), 1e-15)
                    z_values = np.linspace(
                        0.05 * zmax,
                        1.05 * zmax,
                        int(args.axial_z_planes),
                    )
                    ref_profile = axial_profile(
                        ref["post_axicon"],
                        ref["grid"],
                        wavelength_m=lam,
                        z_values_m=z_values,
                        charge=charge,
                        keep_planes_m=(),
                    )
                    z_ref = float(ref_profile["z_at_peak_m"])
                    ref_keep = axial_profile(
                        ref["post_axicon"],
                        ref["grid"],
                        wavelength_m=lam,
                        z_values_m=[z_ref],
                        charge=charge,
                        keep_planes_m=[z_ref],
                    )
                    ref_plane = ref_keep["planes"][z_ref]
                    ref_peak = float(ref_profile["peak_intensity"])
                    ref_zone = float(ref_profile["bessel_zone_fwhm_m"])

                    for frac, route in zip(fractions, routes):
                        prof = axial_profile(
                            route["post_axicon"],
                            route["grid"],
                            wavelength_m=lam,
                            z_values_m=z_values,
                            charge=charge,
                            keep_planes_m=[z_ref],
                        )
                        at_ref = prof["planes"][z_ref]
                        pm = plane_metrics(
                            at_ref,
                            route["grid"],
                            charge=charge,
                            reference=ref_plane,
                        )
                        idx_ref = int(np.argmin(np.abs(np.asarray(prof["z_m"]) - z_ref)))
                        row = {
                            "architecture": architecture,
                            "charge": int(charge),
                            "beam_radius_px": float(radius_px),
                            "panel": panel,
                            "axis": axis,
                            "offset_fraction_px": float(frac),
                            "offset_um": float(frac) * pitch * 1e6,
                            "reference_z_peak_m": z_ref,
                            "z_at_peak_m": float(prof["z_at_peak_m"]),
                            "peak_intensity": float(prof["peak_intensity"]),
                            "peak_intensity_ratio": float(prof["peak_intensity"]) / max(ref_peak, 1e-30),
                            "bessel_zone_fwhm_m": float(prof["bessel_zone_fwhm_m"]),
                            "bessel_zone_ratio": (
                                float(prof["bessel_zone_fwhm_m"]) / max(ref_zone, 1e-30)
                                if np.isfinite(ref_zone) and ref_zone > 0.0
                                else float("nan")
                            ),
                            "ring_radius_at_reference_z_m": float(np.asarray(prof["ring_radius_z"])[idx_ref]),
                            "core_darkness_at_reference_z": float(np.asarray(prof["core_darkness_z"])[idx_ref]),
                        }
                        for key, value in pm.items():
                            row[f"propagated_{key}"] = value
                        rows.append(row)

                    routes.clear()

    path = outdir / "axial_metrics.csv"
    if rows:
        fields = sorted({k for row in rows for k in row})
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    return path


def run_representative_xy(args: argparse.Namespace, outdir: Path) -> Path:
    """Save cropped representative pre-axicon and propagated XY profiles.

    Both SLM1-only and SLM2-only registration sweeps are retained for both
    architectures.  Only the band-limited relay-grid fields are stored; the
    fine-grid post-SLM fields are intentionally not accumulated in memory.
    """

    hw = canonical_registration_hardware()
    pitch = float(hw["pixel_pitch_m"])
    lam = float(hw["wavelength_m"])
    sampling = RegistrationSampling(
        fine_grid_n=int(args.fine_grid_n),
        relay_grid_n=int(args.relay_grid_n),
        window_m=float(args.window_mm) * 1e-3,
    )
    correction = _correction_from_npy(args.correction_npy, sampling.window_m)
    charge = int(args.xy_charge)
    radius_px = float(args.xy_beam_radius_px)
    fractions = _parse_csv_numbers(args.xy_offset_fractions, float)
    panels = [v.strip() for v in args.xy_panels.split(",") if v.strip()]
    for panel in panels:
        if panel not in ("slm1", "slm2"):
            raise ValueError("--xy-panels may contain only slm1 and slm2")

    payload: dict[str, Any] = {}
    half_width_m = float(args.xy_half_width_um) * 1e-6

    for architecture in [v.strip() for v in args.architectures.split(",") if v.strip()]:
        for panel in panels:
            # Build the zero-registration route first to define the matched
            # propagated reference plane.
            ref = build_architecture_registration_route(
                _case_id(charge),
                architecture=architecture,
                registration=RegistrationState(),
                sampling=sampling,
                beam_radius_m=radius_px * pitch,
                correction_command=correction,
                fill_factor_model=args.fill_factor_model,
                pixel_value_model=args.pixel_value_model,
                keep_intermediate_fields=False,
                keep_fine_post_iris=False,
            )
            kr = abs(float(ref["metadata"]["axicon"]["exact_kr_m_inv"]))
            k0 = 2.0 * math.pi / lam
            theta = math.asin(min(1.0, kr / k0))
            zmax = radius_px * pitch / max(math.tan(theta), 1e-15)
            z_values = np.linspace(0.05 * zmax, 1.05 * zmax, int(args.xy_z_planes))
            ref_ax = axial_profile(
                ref["post_axicon"],
                ref["grid"],
                wavelength_m=lam,
                z_values_m=z_values,
                charge=charge,
                keep_planes_m=(),
            )
            z_peak = float(ref_ax["z_at_peak_m"])

            x_full = np.asarray(ref["grid"]["x"], dtype=float)
            roi = np.where(np.abs(x_full) <= half_width_m)[0]
            if roi.size < 8:
                raise ValueError(
                    "--xy-half-width-um is too small for the relay-grid sampling"
                )
            x_roi = x_full[roi]

            key = f"{architecture}__{panel}"
            payload[f"{key}__offset_fraction_px"] = np.asarray(fractions, dtype=float)
            payload[f"{key}__x_m"] = x_roi
            payload[f"{key}__z_peak_m"] = np.asarray([z_peak], dtype=float)

            pre = []
            propagated = []
            for frac in fractions:
                d = float(frac) * pitch
                state = (
                    RegistrationState(slm1_dx_m=d)
                    if panel == "slm1"
                    else RegistrationState(slm2_dx_m=d)
                )
                route = (
                    ref
                    if float(frac) == 0.0
                    else build_architecture_registration_route(
                        _case_id(charge),
                        architecture=architecture,
                        registration=state,
                        sampling=sampling,
                        beam_radius_m=radius_px * pitch,
                        correction_command=correction,
                        fill_factor_model=args.fill_factor_model,
                        pixel_value_model=args.pixel_value_model,
                        keep_intermediate_fields=False,
                        keep_fine_post_iris=False,
                    )
                )
                field_pre = np.asarray(route["field_on_axicon_plane"], dtype=np.complex64)
                pre.append(field_pre[np.ix_(roi, roi)])

                ax = axial_profile(
                    route["post_axicon"],
                    route["grid"],
                    wavelength_m=lam,
                    z_values_m=[z_peak],
                    charge=charge,
                    keep_planes_m=[z_peak],
                )
                field_z = np.asarray(ax["planes"][z_peak], dtype=np.complex64)
                propagated.append(field_z[np.ix_(roi, roi)])
                if route is not ref:
                    del route

            payload[f"{key}__field_pre_axicon"] = np.stack(pre)
            payload[f"{key}__field_at_reference_z"] = np.stack(propagated)
            del ref

    path = outdir / "representative_xy_fields.npz"
    np.savez_compressed(path, **payload)
    return path

def write_manifest(args: argparse.Namespace, outdir: Path, artifacts: list[Path]) -> Path:
    hw = canonical_registration_hardware()
    manifest = {
        "study": "slm_registration_architecture_comparison",
        "architectures": {
            "upstream_vortex": {
                "SLM1": "vortex + carrier/blaze",
                "SLM2": "correction + carrier/blaze",
            },
            "downstream_vortex": {
                "SLM1": "correction + carrier/blaze",
                "SLM2": "vortex + carrier/blaze",
            },
        },
        "hard_requirement": "carrier/blaze present on BOTH SLM1 and SLM2 in both architectures",
        "correction_map": args.correction_npy or "none: flat-zero physics-isolation baseline",
        "hardware": _jsonable(hw),
        "sampling": {
            "fine_grid_n": int(args.fine_grid_n),
            "relay_grid_n": int(args.relay_grid_n),
            "window_mm": float(args.window_mm),
            "pixel_value_model": args.pixel_value_model,
            "fill_factor_model": args.fill_factor_model,
        },
        "claim_boundary": (
            "geometric pixel-registration sensitivity on the accepted effective-channel "
            "4F route; no unmeasured SLM1-to-SLM2 propagation distance is fabricated"
        ),
        "artifacts": [str(p.relative_to(outdir)) for p in artifacts],
    }
    path = outdir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="core")
    p.add_argument("--stages", default="sweep,xy")
    p.add_argument("--architectures", default="upstream_vortex,downstream_vortex")
    p.add_argument("--charges", default="0,1,3,5,10,20")
    p.add_argument("--beam-radii-px", default="25,50,100,250")
    p.add_argument("--offset-fractions", default="0,0.125,0.25,0.375,0.5")
    p.add_argument("--axes", default="x,y,diagonal")
    p.add_argument("--panels", default="slm1,slm2,common,differential")

    p.add_argument("--fine-grid-n", type=int, default=2500)
    p.add_argument("--relay-grid-n", type=int, default=1024)
    p.add_argument("--window-mm", type=float, default=10.0)
    p.add_argument("--fill-factor-model", default="throughput_only")
    p.add_argument("--pixel-value-model", default="area_average")
    p.add_argument("--correction-npy", default=None)

    p.add_argument("--unit-cell-panels", default="slm1,slm2")
    p.add_argument("--unit-cell-charges", default="10,20")
    p.add_argument("--unit-cell-beam-radii-px", default="50,250")
    p.add_argument("--unit-cell-subdivisions", type=int, default=8)

    p.add_argument("--interpanel-charge", type=int, default=20)
    p.add_argument("--interpanel-beam-radius-px", type=float, default=50.0)
    p.add_argument("--interpanel-axis", default="x")
    p.add_argument("--interpanel-fractions", default="-0.5,-0.25,0,0.25,0.5")

    p.add_argument("--axial-charges", default="10,20")
    p.add_argument("--axial-beam-radii-px", default="50,250")
    p.add_argument("--axial-panels", default="slm1,slm2")
    p.add_argument("--axial-axis", default="x")
    p.add_argument("--axial-offset-fractions", default="0,0.25,0.5")
    p.add_argument("--axial-z-planes", type=int, default=24)

    p.add_argument("--xy-charge", type=int, default=20)
    p.add_argument("--xy-beam-radius-px", type=float, default=50.0)
    p.add_argument("--xy-offset-fractions", default="0,0.125,0.25,0.375,0.5")
    p.add_argument("--xy-panels", default="slm1,slm2")
    p.add_argument("--xy-z-planes", type=int, default=24)
    p.add_argument("--xy-half-width-um", type=float, default=600.0)
    return p


def main() -> None:
    args = build_parser().parse_args()
    outdir = DEFAULT_OUTPUT / str(args.tag)
    outdir.mkdir(parents=True, exist_ok=True)

    stages = {v.strip() for v in args.stages.split(",") if v.strip()}
    artifacts: list[Path] = []
    if "sweep" in stages:
        artifacts.append(run_sweep(args, outdir))
    if "unit_cell" in stages:
        artifacts.append(run_unit_cell(args, outdir))
    if "interpanel" in stages:
        artifacts.append(run_interpanel(args, outdir))
    if "axial" in stages:
        artifacts.append(run_axial(args, outdir))
    if "xy" in stages:
        artifacts.append(run_representative_xy(args, outdir))
    artifacts.append(write_manifest(args, outdir, artifacts))

    print("SLM registration architecture study complete")
    for path in artifacts:
        print(path)


if __name__ == "__main__":
    main()
