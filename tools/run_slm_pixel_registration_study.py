"""Run the sub-pixel SLM pixel-lattice registration sensitivity study.

Stages
------
``convergence``
    Vary only the fine-grid density at fixed physics.  Establishes the sampling
    at which the registration metric itself stops moving, which is a
    prerequisite for any claim: the accepted Phase 2E source grids do not
    resolve an 8 um pixel and therefore cannot represent this effect at all.

``registration``
    Canonical beam.  Sweeps the sub-pixel offset for each registration degree
    of freedom (SLM1 alone, SLM2 alone, both in common, both differentially)
    and each vortex charge.

``beamsize``
    Bench arm, SLM1 registration, with the beam radius stepped down towards the
    pixel pitch.  Reports the iris-clipping confound alongside, because
    shrinking the beam also fills the Fourier-plane iris.

``isolation``
    SLM1 -> axicon only, no carrier and no iris, reaching beam radii of a few
    pixels that the bench architecture cannot support.  Mechanism study only.

Every route is compared against two references: the continuous (unpixelated)
route, which gives the absolute pixelation penalty, and the zero-offset
pixelated route, which gives the registration-induced variability.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from vbb_study.digital_twin.slm_pixel_registration import (
    BENCH_WINDOW_M,
    ISOLATION_WINDOW_M,
    RegistrationSampling,
    build_registration_route,
    canonical_registration_hardware,
    registration_panels,
)
from vbb_study.digital_twin.slm_registration_metrics import (
    axial_profile,
    complex_fidelity,
    plane_metrics,
    relative_l2_phase_aligned,
    sweep_spread,
    translation_registered_fidelity,
)

DEFAULT_CASES = ("B0", "V1", "V3", "V5", "V10", "V20")
DEFAULT_DOFS = ("slm1", "slm2", "common", "differential")


def _z_grid(arm: str, beam_radius_m: float, kr_m_inv: float, wavelength_m: float,
            n_planes: int) -> np.ndarray:
    """Bessel-zone sampling matched to the geometric zone length ``w / tan(beta)``."""

    k = 2.0 * math.pi / float(wavelength_m)
    sin_beta = min(0.999, float(kr_m_inv) / k)
    tan_beta = sin_beta / math.sqrt(max(1.0 - sin_beta ** 2, 1e-12))
    z_max = float(beam_radius_m) / max(tan_beta, 1e-12)
    return np.linspace(0.02 * z_max, 1.05 * z_max, int(n_planes))


def _route(case: str, *, sampling: RegistrationSampling, dof: str | None,
           delta_m: float, axis: str, arm: str, beam_radius_m: float | None,
           pixelate_phase: bool, fill_factor_model: str,
           pixel_value_model: str = "area_average",
           keep_fine: bool = False) -> dict[str, Any]:
    if dof is None or delta_m == 0.0:
        s1, s2 = registration_panels(0.0, dof="slm1", axis=axis)
    else:
        s1, s2 = registration_panels(delta_m, dof=dof, axis=axis)
    return build_registration_route(
        case,
        sampling=sampling,
        slm1=s1,
        slm2=s2,
        arm=arm,
        pixelate_phase=pixelate_phase,
        fill_factor_model=fill_factor_model,
        beam_radius_m=beam_radius_m,
        pixel_value_model=pixel_value_model,
        keep_fine_post_iris=keep_fine,
    )


def _evaluate(route: dict[str, Any], *, charge: int, refs: dict[str, np.ndarray],
              n_z: int, beam_radius_m: float, arm: str,
              keep_axial: bool,
              fine_refs: dict[str, np.ndarray] | None = None) -> dict[str, Any]:
    grid = route["grid"]
    meta = route["metadata"]
    before = route["field_on_axicon_plane"]
    row: dict[str, Any] = {}

    # ---- immediately after the physical iris, on the fine grid -------------
    # On the bench arm the registration signal is smaller than the power the
    # spectral crop discards, so this crop-free comparison is the primary one.
    if fine_refs and "fine_post_iris" in route:
        fine = route["fine_post_iris"]
        fine_dx = float(route["fine_grid"]["dx"])
        for name, ref in fine_refs.items():
            fid = complex_fidelity(fine, ref)
            row[f"postiris_fidelity_vs_{name}"] = fid
            row[f"postiris_infidelity_vs_{name}"] = 1.0 - fid
            row[f"postiris_relative_l2_vs_{name}"] = relative_l2_phase_aligned(
                fine, ref)
            registered = translation_registered_fidelity(
                fine, ref, route["fine_grid"]
            )
            row[f"postiris_translation_registered_fidelity_vs_{name}"] = registered["fidelity"]
            row[f"postiris_translation_registered_infidelity_vs_{name}"] = registered["infidelity"]
            row[f"postiris_translation_alignment_shift_x_m_vs_{name}"] = registered["alignment_shift_x_m"]
            row[f"postiris_translation_alignment_shift_y_m_vs_{name}"] = registered["alignment_shift_y_m"]
            row[f"postiris_centroid_separation_m_vs_{name}"] = registered["centroid_separation_m"]
        row["postiris_power"] = float(np.sum(np.abs(fine) ** 2) * fine_dx ** 2)

    # ---- axicon input plane -------------------------------------------------
    for name, ref in refs.items():
        m = plane_metrics(before, grid, charge=charge, reference=ref)
        for k, v in m.items():
            row[f"before_{k}_vs_{name}"] = v
    base = plane_metrics(before, grid, charge=charge)
    for k, v in base.items():
        row[f"before_{k}"] = v

    if not keep_axial:
        return row

    # ---- Bessel region ------------------------------------------------------
    z = _z_grid(arm, beam_radius_m, float(meta["axicon"]["exact_kr_m_inv"]),
                float(meta["wavelength_m"]), n_z)
    prof = axial_profile(
        route["post_axicon"], grid,
        wavelength_m=float(meta["wavelength_m"]),
        z_values_m=z, charge=charge,
    )
    row.update({
        "after_z_at_peak_m": prof["z_at_peak_m"],
        "after_peak_intensity": prof["peak_intensity"],
        "after_bessel_zone_fwhm_m": prof["bessel_zone_fwhm_m"],
        "after_core_darkness_at_peak": float(
            prof["core_darkness_z"][int(np.argmax(prof["peak_intensity_z"]))]),
        "after_ring_radius_at_peak_m": float(
            prof["ring_radius_z"][int(np.argmax(prof["peak_intensity_z"]))]),
        "after_z_min_m": float(z[0]),
        "after_z_max_m": float(z[-1]),
    })
    row["_axial"] = prof
    return row


def run_stage(args: argparse.Namespace) -> list[dict[str, Any]]:
    hw = canonical_registration_hardware()
    pitch = hw["pixel_pitch_m"]
    rows: list[dict[str, Any]] = []
    axial_rows: list[dict[str, Any]] = []

    if args.stage == "isolation":
        arm, window = "isolation", ISOLATION_WINDOW_M
    else:
        arm, window = "bench", BENCH_WINDOW_M
    if args.window_m and args.window_m > 0:
        window = float(args.window_m)

    deltas = [float(d) * pitch for d in args.delta_fractions]
    beams = [None if b <= 0 else float(b) for b in args.beam_radii_m] or [None]
    fine_ns = list(args.fine_grid_n)

    for fine_n in fine_ns:
        relay_n = min(args.relay_grid_n, fine_n)
        sampling = RegistrationSampling(
            fine_grid_n=fine_n,
            relay_grid_n=fine_n if arm == "isolation" else relay_n,
            window_m=window,
        )
        for case in args.cases:
            charge = 0 if case == "B0" else int(case[1:])
            for beam_radius_m in beams:
                w = hw["beam_radius_on_slm_m"] if beam_radius_m is None else beam_radius_m
                t0 = time.time()
                # references
                cont = _route(case, sampling=sampling, dof=None, delta_m=0.0,
                              axis=args.axis, arm=arm, beam_radius_m=beam_radius_m,
                              pixelate_phase=False,
                              fill_factor_model=args.fill_factor_model,
                              pixel_value_model=args.pixel_value_model,
                              keep_fine=not args.skip_fine_metric)
                zero = _route(case, sampling=sampling, dof=None, delta_m=0.0,
                              axis=args.axis, arm=arm, beam_radius_m=beam_radius_m,
                              pixelate_phase=True,
                              fill_factor_model=args.fill_factor_model,
                              pixel_value_model=args.pixel_value_model,
                              keep_fine=not args.skip_fine_metric)
                refs = {
                    "continuous": cont["field_on_axicon_plane"],
                    "registered0": zero["field_on_axicon_plane"],
                }
                fine_refs = {}
                if "fine_post_iris" in cont:
                    fine_refs["continuous"] = cont["fine_post_iris"]
                if "fine_post_iris" in zero:
                    fine_refs["registered0"] = zero["fine_post_iris"]
                common = {
                    "arm": arm, "case_id": case, "charge": charge,
                    "beam_radius_m": w, "beam_radius_px": w / pitch,
                    "fine_grid_n": fine_n,
                    "relay_grid_n": sampling.relay_grid_n,
                    "window_m": window,
                    "fine_dx_m": sampling.fine_dx_m,
                    "samples_per_pixel": sampling.samples_per_pixel(pitch),
                    "pitch_commensurate": abs(
                        sampling.samples_per_pixel(pitch)
                        - round(sampling.samples_per_pixel(pitch))) < 1e-9,
                    "fill_factor_model": args.fill_factor_model,
                    "pixel_value_model": args.pixel_value_model,
                    "axis": args.axis,
                }
                for label, route, dof, delta in (
                    ("continuous", cont, "none", float("nan")),
                    ("registered", zero, "none", 0.0),
                ):
                    r = _evaluate(route, charge=charge, refs=refs, n_z=args.n_z,
                                  beam_radius_m=w, arm=arm,
                                  keep_axial=not args.skip_axial,
                                  fine_refs=fine_refs)
                    ax = r.pop("_axial", None)
                    rows.append({**common, "series": label, "dof": dof,
                                 "delta_m": delta, "delta_px": delta / pitch if delta == delta else float("nan"),
                                 "iris_selected_fraction": _iris(route),
                                 "decimation_discarded": _discard(route), **r})
                    if ax is not None:
                        axial_rows.extend(_axial_rows({**common, "series": label,
                                                       "dof": dof, "delta_px": 0.0}, ax))
                del cont

                for dof in args.dofs:
                    for delta in deltas:
                        if delta == 0.0:
                            continue
                        route = _route(case, sampling=sampling, dof=dof,
                                       delta_m=delta, axis=args.axis, arm=arm,
                                       beam_radius_m=beam_radius_m,
                                       pixelate_phase=True,
                                       fill_factor_model=args.fill_factor_model,
                                       pixel_value_model=args.pixel_value_model,
                                       keep_fine=not args.skip_fine_metric)
                        r = _evaluate(route, charge=charge, refs=refs, n_z=args.n_z,
                                      beam_radius_m=w, arm=arm,
                                      keep_axial=not args.skip_axial,
                                      fine_refs=fine_refs)
                        ax = r.pop("_axial", None)
                        rows.append({**common, "series": "sweep", "dof": dof,
                                     "delta_m": delta, "delta_px": delta / pitch,
                                     "iris_selected_fraction": _iris(route),
                                     "decimation_discarded": _discard(route), **r})
                        if ax is not None:
                            axial_rows.extend(_axial_rows(
                                {**common, "series": "sweep", "dof": dof,
                                 "delta_px": delta / pitch}, ax))
                        del route
                del zero
                print(f"  [{time.time()-t0:6.1f}s] {arm} {case} w={w*1e6:7.1f}um "
                      f"({w/pitch:6.1f}px) fine_n={fine_n} done", flush=True)

    args._axial_rows = axial_rows
    return rows


def _iris(route: dict[str, Any]) -> float:
    f = route["metadata"].get("fourf")
    if isinstance(f, dict):
        return float(f.get("iris_selected_power_fraction", float("nan")))
    return float("nan")


def _discard(route: dict[str, Any]) -> float:
    d = route["metadata"].get("decimation", {})
    return float(d.get("discarded_power_fraction", float("nan")))


def _axial_rows(common: dict[str, Any], prof: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for i, z in enumerate(prof["z_m"]):
        out.append({**common, "z_m": float(z),
                    "peak_intensity": float(prof["peak_intensity_z"][i]),
                    "core_darkness": float(prof["core_darkness_z"][i]),
                    "ring_radius_m": float(prof["ring_radius_z"][i])})
    return out


def summarise(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Registration-induced spread per (case, beam, dof, sampling)."""

    keys = ("arm", "case_id", "charge", "beam_radius_m", "beam_radius_px",
            "fine_grid_n", "samples_per_pixel", "dof")
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for r in rows:
        if r["series"] != "sweep":
            continue
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    out = []
    for key, items in sorted(groups.items(), key=lambda kv: str(kv[0])):
        rec = dict(zip(keys, key))
        zero = [r for r in rows
                if r["series"] == "registered"
                and all(r[k] == rec[k] for k in keys if k != "dof")]
        pool = items + zero
        for metric in ("postiris_infidelity_vs_registered0",
                       "postiris_translation_registered_infidelity_vs_registered0",
                       "postiris_infidelity_vs_continuous",
                       "postiris_translation_registered_infidelity_vs_continuous",
                       "before_infidelity_vs_continuous",
                       "before_translation_registered_infidelity_vs_continuous",
                       "before_infidelity_vs_registered0",
                       "before_translation_registered_infidelity_vs_registered0",
                       "before_azimuthal_purity",
                       "before_core_darkness",
                       "before_ring_asymmetry_rms",
                       "before_power",
                       "after_peak_intensity",
                       "after_core_darkness_at_peak",
                       "after_bessel_zone_fwhm_m"):
            vals = [r[metric] for r in pool if metric in r]
            if not vals:
                continue
            s = sweep_spread(vals)
            for stat, v in s.items():
                rec[f"{metric}__{stat}"] = v
        rec["n_offsets"] = len(pool)
        out.append(rec)
    return out


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = [dict(r) for r in rows]
    if not rows:
        return
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", default="registration",
                   choices=("convergence", "registration", "beamsize", "isolation"))
    p.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES))
    p.add_argument("--dofs", nargs="+", default=list(DEFAULT_DOFS))
    p.add_argument("--axis", default="x", choices=("x", "y", "diagonal"))
    p.add_argument("--delta-fractions", nargs="+", type=float,
                   default=[0.0, 0.125, 0.25, 0.375, 0.5],
                   help="sub-pixel offsets as a fraction of the 8 um pitch")
    p.add_argument("--fine-grid-n", nargs="+", type=int, default=[4096])
    p.add_argument("--relay-grid-n", type=int, default=2048)
    p.add_argument("--window-m", type=float, default=0.0,
                   help="override the arm window; choose it so that the 8 um pitch "
                        "is an exact integer number of samples (window = N*pitch/m), "
                        "otherwise neighbouring pixels hold different sample counts "
                        "and the sweep measures quadrature noise")
    p.add_argument("--beam-radii-m", nargs="+", type=float, default=[-1.0],
                   help="beam radii on the SLM; -1 means the canonical 2 mm")
    p.add_argument("--skip-fine-metric", action="store_true",
                   help="do not keep the pre-decimation fine field (saves memory)")
    p.add_argument("--fill-factor-model", default="throughput_only",
                   choices=("resolved_pixel_aperture", "throughput_only"))
    p.add_argument("--pixel-value-model", default="area_average",
                   choices=("area_average", "centre_sample"),
                   help="how a pixel takes its value from the continuous command")
    p.add_argument("--n-z", type=int, default=24)
    p.add_argument("--skip-axial", action="store_true")
    p.add_argument("--output-root", type=Path,
                   default=Path("outputs/validation/slm_pixel_registration"))
    p.add_argument("--tag", default=None)
    args = p.parse_args()

    tag = args.tag or args.stage
    root = args.output_root / tag
    root.mkdir(parents=True, exist_ok=True)

    print(f"stage={args.stage} tag={tag}", flush=True)
    t0 = time.time()
    rows = run_stage(args)
    elapsed = time.time() - t0

    write_csv(root / "registration_metrics.csv", rows)
    write_csv(root / "registration_axial.csv", getattr(args, "_axial_rows", []))
    write_csv(root / "registration_summary.csv", summarise(rows))

    hw = canonical_registration_hardware()
    manifest = {
        "stage": args.stage,
        "tag": tag,
        "elapsed_s": elapsed,
        "n_routes": len(rows),
        "cases": list(args.cases),
        "dofs": list(args.dofs),
        "axis": args.axis,
        "delta_fractions_of_pitch": list(args.delta_fractions),
        "fine_grid_n": list(args.fine_grid_n),
        "relay_grid_n": args.relay_grid_n,
        "window_m_override": args.window_m or None,
        "fill_factor_model": args.fill_factor_model,
        "pixel_value_model": args.pixel_value_model,
        "pixel_pitch_m": hw["pixel_pitch_m"],
        "canonical_beam_radius_m": hw["beam_radius_on_slm_m"],
        "references": {
            "continuous": "same route with the pixel lattice bypassed",
            "registered0": "same route pixelated at zero registration offset",
        },
        "provenance": (
            "registration offsets are geometric sensitivity values, not measured "
            "panel positions; absolute claims additionally require the measured "
            "grey-to-phase LUT, static panel phase map and a fitted fringing kernel"
        ),
        "report_status": "screening; not authorised as report evidence",
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {len(rows)} rows to {root} in {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
