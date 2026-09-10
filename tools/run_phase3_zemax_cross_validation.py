from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from vbb_study.integrations.zemax.comparison import compare_field_planes
from vbb_study.integrations.zemax.connection import ZemaxSession
from vbb_study.integrations.zemax.field_exchange import load_field_npz, validate_zbf_input
from vbb_study.integrations.zemax.models import FieldPlane, PopRequest
from vbb_study.integrations.zemax.plotting import save_cross_validation_figures
from vbb_study.integrations.zemax.pop import run_pop
from vbb_study.integrations.zemax.prescription import inspect_model, length_unit_to_m, load_surface_map, prepare_working_copy, validate_surface_map

CASES = ("G0", "B0", "V1", "V3")


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _mapped_surface(surface_map: dict[str, int | None], primary: str, fallback: str | None = None) -> int | None:
    value = surface_map.get(primary)
    if value is not None:
        return value
    return None if fallback is None else surface_map.get(fallback)


def _model_wavelength_m(metadata: object, index: int) -> float:
    table = getattr(metadata, "wavelength_table")
    if index < 1 or index > len(table):
        raise SystemExit(f"Wavelength index {index} is outside the prescription wavelength table.")
    value = float(table[index - 1]["wavelength_m"])
    if not np.isfinite(value) or value <= 0:
        raise SystemExit("Selected Zemax prescription wavelength is invalid; refusing to manufacture a wavelength.")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 3A Zemax POP cross-validation runner.")
    parser.add_argument("--model", required=True, type=Path); parser.add_argument("--surface-map", required=True, type=Path); parser.add_argument("--case", required=True, choices=CASES); parser.add_argument("--wavelength-index", type=int, default=1); parser.add_argument("--field-index", type=int, default=1)
    parser.add_argument("--beam-file", type=Path, default=None, help="Supported pre-existing ZBF input. Required for B0/V1/V3 until a supported writer is proven.")
    reference = parser.add_mutually_exclusive_group(); reference.add_argument("--python-field", type=Path, default=None, help="Audited NPZ FieldPlane at the same physical output plane."); reference.add_argument("--python-reference", choices=("phase2c-vector-focus",), default=None, help="Build through the existing canonical Phase 2C vector-Debye route.")
    parser.add_argument("--confirm-same-plane", action="store_true", help="Required with --python-reference: confirms mapped Zemax output is the same physical plane."); parser.add_argument("--sampling", type=int, default=256); parser.add_argument("--x-width-mm", type=float, default=5.0); parser.add_argument("--y-width-mm", type=float, default=5.0); parser.add_argument("--g0-waist-mm", type=float, default=2.0); parser.add_argument("--output-root", type=Path, default=Path("outputs/validation/phase3_zemax")); parser.add_argument("--mode", choices=("standalone", "interactive"), default="standalone"); parser.add_argument("--dry-run", action="store_true"); parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.python_reference and not args.confirm_same_plane:
        raise SystemExit("--python-reference requires --confirm-same-plane; Phase 3 will not silently compare unmatched physical planes.")
    if args.sampling < 16 or args.x_width_mm <= 0 or args.y_width_mm <= 0 or args.g0_waist_mm <= 0:
        raise SystemExit("POP sampling and physical widths/waist must be positive and physically meaningful.")
    case_dir = args.output_root / args.case; working_model, source_sha = prepare_working_copy(args.model, case_dir / "work"); surface_map = load_surface_map(args.surface_map)
    provenance = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "git_sha": _git_sha(), "validation_backend": "zemax_pop", "canonical_solver_replaced": False, "case_id": args.case, "source_model_absolute_path": str(args.model.expanduser().resolve()), "source_model_sha256": source_sha, "working_copy": str(working_model), "software_validation_is_physical_validation": False, "physical_validation_is_experimental_validation": False, "longitudinal_component_exchange_supported": False}
    with ZemaxSession(mode=args.mode) as session:
        system = session.open_model(working_model); metadata = inspect_model(system, source_path=args.model, source_sha256=source_sha); validate_surface_map(surface_map, metadata.surface_count); _write_json(case_dir / "provenance.json", provenance); _write_json(case_dir / "zemax_model_metadata.json", metadata.to_dict()); _write_json(case_dir / "surface_map.json", surface_map)
        if args.verbose: print(json.dumps(metadata.to_dict(), indent=2, default=str))
        if args.dry_run: print("ZEMAX MODEL INSPECTION: PASS"); print("POP EXECUTION: SKIPPED (--dry-run)"); return 0
        start_surface = _mapped_surface(surface_map, "objective_entrance", "input_plane"); end_surface = _mapped_surface(surface_map, "sample_surface")
        if start_surface is None or end_surface is None: raise SystemExit("Surface map must define input_plane/objective_entrance and sample_surface before POP can run.")
        if args.case != "G0" and args.beam_file is None: raise SystemExit(f"{args.case} is blocked: arbitrary Python -> ZBF writing is not yet a verified supported mechanism. Supply a documented pre-existing --beam-file .zbf.")
        beam_type = "GaussianWaist"; beam_file = ""; beam_parameters_m: dict[str, float] | None = {"Waist X": args.g0_waist_mm * 1e-3, "Waist Y": args.g0_waist_mm * 1e-3, "Decenter X": 0.0, "Decenter Y": 0.0}
        if args.beam_file is not None: beam_file = str(validate_zbf_input(args.beam_file)); beam_type = "File"; beam_parameters_m = None
        request = PopRequest(model_path=str(working_model), wavelength_index=args.wavelength_index, field_index=args.field_index, start_surface=start_surface, end_surface=end_surface, x_sampling=args.sampling, y_sampling=args.sampling, x_width_m=args.x_width_mm * 1e-3, y_width_m=args.y_width_mm * 1e-3, polarization=True, beam_type=beam_type, beam_file=beam_file, beam_parameters_m=beam_parameters_m, use_total_power=True, total_power=1.0, use_peak_irradiance=False)
        _write_json(case_dir / "pop_request.json", request.to_dict()); unit_to_m = length_unit_to_m(metadata.system_units); pop_result = run_pop(system, request, system_length_unit_to_m=unit_to_m); zemax_wavelength_m = _model_wavelength_m(metadata, args.wavelength_index)
    np.savez_compressed(case_dir / "zemax_pop_intensity.npz", x_m=pop_result.x_m, y_m=pop_result.y_m, wavelength_m=zemax_wavelength_m, irradiance=pop_result.irradiance)
    python_plane = load_field_npz(args.python_field) if args.python_field is not None else None
    if args.python_reference == "phase2c-vector-focus":
        from vbb_study.integrations.zemax.python_reference import build_phase2c_vector_focus_reference
        python_plane = build_phase2c_vector_focus_reference(args.case)
    if python_plane is None:
        _write_json(case_dir / "status.json", {"zemax_integration": "READY", "pop": "PASS", "python_zemax_comparison": "BLOCKED — matching Python output-plane field not supplied", "real_bench_cross_validation": "NOT CLAIMED", "experimental_validation": False}); print("ZEMAX POP: PASS"); print("PYTHON ↔ ZEMAX COMPARISON: BLOCKED — provide a same-plane Python reference"); return 0
    zemax_plane = FieldPlane(x_m=pop_result.x_m, y_m=pop_result.y_m, wavelength_m=zemax_wavelength_m, intensity=pop_result.irradiance, plane_name="zemax_pop_output", source_solver="zemax_pop", metadata=pop_result.metadata); result = compare_field_planes(python_plane, zemax_plane, case_id=args.case, provenance=provenance); _write_json(case_dir / "comparison_metrics.json", result.metrics)
    with (case_dir / "comparison_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["metric", "value"])
        for key, value in result.metrics.items(): writer.writerow([key, json.dumps(value) if isinstance(value, (list, dict)) else value])
    save_cross_validation_figures(result, case_dir); _write_json(case_dir / "status.json", {"zemax_integration": "READY", "pop": "PASS", "python_zemax_grid_comparison": "PASS", "comparison_plane_confirmed_by_user": bool(args.confirm_same_plane) if args.python_reference else None, "longitudinal_component_validated": False, "experimental_validation": False}); print("ZEMAX POP: PASS"); print("PYTHON ↔ ZEMAX GRID COMPARISON: PASS"); print("Ez VALIDATION: NOT SUPPORTED BY THIS TRANSVERSE POP/ZBF EXCHANGE"); return 0


if __name__ == "__main__":
    raise SystemExit(main())
