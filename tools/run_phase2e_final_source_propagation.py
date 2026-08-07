"""Run the governed Phase 2E final source propagation workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vbb_study.digital_twin.phase2e_final_source_metrics import FixedRegion
from vbb_study.digital_twin.phase2e_final_source_propagation import (
    CASE_CHARGES,
    VALIDATION_ROOT,
    FinalSourcePropagationConfig,
    load_final_source_result,
    finalize_z_step_convergence,
    result_cache_stem,
    run_final_resolution_gate,
    run_final_source_propagation,
    run_z_step_convergence,
    save_final_source_result,
    validate_production_backend,
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fixed_regions() -> dict[str, FixedRegion]:
    gate = _read_json(VALIDATION_ROOT / "final_resolution_gate.json")
    if gate.get("selected_production_grid_n") != 3072:
        raise RuntimeError("the completed source-resolution gate does not authorise N=3072")
    return {
        case_id: FixedRegion(**gate["fixed_regions"][case_id])
        for case_id in CASE_CHARGES
    }


def _run_routes(routes: tuple[str, ...], *, force: bool) -> list[dict]:
    backend = _read_json(VALIDATION_ROOT / "production_backend_validation.json")
    if backend.get("status") != "passed":
        raise RuntimeError("production backend validation has not passed")
    regions = _fixed_regions()
    summaries: list[dict] = []
    for route in routes:
        for case_id in CASE_CHARGES:
            stem = result_cache_stem(case_id, route)
            if stem.with_suffix(".npz").exists() and not force:
                result = load_final_source_result(case_id, route)
                action = "reused"
            else:
                result = run_final_source_propagation(FinalSourcePropagationConfig(
                    case_id=case_id,
                    grid_n=3072,
                    aperture_route=route,
                    fixed_region=regions[case_id],
                ))
                save_final_source_result(result)
                action = "generated"
            summary = {
                "case_id": case_id,
                "route_id": route,
                "action": action,
                "runtime_seconds": result.metadata["runtime_seconds"],
                "maximum_edge_energy_fraction": result.metadata["maximum_edge_energy_fraction"],
                "maximum_propagation_power_drift_fraction": result.metadata[
                    "maximum_propagation_power_drift_fraction"
                ],
                "zones": result.metadata["zones"],
            }
            summaries.append(summary)
            print(json.dumps(summary), flush=True)
            del result
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--gate-only", action="store_true")
    action.add_argument("--backend-validation", action="store_true")
    action.add_argument("--nominal-production", action="store_true")
    action.add_argument("--z-convergence", action="store_true")
    action.add_argument("--z-convergence-case", choices=tuple(CASE_CHARGES))
    action.add_argument("--finalize-z-convergence", action="store_true")
    action.add_argument("--aperture-production", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.gate_only:
        result = run_final_resolution_gate()
        print(json.dumps({
            "status": result["status"],
            "selected_production_grid_n": result["selected_production_grid_n"],
            "grid_pass": result["grid_pass"],
            "runtime_seconds": result["runtime_seconds"],
        }, indent=2))
        return 0 if result["status"] == "passed" else 2
    if args.backend_validation:
        result = validate_production_backend()
        print(json.dumps({"status": result["status"]}, indent=2))
        return 0 if result["status"] == "passed" else 2
    if args.z_convergence:
        result = run_z_step_convergence()
        print(json.dumps({
            "status": result["status"],
            "selected_z_step_m": result["selected_z_step_m"],
            "runtime_seconds": result["runtime_seconds"],
        }, indent=2))
        return 0 if result["status"] == "passed" else 2
    if args.z_convergence_case:
        result = run_z_step_convergence(case_ids=(args.z_convergence_case,))
        print(json.dumps({
            "status": result["status"],
            "case_ids": result["case_ids"],
            "runtime_seconds": result["runtime_seconds"],
        }, indent=2))
        return 0 if result["status"] == "passed" else 2
    if args.finalize_z_convergence:
        result = finalize_z_step_convergence()
        print(json.dumps({
            "status": result["status"],
            "selected_z_step_m": result["selected_z_step_m"],
        }, indent=2))
        return 0 if result["status"] == "passed" else 2
    routes = (
        ("nominal_no_additional_aperture",)
        if args.nominal_production
        else ("soft_aperture_sensitivity", "hard_aperture_diagnostic")
    )
    _run_routes(routes, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
