from __future__ import annotations

"""Run measured z-stack -> physical digital-twin error screening.

Examples
--------
Real data::

    python tools/run_experiment_simulation_match.py \
      --manifest data/my_stack/manifest.json --case V3 --grid-n 512

Synthetic round-trip validation::

    python tools/run_experiment_simulation_match.py \
      --synthetic-known axicon_decentre_x=0.0005 --case V1 --grid-n 128 \
      --families axicon_decentre_x \
      --values axicon_decentre_x=0,0.0005,0.001
"""

import argparse
import csv
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from vbb_study.inference.candidate_models import DEFAULT_MATCH_FAMILIES, FAMILY_REGISTRY
from vbb_study.inference.experimental_stack import IntensityStack, load_experimental_stack
from vbb_study.inference.matcher import match_error_candidates, simulate_candidate_stack
from vbb_study.inference.miao_handoff import build_miao_handoff


def _assignment(text: str) -> tuple[str, str]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("expected FAMILY=value")
    key, value = text.split("=", 1)
    key = key.strip()
    if key not in FAMILY_REGISTRY:
        raise argparse.ArgumentTypeError(f"unknown family {key!r}")
    return key, value.strip()


def _parse_value_overrides(items: list[tuple[str, str]]) -> dict[str, tuple[float, ...]]:
    result: dict[str, tuple[float, ...]] = {}
    for family, text in items:
        values = tuple(float(v) for v in text.split(",") if v.strip())
        if not values:
            raise ValueError(f"no values supplied for {family}")
        result[family] = values
    return result


def _synthetic_measurement(
    *,
    case_id: str,
    family: str,
    value: float,
    grid_n: int,
    z_mm: list[float],
    gain: float,
    noise_fraction: float,
    seed: int,
) -> IntensityStack:
    truth = simulate_candidate_stack(
        case_id=case_id,
        family=family,
        value=float(value),
        z_m=np.asarray(z_mm, dtype=float) * 1e-3,
        grid_n=int(grid_n),
    )
    values = float(gain) * np.asarray(truth.intensity, dtype=float)
    if float(noise_fraction) > 0.0:
        rng = np.random.default_rng(int(seed))
        sigma = float(noise_fraction) * max(float(np.max(values)), np.finfo(float).tiny)
        values = values + rng.normal(0.0, sigma, size=values.shape)
    values = np.maximum(values, 0.0)
    metadata = {
        **dict(truth.metadata),
        "source": "synthetic_round_trip_measurement",
        "z_reference": "post_axicon_plane",
        "truth_family": family,
        "truth_value": float(value),
        "synthetic_global_gain": float(gain),
        "synthetic_gaussian_noise_fraction_of_stack_peak": float(noise_fraction),
        "synthetic_seed": int(seed),
        "warning": "validation fixture only; not experimental evidence",
    }
    return replace(truth, intensity=values, metadata=metadata)


def _write_outputs(result, handoff, output_dir: Path, source_metadata: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    score_fields = [
        "rank", "family", "value", "parameter_unit", "physical_plane", "operator_class",
        "score", "global_gain", "global_correlation", "valid_fraction",
        "mean_plane_nrmse", "max_plane_nrmse", "mean_shape_nrmse",
    ]
    with (output_dir / "candidate_ranking.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=score_fields)
        writer.writeheader()
        for rank, item in enumerate(result.scores, start=1):
            writer.writerow({
                "rank": rank,
                "family": item.family,
                "value": item.value,
                "parameter_unit": item.metadata["parameter_unit"],
                "physical_plane": item.metadata["physical_plane"],
                "operator_class": item.metadata["operator_class"],
                "score": item.score,
                "global_gain": item.global_gain,
                "global_correlation": item.global_correlation,
                "valid_fraction": item.valid_fraction,
                "mean_plane_nrmse": item.mean_plane_nrmse,
                "max_plane_nrmse": item.max_plane_nrmse,
                "mean_shape_nrmse": item.mean_shape_nrmse,
            })

    payload = asdict(result)
    payload["source_stack_metadata"] = source_metadata
    payload["provenance_warning"] = (
        "Ranks are single-mechanism forward-model hypotheses. They are not proof of a unique bench error."
    )
    (output_dir / "match_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "miao_handoff.json").write_text(
        json.dumps(asdict(handoff), indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path)
    source.add_argument("--synthetic-known", type=_assignment, metavar="FAMILY=VALUE")
    parser.add_argument("--case", choices=("B0", "V1", "V3"))
    parser.add_argument("--grid-n", type=int, default=256)
    parser.add_argument("--families", nargs="+", choices=tuple(FAMILY_REGISTRY), default=None)
    parser.add_argument(
        "--values", action="append", type=_assignment, default=[], metavar="FAMILY=v1,v2,v3",
        help="override candidate grid for one family; may be repeated",
    )
    parser.add_argument("--model-z-offset-mm", type=float, default=None)
    parser.add_argument("--maximum-invalid-fraction", type=float, default=0.02)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiment_simulation_match"))
    parser.add_argument("--synthetic-z-mm", nargs="+", type=float, default=[40.0, 60.0, 80.0])
    parser.add_argument("--synthetic-gain", type=float, default=1.7)
    parser.add_argument("--synthetic-noise-fraction", type=float, default=0.0)
    parser.add_argument("--synthetic-seed", type=int, default=20260907)
    args = parser.parse_args()

    overrides = _parse_value_overrides(args.values)
    families = tuple(args.families or DEFAULT_MATCH_FAMILIES)

    if args.manifest is not None:
        measurement = load_experimental_stack(args.manifest)
        case_id = args.case or measurement.metadata.get("case_id")
        if case_id not in {"B0", "V1", "V3"}:
            raise SystemExit("case must be B0/V1/V3 via --case or manifest case_id")
    else:
        family, value_text = args.synthetic_known
        case_id = args.case or "V1"
        measurement = _synthetic_measurement(
            case_id=case_id,
            family=family,
            value=float(value_text),
            grid_n=int(args.grid_n),
            z_mm=list(args.synthetic_z_mm),
            gain=float(args.synthetic_gain),
            noise_fraction=float(args.synthetic_noise_fraction),
            seed=int(args.synthetic_seed),
        )

    model_z_offset = None if args.model_z_offset_mm is None else float(args.model_z_offset_mm) * 1e-3
    result = match_error_candidates(
        measurement,
        case_id=str(case_id),
        grid_n=int(args.grid_n),
        families=families,
        candidate_values=overrides,
        model_z_offset_m=model_z_offset,
        maximum_invalid_fraction=float(args.maximum_invalid_fraction),
    )
    handoff = build_miao_handoff(result)
    _write_outputs(result, handoff, args.output_dir, dict(measurement.metadata))

    best = result.best
    summary = {
        "outcome": "EXPERIMENT_SIMULATION_SINGLE_ERROR_SCREEN_V1",
        "case_id": str(case_id),
        "best_family": best.family,
        "best_value": best.value,
        "best_score": best.score,
        "global_gain": best.global_gain,
        "miao_handoff_status": handoff.status,
        "correction_map_ready": handoff.correction_map_ready,
        "output_dir": str(args.output_dir),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
