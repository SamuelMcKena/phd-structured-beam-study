"""q=20 correction v14: coarse-to-fine annular-spliced modal solve.

This is the rapid-iteration companion to v13.  The modal search is performed on
1024 samples to reduce optimisation cost, while candidate selection remains
multi-plane and topology guarded.  After coefficients are frozen, the exact
same command is evaluated by v13/v11 production machinery at 4096 samples.
Thus the speed-up applies only to search; the final evidence bar is unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
EXP = ROOT / "notebooks" / "experimental" / "axicon_aberration_correction"
for p in (ROOT, TOOLS, EXP):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import solve_q20_slm2_multiplane_circular_v10 as v10  # noqa: E402
import solve_q20_slm2_annular_splice_v13 as v13  # noqa: E402

SEARCH_N = 1024


def run(source_dir: Path, candidate_json: Path, residual_json: Path, out: Path) -> dict:
    # v13 and its imported helpers read v10.OPT_N at call time.  Override only
    # the optimisation/evaluation grid; v11.production remains hard-coded to
    # 4096 samples for the final evidence package.
    old_n = int(v10.OPT_N)
    old_steps = tuple(v13.STEP_SCHEDULE_RAD)
    try:
        v10.OPT_N = SEARCH_N
        v13.STEP_SCHEDULE_RAD = (0.24, 0.12, 0.060)
        result = v13.run(source_dir, candidate_json, residual_json, out)
    finally:
        v10.OPT_N = old_n
        v13.STEP_SCHEDULE_RAD = old_steps

    result = dict(result)
    result["status"] = "q20_annular_spliced_coarse_to_fine_v14"
    result["search_grid_n"] = SEARCH_N
    result["production_grid_n"] = 4096
    result["coarse_to_fine_policy"] = (
        "modal coefficients selected on 1024-sample multi-plane model; frozen command "
        "must independently satisfy the full 4096 production/topology/concentricity gate"
    )
    out = Path(out)
    (out / "v14_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source-dir", type=Path, default=EXP / "outputs" / "digital_twin_correction")
    p.add_argument("--candidate-json", type=Path, default=EXP / "candidates" / "q20_detector_aware_model_v3_candidate.json")
    p.add_argument("--residual-json", type=Path, default=EXP / "candidates" / "q20_miao_initialized_complex_residual_v1.json")
    p.add_argument("--out", type=Path, default=ROOT / "outputs" / "validation" / "q20_slm2_annular_coarsefine_v14")
    a = p.parse_args(); run(a.source_dir, a.candidate_json, a.residual_json, a.out)


if __name__ == "__main__":
    main()
