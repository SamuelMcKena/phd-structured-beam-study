from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vbb_study.integrations.zemax.field_exchange import save_field_npz
from vbb_study.integrations.zemax.python_reference import PHASE3_CASES, build_phase2c_vector_focus_reference


def main() -> int:
    parser = argparse.ArgumentParser(description="Export an accepted Phase 2C vector-Debye field into the Phase 3 NPZ contract.")
    parser.add_argument("--case", required=True, choices=PHASE3_CASES)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--high-resolution", action="store_true")
    args = parser.parse_args()
    output = args.output or Path("outputs/validation/phase3_zemax/python_reference") / f"{args.case}_phase2c_vector_focus.npz"
    plane = build_phase2c_vector_focus_reference(args.case, high_resolution=args.high_resolution)
    save_field_npz(output, plane)
    print(f"Saved {args.case} canonical Python reference: {output}")
    print("Plane: phase2c_matched_objective_focal_plane_z0")
    print("Use this for Zemax comparison only when the mapped POP output is the same physical plane.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
