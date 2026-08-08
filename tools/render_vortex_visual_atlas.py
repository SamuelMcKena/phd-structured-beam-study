from __future__ import annotations

import argparse
import json
from pathlib import Path

from vbb_study.digital_twin.vortex_visual_atlas_figures import run_visual_atlas_figures


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render repaired-route vortex Bessel visual atlas figures.")
    parser.add_argument("--figure-root", type=Path, default=Path("outputs/figures/vortex_visual_atlas"))
    parser.add_argument("--grid-n", type=int, default=3072)
    parser.add_argument("--cases", nargs="+", default=["B0", "V1", "V3"])
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_visual_atlas_figures(
        figure_root=args.figure_root,
        grid_n=int(args.grid_n),
        cases=tuple(args.cases),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
