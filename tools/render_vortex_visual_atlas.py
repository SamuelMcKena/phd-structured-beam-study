from __future__ import annotations

import argparse
import json
from pathlib import Path

from vbb_study.digital_twin.vortex_visual_atlas_figures import run_visual_atlas_figures


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render physically grounded vortex-Bessel visual atlas figures."
    )
    parser.add_argument(
        "--figure-root",
        type=Path,
        default=Path("outputs/figures/vortex_visual_atlas"),
    )
    parser.add_argument("--grid-n", type=int, default=3072)
    parser.add_argument("--cases", nargs="+", default=["B0", "V1", "V3"])
    parser.add_argument(
        "--family",
        choices=["parameter", "manufacturing", "aberration", "alignment"],
        default=None,
        help="Render only one sweep family.",
    )
    parser.add_argument(
        "--parameter",
        default=None,
        help="Render only one parameter within the selected family.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_visual_atlas_figures(
        figure_root=args.figure_root,
        grid_n=int(args.grid_n),
        cases=tuple(args.cases),
        family_filter=args.family,
        parameter_filter=args.parameter,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
