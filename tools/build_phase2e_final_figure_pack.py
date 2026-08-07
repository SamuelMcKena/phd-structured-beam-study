"""Build the authorised Phase 2E final source-propagation figure pack."""

from __future__ import annotations

import json

from vbb_study.digital_twin.phase2e_final_source_figures import build_final_figure_pack


def main() -> int:
    result = build_final_figure_pack()
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
