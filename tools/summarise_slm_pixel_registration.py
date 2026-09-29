"""Print the headline table for one registration-study run.

Reports, per vortex charge and beam radius, the registration-induced spread
across the sub-pixel offset sweep, the fixed pixelation penalty relative to the
continuous ideal, and the Bessel-region consequence.  Reads only the CSVs, so it
never re-runs any physics.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any


def _f(row: dict[str, Any], key: str) -> float:
    v = row.get(key, "")
    if v in ("", None, "nan"):
        return float("nan")
    try:
        return float(v)
    except ValueError:
        return float("nan")


def _finite(values):
    return [v for v in values if v == v]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path,
                   default=Path("outputs/validation/slm_pixel_registration"))
    p.add_argument("--tag", default="isolation")
    p.add_argument("--metric", default=None,
                   help="override the registration metric column")
    args = p.parse_args()

    path = args.root / args.tag / "registration_metrics.csv"
    if not path.exists():
        raise SystemExit(f"no metrics at {path}")
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if not rows:
        raise SystemExit("metrics file is empty")

    # Prefer the crop-free post-iris metric when the run recorded it.
    metric = args.metric
    if metric is None:
        metric = ("postiris_infidelity_vs_registered0"
                  if any(_f(r, "postiris_infidelity_vs_registered0") == _f(
                      r, "postiris_infidelity_vs_registered0") for r in rows)
                  else "before_infidelity_vs_registered0")
    penalty = metric.replace("_vs_registered0", "_vs_continuous")

    arm = rows[0].get("arm", "?")
    spp = _f(rows[0], "samples_per_pixel")
    comm = rows[0].get("pitch_commensurate", "?")
    print(f"tag={args.tag}  arm={arm}  samples/pixel={spp:.2f}  "
          f"pitch_commensurate={comm}")
    print(f"registration metric: {metric}")
    print()

    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r["series"] == "continuous":
            continue
        groups[(_f(r, "beam_radius_px"), int(_f(r, "charge")), r["dof"])].append(r)

    hdr = (f"{'w[px]':>8} {'l':>4} {'dof':>13} {'reg spread':>12} {'pix penalty':>12} "
           f"{'ring asym':>11} {'core dark':>10} {'zone spread':>12}")
    print(hdr)
    print("-" * len(hdr))
    for key in sorted(groups):
        w, ell, dof = key
        items = groups[key]
        if dof == "none" and len(groups) > 1:
            # zero-offset reference row folds into each dof group below
            continue
        pool = items + [r for r in rows
                        if r["series"] == "registered"
                        and _f(r, "beam_radius_px") == w
                        and int(_f(r, "charge")) == ell]
        reg = _finite([_f(r, metric) for r in pool])
        pen = _finite([_f(r, penalty) for r in pool])
        asym = _finite([_f(r, "before_ring_asymmetry_rms") for r in pool])
        core = _finite([_f(r, "before_core_darkness") for r in pool])
        zone = _finite([_f(r, "after_bessel_zone_fwhm_m") for r in pool])
        if not reg:
            continue
        zone_spread = ((max(zone) - min(zone)) / max(abs(sum(zone) / len(zone)), 1e-30)
                       if len(zone) > 1 else float("nan"))
        print(f"{w:8.1f} {ell:4d} {dof:>13} {max(reg):12.3e} "
              f"{(sum(pen)/len(pen) if pen else float('nan')):12.3e} "
              f"{(max(asym)-min(asym) if len(asym) > 1 else float('nan')):11.3e} "
              f"{(sum(core)/len(core) if core else float('nan')):10.4f} "
              f"{zone_spread:12.3e}")

    print()
    print("reg spread   = max infidelity across the sub-pixel offset sweep, "
          "vs the zero-offset route")
    print("pix penalty  = mean infidelity vs the continuous (unpixelated) ideal")
    print("ring asym    = spread of ring azimuthal asymmetry across the sweep")
    print("zone spread  = relative spread of Bessel-zone length across the sweep")


if __name__ == "__main__":
    main()
