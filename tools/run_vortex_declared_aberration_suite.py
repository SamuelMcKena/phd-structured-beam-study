from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from vbb_study.digital_twin.vortex_system_route import build_system_route
from vbb_study.digital_twin.vortex_wavefront_errors import (
    ZERNIKE_NAMES,
    opd_to_phase_rad,
    zernike_opd_map_m,
)
from vbb_study.equations.fields import make_xy_grid
from vbb_study.equations.propagation import angular_spectrum_propagate_bl


EPS = np.finfo(float).tiny


def _metrics(I: np.ndarray, grid) -> dict[str, float]:
    arr = np.asarray(I, dtype=float)
    total = float(np.sum(arr))
    X = np.asarray(grid["X"], dtype=float)
    Y = np.asarray(grid["Y"], dtype=float)
    return {
        "peak_au": float(np.max(arr)),
        "power_au": total * float(grid["dx"]) ** 2,
        "centroid_x_m": float(np.sum(arr * X) / max(total, EPS)),
        "centroid_y_m": float(np.sum(arr * Y) / max(total, EPS)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run declared-plane Zernike OPD sensitivity studies.")
    parser.add_argument("--cases", nargs="+", default=["B0", "V1", "V3"])
    parser.add_argument("--planes", nargs="+", choices=["input", "lens1", "lens2"], default=["input", "lens1", "lens2"])
    parser.add_argument("--modes", nargs="+", choices=list(ZERNIKE_NAMES), default=list(ZERNIKE_NAMES))
    parser.add_argument("--waves", nargs="+", type=float, default=[-0.5, -0.25, 0.0, 0.25, 0.5])
    parser.add_argument("--grid-n", type=int, default=1536)
    parser.add_argument("--pupil-radius-mm", type=float, default=2.0)
    parser.add_argument("--z-mm", type=float, default=60.0)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/validation/vortex_declared_aberrations"))
    args = parser.parse_args()

    N = int(args.grid_n)
    window = 10e-3
    grid_for_map = make_xy_grid(N, window / N)
    wavelength = 1029e-9
    pupil = float(args.pupil_radius_mm) * 1e-3
    rows = []

    for case_id in args.cases:
        for plane in args.planes:
            for mode in args.modes:
                for waves in args.waves:
                    opd = zernike_opd_map_m(
                        mode,
                        grid_for_map,
                        wavelength_m=wavelength,
                        waves_rms=float(waves),
                        pupil_radius_m=pupil,
                    )
                    kwargs = {}
                    if plane == "input":
                        # A phase screen immediately before SLM1 commutes with the
                        # phase-only SLM1 action at the same transverse plane.
                        kwargs["slm1_static_phase_map_rad"] = opd_to_phase_rad(opd, wavelength)
                    elif plane == "lens1":
                        kwargs["lens1_opd_map_m"] = opd
                    else:
                        kwargs["lens2_opd_map_m"] = opd

                    route = build_system_route(case_id, grid_n=N, **kwargs)
                    prop = angular_spectrum_propagate_bl(
                        route["post_axicon"],
                        dict(route["grid"]),
                        wavelength,
                        float(args.z_mm) * 1e-3,
                        n_medium=1.0,
                        bandlimit=True,
                        include_evanescent=True,
                    )
                    rows.append({
                        "case_id": case_id,
                        "application_plane": plane,
                        "mode": mode,
                        "waves_rms": float(waves),
                        "pupil_radius_m": pupil,
                        "grid_n": N,
                        "z_reference_m": float(args.z_mm) * 1e-3,
                        "fourf_selected_fraction": route["metadata"]["fourf"]["iris_selected_power_fraction"],
                        **_metrics(np.abs(prop) ** 2, route["grid"]),
                    })

    args.output_root.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_root / "declared_aberration_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    manifest = {
        "outcome": "VORTEX-DECLARED-ABERRATION-SENSITIVITY",
        "report_figures_authorised": False,
        "planes": list(args.planes),
        "modes": list(args.modes),
        "waves_rms": list(args.waves),
        "pupil_radius_mm": float(args.pupil_radius_mm),
        "policy": (
            "These are declared OPD sensitivities at named planes. They are not physical substitutes "
            "for lens/axicon/beam misalignment. Absolute values require measured wavefront or lens OPD data."
        ),
        "input_plane_note": (
            "The input phase screen is injected through the co-located SLM1 static-phase hook; phase factors commute at that plane."
        ),
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
