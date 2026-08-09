from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from vbb_study.digital_twin.phase2e_spectral_propagation import build_dense_spectral_propagation
from vbb_study.digital_twin.vortex_system_error_sweeps import (
    blocked_or_data_driven_families,
    system_sweep_registry,
)
from vbb_study.digital_twin.vortex_system_route import build_system_route
from vbb_study.equations.propagation import angular_spectrum_propagate_bl


EPS = np.finfo(float).tiny


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _metrics(I: np.ndarray, grid: dict[str, Any]) -> dict[str, float]:
    arr = np.maximum(np.asarray(I, dtype=float), 0.0)
    total = float(np.sum(arr))
    X = np.asarray(grid["X"], dtype=float)
    Y = np.asarray(grid["Y"], dtype=float)
    peak = float(np.max(arr))
    return {
        "peak_au": peak,
        "power_au": total * float(grid["dx"]) ** 2,
        "centroid_x_m": float(np.sum(arr * X) / max(total, EPS)),
        "centroid_y_m": float(np.sum(arr * Y) / max(total, EPS)),
    }


def _crop_square(arr: np.ndarray, x: np.ndarray, halfwidth: float) -> tuple[np.ndarray, np.ndarray]:
    mask = np.abs(x) <= float(halfwidth)
    return x[mask], np.asarray(arr)[np.ix_(mask, mask)]


def _family_component_image(family: str, route: dict[str, Any], nominal: dict[str, Any]) -> tuple[np.ndarray, str, str]:
    if family.startswith("beam_"):
        return np.abs(route["input_beam"]) ** 2, "input beam intensity", "intensity"
    if family.startswith("slm") or family.startswith("fourf"):
        return np.abs(route["fourier_plane_before_iris"]) ** 2, "4F Fourier plane before iris", "intensity"
    # Axicon families: show phase difference of the physical axicon action relative
    # to nominal, removing field amplitude wherever the incident field is tiny.
    ratio = route["post_axicon_local"] / np.where(
        np.abs(route["field_on_axicon_plane"]) > 1e-9,
        route["field_on_axicon_plane"],
        1.0,
    )
    ratio0 = nominal["post_axicon_local"] / np.where(
        np.abs(nominal["field_on_axicon_plane"]) > 1e-9,
        nominal["field_on_axicon_plane"],
        1.0,
    )
    return np.angle(ratio * np.conj(ratio0)), "axicon phase difference vs nominal", "phase"


def run_family(
    family: str,
    *,
    case_id: str,
    grid_n: int,
    z_reference_m: float,
    output_root: Path,
    figure_root: Path,
) -> list[dict[str, Any]]:
    import matplotlib.pyplot as plt

    registry = system_sweep_registry()
    spec = registry[family]
    values = tuple(spec["values"])
    nominal_config = spec["builder"](values[int(np.argmin([abs(float(v)) if np.isfinite(v) else 1e99 for v in values]))])
    # For ratio/scale sweeps the nominal is 1 rather than 0/inf.
    if any(np.isclose(float(v), 1.0) for v in values if np.isfinite(v)):
        nominal_value = min((v for v in values if np.isfinite(v)), key=lambda v: abs(float(v) - 1.0))
    elif any(not np.isfinite(v) for v in values):
        nominal_value = next(v for v in values if not np.isfinite(v))
    else:
        nominal_value = min(values, key=lambda v: abs(float(v)))
    nominal_config = spec["builder"](nominal_value)
    nominal_route = build_system_route(case_id, grid_n=grid_n, config=nominal_config)

    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    z = np.arange(5e-3, 140e-3 + 2e-3, 2e-3)
    transverse = np.linspace(-0.18e-3, 0.18e-3, 321)

    for value in values:
        config = spec["builder"](value)
        route = build_system_route(case_id, grid_n=grid_n, config=config)
        grid = route["grid"]
        wavelength = float(route["metadata"]["wavelength_m"])
        propagated = angular_spectrum_propagate_bl(
            route["post_axicon"],
            dict(grid),
            wavelength,
            z_reference_m,
            n_medium=1.0,
            bandlimit=True,
            include_evanescent=True,
        )
        Ixy = np.abs(propagated) ** 2
        dense = build_dense_spectral_propagation(
            grid=grid,
            wavelength_m=wavelength,
            z_values_m=z,
            transverse_coordinates_m=transverse,
            scalar_field=route["post_axicon"],
            source_label=f"{case_id}:{family}={value}",
        )
        component, component_label, component_kind = _family_component_image(family, route, nominal_route)
        row = {
            "case_id": case_id,
            "family": family,
            "value": value,
            "units": spec["units"],
            "fidelity": spec["fidelity"],
            "grid_n": grid_n,
            "z_reference_m": z_reference_m,
            "fourf_iris_selected_fraction": route["metadata"]["fourf"]["iris_selected_power_fraction"],
            "axicon_tilt_status": route["metadata"]["axicon_tilt_status"],
            **_metrics(Ixy, dict(grid)),
        }
        rows.append(row)
        records.append({
            "value": value,
            "route": route,
            "Ixy": Ixy,
            "xz": np.asarray(dense.xz_intensity),
            "component": component,
            "component_label": component_label,
            "component_kind": component_kind,
        })

    x = np.asarray(nominal_route["grid"]["x"], dtype=float)
    xy_common = max(float(np.max(rec["Ixy"])) for rec in records)
    xz_common = max(float(np.max(rec["xz"])) for rec in records)
    comp_abs = max(float(np.max(np.abs(rec["component"]))) for rec in records)

    fig, axes = plt.subplots(3, len(records), figsize=(3.1 * len(records), 8.5), constrained_layout=True, squeeze=False)
    for col, rec in enumerate(records):
        value = rec["value"]
        comp = rec["component"]
        if rec["component_kind"] == "intensity":
            # Full-ish component plane crop: Fourier planes need millimetres.
            half = 3.0e-3 if (family.startswith("slm") or family.startswith("fourf")) else 2.5e-3
            xc, cc = _crop_square(comp, x, half)
            im0 = axes[0, col].imshow(
                cc / max(comp_abs, EPS), origin="lower",
                extent=[xc[0]*1e3, xc[-1]*1e3, xc[0]*1e3, xc[-1]*1e3],
                vmin=0, vmax=1, cmap="inferno"
            )
            axes[0, col].set_xlabel("x (mm)")
            axes[0, col].set_ylabel("y (mm)")
        else:
            xc, cc = _crop_square(comp, x, 2.5e-3)
            im0 = axes[0, col].imshow(
                cc, origin="lower",
                extent=[xc[0]*1e3, xc[-1]*1e3, xc[0]*1e3, xc[-1]*1e3],
                vmin=-np.pi, vmax=np.pi, cmap="twilight"
            )
            axes[0, col].set_xlabel("x (mm)")
            axes[0, col].set_ylabel("y (mm)")
        axes[0, col].set_title(str(value))

        xc2, Icrop = _crop_square(rec["Ixy"], x, 0.18e-3)
        im1 = axes[1, col].imshow(
            Icrop / max(xy_common, EPS), origin="lower",
            extent=[xc2[0]*1e6, xc2[-1]*1e6, xc2[0]*1e6, xc2[-1]*1e6],
            vmin=0, vmax=1, cmap="inferno"
        )
        axes[1, col].set_xlabel("x (µm)")
        axes[1, col].set_ylabel("y (µm)")

        im2 = axes[2, col].imshow(
            (rec["xz"] / max(xz_common, EPS)).T,
            origin="lower", aspect="auto",
            extent=[z[0]*1e3, z[-1]*1e3, transverse[0]*1e6, transverse[-1]*1e6],
            vmin=0, vmax=1, cmap="inferno"
        )
        axes[2, col].set_xlabel("z from axicon (mm)")
        axes[2, col].set_ylabel("x (µm)")

    fig.colorbar(im0, ax=axes[0, :].tolist(), label=records[0]["component_label"])
    fig.colorbar(im1, ax=axes[1, :].tolist(), label="I / sweep-global max")
    fig.colorbar(im2, ax=axes[2, :].tolist(), label="I / sweep-global max")
    fig.suptitle(f"{case_id} — {family} | {spec['fidelity']}")
    path = figure_root / case_id / f"{family}_system_diagnostic.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run physically placed vortex/Bessel system-error sweeps.")
    parser.add_argument("--cases", nargs="+", default=["B0", "V1", "V3"])
    parser.add_argument("--families", nargs="+", default=["all"])
    parser.add_argument("--grid-n", type=int, default=1536)
    parser.add_argument("--z-mm", type=float, default=60.0)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/validation/vortex_system_errors"))
    parser.add_argument("--figure-root", type=Path, default=Path("outputs/figures/vortex_system_errors"))
    args = parser.parse_args()

    registry = system_sweep_registry()
    families = list(registry) if args.families == ["all"] else list(args.families)
    unknown = [name for name in families if name not in registry]
    if unknown:
        raise SystemExit(f"unknown families: {unknown}")

    rows: list[dict[str, Any]] = []
    for case_id in args.cases:
        for family in families:
            print(f"running {case_id} / {family}", flush=True)
            rows.extend(
                run_family(
                    family,
                    case_id=case_id,
                    grid_n=int(args.grid_n),
                    z_reference_m=float(args.z_mm) * 1e-3,
                    output_root=args.output_root,
                    figure_root=args.figure_root,
                )
            )
    _write_csv(args.output_root / "system_error_metrics.csv", rows)
    manifest = {
        "outcome": "VORTEX-SYSTEM-ERROR-RESEARCH-SUITE",
        "report_figures_authorised": False,
        "cases": list(args.cases),
        "families": families,
        "grid_n": int(args.grid_n),
        "z_reference_mm": float(args.z_mm),
        "executed_registry": {
            name: {k: v for k, v in registry[name].items() if k != "builder"}
            for name in families
        },
        "blocked_or_data_driven": blocked_or_data_driven_families(),
        "policy": (
            "Every executable family is a controlled relative-sensitivity study. "
            "Absolute bench-error magnitudes require the listed calibration data and independent validation gates."
        ),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "system_error_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()
