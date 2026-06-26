"""Stage 9B.0 nominal 4F candidate-beam atlas.

This module builds opt-in, unvalidated candidate packages around the nominal
F300 scalar 4F forward model. It does not make the physical 4F route ready,
does not model a camera, and does not introduce materials, inverse correction,
AI, or an axicon phase on SLM2.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from vbb_study.equations.fields import make_xy_grid
from vbb_study.equations.holography import wrap_phase_rad
from vbb_study.digital_twin.nominal_f300_4f import (
    CLAIM_BOUNDARY_LABELS,
    FINAL_EXPORT_ALLOWED,
    MODEL_LABEL,
    NominalF300Config,
    NominalF300Run,
    carrier_phase,
    config_from_profile,
    load_nominal_f300_profile,
    nominal_4f_sanity_report,
    phase_export_payload,
    replace_config,
    run_nominal_f300_4f,
    run_to_manifest,
    vortex_phase,
    write_csv,
    write_json,
)


STAGE = "9B.0"
STUDY_NAME = "cslm_nominal_4f_candidate_atlas_v1"
STUDY_PATH = Path("configs/studies/cslm_nominal_4f_candidate_atlas_v1.json")
DEFAULT_RUN_ID = "stage9b0_nominal_4f_candidate_atlas"
PACKAGE_ROOT = Path("outputs/nominal_4f_candidate_runs")
FIGURE_COMPONENT_SEQUENCE = Path("outputs/figures/digital_twin/stage9b0_nominal_f300_4f_component_sequence.png")
FIGURE_STOP_ROBUSTNESS = Path("outputs/figures/digital_twin/stage9b0_nominal_f300_4f_stop_robustness.png")
FIGURE_CANDIDATE_ATLAS = Path("outputs/figures/digital_twin/stage9b0_nominal_f300_candidate_atlas.png")

CANDIDATE_STATUS_LABELS = (
    "nominally_simulated",
    "hardware_command_exportable",
    "not_bench_validated",
)

HANDOFF_MODES = (
    "unknown_not_simulated",
    "nominal_user_scenario",
    "later_measured_bench_geometry",
)


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    candidate_family: str
    ell: int = 0
    input_beam_radius_m: float = 0.0008
    input_beam_decentre_x_m: float = 0.0
    input_beam_decentre_y_m: float = 0.0
    input_beam_ellipticity: float = 1.0
    input_beam_rotation_deg: float = 0.0
    command_domain_carrier_cycles_x: float = 8.0
    command_domain_carrier_cycles_y: float = 0.0
    numerical_model_carrier_cycles_x: float = 8.0
    numerical_model_carrier_cycles_y: float = 0.0
    pinhole_radius_m: float = 0.00018
    pinhole_offset_x_m: float = 0.00031
    pinhole_offset_y_m: float = 0.0
    lens_clear_radius_m: float = 0.0032
    phase_generator: str = "SLM1_vortex_or_flat_plus_SLM2_carrier"
    hardware_realizability_status: str = "hardware_command_exportable_not_bench_validated"
    export_permission: str = "hardware_command_exportable_not_final"
    required_components: tuple[str, ...] = (
        "SLM1_phase_plane",
        "SLM2_carrier_phase_plane",
        "nominal_F300_4F_relay",
        "nominal_Fourier_stop",
    )
    known_model_limits: tuple[str, ...] = (
        "nominal 4F geometry only",
        "no physical 4F readiness",
        "no camera model",
        "no material response",
        "SLM2 contains carrier/future correction only, not an axicon phase",
    )


def build_candidate_atlas_config() -> dict[str, Any]:
    """Return the machine-readable Stage 9B.0 atlas contract."""
    return {
        "stage": STAGE,
        "study_name": STUDY_NAME,
        "study_status": "nominal_4f_candidate_atlas_not_bench_validated",
        "model_label": MODEL_LABEL,
        "claim_boundary_labels": list(CLAIM_BOUNDARY_LABELS),
        "candidate_status_labels": list(CANDIDATE_STATUS_LABELS),
        "final_export_allowed": FINAL_EXPORT_ALLOWED,
        "slm_role_contract": {
            "SLM1": "vortex/structured phase conditioning source",
            "SLM2": "carrier grating and future correction map only",
            "SLM2_forbidden_content": ["axicon phase", "material correction", "validated inverse correction"],
        },
        "nominal_4f_contract": {
            "component_sequence": [
                "SLM2",
                "300 mm free-space segment",
                "Lens 1 f=300 mm",
                "300 mm free-space segment",
                "Fourier/pinhole plane",
                "300 mm free-space segment",
                "Lens 2 f=300 mm",
                "300 mm free-space segment",
                "nominal relay output",
            ],
            "physical_4f_readiness": "blocked",
            "carrier_coordinate_status": "nominal_model_not_bench_calibrated",
        },
        "candidate_families": {
            "gaussian_reference": {"ell_values": [0]},
            "vortex_charge_sweep": {"ell_values": [1, 2, 3, 4]},
            "input_beam_size_variants": {"beam_radius_m": [0.00055, 0.0008, 0.00105]},
            "input_decentre_variants": {
                "decentre_m": [[0.0, 0.0], [-0.00008, 0.0], [0.00008, 0.0], [0.0, -0.00008], [0.0, 0.00008]]
            },
            "pinhole_robustness_variants": {
                "pinhole_radius_m": [0.00012, 0.00018, 0.00026],
                "pinhole_offset_m": [[0.00031, 0.0], [0.00023, 0.0], [0.00039, 0.0], [0.00031, 0.00008]],
            },
        },
        "robustness_sweeps": {
            "carrier_vs_stop_offset_transmission": {
                "carrier_cycles_x": [-12, -8, -4, 0, 4, 8, 12],
                "stop_offset_delta_x_m": [-0.00016, -0.00008, 0.0, 0.00008, 0.00016],
            },
            "pinhole_radius_vs_stop_offset_transmission": {
                "pinhole_radius_m": [0.00012, 0.00018, 0.00026, 0.00034],
                "stop_offset_delta_x_m": [-0.00012, 0.0, 0.00012],
            },
            "beam_radius_vs_relay_quality": {"beam_radius_m": [0.00055, 0.0008, 0.00105]},
            "input_decentre_vs_output_centroid": {
                "decentre_m": [[-0.00008, 0.0], [0.0, 0.0], [0.00008, 0.0], [0.0, -0.00008], [0.0, 0.00008]]
            },
        },
        "package_contract": {
            "root": str(PACKAGE_ROOT).replace("\\", "/"),
            "required_files": [
                "run_manifest.json",
                "candidate_manifest.json",
                "nominal_4f_profile_snapshot.json",
                "SLM1 phase_rad.npy",
                "SLM1 quantised_rad.npy",
                "SLM1 gray.png",
                "SLM2 phase_rad.npy",
                "SLM2 quantised_rad.npy",
                "SLM2 gray.png",
                "fourier_plane_pre_stop.png",
                "fourier_stop_transmission.png",
                "fourier_plane_post_stop.png",
                "nominal_relay_output_xy.png",
                "energy_ledger.csv",
                "robustness_summary.csv",
                "claim_boundary.md",
            ],
        },
    }


def write_candidate_atlas_config(path: str | Path = STUDY_PATH) -> Path:
    return write_json(path, build_candidate_atlas_config())


def build_candidate_specs(config: Mapping[str, Any] | None = None) -> list[CandidateSpec]:
    """Build the default atlas candidates from the Stage 9B.0 contract."""
    _ = config or build_candidate_atlas_config()
    specs: list[CandidateSpec] = [
        CandidateSpec("gaussian_reference", "gaussian_reference", ell=0),
    ]
    specs.extend(
        CandidateSpec(f"vortex_ell_{ell}", "vortex_charge_sweep", ell=ell)
        for ell in (1, 2, 3, 4)
    )
    specs.extend(
        [
            CandidateSpec("beam_radius_small", "input_beam_size_variants", ell=2, input_beam_radius_m=0.00055),
            CandidateSpec("beam_radius_nominal", "input_beam_size_variants", ell=2, input_beam_radius_m=0.0008),
            CandidateSpec("beam_radius_large", "input_beam_size_variants", ell=2, input_beam_radius_m=0.00105),
        ]
    )
    specs.extend(
        [
            CandidateSpec("input_decentre_centered", "input_decentre_variants", ell=2),
            CandidateSpec("input_decentre_x_minus", "input_decentre_variants", ell=2, input_beam_decentre_x_m=-0.00008),
            CandidateSpec("input_decentre_x_plus", "input_decentre_variants", ell=2, input_beam_decentre_x_m=0.00008),
            CandidateSpec("input_decentre_y_minus", "input_decentre_variants", ell=2, input_beam_decentre_y_m=-0.00008),
            CandidateSpec("input_decentre_y_plus", "input_decentre_variants", ell=2, input_beam_decentre_y_m=0.00008),
        ]
    )
    specs.extend(
        [
            CandidateSpec("pinhole_centered", "pinhole_robustness_variants", ell=2, pinhole_offset_x_m=0.00031),
            CandidateSpec("pinhole_offset_x_minus", "pinhole_robustness_variants", ell=2, pinhole_offset_x_m=0.00023),
            CandidateSpec("pinhole_offset_x_plus", "pinhole_robustness_variants", ell=2, pinhole_offset_x_m=0.00039),
            CandidateSpec("pinhole_offset_y_plus", "pinhole_robustness_variants", ell=2, pinhole_offset_x_m=0.00031, pinhole_offset_y_m=0.00008),
            CandidateSpec("pinhole_radius_small", "pinhole_robustness_variants", ell=2, pinhole_radius_m=0.00012),
            CandidateSpec("pinhole_radius_large", "pinhole_robustness_variants", ell=2, pinhole_radius_m=0.00026),
        ]
    )
    return specs


def _grid_for_config(config: NominalF300Config) -> dict[str, Any]:
    return make_xy_grid(int(config.simulation_grid_size), config.dx_m)


def config_for_candidate(spec: CandidateSpec, base_config: NominalF300Config | None = None) -> NominalF300Config:
    base = base_config or config_from_profile()
    return replace_config(
        base,
        input_beam_radius_m=spec.input_beam_radius_m,
        input_beam_decentre_x_m=spec.input_beam_decentre_x_m,
        input_beam_decentre_y_m=spec.input_beam_decentre_y_m,
        input_beam_ellipticity=spec.input_beam_ellipticity,
        input_beam_rotation_deg=spec.input_beam_rotation_deg,
        command_domain_carrier_cycles_x=spec.command_domain_carrier_cycles_x,
        command_domain_carrier_cycles_y=spec.command_domain_carrier_cycles_y,
        numerical_model_carrier_cycles_x=spec.numerical_model_carrier_cycles_x,
        numerical_model_carrier_cycles_y=spec.numerical_model_carrier_cycles_y,
        pinhole_radius_m=spec.pinhole_radius_m,
        pinhole_offset_x_m=spec.pinhole_offset_x_m,
        pinhole_offset_y_m=spec.pinhole_offset_y_m,
        lens_clear_radius_m=spec.lens_clear_radius_m,
    )


def simulate_candidate(spec: CandidateSpec, base_config: NominalF300Config | None = None) -> NominalF300Run:
    cfg = config_for_candidate(spec, base_config)
    grid = _grid_for_config(cfg)
    phase = vortex_phase(grid, spec.ell)
    return run_nominal_f300_4f(cfg, slm1_phase_rad=phase)


def relay_output_candidate_metrics(run: NominalF300Run) -> dict[str, Any]:
    cfg = run.config
    input_power = float(run.component_energy_ledger[0]["energy_before_arb_m2"])
    output_power = float(run.component_energy_ledger[-1]["energy_after_arb_m2"])
    output = run.diagnostics["nominal_relay_output_metrics"]
    fourier = run.diagnostics["fourier_plane_pre_stop_metrics"]
    half = max(0.5 * cfg.simulation_plane_width_m, 1e-30)
    centroid_radius = float(np.hypot(output["centroid_x_m"], output["centroid_y_m"]))
    margin_score = float(np.clip(output["field_of_view_margin_m"] / half, 0.0, 1.0))
    transmitted = float(output_power / max(input_power, 1e-30))
    warning_penalty = 0.10 * len(run.warnings)
    score = float(np.clip(0.55 * min(transmitted, 1.0) + 0.25 * margin_score + 0.20 * (1.0 - min(centroid_radius / half, 1.0)) - warning_penalty, 0.0, 1.0))
    return {
        "relative_transmitted_energy": transmitted,
        "pinhole_transmitted_fraction": float(run.diagnostics["pinhole_transmitted_fraction"]),
        "lens1_pupil_transmitted_fraction": float(run.diagnostics["lens1_pupil_transmitted_fraction"]),
        "lens2_pupil_transmitted_fraction": float(run.diagnostics["lens2_pupil_transmitted_fraction"]),
        "output_centroid_x_m": float(output["centroid_x_m"]),
        "output_centroid_y_m": float(output["centroid_y_m"]),
        "output_second_moment_width_x_m": float(output["second_moment_width_x_m"]),
        "output_second_moment_width_y_m": float(output["second_moment_width_y_m"]),
        "output_field_of_view_margin_m": float(output["field_of_view_margin_m"]),
        "fourier_centroid_x_m": float(fourier["centroid_x_m"]),
        "fourier_centroid_y_m": float(fourier["centroid_y_m"]),
        "relay_quality_score": score,
        "pupil_clipping_flag": bool(
            run.diagnostics["lens1_pupil_transmitted_fraction"] < 0.98
            or run.diagnostics["lens2_pupil_transmitted_fraction"] < 0.98
        ),
        "pinhole_clipping_flag": bool(run.diagnostics["pinhole_transmitted_fraction"] < 0.95),
        "warnings": "; ".join(run.warnings),
    }


def _base_fast_config(base_config: NominalF300Config | None = None) -> NominalF300Config:
    return base_config or NominalF300Config.fast()


def run_candidate_robustness(spec: CandidateSpec, base_config: NominalF300Config | None = None) -> list[dict[str, Any]]:
    base = _base_fast_config(base_config)
    rows: list[dict[str, Any]] = []

    def _append(sweep_name: str, cfg: NominalF300Config, extra: Mapping[str, Any]) -> None:
        local_spec = CandidateSpec(
            candidate_id=spec.candidate_id,
            candidate_family=spec.candidate_family,
            ell=spec.ell,
            input_beam_radius_m=cfg.input_beam_radius_m,
            input_beam_decentre_x_m=cfg.input_beam_decentre_x_m,
            input_beam_decentre_y_m=cfg.input_beam_decentre_y_m,
            input_beam_ellipticity=cfg.input_beam_ellipticity,
            input_beam_rotation_deg=cfg.input_beam_rotation_deg,
            command_domain_carrier_cycles_x=cfg.command_domain_carrier_cycles_x,
            command_domain_carrier_cycles_y=cfg.command_domain_carrier_cycles_y,
            numerical_model_carrier_cycles_x=cfg.numerical_model_carrier_cycles_x,
            numerical_model_carrier_cycles_y=cfg.numerical_model_carrier_cycles_y,
            pinhole_radius_m=cfg.pinhole_radius_m,
            pinhole_offset_x_m=cfg.pinhole_offset_x_m,
            pinhole_offset_y_m=cfg.pinhole_offset_y_m,
            lens_clear_radius_m=cfg.lens_clear_radius_m,
        )
        run = simulate_candidate(local_spec, cfg)
        metrics = relay_output_candidate_metrics(run)
        rows.append(
            {
                "candidate_id": spec.candidate_id,
                "sweep_name": sweep_name,
                "simulation_status": "nominal_unvalidated",
                "physical_4f_readiness": "blocked",
                "final_export_allowed": False,
                **dict(extra),
                **metrics,
            }
        )

    for cycles in (-12, -8, -4, 0, 4, 8, 12):
        cfg = replace_config(
            config_for_candidate(spec, base),
            command_domain_carrier_cycles_x=float(cycles),
            numerical_model_carrier_cycles_x=float(cycles),
        )
        _append("carrier_vs_nominal_stop_transmission", cfg, {"carrier_cycles_x": cycles})
    for delta_x in (-0.00012, -0.00006, 0.0, 0.00006, 0.00012):
        cfg = replace_config(config_for_candidate(spec, base), pinhole_offset_x_m=spec.pinhole_offset_x_m + delta_x)
        _append("stop_offset_x_transmission", cfg, {"stop_offset_delta_x_m": delta_x})
    for radius in (0.00012, 0.00018, 0.00026, 0.00034):
        cfg = replace_config(config_for_candidate(spec, base), pinhole_radius_m=radius)
        _append("pinhole_radius_transmission", cfg, {"pinhole_radius_m": radius})
    for radius in (0.00055, 0.0008, 0.00105):
        cfg = replace_config(config_for_candidate(spec, base), input_beam_radius_m=radius)
        _append("beam_radius_vs_relay_quality", cfg, {"input_beam_radius_m": radius})
    for dx, dy in ((-0.00008, 0.0), (0.0, 0.0), (0.00008, 0.0), (0.0, -0.00008), (0.0, 0.00008)):
        cfg = replace_config(config_for_candidate(spec, base), input_beam_decentre_x_m=dx, input_beam_decentre_y_m=dy)
        _append("input_decentre_vs_output_centroid", cfg, {"input_decentre_x_m": dx, "input_decentre_y_m": dy})
    return rows


def candidate_robustness_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_sweep: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_sweep.setdefault(str(row["sweep_name"]), []).append(row)
    summaries: dict[str, Any] = {}
    for sweep, sweep_rows in by_sweep.items():
        transmissions = [float(r["relative_transmitted_energy"]) for r in sweep_rows]
        scores = [float(r["relay_quality_score"]) for r in sweep_rows]
        summaries[sweep] = {
            "num_points": len(sweep_rows),
            "min_relative_transmitted_energy": min(transmissions),
            "max_relative_transmitted_energy": max(transmissions),
            "transmission_span": max(transmissions) - min(transmissions),
            "mean_relay_quality_score": float(np.mean(scores)),
        }
    score_values = [float(row["relay_quality_score"]) for row in rows]
    return {
        "candidate_robustness_summary": summaries,
        "overall_nominal_score": float(np.mean(score_values)) if score_values else 0.0,
        "simulation_status": "nominal_unvalidated",
        "physical_4f_readiness": "blocked",
        "final_export_allowed": FINAL_EXPORT_ALLOWED,
    }


def _write_png(path: Path, array: np.ndarray, *, cmap: str = "viridis") -> Path:
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(array)
    if arr.ndim == 2 and np.issubdtype(arr.dtype, np.integer):
        plt.imsave(path, arr, cmap=cmap, vmin=0, vmax=255)
    else:
        plt.imsave(path, arr, cmap=cmap)
    return path


def _write_phase_payload(prefix: str, payload: Mapping[str, Any], out_dir: Path) -> dict[str, str]:
    paths = {
        "phase_rad": out_dir / f"{prefix} phase_rad.npy",
        "quantised_rad": out_dir / f"{prefix} quantised_rad.npy",
        "gray": out_dir / f"{prefix} gray.png",
    }
    np.save(paths["phase_rad"], np.asarray(payload["phase_rad"], dtype=np.float32))
    np.save(paths["quantised_rad"], np.asarray(payload["quantised_rad"], dtype=np.float32))
    _write_png(paths["gray"], np.asarray(payload["gray"], dtype=np.uint8), cmap="gray")
    return {k: str(v).replace("\\", "/") for k, v in paths.items()}


def make_claim_boundary_markdown(spec: CandidateSpec, run: NominalF300Run) -> str:
    return f"""# Stage 9B.0 Candidate Claim Boundary

Candidate: `{spec.candidate_id}`

Status labels:
- nominally_simulated
- hardware_command_exportable
- not_bench_validated

This package is a nominal F300 4F forward-model diagnostic. It is not a
bench-calibrated physical 4F model, not a camera model, not an inverse
correction result, not an AI estimate, and not a material-response prediction.

SLM role contract:
- SLM1 carries the flat/vortex/structured phase conditioning.
- SLM2 carries the command-domain carrier and later may carry correction maps.
- SLM2 does not contain an axicon phase.

Readiness:
- physical_4f_readiness = `{run.physical_4f_readiness}`
- carrier_coordinate_status = `{run.carrier_coordinate_status}`
- final_export_allowed = `{run.final_export_allowed}`
"""


def candidate_manifest(spec: CandidateSpec, run: NominalF300Run, robustness_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = relay_output_candidate_metrics(run)
    return {
        "stage": STAGE,
        "study_name": STUDY_NAME,
        "candidate_id": spec.candidate_id,
        "candidate_family": spec.candidate_family,
        "candidate_status_labels": list(CANDIDATE_STATUS_LABELS),
        "simulation_status": "nominal_unvalidated",
        "hardware_realizability_status": spec.hardware_realizability_status,
        "export_permission": spec.export_permission,
        "model_label": MODEL_LABEL,
        "claim_boundary_labels": list(CLAIM_BOUNDARY_LABELS),
        "slm_role_contract": {
            "SLM1": "vortex/structured phase source; ell=%d" % int(spec.ell),
            "SLM2": "carrier/future correction only",
            "SLM2_contains_axicon_phase": False,
        },
        "phase_generator": spec.phase_generator,
        "required_components": list(spec.required_components),
        "known_model_limits": list(spec.known_model_limits),
        "physical_4f_readiness": run.physical_4f_readiness,
        "carrier_coordinate_status": run.carrier_coordinate_status,
        "camera_validation": "absent",
        "material_prediction": "absent",
        "relay_output_to_axicon_mode": run.config.relay_output_to_axicon_mode,
        "handoff_modes_supported_by_contract": list(HANDOFF_MODES),
        "metrics": metrics,
        "robustness": candidate_robustness_summary(robustness_rows),
        "warnings": list(run.warnings),
        "config": asdict(run.config),
        "final_export_allowed": bool(run.final_export_allowed),
    }


def export_candidate_package(
    spec: CandidateSpec,
    *,
    run_id: str = DEFAULT_RUN_ID,
    output_root: str | Path = PACKAGE_ROOT,
    base_config: NominalF300Config | None = None,
) -> dict[str, Path]:
    cfg_base = base_config or config_from_profile()
    run = simulate_candidate(spec, cfg_base)
    robustness_rows = run_candidate_robustness(spec, NominalF300Config.fast())
    out_dir = Path(output_root) / run_id / spec.candidate_id
    out_dir.mkdir(parents=True, exist_ok=True)

    slm1_payload = phase_export_payload(run.slm1_phase_rad, mask_id=f"{spec.candidate_id}_SLM1", slm_id="SLM1", config=run.config)
    slm2_payload = phase_export_payload(run.slm2_phase_rad, mask_id=f"{spec.candidate_id}_SLM2", slm_id="SLM2", config=run.config)

    paths: dict[str, Path] = {}
    paths["run_manifest"] = write_json(out_dir / "run_manifest.json", run_to_manifest(run))
    paths["candidate_manifest"] = write_json(out_dir / "candidate_manifest.json", candidate_manifest(spec, run, robustness_rows))
    paths["nominal_4f_profile_snapshot"] = write_json(out_dir / "nominal_4f_profile_snapshot.json", load_nominal_f300_profile())
    _write_phase_payload("SLM1", slm1_payload, out_dir)
    _write_phase_payload("SLM2", slm2_payload, out_dir)
    paths["fourier_plane_pre_stop"] = _write_png(out_dir / "fourier_plane_pre_stop.png", np.abs(run.fourier_plane_field_pre_stop) ** 2)
    paths["fourier_stop_transmission"] = _write_png(out_dir / "fourier_stop_transmission.png", run.fourier_stop_transmission, cmap="gray")
    paths["fourier_plane_post_stop"] = _write_png(out_dir / "fourier_plane_post_stop.png", np.abs(run.fourier_plane_field_post_stop) ** 2)
    paths["nominal_relay_output_xy"] = _write_png(out_dir / "nominal_relay_output_xy.png", np.abs(run.nominal_relay_output_field) ** 2)
    paths["energy_ledger"] = write_csv(out_dir / "energy_ledger.csv", run.component_energy_ledger)
    paths["robustness_summary"] = write_csv(out_dir / "robustness_summary.csv", robustness_rows)
    paths["claim_boundary"] = out_dir / "claim_boundary.md"
    paths["claim_boundary"].write_text(make_claim_boundary_markdown(spec, run), encoding="utf-8")
    return paths


def build_stop_robustness_rows(base_config: NominalF300Config | None = None) -> list[dict[str, Any]]:
    base = _base_fast_config(base_config)
    rows: list[dict[str, Any]] = []
    for cycles in (-12, -8, -4, 0, 4, 8, 12):
        for delta in (-0.00016, -0.00008, 0.0, 0.00008, 0.00016):
            spec = CandidateSpec(
                "stop_robustness_probe",
                "robustness_probe",
                ell=2,
                command_domain_carrier_cycles_x=float(cycles),
                numerical_model_carrier_cycles_x=float(cycles),
                pinhole_offset_x_m=0.00031 + delta,
            )
            run = simulate_candidate(spec, base)
            metrics = relay_output_candidate_metrics(run)
            rows.append(
                {
                    "carrier_cycles_x": cycles,
                    "stop_offset_delta_x_m": delta,
                    "pinhole_transmitted_fraction": metrics["pinhole_transmitted_fraction"],
                    "relative_transmitted_energy": metrics["relative_transmitted_energy"],
                    "relay_quality_score": metrics["relay_quality_score"],
                }
            )
    return rows


def plot_stop_robustness(
    output_path: str | Path = FIGURE_STOP_ROBUSTNESS,
    *,
    base_config: NominalF300Config | None = None,
) -> Path:
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    rows = build_stop_robustness_rows(base_config)
    cycles = sorted({int(row["carrier_cycles_x"]) for row in rows})
    deltas = sorted({float(row["stop_offset_delta_x_m"]) for row in rows})
    mat = np.zeros((len(deltas), len(cycles)), dtype=float)
    for row in rows:
        i = deltas.index(float(row["stop_offset_delta_x_m"]))
        j = cycles.index(int(row["carrier_cycles_x"]))
        mat[i, j] = float(row["relative_transmitted_energy"])
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), facecolor="white")
    im = axes[0].imshow(mat, origin="lower", aspect="auto", cmap="viridis")
    axes[0].set_xticks(range(len(cycles)), labels=[str(c) for c in cycles])
    axes[0].set_yticks(range(len(deltas)), labels=[f"{d*1e6:+.0f}" for d in deltas])
    axes[0].set_xlabel("SLM2 numerical carrier cycles x")
    axes[0].set_ylabel("stop x offset delta (um)")
    axes[0].set_title("Nominal stop robustness")
    fig.colorbar(im, ax=axes[0], label="relative output energy")
    for delta in deltas:
        ys = [
            float(row["relative_transmitted_energy"])
            for row in rows
            if abs(float(row["stop_offset_delta_x_m"]) - delta) < 1e-15
        ]
        axes[1].plot(cycles, ys, marker="o", label=f"{delta*1e6:+.0f} um")
    axes[1].set_xlabel("SLM2 numerical carrier cycles x")
    axes[1].set_ylabel("relative output energy")
    axes[1].set_title("Carrier versus stop setting")
    axes[1].legend(fontsize=8, ncols=2)
    fig.suptitle(
        "Stage 9B.0 nominal F300 4F stop sensitivity - not bench calibrated",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.01,
        "nominal_4f_forward_model; physical_4f_readiness=blocked; no camera/material model; final_export_allowed=False",
        fontsize=8.5,
        family="monospace",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_candidate_atlas(
    output_path: str | Path = FIGURE_CANDIDATE_ATLAS,
    *,
    specs: Sequence[CandidateSpec] | None = None,
    base_config: NominalF300Config | None = None,
) -> Path:
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    specs = list(specs or build_candidate_specs()[:12])
    base = base_config or NominalF300Config.fast()
    runs = [simulate_candidate(spec, base) for spec in specs]
    ncols = 4
    nrows = int(np.ceil(len(runs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15.0, max(7.2, 3.1 * nrows)), facecolor="white")
    axes_arr = np.asarray(axes).ravel()
    vmax = max(float(np.max(np.abs(run.nominal_relay_output_field) ** 2)) for run in runs)
    x_mm = np.asarray(runs[0].grid["x"], float) * 1e3
    extent = (float(x_mm.min()), float(x_mm.max()), float(x_mm.min()), float(x_mm.max()))
    for ax, spec, run in zip(axes_arr, specs, runs):
        metrics = relay_output_candidate_metrics(run)
        ax.imshow(np.abs(run.nominal_relay_output_field) ** 2, origin="lower", extent=extent, cmap="magma", vmin=0.0, vmax=vmax)
        ax.set_title(
            f"{spec.candidate_id}\nscore={metrics['relay_quality_score']:.2f}, T={metrics['relative_transmitted_energy']:.2f}",
            fontsize=8.5,
        )
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
    for ax in axes_arr[len(runs):]:
        ax.axis("off")
    fig.suptitle("Stage 9B.0 nominal F300 candidate atlas - unvalidated relay output", fontsize=14, fontweight="bold")
    fig.text(
        0.01,
        0.01,
        "SLM1: vortex/structured phase. SLM2: carrier/future correction only, no axicon. physical_4f_readiness=blocked; final_export_allowed=False.",
        fontsize=8.5,
        family="monospace",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out


__all__ = [
    "CANDIDATE_STATUS_LABELS",
    "DEFAULT_RUN_ID",
    "HANDOFF_MODES",
    "PACKAGE_ROOT",
    "STUDY_NAME",
    "CandidateSpec",
    "build_candidate_atlas_config",
    "build_candidate_specs",
    "build_stop_robustness_rows",
    "candidate_manifest",
    "candidate_robustness_summary",
    "config_for_candidate",
    "export_candidate_package",
    "make_claim_boundary_markdown",
    "plot_candidate_atlas",
    "plot_stop_robustness",
    "relay_output_candidate_metrics",
    "run_candidate_robustness",
    "simulate_candidate",
    "write_candidate_atlas_config",
]
