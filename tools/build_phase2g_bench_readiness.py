from __future__ import annotations

import argparse
import json
from pathlib import Path

from vbb_study.calibration.schema import CalibrationBundle, canonical_calibration_template
from vbb_study.calibration.validation import calibration_readiness_for_claim, validate_calibration_bundle
from vbb_study.vector_arm_config import SLMPanelConfig


PHASE2G_GATES = (
    "slm_phase_fidelity",
    "fourier_filter_geometry",
    "wavefront_correction",
    "refractive_axicon_tilt",
    "objective_sample_vector_field",
    "absolute_dimensions",
    "absolute_fluence",
    "spatiotemporal_field",
    "experimental_agreement",
    "material_response",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase 2G bench-calibration readiness evidence.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/validation/phase2g/bench_readiness.json"),
    )
    parser.add_argument(
        "--template-output",
        type=Path,
        default=Path("calibration/templates/phase2g_bench_calibration_template.json"),
    )
    args = parser.parse_args()

    template = canonical_calibration_template()
    bundle = CalibrationBundle(template)
    validation = validate_calibration_bundle(bundle)
    panel = SLMPanelConfig()
    readiness = {}
    for gate in PHASE2G_GATES:
        row = calibration_readiness_for_claim(bundle, gate)
        readiness[gate] = {
            "ready": bool(row.ready),
            "status": row.status,
            "missing_measurements": list(row.missing_measurements),
            "non_calibrated_measurements": list(row.non_calibrated_measurements),
            "satisfied_measurements": list(row.satisfied_measurements),
        }

    report = {
        "outcome": "PHASE2G-CALIBRATION-READY-CODE-MEASUREMENTS-PENDING",
        "schema_version": template["schema_version"],
        "template_valid": bool(validation.valid_schema),
        "template_errors": list(validation.errors),
        "known_physical_bench_contract": {
            "slm_make": template["slm"]["make"],
            "slm_model": template["slm"]["model"],
            "slm_resolution_px": template["slm"]["active_resolution"],
            "slm_pixel_pitch_m": template["slm"]["pixel_pitch_m"],
            "slm_fill_factor": template["slm"]["fill_factor"],
            "slm_phase_bits": template["slm"]["phase_bits"],
            "carrier_period_px": int(template["slm"]["carrier_period_px"]),
            "carrier_lp_per_mm": float(template["slm"]["carrier_lp_per_mm"]),
            "carrier_period_from_config_px": float(panel.carrier_period_px),
            "wavelength_m": template["laser"]["wavelength_m"]["value"],
            "camera_model": template["camera"]["model"],
        },
        "readiness": readiness,
        "physics_components_available": [
            "audited scalar/vector free-space propagation",
            "physical SLM pixelation/quantisation/fill-factor/order filtering",
            "measured LUT inversion and greyscale export",
            "Shack-Hartmann Southwell least-squares OPD reconstruction",
            "topology-preserving additive SLM correction",
            "explicit 4F selected-order propagation",
            "explicit two-surface refractive axicon reference for rigid tilt",
            "vector Debye/Richards-Wolf objective focusing",
            "spectral vector Fresnel sample interface",
            "linear ultrafast spectral/time reconstruction",
            "fluence/peak-intensity/discrete scan exposure",
            "camera-calibrated simulation/experiment comparison",
            "calibration-bound empirical material-response interface",
        ],
        "absolute_claim_policy": (
            "Code readiness is not measurement readiness. A gate remains blocked until the corresponding "
            "laboratory values are present with calibrated provenance."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.template_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.template_output.write_text(json.dumps(template, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
