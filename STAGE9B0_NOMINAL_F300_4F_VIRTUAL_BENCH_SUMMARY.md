# Stage 9B.0 Nominal F300 4F Virtual Bench Summary

Starting checkpoint: `b62ede4` Stage 9A.1B downstream carrier stop
characterisation.

Stage 9B.0 adds an opt-in nominal scalar F300 4F forward model and candidate
beam atlas. It does not alter the active CSLM route, does not mark physical 4F
readiness ready, and does not add camera, inverse correction, AI, GUI, axicon
handoff, material response, or Stage 8D work.

Stage 9B.0.1 supersedes one atlas shortcut: candidate fields now arrive at SLM2
through the existing CSLM component route, and SLM1 phase is no longer accepted
as a direct input to the nominal F300 model. SLM2 carrier handling is labelled
as an ideal continuous-ramp surrogate, not pixelated-SLM diffraction-order
physics.

## Added

- `configs/hardware/cslm_f300_nominal_4f_profile.json`
- `configs/studies/cslm_nominal_4f_candidate_atlas_v1.json`
- `vbb_study/digital_twin/nominal_f300_4f.py`
- `vbb_study/digital_twin/candidate_beam_atlas.py`
- `notebooks/digital_twin/01_nominal_f300_4f_virtual_bench_and_candidate_atlas.ipynb`
- `docs/46_nominal_f300_4f_virtual_bench.md`
- `docs/47_candidate_beam_atlas_contract.md`
- `tests/test_stage9b0_nominal_f300_4f_virtual_bench.py`

## Figures

- `outputs/figures/digital_twin/stage9b0_nominal_f300_4f_component_sequence.png`
- `outputs/figures/digital_twin/stage9b0_nominal_f300_4f_stop_robustness.png`
- `outputs/figures/digital_twin/stage9b0_nominal_f300_candidate_atlas.png`

## Boundary

```text
nominal_4f_forward_model
not_bench_calibrated
not_physical_4f_readiness_ready
not_camera_modelled
not_material_modelled
final_export_allowed = false
```

SLM1 remains the vortex/structured phase source. SLM2 carries carrier/future
correction only and does not generate an axicon phase.

## Candidate Packages

Candidate packages are generated under:

```text
outputs/nominal_4f_candidate_runs/<run_id>/<candidate_id>/
```

They are `command_masks_exportable_unvalidated` but not final, not bench
validated, and not publication-ready physical outputs.

## Unsupported

Still unsupported: measured 4F geometry, true stop position/radius,
carrier-to-Fourier-plane calibration, real SLM phase response at 1030 nm,
camera-plane modelling, inverse correction, AI, downstream axicon handoff
geometry, and material response.
