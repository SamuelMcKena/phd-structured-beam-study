# Stage 9A — Calibration Acquisition and Measured-Data Ingestion (summary)

**Goal:** the acquisition / storage / ingestion / pixel-analysis / experiment-package tooling for a
real CSLM → 4F → physical-axicon calibration campaign. No optical physics added.

## Added
- `vbb_study/digital_twin/calibration_acquisition.py`: `CalibrationCapture` schema (CaptureKind /
  DataStatus vocab), sha256/run-id/git helpers, `build_calibration_campaign_v1` (families 0–6),
  `create_calibration_acquisition_package`, `validate_capture_manifest`,
  `ingest_calibration_capture` / `ingest_calibration_run` (raw immutable + SHA256),
  `write_capture_manifest`, `save_derived_artifact`, `plot_calibration_campaign_overview`.
- `vbb_study/digital_twin/measured_image_metrics.py`: `load_image` (PNG/TIFF/NPY),
  `compute_measured_image_metrics` (pixel metrics; physical blocked until calibrated),
  `image_quality_report`, `compare_measured_to_model` (comparison boundary).
- `configs/studies/cslm_physical_axicon_calibration_campaign_v1.json` (families 0–6, 79 planned captures).
- `data/calibration_runs/` and `outputs/calibration_runs/` scaffolding (`.gitkeep` / README /
  `.gitignore`; generated raw data not committed).
- Notebook section "Stage 9A — Calibration Acquisition and Measured-Data Ingestion".
- `docs/42_calibration_acquisition_and_measured_data_ingestion.md`.
- `tests/test_stage9a_calibration_acquisition_and_ingestion.py` (10 tests; synthetic fixtures only).
- Figure `outputs/figures/digital_twin/stage9a_calibration_campaign_overview.png`.

## Campaign families
F0 dark/background, F1 input beam (pixels), **F2 SLM2 carrier→Fourier mapping (highest priority)**,
F3 Fourier-stop scan (measured only), F4 Gaussian-through-axicon z-stack, F5 vortex atlas ℓ=1,2,3,
F6 future controlled perturbations (`planned_future_calibration` / `not_implemented_in_current_stage`).

## Acquisition package
`outputs/calibration_runs/<run_id>/`: run_manifest.json, acquisition_plan.csv,
capture_manifest_template.csv, hardware_profile_snapshot.json, bench_inventory_snapshot.json,
coordinate_contract_snapshot.json, experiment_package/ (bench setup md+csv, camera checklist, energy
log, axicon alignment log, fused-silica observation template, operator notes). Raw under
`data/calibration_runs/<run_id>/{raw,manifests,derived,figures}`.

## Ingestion / formats / metrics
Formats: PNG, TIFF, NumPy `.npy`. Raw copied byte-for-byte + SHA256 verified; never transformed;
re-ingest blocked. Derived ops saved separately + recorded; raw never overwritten. Pixel metrics
valid pre-calibration; physical µm/mm metrics `blocked_coordinate_uncalibrated` until a declared
camera scale + named reference-plane relation exist. Non-annular captures flagged, not force-fit.

## Comparison boundary
Absolute physical comparison only with declared camera scale + named reference-plane relation +
coordinate-calibrated capture; otherwise `comparison_not_physically_calibrated`. Shape-only
descriptors labelled `not_absolute_physical_validation`. No fitting / inverse / AI.

## Tests
10 Stage 9A tests pass; R5/R5.1/R5.2/R5.3/physical-axicon regression green (venv2).

## Claim boundary
`n=1.0` free-space optical/fluence diagnostic; no physical 4F field; no camera-imaging model; no
material model; no inverse correction; no AI; `final_export_allowed=False`.

## Data required before physical 4F modelling
Family 2 carrier→Fourier mapping (Fourier-plane physical-position convention + carrier sign), SLM2
pixel pitch / transverse scale, lens-1/2 focal lengths + clear apertures, SLM2→lens1 /
lens1→Fourier / Fourier→lens2 / lens2→output distances, Fourier-stop centre/radius/shape — each with
real provenance (see docs/41 level C).

## Data required before inverse aberration correction
Calibrated camera scale + named reference-plane relation; measured Family 4/5 Gaussian + vortex
post-axicon z-stacks under a fixed camera convention; declared SLM2→lab / Fourier→lab / camera→lab
transforms (docs/41 level D). Inverse correction itself is out of scope until then.
