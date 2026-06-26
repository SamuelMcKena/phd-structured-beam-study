# Stage 9A.1 — First Fourier Carrier Calibration Session Pack (summary)

**Goal:** generate the exact SLM masks, capture plan, manifests, and lab setup sheet for the first
physical CSLM → 4F session (SLM1 flat, SLM2 command-domain carrier only, axicon removed). Files +
bench procedure only; no new physics.

## Added
- `vbb_study/digital_twin/slm_calibration_masks.py`: `CarrierSweepConfig`, `command_carrier_phase`,
  `pixels_per_cycle`, `validate_carrier_sampling` (sampling guard), `build_carrier_mask`,
  `build_carrier_sweep_masks`, `export_mask` (phase npy / quantised npy / grayscale PNG / metadata
  JSON, reusing `wrap_phase_rad`/`quantize_phase_rad`/`phase_to_gray`),
  `build_carrier_calibration_study`, `create_fourier_carrier_calibration_session` (extends Stage 9A),
  `plot_command_domain_carrier_mask_atlas`.
- `configs/studies/cslm_fourier_carrier_calibration_minimal_v1.json`.
- Notebook section "Stage 9A.1 — First Fourier-Plane Carrier Calibration Session".
- `docs/43_first_fourier_plane_carrier_calibration_session.md`.
- `tests/test_stage9a1_fourier_carrier_session_pack.py` (10 tests).
- Figure `outputs/figures/digital_twin/stage9a1_command_domain_carrier_mask_atlas.png`.

## Command-domain rule
`phi = wrap(2π·N_x·px/W + 2π·N_y·py/H)`; N_x/N_y are signed cycles across the displayed command
area. Metadata `physical_frequency_status = uncalibrated_command_domain`;
`phase_response_calibration_status = unknown_or_unverified`. Never labelled cycles/mm or cycles/m.

## Default sweep
dark 5, flat-ref 3, carrier_cycles [-24,-16,-8,0,8,16,24], axes x+y, diagonals (8,8)/(8,-8)/(-8,8)/
(-8,-8), capture_repeats 1, minimum_pixels_per_carrier_cycle 8.

## Sampling guard
`display/|cycles| < min_pixels_per_carrier_cycle` → validation fails (no silent aliasing); e.g.
64 px display + 24 cycles = 2.7 px/cycle → rejected.

## Mask export
Per mask: `<id>_phase_rad.npy`, `<id>_quantised_rad.npy`, `<id>_gray.png`, `<id>_metadata.json`
(mask/SLM id, command display W/H, carrier cycles x/y, pixels/cycle, phase-wrap convention,
quantisation levels, calibration status, physical-frequency status, coordinate frame, sha256
checksum, timestamp, git commit). No vortex/axicon/correction-map/aperture term.

## Session package
`outputs/calibration_runs/<run_id>/` with run_manifest, acquisition_plan, capture_manifest_template,
hardware/bench/coordinate snapshots, phase_masks/{slm1,slm2}, figures/, and experiment_package/
(LAB_README_FIRST_FOURIER_SESSION.md, bench setup md+csv, camera checklist, carrier_sweep_log,
fourier_plane_observation_template, operator_notes). Empty raw/ dirs (no fabricated images);
generated packages gitignored.

## Tests
10 Stage 9A.1 tests pass; Stage 9A + R5/R5.1/R5.2/R5.3/physical-axicon regression green (venv2).

## Claim boundary
command-domain carrier cycles; no physical 4F model, no camera model, no material model;
`final_export_allowed=False`.

## Data to bring back for Stage 9B
Per capture: dark frames; zero-carrier reference order position; observed zero/+1/−1 order
positions (px) vs command carrier cycles for x and y (and diagonals); order movement direction
under sign reversal; saturation/clipping notes; Fourier-stop state; exposure/gain; camera location;
plus the recorded SLM command display resolution and any measured SLM/Fourier-plane geometry. That
dataset yields the carrier→order-position mapping (Fourier-plane coordinate convention + sign) that
unblocks docs/41 level C.
