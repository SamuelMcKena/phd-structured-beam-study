# Stage 8C.3R.5.3 — Bench Inventory, Coordinates, and 4F Readiness Gate (summary)

**Goal:** turn the R5.2 controls/profiles into a disciplined bench record + coordinate-convention
contract + a four-level physical-4F readiness gate. No optical transform implemented.

## Added
- `vbb_study/digital_twin/coordinate_contract.py`: `CoordinateFrame`, `CoordinateTransformDeclaration`,
  10 frames, 9 transforms, validation + readiness helpers.
- `vbb_study/digital_twin/bench_inventory.py`: `BenchInventoryItem`, `build_bench_inventory`,
  evidence overlay, `build_bench_inventory_profile`, save/load, `evaluate_physical_4f_readiness`
  (levels A–D), `readiness_summary_rows`, `measurement_checklist`, `plot_physical_4f_readiness_gate`.
- `configs/hardware/cslm_physical_axicon_bench_inventory.json` (diagnostic demo inventory;
  placeholder values with evidence, unknowns null).
- Notebook section "Stage 8C.3R.5.3 — Bench Inventory, Coordinates, and 4F Readiness Gate".
- `docs/41_measured_bench_inventory_and_4f_readiness.md`.
- `tests/test_stage8c3r5_3_bench_inventory_and_4f_readiness.py` (9 tests).
- Figure: `outputs/figures/digital_twin/stage8c3r5_3_physical_4f_readiness_gate.png` (diagnostic only).

## Coordinate frames / transforms
10 frames; the model frames (`lab_beam_frame`, phase-map frames, FFT frequency frame) are
`declared_model_convention`; the physical SLM-pixel, Fourier physical-position, and camera frames
are `unknown`. 9 transforms; only the model-internal identity transforms are `modelled`; the
SLM2→lab, Fourier-frequency→physical-position, Fourier→lab, and camera transforms are
`calibration_required` and not modelled.

## Readiness results (demo)
- A active CSLM diagnostic: **READY**.
- B ideal axicon benchmark: **READY** (executable; placeholders).
- C initial scalar 4F model: **BLOCKED** (5 blockers: Fourier coordinate convention unknown,
  SLM2 transverse scale unknown, lens-1/lens-2 clear apertures unknown).
- D measured bench / camera: **BLOCKED** (26 blockers: nothing measured + undeclared transforms).
Bench inventory by evidence bucket: measured 0 / placeholder 30 / unknown 27.

## Hard blockers for physical 4F
wavelength, SLM2 pixel pitch (transverse scale), SLM2 carrier in physical coords, SLM2→lens1,
lens-1 focal length + clear aperture, lens1→Fourier-plane, Fourier-stop centre/radius/shape,
Fourier-plane→lens2, lens-2 focal length + clear aperture, lens2→output — plus the Fourier-plane
physical-position coordinate convention.

## From repository records vs unknown
Recorded as `diagnostic_placeholder` (demo config defaults, not measured): wavelength, SLM1/SLM2
distances, 4F lens focal lengths and distances, Fourier-stop centre/radius/shape, axicon cone/
aperture/centre/reference distance. Remain `unknown` (null): SLM pixel pitch/active-area/fill/
resolution/calibration, lens clear apertures, camera everything, beam ellipticity/aperture,
spatial correction map, axicon axial offset / mechanical tip-tilt.

## Tests
9 R5.3 tests pass; R5/R5.1/R5.2/R5.3/physical-axicon regression green (venv2).

## Claim boundary
n=1.0 free-space optical/fluence diagnostic; no physical 4F field generated; changing 4F
inventory values updates readiness only; `fourier_filter_physics_available=False`;
`final_export_allowed=False`.

## Criteria before Stage 8C.3R.5.4 (physical 4F) can begin
Every physical-4F hard blocker non-null with at least estimated/manufacturer/measured provenance,
AND the Fourier-plane physical-position coordinate convention + SLM2 transverse coordinate scale
explicitly declared (carrier-order x/y placement defined), AND a component-owned scalar thin-lens
transform plan. Measured-bench/camera comparison additionally needs measured provenance and the
declared SLM2→lab / Fourier→lab / camera→lab transforms + reference plane + a beam-profile capture.
