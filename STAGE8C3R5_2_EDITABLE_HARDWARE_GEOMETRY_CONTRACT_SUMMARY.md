# Stage 8C.3R.5.2 — Editable Hardware / Geometry Contract (summary)

**Goal:** one source of truth → editable controls → validated configuration → route execution →
figures → saved run metadata. No new optical physics.

## What was added
- `vbb_study/digital_twin/control_contract.py`: `EditableControl` + status/provenance vocabularies,
  `build_cslm_editable_control_registry`, `editable_control_rows`, `apply_editable_control_overrides`,
  `build_default_demo_profile`, `build_measured_bench_template`, `validate_hardware_profile`,
  `hardware_profile_completeness_report`, `save_/load_hardware_profile`, `config_from_profile`,
  `status_counts`, `provenance_counts`, `plot_hardware_profile_completeness`.
- `configs/hardware/cslm_physical_axicon_demo_profile.json` (diagnostic demo, not measured).
- `configs/hardware/cslm_physical_axicon_measured_bench_template.json` (blank; all null/unknown).
- Notebook section "Stage 8C.3R.5.2 — Editable Hardware and Geometry Contract".
- `docs/40_editable_hardware_geometry_contract.md`.
- `tests/test_stage8c3r5_2_editable_hardware_contract.py` (12 tests).

## Control counts (66)
physics_active 15 · benchmark_only 6 · warning_only 29 · future_not_implemented 8 ·
numerical_advanced 5 · derived_read_only 3.

## Status / provenance vocabularies
Status: physics_active / benchmark_only / warning_only / future_not_implemented /
numerical_advanced / derived_read_only.
Provenance: measured / manufacturer_specification / estimated / diagnostic_placeholder /
unknown / derived. Demo profile has zero `measured` controls.

## Affects current physics vs recorded for future
- Active field: SLM1 charge/mode/ramp, SLM1→SLM2 distance, SLM2 conjugate/carrier/piston/
  quantisation, SLM2→pre-4F distance, order_handoff_mode, wavelength, source energy/radius, n.
- Benchmark only: 6 physical-axicon controls (when surrogate mode active).
- Numerical: grid_N, dx_um, n_z, z_max_um, bandlimit.
- Recorded for future (no field change): 4F lens distances/apertures, Fourier stop, camera,
  SLM pixel pitch/fill/active-area, beam ellipticity/rotation/aperture, spatial correction map.

## Readiness
Active CSLM diagnostic: complete (placeholders). Ideal axicon benchmark: complete.
Physical 4F route: blocked (no component-owned 4F model). Measured lab route: blocked (nothing
measured). Camera comparison: blocked.

## Claim boundary
n=1.0 free-space optical/fluence diagnostic; `fourier_filter_physics_available=False`;
no material/4F/camera physics; `diagnostic_only`; `final_export_allowed=False`.

## Remaining before physical 4F modelling
Measured SLM2 pixel pitch / continuous coordinate scale, SLM2→lens1 distance, lens-1 focal length
and clear aperture, lens1→Fourier-plane distance, Fourier-plane coordinate convention + carrier
frequency in physical SLM coordinates, Fourier-stop centre/radius/shape, Fourier-plane→lens2,
lens-2 focal length/aperture, lens2→output distance — each with real provenance, plus a
component-owned scalar thin-lens transform.
