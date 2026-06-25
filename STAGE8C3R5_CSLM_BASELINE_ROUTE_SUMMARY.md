# Stage 8C.3R.5 CSLM Baseline Route Summary

Stage 8C.3R.5 adds a component-owned concatenated-SLM baseline route for the
programmable vortex-Bessel path.

Starting point: Stage 8C.3R.4.1 physical-axicon route semantics cleanup
(`6e6c6b1`).  The physical-axicon route remains frozen and separate.

## Scope

This is a free-space optical-field and fluence diagnostic route only:

```text
n = 1.0
diagnostic_only
final_export_allowed=False
no material model
```

It does not start Stage 8D and does not add objective, material interface,
nonlinear, plasma, thermal, dose, writing, calibration, GUI, or 3D work.

## Created

- `vbb_study/digital_twin/cslm_route.py`
- `docs/38_component_owned_cslm_route.md`
- `tests/test_stage8c3r5_cslm_component_route.py`

## Modified

- `vbb_study/digital_twin/__init__.py`
- `notebooks/digital_twin/00_full_beam_to_write_cockpit_MVP.ipynb`

## Conceptual Route

```text
source_field
-> input_conditioning_boundary
-> SLM1_phase_plane
-> SLM1_to_SLM2_segment
-> SLM2_phase_plane
-> SLM2_to_fourier_lens_segment
-> Fourier_lens_1
-> Fourier_plane
-> plus_one_order_filter
-> Fourier_lens_2
-> 4F_output_plane
-> free_space_reference_plane
```

## Executed Route

```text
source_field
-> input_conditioning_boundary
-> SLM1_phase_plane
-> SLM1_to_SLM2_segment
-> SLM2_phase_plane
-> SLM2_to_reference_segment
-> free_space_reference_plane
```

## SLM Roles

SLM1 is implemented as `phase_only_conditioning`; in the baseline it owns the
vortex/topological-charge phase, applies a phase map, and then propagates to
SLM2.  It does not claim validated independent amplitude shaping.

SLM2 is implemented as `phase_correction_and_carrier_preserve_vortex`; it
composes carrier/blaze phase and optional correction phase only, with wrapping
and phase quantisation before downstream propagation.

Correction: SLM2 does not produce the axicon phase.  Any axicon/Bessel-like
preview is an external/non-executed reference and is not part of the executed
CSLM route.

## 4F Decision

`fourier_filter_physics_available=False`.

The 4F lenses, Fourier plane, +1 order filter, and 4F output plane are retained
as route declarations and warning-only configuration contracts.  No fake
filtered field is generated.

Missing before activation:

- measured SLM pixel pitch and active area;
- component-owned thin-lens transforms;
- Fourier-plane coordinate mapping;
- measured +1 filter centre, radius, and shape;
- measured lens and filter separations;
- order-resolved zero/+1/residual energy calibration.

## Diagnostic Previews

```text
outputs/figures/digital_twin/stage8c3r5_cslm_route_inspection.png
outputs/figures/digital_twin/stage8c3r5_slm_phase_and_field_baselines.png
outputs/figures/digital_twin/stage8c3r5_fourier_order_selection_audit.png
outputs/figures/digital_twin/stage8c3r5_phase_masks.npz
```

## Baseline Validations

The baseline route reports:

- zero-phase propagation power ratio;
- SLM1 vortex-only reference-plane field;
- external/non-executed axicon reference-plane field;
- external axicon plus SLM1 vortex reference, explicitly not executed by SLM2;
- measurable change under SLM1 topological-charge increment;
- phase wrapping and quantisation before propagation;
- energy conservation across phase-only SLM stages;
- inactive 4F order selection returning no filtered field.

## Governance

Core scalar propagation and holography primitives are reused, not replaced.
The physical-axicon route is not semantically changed.  No material response or
experimental calibration claim is introduced.
