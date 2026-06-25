# Stage 8C.3R.5.1 CSLM Axicon Handoff Summary

Starting point: Stage 8C.3R.5 corrected SLM attribution (`273a065`).

Stage 8C.3R.5.1 adds an opt-in ideal selected-order benchmark that connects the
CSLM route to the downstream physical axicon without pretending that a physical
4F filter exists.

## Scope

```text
free-space n = 1.0
optical-field / fluence diagnostics only
no material model
diagnostic_only
final_export_allowed=False
fourier_filter_physics_available=False
```

No Stage 8D, material, calibration, dose, writing, thermal, nonlinear, GUI, or
3D work is introduced.

## Active CSLM Route

```text
source_field
-> input_conditioning_boundary
-> SLM1_phase_plane
-> SLM1_to_SLM2_segment
-> SLM2_phase_plane
-> SLM2_to_pre_4F_diagnostic_segment
-> post_SLM2_pre_4F_diagnostic_plane
```

This branch produces `post_slm2_unfiltered_diagnostic_field`, which still
contains the SLM2 carrier and is not a final lab output.

## Ideal Benchmark Route

```text
ideal_selected_order_handoff_plane
-> physical_axicon_benchmark_plane
-> post_axicon_benchmark_segment
-> ideal_axicon_benchmark_reference_plane
```

The surrogate field is built from the SLM2 input field after SLM1 vortex
propagation plus SLM2 correction phase only.  The carrier is removed as an
explicit non-physical desired-order surrogate:

```text
claim_boundary = carrier-free desired-order benchmark; not a physical 4F prediction
physical_filter_modelled = False
zero_order_rejection_modelled = False
order_efficiency_modelled = False
selected_order_energy_uJ = None
```

The physical axicon benchmark uses the existing scalar physical-axicon
transmission and models only represented aperture loss.

## Roles

- SLM1: vortex/topological-charge phase and phase-only conditioning.
- SLM2: correction phase, carrier/blaze, wrapping, quantisation; no axicon
  phase.
- Physical axicon: downstream Bessel-forming scalar phase and clear aperture in
  the opt-in benchmark branch.
- 4F optics/filter: warning-only declarations.

## Created

- `tests/test_stage8c3r5_1_cslm_axicon_handoff.py`
- `docs/39_cslm_to_physical_axicon_handoff.md`
- `STAGE8C3R5_1_CSLM_AXICON_HANDOFF_SUMMARY.md`

## Modified

- `vbb_study/digital_twin/cslm_route.py`
- `vbb_study/digital_twin/__init__.py`
- `docs/38_component_owned_cslm_route.md`
- `STAGE8C3R5_CSLM_BASELINE_ROUTE_SUMMARY.md`
- `notebooks/digital_twin/00_full_beam_to_write_cockpit_MVP.ipynb`

## Required Figures

```text
outputs/figures/digital_twin/stage8c3r5_1_cslm_to_axicon_handoff_audit.png
outputs/figures/digital_twin/stage8c3r5_1_unfiltered_vs_ideal_handoff.png
outputs/figures/digital_twin/stage8c3r5_1_ideal_vortex_bessel_axicon_benchmark.png
```

## Remaining Unsupported

Physical 4F filtering remains unavailable until measured SLM geometry,
component-owned lens propagation, Fourier-plane coordinate calibration, filter
mask placement, and order-resolved energy calibration are added and validated.
