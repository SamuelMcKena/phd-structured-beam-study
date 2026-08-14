# Current presentation figure state

**Updated:** 2026-08-14

This is the canonical pointer for the current first-year presentation figure set. It exists so future coding / ChatGPT sessions do not accidentally fall back to historical presentation assets.

## Phase 2J visual standard

The full live-presentation figure set is now regenerated under one Phase 2J rendering policy while keeping the validated Phase 2I/2J optical physics and claim boundaries intact.

Visual policy:

- intensity heatmaps use `inferno`: black -> red -> orange -> yellow;
- main presentation simulations use native model grid `N=1536`;
- high-resolution export uses 420 dpi;
- display interpolation is `lanczos` only after the numerical field has been calculated;
- main longitudinal maps use 64 z samples;
- x-z evidence stays in fixed laboratory coordinates with no per-z recentring;
- tighter crops are presentation framing only and never alter the simulated field;
- comparative error figures preserve common nominal normalisation;
- cyclic phase maps remain cyclic rather than being incorrectly rendered as intensity.

Shared style:

`tools/presentation_phase2j_style.py`

Master renderer:

`tools/build_phase2j_presentation_suite.py`

## Canonical live-presentation figures

### 1. Computational model / numerical route

`outputs/figures/presentation_phase2j/01_computational_route_phase2j.png`

### 2. Ideal beam profile shaping

`outputs/figures/presentation_phase2j/02_beam_profile_shaping_B0_V1_V3_inferno_tight.png`

Renderer retained for the dedicated ideal figure:

`tools/build_phase2j_ideal_beam_profile_figure.py`

Presentation settings include N=1536, +/-0.20 mm transverse crop, +/-0.22 mm longitudinal crop and 64 z samples.

### 3. Moving towards the real experimental system

`outputs/figures/presentation_phase2j/08_V1_real_error_fingerprints_inferno_tight.png`

Nominal V1 is compared with representative input pointing, SLM1 registration and 4F iris-offset errors using the same forward model and fixed-laboratory propagation convention.

### 4. Axicon lateral decentre

`outputs/figures/presentation_phase2j/04_V1_axicon_decentre_fixed_lab_inferno_tight.png`

The physical perturbations remain -500 µm / aligned / +500 µm. The Phase 2J change is visual only: inferno, tighter crop, denser z sampling and higher export quality.

### 5. Non-ideal axicon tip

`outputs/figures/presentation_phase2j/05_V1_nonideal_tip_fixed_lab_inferno_tight.png`

The physical cases remain ideal sharp / 200 µm / 800 µm radial rounding and continue to pass the existing tip-resolution gate.

### 6. Experiment / simulation closure

`outputs/figures/presentation_phase2j/06_simulation_experiment_closure_phase2j.png`

### Supporting figures

Tip-avoidance planning proxy:

`outputs/figures/presentation_phase2j/09_tip_avoidance_planning_proxy_inferno.png`

Synthetic z-stack inverse recovery:

`outputs/figures/presentation_phase2j/10_synthetic_zstack_inverse_recovery_inferno.png`

Master provenance / metrics manifest:

`outputs/figures/presentation_phase2j/presentation_phase2j_manifest.json`

## Scientific boundaries retained

1. Presentation styling does not alter optical fields, propagation physics or calibration assumptions.
2. Illustrative perturbation magnitudes are not automatically measured bench tolerances.
3. The tip-avoidance figure remains a planning proxy unless a calibrated phase-only implementation is demonstrated.
4. The z-stack inverse figure remains synthetic model-to-model validation until real calibrated camera data are fitted.
5. Correction remains an additive residual-wavefront correction problem rather than conjugation of the complete structured field.

## Rule for future presentation edits

Use repository-generated canonical Phase 2J figures only. Do not substitute locally approximated, AI-generated, or historical figures when a current repository renderer exists. Any later presentation-only visual change should be implemented in the shared Phase 2J style/master renderer so the whole figure set stays coherent.
