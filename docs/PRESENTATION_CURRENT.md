# Current presentation figure state

**Updated:** 2026-08-14

This is the canonical pointer for the current first-year presentation figure set. It exists so future coding / ChatGPT sessions do not accidentally fall back to historical presentation assets.

## Slide 2 — ideal beam profile shaping

Current preferred figure:

`outputs/figures/presentation_phase2j/02_beam_profile_shaping_B0_V1_V3_inferno_tight.png`

Renderer:

`tools/build_phase2j_ideal_beam_profile_figure.py`

This Phase 2J refinement supersedes the older presentation-only B0/V1/V3 hero rendering for the live presentation. It is regenerated from the current dual-SLM -> explicit 4F -> physical axicon optical route and preserves fixed-laboratory longitudinal coordinates.

Presentation rendering choices:

- `inferno` heatmap: black -> red -> orange -> yellow;
- native model grid `N=1536`;
- tighter transverse crop: +/-0.20 mm;
- tighter longitudinal crop: +/-0.22 mm;
- 64 z samples;
- high-resolution export;
- display interpolation is rendering-only and does not alter the underlying complex field;
- per-case peak normalisation is used only for ideal morphology comparison.

Provenance sidecar:

`outputs/figures/presentation_phase2j/02_beam_profile_shaping_B0_V1_V3_inferno_tight.txt`

## Other current presentation figures

Unless subsequently refined in Phase 2J, retain the audited Phase 2I presentation figures for:

- computational route: `outputs/figures/presentation_phase2i/01_computational_route.png`;
- realistic error fingerprints: `outputs/figures/presentation_phase2i/08_V1_real_error_fingerprints.png`;
- V1 axicon lateral decentre: `outputs/figures/presentation_phase2i/04_V1_axicon_decentre_fixed_lab.png`;
- V1 non-ideal tip: `outputs/figures/presentation_phase2i/05_V1_nonideal_tip_fixed_lab.png`;
- tip-avoidance planning proxy: `outputs/figures/presentation_phase2i/09_tip_avoidance_planning_proxy.png`;
- synthetic z-stack inverse recovery: `outputs/figures/presentation_phase2i/10_synthetic_zstack_inverse_recovery.png`;
- experiment/simulation closure loop: `outputs/figures/presentation_phase2i/06_simulation_experiment_closure.png`.

## Rule for future presentation edits

Use repository-generated figures only. Do not substitute locally approximated, AI-generated, or historical figures when a current validated repository renderer exists. Presentation-only visual changes must not alter the underlying physics or claim boundaries.
