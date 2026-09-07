# Legacy error-simulation cleanup — authoritative status

This audit exists because several generations of structured-beam error models coexist in the repository. A figure is **not** physical evidence merely because its label names a physical error. The operator must act on the complex field at the physical plane where the error occurs, and the downstream field must then be propagated.

## Corrections made in this cleanup

1. **Miao benchmark wording and objective corrected.** The published synthetic wavefront contains **astigmatism, trefoil, quadrafoil and spherical aberration**. Spherical remains in anything labelled as the published Miao synthetic benchmark. Miao separates radial and azimuthally varying phase. The `m>1` statement for the tested reflective-axicon edge errors is an experiment-specific simplification, not a universal restriction of the retrieval method.
2. **The faint q=20 hollow-core residual was traced to an implementation bias.** The port had added an always-on L2 penalty (`reg=2e-4`) to non-zero modal coefficients. That penalty is not part of the published intensity-mismatch objective and attenuated genuine recovered aberration content, leaving weak core structure after correction. The default is now unregularised (`reg=0`). Regularisation remains available only as an explicit opt-in numerical stabiliser and may not be used as correction evidence without a sensitivity study.
3. **Hollow-core validation is now a hard regression gate.** Global image NRMSE is insufficient for a high-order vortex/Bessel beam. Validation reports centre intensity, inner-core intensity and the candidate-minus-ideal residual inside 65% of the first bright-annulus radius. The first bright-annulus radius is determined from the first positive zero of `J_q'`. `scripts/validate_miao_hollow_core.py` runs the full-strength synthetic q=20 angular benchmark and CI rejects a correction that reintroduces the faint-core-ring failure.
4. **Generic Zernike basis completed.** The declared-plane wavefront module includes trefoil and quadrafoil x/y terms in addition to defocus, astigmatism, coma and spherical.
5. **Legacy evidence is gated centrally.** `legacy_error_policy.py` records whether each historical module is canonical, calibration-limited, compatibility-only, diagnostic-only, deprecated/redirected or reference-only. Post-engine diagnostic routes are explicitly blocked from correction evidence.
6. **The old `vortex_physical_errors.py` bypass has been removed at source.** Its routed public API is now a compatibility wrapper over `vortex_system_route.py`. A standalone non-zero rigid axicon tilt is explicitly refused because a tilted optic cannot be represented by one lab-plane multiplicative phase screen; the canonical route rotates the angular spectrum to the tilted optic plane, applies the local transmission, then rotates back.
7. **Visual atlas and research figures were rerouted.** `vortex_visual_atlas.py` and `vortex_error_research_figures.py` now obtain physical perturbations from the integrated canonical route rather than the older prototype. Generic wavefront sweeps share `vortex_wavefront_errors.py` rather than maintaining a divergent Zernike implementation.
8. **Phase-2A error registry source and generated CSV were repaired.** Rigid axicon tilt is no longer described as a linear phase ramp. Sample-interface tilt is assigned to the dedicated downstream vector/interface branch rather than the post-engine diagnostic layer.
9. **SLM fill-factor claim tightened.** The Phase-2A 10 mm validation grid deliberately uses a uniform `sqrt(FF)` throughput model because it does not resolve the 8 µm pixel borders. That is valid only for power bookkeeping. Morphology-sensitive pixel/dead-space work must use `vbb_study.slm_model`, whose resolved pixel-aperture path enforces adequate sampling and whose coherent dead-space path retains the unmodulated field.
10. **Numerical sampling guards are enforced rather than bypassed.** CI smoke grids were raised where a 256² grid under-sampled morphology crops. Under-resolved tip/pixel/crop studies fail instead of producing a misleading image.

## Module-by-module disposition

| Module | Disposition | Use going forward |
|---|---|---|
| `vortex_system_route.py` | canonical physical | Primary scalar bench error route |
| `vortex_system_error_sweeps.py` | compatibility restricted | Generic sensitivity definitions; current runners gate axicon families to axicon-physics-v3; direct registry output is not correction evidence |
| `vortex_beam_slm_errors.py` | calibration-limited | Registration/LUT/stroke/fringing; measured panel data needed for absolute claims |
| `slm_model.py` | canonical physical | Pixelation/fill-factor/dead-space/quantisation model |
| `vortex_explicit_4f.py` | canonical physical | Propagated Fourier plane and physical iris errors |
| `vortex_rotated_plane.py` | calibration-limited | Scalar tilted-plane propagation; not full vector Snell/Fresnel |
| `vortex_rotated_plane_baseband.py` | reference only | Numerical cross-check only |
| `vortex_error_reference_models.py` | reference only | Analytic/literature validation targets |
| `vortex_physical_errors.py` | compatibility restricted | Historical API now delegates to `vortex_system_route.py`; use canonical route directly for evidence |
| `vortex_error_research_figures.py` | calibration-limited | Rerouted to canonical system route; still a sensitivity figure generator, not measured validation |
| `vortex_visual_atlas.py` | calibration-limited | Canonical-route atlas; not measured validation |
| `component_plane_pipeline.py` | compatibility restricted | Useful for selected pre-propagation checks; not correction evidence; fill-factor is throughput-only |
| `lab_perturbations.py` | diagnostic only | Cockpit/display diagnostic only |
| `lab_realism_controls.py` | compatibility restricted | UI/control definitions, not evidence |
| `phase2b_visual_cases.py` | diagnostic only | Visual comparison only |
| `phase2b_visual_diagnostics.py` | diagnostic only | Display diagnostics only |
| `vortex_wavefront_errors.py` | canonical physical | Declared-plane generic OPD sensitivity |
| `vortex_axicon_tip_reference.py` | reference only | Tip benchmark; propagated studies use axicon-physics-v3 |
| `vortex_round_tip_reference.py` | reference only | Legacy tip benchmark only |
| `vortex_axicon_oblique_reference.py` | reference only | Oblique-incidence benchmark only |
| `vortex_axicon_oblique_wave.py` | deprecated redirect | Integrated route preferred |
| `vortex_refractive_axicon.py` | deprecated redirect | Regression/reference comparison only |
| `vortex_refractive_axicon_wave.py` | deprecated redirect | Regression/reference comparison only |
| `vector_refractive_axicon.py` | calibration-limited | Vector sensitivity branch |
| `vector_refractive_axicon_eikonal.py` | reference only | Eikonal benchmark only |
| `vector_tilt_study.py` | calibration-limited | Vector tilt sensitivity, not absolute bench proof |

The machine-readable source of this table is `vbb_study/digital_twin/legacy_error_policy.py`.

## Physics basis checked

- Miao et al., *Optics Express* 30, 11360–11371 (2022), DOI 10.1364/OE.454796: intensity-only Bessel phase retrieval; radial/angular wavefront separation; published synthetic phase; complex modal fitting; high-order Bessel correction.
- Brzobohaty, Cizmar & Zemanek, *Optics Express* 16, 12688–12700 (2008), DOI 10.1364/OE.16.012688: rounded axicon tip produces a second refracted component which interferes with the quasi-Bessel beam.
- SLM pixel/dead-space literature: finite rectangular pixel aperture produces a diffraction-order envelope and is distinct from phase quantisation. Therefore a scalar throughput factor cannot stand in for spatial fill-factor diffraction.
- Oblique Bessel/axicon literature: oblique incidence introduces astigmatic degradation; a post-propagation shift or generic phase ramp is not a complete physical substitute for tilted-axicon/interface geometry.

## What this does and does not prove

The synthetic q=20 regression proves that the matched numerical Miao model no longer creates the previous regularisation-induced hollow-core residual. It does **not** prove that an experimental correction will be ring-free. Absolute laboratory claims still require measured SLM phase LUT/static maps, actual SLM incidence/polarisation geometry, measured 4F/iris geometry, axicon apex/surface data where relevant, independent direct/conjugate branch resolution and a new measured pre/post-correction z-stack.
