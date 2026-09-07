# Legacy error-simulation cleanup — authoritative status

This audit exists because several generations of structured-beam error models coexist in the repository. A figure is **not** physical evidence merely because its label names a physical error. The operator must act on the complex field at the physical plane where the error occurs, and the downstream field must then be propagated.

## Corrections made in this cleanup

1. **Miao benchmark wording corrected.** The published synthetic wavefront contains **astigmatism, trefoil, quadrafoil and spherical aberration**. Spherical must remain in an implementation labelled as the published Miao synthetic benchmark. Miao also separates the wavefront into a radial component and an azimuthally varying component. Their statement that the tested reflective-axicon errors could be described by primary `m>1` terms is an experiment-specific simplification, not a universal restriction of the retrieval method.
2. **Hollow-core validation added.** Global image NRMSE is insufficient for a high-order vortex/Bessel beam. Validation must report centre intensity and leakage within a conservative fraction of the first bright annulus. The first bright-annulus radius is determined from the first positive zero of `J_q'`.
3. **Generic Zernike basis completed.** The declared-plane wavefront module now includes trefoil and quadrafoil x/y terms in addition to defocus, astigmatism, coma and spherical.
4. **Legacy evidence is gated centrally.** `legacy_error_policy.py` records whether each historical module is canonical, calibration-limited, compatibility-only, diagnostic-only, deprecated/redirected or reference-only. Post-engine diagnostic routes are explicitly blocked from correction evidence.
5. **SLM fill-factor claim tightened.** `component_plane_pipeline.py` uses a uniform `sqrt(FF)` amplitude factor. That is a throughput-only approximation and cannot predict spatial diffraction from pixel dead space. Morphology-sensitive SLM work must use `vbb_study.slm_model`, whose resolved pixel-aperture path enforces adequate sampling and whose coherent dead-space path keeps the unmodulated field.

## Module-by-module disposition

| Module | Disposition | Use going forward |
|---|---|---|
| `vortex_system_route.py` | canonical physical | Primary scalar bench error route |
| `vortex_system_error_sweeps.py` | canonical physical | Sensitivity sweeps; values are not measurements |
| `vortex_beam_slm_errors.py` | calibration-limited | Registration/LUT/stroke/fringing; measured panel data needed for absolute claims |
| `slm_model.py` | canonical physical | Pixelation/fill-factor/dead-space/quantisation model |
| `vortex_explicit_4f.py` | canonical physical | Fourier plane and iris errors |
| `vortex_rotated_plane.py` | calibration-limited | Scalar tilted-plane propagation; not full vector Snell/Fresnel |
| `vortex_rotated_plane_baseband.py` | reference only | Numerical cross-check only |
| `vortex_error_reference_models.py` | reference only | Analytic/literature validation targets |
| `vortex_physical_errors.py` | deprecated redirect | Superseded by `vortex_system_route.py` |
| `vortex_error_research_figures.py` | deprecated redirect | Must be regenerated from canonical route before report use |
| `component_plane_pipeline.py` | compatibility restricted | Useful for pre-propagation checks; not correction evidence; fill-factor is throughput-only |
| `lab_perturbations.py` | diagnostic only | Cockpit/display diagnostic only |
| `lab_realism_controls.py` | compatibility restricted | UI/control definitions, not evidence |
| `phase2b_visual_cases.py` | diagnostic only | Visual comparison only |
| `phase2b_visual_diagnostics.py` | diagnostic only | Display diagnostics only |
| `vortex_wavefront_errors.py` | canonical physical | Declared-plane generic OPD sensitivity |
| `vortex_axicon_tip_reference.py` | reference only | Tip benchmark only |
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

- Miao et al., *Optics Express* 30, 11360–11371 (2022), DOI 10.1364/OE.454796: intensity-only Bessel phase retrieval; radial/angular wavefront separation; published synthetic phase; high-order Bessel correction.
- Brzobohaty, Cizmar & Zemanek, *Optics Express* 16, 12688–12700 (2008), DOI 10.1364/OE.16.012688: rounded axicon tip produces a second refracted component which interferes with the quasi-Bessel beam.
- SLM pixel/dead-space literature: finite rectangular pixel aperture produces a sinc-squared diffraction-order envelope and is distinct from phase quantisation. Therefore a scalar throughput factor cannot stand in for spatial fill-factor diffraction.
- Oblique Bessel/axicon literature: oblique incidence introduces astigmatic degradation; a mere post-propagation shift or generic phase ramp is not a complete physical substitute for tilted-axicon/interface geometry.

## Remaining laboratory limitation

None of these corrections manufacture missing calibration. Absolute claims about the actual bench still require the measured SLM phase LUT/static maps, actual SLM incidence/polarisation geometry, measured 4F/iris geometry, axicon apex/surface data where relevant, and independent measured pre/post-correction z-stacks.
