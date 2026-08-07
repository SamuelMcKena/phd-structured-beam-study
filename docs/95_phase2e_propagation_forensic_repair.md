# Phase 2E Propagation Forensic Repair

**Status:** historical forensic checkpoint, superseded by
`docs/95_phase2e_final_source_scale_bessel_propagation.md`. Its original outcome was
`PHASE2E-PROP-B`: exact-route canonical baselines had been regenerated, but the
scalar propagation was not authorised as publication-converged. That checkpoint did not change the
accepted 3D surfaces, parameter sweeps, realism, energy, or hero figures.

## Previous Route

The disputed Phase 2E figure propagated a post-SLM, Fourier-filtered, once-pupil-masked and
once-axicon-phased field from the numerical axicon-output plane. The field was not already focused,
and no objective transform was applied. The separate Phase 2E renderer then evaluated x-z/y-z
spectral lines directly instead of reusing the accepted Phase 2B full-plane propagation product.

At the first common axicon-output plane, the N=1024 Phase 2E source reproduces the accepted N=512
source above the predeclared 0.999 complex-overlap gate after interpolation and one global phase
alignment. The minimum B0/V1/V3 overlap is about 0.99956. This validates the source-plane family,
but not the separate propagation renderer: its B0 x-z/y-z correlations to the accepted full-plane
route are only about 0.98356.

## Corrected Observables

The replacement B0 trace is the unsmoothed on-axis intensity. Its integrated observable is power in
a fixed circular bucket whose 48.83 um radius is the first native radial minimum at z=60 mm. V1 and
V3 use fixed-reference ring intensity and fixed annuli defined once at z=60 mm. Plane maximum is a
thin grey diagnostic only because it can follow different radial features.

The accepted Phase 2A interval remains 20-60 mm. The 112.5 mm hard-pupil and 125 mm Gaussian-radius
geometric estimates appear only in the explicitly clipped low-intensity diagnostic; they do not
replace the accepted interval.

## Pupil Audit

No-pupil, one-hard-pupil, one-soft-pupil and canonical-realistic controls all begin with the same
pre-pupil complex field. The canonical realistic B0 route is exactly the one-hard-pupil case because
its configured aberration is zero. The hard pupil retains about 0.80223 of the no-pupil power; the
soft 15% edge taper retains about 0.73516 and greatly reduces edge energy. These differences are
classified as `nominal_model_consequence`, not experimentally physical pupil behavior.

## Numerical Outcome

Power drift over the primary 0-100 mm products is below 8e-16. The ideal B0 x-z/y-z correlation is
about 0.999999999, so there is no transpose or centering failure. Enlarging the physical window from
10 mm to 15 mm and 20 mm and refining z from 1.0 mm to 0.5 mm and 0.25 mm pass the predeclared gates.
Maximum tested edge energy is below 9.4e-5.

Grid convergence fails. Relative to accepted N=512, N=1024 changes the z=60 mm fixed-core power by
about 14.1% and the normalised on-axis trace by about 6.4%; N=1536 changes them by about 11.6% and
8.1%. The 8 um SLM pitch is also unresolved: a 10 mm window gives only 0.41, 0.82 and 1.23
computational samples per SLM pixel at N=512, 1024 and 1536 respectively, below the SLM model's
two-samples-per-pixel resolved threshold. The canonical figures are therefore forensic baselines,
not publication-authorised propagation results.

## Structure Classification

- Global-linear structure: accepted canonical baseline structure, currently blocked by grid
  convergence.
- One-percent clipped diagonal/vertical structure: display-amplified weak structure.
- Hard-pupil differences: nominal hard-pupil diffraction consequences of the assumed model.
- Experimentally physical hard-pupil claim: not authorised until the effective pupil and fill are
  measured.

## Outputs

Primary baseline paths are `01b_propagation_maps/b0_canonical_propagation_primary.png`,
`v1_canonical_propagation_primary.png`, and `v3_canonical_propagation_primary.png` under the Phase 2E
figure root. The B0 low-intensity, pupil-model, and canonical-versus-direct figures are diagnostics.
All forensic tables and the machine-readable outcome are under
`outputs/validation/phase2e_propagation_repair/`.
