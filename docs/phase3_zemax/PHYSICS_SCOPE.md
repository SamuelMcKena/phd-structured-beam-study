# Physics scope

## What Zemax POP can cross-check

- prescription-dependent propagation and diffraction;
- finite apertures/stops and real optical surfaces;
- geometrical separations, decentres and tilts represented by the prescription;
- prescription aberrations;
- selected physical-optics intensity/morphology and, where a supported complex-field route is available, transverse complex-field agreement.

## What this does not establish

- a full electromagnetic Maxwell solution;
- longitudinal-field (`Ez`) agreement through traditional transverse ZBF exchange;
- nonlinear material response, plasma generation, pulse accumulation, thermal response or permanent modification;
- quantum optical state evolution;
- experimental agreement or calibration.

The canonical solver policy remains unchanged: scalar FFT/ASM is the screening route; vector Debye is the quantitative focal/vector reference where required; vector spectral Fresnel is the component/high-angle interface reference. Metadata therefore records `validation_backend = zemax_pop`, never `canonical_solver = zemax`.

For V1/V3, annular irradiance is not evidence of topological winding. If only irradiance is retrieved, Phase 3 reports **intensity morphology comparison only**.
