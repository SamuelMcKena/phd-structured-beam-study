# Model Limitations — Current Solver State

This document records the current limits of the simulation framework. It replaces
older statements that described only the historical scalar route and therefore
understated later Phase 2C, Phase 2H, Phase 3A and Phase 3B capabilities.

The core rule is unchanged: a feature being implemented in code does **not** make
it experimentally validated, and a sensitivity value does not become a measured
bench parameter unless laboratory or manufacturer evidence supplies it.

---

## 1. Solver hierarchy

The repository now contains several solver levels with different authority:

- scalar FFT / angular-spectrum propagation for screening and source-scale
  morphology;
- vector angular-spectrum propagation where vector free-space propagation is
  required;
- the Phase 2H common-eikonal vector two-surface refractive axicon model for
  the implemented flat-first macroscopic misaligned/tilted axicon geometry;
- vector Debye/Richards-Wolf focusing for quantitative high-NA focal/vector
  observables;
- vector spectral Fresnel transmission for planar dielectric interfaces;
- optional OpticStudio POP integration as an independent prescription-dependent
  cross-validation backend;
- Phase 3B broadband orchestration that reruns the selected single-wavelength
  optical route across an explicit spectrum.

The solver policy in `vbb_study/solver_policy.py` remains authoritative for which
solver is eligible for a claim.

---

## 2. Angular-spectrum propagation

The free-space propagation routes use sampled angular-spectrum methods. Their
limits are numerical rather than conceptual:

- finite field of view and finite spatial sampling;
- band-limit / aliasing constraints at each propagation distance;
- truncation if the field reaches the computational boundary;
- interpolation error for rotated-plane spectral transforms;
- evanescent handling depends on the selected propagation route.

Grid-convergence and power bookkeeping therefore remain required for quantitative
results.

---

## 3. SLM model

The current SLM implementation **does** include:

- finite active area;
- physical pixel pitch;
- phase pixelation;
- phase quantisation;
- blaze/carrier phase;
- several fill-factor models;
- coherent unmodulated dead-space / zero-order handling;
- hooks for panel-specific phase LUTs and static phase maps in the calibrated
  digital-twin paths;
- registration and fringing/crosstalk sensitivity machinery in the system-error
  branch.

Remaining limitations are calibration-driven:

- the actual SLM1/SLM2 grey-to-phase LUT at the bench wavelength, incidence and
  polarisation must be measured;
- spatial phase nonuniformity/flatness must come from a measured map for an
  absolute bench claim;
- fringing-field parameters remain a surrogate until fitted to panel data;
- temporal LCOS phase flicker is not yet a measured dynamic model.

---

## 4. Explicit 4F relay

The research route now contains an explicit physical 4F path:

`object -> L1 -> Fourier-plane iris -> L2 -> output`

It supports physical propagation distances, lens despace, focal-length error,
lens decentre, finite apertures, user-supplied OPD maps, iris offset/radius and
rigid lens-plane tilt through rotated angular-spectrum mapping.

The principal remaining limitation is that L1/L2 are still thin/paraxial lens
models. Strongly tilted or thick multi-element lenses require a dedicated
surface-by-surface prescription treatment; the Phase 3A OpticStudio backend is
the current independent prescription-dependent route for those cases.

---

## 5. Refractive axicon

There are two distinct axicon fidelity levels and they must not be conflated.

The scalar/source-scale research branch includes:

- exact normal-incidence Snell cone angle rather than only the shallow-cone
  approximation;
- lateral beam/apex decentre;
- rotated-plane tilt sensitivity;
- finite clear aperture when supplied;
- base-angle and refractive-index variation;
- rounded/hyperboloidal and flat/blunt apex defects;
- optional measured surface-height error map.

For quantitative non-zero rigid axicon tilt in the currently implemented
flat-first macroscopic geometry, **Phase 2H is the authoritative in-repository
route**. It represents two real dielectric surfaces, performs exact vector Snell
refraction at the entrance and conical exit surfaces, solves the actual ray/cone
intersection, includes finite glass optical path and reconstructs a common-eikonal
vector boundary field on a fixed laboratory plane before vector propagation.
The calibrated bench route explicitly refuses to substitute the older rotated
thin-phase surrogate for this case.

Phase 3B additionally exposes a generic arbitrary-normal local vector
Snell/Fresnel boundary primitive for reuse outside the axicon-specific Phase 2H
implementation. It does not replace the Phase 2H solver.

The remaining limitation is that Phase 2H is a **vector geometrical-optics /
eikonal boundary-field model**, not a full-volume FDTD/FEM electromagnetic
solution throughout the glass. Its common-eikonal and geometric validity gates
therefore remain part of the claim boundary. Unsupported surface ordering,
microscopic apex physics beyond the model resolution, or regimes outside those
gates must not be silently extrapolated.

---

## 6. Wavelength dependence and femtosecond bandwidth

Phase 3B adds broadband linear-optics support:

- measured spectrum CSV ingestion;
- per-wavelength rerunning of an existing optical route;
- wavelength-dependent refractive index through explicit material models;
- slow-detector spectral intensity integration;
- optional spectral phase with GDD/TOD;
- optional relative coherent temporal-field reconstruction.

This does **not** automatically make every run broadband-calibrated. A measured
laser spectrum is required for a measured-spectrum claim. A transform-limited
Gaussian spectrum is available only as a labelled numerical control.

The code includes a named fused-silica Malitson Sellmeier model. Other glasses
must be explicitly identified or supplied. A single constant refractive index is
allowed only as a labelled non-dispersive comparison model and is not sufficient
for a calibrated broadband-dispersion claim.

---

## 7. Objective and vector focusing

The quantitative focal solver is vector Debye/Richards-Wolf and produces
`Ex`, `Ey` and `Ez`. It assumes an aplanatic sine-condition objective unless
additional measured pupil information is supplied.

Phase 3B can apply a measured objective pupil amplitude-transmission map, OPD map
and validity mask before the vector Debye calculation.

Remaining limitations:

- the objective is not automatically a full manufacturer prescription;
- measured pupil maps must be registered externally and supplied on the declared
  simulation grid;
- strong objective misalignment or thick multi-element surface effects are not
  replaced by arbitrary Zernikes;
- the omitted absolute Debye prefactor means the Debye field remains a relative
  vector reference unless an independently calibrated energy mapping is applied.

---

## 8. Sample/interface

The vector interface solver handles a planar dielectric boundary spectrally:

- transverse wavevector conservation;
- local s/p decomposition;
- Fresnel transmission and reflection;
- `Ex`, `Ey`, `Ez` reconstruction;
- energy/transversality diagnostics.

Limitations:

- the canonical solver is planar;
- curved, rough or structured interfaces require a separate model;
- sample tilt must use the dedicated tilted-interface/vector route rather than a
  post-processing image shift;
- material dispersion must be supplied wavelength-by-wavelength for broadband
  propagation.

---

## 9. Detector/camera prediction

The repository now contains calibrated camera-coordinate comparison and Phase 3B
adds a detector-transfer layer that can include:

- supplied PSF convolution;
- relative detector-response map;
- background;
- saturation;
- calibrated object-plane pixel scale, rotation and centre through the existing
  camera comparison code.

These quantities are never fitted silently to make simulation and experiment
agree. Missing detector calibration remains missing calibration.

---

## 10. Uncertainty

Two uncertainty levels now exist:

1. reduced metric propagation from supplied calibration uncertainties;
2. Phase 3B full-field Monte Carlo reruns that return pixelwise mean, standard
   deviation and confidence limits.

The full-field path requires every Monte Carlo sample to remain on the same
physical output grid. Hidden registration, scale fitting or image recentering is
forbidden.

A probability distribution is only physically meaningful when its spread comes
from measured/manufacturer uncertainty or explicitly labelled sensitivity
assumptions.

---

## 11. Material response and nonlinear ultrafast physics

The optical digital twin remains fundamentally a **linear optical model**.
Existing fluence, threshold and empirical response layers are planning or
calibrated statistical tools. They do not mechanistically solve:

- Kerr self-focusing;
- multiphoton or tunnel ionisation;
- free-carrier/plasma generation and defocusing;
- nonlinear absorption;
- self-phase modulation;
- heat accumulation;
- melt flow;
- stress and cracking;
- void/channel formation;
- permanent refractive-index change.

Therefore a prediction of optical field/fluence is not automatically a
prediction of permanent material modification.

A future nonlinear-material branch should be treated as a separate physical
model with its own validation evidence rather than folded into the linear solver
without provenance.

---

## 12. Zemax / OpticStudio cross-validation

Phase 3A provides an optional OpticStudio POP backend. Its purpose is independent
prescription-dependent cross-validation, not replacement of the Python solver
hierarchy.

Important limits:

- POP is not a full Maxwell/material-response solver;
- traditional ZBF exchange is a transverse-field interchange and must not be
  used to claim independent `Ez` validation;
- a live OpticStudio licence and actual `.ZOS/.ZMX` prescription are required
  for a real cross-validation run;
- software agreement does not equal experimental validation.

---

## 13. Experimental validation status

No numerical route becomes experimentally validated because it passes unit tests,
looks physically plausible or agrees with a second numerical solver.

Experimental validation requires:

- a declared bench state;
- measured calibration values;
- independent camera/power/wavefront data;
- fixed comparison coordinates and preprocessing;
- predeclared metrics/acceptance criteria;
- provenance linking the measurement and simulation inputs.

Phase 3B is designed to make that comparison possible without changing the
accepted Phase 1–2C evidence base.
