# q=20 inverse/correction literature comparison

This note records the literature basis used for the September 2026 model/poster redesign. It is not itself experimental evidence.

## Miao et al. (2022): intensity-only Bessel wavefront retrieval

B. Miao, L. Feder, J. E. Shrock and H. M. Milchberg, **"Phase front retrieval and correction of Bessel beams,"** *Optics Express* **30**, 11360-11371 (2022), DOI `10.1364/OE.454796`.

Key method elements:

- retrieve the complex amplitude of a Bessel beam from intensity measurements alone;
- fit a Bessel-mode representation and transverse wavenumber;
- use the Bessel radial-to-axial/stationary-phase mapping to associate focal-line positions with input-aperture annuli;
- reconstruct the input phase, decompose it into low-order Zernike content, and apply conjugate correction with a deformable mirror;
- demonstrate measured post-correction improvement for J0 and J16 beams.

Use in this repository:

- motivates a compact physically interpretable low-order pupil-phase basis rather than an unrestricted arbitrary phase fit;
- motivates separating radial/axisymmetric and non-axisymmetric aberration terms;
- provides the main published benchmark for intensity-only Bessel correction.

Difference from this repository:

- the q=20 workflow propagates every candidate through the explicit SLM2 carrier, finite 4F/+1-order iris, refractive axicon, free-space propagation and measured 5.5 um detector response;
- absolute z and uncalibrated bench nuisance parameters are fitted on train planes only and then frozen;
- odd z planes are held out from parameter selection;
- the diagnosed axicon-input phase is not assumed to be directly equal to the upstream SLM2 command;
- SLM2 candidates are rejected when they change q=20 winding, even if intensity closure improves;
- there is no post-correction BeamGage measurement yet, so numerical correction must remain labelled prediction.

## Li et al. (2020): high-order Bessel-Gaussian system correction

J. Li et al., **"The aberration correction of high-order Bessel-Gaussian beams,"** *Optik* **221**, 163968 (2020), DOI `10.1016/j.ijleo.2019.163968`.

The paper demonstrates correction of high-order Bessel-Gaussian beam aberration using in-situ wavefront correction based on orthogonal mode decomposition / constructive interference, with the SLM carrying both the structured-beam hologram and compensation.

Design implication here:

- high-order vortex/Bessel fields are sufficiently sensitive to system aberration that correction should be evaluated on the complete structured field, not merely a Gaussian proxy;
- combining generation and correction on a phase-only SLM is experimentally established, but the actuator-to-diagnostic-plane mapping must remain explicit in the digital twin.

## Cheng et al. (2020): physically structured phase correction across Bessel depth of field

H. Cheng, C. Xia, S. M. Kuebler and X. Yu, **"Aberration correction for SLM-generated Bessel beams propagating through tilted interfaces,"** *Optics Communications* **475**, 126213 (2020), DOI `10.1016/j.optcom.2020.126213`.

Key result:

- a single SLM phase map with spatially varying ellipticity can compensate aberration from oblique/tilted interfaces over the Bessel depth of field;
- beam-quality metrics are evaluated across the propagation range, not at one plane only.

Design implication here:

- prefer smooth low-dimensional phase families that correspond to physically plausible misalignment/astigmatic distortions;
- score correction over the complete z-stack and reject one-plane solutions.

## Basu et al. (2024): topology/singularity-aware structured-light phase compensation

D. Basu, S. Chejarla, S. Maji, S. Bhattacharya and B. Srinivasan, **"An adaptive optical technique for structured beam generation based on phase retrieval using modified Gerchberg-Saxton algorithm,"** *Optics & Laser Technology* **170**, 110244 (2024), DOI `10.1016/j.optlastec.2023.110244`.

Key result:

- modified Gerchberg-Saxton phase retrieval with aperture optimisation for structured/OAM beams;
- explicit phase-singularity detection and avoidance of spurious phase jumps;
- reported modal-purity enhancement and crosstalk reduction after compensation.

Design implication here:

- a lower intensity loss is not sufficient for vortex correction: topological/singularity preservation must be an explicit acceptance condition;
- this motivates the q=20 winding-contour gate used by the SLM2 correction workflow.

## Katkovnik & Astola (2012): joint phase/amplitude inverse in a 4F system

V. Katkovnik and J. Astola, **"Phase retrieval via spatial light modulator phase modulation in 4f optical setup: numerical inverse imaging with sparse regularization for phase and amplitude,"** *JOSA A* **29**, 105-116 (2012), DOI `10.1364/JOSAA.29.000105`.

Design implication here:

- intensity-only inverse problems in 4F systems can contain both phase and amplitude mismatch;
- fitting a small diagnostic amplitude nuisance alongside phase can prevent the phase estimate from absorbing throughput/filtering mismatch;
- the diagnostic amplitude term must not be mislabelled as phase-only SLM correction.

## Current repository design rule

The redesign therefore separates three questions:

1. **Does the bench-matched nominal model reproduce the real q=20 stack?**  Fit only model-bound nuisance values on train planes, then score held-out planes.
2. **What smooth complex residual is supported by the real data?**  Fit low-order phase and a separate low-order amplitude nuisance; use held-out z planes as external validation.
3. **What can the phase-only SLM2 actually correct?**  Propagate only the phase-correctable residual upstream through the explicit 4F route, optimise a zero-winding SLM2 addition, and reject any candidate that changes q=20 winding on the production grid.

This structure is intentionally more conservative than treating any high-correlation predicted camera image as measured correction evidence.
