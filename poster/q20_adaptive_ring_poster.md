# Programmable High-Order Vortex-Bessel Beams
## Detector-aware modelling and camera-in-loop correction of q = 20 ring distortions

**Poster source / claim ledger**

This file records the scientific claims used in the q=20 research poster. It intentionally separates measured data, numerical modelling, synthetic metric validation and proposed experimental correction.

### Aim
Generate and characterise high-order vortex/Bessel beams in the dual-SLM + finite-4F + axicon system, then improve the concentricity of the experimentally observed q=20 ring train without requiring a unique reconstruction of every unknown system aberration.

### Optical route represented on the poster
SLM1 -> SLM2 -> finite 4F / +1-order spatial filter -> refractive axicon -> propagation -> BeamGage detector.

The detector-aware numerical route is required to include the finite 4F order-selection step rather than applying an ideal analytical Bessel field directly at the detector.

### What is established in the repository
- A detector-aware numerical model of the q=20 observation route has been developed and used in the v2-v14 correction studies.
- Several inverse/model-space correction approaches were tested (complex-field closure, circular selected-order targets, finite-4F IFTA, multi-plane projection and annular low-order refinements).
- The v12-v14 optimisation lineage inherited a fixed-contour post-axicon winding gate. The nominal zero-correction baseline is rejected by that gate on the chosen 1.0-1.5 mm contours. This is therefore not valid evidence that the nominal q=20 beam has lost its physical topological charge; the diagnostic must be reformulated using a physically justified field plane and contours with adequate amplitude support.
- `tools/q20_sensorless_ring_metric_v15.py` implements an intensity-domain camera metric for ring quality and low-order annular trial phase maps (m = 2, 3, 4).
- `.github/workflows/q20-sensorless-ring-metric-v15.yml` validates that the metric ranks a circular multi-ring synthetic field as better than a deliberately twofold/fourfold-distorted synthetic ring train. This is a metric sanity check only.

### Ring-quality objective
For each significant bright annulus, the v15 metric measures:
- angular coefficient of variation;
- azimuthal Fourier distortion energy at m = 2, 3 and 4;
- radial peak wobble standard deviation;
- radial peak-to-peak variation.

A lower aggregate objective corresponds to a more concentric and angularly uniform ring train.

### Camera-in-loop correction concept
Keep the nominal q=20 hologram fixed and add a small correction phase on SLM2. Probe annular harmonic modes such as m = 2, 3 and 4 at several trial amplitudes, capture a BeamGage intensity image for each trial, evaluate the v15 ring metric, and retain the coefficient that improves the measured ring train. Repeat or jointly refine over several z planes when experimental acquisition supports it.

This is an image-based adaptive optimisation problem, not a claim that the current intensity stack uniquely identifies the physical phase error of every optical element.

### Evidence labels used on the poster
- **Measured:** pre-correction BeamGage data exist in the experimental campaign, but no corrected BeamGage capture is claimed in this poster source.
- **Numerical:** detector-aware propagation/correction studies and legacy structured-beam simulations.
- **Synthetic metric sanity check:** circular and deliberately distorted ring trains generated only to validate the v15 score. They are not experimental before/after images.
- **Proposed experiment:** camera-in-loop SLM modal sweep and multi-plane extension.

### Current limitation
A corrected experimental BeamGage q=20 image has not yet been acquired. The poster therefore does not present any synthetic or numerical field as an experimental post-correction measurement.

### Near-term experimental success criterion
A successful correction is an experimentally reproducible reduction in cross/fan angular structure and radial ring wobble while preserving the intended high-order vortex/Bessel field. Perfect recovery of a unique system wavefront is not required for this near-term result.

### Literature basis
1. B. Miao, L. Feder, J. E. Shrock, H. M. Milchberg, "Phase front retrieval and correction of Bessel beams," Optics Express 30, 11360-11371 (2022). DOI: 10.1364/OE.454796.
2. H. Kim et al., "Modal focal adaptive optics for Bessel-focus two-photon fluorescence microscopy," Optics Express 33, 680-693 (2025). DOI: 10.1364/OE.541033.
3. F. Luo et al., "Adaptive optics correction for Bessel beam propagating through spatially varying biological aberrations," Optics Express 34, 26872-26882 (2026). DOI: 10.1364/OE.606056.

### Poster takeaway
**The immediate correction target is experimentally verifiable ring concentricity, not an over-constrained claim of unique full-system wavefront reconstruction.**
