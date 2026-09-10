# Phase 3B laboratory measurement checklist

The purpose of this checklist is to move the digital twin from a fixed-bench
nominal prediction to a measured-bench prediction without inventing component
parameters.

## Laser/input state

Required for dimensional bench matching:

- central wavelength and measured optical spectrum;
- pulse duration and, when temporal reconstruction is required, spectral phase
  or equivalent GDD/TOD evidence;
- pulse energy at a declared reference plane;
- x/y 1/e field-amplitude beam radii at the SLM plane;
- beam-centre coordinates relative to the SLM active area;
- x/y pointing angle;
- wavefront curvature or measured input OPD;
- time-series statistics for pointing/pulse-energy jitter when uncertainty from
  shot-to-shot variation is required.

## SLM1 / SLM2

For each panel separately:

- panel serial/ID;
- active orientation on the bench;
- wavelength, incidence angle and polarisation used for calibration;
- grey-level to phase LUT;
- usable phase stroke;
- static panel OPD/phase map after factory correction state is declared;
- fill factor / pixel geometry confirmation;
- measured or fitted fringing/crosstalk response if quantitative diffraction
  efficiency across sharp phase features is required;
- SLM1-to-SLM2 coordinate translation, rotation and scale mapping.

## 4F relay and Fourier filter

- L1/L2 part numbers and focal lengths;
- object-L1, L1-iris, iris-L2 and L2-output distances;
- L1/L2 clear apertures;
- lens OPD/wavefront maps if manufacturer or measured data are available;
- physical +1 order position;
- iris centre and radius;
- selected-order transmitted power fraction.

## Axicon

- manufacturer and part number;
- glass/material identity;
- exact angle convention and nominal/measured value;
- clear aperture;
- centre thickness if a thick-surface model is required;
- orientation: flat/conical face order;
- coating state;
- apex centre relative to the incoming beam;
- tip profile/rounding/flat radius when resolvable;
- surface-height/figure map when available.

A single quoted refractive index is not enough to claim calibrated broadband
axicon dispersion unless the glass identity or wavelength-dependent index is
known.

## Objective / relay to sample

- objective make/model;
- NA;
- effective focal length;
- effective entrance-pupil radius;
- beam-to-pupil magnification;
- pupil fill;
- pupil centre/rotation relative to the simulated field;
- measured pupil amplitude transmission if available;
- objective OPD/wavefront map if available;
- immersion/environment refractive index where relevant.

## Sample/interface

- sample material identity;
- surface orientation;
- refractive-index/dispersion model;
- coating state;
- surface position and tilt;
- depth of the comparison plane inside material.

For fused silica, the code contains a named Malitson Sellmeier model. It is not
used automatically for an axicon or any other component merely because its
single-wavelength index is similar.

## Camera / beam profiler

- object-plane metres per pixel;
- camera rotation;
- optical-axis/reference centre pixel;
- dark/background frame;
- saturation level;
- exposure/attenuation state;
- measured PSF or MTF when detector blur is non-negligible;
- relative pixel-response map when required.

## Energy ledger

Absolute fluence additionally requires measured transmission/efficiency for:

- SLM1;
- SLM2;
- +1 order selection;
- remaining 4F path;
- axicon;
- objective;
- sample interface.

The code must not renormalise away passive losses after they have been applied.

## Minimum useful first campaign

If time is limited, collect these first:

1. measured laser spectrum;
2. actual x/y beam size and centre at SLM1/SLM2;
3. SLM1 and SLM2 phase LUTs at 1030 nm bench conditions;
4. SLM static wavefront/phase maps;
5. real iris radius/centre and 4F separations;
6. axicon manufacturer part number, glass and exact angle convention;
7. camera object-plane scale/rotation;
8. one multi-z camera stack for G0/B0/V1/V3 using a declared optical setup state.

Those measurements will improve predictive accuracy more than adding another
nominal propagation algorithm.
