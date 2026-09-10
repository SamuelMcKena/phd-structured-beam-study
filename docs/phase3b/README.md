# Phase 3B — Bench-Calibrated Broadband Digital Twin

## Purpose

Phase 3B improves the accuracy of the existing structured-beam simulation by
feeding the accepted optical solvers wavelength-dependent and measured bench
information. It does **not** replace the Phase 1–2C solver evidence and it does
not claim experimental validation merely because a calibrated data path exists.

The intended chain is:

```text
measured laser spectrum / spectral phase
        |
measured input beam + SLM calibration
        |
explicit physical 4F relay
        |
dispersive axicon + Phase 2H two-surface vector/eikonal route when required
        |
per-wavelength free-space propagation
        |
measured-pupil vector Debye objective
        |
vector spectral Fresnel sample interface
        |
energy-weighted broadband detector integration
        |
measured detector PSF/response/background/saturation
        |
calibrated camera-coordinate comparison
        |
full-field Monte Carlo uncertainty
```

## New modules

- `vbb_study/equations/dispersion.py`
  - explicit three-term Sellmeier material model;
  - named fused-silica Malitson coefficients;
  - explicit constant-index comparison model;
  - no implicit material selection.
- `vbb_study/digital_twin/broadband_propagation.py`
  - measured-spectrum CSV ingestion;
  - transform-limited Gaussian control spectrum for tests only;
  - per-wavelength execution of an existing optical solver;
  - incoherent energy-weighted intensity integration for slow detectors;
  - optional GDD/TOD spectral phase and coherent temporal reconstruction.
- `vbb_study/equations/calibrated_objective.py`
  - measured objective amplitude transmission;
  - measured OPD map in metres;
  - measured pupil-validity mask;
  - passes the calibrated pupil into the existing vector Debye solver.
- `vbb_study/calibration/detector_transfer.py`
  - measured PSF convolution;
  - relative detector-response map;
  - supplied background;
  - explicit saturation.
- `vbb_study/calibration/full_field_uncertainty.py`
  - reruns the optical solver for every Monte Carlo sample;
  - returns pixelwise mean, standard deviation and confidence limits;
  - forbids hidden output-grid registration.
- `vbb_study/equations/vector_surface_refraction.py`
  - generic exact local vector Snell/Fresnel boundary condition at an arbitrary
    surface normal;
  - reusable boundary primitive only; it does not replace the established
    Phase 2H axicon solver.
- `vbb_study/digital_twin/phase3b_bench_broadband.py`
  - binds calibration bundle, spectrum, dispersive material state, measured
    pupil assets and detector assets around an existing single-wavelength
    optical route.

## Existing Phase 2H axicon authority

Phase 3B deliberately reuses, rather than duplicates, the established Phase 2H
refractive-axicon machinery. The Phase 2H route represents the macroscopic
plano-conical axicon with **two real dielectric surfaces**, exact vector Snell
refraction at both surfaces, exact ray/cone intersection, finite glass path, and
a common-eikonal vector boundary field on a fixed laboratory plane. That field
can then be propagated by vector ASM and handed to the Debye branch.

For the currently implemented flat-first macroscopic axicon geometry, Phase 2H
is therefore the authoritative in-repository route for non-zero rigid axicon
tilt. The older scalar rotated-thin-phase axicon approximation must not be used
as quantitative evidence for that case.

The remaining boundary is narrower than the earlier roadmap implied: Phase 2H
is a **vector geometrical-optics/eikonal boundary-field model**, not a full-volume
FDTD/FEM solution of the electromagnetic field inside the glass. Its own
validity/common-eikonal gates still apply. General thick multi-element lens
surface modelling is also not supplied by Phase 2H; OpticStudio remains the
independent prescription-dependent route for those components.

## Spectrum CSV contract

A measured spectrum file must contain:

```csv
wavelength_nm,spectral_intensity
1020.0,0.12
1025.0,0.65
1029.0,1.00
1033.0,0.62
1038.0,0.10
```

`wavelength_m` may be used instead of `wavelength_nm`. `energy_weight` may be
used instead of `spectral_intensity`. An optional `spectral_phase_rad` column may
be included in the same file. A separate phase CSV may also be supplied by the
calibration bundle.

The weights are normalised to pulse-energy fractions. They are **not** treated
as absolute joules unless pulse energy is independently calibrated.

## Detector integration

A beam profiler or camera integrates over many optical cycles, so the canonical
Phase 3B camera prediction is

```math
I_\mathrm{det}(x,y)=\sum_i w_i\left(|E_x(x,y,\lambda_i)|^2+|E_y(x,y,\lambda_i)|^2+|E_z(x,y,\lambda_i)|^2\right).
```

Cross-frequency optical interference is therefore not included in the slow
camera image. Coherent time-domain reconstruction is a separate optional output
and requires a supplied or deliberately declared spectral phase.

## Material policy

Fused silica may use the named Malitson Sellmeier model. Other materials are not
silently treated as fused silica. If only a single measured index is available,
Phase 3B can use a `ConstantIndexMaterial`, but its status is
`constant_index_no_dispersion` and broadband absolute-dispersion claims remain
blocked.

The axicon glass is especially important: a nominal index near 1.46 is not enough
to identify its dispersion law. Manufacturer glass identity or measured
wavelength dependence is required before treating broadband axicon propagation
as calibrated.

## Objective policy

The accepted vector Debye/Richards–Wolf solver remains unchanged. Measured
objective pupil calibration is applied to the transverse pupil field before that
solver:

```math
\mathbf{E}_\mathrm{pupil}'(x,y,\lambda)
= T_A(x,y)\exp\left[i\frac{2\pi}{\lambda}\,\mathrm{OPD}(x,y)\right]
M(x,y)\mathbf{E}_\mathrm{pupil}(x,y,\lambda).
```

`T_A` is amplitude transmission, OPD is in metres, and `M` is a supplied valid
pupil mask. Map resizing or fit-to-image registration is not performed.

## Material-response boundary

Phase 3B remains a **linear optical digital twin**. It can predict wavelength-
dependent optical field, fluence and uncertainty. It does not simulate:

- Kerr self-focusing;
- multiphoton/tunnel ionisation;
- free-carrier/plasma dynamics;
- nonlinear absorption;
- heat accumulation;
- stress, cracking or melt flow;
- permanent refractive-index change;
- void/channel formation.

Existing threshold and empirical response modules remain planning/calibrated
response layers, not mechanistic nonlinear propagation.

## Example configuration

See `config/phase3b_bench_calibrated.example.json`.

## Evidence rule

A Phase 3B run may be called **bench-calibrated** only for quantities whose
required measurements are actually supplied. A simulation output is not
experimental validation. Experimental validation requires an independently
measured comparison dataset and a predeclared acceptance criterion.
