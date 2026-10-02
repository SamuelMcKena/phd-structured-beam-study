# Conditional 200 mm inter SLM registration study

The subsequent user clarification is SLM1 → approximately 200 mm free space →
SLM2 → 4F order selection → physical axicon. The distance is approximate and
user-reported, not a measured transfer calibration. Both panels retain their
own 20-pixel blaze. Correction is zero: **FLAT-CORRECTION PHYSICS-ISOLATION
BASELINE**. Aligned unfolded coordinates, identical ideal 256-level responses,
same-sign x carriers and a unit-magnification ideal 4F image are conditional
assumptions, not measured hardware facts.

The earlier local registration results and exact identity-transfer A/B
role-exchange validation remain valid in their stated scope. They are not an
architecture-ranking theorem for separated panels: vortex multiplication and
free-space propagation do not commute. A conditional separated-panel
comparison is now physically definable even without measured panel LUTs;
predicting an experimental architecture winner remains calibration-dependent.

At 1029 nm, 8 µm pitch and a 20-pixel carrier, the nominal upstream carrier
walks 1.28628 mm over 200 mm. SLM2's numerical origin is fixed on that nominal
ray for all registration states. No per-state centroid recentering is applied.
The imposed beam/hologram-axis decentre is zero; any propagated centroid
change is recorded separately as an outcome. Actual intensity-centroid
alignment and nominal-axis alignment must not be conflated.

The first carrier is demodulated only in the coordinate representation. The
SLM commands still contain BOTH carriers, and the physical selected channel
contains their sum. No iris is placed between panels. The numerical source
frequency radius is expanded in convergence controls independently of the
downstream diagnostic iris. That iris is adjusted per charge and beam radius,
then fixed for all offsets and both architectures in that case. Its physical
order purity remains uncalibrated.

SLM1 subcells have analytic sinc Fourier integrals. A chunked Bluestein sum
evaluates the propagated field on SLM2's actual fractional-origin coordinates.
Illumination quadrature changes neither the commanded 8 µm pixels nor their
phase allocation. Zero distance uses the independent exact intersection-of-
lattices integral; this control does not claim convergence of a finite-band
algorithm arbitrarily close to zero distance. Overlaps accumulate in double
precision to resolve small bench infidelities.

The downstream scalar ideal-cone axicon is conditional on a 20-degree BASE
angle and index 1.458; exact optic identity, orientation, effective cone angle
and image-plane position remain unmeasured. No immediate intensity change is
ascribed to its phase-only action. All post-axicon images refer to specified
nonzero propagation distances. This is broad-morphology screening, not a vector
objective focal-detail or material-processing prediction.

## Reproduction

From the repository root, with dependencies installed:

```bash
export PYTHONPATH=.
export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2
python -m pytest -q tests/test_slm_pixel_registration.py tests/test_slm_registration_architectures.py tests/test_registration_reference.py tests/test_registration_interpanel.py
python -m tools.run_registration_interpanel_study --stage sweep
python -m tools.run_registration_interpanel_study --stage controls
python -m tools.run_registration_interpanel_study --stage postcontrols
python -m tools.run_registration_interpanel_study --stage extra
python -m tools.plot_registration_interpanel_study --data outputs/validation/registration_interpanel_200mm --output outputs/validation/registration_interpanel_200mm/figures
python -m tools.build_registration_interpanel_report --data outputs/validation/registration_interpanel_200mm --output outputs/reports/registration_interpanel
```

The dedicated Actions workflow retains CSV, complex-field NPZ, source hashes,
manifests, PNG figures and reports as artifacts. Heavy evidence is excluded
from Git. Primary 2 µm image-space metrics are accompanied by 1 µm controls;
pixel-maximum, fitted radius and azimuthal metrics must not be treated as
converged solely because a complex-field overlap passes. Prior evidence is
preserved as an explicitly identity-transfer reference, not silently relabelled
as the 200 mm laboratory route.
