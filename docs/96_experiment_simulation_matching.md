# Experiment -> simulation matching contract

## Status

This stage is a **single-physical-mechanism screening layer** between measured
multi-plane intensity data and the existing correction work.  It is not yet a
joint inverse solution and it must not be described as uniquely identifying the
laboratory error.

The matcher regenerates every candidate from the canonical complex-field route:

`Gaussian -> SLM1 -> SLM2/carrier -> propagated 4F + physical iris -> selected order -> physical axicon -> free-space z planes`

Rendered error-atlas PNG/SVG files are never used as inference templates.

## Required measured-stack information

Use `examples/experimental_stack_manifest.example.json` as the schema example.
For real data replace every placeholder with measured values.  The loader
requires:

- a physical pixel pitch in x and y;
- the camera-array centre coordinate relative to the declared laboratory axis;
- explicit transpose/x-flip/y-flip orientation;
- an explicit z coordinate for every plane;
- a declared z reference;
- one scalar 2-D intensity array per plane.

The current loader accepts NPY, NPZ, TXT, CSV, BMP, PNG and TIFF scalar arrays.
Colour images are rejected because conversion to scalar intensity would be an
unrecorded radiometric assumption.

If the camera-stage z=0 is not the model's post-axicon plane, pass an explicit
`--model-z-offset-mm`.  The code does not silently slide the experimental stack
along z to make it look better.

## Preprocessing policy

Default preprocessing is deliberately **none**.  In particular the loader does
not automatically:

- subtract a background;
- clip negative values;
- recenter each image;
- resize to the simulated grid;
- normalise each image to its own peak;
- equalise power between z planes.

An explicitly supplied dark frame or scalar background can be applied through
`preprocess_stack`; every operation is appended to stack metadata.  Simulation
intensities are interpolated onto the *measured physical x/y coordinates* only at
the comparison boundary.  Complex-field propagation itself remains on the
native simulation grid.

## Objective function

For a measured stack `E` and simulated intensity stack `S`, one non-negative
global radiometric gain is fitted across every valid pixel and every z plane:

`g = max(0, <E,S>/<S,S>)`.

The default rank score is

`sqrt(sum((E-gS)^2) / sum(E^2))`.

This single gain handles an unknown overall camera/laser scale without erasing
relative throughput changes between z planes or between physical candidates.
Per-plane NRMSE, correlation and own-peak shape residuals are recorded as
diagnostics, but own-peak-normalised quantities do **not** determine the rank.

There is no hidden x/y registration.  That is intentional: free registration is
degenerate with physical beam/axicon decentre.  If the camera origin is uncertain,
it should later be fitted as an explicit nuisance parameter with an experimental
prior rather than removed independently in every frame.

## Candidate mechanisms

The default screen matches the seven mechanisms used in the B0/V1/V3 atlas:

1. input-beam x decentre;
2. 4F iris x offset;
3. dual-SLM phase-stroke scale;
4. axicon x decentre;
5. hyperboloidal rounded axicon tip;
6. declared L1 astigmatism OPD;
7. declared L1 trefoil OPD.

Declared L1 quadrafoil is available as an optional eighth family because it is
useful for the Miao benchmark/phase-retrieval family.  The current parameter
grids are **unmeasured sensitivity levels**.  They are useful for code validation
and coarse hypothesis screening only.  Real inference ranges should be narrowed
or expanded using stage calibration, camera calibration, SLM calibration,
mechanical tolerances, profilometry/interferometry or other independent evidence.

## Miao handoff

A low intensity residual is not a retrieved phase and never becomes a correction
map automatically.

- aperture/clipping -> fix/model the amplitude mechanism;
- SLM stroke -> calibrate the panel response first;
- axicon/beam decentre -> treat as alignment/coordinate hypotheses first;
- rounded tip -> retain the surface-geometry forward model;
- declared astigmatism/trefoil/quadrafoil -> eligible to motivate complex phase
  retrieval, but the fitted scalar is **not** the correction map.

For a phase-like winner the next stage is to retrieve the actual complex phase
from the measured z-stack, then propagate/map that phase to the intended SLM
correction plane before applying a conjugate.

The code records `correction_map_ready=false` for every output of this screening
stage.

## CLI

Real stack:

```bash
python tools/run_experiment_simulation_match.py \
  --manifest path/to/manifest.json \
  --case V3 \
  --grid-n 512 \
  --output-dir outputs/my_measurement_match
```

Override one candidate grid:

```bash
python tools/run_experiment_simulation_match.py \
  --manifest path/to/manifest.json \
  --case V3 \
  --families axicon_decentre_x lens1_astigmatism \
  --values axicon_decentre_x=0,0.00025,0.0005,0.00075,0.001 \
  --values lens1_astigmatism=0,0.1,0.2,0.3,0.4
```

Synthetic full-route validation:

```bash
python tools/run_experiment_simulation_match.py \
  --synthetic-known axicon_decentre_x=0.0005 \
  --case V1 \
  --grid-n 128 \
  --families axicon_decentre_x \
  --values axicon_decentre_x=0,0.0005,0.001
```

Outputs are `candidate_ranking.csv`, `match_result.json` and
`miao_handoff.json`.

## Next extension after this contract is validated

The next scientific step is not to add arbitrary image metrics.  It is to test
**identifiability and combined errors**: introduce a small number of physically
justified nuisance/calibration parameters, quantify parameter correlations, and
only then optimise more than one mechanism at once.  Real measured data should
enter before this is promoted from screening to quantitative inverse modelling.
