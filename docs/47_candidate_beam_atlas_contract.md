# Stage 9B.0 Candidate Beam Atlas Contract

The candidate atlas uses the nominal F300 4F virtual bench to compare
hardware-exportable command candidates before bench calibration exists.

It is explicitly:

```text
nominally_simulated
hardware_command_exportable
not_bench_validated
final_export_allowed = false
```

## SLM Role Contract

```text
SLM1: vortex / structured phase conditioning
SLM2: command-domain carrier and future correction map only
```

SLM2 must not be used to generate an axicon phase in this atlas.

## Candidate Families

The initial atlas covers:

- Gaussian reference,
- vortex charge sweep,
- input beam size variants,
- input decentre variants,
- pinhole offset and radius robustness variants.

The study contract is stored in:

```text
configs/studies/cslm_nominal_4f_candidate_atlas_v1.json
```

## Robustness Sweeps

The atlas reports nominal sensitivity only:

- carrier cycles versus stop-offset transmission,
- pinhole radius versus stop-offset transmission,
- beam radius versus relay quality,
- input decentre versus output centroid.

These sweeps are not physical 4F validation and do not infer camera-plane
coordinates.

## Candidate Package

Each exported candidate package is written under:

```text
outputs/nominal_4f_candidate_runs/<run_id>/<candidate_id>/
```

Required files:

```text
run_manifest.json
candidate_manifest.json
nominal_4f_profile_snapshot.json
SLM1 phase_rad.npy
SLM1 quantised_rad.npy
SLM1 gray.png
SLM2 phase_rad.npy
SLM2 quantised_rad.npy
SLM2 gray.png
fourier_plane_pre_stop.png
fourier_stop_transmission.png
fourier_plane_post_stop.png
nominal_relay_output_xy.png
energy_ledger.csv
robustness_summary.csv
claim_boundary.md
```

Every manifest states:

```text
simulation_status = nominal_unvalidated
physical_4f_readiness = blocked
camera_validation = absent
material_prediction = absent
final_export_allowed = false
```

## Unsupported

Unsupported remains: direct Fourier-plane calibration, measured carrier-to-stop
mapping, camera model, local SLM phase calibration, physical order purity,
validated correction maps, physical axicon handoff, material response, and any
publication-ready final export.
