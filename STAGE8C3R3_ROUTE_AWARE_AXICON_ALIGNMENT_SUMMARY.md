# Stage 8C.3R.3 Route-Aware Axicon Alignment Summary

Starting point for Stage 8C.3R.3: `47fc599` (`Stage 8C.3R.2 annular axis tracking and sensitivity study lock`).

Stage 8C.3R.3 adds a route-aware physical-axicon diagnostic path. It corrects
the architecture so field tilt, beam decentre, aperture changes, phase elements,
and future mechanical tilts are tied to an explicit injection location and
downstream route graph rather than being exposed as location-free controls.

Stage 8C.3R.4 supersedes the too-narrow C3R.3 interpretation by making the
route an ordered component/segment chain. Supported errors are now owned by
named components and field-state controls are labelled boundary conditions, not
generic component-misalignment models.

## Created

- `vbb_study/digital_twin/route_aware_axicon.py`
- `tests/test_stage8c3r3_route_aware_axicon.py`
- `docs/36_route_aware_physical_axicon_alignment.md`
- `STAGE8C3R3_ROUTE_AWARE_AXICON_ALIGNMENT_SUMMARY.md`

## Modified

- `vbb_study/digital_twin/__init__.py`
- `notebooks/digital_twin/00_full_beam_to_write_cockpit_MVP.ipynb`

## Route Model

Represented physical-axicon route:

```text
source_plane
-> after_beam_conditioning
-> before_physical_axicon
-> physical_axicon_plane
-> after_physical_axicon
-> after_objective
-> free_space_reference_plane
```

The propagation segment distances are editable. Defaults are diagnostic demo
geometry only and are not measured laboratory distances.

## Location-Aware Perturbations

Each perturbation record declares:

- perturbation type
- magnitude and units
- injection location
- route
- implementation plane
- downstream elements affected
- active / warning-only / future status

Generic `beam_tilt` is no longer the route-aware primitive. Field tilt is
represented as:

```text
field_tilt_x_mrad
field_tilt_y_mrad
field_tilt_location
```

Beam decentre is represented as:

```text
beam_decentre_x_um
beam_decentre_y_um
beam_decentre_location
```

Compatibility aliases from earlier C3R work are normalised into these generic
controls.

## Physical Distinction

Upstream field tilt propagates to the fixed physical axicon and changes the
incident centroid/angle/overlap before the axicon transmission is applied.

Field tilt injected after the physical axicon is labelled:

```text
post_axicon_steering_test
```

It is an analytical downstream steering validation, not an upstream axicon fault.

Field tilt injected after the objective/downstream steering plane affects only
the remaining represented downstream segment and does not change axicon
incidence.

## Preview Figures

```text
outputs/figures/digital_twin/stage8c3r3_route_aware_axicon_pipeline.png
outputs/figures/digital_twin/stage8c3r3_upstream_vs_post_axicon_tilt_comparison.png
outputs/figures/digital_twin/stage8c3r3_axicon_alignment_sensitivity_atlas.png
```

The comparison figure is now a field-tilt injection-location sweep.

## Governance

The model remains free-space optical/fluence diagnostic only (`n = 1.0`,
`final_export_allowed=False`). No material/interface, writing trajectory, 3D
visualisation, GUI work, plasma, nonlinear propagation, thermal physics, dose
accumulation, or calibrated material-response prediction has been added. Stage
8D has not been started.
