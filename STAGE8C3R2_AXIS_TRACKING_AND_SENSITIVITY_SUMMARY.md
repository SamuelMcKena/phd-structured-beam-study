# Stage 8C.3R.2 Axis Tracking and Sensitivity Summary

Starting point for Stage 8C.3R.2 recovery: committed Stage 8C.3R.1 baseline
`6726007`.

Stage 8C.3R.2 locks annular beam-axis diagnostics for the free-space
component-plane model. It demotes the raw brightest annular pixel to a labelled
diagnostic, uses fitted ring/core/ROI axis tracking for steering and FOV
convergence, and adds multi-point diagnostic response curves for the physically
represented perturbation families.

## Created

- `vbb_study/digital_twin/annular_axis_tracking.py`
- `docs/35_annular_axis_tracking_and_free_space_sensitivity_lock.md`
- `STAGE8C3R2_AXIS_TRACKING_AND_SENSITIVITY_SUMMARY.md`
- `tests/test_stage8c3r2_annular_axis_tracking.py`

## Modified

- `vbb_study/digital_twin/component_plane_metrics.py`
- `vbb_study/digital_twin/component_plane_validation.py`
- `vbb_study/digital_twin/component_plane_figures.py`
- `vbb_study/digital_twin/__init__.py`
- `notebooks/digital_twin/00_full_beam_to_write_cockpit_MVP.ipynb`

## Axis Estimator

Primary axis selection is:

1. fitted annular ring centre
2. fitted dark-core centre
3. central ROI intensity centroid
4. phase singularity when a valid complex phase field is supplied
5. no reliable axis

The raw brightest pixel remains available as:

```text
not_a_primary_axis_metric_for_annular_fields
```

It is not used for steering validation, FOV classification, or headline axis
error. Deformed or crop-limited planes return caution/unreliable states instead
of fake precision.

## Validation And FOV

`validate_beam_tilt` now reports expected and measured slopes using fitted
annular-axis trajectory:

- `expected_slope_x`, `expected_slope_y`
- `measured_slope_x`, `measured_slope_y`
- `absolute_error`
- `relative_error`
- `fit_quality`
- `grid_resolved_displacement`

`fov_convergence_check` reports ring/core/trajectory differences and keeps raw
peak movement as diagnostic-only. Clean annular baselines remain
`numerically_reliable` even when raw brightest-pixel azimuth changes between
standard and expanded FOV. Undersized FOV remains a deliberate
`invalid_out_of_frame` demonstration.

## Response Curves

Stage 8C.3R.2 adds ten multi-point diagnostic families:

- vortex-centre offset, axicon fixed
- axicon-centre offset, vortex fixed
- input beam decentre with fixed active area
- input beam tilt
- pupil decentre/clipping
- defocus
- astigmatism
- coma
- spherical aberration
- zero-order leakage fraction

Each response row is labelled:

```text
Diagnostic sensitivity sweep. Not an experimentally measured laboratory tolerance.
```

## Preview Figures

```text
outputs/figures/digital_twin/stage8c3r2_annular_axis_tracking_validation.png
outputs/figures/digital_twin/stage8c3r2_individual_response_curves.png
outputs/figures/digital_twin/stage8c3r2_free_space_study_summary.png
```

## Governance

The model remains a free-space optical/fluence diagnostic with `n = 1.0` and
`final_export_allowed=False`. No material propagation, sample interface, plasma,
nonlinear/thermal response, dose accumulation, writing simulation, 3D
visualisation, GUI work, or calibrated material-response prediction is added.
Stage 8D has not been started.
