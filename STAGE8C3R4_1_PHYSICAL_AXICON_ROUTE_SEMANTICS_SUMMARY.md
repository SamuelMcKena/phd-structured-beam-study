# Stage 8C.3R.4.1 Physical-Axicon Route Semantics Summary

Starting point for Stage 8C.3R.4.1: `4c88f08` (`Stage 8C.3R.4 component-owned route architecture correction`).

This cleanup keeps the component-owned route scaffold and removes phantom
objective language from the executed physical-axicon route. The current
physical-axicon route has the axicon as the final represented optical element.
No objective transformation is represented.

## Before

```text
source_field
-> source_boundary_condition
-> input_aperture
-> source_to_physical_axicon
-> physical_axicon_input_boundary
-> physical_axicon
-> after_physical_axicon_boundary
-> physical_axicon_to_after_objective
-> after_objective_boundary
-> after_objective_to_reference
-> reference_plane
```

## After

```text
source_field
-> source_boundary_condition
-> input_aperture
-> source_to_physical_axicon
-> physical_axicon_input_boundary
-> physical_axicon
-> after_physical_axicon_boundary
-> post_axicon_free_space_segment
-> post_axicon_diagnostic_boundary
-> post_axicon_to_reference_segment
-> reference_plane
```

## Scope

Component-owned physical-axicon route scaffold.

Active physics:

- source complex field
- input aperture
- free-space propagation segments
- thin scalar physical axicon phase and clear aperture
- free-space reference-plane diagnostics

Warning-only / future:

- steering mirrors
- SLM1/SLM2
- 4F Fourier filtering
- relay optics
- pupil/objective optics
- mechanical axicon tilt

The route-inspection rows now include `transform_applied` so real optical
transforms, free-space propagation segments, inactive diagnostic boundaries, and
active boundary perturbations can be distinguished.

The model remains free-space optical/fluence diagnostic only (`n = 1.0`,
`final_export_allowed=False`). No material/interface, writing trajectory, 3D
visualisation, GUI work, plasma, nonlinear propagation, thermal physics, dose
accumulation, calibrated material-response prediction, objective model, or Stage
8D work has been added.
