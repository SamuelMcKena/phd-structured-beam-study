# Stage 8C.3R.4 Component-Owned Physical-Axicon Route Scaffold Summary

Starting point for Stage 8C.3R.4: `b9f3927` (`Stage 8C.3R.3 route-aware physical axicon alignment study`).

Stage 8C.3R.4 corrects the C3R.3 route-aware architecture. The physical-axicon
route is now a component-owned physical-axicon route scaffold, not a complete
laboratory digital twin. Supported lab-realism errors belong to named
components and are applied in that component plane, then propagated through all
downstream represented segments and elements.

## Created

- `tests/test_stage8c3r4_component_route_architecture.py`
- `docs/37_component_owned_route_architecture.md`
- `STAGE8C3R4_COMPONENT_ROUTE_ARCHITECTURE_SUMMARY.md`

## Modified

- `vbb_study/digital_twin/route_aware_axicon.py`
- `vbb_study/digital_twin/__init__.py`
- `notebooks/digital_twin/00_full_beam_to_write_cockpit_MVP.ipynb`
- `STAGE8C3_ACTIVE_LAB_REALISM_SUMMARY.md`

## Executed Route

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

The route records component pose, local transform, segment distance, incoming
and outgoing field metrics, energy, centroid, angle, aperture overlap, downstream
consequences, status, and warnings at every executed stage.

## Active Physics

Currently active represented physics includes:

- source complex field
- input aperture
- free-space propagation segments
- thin scalar physical axicon phase and clear aperture
- free-space reference-plane diagnostics

Currently active represented component controls include input aperture
radius/decentre, physical axicon lateral decentre, physical axicon axial offset
through adjacent segment distances, physical axicon clear aperture, and scalar
axicon cone phase parameter.

Field-state controls remain only as explicit boundary conditions at named
planes. They declare the physical approximation, possible upstream hardware
error they emulate, and downstream components that consume them.

## Warning / Future

Steering mirrors, SLM1/SLM2, explicit 4F Fourier filtering, relay optics,
pupil/objective optics, and physical axicon mechanical tilt remain warning-only
or future-stage. No unsupported optic is silently represented by generic field
tilt.

## Preview

```text
outputs/figures/digital_twin/stage8c3_component_route_inspection.png
```

Legacy C3R.3 figures still exist, but the route-inspection view is the required
architecture diagnostic.

## Governance

The model remains free-space optical/fluence diagnostic only (`n = 1.0`,
`final_export_allowed=False`). No material/interface, writing trajectory, 3D
visualisation, GUI work, plasma, nonlinear propagation, thermal physics, dose
accumulation, calibrated material-response prediction, or Stage 8D work has been
added.
