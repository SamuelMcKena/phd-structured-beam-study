# Stage 8C.3R.4 Component-Owned Route Architecture

Stage 8C.3R.4 corrects the C3R.3 route-aware implementation. The route is no
longer treated as generic field perturbations injected at arbitrary planes. It
is now an ordered component/segment chain:

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

Each represented component or segment declares:

- `component_id`
- `component_type`
- nominal z position
- distance from previous component
- `distance_to_next_element_mm`
- enabled state
- physical pose: decentre, axial offset, tip, roll
- component-specific parameters
- clear aperture
- active / warning-only / future status
- whether the current engine represents it
- whether a physical model is available
- supported misalignment modes
- downstream elements affected

## Active Represented Components

The physical-axicon route currently executes only free-space scalar optics:

- source field generation
- input aperture at the conditioning plane
- free-space propagation to the physical axicon
- thin physical axicon phase plus clear aperture
- free-space propagation to the downstream/reference plane

Supported local component errors include input-aperture decentre/radius,
physical-axicon lateral decentre, physical-axicon axial offset through adjacent
segment distances, physical-axicon clear aperture, and the scalar axicon cone
phase parameter.

Physical axicon mechanical tilt is not silently converted into field tilt. It
remains `future_not_implemented` unless a documented thin-element/paraxial
approximation is added.

## Boundary Conditions

Field-state controls remain available only as labelled boundary conditions:

```text
field_tilt_x_mrad / field_tilt_y_mrad at field_tilt_location
beam_decentre_x_um / beam_decentre_y_um at beam_decentre_location
```

Every such record declares the boundary plane, physical approximation, which
upstream hardware error it could emulate, and which downstream components consume
it. These controls are not generic component misalignment models.

## Unsupported Components

The following remain explicitly warning-only or future-stage in this physical
axicon engine:

- steering mirror tilt without a represented reflection plane
- SLM1 / SLM2 physical route
- 4F Fourier filter plane
- relay lens plane
- objective pupil clipping/Zernike model
- objective physics
- material/interface/dose/nonlinear/thermal/calibrated response

## Route Inspection

The pipeline now records a route-inspection row for every executed stage:

- component name/type
- nominal location
- actual pose error
- incoming/outgoing field metrics
- energy before/after
- centroid before/after
- angle before/after
- aperture overlap where relevant
- downstream consequences
- model status and warnings

Preview:

```text
outputs/figures/digital_twin/stage8c3_component_route_inspection.png
```

The model remains diagnostic free-space optical/fluence only:

```text
n = 1.0
final_export_allowed = False
no material response
no Stage 8D
```
