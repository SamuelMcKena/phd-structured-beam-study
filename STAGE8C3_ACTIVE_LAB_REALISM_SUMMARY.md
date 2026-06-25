# Stage 8C.3 Active Lab-Realism Coupling Summary

Starting point for Stage 8C.3D repair: committed Stage 8C.3C baseline `0a986b56`.

Stage 8C.3 adds active, diagnostic lab-realism coupling without changing locked propagation physics. The cockpit now classifies every lab-realism control, declares affected outputs, applies physically feasible perturbations to canonical optical stacks, reports degradation metrics, and flags enabled controls that remain warning-only or future-stage.

## Created

- `vbb_study/digital_twin/lab_perturbations.py`
- `vbb_study/digital_twin/active_realism_metrics.py`
- `docs/32_active_lab_realism_coupling.md`
- `tests/test_stage8c3_lab_perturbations.py`
- `tests/test_stage8c3_active_control_classification.py`
- `tests/test_stage8c3_misalignment_metrics.py`
- `tests/test_stage8c3_notebook_active_controls.py`
- `tests/test_stage8c3_governance.py`

## Modified

- `vbb_study/digital_twin/lab_realism_controls.py`
- `vbb_study/digital_twin/cockpit_dashboard.py`
- `vbb_study/digital_twin/__init__.py`
- `notebooks/digital_twin/00_full_beam_to_write_cockpit_MVP.ipynb`

## Active Controls

`physics_active` is now reserved for future engine-level controls consumed before propagation. Stage 8C.3 stack/image perturbations are `diagnostic_active` and report both intended `physical_placement` and actual `implementation_stage`; post-engine stack transforms are not called `physics_active`.

`diagnostic_active` optical controls include beam decentre, beam tilt, beam ellipticity, input/relay/pupil clipping, SLM active-area clipping, seeded SLM dead pixels, seeded SLM phase noise, SLM pixel/fill/quantisation proxies, SLM/vortex/axicon centre offsets, physical axicon apex offset/tilt proxy, relay decentre/tilt/magnification/aperture, zero-order leakage, unwanted-order leakage, and low-order Zernike aberrations.

`energy_active` controls include pulse energy before optics, pulse-energy jitter, repetition-rate error, pulse-duration error, transmission chain controls, selected first-order fraction, interface reflection/Fresnel estimate, and average-power-limit warnings.

`geometry_active` controls include focus offset, focus-depth error, sample tilt/surface offset metadata, refractive-index error bookkeeping, sample-thickness limit warning geometry, and exposure-plan controls such as scan speed.

`diagnostic_active` controls include selected z, central ROI, display scaling, camera crop/noise display metadata, and engine/field-acquisition diagnostics.

## Report-Only / Future

`warning_only` controls include first-order field-active Fourier decentre/clipping, physical axicon angle error, physical axicon apex defect, SLM/mask rotation resampling, pointing jitter, stage-position jitter, and focus drift.

`metadata_only` controls include scalar-mode polarisation and route/source descriptors that are recorded but are not active scalar primitives.

`future_not_implemented` controls remain disabled: material response, threshold proxy, dose accumulation, nonlinear proxy, thermal proxy, microscope proxy, and calibrated prediction.

## Poynting Boundary

In the current scalar optical model, direct Poynting-vector editing is not a primitive active input. Beam direction changes are represented by input beam tilt / angular-spectrum phase ramp. Polarisation and true vectorial energy-flow effects require a vector engine and are metadata-only unless the vector path is active.

## Preview

Required diagnostic preview:

```text
outputs/figures/digital_twin/stage8c3_baseline_vs_perturbed_preview.png
```

The comparison uses shared baseline/perturbed colour scales and includes metric deltas. It is optical fluence only; no material response is predicted.

Stage 8C.3B visual rescue preview:

```text
outputs/figures/digital_twin/stage8c3_misalignment_sensitivity_sweep_preview.png
```

This polished sensitivity figure shows aligned, mild, severe, and severe-minus-baseline columns with XY fluence, XZ fluence, central ROI/core fluence, and metric cards. It uses shared colour scales for baseline/mild/severe panels and embeds the claim boundary directly on the figure.

Scenario coverage:

- Scenario A: SLM/vortex/axicon centre offset.
- Scenario B: beam tilt / pointing.
- Scenario C: pupil clipping / decentre.
- Scenario D: zero-order leakage.

The severe case is required to degrade at least one scenario metric more than the mild case. The displayed metric cards include centroid shift, peak shift, peak fluence change, azimuthal uniformity, ring circularity, core fill fraction, symmetry score, pupil clipped power, and baseline similarity.

Stage 8C.3C genuine degradation preview:

```text
outputs/figures/digital_twin/stage8c3c_genuine_degradation_sweep_preview.png
```

Stage 8C.3C separates translation from deformation. The metric set now includes `unregistered_similarity_score`, `registered_similarity_score`, `centroid_shift_um`, `translation_dominated_boolean`, and `residual_shape_deformation_score`. Co-shifted vortex+axicon is kept only as a translation diagnostic; relative vortex/axicon misregistration is the primary Scenario A. The replacement sweep covers relative vortex/axicon offsets, beam decentre with finite SLM aperture, beam tilt with finite pupil, objective pupil decentre/clipping, coma/astigmatism/defocus/spherical aberration, zero-order leakage, and a combined diagnostic stress test.

Stage 8C.3D conservation/axis diagnostic preview:

```text
outputs/figures/digital_twin/stage8c3d_conservation_axis_diagnostics_preview.png
```

Stage 8C.3D does not claim a full pre-propagation component beamline engine. It repairs the diagnostic stress visualizer so passive clipping/loss controls reduce the perturbed pulse energy instead of being silently restored by per-plane fluence scaling. The audit reports input pulse energy, energy before perturbation, energy after passive loss, transmitted fraction, peak-to-total-energy ratio, and renormalisation factor applied.

Post-engine spatial clipping is disabled in the headline preview because multiplying upstream aperture masks through already-propagated z-planes produced harsh straight XZ cutoffs that were diagnostic artifacts, not true propagated physics. Passive aperture/pupil/SLM active-area losses are shown as throughput loss unless artifact-risk post-engine clipping is explicitly enabled for investigation.

The saved C3D headline preview now uses low-order aberrations as the visual scenario so the main field panels show smooth diagnostic deformation rather than clipping-heavy stress artifacts. Combined stress and passive clipping cases remain in the scenario audit table and throughput ledger only until a real upstream aperture/pupil propagation path is available.

The C3D metric set also adds commanded-axis versus actual-axis diagnostics: commanded x/y, fitted ring centre, brightest-point offset, fitted beam-axis surface intercept, target-plane axis offset, steering angle, field-of-view margin, out-of-frame fraction, and crop-edge energy fraction. Co-shifted vortex+axicon remains a translation diagnostic; genuine degradation scenarios remain separated from pure translation by registered similarity and residual shape deformation.

Stage 8C.3R.4 component-owned physical-axicon route scaffold:

```text
outputs/figures/digital_twin/stage8c3_component_route_inspection.png
```

Stage 8C.3R.4 corrects the too-narrow route-aware interpretation. The physical-axicon route is now an ordered component/segment scaffold: source complex field, source boundary condition, input aperture, source-to-axicon propagation, axicon input boundary, physical axicon, post-axicon boundary, post-axicon free-space segment, post-axicon diagnostic boundary, post-axicon-to-reference segment, and free-space reference plane. Supported errors are attached to represented components and applied in their local planes before downstream propagation continues.

Field-state controls still exist only as labelled boundary conditions at named planes. They declare the physical approximation, the upstream hardware error they could emulate, and the downstream components that consume them. They are not generic representations of arbitrary component misalignment.

Active physics is limited to source complex field, input aperture, free-space propagation segments, thin scalar physical axicon phase and clear aperture, and free-space reference-plane diagnostics. Steering mirrors, SLM1/SLM2, 4F Fourier filtering, relay optics, pupil/objective optics, and mechanical axicon tilt remain warning-only or future-stage.

The route-inspection table records component name/type, nominal location, pose error, incoming/outgoing field metrics, energy before/after, centroid before/after, angle before/after, aperture overlap, actual segment distance, transform-applied boolean, downstream consequences, model status, and warnings.

## Governance

Core optical physics, scalar/vector propagation equations, axicon physics, characterization locks, validation baselines, and production baselines were not changed. No material-response claim is introduced. Stage 8C.3 remains a diagnostic optical/fluence cockpit pass only, with actual post-engine diagnostic implementation stage reported separately from intended physical placement.
