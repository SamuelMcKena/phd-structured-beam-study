# Stage 8C.3 Active Lab-Realism Coupling Summary

Starting point: accepted commit `cb42bc7`.

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

`physics_active` controls include beam decentre, beam tilt, beam ellipticity, input/relay/pupil clipping, SLM active-area clipping, seeded SLM dead pixels, seeded SLM phase noise, SLM pixel/fill/quantisation proxies, SLM/vortex/axicon centre offsets, physical axicon apex offset/tilt proxy, relay decentre/tilt/magnification/aperture, zero-order leakage, unwanted-order leakage, and low-order Zernike aberrations.

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

## Governance

Core optical physics, scalar/vector propagation equations, axicon physics, characterization locks, validation baselines, and production baselines were not changed. No material-response claim is introduced. Stage 8C.3 remains a diagnostic optical/fluence cockpit pass only.
