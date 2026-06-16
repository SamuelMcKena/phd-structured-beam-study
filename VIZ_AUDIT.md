# VIZ_AUDIT.md — Stage 6A visualisation audit

Every entry below names the notebook, the specific cell or function producing
the plot, the defect, and the current status (REPLACED or DEFERRED).

---

## Replaced (holographic route — Stage 6A)

### NB01-A · `notebooks/lab_realism/01_holographic_axicon_route.ipynb` — `plot_train_visualiser` at fast preset

**Defect:** Train visualiser runs at `PRESET = "fast"` (triage-quality grid,
N=512, device_downsample=4, dx=0.25 µm).  Phase is shown as a separate
`twilight`-cmap row without amplitude weighting: in the dark vortex core
(|U|≈0) the phase is noise-dominated and the twilight colormap shows random
colours, making the ℓ=3 winding invisible.  All five sub-panels show
amplitude and flat phase; no domain-coloured combined image.

**Status: REPLACED** — cell `vbb_train_viz.plot_train_visualiser` at
fast preset replaced by `linked_field_views` at paper preset (N=2048,
device_downsample=1, 181 z-planes).  The new panel (a) shows |U|² at peak z,
(b) shows domain-coloured phase (HUE=phase via colorcet CET_C2, VALUE=|U|/peak)
at the surface plane, making the ℓ=3 winding visibly wrap 3 times, (c) shows
the longitudinal x–z slice, and (d) shows the radial profile with the analytic
|J₃(kᵣr)|² overlay and measured ring radius marked.  Figures saved for both
`general` and `limits` holographic regimes.

---

### NB01-B · `notebooks/lab_realism/01_holographic_axicon_route.ipynb` — summary pivot table only (no phase figure)

**Defect:** The notebook's main output (cells 6–7) is a numeric summary
DataFrame and a pivot table of scalar metrics only.  There is no field-view
figure showing the intensity distribution or phase structure of the holographic
beam at publication preset.  The reader cannot verify the ℓ=3 winding from
the notebook's saved figures.

**Status: REPLACED** — new cell added (Stage 6A) producing `linked_field_views`
at paper preset for general/holographic and limits/holographic cases.

---

## Resolved in Stage 6C (physical route — labelled, charge=0 documented)

### NB02-A · `notebooks/lab_realism/02_physical_axicon_route.ipynb` — all figures

**Status: RESOLVED** — Stage 6C adds:
- `nb02_results` dict storing full result objects for charge measurement.
- `measured_charge_label` on `plot_train_visualiser(method='physical')` call.
- Hero `linked_field_views` at `balanced` preset (N=1024, ds=2) for both
  regimes, with charge label and preset/N/ds in caption.
- `azimuthal_order_panel` for both regimes, showing m=0 dominance (charge=0).
All figures carry the F-A3p label: "measured winding ≈ 0 (design ℓ=3; SLM2
conjugate_mode='full' strips the helical phase)".

---

### NB03-A · `notebooks/lab_realism/03_holographic_vs_physical_axicon.ipynb` — method comparison

**Status: RESOLVED** — Stage 6C adds:
- Markdown cell explaining holographic vs physical charge discrepancy (F-A3p).
- Measured charge labels on both `plot_train_visualiser` calls (computed from
  fast run, not hardcoded).

---

### NB05-A · `notebooks/lab_realism/05_through_sample_interface.ipynb` — `plot_sample_result_comparison`

**Status: RESOLVED** — Stage 6C expands from 2 to all 8 computed cases, with
measured charge label in every figure title.

---

### NB06-A · `notebooks/lab_realism/06_full_source_to_sample_journey.ipynb` — `_shared_linear_display`

**Status: RESOLVED** — Stage 6C:
- Replaces linear clip with `vbb_style.display_scale(gamma=0.45)`.
- Adds `charge_label` kwarg to `plot_journey_grid` (suptitle + caption).
- Computes charge from air_result surface_field in main loop.
- Adds holographic phase hero cell after main loop.
Validity-stamping logic (`_stamp_invalid`, `_is_first_order_impossible`,
NOT ACHIEVABLE blanking) preserved unchanged.

---

## Deferred (physical route — pending SLM2 decision)

### NB02-A · `notebooks/lab_realism/02_physical_axicon_route.ipynb` — all figures

**Defect:** Physical route carries topological charge 0 (Finding F-A3p: the
`slm2_conjugate_mode='full'` strips the SLM1 vortex phase before the axicon;
winding=0.000, not ell=3).  All intensity and phase plots in this notebook
describe a charge-0 beam, not the intended ℓ=3 design.  Running at `PRESET =
"fast"` further degrades resolution.  Showing these figures in publication
context without explicit labelling would misrepresent the physical-route beam.

**Status: DEFERRED** — pending SLM2 decision.  If shown, figures MUST carry
the label "current full-conjugate config, charge=0, pending SLM2 decision".
Do NOT present these as ℓ=3 holographic-equivalent figures.

---

### NB03-A · `notebooks/lab_realism/03_holographic_vs_physical_axicon.ipynb` — method comparison

**Defect:** Method-comparison `plot_train_visualiser` at fast preset compares
holographic (charge=3) and physical (charge=0) beams without labelling the
charge discrepancy.  The comparison is misleading: it implies the two routes
produce equivalent beams when they do not (Finding F-A3p).

**Status: DEFERRED** — holographic side is valid; physical side blocked by
SLM2 charge-0 issue.  Do not present the comparison as holographic vs physical
for equivalent ℓ=3 beams.

---

### NB06-A · `notebooks/lab_realism/06_full_source_to_sample_journey.ipynb` — `_shared_linear_display`

**Defect:** Function `_shared_linear_display(values, vmax)` applies linear
clipping (`clip(arr/vmax, 0, 1)`) without gamma correction.  For Bessel beams,
this suppresses sidelobe and ring visibility: the bright core or ring
saturates, and the weaker outer rings are invisible.  The notebook mixes
holographic and physical routes without charge labels.

**Status: DEFERRED** — physical route blocked by SLM2 charge-0 issue.  Once
the physical-route charge configuration is resolved, replace
`_shared_linear_display` with `vbb_style.display_scale(..., gamma=0.45)` and
add domain-coloured phase panels via `complex_field_image`.

---

### NB05-A · `notebooks/lab_realism/05_through_sample_interface.ipynb` — `plot_sample_result_comparison`

**Defect:** Intensity-only field comparison before/after interface correction.
Phase structure not shown; does not illustrate whether ℓ=3 winding survives
propagation through the glass–sample interface.

**Status: DEFERRED** — physical route included in this notebook and blocked by
SLM2 charge-0 issue.  Add domain-coloured phase panels when physical-route
charge configuration is finalised.

---

## Noted (non-holographic, low priority)

### NB04-A · `notebooks/scalar/04_scalar_parameter_sweeps.ipynb` — capability-slice heatmap

**Defect:** Sampling-capability slice (cell 21) encodes the discrete QA state
`{fail=0, marginal=1, pass=2}` as a continuous `viridis` heatmap.  Continuous
colourmap on ordinal data implies a false gradient between states.  A
diverging-discrete or categorical colourmap (e.g. two-colour fail/pass) would
be clearer.

**Status: NOTED** — not replaced in Stage 6A (no holographic phase information
involved).  Improve in a later style pass.

---

### NB02-B · `notebooks/scalar/02_scalar_ideal_vs_lab_diagnostics.ipynb` — `plot_axial_plane_montage`

**Defect:** Plane montage shows intensity-only cross-sections.  For an ℓ=3
vortex beam, the dark core is informative only in context of the phase; without
it the montage cannot distinguish a true vortex from a ring-shaped intensity
artefact.

**Status: NOTED** — this is the scalar diagnostic notebook; it covers both
ℓ=0 and ℓ=3.  For ℓ=3 cases, add a domain-coloured phase panel alongside the
montage in a future scalar phase-audit pass.

---

## Summary table

| ID     | Notebook                                         | Function / cell            | Defect class                              | Status   |
|--------|--------------------------------------------------|----------------------------|-------------------------------------------|----------|
| NB01-A | lab_realism/01_holographic_axicon_route          | plot_train_visualiser       | flat phase (no domain colour), fast grid  | REPLACED |
| NB01-B | lab_realism/01_holographic_axicon_route          | summary pivot only          | no field-view figure at paper preset      | REPLACED |
| NB02-A | lab_realism/02_physical_axicon_route             | all figures                 | charge=0 (F-A3p), unlabelled             | DEFERRED |
| NB03-A | lab_realism/03_holographic_vs_physical_axicon    | method comparison           | charge mismatch unlabelled                | DEFERRED |
| NB05-A | lab_realism/05_through_sample_interface          | plot_sample_result_comparison | intensity-only, physical route blocked  | DEFERRED |
| NB06-A | lab_realism/06_full_source_to_sample_journey     | _shared_linear_display      | no gamma, no phase, physical charge-0     | DEFERRED |
| NB04-A | scalar/04_scalar_parameter_sweeps                | capability-slice heatmap   | categorical data on continuous cmap       | NOTED    |
| NB02-B | scalar/02_scalar_ideal_vs_lab_diagnostics        | plot_axial_plane_montage   | intensity-only, no phase for ℓ=3 cases   | NOTED    |
