# STAGE6C_SUMMARY.md — Lab Realism Viz Fixes (Not Blocked on SLM2 Physics)

**Date:** 2026-06-16  
**Branch:** master  
**Commits:** 2 gated commits (c74b25f, 6735f5b)

---

## What was done

### Task A — Charge-measurement helpers (`vbb_study/viz_fields.py` + `vbb_train_viz.py`)

**`vbb_study/viz_fields.py`** — two new public symbols added:

| Symbol | Description |
|---|---|
| `_phase_winding(field_2d, grid, sample_radius_m, n_phi=256)` | Closed-loop incremental phase accumulation (ported from `tests/test_physics_validation.py` A3). Returns winding in turns as a float. Robust to phase wrapping. |
| `measured_charge_label(field_2d, grid, sample_radius_m, *, design_ell, conjugate_mode=None, n_phi=256)` | Wraps `_phase_winding`. Returns a human-readable string: "measured winding = X.XX (design ℓ=N; ✓ charge preserved)" if winding ≈ design_ell within 0.15 turns, or "... charge stripped; SLM2 conjugate_mode='full' strips the helical phase" otherwise. MEASURED from field, not hardcoded. |

**`vbb_study/vbb_train_viz.py`** — `plot_train_visualiser` gains `charge_label: str | None = None` keyword. When supplied, it is appended to the figure suptitle and the `save_figure` caption string.

---

### Task B — NB02: Physical route hero figures + charge label

`notebooks/lab_realism/02_physical_axicon_route.ipynb`:

1. **Cell `a65fb4fd`**: Added `nb02_results = {}` dict populated during the main run loop (`nb02_results[(regime, label)] = result`). Enables downstream charge measurement without re-running cases.

2. **Cell `dc6a5a29`**: Computes `measured_charge_label` from `nb02_results[("general", "lab")]` surface field. Passes label to `plot_train_visualiser(method='physical')`.

3. **New markdown cell** (`8f72cd06`): Documents F-A3p finding and that charge labels are measured.

4. **New hero code cell** (`e0781eb1`): Runs balanced-preset (N=1024, ds=2) lab config for both regimes. Saves:
   - `nb02_physical_{regime}_hero_linked_field_views.png` — 4-panel figure
   - `nb02_physical_{regime}_azimuthal_order.png` — ring + power spectrum showing m=0 dominance
   
   Charge label in every caption; preset/N/device_downsample stated per Task G.

---

### Task C — NB03: Charge label + comparison markdown

`notebooks/lab_realism/03_holographic_vs_physical_axicon.ipynb`:

1. **New markdown cell** (`727cb17d`): Before the comparison table — explains F-A3p finding, tabulates expected vs measured winding per route, links to `measured_charge_label`.

2. **Cell `52d24e28`**: Runs two fast cases (holographic general/lab + physical general/lab) to get surface fields. Passes measured charge labels to both `plot_train_visualiser` calls.

---

### Task D — NB04: `out_fig` + pupil-fill schematic + carrier-cone diagram

`notebooks/lab_realism/04_objective_pupil_and_first_order_filtering.ipynb`:

1. **Cell `21ebc968`**: Added `out_fig = PATHS["figures"] / "stage_c"` with `mkdir`.

2. **New pupil-fill schematic cell** (`2a729793`): Plots Gaussian beam 1/e² footprint over objective pupil circle for both regimes. Fills 19.8% clipped fraction visualised. Saved as `nb04_pupil_fill_schematic.png`.

3. **New carrier-cone diagram cell** (`c99c8bc3`): Sweeps blaze period (12/20/32 px) and plots carrier spatial frequency vs axicon-cone ring frequency per regime. Annotates ✓/✗ achievability markers. Saves `nb04_carrier_cone_diagram.png`.

---

### Task E — NB05: All 8 cases + charge labels

`notebooks/lab_realism/05_through_sample_interface.ipynb`:

**Cell `24b66f5a`**: Replaced 2-case loop with all-8-cases iteration over `sorted(results.keys())`. Each call to `plot_sample_result_comparison` receives `measured_charge_label` from the beam's surface field, plus preset/N/device_downsample in title (Task G).

---

### Task F — NB06: Gamma fix + holographic phase hero + physical charge labels

`notebooks/lab_realism/06_full_source_to_sample_journey.ipynb`:

1. **Cell `36fa32d8`** (`_shared_linear_display` + `plot_journey_grid`):
   - `_shared_linear_display`: replaced linear clip with `vbb_style.display_scale(normalised, gamma=0.45)`. Sidelobe and ring visibility improved.
   - `plot_journey_grid`: added `*, charge_label=None` kwarg. If supplied: appended to suptitle and caption string.
   - Colourbar label updated: "shared gamma-corrected (γ=0.45) XZ intensity".
   - **Preserved unchanged**: `_is_first_order_impossible`, `_stamp_invalid`, NOT ACHIEVABLE blanking logic, OUT OF VALIDITY stamping, shared_vmax computation.

2. **Cell `7aba1b29`**: Imports `measured_charge_label` at cell top. After each `(regime, method)` block, tries to get `surface_field` and `design` from `lab_journey.air_result`. If accessible, calls `measured_charge_label` and passes `charge_label` to `plot_journey_grid`. Graceful fallback (`charge_label=None`) if `air_result` doesn't carry `surface_field`.

3. **New holographic phase hero cell** (`5bc042a8`): After main loop, iterates holographic lab corrected journeys, accesses `air_result.surface_field`, calls `complex_field_image` to render domain-coloured phase. Saves `nb06_holographic_{regime}_phase_hero.png`. Prints measured charge label confirming winding ≈ 3.

---

### Task G — Preset/N/device_downsample in every caption

All figures added in Stage 6C carry preset/N/device_downsample in their caption or title. Geometry-only figures (NB04) note "geometry only" instead of N/ds since no propagation is run.

---

## Constraints respected

| Constraint | Status |
|---|---|
| `slm2_conjugate_mode` not changed anywhere | ✓ |
| No second SLM2 mode added | ✓ |
| NB06 validity-stamping logic preserved | ✓ `_stamp_invalid`, `_is_first_order_impossible`, NOT ACHIEVABLE pattern — untouched |
| Engine (`bessel_twin_core.py`) untouched | ✓ |
| Charge labels MEASURED from field, not hardcoded | ✓ via `measured_charge_label` → `_phase_winding` |
| Both locks green | ✓ fast: 9/9, prod: see below |
| Viz tests green | ✓ 9/9 |

---

## Lock results

| Test suite | Result |
|---|---|
| `tests/test_viz_fields.py` (9 tests) | **9/9 PASS** |
| `tests/test_characterisation_lock.py` (fast, 9 tests) | **9/9 PASS** |
| `tests/test_characterisation_lock_prod.py` (paper, 9 tests) | pending (running) |

---

## Commits

| Hash | Description |
|---|---|
| `c74b25f` | Stage 6C-A: add _phase_winding + measured_charge_label helpers |
| `6735f5b` | Stage 6C-B/F: lab_realism viz fixes (NB02-06) |
| (this) | Stage 6C-G: STAGE6C_SUMMARY.md + VIZ_AUDIT.md update |

---

## What remains

- NB04-A (capability-slice categorical colourmap) — still NOTED.
- NB02-B (scalar/02 intensity-only plane montage) — still NOTED.
- `cmocean` install still blocked by network; matplotlib `twilight` fallback used.
