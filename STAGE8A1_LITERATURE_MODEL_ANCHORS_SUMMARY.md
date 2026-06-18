# STAGE 8A.1 — Literature and Model Anchors: Summary

**Completed:** 2026-06-18  
**Branch:** master  
**Follows:** Stage 8A commit `63effc9`

---

## Files Created

### Documents

| File | Description |
|---|---|
| `docs/25_literature_model_anchors.md` | Human-readable literature anchor report: categories, engine mapping, what can be borrowed, what must be calibrated, 8 calibration sweeps, next stage |
| `docs/26_literature_anchor_claim_boundaries.md` | 8 explicit claim-boundary rules with quick-reference table |

### CSVs

| File | Rows | Description |
|---|---|---|
| `outputs/csv/digital_twin/literature_model_anchors.csv` | 24 | Main anchor table: 11 categories, paper placeholders, parameter borrowability, claim boundaries |
| `outputs/csv/digital_twin/material_parameter_anchor_candidates.csv` | 23 | Material parameters with safe_use_level; ZnSe-family vs silica separated; calibration flags |
| `outputs/csv/digital_twin/model_equation_anchor_map.csv` | 15 | All required equations mapped to engine layer, model status, literature anchors, calibration requirement |
| `outputs/csv/digital_twin/material_response_calibration_targets.csv` | 16 | Explicit calibration targets: measurement method, parameter informed, promotion pathway |

### Tests

| File | Tests | Result |
|---|---|---|
| `tests/test_stage8a1_literature_anchors.py` | 48 | (run after commit — see below) |

---

## What Literature Categories Are Now Represented

All 11 required categories are present in the anchor CSV:

| Category | Rows | Coverage |
|---|---|---|
| `bessel_beam_material_modification` | 3 | Bhuyan / Courvoisier / Stoian group placeholders |
| `fused_silica_bessel_processing` | 2 | Mathis / Bhuyan single-shot; etch enhancement |
| `ultrafast_waveguide_inscription` | 3 | Davis 1996; Thomson review; Osellame review |
| `crznse_or_zns_family_waveguide_inscription` | 2 | Lancaster / Bernier group placeholders |
| `heat_accumulation_scan_speed` | 2 | Eaton / Thomson thermal regime papers |
| `incubation_multi_pulse_threshold` | 2 | Jee 1988; Lenzner dielectrics |
| `nonlinear_propagation_filamentation` | 2 | Couairon review; ZnSe nonlinear anchor |
| `surface_ablation_modelling` | 2 | Liu 1982 spot method; Stuart dielectrics |
| `refractive_index_reconstruction` | 2 | QPM methodology; ZnSe-family QPM |
| `waveguide_mode_modelling` | 2 | Marcuse theory; FD-BPM numerical method |
| `microscope_or_etch_observable` | 2 | DIC microscopy; HF etch fused silica |

---

## What Remains Placeholder / Needs Lookup

All 22 paper-based anchors are marked `source_status = "needs_literature_lookup"`.

**Before any anchor parameter value is used in a simulation output:**
1. Verify the exact paper citation (author, journal, year, DOI) from Zotero/PDF.
2. Extract the specific parameter values and experimental conditions.
3. Update `source_status` to `"confirmed_from_pdf"` or `"confirmed_textbook"`.
4. Update the parameter value in `material_parameter_anchor_candidates.csv`.

Two anchors are already confirmed:
- A021 (Marcuse textbook): `source_status = "confirmed_textbook"`
- A022 (FD-BPM method): `source_status = "confirmed_method"`

---

## Claim Boundary Rules Implemented

8 explicit rules in `docs/26_literature_anchor_claim_boundaries.md`:

| Rule | Summary |
|---|---|
| Rule 1 | Fused silica does NOT calibrate Cr:ZnSe |
| Rule 2 | Gaussian-focus papers do NOT calibrate Bessel/vortex-Bessel writing |
| Rule 3 | Surface ablation thresholds do NOT apply to internal modification |
| Rule 4 | Thresholded fluence/dose model is NOT damage prediction |
| Rule 5 | Heat accumulation models require material thermal constants and context |
| Rule 6 | Nonlinear propagation models require material-specific coefficients |
| Rule 7 | Cr:ZnSe/ZnSe-family anchors are priors, not calibration |
| Rule 8 | Literature anchors initialise; lab calibration earns prediction status |

---

## Core Engine Physics Changed?

**No.** Stage 8A.1 is documentation and data-architecture only. No Python source files modified.

---

## Calibration Targets Defined

16 calibration targets in `material_response_calibration_targets.csv`:

| Priority | Target | Informs |
|---|---|---|
| 1 | Surface ablation threshold (Liu method) | F_th_surface; Engine 2 upper bound |
| 2 | Internal modification threshold | F_th_internal; primary Engine 2 calibration |
| 3 | Incubation S and F_th(1) via static sweep | Jee-Becker model; full threshold proxy upgrade |
| 4 | Track width and depth vs energy | Engine 3 geometry calibration |
| 5 | Feature length (Bessel zone observable) | Engine 1 validation |
| 6 | Track continuity / minimum scan speed | N_spot and dose proxy calibration |
| 7 | Refractive-index change Δn | Waveguide mode solver; Engine 3 calibration |
| 8 | Waveguide mode size and propagation loss | Engine 3 validation; application metric |
| + | Crack onset; etch response; microscope contrast; side-lobe marks | Writing window; observable calibration |

---

## Recommended Next Stage

The literature anchor architecture is now stable.  Two paths are recommended:

**Path A (parallel):**
> **Stage 8A.2** — Fill literature anchors from actual PDFs / Zotero / manual extraction  
> Replace all `needs_literature_lookup` rows with verified citations and parameter values.

**Path B (can proceed now):**
> **Stage 8B** — Optical cockpit + energy accounting  
> Engine 2 energy accounting formulas do not depend on literature material parameters.  
> Stage 8B can proceed in parallel with 8A.2.

**Recommended:** Proceed to Stage 8B.  Run Stage 8A.2 as a parallel documentation task
once actual PDFs are accessible.
