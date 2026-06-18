# STAGE 8C — SurfaceField Energy-Scaled Optical Cockpit: Summary

**Completed:** 2026-06-18
**Branch:** master
**Follows:** Stage 8A (`63effc9`), Stage 8A.1 (`cf11f1a`), Stage 8B (`19fe4f9`)

---

## 1. Files Created

### Code modules
| File | Description |
|---|---|
| `vbb_study/digital_twin/field_coupling.py` | Canonical optical-field adapters: `OpticalFieldPlane`/`OpticalFieldStack`, validators, array constructors, `SurfaceField`/volume extractors, custom errors |
| `vbb_study/digital_twin/field_fluence.py` | Energy-conserving field→fluence scaling (plane + per-plane stack), integrals, peak-intensity bridge, summary |
| `vbb_study/digital_twin/field_figures.py` | Diagnostic preview figure builder (stamped `final_export_allowed=False`) |

### Updated
| File | Change |
|---|---|
| `vbb_study/digital_twin/__init__.py` | Exports all Stage 8C names; stage status + allowed/forbidden statuses updated |

### Notebook
| File | Description |
|---|---|
| `notebooks/digital_twin/02_surfacefield_energy_scaled_optical_cockpit.ipynb` | 18 cells; runs the real engine, adapts the field, scales to fluence, reports, diagnostic figure, guarded save |

### Documentation
| File | Description |
|---|---|
| `docs/28_surfacefield_energy_scaled_cockpit.md` | 11-section reference (purpose, 8B link, Mode B, conventions, scaling equation, volume-integral caveat, drift, claims, Stage 8D, failure modes) |
| `STAGE8C_SURFACEFIELD_ENERGY_SCALED_COCKPIT_SUMMARY.md` | This file |

### Tests
| File | Tests | Description |
|---|---|---|
| `tests/test_stage8c_field_coupling.py` | adapters, validators, [z,y,x] convention, extractors, real-engine roundtrip |
| `tests/test_stage8c_field_fluence.py` | integrals, energy conservation, drift, peak intensity, summaries |
| `tests/test_stage8c_notebook_wiring.py` | controls, caveats, save guard, demo gating, no placeholder beam |
| `tests/test_stage8c_governance.py` | docs, claim boundary, forbidden-phrase negation check, figure metadata, lock-sensitive files |

### Optional example outputs
- `outputs/csv/digital_twin/stage8c_field_fluence_summary_example.csv` (committable)
- `outputs/figures/digital_twin/stage8c_surfacefield_energy_scaled_preview.png` (gitignored; local only)

---

## 2. Code Modules

### `field_coupling.py`
- Frozen dataclasses `OpticalFieldPlane` (`intensity[y,x]`) and `OpticalFieldStack` (`intensity[z,y,x]`).
- Validators `validate_intensity_plane` / `validate_intensity_stack` (finite, non-negative, correct ndim).
- Constructors `plane_from_arrays` / `stack_from_arrays` (shape/coordinate/monotonicity checks).
- Extractors `extract_plane_from_surfacefield` (handles repo `SurfaceField`: `|Ex|²+|Ey|²+|Ez|²`, grid in metres → µm) and `extract_stack_from_surfacefield` (handles `propagate_volume` dict / nested result).
- Errors `MissingOpticalFieldError`, `InvalidOpticalFieldError`, `UnsupportedSurfaceFieldError`.
- Source-status governance: `synthetic_placeholder` flagged non-governed; `unit_test_fixture` allowed for tests.

### `field_fluence.py`
- `FluencePlaneResult`, `FluenceStackResult` (frozen).
- `scale_plane_to_fluence` — energy-conserving `F = E·I/(ΣI·dA)`, reuses Stage 8B `scale_intensity_to_fluence_j_cm2`.
- `scale_stack_to_fluence` — per-plane transverse-energy normalisation; reports `propagation_energy_drift_fraction` from raw transverse integrals.
- `transverse_integral_um2`, `integrated_energy_uJ_from_fluence`, `peak_intensity_from_fluence_result` (Stage 8B `I≈F/τ`), `field_fluence_summary`.
- Status `fluence_prediction`; `final_export_allowed=False`; strong caveats.

### `field_figures.py`
- `plot_stage8c_field_fluence_preview` — XY intensity, XY fluence, XZ fluence, peak-fluence-vs-z, per-plane energy, text/caveat panel.
- PNG metadata stamps `final_export_allowed=False`, `model_status=fluence_prediction`, `figure_status=diagnostic_allowed`.
- Saving with `show_caveats=False` raises `CaveatsRequiredError`. No threshold contours / modification regions / microscopy proxies.

---

## 3. Notebook
- 20 visible controls including `field_source_mode`, `require_real_field`, `allow_synthetic_demo_field` (False by default), `show_caveats`, `save_outputs`.
- Runs `bessel_twin_core.run_case` live for a real field; raises `MissingOpticalFieldError` if no real field and demo disabled.
- Save guard: `ValueError` if `save_outputs=True` and `show_caveats=False`; refuses to save a non-governed (synthetic/demo) field.

---

## 4. Tests Run

```
tests/test_stage8c_field_coupling.py
tests/test_stage8c_field_fluence.py
tests/test_stage8c_notebook_wiring.py
tests/test_stage8c_governance.py
  -> 91 passed, 1 skipped (example PNG gitignored)

Regression:
tests/test_stage8b_*                  -> all pass
tests/test_stage8a_blueprint_docs.py  -> pass
tests/test_stage8a1_literature_anchors.py -> pass
tests/test_characterisation_lock.py   -> 9/9 pass (fast lock green)
```

(Exact counts confirmed in the commit message.)

---

## 5. Pass / Fail Status
**Pass.** All Stage 8C tests green; Stage 8A/8A.1/8B regression green; fast characterisation lock 9/9.

## 6. Existing core optical physics changed?
**No.** Only new files under `vbb_study/digital_twin/` plus an `__init__.py` export update. `bessel_twin_core.py`, propagation/scalar equation modules, and the characterisation lock are untouched (asserted by `test_lock_sensitive_files_not_modified`).

## 7. Material response implemented?
**No.** Stage 8C is optical/fluence only. No thresholds, no dose accumulation, no nonlinear/thermal deposition, no material modification.

## 8. Fake optical fields in production paths?
**No.** Adapters wrap real arrays; the notebook runs the real engine. The only synthetic path is the notebook demo, disabled by default, labelled `unit_test_or_demo_only`, and never saved as a governed output.

## 9. Caveats
- `propagation_energy_drift_fraction` can be large for a real volume because the crop window captures a shrinking fraction of the field away from the Bessel zone; it is a diagnostic of captured transverse power, not physical energy loss.
- Mode B fluence is an **optical** prediction; it is not absorbed energy, dose, or material modification.
- Peak intensity is the approximate `I ≈ F/τ` flat-top estimate (no nonlinear reshaping, plasma, or thermal feedback).
- The notebook is verified by wiring tests, not executed in CI.

## 10. Recommended next stage
```
Stage 8D — 3D beam-to-sample visualiser
```
Stage 8C connects real field outputs successfully, so Stage 8D can build a 3D
visualisation layer over the `OpticalFieldStack` / `FluenceStackResult` with no
new physics and the same `fluence_prediction` claim boundary.
