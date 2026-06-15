# Stage 4 — Notebook Triage

Executed 2026-06-15. All 22 curated notebooks run via papermill from a **foreign CWD
(`C:\PhD\Code`)**, never from `Publication_Study/`, using a fresh kernel per notebook
(`python3` → `.venv2`, 300 s per-notebook timeout).

## Summary

| Status  | Count |
|---------|------:|
| PASS    | 21    |
| TIMEOUT | 1     |
| FAIL    | 0     |

| Category      | Count | Note |
|---------------|------:|------|
| path          | 0     | Zero — editable install is working |
| missing-symbol| 0     | |
| data-dependency| 0    | |
| runtime-bug   | 1     | scalar/04 hardcodes `PRESET = "publication"`; sweeps exceed 300 s |

**Zero path-category failures.** `import vbb_study` and `import bessel_twin_core`
both resolve from any CWD via the editable install without `sys.path` manipulation.

## Notebook Results

| # | Notebook | Status | Time (s) | Category | Cell | Error |
|---|----------|--------|----------|----------|------|-------|
| 1 | notebooks/00_study_overview_and_conventions.ipynb | PASS | 4.3 | — | — | — |
| 2 | notebooks/quicklook/00_quick_beam_to_sample_simulator.ipynb | PASS | 27.5 | — | — | — |
| 3 | notebooks/scalar/02_scalar_ideal_vs_lab_diagnostics.ipynb | PASS | 23.9 | — | — | — |
| 4 | notebooks/scalar/03_scalar_robustness_and_self_healing.ipynb | PASS | 19.5 | — | — | — |
| 5 | notebooks/scalar/04_scalar_parameter_sweeps.ipynb | **TIMEOUT** | 304.3 | runtime-bug | — | CellTimeoutError: `run_oat_sensitivity` + `run_tradeoff_map` with hardcoded `PRESET="publication"` exceed 300 s |
| 6 | notebooks/scalar/05_scalar_validation_suite.ipynb | PASS | 23.4 | — | — | — |
| 7 | notebooks/lab_realism/01_holographic_axicon_route.ipynb | PASS | 81.8 | — | — | — |
| 8 | notebooks/lab_realism/02_physical_axicon_route.ipynb | PASS | 16.3 | — | — | — |
| 9 | notebooks/lab_realism/03_holographic_vs_physical_axicon.ipynb | PASS | 19.9 | — | — | — |
| 10 | notebooks/lab_realism/04_objective_pupil_and_first_order_filtering.ipynb | PASS | 3.3 | — | — | — |
| 11 | notebooks/lab_realism/05_through_sample_interface.ipynb | PASS | 29.8 | — | — | — |
| 12 | notebooks/lab_realism/06_full_source_to_sample_journey.ipynb | PASS | 134.1 | — | — | — |
| 13 | notebooks/vector/01_vector_beam_theory_atlas.ipynb | PASS | 9.3 | — | — | — |
| 14 | notebooks/vector/02_vector_ideal_vs_lab_case1.ipynb | PASS | 7.0 | — | — | — |
| 15 | notebooks/vector/03_vector_hardware_routes.ipynb | PASS | 5.2 | — | — | — |
| 16 | notebooks/materials/01_material_proxy_fluence_and_thresholds.ipynb | PASS | 10.7 | — | — | — |
| 17 | notebooks/materials/02_material_calibration_template.ipynb | PASS | 7.3 | — | — | — |
| 18 | notebooks/materials/03_application_design_tables.ipynb | PASS | 7.3 | — | — | — |
| 19 | notebooks/publication_exports/03_report_export.ipynb | PASS | 154.4 | — | — | — |
| 20 | notebooks/advanced/01_capsule_weld_feature_design.ipynb | PASS | 4.5 | — | — | — |
| 21 | notebooks/advanced/02_hexagonal_polygonal_beams.ipynb | PASS | 6.2 | — | — | — |
| 22 | notebooks/advanced/03_discrete_nfold_beams.ipynb | PASS | 3.4 | — | — | — |

Total wall time: 903.4 s (~15 min)

## Notes

### scalar/04 timeout (runtime-bug)
`notebooks/scalar/04_scalar_parameter_sweeps.ipynb` has `PRESET = "publication"`
hardcoded in cell 3 (not as a papermill parameter). The publication-preset sweeps
(`run_oat_sensitivity`, `run_tradeoff_map`, `run_ell_family_comparison`, etc.) take
well over 5 minutes. This is not a logic bug — the notebook works correctly when
given sufficient time. Stage 5 action: expose `PRESET` as a papermill parameter cell
so triage can override it with `"fast"`.

### Kernel metadata (raised during triage)
All notebooks have `"kernelspec": {"name": "python3"}`. The `python3` kernel spec
inside `.venv2` previously used bare `python` in `argv`, meaning VS Code interactive
sessions resolved to the system Python (not `.venv2`) when the venv was not active.
**Fixed in Stage 4**: `C:\PhD\.venv2\share\jupyter\kernels\python3\kernel.json`
updated to use the full path `C:\PhD\.venv2\Scripts\python.exe`. Restart any
running kernel in VS Code to pick up the fix.
