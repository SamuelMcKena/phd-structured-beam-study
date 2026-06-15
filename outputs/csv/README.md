# CSV output naming

Stage 8.6 adds a publication-readiness gate for figures and output families.
The machine-readable registry lives under `outputs/csv/publication_exports/`:

- `figure_output_registry.csv`
- `figure_visual_quality_audit.csv`

Use `vbb_study.publication.figure_registry.assert_export_allowed(path)` before
any final report or paper export consumes an artifact. Unknown, diagnostic,
legacy, and rejected paths are blocked by default. Compatibility outputs remain
available for traceability, but they are not final-export sources unless the
registry explicitly marks them `final_export_allowed`.

The registry status meanings are:

| status | export meaning |
| --- | --- |
| `final_export_allowed` | Can be used later for final report/paper export with the required caveat/QA labels. |
| `diagnostic_allowed` | Useful for research/debug/review, blocked from final export by default. |
| `exploratory_only` | Development-only, blocked. |
| `legacy_quarantined` | Old or compatibility output retained for traceability, blocked. |
| `rejected` | Misplaced, stale, unknown, or unsafe artifact, blocked. |

## Stage 8.7 quicklook

The quick-look simulator writes diagnostic outputs under:

- `outputs/csv/quicklook/quicklook_metric_summary.csv`
- `outputs/csv/quicklook/quicklook_four_condition_phase_comparison_summary.csv`
- `outputs/csv/quicklook/four_condition_phase_comparison_metrics.csv`
- `outputs/json/quicklook/quicklook_config.json`
- `outputs/figures/quicklook/`

These outputs include `run_id`, `generated_at_utc`,
`source_schema_version`, `computational_mode`, parameter metadata, and caveats.
The notebook default is live plotting with `save_outputs=False`; writing these
files is opt-in. The registry marks the quicklook families as
`diagnostic_allowed`, not `final_export_allowed`. Labelled display interpolation
may be used to reduce pixelated appearance, but it is visual only; rerun
balanced/publication mode for numerical interpretation.

Stage 8.7D quicklook rows include raw-array visual sanity guardrails such as
`centre_intensity_fraction`, `ringness_score`, `xz_structure_score`,
`beam_centre_offset_um`, and first-order/filter sanity fields. A lab-realistic
or material-facing row with `visual_sanity_label=fail` or `blocked` is a
diagnostic failure, not a beam prediction.

Stage 5 vector notebooks write canonical outputs under `outputs/csv/vector/`.
Compatibility copies for older publication-export names are kept under `outputs/csv/publication_study/` where useful.

| Old name | Canonical Stage 5 name |
| --- | --- |
| `vector_atlas_scalar_sas_summary.csv` | `vector/vector_atlas_scalar_sas_summary.csv` and `vector/vector_beam_theory_atlas.csv` |
| `vector_atlas_jones_summary.csv` | `vector/vector_atlas_jones_summary.csv` and `vector/vector_beam_theory_atlas.csv` |
| `stage6_fidelity_ladder_summary.csv` | `vector/vector_ideal_vs_lab_case1_summary.csv` |
| `stage6_fidelity_delta_table.csv` | `vector/stage6_fidelity_delta_table.csv` compatibility diagnostic |
| `stage6_slm_encoded_vector_summary.csv` | `vector/vector_ideal_vs_lab_case1_summary.csv` filtered to current Case 1 |
| `stage6_paper_replica_vector_summary.csv` | `vector/vector_ideal_vs_lab_case1_summary.csv` filtered to paper-replica diagnostics |

## Stage 6 materials

Stage 6 material notebooks write canonical outputs under `outputs/csv/materials/`.
Rows include native material schema metadata: `run_id`, `generated_at_utc`,
`source_schema_version`, `material_model_status`, `material_response_model`,
`calibration_status`, `threshold_source`, and `qa_status`.

| Old or compatibility name | Canonical Stage 6 name |
| --- | --- |
| `stage7_materials/07_materials_design_table.csv` | `materials/material_application_design_table.csv` |
| `materials/07_materials_design_table.csv` | `materials/material_proxy_fluence_threshold_summary.csv` compatibility copy |
| `publication_study/calibration_template.csv` | `materials/material_calibration_template.csv` |
| `stage_f/F_capsule_sweep.csv` | Not a Stage 6 canonical output; belongs to later advanced/capsule work |
| `stage_f/F_design_solver.csv` | Not a Stage 6 canonical output; belongs to later advanced/capsule work |

## Stage 7 capsule

Stage 7 capsule/weld-feature notebooks write canonical outputs under
`outputs/csv/capsule/`. Rows include native capsule schema metadata:
`run_id`, `generated_at_utc`, `source_schema_version`,
`geometry_model_status`, `material_model_status`, `material_response_model`,
`calibration_status`, and `qa_status`.

| Old or compatibility name | Canonical Stage 7 name |
| --- | --- |
| `stage9_capsule/09_capsule_design_feasible_set.csv` | `capsule/capsule_candidate_ranking.csv` |
| `stage9_capsule/09_capsule_apodization_sweep.csv` | `capsule/capsule_weld_feature_design_summary.csv` |
| `stage9_capsule/09_capsule_acceptance_summary.csv` | `capsule/capsule_acceptance_summary.csv` |
| `stage_f/F_capsule_sweep.csv` | old Stage F compatibility output, superseded by `capsule/capsule_weld_feature_design_summary.csv` |

## Stage 8 advanced hexagonal/polygonal/discrete

Stage 8 advanced notebooks write canonical outputs under `outputs/csv/advanced/`.
Rows include native advanced-beam schema metadata: `run_id`,
`generated_at_utc`, `source_schema_version`, `beam_family`, `model_level`,
`generation_method`, `hardware_status`, `qa_status`,
`propagation_stability_status`, `focal_plane_only`, `propagation_tested`,
`phase_only_compatible`, `complex_amplitude_required`, and
`simulation_only`.

| Old or compatibility name | Canonical Stage 8 name |
| --- | --- |
| `stage_h2/H2_air_knob_sweep.csv` | `advanced/hexagonal_polygonal_beam_summary.csv` |
| `stage_h2/H2_survival_summary.csv` | `advanced/hexagonal_polygonal_acceptance_summary.csv` |
| `stage_h2/H2_transient_hexlike_scan.csv` | `advanced/hexagonal_polygonal_beam_summary.csv` compatibility copy |
| `polygonal_hex_ring/11_polygonal_hex_ring_acceptance_metrics.csv` | `advanced/hexagonal_polygonal_beam_summary.csv` |
| `polygonal_hex_ring/11_polygonal_hex_ring_z_stability.csv` | z-profile compatibility rows for `advanced/hexagonal_polygonal_beam_summary.csv` |
| `polygonal_hex_ring/11_polygonal_hex_ring_materials_proxy.csv` | superseded by optical/geometry-only Stage 8 rows; no material-writing claim |
| `polygonal_hex_ring/12_hollow_hex_sidelobe_ideal_sweep.csv` | `advanced/hexagonal_polygonal_beam_summary.csv` compatibility copy |
| `polygonal_hex_ring/12_hollow_hex_sidelobe_lab_shortlist.csv` | `advanced/hexagonal_polygonal_beam_summary.csv` compatibility copy |
| `hex_outline/13_hollow_hex_outline_checkpoint.csv` | `advanced/hexagonal_polygonal_beam_summary.csv` compatibility copy |
| `hex_outline/14_hexlike_transient_vs_outline.csv` | `advanced/hexagonal_polygonal_beam_summary.csv` compatibility copy |
| `hex_outline/15_hybrid_transient_seed_lab_gate.csv` | `advanced/hexagonal_polygonal_beam_summary.csv` compatibility copy |
| `hex_outline/16_hex_bessel_like_summary.csv` | `advanced/hexagonal_polygonal_beam_summary.csv` |
| `hex_outline/16_hex_bessel_like_z_profile.csv` | z-profile compatibility rows for `advanced/hexagonal_polygonal_beam_summary.csv` |
| `hex_outline/17_zernike_hex_bessel_sweep.csv` | `advanced/hexagonal_polygonal_beam_summary.csv` compatibility copy |
| `stage10_discrete/10_discrete_pattern_summary.csv` | `advanced/discrete_nfold_beam_summary.csv` |
| `stage10_discrete/10_discrete_acceptance_summary.csv` | `advanced/discrete_nfold_acceptance_summary.csv` |
| `stage10_discrete/10_discrete_cgh_exports.csv` | `advanced/discrete_nfold_beam_summary.csv` compatibility copy |
| `stage10_discrete/10_discrete_encoding_comparison.csv` | `advanced/discrete_nfold_beam_summary.csv` compatibility copy |
| `stage10_discrete/10_discrete_continuous_limit.csv` | `advanced/discrete_nfold_beam_summary.csv` compatibility copy |
