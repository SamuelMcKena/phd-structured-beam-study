# Stage 6X Visual Cross-Check Inventory

Scope: the 22 notebooks in `Publication_Study/run_study.py` and the active
helpers they call under `Publication_Study/vbb_study/`, plus
`Publication_Study/publication_diagnostics.py` where notebooks call plotting
helpers directly. The root-level compatibility `vbb_study/` tree was not used
by the canonical notebooks.

Note: `.github/copilot-instructions.md` was requested but is not present in
this checkout (`rg --files --hidden` found no copy under `C:\PhD`).

Status keys:

- `TRUSTED`: consumes `bt.run_case()` result dicts, `metrics`, `volume`,
  or tables built directly from those results.
- `DISPLAY_ONLY`: low-level renderer; trust is inherited from the caller.
- `UNGUARDED`: builds or displays a separately constructed field, phase,
  propagation, or geometry not taken directly from a run-case result.
- `INLINE`: notebook cell code rather than a named helper.

| Source / function | Notebooks that call it | Data path | Status |
|---|---|---|---|
| `vbb_study/publication/notebook_widgets.py::_run_and_plot` via `interactive_quicklook` | scalar 02-05; lab 01-06; vector 01-03; materials 01-03; advanced 01-03 | Calls `bt.run_case(..., path="ideal")`; plots result volume/metrics in an interactive widget only. | TRUSTED |
| `vbb_study/publication/quicklook.py::plot_slm_preview` | quicklook/00 | Displays phase from `run_slm_phase_preview()` -> `_phase_from_config()` -> `build_realistic_slm_field()`, not from the adjacent run-case lab preview. | UNGUARDED |
| `vbb_study/publication/quicklook.py::plot_ideal_preview` | quicklook/00 | Trusted when given `run_conical_axicon_preview()`; unguarded when given `run_ideal_beam_preview()` because that builds a true Bessel-Gauss target field directly. | UNGUARDED/MIXED |
| `vbb_study/publication/quicklook.py::plot_lab_preview` | quicklook/00 | Displays `run_lab_realistic_preview()` output from `bt.run_case(..., path="realistic")`. | TRUSTED |
| `vbb_study/publication/quicklook.py::plot_through_sample_preview` | quicklook/00 | Displays `run_through_sample_preview()` output built from a run-case air surface field plus `vbb_sample_study.run_through_sample()`. | UNGUARDED |
| `vbb_study/publication/quicklook.py::plot_material_proxy_preview` | quicklook/00 | Displays fluence/threshold proxy from through-sample peak plane. | UNGUARDED |
| `vbb_study/publication/quicklook.py::plot_config_comparison` | quicklook/00 | Notebook uses `path="conical"`, which compares run-case conical previews. | TRUSTED |
| `vbb_study/publication/quicklook.py::plot_parameter_sweep_preview` | quicklook/00 | Plots metrics from `run_lab_realistic_preview()` sweeps. | TRUSTED |
| `vbb_study/publication/quicklook.py::plot_four_condition_phase_comparison` | quicklook/00 | XY/XZ panels come from run-case results; phase-mask panel uses `_phase_from_config()` explanatory masks. | UNGUARDED/MIXED |
| `vbb_study/publication/visuals.py::plot_xy_intensity`, `plot_xz_intensity`, `plot_xy_xz_pair`, `plot_slm_phase_mask`, `plot_case_comparison_grid`, `plot_metric_summary_box` | quicklook/00 via quicklook helpers | Generic renderers only. | DISPLAY_ONLY |
| `Publication_Study/publication_diagnostics.py::plot_case_diagnostics` | scalar/02 | Uses `bundle["result"]["volume"]`, `metrics`, `radial` from `build_shortlist_bundles()` run-case bundles. | TRUSTED |
| `Publication_Study/publication_diagnostics.py::plot_axial_plane_montage` | scalar/02 | Uses run-case `volume["planes"]` and `volume["z"]`. | TRUSTED |
| scalar/02 inline summary panel | scalar/02 | Plots summary dataframe from run-case bundles. | TRUSTED/INLINE |
| `Publication_Study/publication_diagnostics.py::plot_self_healing_diagnostics` | scalar/03 | Displays `build_self_healing_bundle()` custom obstacle propagation. | UNGUARDED |
| scalar/03 inline recovery-curve cell | scalar/03 | Plots custom self-healing recovery traces. | UNGUARDED/INLINE |
| `Publication_Study/publication_diagnostics.py::plot_metric_heatmap` | scalar/04 | Plots run-case/sweep summary tables; no field construction. | TRUSTED |
| scalar/04 inline lollipop, heatmap, tornado, and sweep panels | scalar/04 | Plots tables from `bt.run_*` study functions and run-case bundles. | TRUSTED/INLINE |
| scalar/05 inline SAS validation montage and radial-profile cells | scalar/05 | Builds Gaussian/aperture/conical fields and compares BL-ASM vs SAS directly. | UNGUARDED/INLINE |
| `vbb_study/vbb_train_viz.py::plot_train_visualiser` | lab 01, lab 02, lab 03 | Calls `build_train_frames()` and displays intermediate fields, including z=0 seed/post-axicon fields. | UNGUARDED |
| `vbb_study/vbb_train_viz.py::plot_sampling_qa` | lab 01, lab 02, lab 03 | Plots analytic sampling/validity margins from `vbb_regime.validity_map()`. | UNGUARDED |
| `vbb_study/vbb_train_viz.py::plot_holographic_carrier_filter_tradeoff` | lab 01 | Plots table from `holographic_carrier_filter_sweep()`, which calls `bt.run_case()` for each row. | TRUSTED |
| `vbb_study/viz_fields.py::linked_field_views` | lab 02 | Consumes `bt.run_case()` result dict directly. | TRUSTED |
| `vbb_study/viz_fields.py::azimuthal_order_panel` | lab 02 | Uses ring samples from a run-case `surface_field`; A4 spectrum helper is validated. | TRUSTED |
| lab/04 inline pupil-fill schematic | lab 04 | Computes objective and Gaussian-footprint geometry directly. | UNGUARDED/INLINE |
| lab/04 inline carrier-cone diagram | lab 04 | Computes first-order carrier/cone geometry directly in the notebook. | UNGUARDED/INLINE |
| `vbb_study/vbb_sample_study.py::plot_sample_result_comparison` | lab 05 | Displays `run_through_sample()` results from a run-case air surface field. | UNGUARDED |
| lab/06 inline `plot_journey_grid` | lab 06 | Displays stitched full air-to-sample journeys from `run_full_source_to_sample()`. | UNGUARDED/INLINE |
| `vbb_study/viz_fields.py::complex_field_image` | lab 06 | Notebook passes run-case `air_result.surface_field` and measured charge label. | TRUSTED |
| `vbb_study/vbb_vector.py::plot_analyzer_family_grid` | vector/01 | Displays analytic vector target fields built by `build_analytic_vector_mode()`. | UNGUARDED |
| `vbb_study/vbb_vector.py::plot_polarization_quiver` | vector/01 | Displays analytic vector target polarization field. | UNGUARDED |
| `vbb_study/vbb_vector.py::plot_total_and_analyzer_panel` | vector/02 | Displays analytic or lab-vector approximations built outside scalar `run_case()`. | UNGUARDED |
| publication_exports inline comparison, charge, heatmap, radial, axial, and sensitivity panels | publication_exports/03 | Plots run-case bundle metrics, radial profiles, and volumes from `publication_diagnostics`. | TRUSTED/INLINE |

