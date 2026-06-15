# Stage 2 Move Plan

Stage 2 starts from `748dc3f` with `tests/test_characterisation_lock.py`
green and 10 in-function lazy imports in `bessel_twin_core.py`.

The goal for this pass is dependency inversion by moving leaf engine helpers
into `vbb_study/equations/` modules that do not import `bessel_twin_core`.
`bessel_twin_core.py` remains the public facade by re-exporting the moved
helpers at their original names.

## Planned Moves

| Move | Function(s) | Target | Lazy import touched | Dependency note |
| --- | --- | --- | --- | --- |
| 1 | `make_xy_grid`, `make_rect_grid`, `fft2c`, `ifft2c`, `phase_wrap`, `quantize_phase`, `phase_to_gray`, `gray_to_phase`, `gaussian_amplitude`, `next_power_of_two`, `compute_kr` | `vbb_study/equations/fields.py` | None | Leaf scalar grid, FFT, phase, and amplitude primitives. |
| 2 | `objective_map_from_design_inputs`, `headline_length_tags` | `vbb_study/equations/objective_pupil.py` | `from vbb_study import vbb_planes` at Stage-1 lines 597 and 703 | Uses `vbb_planes`, which does not import the monolith. |
| 3 | `interface_aberration_pupil`, `interface_correction_phase`, `fit_interface_zernike_terms` | `vbb_study/equations/interface.py` | None | Pupil-interface wrappers built on arrays and config-like objects. |
| 4 | `_kz_medium`, `_transfer_function_medium`, `bandlimit_mask_matsushima`, `angular_spectrum_propagate_bl`, `make_bl_asm_propagator`, `_zero_pad_center`, `_zero_unpad_center`, `_sas_z_limit_m`, `sas_validity_report`, `scalable_angular_spectrum_propagate`, `focus_to_focal_plane` | `vbb_study/equations/propagation.py` | None | Core propagation kernels; depends only on arrays, grids, scalar attributes, and field primitives. |
| 5 | `build_conical_axicon_field_ideal`, `build_bessel_gauss_field_ideal`, `build_sample_field_ideal` | `vbb_study/equations/scalar_bessel.py` | None | Ideal field constructors; keep compatibility wrapper warning unchanged. |
| 6 | `extract_radial_metrics`, `_contiguous_mask_zone`, `bessel_region_metrics` | `vbb_study/equations/metrics.py` | `from vbb_study import vbb_metrics` at Stage-1 lines 1793 and 1938 | Metric adapters around `vbb_metrics`, which does not import the monolith. |

## Deferred From This Stage

These functions are pure or mostly pure but remain coupled to config
dataclasses, public export behaviour, or orchestration. They are not moved in
this pass unless an earlier move makes them trivially safe:

- `compute_design_from_targets`: constructs `BeamDesign`; moving it first would
  require relocating config dataclasses.
- `objective_map_from_config`: falls back to `compute_design_from_targets`
  when relay metadata is absent.
- `_continuous_phase`, `render_device_hologram`, `_reduced_device_grid`,
  `_pad_rect_to_square`, `_pad_mask_to_square`, `fill_factor_amplitude`,
  `build_realistic_slm_field`, `isolate_first_order`,
  `first_order_filter_geometry`: device-realism helpers still interleave
  dataclass config objects, image export, and public facade behaviour.
- `propagate_volume`: creates a default `PropagationConfig`; move after config
  dataclasses are inverted.
- `realistic_slm_to_sample`, `_run_full_source_to_sample_case`, `run_case`,
  and `run_*` study functions: orchestration, not leaf engine helpers.

## Per-Move Gate

For each move:

1. Move bodies into the target module and replace the original definitions in
   `bessel_twin_core.py` with imports from the target module.
2. Run `C:\PhD\.venv2\Scripts\python.exe -c "import bessel_twin_core"` from
   `Publication_Study`.
3. Re-grep lazy imports in `bessel_twin_core.py`.
4. Run `C:\PhD\.venv2\Scripts\python.exe -m pytest tests/test_characterisation_lock.py`.
5. Green: commit the move. Red: revert to the previous commit, document the
   reverted move in `STAGE2_BLOCKERS.md`, and continue with the next
   independent move.
