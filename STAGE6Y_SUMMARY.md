# Stage 6Y Summary

## Pre-fix root cause gate

This section was written before applying any Stage 6Y source, notebook, or output fix.

Stage 6X DIVERGES #1 and #2 share the same physical mistake, but not the same code helper:

- `vbb_study/publication/quicklook.py::run_slm_phase_preview()` reaches the SLM design through `quicklook._phase_from_config(twin)`, which computed `compute_design_from_targets(twin.laser, twin.target, twin.material)` directly on the sample-medium `TwinConfig`.
- `notebooks/lab_realism/04_objective_pupil_and_first_order_filtering.ipynb` duplicated the same pattern inline in the carrier-cone plot and first-order geometry CSV cells: `cfg = vbb_regime.config_for_regime(base, regime)`, then `bt.compute_design_from_targets(cfg.laser, cfg.target, cfg.material)`.
- `Publication_Study/tools/update_lab_realism_notebooks.py` also contains the lab04 first-order geometry CSV template with the same sample-context design call, so that generator must be corrected with the notebook to avoid restoring the stale calculation.

The two call sites are therefore independent duplications of the same air-vs-sample context error. There is no shared plotting helper to fix once.

The air-path context is the physically correct one for SLM-plane phase, SLM cone frequency, blaze-period carrier, and first-order filter geometry. The SLM and Fourier-order filtering are upstream of the objective and sample, in air; they cannot depend directly on the Cr:ZnSe refractive index. The validated `run_case(path="realistic")` beam-to-surface route already reflects this by calling `vbb_studies.beam_air_config()` before computing the holographic SLM field and first-order geometry. The sample-side material model and interface mapping are what connect that air-side beam to the in-medium target. Using the sample-medium `k_r` directly as the SLM cone frequency skips that mapping and produces the observed factor-of-2.44 error.

No evidence was found that the sample-context calculation was a deliberate exploratory "aim the SLM directly at the in-sample target" mode in these quicklook or lab04 outputs.

## Fix status

## Fix 1: quicklook SLM preview

Changed `Publication_Study/vbb_study/publication/quicklook.py::_phase_from_config()` to derive the SLM design and phase field from `vbb_studies.beam_air_config(twin)`, matching the validated `run_case(path="realistic")` beam-to-surface path.

Re-run comparison for the default quicklook case:

| Quantity | Pre-fix sample-context value | Fixed quicklook value | Validated run-case value |
|---|---:|---:|---:|
| `gamma_slm_deg` | 0.2428511254567136 | 0.5925391693126465 | 0.5925391693126465 |
| SLM cone frequency (lp/mm) | 2.059561279829788 | 5.025329522784684 | 5.025329522784684 |
| Active-aperture wrapped phase RMS delta (rad) | 1.84141 | 0.0 | 0.0 target |
| Active pixels with `abs(delta) > 0.1 rad` | 0.9665 | 0.0 | 0.0 target |
| Max active wrapped phase delta (rad) | not re-run | 0.0 | 0.0 target |

Regenerated ignored output figure:

- `Publication_Study/outputs/figures/quicklook/01_quicklook_slm_phase_mask.png`

Lock status after this fix:

- `C:\PhD\.venv2\Scripts\python.exe -m pytest tests\test_characterisation_lock.py tests\test_characterisation_lock_prod.py`
- Result: 18 passed, 1 cache-permission warning (`C:\PhD\Code\.pytest_cache\v\cache\nodeids` could not be written).

## Fix 2: lab04 carrier-cone diagram and CSV

Changed the lab04 carrier-cone plot cell, lab04 first-order geometry CSV cell, and the lab04 CSV template in `Publication_Study/tools/update_lab_realism_notebooks.py` to compute first-order cone/filter geometry from `vbb_studies.beam_air_config(cfg)`.

Regenerated outputs:

- `Publication_Study/outputs/figures/stage_c/nb04_carrier_cone_diagram.png` (ignored by Git; regenerated in place, 77,242 bytes)
- `Publication_Study/outputs/csv/stage_c/first_order_filter_geometry_summary.csv`
- Cleared stale stored output from the affected lab04 CSV display cell so the notebook no longer embeds the old sample-context table.

Re-run carrier/cone comparison:

| Case | Pre-fix sample cone (lp/mm) | Fixed air cone (lp/mm) | Validated run-case cone (lp/mm) | Carrier (lp/mm) | Pre-fix valid? | Fixed valid? | Run-case valid? |
|---|---:|---:|---:|---:|---|---|---|
| `general_blaze12` | 2.059561279829788 | 5.025329522784684 | not separately propagated | 10.416666666666666 | true | true | not separately propagated |
| `general_blaze20` | 2.059561279829788 | 5.025329522784684 | not separately propagated | 6.250000000000001 | true | true | not separately propagated |
| `general_blaze32` | 2.059561279829788 | 5.025329522784684 | 5.025329522784684 | 3.90625 | true | false | false |
| `limits_blaze12` | 9.268025759234048 | 22.613982852531077 | 22.613982852531077 | 10.416666666666666 | true | false | false |
| `limits_blaze20` | 9.268025759234048 | 22.613982852531077 | not separately propagated | 6.250000000000001 | false | false | not separately propagated |
| `limits_blaze32` | 9.268025759234048 | 22.613982852531077 | not separately propagated | 3.90625 | false | false | not separately propagated |

Resolved achievability flips:

- `general_blaze32`: before `current_lab_realizable` / valid=true; after `diagnostic_only` / valid=false; run-case valid=false.
- `limits_blaze12`: before `current_lab_realizable` / valid=true; after `diagnostic_only` / valid=false; run-case valid=false.

Lock status after this fix:

- First lock attempt completed all tests with `18 passed`, but the shell command hit the 20-minute timeout after printing the pytest summary, so it returned exit code 124.
- Clean rerun with a longer timeout: `C:\PhD\.venv2\Scripts\python.exe -m pytest tests\test_characterisation_lock.py tests\test_characterisation_lock_prod.py`
- Result: 18 passed, 1 cache-permission warning (`C:\PhD\Code\.pytest_cache\v\cache\nodeids` could not be written).

## Final status

Both Stage 6X DIVERGES findings are resolved without engine changes. The only source/template changes are:

- `Publication_Study/vbb_study/publication/quicklook.py`
- `Publication_Study/notebooks/lab_realism/04_objective_pupil_and_first_order_filtering.ipynb`
- `Publication_Study/tools/update_lab_realism_notebooks.py`

Tracked regenerated output:

- `Publication_Study/outputs/csv/stage_c/first_order_filter_geometry_summary.csv`

Ignored regenerated output figures:

- `Publication_Study/outputs/figures/quicklook/01_quicklook_slm_phase_mask.png`
- `Publication_Study/outputs/figures/stage_c/nb04_carrier_cone_diagram.png`
