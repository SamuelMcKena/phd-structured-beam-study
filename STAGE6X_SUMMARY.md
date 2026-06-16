# Stage 6X Summary

Diagnostic-only sweep completed against the current active source. No source
code, notebooks, baselines, or lock files were edited. Deliverables:

- `VISUAL_CROSSCHECK_INVENTORY.md`
- `VISUAL_CROSSCHECK_FINDINGS.md`
- `STAGE6X_SUMMARY.md`

Scope notes:

- `.github/copilot-instructions.md` was requested but is absent from this checkout.
- The active notebook package is `Publication_Study/vbb_study/`; the root-level
  `vbb_study/` compatibility tree is not what the 22 canonical notebooks use.
- Pre-existing dirty/untracked files were left untouched.

Classification counts for unguarded plotting paths:

| Classification | Count |
|---|---:|
| DIVERGES | 2 |
| DIFFERENT_PLANE | 1 |
| AGREES | 2 |
| UNVERIFIABLE | 12 |

Ranked DIVERGES findings needing follow-up:

1. `quicklook.plot_slm_preview` / `run_slm_phase_preview()` uses the sample-config
   design for the SLM mask, while `bt.run_case(..., path="realistic")` uses the
   beam-to-surface air-path design. Active-aperture wrapped-phase RMS delta is
   `1.84141 rad`; preview cone `2.05956 lp/mm` vs run-case cone `5.02533 lp/mm`.
2. `notebooks/lab_realism/04_objective_pupil_and_first_order_filtering.ipynb`
   carrier-cone diagram uses the sample-medium design and disagrees with
   run-case first-order geometry. It flips achievability for general blaze
   32 px and limits blaze 12 px.

Known focused-seed class:

- `vbb_train_viz.plot_train_visualiser` is `DIFFERENT_PLANE`, not a same-plane
  divergence in the current source. The displayed z=0 seed/post-axicon field
  differs from the run-case peak-plane metric, but propagating that same field
  to the run-case peak z reproduces the metric within numerical tolerance.

Lock status:

- Lock files and baseline directories were not edited by this stage.
- `tests/test_characterisation_lock.py`: 9 passed.
- `tests/test_characterisation_lock_prod.py`: 9 passed.
- Both pytest runs emitted a cache-only warning because pytest could not write
  `C:\PhD\Code\.pytest_cache\v\cache\nodeids`; the lock assertions passed.
