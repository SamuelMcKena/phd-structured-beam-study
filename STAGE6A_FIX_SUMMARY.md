# Stage 6A-FIX Retry Summary

## Decision Gate

Stage 6X already established that the train visualiser discrepancy is `DIFFERENT_PLANE`, not a divergent metric bug. I did not re-derive the full sweep.

Numbers carried forward from Stage 6X:

| Route/panel | z=0 pre-zone value | Validated peak-plane value | Validated peak z | Interpretation |
|---|---:|---:|---:|---|
| Holographic focused surface seed | ring radius `1.38758 um` | ring radius `1.63542 um` | `-50.3309 um` | Different plane; BL-ASM propagation from z=0 to peak reproduces the validated value. |
| Physical after physical axicon | ring radius `4.64934 um` | ring radius `2.38750 um` | `75.6798 um` | Different plane; BL-ASM propagation from z=0 to peak reproduces the validated value. |

Fix decision: keep the z=0 route endpoint visible, label it explicitly as pre-zone, and add a canonical peak-z core panel whose plane matches the `ring_radius_um` metrics/CSV plane.

## Generator Template Check

`Publication_Study/tools/update_lab_realism_notebooks.py` was checked for train-visualiser template cells for lab notebooks 01, 02, and 03. It contains setup/summary templates for those notebooks, but no `plot_train_visualiser(...)` template cells. Therefore there was no train-visualiser generator template to patch. The shared source function `vbb_study/vbb_train_viz.py::plot_train_visualiser()` controls the affected figures for all three notebooks.

## Source Change

Edited `Publication_Study/vbb_study/vbb_train_viz.py`.

- Holographic and physical train frame builders now emit six columns.
- Existing route endpoint labels are explicit:
  - `focused surface seed (z=0, pre-zone)`
  - `after physical axicon (z=0, pre-zone)`
- New sixth column:
  - `Bessel zone core (z=peak, canonical)`
- The new core field is propagated from the visualiser's own z=0 endpoint field to the validated `run_case(...).volume["z"][peak_index]` plane using the BL-ASM propagator.
- The core panel's upper view uses `vbb_study.viz_fields.complex_field_image()` for domain-coloured phase hue with amplitude brightness.
- The figure suptitle/caption state that `ring_radius_um` is measured at z=peak, not in the z=0 pre-zone panel.

Source edit timestamp:

- `Publication_Study/vbb_study/vbb_train_viz.py`: `2026-06-16T14:44:33.4529403Z`

Frame sanity check after edit:

| Method | Frame count | z=0 endpoint | z=peak core metadata |
|---|---:|---|---|
| Holographic | 6 | `focused surface seed (z=0, pre-zone)` | `peak_z_um=-50.33086756475486`, `ring_radius_um=1.635420391088523` |
| Physical | 6 | `after physical axicon (z=0, pre-zone)` | `peak_z_um=75.67982781475047`, `ring_radius_um=2.3874999999999997` |

## Notebook Execution

Executed end-to-end with `C:\PhD\.venv2` Python/Jupyter:

- `notebooks/lab_realism/01_holographic_axicon_route.ipynb`
- `notebooks/lab_realism/02_physical_axicon_route.ipynb`
- `notebooks/lab_realism/03_holographic_vs_physical_axicon.ipynb`

Jupyter on this Windows sandbox could not set ACLs on connection files unless `JUPYTER_ALLOW_INSECURE_WRITES=1` was set. Re-execution used:

- `JUPYTER_ALLOW_INSECURE_WRITES=1`
- `JUPYTER_RUNTIME_DIR=C:\PhD\Code\Publication_Study\outputs\jupyter_runtime`

Final notebook timestamps:

| Notebook | Last write time UTC |
|---|---|
| `01_holographic_axicon_route.ipynb` | `2026-06-16T14:47:27Z` |
| `02_physical_axicon_route.ipynb` | `2026-06-16T14:48:32Z` |
| `03_holographic_vs_physical_axicon.ipynb` | `2026-06-16T14:49:43Z` |

## PNG Timestamp Proof

The final train PNGs are newer than the source edit timestamp (`2026-06-16T14:44:33.4529403Z`):

| PNG | Last write time UTC | Size |
|---|---:|---:|
| `outputs/figures/stage_c/stage_c_train_holographic.png` | `2026-06-16T14:49:10Z` | `4,721,184` bytes |
| `outputs/figures/stage_c/stage_c_train_physical.png` | `2026-06-16T14:49:27Z` | `3,028,344` bytes |

Caption sidecars were regenerated at the same times and explicitly state that the z=0 pre-zone endpoint differs from the z=peak canonical core, and that `ring_radius_um` is a z=peak metric.

## Visual Inspection Evidence

Opened and inspected the regenerated figures.

Notebook 01 / holographic:

- `stage_c_train_holographic.png` now has six columns.
- Column 5 is labelled `focused surface seed (z=0, pre-zone)` and shows the large bright pre-zone seed annulus in the surface-in-air grid.
- Column 6 is labelled `Bessel zone core (z=peak, canonical)` with `z=-50.3 um, r=1.64 um`.
- The new domain-coloured peak-core panel shows a compact central vortex core with approximately three visible phase-colour cycles/spiral arms, matching the measured label `measured winding = 3.00 (design ell=3; charge preserved)`.

Notebook 02 / physical:

- `stage_c_train_physical.png` now has six route columns.
- Column 5 is labelled `after physical axicon (z=0, pre-zone)` and shows the pre-zone post-axicon field.
- Column 6 is labelled `Bessel zone core (z=peak, canonical)`. The lab row is labelled `z=75.7 um, r=2.39 um`.
- The physical peak-core panel is tighter at the centre but shows concentric/no-helical phase structure rather than a three-arm winding, matching the preserved measured-charge label `measured winding = 0.00 (design ell=3; charge stripped; SLM2 conjugate_mode='full' strips the helical phase)`.

Notebook 03 / comparison:

- Re-execution regenerated both shared train figures from the comparison notebook after notebooks 01 and 02.
- The final holographic output carries the `measured winding = 3.00` label and the z=peak core column.
- The final physical output carries the `measured winding = 0.00` label and the z=peak core column.

## Tests

- `C:\PhD\.venv2\Scripts\python.exe -m pytest tests\test_viz_fields.py`
  - Result: 9 passed, 1 pytest cache-permission warning.
- `C:\PhD\.venv2\Scripts\python.exe -m pytest tests\test_characterisation_lock.py tests\test_characterisation_lock_prod.py`
  - Result: 18 passed, 1 pytest cache-permission warning.

The warning in both runs was the existing inability to write `C:\PhD\Code\.pytest_cache\v\cache\nodeids`.

Engine/baseline diff check:

- `bessel_twin_core.py`, `vbb_studies.py`, `design.py`, `equations/propagation.py`, `baselines`, and `baselines_prod` had no diff from this fix.

## Commits

- `3fcc097 Update publication study notebooks and validation outputs` postdates prior Stage 6A and Stage 6Y commits and contains the source/notebook/output part of this retry, including `vbb_train_viz.py`, `FOCUSED_SEED_FINDING.md`, and this summary file.
