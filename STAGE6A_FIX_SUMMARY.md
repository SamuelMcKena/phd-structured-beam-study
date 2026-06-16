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

- `Publication_Study/vbb_study/vbb_train_viz.py`: `2026-06-16T14:27:03.5190940Z`

Frame sanity check after edit:

| Method | Frame count | z=0 endpoint | z=peak core metadata |
|---|---:|---|---|
| Holographic | 6 | `focused surface seed (z=0, pre-zone)` | `peak_z_um=-50.33086756475486`, `ring_radius_um=1.635420391088523` |
| Physical | 6 | `after physical axicon (z=0, pre-zone)` | `peak_z_um=75.67982781475047`, `ring_radius_um=2.3874999999999997` |

## Notebook Execution

Pending.

## PNG Timestamp Proof

Pending.

## Visual Inspection Evidence

Pending.

## Tests

Pending.

## Commits

Pending.
