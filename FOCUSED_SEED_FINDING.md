# Focused Seed Finding

Stage 6X classified `vbb_study.vbb_train_viz.plot_train_visualiser` as `DIFFERENT_PLANE`, not a physics bug.

Confirmed Stage 6X values:

| Route/panel | z=0 pre-zone visual value | Validated peak-plane value | Validated peak z | Propagation check |
|---|---:|---:|---:|---|
| Holographic focused surface seed | ring radius `1.38758 um` | ring radius `1.63542 um` | `-50.3309 um` | Propagating the z=0 field with the same BL-ASM engine reproduced `1.63542 um`. |
| Physical after physical axicon | ring radius `4.64934 um` | ring radius `2.38750 um` | `75.6798 um` | Propagating the z=0 field with the same BL-ASM engine reproduced `2.38750 um`. |

Decision: keep the z=0 route endpoint visible, but label it explicitly as pre-zone and add a canonical peak-z core panel so the visual plane matches the `ring_radius_um` plane used by the metrics and CSVs.
