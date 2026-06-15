# Stage 3 Summary

Stage 3 started from `f2ec5cc` and completed the dependency inversion without
touching notebooks, `run_*.py`, or `tools/`.

## Moves

- `3919a8e` - documented the Stage 3 move plan and blocker log.
- `bd656d0` - moved config dataclasses, literals, and units to `vbb_study.config`.
- `ee41617` - moved pure design builders to `vbb_study.design`.
- `674cee5` - added the lazy facade adapter for still-monolithic runtime calls.
- `00f3c2e` - removed `vbb_regime` facade import.
- `44d0bf5` - removed `vbb_axicon` facade import.
- `aa47a58` - removed `vbb_studies` facade import.
- `254818e` - removed scalar cases facade import.
- `281e11f` - removed `vbb_viz` facade import.
- `2f5dd8a` - removed `vbb_train_viz` facade import.
- `102e1a8` - removed `vbb_vector` facade import.
- `a462f63` - removed `vbb_polarized_train` facade import.
- `7e7a0e7` - removed `vbb_discrete` facade import.
- `82dfa80` - removed `vbb_hexagon_study` facade import.
- `0214255` - removed `vbb_hex_outline` facade import.
- `808b069` - removed `vbb_polygonal` facade import.
- `a2582f0` - removed `vbb_materials` facade import.
- `a982a8d` - removed `vbb_materials_study` facade import.
- `86e01ce` - removed `vbb_capsule` facade import.
- `3ee689d` - removed `vbb_sample_study` facade import.
- `3cd9f95` - removed `vbb_validation` facade import.
- `aea07f5` - removed quicklook facade import.
- `f0d2c02` - promoted facade study imports to module level.

## Results

- In-function lazy ``vbb_study`` imports in `bessel_twin_core.py`: 6 -> 0.
- Top-level `bessel_twin_core` imports under `vbb_study/`: 18 -> 0.
- Fresh imports verified for `bessel_twin_core`, `vbb_study.vbb_axicon`, and
  `vbb_study.vbb_studies`.
- Config/design names remain importable from the facade, including
  `BeamDesign`, `TwinConfig`, and `compute_design_from_targets`.
- No blockers or lock-red moves were encountered.

## Final Gate

The characterisation lock remained green after every committed move.
