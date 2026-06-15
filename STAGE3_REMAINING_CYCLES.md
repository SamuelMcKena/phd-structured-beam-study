# Stage 3 Remaining Cycles

No remaining import cycle requires an in-function ``vbb_study`` import in
``bessel_twin_core.py``.

Final counts:

- In-function lazy ``vbb_study`` imports in ``bessel_twin_core.py``: 0
  (Stage 2 baseline: 6).
- Top-level ``bessel_twin_core`` imports under ``vbb_study/``: 0
  (Stage 3 start: 18).

Some ``vbb_study`` modules still call monolithic facade-owned orchestration
helpers at runtime through ``vbb_study.facade.core()``. That adapter performs no
top-level facade import and is a temporary compatibility boundary until those
helpers are moved in later stages.
