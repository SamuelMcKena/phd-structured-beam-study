# Stage 2 Remaining Lazy Imports

Stage 2 reduced in-function lazy imports in `bessel_twin_core.py` from 10 to 6.
The remaining imports are orchestration or config/dataclass coupling points, not
leaf propagation/equation helpers.

| Current line | Function | Import | Why it remains |
| --- | --- | --- | --- |
| 574 | `objective_map_from_config` | `from vbb_study import vbb_planes` | This helper still falls back to `compute_design_from_targets`, which constructs the monolith-local `BeamDesign` dataclass. Move after config/design dataclasses are inverted. |
| 1725 | `_case_validity_report` | `from vbb_study import vbb_regime` | Validity reporting is case orchestration around regime policy and result dictionaries. |
| 1772 | `_run_full_source_to_sample_case` | `from vbb_study import vbb_axicon` | Physical axicon route calls `PhysicalAxicon`, whose module still imports `bessel_twin_core` at top level. |
| 1807 | `_run_full_source_to_sample_case` | `from vbb_study import vbb_axicon` | Holographic ideal route calls `HolographicAxicon`, with the same current top-level reverse dependency. |
| 1893 | `run_case` | `from vbb_study import vbb_studies` | `run_case` is the public orchestration shim; `vbb_studies` still imports the facade. |
| 2139 | `run_sampling_feasibility_envelope` | `from vbb_study import vbb_regime, vbb_studies` | Study-runner orchestration; outside the leaf engine/equation move set. |
