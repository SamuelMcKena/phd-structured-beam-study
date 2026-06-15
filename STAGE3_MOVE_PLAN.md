# Stage 3 Move Plan

Stage 3 starts from `f2ec5cc` with the characterisation lock green.

Baseline counts:

- In-function lazy imports in `bessel_twin_core.py`: 6.
- Top-level facade imports under `Publication_Study/vbb_study/`: 18.

## Config And Design Inventory

| Name | Kind | Hard dependencies | Current callers |
| --- | --- | --- | --- |
| `LaserConfig` | dataclass | units, `TWOPI` | `TwinConfig`; `vbb_sample_study` default wavelength; facade functions using `config.laser`. |
| `SLMConfig` | dataclass | units | `TwinConfig`; facade SLM/device helpers; `vbb_axicon`, `vbb_studies`, and study helpers through `config.slm`. |
| `ObjectiveConfig` | dataclass | units, `EPS` | `TwinConfig`; objective/pupil/focal-plane helpers through `config.objective`. |
| `RelayConfig` | dataclass | units | `TwinConfig`; `objective_map_from_config`; config summary/reporting. |
| `MaterialConfig` | dataclass | units, `EPS` | `TwinConfig`; `compute_design_from_targets`; `vbb_materials*`; `vbb_regime`; surface/sample study helpers. |
| `EnergyBudget` | dataclass | units | `TwinConfig`; `vbb_studies.beam_air_config`; energy and fluence reporting. |
| `BeamTarget` | dataclass | units | `TwinConfig`; `compute_design_from_targets`; regime/case generation. |
| `BeamDesign` | dataclass | none beyond typing | `compute_design_from_targets`; facade metrics/propagation; `vbb_axicon`, `vbb_studies`, `vbb_regime`, `vbb_train_viz`, `vbb_polygonal`, `vbb_sample_study`. |
| `GridConfig` | dataclass | units | `SimulationPreset`, `TwinConfig`, `get_preset`, propagation/grid helpers. |
| `PropagationConfig` | dataclass | `PropagationMethod` literal | `TwinConfig`; `propagate_volume`; sample/study propagation helpers. |
| `PhysicalAxiconConfig` | dataclass | `Slm2ConjugateMode` literal, units | `TwinConfig`; physical-axicon route in `vbb_axicon` and full-source case assembly. |
| `SimulationPreset` | dataclass | `GridConfig` | `get_preset`, `default_config`. |
| `TwinConfig` | dataclass | all config dataclasses and literals | Primary config object across the facade and most `vbb_study` study modules. |
| `get_preset` | builder | `SimulationPreset`, `GridConfig`, units | `default_config`; external API via `bessel_twin_core.get_preset`. |
| `default_config` | builder | `TwinConfig`, `get_preset` | facade run helpers; `vbb_materials`, `quicklook`, `scalar_cases`, `vbb_validation`; external API. |
| `axial_scan_values` | helper | `numpy`, `TwinConfig`, `BeamDesign` | facade realistic/ideal paths; `vbb_regime.sampling_validity`; `vbb_studies` air scan helpers. |
| `compute_design_from_targets` | builder | `BeamDesign`, `objective_map_from_design_inputs`, `scipy.special`, `math` | facade paths and reports; `vbb_axicon`, `vbb_studies`, `vbb_regime`, `vbb_train_viz`, `vbb_validation`, `quicklook`, `vbb_materials*`, `vbb_polygonal`, `vbb_sample_study`. |
| `objective_map_from_config` | design helper | `vbb_planes`, `compute_design_from_targets` | facade metrics and summaries; remaining lazy import site. |

## Move Sequence

| Move | Scope | Commit gate |
| --- | --- | --- |
| 1 | Inventory only: write this plan and `STAGE3_BLOCKERS.md`. | Commit after baseline lock/import checks. |
| 2 | Create `vbb_study/config.py` as a leaf config module and move units, literals, and pure config/design dataclasses there. Re-export all names from `bessel_twin_core.py`. | Lock + `import bessel_twin_core` + re-export import test. |
| 3 | Create `vbb_study/design.py` and move `get_preset`, `default_config`, `axial_scan_values`, `compute_design_from_targets`, and `objective_map_from_config`. Re-export from facade. | Lock + import checks. |
| 4 | Remove top-level facade imports under `vbb_study/` one module at a time. Prefer `vbb_study.config`, `vbb_study.design`, and `vbb_study.equations`; use function-local facade imports only for still-facade-owned orchestration helpers. | Lock + `import bessel_twin_core` + target module fresh import after each module. |
| 5 | Convert the remaining in-function lazy imports in `bessel_twin_core.py` to module-level imports if the reverse top-level cycle is gone. Leave and document any remaining lazy imports in `STAGE3_REMAINING_CYCLES.md`. | Lock + fresh import checks. |
| 6 | Write `STAGE3_SUMMARY.md` with move/revert list, count changes, final import checks, and final lock result. | Final lock and clean status. |

## Top-Level Facade Imports To Remove

The acceptance criterion is stronger than the named examples: no top-level
`import bessel_twin_core` or `from bessel_twin_core import ...` should remain
anywhere under `vbb_study/`.

Initial top-level import sites:

- `vbb_study/vbb_axicon.py`
- `vbb_study/studies/scalar_cases.py`
- `vbb_study/vbb_capsule.py`
- `vbb_study/vbb_discrete.py`
- `vbb_study/publication/quicklook.py`
- `vbb_study/vbb_hexagon_study.py`
- `vbb_study/vbb_hex_outline.py`
- `vbb_study/vbb_materials.py`
- `vbb_study/vbb_materials_study.py`
- `vbb_study/vbb_polarized_train.py`
- `vbb_study/vbb_polygonal.py`
- `vbb_study/vbb_regime.py`
- `vbb_study/vbb_sample_study.py`
- `vbb_study/vbb_studies.py`
- `vbb_study/vbb_train_viz.py`
- `vbb_study/vbb_validation.py`
- `vbb_study/vbb_vector.py`
- `vbb_study/vbb_viz.py`

## Deferred Behaviour

`run_case`, study runners, notebook-facing commands, and physics calculations
remain in the facade unless explicitly moved above. Notebook, `run_*.py`, and
`tools/` files are out of scope for this stage.
