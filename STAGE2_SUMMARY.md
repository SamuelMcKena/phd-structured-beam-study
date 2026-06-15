# Stage 2 Summary

Stage 2 started from `748dc3f` with the Stage 1 characterisation lock green and
10 in-function lazy imports in `bessel_twin_core.py`.

## Moves Made

| Commit | Move |
| --- | --- |
| `09bbdef` | Moved scalar grid, FFT, phase, Gaussian-amplitude, and `compute_kr` primitives to `vbb_study/equations/fields.py`. |
| `e6d6fb2` | Moved `objective_map_from_design_inputs` and `headline_length_tags` to `vbb_study/equations/objective_pupil.py`. |
| `e4b8dc6` | Moved interface pupil helpers to `vbb_study/equations/interface.py`. |
| `498c3db` | Moved BL-ASM/SAS propagation kernels to `vbb_study/equations/propagation.py`. |
| `f6ef1dc` | Moved ideal field constructors to `vbb_study/equations/scalar_bessel.py`. |
| `bc3af91` | Moved radial metric adapters and strict Bessel-region helper to `vbb_study/equations/metrics.py`. |

`bessel_twin_core.py` remains the public facade by importing and re-exporting
the moved names. `run_case` and the notebook/script-facing API signatures were
not changed.

## Reverted Moves

None. No planned move turned the characterisation lock red.

## Lazy Import Count

- Stage-1 count in `bessel_twin_core.py`: 10.
- Final Stage-2 count in `bessel_twin_core.py`: 6.
- Remaining imports are documented in `STAGE2_REMAINING_CYCLES.md`.

Final remaining lazy imports:

```text
574:    from vbb_study import vbb_planes
1725:    from vbb_study import vbb_regime
1772:        from vbb_study import vbb_axicon
1807:        from vbb_study import vbb_axicon
1893:    from vbb_study import vbb_studies
2139:    from vbb_study import vbb_regime, vbb_studies
```

## Import Check

```text
C:\PhD\.venv2\Scripts\python.exe -c "import bessel_twin_core; print('import ok')"
import ok
```

No `from bessel_twin_core import ...` or `import bessel_twin_core` entries were
introduced under `Publication_Study/vbb_study/equations/`.

## Final Lock Result

```text
tests/test_characterisation_lock.py .........                            [100%]
9 passed, 1 warning in 16.68s
```

The warning is the existing pytest-cache access warning under
`C:\PhD\Code\.pytest_cache`; it does not affect the physics lock.
