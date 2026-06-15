# Stage 1 Characterisation Lock Summary

## Determinism Gate

Status: PASSED.

Each canonical tripwire case was run twice in separate subprocesses with `preset="fast"` and the full serialized metrics dictionaries matched exactly.

## Cases Captured

- `general_holographic_ideal`
- `general_holographic_lab`
- `general_physical_ideal`
- `general_physical_lab`
- `limits_holographic_ideal`
- `limits_holographic_lab`
- `limits_physical_ideal`
- `limits_physical_lab`

The holographic cases follow the lab-realism notebook construction: `generation_method="holographic"`, `path="ideal"` for ideal and `path="realistic"` for lab. The physical cases follow the physical-route notebook construction: `generation_method="physical"`, `path="ideal"`, `slm2_stroke_levels=None` for ideal and `256` for lab, with `slm2_conjugate_mode="full"`.

## Exactness Split

Bit-exact:

- Full `metrics` dict.
- `design`.
- `axicon_metadata`.
- Array-bearing result fields, including volumes, fields, grids, order masks, and surface-field objects, stored as `sha256(np.ascontiguousarray(a).tobytes())` plus dtype, dtype byte-order string, shape, and C-contiguity.
- Scalar floats are stored as Python `float.hex()`.

Tolerance-based:

- None. No Stage 1 assertion silently relaxes to tolerance.

Endianness assumption: array hashes use native-endian NumPy bytes on this platform; `dtype_str` records byte order and `baselines/ENVIRONMENT.json` records `sys.byteorder`.

## Fingerprint Results

All human-facing fingerprints were classified IN RANGE in `baselines/FINGERPRINT_REPORT.md`:

- Objective demagnification: `0.008071063517253126`.
- Physical-axicon `k_r`: `1603333.3333333333 m^-1`.
- SLM2 residual phase RMS: before `1.8138168507664694 rad`, after `0.00712673122363286 rad`.
- Holographic first-order selected fraction: `0.732111295550108`.
- Bessel zone: holographic `38.01556727371095 um`, physical `113.94754279265943 um`.

## Verification

Active interpreter used for capture:

```text
C:\PhD\.venv2\Scripts\python.exe
Python 3.13.7
```

New lock only:

```text
.........                                                                [100%]
9 passed, 1 warning in 38.18s
```

Full `Publication_Study/tests` suite:

```text
................                                                         [100%]
16 passed, 1 warning in 64.63s
```

Perturbation proof:

```text
FAILED tests/test_characterisation_lock.py::test_characterisation_lock_matches_baseline[general_holographic_lab]
general_holographic_lab changed at 1 exact key(s).
- payload.metrics.bessel_zone_um.hex: expected "0x1.301fe1bc1bcfap+5", got "0x1.301fe1bc1bcfbp+5"
```

The baseline value was restored after this proof, and the new lock was rerun successfully.

Fresh venv verification:

```text
pip install -r requirements.txt
.........                                                                [100%]
9 passed in 23.94s
```

## Surprises

- `python` and `C:\PhD\Code\.venv` did not have NumPy installed; the active dependency-bearing interpreter was `C:\PhD\.venv2\Scripts\python.exe`.
- `git` was not on PATH and no `.git` metadata was visible from the workspace, so `engine_git_commit` is recorded as unavailable rather than fabricated.
- Pytest could not write cache files under `C:\PhD\Code\.pytest_cache` because of access denial; tests still passed.
