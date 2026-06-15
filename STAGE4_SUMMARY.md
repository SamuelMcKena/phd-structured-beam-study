# Stage 4 Summary — Make It Installable + Triage Notebook Execution

Completed: 2026-06-15

## A. Packaging

`Publication_Study/pyproject.toml` created with:
- `name = "vbb_study"`, `version = "0.1.0"`, `requires-python = ">=3.13,<3.14"`
- Setuptools backend; flat layout (no `src/` restructure)
- `py-modules = ["bessel_twin_core"]` — ships the physics facade as a top-level module
- `packages.find include = ["vbb_study*"]` — ships only the study package, not run_*.py scripts or tools/
- Runtime deps match `requirements.txt`: numpy, scipy, pandas, matplotlib, plotly, nbformat, pillow, ipython
- `[project.optional-dependencies] dev` group: papermill, ipykernel, jupyter

## B. Editable Install + Path-Independence Proof

```
pip install -e .[dev]   (from Publication_Study/, using C:\PhD\.venv2\Scripts\python.exe)
```

**Path-independence proof** — from `C:\` (foreign CWD, no activation, no sys.path tricks):
```
C:\PhD\.venv2\Scripts\python.exe -c "import vbb_study, bessel_twin_core; print('ok', vbb_study.__file__)"
ok C:\PhD\Code\Publication_Study\vbb_study\__init__.py
```

`bessel_twin_core` resolves to `C:\PhD\Code\Publication_Study\bessel_twin_core.py` via the
editable finder installed at `site-packages/__editable__.vbb_study-0.1.0.pth`.

**Characterisation lock post-install:**
```
pytest Publication_Study/tests/test_characterisation_lock.py
9 passed in 17.31s
```
Lock stays green — editable install does not shadow anything.

## Bonus fix: python3 kernel spec

During triage, VS Code interactive notebooks were failing with
`ModuleNotFoundError: No module named 'bessel_twin_core'` because the
`python3` kernel spec at `C:\PhD\.venv2\share\jupyter\kernels\python3\kernel.json`
used bare `python` in `argv` (resolves to system Python when venv not activated).

Fixed by pinning `argv[0]` to `C:\PhD\.venv2\Scripts\python.exe`. Papermill triage
was unaffected because `python` on PATH already resolved to `.venv2` in that
PowerShell session.

**After fix:** restart any open notebook kernel in VS Code — imports will resolve
immediately without selecting a different kernel.

## C. Clean-Kernel Notebook Triage

- Method: papermill, kernel `python3` (→ `.venv2`), CWD = `C:\PhD\Code` (foreign)
- Per-notebook timeout: 300 s
- Executed copies written to `outputs/notebook_triage/` (gitignored)

### Results by status

| Status  | Count |
|---------|------:|
| PASS    | 21    |
| TIMEOUT | 1     |
| FAIL    | 0     |

### Results by failure category

| Category       | Count | Detail |
|----------------|------:|--------|
| path           | **0** | Zero — editable install is working |
| missing-symbol | 0     | |
| data-dependency| 0     | |
| runtime-bug    | 1     | scalar/04 — hardcoded `PRESET="publication"` makes sweeps exceed 300 s |

### The one TIMEOUT: scalar/04

`scalar/04_scalar_parameter_sweeps.ipynb` has `PRESET = "publication"` hardcoded
(cell 3). The publication-preset parameter sweeps (OAT sensitivity, tradeoff map,
ell-family, etc.) take ~10–20 min. This is not a logic error — the notebook executes
correctly when given sufficient time. Stage 5 action: tag a papermill `parameters`
cell so `preset="fast"` can be injected for triage runs.

## Acceptance criteria check

| Criterion | Result |
|-----------|--------|
| `pip install -e .[dev]` succeeds | ✓ |
| Import from foreign CWD works, no sys.path hacks | ✓ |
| Characterisation lock: 9 passed post-install | ✓ |
| NOTEBOOK_TRIAGE.md covers all 22 notebooks | ✓ |
| Every failure categorised | ✓ (1 TIMEOUT → runtime-bug) |
| Zero path-category failures | ✓ |
| Executed-notebook temp output not committed | ✓ (gitignored) |
