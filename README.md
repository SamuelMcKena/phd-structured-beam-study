# Structured-Beam Simulation Atlas

**Folder:** `Publication_Study/` (name kept for compatibility)
**Internal name:** VBB Study / Structured-Beam Atlas

This workspace is a structured-beam simulation atlas for a PHAROS-class
Yb:KGW ultrafast laser system.  The repeated theme is **ideal mathematical
target vs. lab-realistic implementation**: each beam family starts from a
clean analytic model, then successive stages connect it to hardware routes
and record the gap explicitly.

Scalar Bessel-Gauss, vortex Bessel-Gauss, vector Bessel, holographic and
physical axicon routes, through-sample propagation, hexagonal/polygonal
beams, discrete N-fold beams, and material-proxy planning studies are all
first-class branches of this atlas.  They are not side experiments.

See `docs/00_project_overview.md` for the full scope description.

---

## What This Project Does

- Simulates scalar Bessel, vortex Bessel, vector Bessel, polygonal/hexagonal,
  and discrete N-fold structured beams.
- Compares ideal target fields with holographic SLM, physical axicon, and
  other lab-realistic hardware routes.
- Propagates each beam through the relevant optical path including objective
  pupil, first-order filter, and air–sample interface.
- Exports figures, CSV tables, holograms, captions, and run manifests
  under `Publication_Study/outputs/`.
- Maintains documentation for theory, conventions, validation, materials
  proxies, vector hardware status, and study taxonomy under `docs/`.

## What It Does Not Do

- It does not claim every ideal target has a current lab implementation.
- It does not turn threshold maps into calibrated ablation or material-
  modification predictions.  Material outputs are planning proxies.
- It does not import from `reference_kernels/`; those files are provenance
  snapshots only.
- It does not use the root-level `docs/` or `outputs/` folders; those are
  from an earlier project layout.

See `docs/04_model_limitations.md` for the full list of model limitations.

---

## Usability And Visual QA Layer

The repo now separates three kinds of notebook use:

1. **Locked stage notebooks** regenerate canonical study outputs and keep QA labels visible.
2. **Quick-look notebook** (`notebooks/quicklook/00_quick_beam_to_sample_simulator.ipynb`) is the day-to-day adjustable beam-to-sample simulator. Change `CONFIG` inside the notebook to inspect SLM phase masks, XY profiles, XZ propagation maps, through-sample previews, material proxies, and comparison sweeps.
3. **Publication exports** must use the figure registry/caption gate rather than sweeping old output folders.

Every non-quicklook notebook now contains an **Editable Notebook Controls** block near the top. These controls make the intended user-adjustable parameters explicit without silently changing the locked physics path. See `docs/19_visual_usability_and_physics_review.md` for the current visual/physics usability review and `outputs/figures/review/quicklook_visual_contact_sheet.png` for a contact sheet of the current quick-look figures.


## Folder Structure

```
Publication_Study/
  run_study.py                ← canonical study runner (new)
  run_publication_study.py    ← compatibility wrapper → run_study.py
  finalize_outputs.py         ← compatibility wrapper → finalize_publication_outputs.py
  bessel_twin_core.py         ← scalar physics engine (source of truth)
  publication_diagnostics.py  ← study-level helpers

  notebooks/                  ← all study notebooks (organised by topic)
    00_study_overview_and_conventions.ipynb
    scalar/                   ← scalar Bessel baselines and diagnostics
    lab_realism/              ← holographic/physical axicon, pupil, interface
    vector/                   ← vector beam atlas and hardware comparison
    materials/                ← material proxy, fluence, calibration
    advanced/                 ← capsule, hexagonal, polygonal, discrete N-fold
    publication_exports/      ← final paper figures, tables, report

  vbb_study/                  ← active helper package
    equations/                ← scalar_bessel, propagation, holography, vector_jones, materials
    setup_study.py            ← path contract, bootstrap, manifest helpers
    study_taxonomy.py         ← standard label definitions
    vbb_*.py                  ← study submodules

  docs/                       ← canonical study documentation
    00_project_overview.md    ← what this project is
    00_theory.md              ← implemented optical model
    01_conventions.md         ← metric definitions and units
    02_validation.md          ← validation record
    03_materials_application.md ← materials proxy warning
    04_actual_lab_vector_case1.md ← vector hardware status
    04_model_limitations.md   ← model limitations (new)
    05_study_taxonomy.md      ← label definitions
    06_hardware_routes.md     ← hardware route descriptions (new)
    08_refactor_plan.md       ← refactor history and plan
    09_running_the_study.md   ← full run instructions (new)

  tools/                      ← study-specific utilities
    smoke_test_study.py       ← workspace smoke test
    inventory_repo.py         ← print workspace inventory

  outputs/                    ← generated artifacts (not source)
    figures/  csv/  holograms/  manifests/  jupyter_runtime/

  archive/                    ← safely stored old files
    old_notebooks/            ← notebooks superseded by new structure
    old_source_copies/        ← stale/duplicate source files
    backups/                  ← 21 timestamped development snapshots

  reference_kernels/          ← historical provenance snapshots (not imported)
```

---

## Clean Install

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Or with the existing lab environment:

```powershell
C:\PhD\.venv2\Scripts\python.exe -m pip install -r requirements.txt
```

---

## Quick Smoke Test

```powershell
python Publication_Study\tools\smoke_test_study.py
```

Checks imports, path contract, canonical docs, and all notebook paths.
Does not run notebooks.

---

## Run Instructions (summary)

Full details: `docs/09_running_the_study.md`

```powershell
# List notebook order
python Publication_Study\run_study.py --list

# Preview a run
python Publication_Study\run_study.py --dry-run

# Run one stage
python Publication_Study\run_study.py --stage scalar
python Publication_Study\run_study.py --stage lab_realism
python Publication_Study\run_study.py --stage vector
python Publication_Study\run_study.py --stage materials
python Publication_Study\run_study.py --stage advanced

# Run full study
python Publication_Study\run_study.py --timeout-s 1800

# Run one notebook
python Publication_Study\run_study.py --only notebooks\scalar\04_scalar_parameter_sweeps.ipynb

# Run a slice
python Publication_Study\run_study.py `
    --start-at notebooks\scalar\02_scalar_ideal_vs_lab_diagnostics.ipynb `
    --stop-after notebooks\scalar\05_scalar_validation_suite.ipynb

# Clean and re-run
python Publication_Study\run_study.py --clean-output figures csv --stage scalar
```

---

## Output Locations

All generated files live under `Publication_Study/outputs/`.

| Folder | Contents |
|---|---|
| `outputs/figures/` | Plots (PNG, SVG, PDF) |
| `outputs/csv/` | Summary tables |
| `outputs/holograms/` | SLM phase pattern exports |
| `outputs/manifests/` | Run and artifact manifests (JSON) |
| `outputs/jupyter_runtime/` | Jupyter kernel runtime (auto-generated) |

Scientific outputs are never deleted unless `--clean-output` is passed.

---

## Physics and Model Limitations

- `docs/00_theory.md` — implemented optical model.
- `docs/01_conventions.md` — reported metrics and units.
- `docs/02_validation.md` — validation record.
- `docs/03_materials_application.md` — why material/fluence results are proxies.
- `docs/04_actual_lab_vector_case1.md` — why the current lab does not produce
  true radial/azimuthal vector beams.
- `docs/04_model_limitations.md` — scalar paraxial limits, ASM limits, SLM
  limits, interface-correction status, vector status.
- `docs/06_hardware_routes.md` — what each hardware route models and what it
  does not model.

---

## How To Add a New Stage

1. Add the notebook to the appropriate `notebooks/<topic>/` subfolder.
2. Add its relative path (e.g. `"notebooks/scalar/06_new_stage.ipynb"`) to
   `STAGE_NOTEBOOKS` in `run_study.py`.
3. Add it to `REQUIRED_NOTEBOOKS` in `vbb_study/setup_study.py` if it is
   part of the canonical reproducible sequence.
4. Write outputs under the named `paths["figures"]`, `paths["csv"]`, etc.
5. Use the taxonomy labels from `vbb_study/study_taxonomy.py`.
6. Update the relevant doc when the stage changes assumptions or hardware status.

---

## Backward Compatibility

Old command-line usage still works:

```powershell
python Publication_Study\run_publication_study.py --list
python Publication_Study\run_publication_study.py --dry-run
python Publication_Study\run_publication_study.py --only notebooks\scalar\04_scalar_parameter_sweeps.ipynb
```

`run_publication_study.py` is a thin wrapper that delegates to `run_study.py`.

Root-level compatibility shims (`bessel_twin_core.py` and `vbb_study/` at
`c:\PhD\Code\`) are **not changed**.  Older scripts that do
`import bessel_twin_core` or `from vbb_study import ...` continue to work.

---

## Common Failure Modes

| Symptom | Fix |
|---|---|
| `workspace validation failed` | Run `python tools\smoke_test_study.py`; verify notebook paths in `vbb_study/setup_study.py` |
| `Unknown notebook` in `--only` / `--start-at` | Run `--list`; note paths now include subdirectory prefix |
| Jupyter runtime permission error | `outputs/jupyter_runtime/` must be writable; runner sets `JUPYTER_RUNTIME_DIR` automatically |
| Missing Python packages | `pip install -r requirements.txt` |
| `Cannot find repo root` | Run from inside the repo; the anchor is `Publication_Study/bessel_twin_core.py` |
