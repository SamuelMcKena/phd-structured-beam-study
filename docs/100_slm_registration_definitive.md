# Audited SLM registration study

This study supersedes quantitative claims from docs 97–99 and the old sampled
4F report route. Both panels retain 20-pixel blazes. The historical route now
records nominal free-space numerical power loss and rejects its quantitative
eligibility when any step exceeds 5%; audited losses reached 9–55%.

The independent reference integrates actual pixel-grid intersections, commands
phase at pixel centres, and applies ideal 4F filtering in **source** frequency
space. It has no measured inter-panel transfer. Flat correction is explicitly
the **FLAT-CORRECTION PHYSICS-ISOLATION BASELINE**. Swapping equivalent A/B
physical roles is exactly symmetric, not an architecture contest.

## Primary iris-adjusted experiment

The iris is adjusted for each L,w case to
`B = max(2500, 5 L / (2 pi w))` cycles/m and then held fixed for every
registration/architecture in that case. This prevents the narrow fixed iris
from rejecting the high-charge small-beam vortex before the question is tested.
Wide apertures are ideal selected-channel diagnostics; real zero/unwanted
orders and dead-space leakage remain uncalibrated and may be admitted in a lab.

The fixed 2500 cycles/m study is retained separately as a filtering control.
Do not combine rows from different iris policies into one tolerance claim.

## Reproduction

Python 3.12 or later, NumPy, SciPy, pandas, Matplotlib, pytest, python-docx,
Pandoc. LibreOffice is required for PDF export. The repository's full numerical
requirements can also be installed; package-install metadata targets the
original project environment, while these standalone scripts use PYTHONPATH.

From the repository root:

```bash
export PYTHONPATH=.
export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2
python -m pytest -q tests/test_slm_pixel_registration.py tests/test_slm_registration_architectures.py tests/test_registration_reference.py
python tools/run_registration_reference_study.py
python tools/run_registration_extra_controls.py
python tools/complete_registration_diagnostics.py
python tools/plot_registration_reference_study.py
python tools/complete_registration_diagnostics.py
python tools/build_registration_reference_report.py
```

The core runner can be staged with `--stages sweep,unit_cell,roles,convergence`
and then `--stages fields`. Do not run both stages concurrently against the
same destination when producing final manifests. The full runner is sequential
and caches selected coefficients within each invocation.

Primary outputs:

* `outputs/validation/registration_iris_adjusted/`: CSV, NPZ, manifest, controls.
* `outputs/figures/registration_definitive/`: standalone PNG and vector-container
  PDF figures; four presentation figures use the same source fields.
* `outputs/reports/registration_definitive/`: DOCX, Markdown and report audit.

Portable PDF export after creating the DOCX:

```bash
soffice -env:UserInstallation=file:///tmp/registration-lo --headless --convert-to pdf --outdir outputs/reports/registration_definitive outputs/reports/registration_definitive/SLM_Pixel_Registration_Report.docx
```

The delivered report is rendered and visually inspected; a successful export
alone does not constitute that review. Equations are native editable Word math.

Fixed-iris and historical forensic controls:

```bash
python tools/run_registration_reference_study.py --iris-policy fixed --out outputs/validation/registration_definitive
python tools/run_registration_extra_controls.py --out outputs/validation/registration_definitive
python tools/complete_registration_diagnostics.py --data outputs/validation/registration_definitive
python tools/audit_registration_legacy.py
```

The extra-controls severe 25-pixel case intentionally uses its adjusted iris
even when stored alongside the fixed-iris controls; it records its separate
bandwidth and scope. Do not relabel it as a fixed-iris result.

## Interpretation and unresolved measurements

* Pure registration: b=h fixed; lattice origin g moves modulo p. Diagonal offsets
  move both coordinates by the stated fraction, not the same distance as x-only.
* Local raw field change is larger than the selected or propagated intensity
  change. Report throughput and power-normalised morphology separately.
* Centroid registration is a specified complex-field translation, not an
  overlap optimisation; its infidelity can exceed raw infidelity.
* Every propagated image is at z>0. A phase-only axicon changes no immediate
  intensity. Absolute cone dimensions are conditional on the assumed geometry.
* The user-reported lens separation is 300 mm. Individual focal lengths,
  magnification and iris geometry are not thereby known.
* The 20-degree optic is a conditional base-angle cone calculation. Exact part
  number, orientation, index and cone angle need confirmation. A fitted cone
  control is explicitly model-inferred, not independent hardware calibration.
* The demo inter-SLM distance is a placeholder. Actual transfer, parity,
  rotation, magnification, carrier conventions, panel LUTs and a trusted
  correction map are missing. Candidate q20 retrieval bundles remain
  uncalibrated; correction-ON is not fabricated.

Numeric file hashes and source hashes are in the manifests. Large generated
evidence is ignored in Git and uploaded as Actions artifacts. Previous source
and evidence are preserved. The report tables and figures read CSV/NPZ values;
neither screenshots nor rendered figures supply metrics.
