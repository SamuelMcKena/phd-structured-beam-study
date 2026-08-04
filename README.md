# Vortex-Bessel Numerical and Lab-Realistic Study

`Publication_Study` is the evidence repository for a structured-beam programme built around a
PHAROS-class ultrafast laser, dual phase-only SLM routing, Bessel and vortex-Bessel propagation,
vector focusing, and later experimental calibration. The first report supported by this repository
is the vortex-Bessel numerical and lab-realistic study. Hexagonal-vector and experimental-validation
reports remain separate follow-on products.

This repository separates analytic targets, nominal fixed-bench predictions, diagnostics, and
calibration-dependent claims. Passing software tests establishes numerical/software contracts; it
does not by itself establish experimental validity.

## Canonical Cases

| ID | Meaning | Canonical role |
|---|---|---|
| `G0` | Charge-zero Gaussian control | Propagation, scale, and focusing control |
| `B0` | Charge-zero bright-core Bessel beam | Scalar Bessel reference |
| `V1` | Charge-one vortex Bessel beam | Primary vortex case |
| `V3` | Charge-three vortex Bessel beam | Higher-charge vortex comparison |
| `H1` | Continuous vector hexagonal field | Separate hexagonal-vector report branch |

G0, B0, V1, and V3 are the scope of the first report. H1 remains in the code and regression suite but
is not used to turn the vortex report into a hexagonal-beam report.

## Authoritative Physics Stages

| Stage | Outcome | Report meaning |
|---|---|---|
| Phase 1 | `PHASE1-B` | Corrected vortex preservation, wavelength-bearing Fourier geometry, 5% power gate, and explicit mapping modes |
| Phase 1R | `PHASE1R-B` | Reconciled affected artifacts; recovered converged rows and retained blocked historical diagnostics |
| Phase 2A | `PHASE2A-B` | Canonical `fixed_physical_optics` hardware binding, five controlled route variants, and unified energy ledger; absolute claims remain calibration-limited |
| Phase 2B | `PHASE2B-A` | Publication-resolution native diagnostics and transverse intensity surfaces with interpolation used only for display |
| Phase 2C | `PHASE2C-B` | Independent vector Debye and vector Fresnel benchmark; scalar morphology is bounded, while vortex peak detail and components use the vector reference |

Phase 2D adds solver governance and a calibration bridge. It does not alter the accepted Phase 2C
physics and does not unlock real calibrated dimensions or fluence without laboratory measurements.

## Validated Contracts

- Physical-axicon vortex routing preserves the requested winding by default. Full conjugation that
  removes a nonzero vortex requires an explicit diagnostic acknowledgement.
- Report-facing hardware predictions use `fixed_physical_optics`; historical target-matched inverse
  design must remain labelled as such.
- Quantitative propagation is blocked when numerical plane-power drift exceeds `0.05`.
- Selected first-order efficiency appears exactly once in each energy ledger.
- Scalar FFT remains a fast screening route. Quantitative vortex peak location, longitudinal field,
  and component claims use the vector Debye reference.
- Accepted numerical outputs are not silently overwritten.

## Supported Numerical Claims

The repository supports analytic/control definitions; fixed-bench ideal, realistic, mild-error and
degraded route comparisons; topology/winding checks; native ring, dark-core and side-lobe metrics;
nominal axial behaviour; numerical convergence; SLM/filtering semantics; relative energy ledgers;
and bounded scalar-versus-vector objective/interface comparisons. The exact claim-to-evidence mapping
is in `docs/reporting/VORTEX_CLAIM_TO_EVIDENCE.csv`.

## Calibration-Required Claims

Absolute sample dimensions, absolute focal fluence, hardware-calibrated SLM phase fidelity, measured
camera/objective scale, measured per-stage transmission, and experimental agreement remain blocked.
The current material model is linear optical propagation only. Nonlinear response, pulse
accumulation, thermal response, ablation, and material modification are not predicted.

## Repository Map

- `vbb_study/`: active physics and digital-twin source
- `tests/`: numerical, governance, and non-regression tests
- `tools/`: runners, validators, and audit utilities
- `outputs/validation/`: governed machine-readable results and freeze manifests
- `outputs/figures/`: generated figures; report-selected ignored figures require explicit force-add
- `calibration/templates/`: editable measurement templates, not laboratory data
- `docs/reporting/`: report scope, evidence, figure plan, and reproducibility record
- `archive/` and root-level legacy notebooks/ZIPs: historical or duplicate material, not canonical

See `docs/reporting/REPOSITORY_MAP.md` for the complete audit and
`docs/reporting/VORTEX_REPORT_EVIDENCE_INDEX.md` for report navigation.

## Environment

Python `3.13` is the supported interpreter family. Install the report environment with:

```powershell
C:\PhD\.venv2\Scripts\python.exe -m pip install -r requirements-report.txt
```

Mandatory and optional packages are separated in `requirements-report.txt`. Installed package and
Python versions are also recorded in the freeze manifest.

## Tests

Focused report checks:

```powershell
C:\PhD\.venv2\Scripts\python.exe -m pytest tests/test_phase1_critical_physics_repairs.py -q
C:\PhD\.venv2\Scripts\python.exe -m pytest tests/test_phase1r_reconciliation.py -q
C:\PhD\.venv2\Scripts\python.exe -m pytest tests/test_phase2a_canonical_lab_realism.py -q
C:\PhD\.venv2\Scripts\python.exe -m pytest tests/test_phase2b_visual_diagnostics.py -q
C:\PhD\.venv2\Scripts\python.exe -m pytest tests/test_phase2c_objective_interface.py -q
C:\PhD\.venv2\Scripts\python.exe -m pytest tests/test_slm2_preserve_vortex.py tests/test_slm2_preserve_vortex_end_to_end.py -q
```

Collection and compilation:

```powershell
C:\PhD\.venv2\Scripts\python.exe -m pytest --collect-only tests -q
C:\PhD\.venv2\Scripts\python.exe -m compileall -q vbb_study tools tests
git diff --check
```

Tests verify software and numerical contracts only. They do not replace measurement or calibration.

## Report Freeze

The current vortex evidence freeze is under `outputs/validation/report_freeze/`. It records selected
evidence paths and SHA-256 hashes without copying large artifacts. The standalone export should omit
temporary directories, duplicate root notebooks, bulk ZIP archives, and files over GitHub's size
limit.
