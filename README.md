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
| Phase 2E | `PHASE2E-FINAL-A` | Converged governed source-scale scalar B0/V1/V3 propagation, zone metrics, aperture sensitivity and sampling convergence |

Phase 2D adds solver governance and a calibration bridge. It does not alter the accepted Phase 2C
physics and does not unlock real calibrated dimensions or fluence without laboratory measurements.

Phase 2E is the authoritative source for final quantitative source-scale axial results. Its scope is
converged governed source-scale scalar B0/V1/V3 propagation. It is **not** objective/sample-scale
focusing and **not** experimental validation, and it does not alter the accepted Phase 2C focal
contract. Phase 2B x-z/y-z panels remain accepted visual diagnostics but no longer supply the final
source-scale quantitative axial result.

## Two Optical Scales

| Scale | Meaning | Authoritative phase |
|---|---|---|
| Source-scale | Tens of mm axial propagation; tens of um transverse Bessel/ring scale | Phase 2E |
| Objective/sample-scale | Debye focal plane; approximately micron transverse scale | Phase 2C |

Numbers from the two scales are not interchangeable.

The final Phase 2E source-scale route is SLM phase-only modulation, common 4F selected-order
filtering, carrier removal and reconstructed field, no additional real-space aperture, one physical
axicon, then band-limited angular-spectrum propagation in air. Production uses N=3072 on a 10 mm
window with dz=0.25 mm over z=0-180 mm. The `nominal_no_additional_aperture` route is the report
primary; the soft route is an unmeasured sensitivity case and the hard 1.8 mm route is diagnostic
only.

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
final governed source-scale axial zones and metrics from Phase 2E; numerical convergence;
SLM/filtering semantics; relative energy ledgers; and bounded scalar-versus-vector
objective/interface comparisons. The exact claim-to-evidence mapping is in
`docs/reporting/VORTEX_CLAIM_TO_EVIDENCE.csv`.

The 20-60 mm axial interval that appears in historical Phase 2A configuration is a configured
reference only. It is not a measured Bessel zone and not the final source-scale axial prediction.

## Calibration-Required Claims

Absolute sample dimensions, absolute focal fluence, hardware-calibrated SLM phase fidelity, measured
camera/objective scale, measured per-stage transmission, and experimental agreement remain blocked.
The current material model is linear optical propagation only. Nonlinear response, pulse
accumulation, thermal response, ablation, and material modification are not predicted.

A route-level `calibration_required = false` flag, such as the one on the Phase 2E
`nominal_no_additional_aperture` route, means only that no additional aperture calibration is needed
to define that numerical route. It does not mean experimentally calibrated, absolute physical scale
verified, bench validated, or fluence calibrated. For the overall nominal fixed-bench prediction
`experimental_calibration_required = true`, with outstanding blockers including beam radius, SLM
phase LUT/stroke, exact 4F iris centre and radius, physical stop/aperture presence, axicon centring
and geometry, camera scale, z-stage calibration, objective/relay calibration where relevant, and
energy/transmission calibration for fluence.

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
C:\PhD\.venv2\Scripts\python.exe -m pytest tests/test_phase2e_source_sampling_repair.py -q
C:\PhD\.venv2\Scripts\python.exe -m pytest tests/test_phase2e_final_source_propagation.py -q
C:\PhD\.venv2\Scripts\python.exe -m pytest tests/test_vortex_report_freeze_v2.py -q
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

The current vortex evidence freeze is `vortex_bessel_report_final_v2_governance` under
`outputs/validation/report_freeze_v2/`. It records selected evidence paths and SHA-256 hashes without
copying large artifacts, and it includes the final Phase 2E source-scale evidence and figure pack.
Build it with `tools/build_vortex_report_freeze_v2.py`, which only reads and hashes existing files.

`outputs/validation/report_freeze/` is the immutable `historical_vortex_report_freeze_v1` snapshot of
the pre-Phase-2E report state. It is retained for provenance and is not the current evidence set. Its
recorded hashes describe the governance documents as they stood at v1 and therefore no longer match
the current working copies of those documents.

The standalone export should omit temporary directories, duplicate root notebooks, bulk ZIP archives,
and files over GitHub's size limit.
