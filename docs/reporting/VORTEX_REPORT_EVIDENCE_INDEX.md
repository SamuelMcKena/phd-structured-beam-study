# Vortex Report Evidence Index

This index is the report-first route through G0, B0, V1 and V3. The authoritative row-level mapping
is `VORTEX_CLAIM_TO_EVIDENCE.csv`; the current frozen copy and file hashes are under
`outputs/validation/report_freeze_v2/` (`vortex_bessel_report_final_v2_governance`).

`outputs/validation/report_freeze/` is retained unchanged as `historical_vortex_report_freeze_v1`.
It is immutable provenance describing the pre-Phase-2E report state and must not be cited as the
current evidence set.

## Scale Separation

Two different optical scales appear in this repository and must never be merged:

| Scale | Meaning | Authoritative phase |
|---|---|---|
| Source-scale | Tens of mm axial propagation; tens of um transverse Bessel/ring scale | Phase 2E |
| Objective/sample-scale | Debye focal plane; approximately micron transverse scale | Phase 2C |

A source-scale zone length or ring radius is not an objective focal number, and a Phase 2C focal
radius is not a source-scale feature. Quoting one as the other is a reporting error.

## Case Definitions

| Case | Definition | Primary source | Primary data |
|---|---|---|---|
| G0 | Charge-zero Gaussian control | `vbb_study/digital_twin/phase2a_canonical.py` | `outputs/validation/phase2a/canonical_case_summary.csv` |
| B0 | Charge-zero bright-core Bessel reference | same | same |
| V1 | Charge-one vortex Bessel | same plus Phase 2C winding metric | `outputs/validation/phase2c/phase2c_objective_benchmark.csv` |
| V3 | Charge-three vortex Bessel | same plus Phase 2C winding metric | same |

Analytic scalar definitions and metric conventions are in `vbb_study/equations/scalar_bessel.py`,
`vbb_study/equations/fields.py`, and `vbb_study/equations/metrics.py`.

## Target and Physical Routes

Phase 2A defines the analytic target control, ideal optical route, realistic fixed-bench route,
mild-error diagnostic, and deliberately degraded diagnostic. The report must retain those exact
labels. The repaired physical-axicon vortex contract is in `vbb_study/vbb_axicon.py`; the holographic
phase route is in `vbb_study/equations/holography.py`.

Older Stage 6 phase-mask and charge-sweep artifacts remain useful diagnostics, but they predate the
full Phase 1/1R mapping reconciliation. They are not the primary quantitative fixed-bench evidence.
No preview PNG is a calibrated SLM mask.

## Propagation and Topology

- Scalar propagation and the 5% plane-power gate: Phase 1 and Phase 2A.
- Physical-route vortex preservation: Phase 1/1R plus the two `test_slm2_preserve_vortex*` suites.
- V1/V3 winding, ring radius, dark-core radius, side lobes and focal comparison at the
  objective/sample-scale: Phase 2C objective benchmark.
- Final quantitative source-scale axial behaviour, Bessel-zone metrics and aperture sensitivity:
  Phase 2E (see below).
- Phase 2A configured axial intervals and Phase 2B x-z/y-z figures: historical context and accepted
  visual diagnostics; they are no longer the final quantitative source-scale axial evidence.
- Debye quadrature and canonical-grid convergence: Phase 2C convergence outputs.

## Phase 2E - Authoritative Source-Scale Evidence

Phase 2E (`PHASE2E-FINAL-A`) is the authoritative source for:

- final B0/V1/V3 source-scale x-z and y-z propagation;
- final source-scale feature and ring evolution;
- final source-scale Bessel-zone metrics;
- final source-scale aperture sensitivity;
- final source-scale 3D transverse surfaces;
- source-sampling convergence.

The governed route is SLM phase-only modulation, common 4F selected-order filtering, carrier removal
and reconstructed field, no additional real-space aperture, one physical axicon, then band-limited
angular-spectrum propagation in air. Production uses N=3072 over a 10 mm window
(dx=3.255208 um, 19.7646 samples per axicon radial phase period) with dz=0.25 mm over z=0-180 mm.

| Case | Measured FWHM zone (mm) | Strict useful region (mm) | Reference radius (um) | Median width (um) | Median dark core (um) | Winding |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 20.00-120.00 | 20.00-120.00 | 11.257 | 22.513 | n/a | 0 |
| V1 | 17.25-119.25 | 17.25-119.25 | 18.671 | 19.496 | 8.943 | 1 |
| V3 | 20.25-119.50 | 20.50-119.25 | 42.882 | 23.701 | 30.305 | 3 |

Three route identities are preserved. `nominal_no_additional_aperture` is the report primary,
`soft_aperture_sensitivity` is an unmeasured sensitivity case, and `hard_aperture_diagnostic` is
diagnostic only. Pronounced axial beading survives only under the hard 1.8 mm disk and is therefore a
hard-truncation diagnostic, not a nominal experimental prediction.

Governed tables are under `outputs/validation/phase2e_final_propagation/`; the 18-pair figure pack
and its manifests are under `outputs/figures/phase2e_final_source_propagation/`. The narrative record
is `docs/95_phase2e_final_source_scale_bessel_propagation.md`, with the earlier forensic repair in
`docs/95_phase2e_propagation_forensic_repair.md`.

Phase 2E closes source-scale propagation only. It does not alter the Phase 2C objective/sample-scale
contract and makes no objective-focused claim.

## Phase 2B Status After Phase 2E

Phase 2B is not withdrawn and is not globally wrong. Its accepted visualisation contracts stand:
native-metric rendering, display-only interpolation, transverse evolution, profile panels and
transverse intensity surfaces continue to support their own claims.

Phase 2B `03_xz_yz_slices/` source-scale axial plots are now classified as historical and earlier
accepted visual diagnostics. They are superseded for final quantitative source-scale axial detail,
which comes from Phase 2E.

## Phase 2C Status After Phase 2E

Phase 2C remains authoritative and unchanged for:

- vector Debye focal results;
- scalar versus vector focal comparison;
- the longitudinal field;
- the vector Fresnel interface;
- quantitative focal peak-location claims.

Phase 2E adds no objective transform and does not supersede any of these.

Winding is a complex-field phase measurement on the identified dominant transverse component. An
annular intensity or dark core alone is not evidence of topological charge.

## Lab Realism and Energy

The canonical hardware definition is
`outputs/validation/phase2a/canonical_hardware_manifest.json`. SLM models, selected-order efficiency,
pupil capture and the unified power ledger are governed by Phase 2A. Each filtered route contains one
`simulated_selected_first_order` factor. Absolute pulse energy and measured throughput remain blocked.

## Scalar Versus Vector Reference

Phase 2C supports scalar broad morphology for G0/B0. V1/V3 preserve charge and ring radius across the
comparison, but quantitative peak-location and component claims use vector Debye. The bounded scalar
normal-incidence interface comparison is acceptable only in its tested scope; component-resolved or
changed-interface work uses vector Fresnel.

## Figure Families

Primary source-scale family (Phase 2E, `outputs/figures/phase2e_final_source_propagation/`):

- `01_primary_propagation/`: final B0/V1/V3 source-scale x-z and y-z propagation.
- `02_aperture_comparison/`: matched nominal, soft and hard route comparison.
- `03_sampling_convergence/`: source-sampling convergence.
- `04_transverse_snapshots/`: final source-scale transverse evolution.
- `05_profiles_and_metrics/`: final source-scale feature/ring evolution and zone metrics.
- `06_3d_surfaces/`: final source-scale 3D transverse intensity surfaces.
- `07_report_hero_figures/`: composed report heroes.

Objective/sample-scale family (Phase 2C):

- Phase 2C `objective/`: scalar FFT versus vector Debye.
- Phase 2C `interface/`: scalar versus spectral vector Fresnel.
- Phase 2C `profiles/` and `components/`: solver-specific focal metrics and components.

Supporting and historical families (Phase 2B):

- Phase 2B `02_xy_planes/`: transverse evolution and SAS focus views.
- Phase 2B `03_xz_yz_slices/`: historical source-scale axial context and plane-power diagnostics;
  superseded for final quantitative source-scale axial detail by Phase 2E.
- Phase 2B `04_profiles/`: native radial/angular/centre-line profiles.
- Phase 2B `05_3d_maps/`: fixed transverse-plane intensity surfaces for B0, V1 and V3.

Interpolation in these figures is display-only; report metrics come from native arrays. See
`VORTEX_FIGURE_AND_TABLE_PLAN.md` for main/supplementary/diagnostic classifications.

## Limitations

Absolute dimensions, absolute fluence, LUT-calibrated masks, and experimental agreement require real
calibration. The current repository makes no nonlinear material-modification, accumulation, thermal,
or ablation prediction. Software tests establish implementation and numerical contracts only.

### Reading the Phase 2E calibration flags

`outputs/validation/phase2e_final_propagation/source_scale_route_contract.json` records
`calibration_required = false` on `nominal_no_additional_aperture`. That flag means one thing only:

> no additional aperture calibration is required to define that numerical route.

It does **not** mean any of the following:

- experimentally calibrated;
- absolute physical scale verified;
- bench validated;
- fluence calibrated.

For the overall nominal fixed-bench prediction, `experimental_calibration_required = true` until real
measurements exist. Remaining blockers are beam radius; SLM phase LUT/stroke; exact 4F iris centre
and radius; physical stop/aperture presence; axicon centring and geometry; camera scale; z-stage
calibration; objective/relay calibration where relevant; and energy/transmission calibration for
fluence. Phase 2E is `experimental_validation = false`.
