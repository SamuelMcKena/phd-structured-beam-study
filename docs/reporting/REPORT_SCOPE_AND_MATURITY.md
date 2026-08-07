# Report Scope and Claim Maturity

## Report Order

1. Vortex-Bessel numerical and lab-realistic study: G0, B0, V1, V3.
2. Hexagonal-vector Bessel study: H1 and Nathan MODE 2 evidence.
3. Experimental calibration and validation study: measured hardware and measured-output comparison.

The first report may cite H1 only as an out-of-scope branch. It must not borrow H1 success to imply a
vortex result, or vice versa.

## Maturity Vocabulary

| Maturity | Meaning |
|---|---|
| `relative_optical_prediction` | Numerical field/metric under explicit model assumptions |
| `fixed_bench_nominal_prediction` | Canonical hardware binding with assumed or manufacturer values |
| `calibration_ready_prediction` | Software dependencies are explicit, but real measurements are incomplete |
| `calibrated_optical_prediction` | Real optical geometry/scale calibration is supplied |
| `calibrated_fluence_prediction` | Real energy, throughput and physical-area calibration are supplied |
| `experimentally_validated_prediction` | A real measured-output comparison passes accepted criteria |

Levels are not interchangeable. Synthetic calibration data is capped at calibration-ready.

## Supported for the Vortex Report

- Analytic G0/B0/V1/V3 definitions and controlled route taxonomy.
- Repaired vortex preservation and topological winding in simulation.
- Native numerical ring, dark-core, side-lobe, axial and convergence metrics with named planes.
- Fixed-bench nominal ideal/realistic/mild/degraded comparisons.
- Relative energy-ledger closure and single-count selected-order efficiency.
- Phase 2C scalar-versus-Debye and scalar-versus-vector-Fresnel bounded comparisons.
- Phase 2B 2D and transverse intensity-surface presentation with native-metric provenance.
- Phase 2E converged governed source-scale scalar B0/V1/V3 propagation, zone metrics, aperture
  sensitivity and sampling convergence.

## Authoritative Stages

| Stage | Outcome | Scope |
|---|---|---|
| Phase 1 | `PHASE1-B` | Corrected vortex preservation and mapping modes |
| Phase 1R | `PHASE1R-B` | Reconciled artifacts and convergence recovery |
| Phase 2A | `PHASE2A-B` | Canonical fixed-bench hardware binding and energy ledger |
| Phase 2B | `PHASE2B-A` | Native-metric visual diagnostics and transverse surfaces |
| Phase 2C | `PHASE2C-B` | Vector Debye and vector Fresnel objective/sample-scale benchmark |
| Phase 2E | `PHASE2E-FINAL-A` | Converged governed source-scale scalar B0/V1/V3 propagation |

Phase 2E scope is source-scale propagation only. It is **not** objective/sample-scale focusing and
**not** experimental validation. It does not alter the accepted Phase 2C focal contract.

## Scale Separation

Source-scale means tens of mm axial propagation with tens of um transverse Bessel/ring scale, and is
governed by Phase 2E. Objective/sample-scale means the Phase 2C Debye focal plane at approximately
micron transverse scale. The two must not be merged, and a number from one scale must never be
reported as belonging to the other.

## Narrowed Claims

- Scalar vortex morphology does not authorise quantitative peak location; use Debye.
- Ring-radius numbers must identify plane and scalar/vector reference.
- Old target-matched or unreconciled Stage 6 exports are diagnostic, not canonical fixed-bench data.
- Mild/degraded routes are controlled diagnostics, not measured bench states.
- The scalar interface result is bounded to the tested nominal interface and claim class.
- The 20-60 mm axial interval is a historical Phase 2A configuration reference only. It is not a
  measured Bessel zone and not the final source-scale axial prediction. Final measured source-scale
  zones come from Phase 2E.
- Hard 1.8 mm aperture beading is a diagnostic of hard truncation, not a nominal experimental
  prediction. The soft `exp[-(r/1.8 mm)^8]` route is an unmeasured sensitivity case.
- Phase 2B x-z/y-z panels remain accepted visual diagnostics but are superseded for final
  quantitative source-scale axial detail.

## Calibration Blockers

Real camera/objective scale, pulse energy, beam radius, SLM LUT/stroke/orientation, 4F geometry,
objective pupil/fill, relay magnification, axicon geometry, per-stage transmission, and material
interface state remain unresolved or not repository-verified. Therefore absolute dimensions,
absolute fluence, calibrated masks, and experimental agreement remain blocked.

### Reading route-level calibration flags

`outputs/validation/phase2e_final_propagation/source_scale_route_contract.json` records
`calibration_required = false` on the `nominal_no_additional_aperture` route. That flag has exactly
one meaning:

> no additional aperture calibration is required to define that numerical route.

It does **not** mean experimentally calibrated, absolute physical scale verified, bench validated, or
fluence calibrated. A route-level flag governs the numerical route definition only; it never
promotes a claim up the maturity ladder.

For the overall nominal fixed-bench prediction, `experimental_calibration_required = true` until real
measurements exist. Remaining blockers are beam radius; SLM phase LUT/stroke; exact 4F iris centre
and radius; physical stop/aperture presence; axicon centring and geometry; camera scale; z-stage
calibration; objective/relay calibration where relevant; and energy/transmission calibration for
fluence.

Phase 2E is therefore `experimental_validation = false` and remains at most a fixed-bench nominal
prediction.

## Excluded Physics

No nonlinear response, pulse accumulation, thermal response, damage threshold, ablation, or material
modification is predicted. A linear optical field or fluence proxy cannot be relabelled as material
outcome evidence.
