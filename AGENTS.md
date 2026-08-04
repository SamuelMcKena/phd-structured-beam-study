# Instructions for Future Coding Agents

This repository is report evidence. Preserve scientific scope and provenance before optimising
convenience.

## Contracts That Must Not Be Weakened

1. Nonzero physical-axicon vortex cases use vortex-preserving SLM2 correction by default.
   `full` conjugation may remove winding only when `allow_vortex_removal=True` identifies an
   intentional diagnostic.
2. Report-facing hardware predictions use `mapping_mode="fixed_physical_optics"`. Historical
   `target_matched_inverse_design` outputs may be retained but cannot be relabelled as fixed-bench.
3. Quantitative propagation requires `propagation_power_drift_fraction <= 0.05`. Missing drift is
   not zero, and expected physical/filter loss is not numerical drift.
4. Selected first-order efficiency is counted exactly once. Never multiply a configured efficiency
   onto a field/ledger that already contains `simulated_selected_first_order`.
5. Accepted Phase 1/1R/2A/2B/2C outputs are immutable evidence. Do not overwrite or regenerate them
   silently. New presentation-only derivatives must identify their accepted source and must not feed
   metrics.

## Solver Governance

- Scalar FFT/ASM is the fast screening route for broad transverse morphology.
- Vector Debye is the quantitative reference for vortex peak location, longitudinal field,
  polarisation components, and any H1 focal-detail claim governed by `vbb_study/solver_policy.py`.
- V1/V3 ring radii must state whether the scalar or vector focal reference supplies the value.
- Scalar normal-incidence Fresnel is allowed only for the bounded morphology/power comparison already
  validated in Phase 2C. Component-resolved or high-angle interface analysis uses vector Fresnel.
- Never silently select a lower-fidelity solver than the claim policy permits.

## Vortex Preservation

- Check requested and measured winding for V1 and V3.
- A dark core or annular intensity alone is not proof of topological charge.
- Phase winding must be evaluated on an identified complex transverse component and contour.
- Keep analytic target, ideal route, realistic fixed-bench route, mild-error diagnostic, and degraded
  diagnostic labels distinct.

## Claims and Calibration

- Numerical prediction, fixed-bench nominal prediction, calibrated prediction, and experimentally
  validated prediction are different maturity levels.
- A populated synthetic bundle remains `synthetic_not_experimental`.
- Do not claim experimental validation without real measured-output evidence and an accepted
  comparison record.
- Absolute dimensions and fluence remain calibration-required until camera/objective scale, energy,
  transmissions, and other dependencies are measured.
- Do not add nonlinear material response, thermal accumulation, ablation, or modification claims
  without a separately validated model and evidence phase.

## Repository Hygiene

- Do not commit root ZIP archives, pytest/temp trees, `vbb_study - Copy/`, root duplicate notebooks,
  caches, or files above GitHub's hard size limit.
- Generated figures under ignored output paths must be force-added only when listed in the report
  freeze manifest.
- Do not delete historical/superseded files merely because they are old. Classify and exclude them
  from canonical evidence instead.
- Run the report-focused tests, authoritative collection, compileall, and `git diff --check` before a
  report freeze is accepted.
- Do not commit unless the user explicitly requests it.
