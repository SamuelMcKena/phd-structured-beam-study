# Stage 9A.2 Code-to-Evidence Audit Summary

Starting checkpoint: Stage 9A.1 first Fourier carrier calibration session pack (`51071bc`).

Stage 9A.2 creates an evidence-aware project map and research backlog.  No new
optical propagation, calibration fitting, correction, AI, camera model, or
material-response physics is implemented.

## Created

- `docs/44_code_to_evidence_audit.md`
- `configs/evidence/project_claim_registry.json`
- `configs/evidence/research_backlog.json`
- `configs/evidence/literature_search_plan.json`
- `configs/evidence/manufacturer_evidence_register.json`
- `configs/evidence/bench_evidence_register.json`
- `configs/materials/fused_silica_evidence_template.json`
- `references/README.md`
- `references/structured_beam_methods.bib`
- `outputs/figures/digital_twin/stage9a2_code_to_evidence_roadmap.png`
- `tests/test_stage9a2_code_to_evidence_audit.py`

## Claim Counts

- Canonical active claims: 8
- Placeholder/assumption claims: 5
- Manufacturer-data blockers: 10
- Bench-data blockers: 21
- Literature/source blockers: 8

## Backlog Counts

- P0: 8
- P1: 6
- P2: 7
- P3 including legacy optional branches: 9

## Immediate Lab Action

Run the first Fourier-plane carrier calibration session from Stage 9A.1:
record actual SLM/camera/lens/stop/axicon identifiers, capture dark and flat
references, then measure SLM2 command-domain carrier cycles versus observed
Fourier-plane order position without changing the bench mid-run.
