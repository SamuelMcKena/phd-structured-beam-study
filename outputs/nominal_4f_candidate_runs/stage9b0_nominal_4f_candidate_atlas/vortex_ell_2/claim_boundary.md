# Stage 9B.0 Candidate Claim Boundary

Candidate: `vortex_ell_2`

Status labels:
- nominally_simulated
- hardware_command_exportable
- not_bench_validated

This package is a nominal F300 4F forward-model diagnostic. It is not a
bench-calibrated physical 4F model, not a camera model, not an inverse
correction result, not an AI estimate, and not a material-response prediction.

SLM role contract:
- SLM1 carries the flat/vortex/structured phase conditioning.
- SLM2 carries the command-domain carrier and later may carry correction maps.
- SLM2 does not contain an axicon phase.

Readiness:
- physical_4f_readiness = `blocked`
- carrier_coordinate_status = `nominal_model_not_bench_calibrated`
- final_export_allowed = `False`
