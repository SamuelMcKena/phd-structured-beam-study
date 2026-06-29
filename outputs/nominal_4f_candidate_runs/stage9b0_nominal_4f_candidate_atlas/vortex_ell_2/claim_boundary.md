# Stage 9B.0.1 Candidate Claim Boundary

Candidate: `vortex_ell_2`

Status labels:
- nominally_simulated
- command_masks_exportable_unvalidated
- not_bench_validated

This package is a nominal F300 4F forward-model diagnostic. It is not a
bench-calibrated physical 4F model, not a camera model, not an inverse
correction result, not an AI estimate, and not a material-response prediction.

Upstream route:
- SLM1 phase is applied at SLM1.
- SLM1-to-SLM2 propagation is included through the existing CSLM component route.
- The nominal F300 model starts from the field arriving at SLM2.

SLM role contract:
- SLM1 carries the flat/vortex/structured phase conditioning.
- SLM2 carries an ideal continuous carrier surrogate and later may carry correction maps.
- SLM2 does not contain an axicon phase.

Carrier boundary:
- carrier_realism = `ideal_continuous_phase_ramp`
- ideal_blazed_carrier_shift_surrogate = `True`
- pixelated_slm_diffraction_orders_modelled = `False`
- selected_order_purity_predicted = `False`

Stop sampling:
- stop_sampling_status = `convergence_verified`
- convergence_status = `passed_for_nominal_scenario`
- ranking_allowed = `True`

Readiness:
- physical_4f_readiness = `blocked`
- carrier_coordinate_status = `nominal_model_not_bench_calibrated`
- final_export_allowed = `False`
