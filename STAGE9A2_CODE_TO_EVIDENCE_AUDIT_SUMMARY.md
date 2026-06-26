# Stage 9A.2 / 9A.3 Evidence Audit Summary

Starting checkpoint for Stage 9A.3: Stage 9A.2 code evidence audit and research
backlog (`6bbc210`).

Stage 9A.3 integrates the supplied verified seed bibliography and adds explicit
multi-layer evidence records. No optical propagation, physical 4F, camera model,
inverse correction, neural network, or material-response physics is implemented.

## Verified Bibliography

- `references/structured_beam_methods.bib`
- Verified entries: 8
- Source seed: `references/incoming/structured_beam_methods_verified_seed.bib`

## Claims Linked To Verified References

- `angular_spectrum_or_bl_asm_propagation`
- `finite_sampling_and_aliasing_control`
- `phase_only_slm_mask_generation`
- `phase_quantisation_and_grayscale_export`
- `command_domain_carrier_grating`
- `pixelated_slm_zero_order_and_unwanted_orders`
- `multi_plane_phase_retrieval_future`
- `effective_aberration_inference_future`
- `zernike_or_phase_conjugate_correction_future`
- `fused_silica_bessel_channel_or_tgv_future`
- `fused_silica_welding_future`

## Reclassified To Avoid Overstating Physical Validation

- `phase_only_slm_mask_generation`
- `phase_quantisation_and_grayscale_export`
- `command_domain_carrier_grating`
- `pixelated_slm_zero_order_and_unwanted_orders`

## Fused-Silica Split

- `fused_silica_bessel_channel_or_tgv_future`
- `fused_silica_waveguide_future`
- `fused_silica_welding_future`

The broad `fused_silica_application_boundary` is retained only as
`superseded_by_specific_application_claims`.

## Backlog And Search Plan

- Backlog items: 30
- Literature/search-plan entries: 19

## Immediate Lab Action

Run the first Fourier-plane carrier calibration session from Stage 9A.1:
record actual SLM/camera/lens/stop/axicon identifiers, capture dark and flat
references, then measure SLM2 command-domain carrier cycles versus observed
Fourier-plane order position without changing the bench mid-run.
