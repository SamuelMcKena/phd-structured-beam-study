# Stage 9B.0.1 Upstream Bridge And Stop-Sampling Summary

Starting checkpoint: `1591ebe` (`Stage 9B.0 nominal F300 4F virtual bench and candidate atlas`).

Stage 9B.0.1 corrects the nominal F300 atlas semantics before any bench use.
It remains a nominal, unvalidated forward-model scenario.

## Corrected

- `run_nominal_f300_4f` now starts from `field_arriving_at_slm2` or an
  `UpstreamSLM2FieldBridge`.
- Passing `slm1_phase_rad` directly to the F300 model raises an error.
- Candidate simulations use the existing CSLM component route:
  `SLM1_phase_plane -> SLM1_to_SLM2_segment -> field_arriving_at_SLM2`.
- SLM2 carrier handling is labelled as an ideal continuous-ramp/blazed-shift
  surrogate, not pixelated-SLM diffraction-order physics.
- Candidate package status is now `command_masks_exportable_unvalidated`.
- The initial shortlist is limited to `gaussian_reference` and
  `vortex_ell_1` through `vortex_ell_4`.
- Stop-radius/offset ranking is gated by sampling and convergence checks.

## Stop Sampling

```text
exploratory profile: 128 grid, stop diameter 5.76 px, exploratory_only
standard profile:    256 grid, stop diameter 11.52 px, ranking_eligible
ranking rule:        standard profile + passed convergence
```

Convergence thresholds:

```text
energy relative difference <= 0.35
centroid difference <= 250 um
second-moment width relative difference <= 0.35
normalised intensity correlation >= 0.85
```

All five shortlist candidates pass the Stage 9B.0.1 nominal convergence gate.

## Ranking Eligible

| rank | candidate | nominal score |
|---:|---|---:|
| 1 | `gaussian_reference` | 0.992 |
| 2 | `vortex_ell_1` | 0.796 |
| 3 | `vortex_ell_2` | 0.486 |
| 4 | `vortex_ell_3` | 0.387 |
| 5 | `vortex_ell_4` | 0.351 |

## Figures

```text
outputs/figures/digital_twin/stage9b0_1_upstream_cslm_to_nominal_4f_chain.png
outputs/figures/digital_twin/stage9b0_1_stop_sampling_convergence.png
outputs/figures/digital_twin/stage9b0_1_candidate_ranking_validity.png
```

## Still Unsupported

- physical 4F readiness/calibration
- pixelated-SLM diffraction orders, fill-factor effects, zero-order leakage, or
  measured selected-order purity
- camera model or camera-to-relay-plane calibration
- inverse correction, AI, material response, or process prediction
- treating nominal candidate ranking as bench validation
