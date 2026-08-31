# Real q=20 BMG digital-twin correction summary

All corrected fields below are **model predictions**, not post-correction camera measurements.

- Raw input: 72 BMG files (18 z planes x 4 repeats).
- Physical z convention: hexapod -11...+6 mm maps to relative -17...0 mm.
- Effective k_perp: 482741.4 1/m.
- Effective repository-convention base angle from k_perp: 9.738 deg; manufacturer '20 deg' convention remains unverified.
- Measured beam-path slope: 5.417 mrad.
- Miao branch status: GLOBAL_K_PERP_PATH_UNRELIABLE; hardware ready: False.
- Full-route held-out status: SCREENING_PREDICTION_ONLY.
- Final full-route decision: MODEL_SCREENING_SUPPORTED_HARDWARE_BLOCKED.
- SLM2 output is a numerical-grid phase proposal only; native mapping/LUT export is blocked.

- Nominal morphology gate: True (mean radial r=0.913, median ring-radius error=1.50 um).
- Axicon-grid convergence claim allowed: True.

## Mean/median finite-energy-target metrics

| method | mean_pearson_r | median_pearson_r | mean_nrmse | median_nrmse | median_ring_cv | maximum_dark_core_ratio |
| --- | --- | --- | --- | --- | --- | --- |
| Calibrated physical baseline | 0.88316 | 0.98315 | 0.041002 | 0.033371 | 0.16676 | 0.022653 |
| Complete digital-twin correction prediction | 0.88316 | 0.98315 | 0.041002 | 0.033371 | 0.16676 | 0.022653 |
| Ideal finite-energy target | 1 | 1 | 0 | 0 | 0.1595 | 0.071491 |
| Measured BMG | 0.33337 | 0.29744 | 0.15768 | 0.15612 | 0.33768 | 0.13457 |
| Miao-only corrected (same twin) | 0.97857 | 0.98947 | 0.019696 | 0.016802 | 0.16501 | 0.098474 |
| Physical fit + Miao corrected | 0.86964 | 0.97158 | 0.045694 | 0.034787 | 0.17541 | 0.037779 |

## Interpretation

The previous stripe fields were numerical: a 5.632-mm source window clipped the displaced +1 order, while widening that same single grid without refinement aliased the measured high-k_perp axicon phase. The corrected multirate route uses a 10-mm relay window and a finer fixed-window axicon grid. It reproduces the basic q=20 annulus before fitting. The fitted low-order residual improves both train and untouched held-out planes only slightly, so it remains a screening prediction rather than a demonstrated hardware correction. Miao input-plane phase and numerical SLM2 phase are kept as distinct planes. Hardware use remains blocked by camera-stage axis, absolute z, vector high-angle response, SLM2 conjugacy/native coordinates, the 1030-nm LUT, branch sign, and a new post-mask BMG acquisition.
