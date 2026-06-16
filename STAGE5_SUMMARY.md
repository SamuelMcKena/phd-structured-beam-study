# Stage 5 Summary — Physics Validation Harness + First Validated-Quantity Plot

**Date**: 2026-06-15  
**Characterisation lock status at start**: 9/9 PASSED  
**Characterisation lock status at end**: 9/9 PASSED  
**Engine files modified**: NONE (this stage is additive only)

---

## What was done

### Part A — Physics validation harness

**New file**: `tests/test_physics_validation.py` (33 tests)

Six independent correctness checks against closed-form analytic expectations. Tests
run via `pytest tests/test_physics_validation.py`. The engine is called via `run_case`
(real physics), and analytic formulas are computed independently inside the tests.

**Final result: 22 passed, 11 xfailed, 0 unexpected failures**

| Check | Result | Tests |
|-------|--------|-------|
| A1 — Core/ring radius vs analytic | 2 PASS, 4 XFAIL (documented findings) | 6 |
| A2 — Zone length vs geometry | 4 PASS, 2 XFAIL (limits regime finding) | 6 |
| A3 — Topological charge (phase winding) | 3 PASS, 2 XFAIL (physical route finding) | 5 |
| A4 — Azimuthal Fourier tool + degenerate rejection | 7 PASS | 7 |
| A5 — Grid convergence | 1 PASS | 1 |
| A6 — Energy conservation | 4 XFAIL (fast-preset finding) | 4 |

The phase winding test (A3) uses a closed-loop incremental-phase integral, not a
correlation metric. The A4 azimuthal Fourier tool was built and unit-tested on three
synthetic inputs (pure ring, n-fold modulated ring, discrete n-spot comb). The ring
vs lattice distinction uses `power[n]/power[0]` ratio (not `power[n]/total`, which is
wrong for a periodic comb — this was a finding and the test was corrected).

### Part B — Rebuilt `plot_sampling_qa`

**Modified file**: `vbb_study/vbb_train_viz.py` — `plot_sampling_qa` function replaced.

**Two defects fixed**:
1. **FALSE 2D** (previous): every column was identical because the code used the
   boolean `valid` flag, which is purely core-diameter-driven. But the zone axis IS
   real: at zone=25 µm the axial sampling criterion becomes binding, producing
   distinct margin values in the zone=25 column.
2. **FALSE TERNARY** (previous): colormap defined fail/marginal/pass but code only
   wrote `2 if valid else 0` — 'marginal' never appeared and the margin was discarded.

**New plot shows**:
- Continuous per-cell sampling margin = min(spf/2, spr/2, spaz/4), where margins < 1.0
  mean Nyquist violation.
- Perceptually-uniform (viridis) heatmap showing HOW FAR each cell is from the
  threshold, not a boolean.
- White contour at margin = 1.0 (pass/fail boundary).
- Per-cell annotation showing the margin value and the binding criterion:
  `feat` = transverse feature Nyquist, `rad` = radial period Nyquist, `ax` = axial
  zone sampling.
- Caption/figure metadata explains the zone-axis structure.

Key pattern visible in the new plot:
- core = 0.5 µm: fails for all zone lengths (transverse feat binding, margin ≈ 1.00 → just below threshold)
- zone = 25 µm column: axially binding for all core sizes (margin ≈ 1.04)
- zone ≥ 50 µm, core ≥ 0.75 µm: feature-binding, margin scales with core diameter
- Maximum margin: 5.56× Nyquist threshold (core=2.8 µm, zone ≥ 200 µm)

---

## Findings (honest reporting)

Four physics findings were surfaced. None are silently papered over; all are
documented with both numbers in `PHYSICS_VALIDATION_FINDINGS.md`.

**F-A1b (grid-limited)** — Limits-regime holographic ring radius:  
Measured 1.634 µm vs analytic 1.747 µm (−6.5 %, tolerance 5 %).  
Cause: pixel-quantization bound ~7 % at dx=0.25 µm for a 1.75 µm ring. Not a physics bug.

**F-A1c (under-relaxed beam)** — Physical-route ring radius:  
Measured ring radius undershoots analytic J'_ell / k_r by up to 8.9 %.  
Cause: fast-preset grid quantization + beam not fully relaxed to asymptotic Bessel profile at surface plane.

**F-A2c (limits-regime zone)** — Zone at fast preset is severely shortened:  
general regime: 116 µm (78 % of 150 µm target) ← OK.  
limits regime: 104–114 µm (35–38 % of 300 µm target) ← FINDING.  
Cause: BL-ASM severe power drift (220 %) at fast preset for the limits case.

**F-A3p (PRIORITY finding)** — Physical route carries topological charge **0**, not ell=3:  
Measured phase winding = 0.000 turns, expected 3.000 turns.  
Root cause: `slm2_conjugate_mode='full'` applies the full phase conjugate of U2
(which carries the SLM1 vortex phase), effectively cancelling the helical ramp before
the axicon. The physical route produces SCALAR Bessel beams regardless of the design
ell. The holographic route correctly imprints charge=3 (winding=3.000). This requires
a deliberate physics-design decision before any fix is attempted.

**F-A6 (energy conservation)** — All fast-preset cases fail the 5 % drift threshold:  
general: 67–71 % drift; limits: 221 % drift. Cause: BL-ASM propagator bandlimit clipping
under aggressive device_downsample=4. Not a physics bug; a known accuracy/speed trade-off.

---

## Lock discipline

- Lock was confirmed green (9/9) **before** any code was written.
- `tests/test_physics_validation.py` was added (additive only; no engine changes).
- `vbb_study/vbb_train_viz.py` → `plot_sampling_qa` was replaced (viz only; engine untouched).
- Lock was confirmed green (9/9) after each change.
- The `baseline_inspect.py` and `inspect_validity.py` temporary scripts can be deleted; they do not affect the package.

---

## What was NOT verified

- The limits-regime zone length is not a reliable physics prediction at `fast` preset (F-A2c, F-A6).
- The ring radius at the limits regime has ~7 % grid-quantization uncertainty (F-A1b).
- The physical route vortex charge has not been verified — it is confirmed to be 0, not 3 (F-A3p).
- Grid convergence was tested at only one refinement level (2×) for one case (general/holographic/ideal). Other regimes and routes were not tested for convergence.
- The SAS propagator was not tested (all cases use BL-ASM at `fast` preset).

---

## Files created / modified

| File | Action | Description |
|------|--------|-------------|
| `tests/test_physics_validation.py` | Created | 33-test physics correctness suite |
| `PHYSICS_VALIDATION_FINDINGS.md` | Created | Detailed findings with both numbers |
| `vbb_study/vbb_train_viz.py` | Modified | `plot_sampling_qa` rebuilt (viz only) |
| `STAGE5_SUMMARY.md` | Created | This file |
| `baseline_inspect.py` | Created (temp) | Inspection script — safe to delete |
| `inspect_validity.py` | Created (temp) | Inspection script — safe to delete |
