# STAGE6A_SUMMARY.md — Visualisation Foundation + Holographic-Route Overhaul

**Date:** 2026-06-16  
**Branch:** master  
**Commits:** 4 gated commits (eaf3ec8 → ddedc1d)

---

## What was done

### A — Reusable rendering layer (`vbb_study/viz_fields.py`)

New module with three renderers; no engine physics touched.

| Function | Description |
|---|---|
| `complex_field_image(U, grid, ...)` | Domain-coloured complex field. HUE = phase via colorcet CET_C2 (cyclic, perceptually-uniform; falls back to matplotlib `twilight`), VALUE = normalised amplitude. Black in zero-amplitude regions. ℓ=3 winding visibly wraps 3 times. |
| `linked_field_views(result, ...)` | 4-panel figure from real engine result: (1) transverse \|U\|² at selectable z, (2) domain-coloured surface-plane phase, (3) longitudinal x–z intensity slice, (4) radial profile + analytic \|J_ℓ(k_r r)\|² overlay + measured ring marker. Physical µm units, origin lower, inferno intensity cmap. |
| `azimuthal_order_panel(field_on_ring, ...)` | Wraps the validated A4 `_azimuthal_power_spectrum` tool from `tests/test_physics_validation.py`. Polar ring plot + azimuthal-order power bar chart, m=6 highlighted for hexagonal beam readiness. |

**Unit tests (`tests/test_viz_fields.py`, 9 tests, all pass):**
- **V1a** — ℓ=3 3-fold colour symmetry: rendered colours at φ and φ+2π/3 match (primary cyclic assertion)
- **V1b** — ℓ=3 DFT dominant ring frequency = 3
- **V2** — ℓ=0 flat-phase field: no azimuthal colour variation
- **V3** — A4 re-export: uniform ring → DC only; 6-fold modulated → correct orders
- **V4/V5** — `azimuthal_order_panel` smoke tests
- **V6** — `linked_field_views` smoke test on fast-preset holographic result

### B — Viz dependencies

`colorcet>=3.1` already listed in `pyproject.toml` `[project.optional-dependencies] viz`.  
`colorcet 3.2.1` installed and verified (`import colorcet; colorcet.cm.CET_C2` works).  
`cmocean` not installable from PyPI on this machine; matplotlib `twilight` used as fallback.  
`pyproject.toml` `viz` extra retained as-is (both entries stay for future installs).

### C — Holographic figures regenerated at paper preset

`notebooks/lab_realism/01_holographic_axicon_route.ipynb` — 4 new cells added:

1. **Markdown preamble** (cell `bf46b60e`): explains the paper-preset replacement, lists the 4 panels, explicitly notes physical route excluded pending SLM2 decision.
2. **Run cell** (cell `4a673984`): builds paper-preset holographic configs for `general` and `limits` regimes, runs `bt.run_case(... preset="paper" ...)`.
3. **linked_field_views save cell** (cell `63a2305a`): saves 4-panel figure per regime to `outputs/figures/stage_c/stage_6a/`.
4. **azimuthal_order_panel cell** (cell `5448993c`): samples ring intensity from surface field, saves polar + power-spectrum figure per regime.

Figures generated (saved to `outputs/figures/stage_c/stage_6a/`, gitignored; regenerate by running notebook):

| Figure | Content |
|---|---|
| `holographic_general_paper_linked_field_views.png` | General regime, ℓ=3, zone=116.2 µm, ring_r=2.615 µm |
| `holographic_limits_paper_linked_field_views.png` | Limits regime, ℓ=3, zone=179.0 µm, ring_r=1.713 µm |
| `holographic_general_paper_azimuthal_order.png` | General regime azimuthal power spectrum |
| `holographic_limits_paper_azimuthal_order.png` | Limits regime azimuthal power spectrum |

The domain-coloured phase panel in each `linked_field_views` figure shows the ℓ=3 winding visibly completing 3 full colour cycles around the dark vortex core — something the old intensity-only and flat-phase train-visualiser figures could not show.

### D — VIZ_AUDIT.md

Complete audit of all notebooks. 8 plots catalogued:

| Status | Count | Description |
|---|---|---|
| REPLACED | 2 | NB01-A (flat phase/fast-grid train viz), NB01-B (no field-view at paper preset) |
| DEFERRED | 4 | Physical-route notebooks (NB02-A, NB03-A, NB05-A, NB06-A) — blocked by SLM2 charge-0 issue (Finding F-A3p) |
| NOTED | 2 | NB04-A (categorical QA on continuous cmap), NB02-B (intensity-only plane montage) |

### E — scalar/04 parameterised

`notebooks/scalar/04_scalar_parameter_sweeps.ipynb` — inserted code cell `6a1e9c0b` (tagged `parameters`) immediately after the imports cell.  Default: `PRESET = "publication"`.  The old hard-coded `PRESET = "publication"` assignment removed from its original cell.

Triage can now inject `PRESET = "fast"` without editing source:
```
papermill notebooks/scalar/04_scalar_parameter_sweeps.ipynb out.ipynb \
  --parameters PRESET fast
```

### F — Note on analytic HTML toy

`bessel_beam_simulator.html` is an idealised analytic intuition aid, not engine output. It is kept clearly separate; the Stage 6A figures are all from `bt.run_case()` at paper preset.

---

## Lock results

| Test suite | Result |
|---|---|
| `tests/test_characterisation_lock.py` (fast, 9 tests) | **9/9 PASS** |
| `tests/test_characterisation_lock_prod.py` (paper, 9 tests) | **9/9 PASS** |
| `tests/test_viz_fields.py` (new, 9 tests) | **9/9 PASS** |

Engine untouched — `git diff HEAD~4..HEAD -- Publication_Study/bessel_twin_core.py` is empty.

---

## Commits

| Hash | Description |
|---|---|
| `eaf3ec8` | Stage 6A-A: add viz_fields.py rendering layer + unit tests |
| `e1feae2` | Stage 6A-B: parameterise scalar/04 with papermill parameters cell |
| `ddedc1d` | Stage 6A-C: holographic figures at paper preset + VIZ_AUDIT.md |
| (this) | Stage 6A-D: STAGE6A_SUMMARY.md |

---

## What remains (not done in 6A)

- Physical-route figures: blocked until SLM2 charge configuration is resolved (F-A3p)
- scalar/02 plane montage phase panels (NB02-B)
- scalar/04 capability-slice categorical colourmap fix (NB04-A)
- `cmocean` install (network issue; fallback to `twilight` is correct; retry when network allows)
