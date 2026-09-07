# Poster figures

Presentation-only figure workspace for the A0 vortex-Bessel poster. This branch is deliberately separate from the accepted report-evidence branch so poster layout changes do not overwrite accepted scientific outputs.

## Build

From the repository root:

```bash
python poster_figures/build_poster_figures.py
```

Outputs are written to `poster_figures/generated/`:

- `01_gaussian_vs_bessel_like_core.png` - Gaussian focus versus finite Bessel-like interaction region. The Gaussian side uses the paraxial Gaussian-beam waist evolution; the Bessel-like side uses a finite-energy Bessel-Gauss transverse profile with a finite conical-overlap window.
- `02_processing_relevance.png` - processing-motivation schematic. This is explicitly optical-field motivation only and does **not** claim a calibrated material-modification model.
- `03_experimental_modelled_optical_route.png` - white-background component-order schematic matching the numerical route: fs laser -> SLM1 -> SLM2 -> 4f/+1 order -> physical axicon -> camera/sample.
- `04_vortex_bessel_reference_field.png` - clean poster typesetting of the vortex-Bessel reference field.

## Scientific boundaries

These are presentation derivatives, not new evidence. They do not replace the accepted Phase 1/2 figures or alter any metrics. Quantitative report-facing claims continue to use the validated repository renderers and fixed-physical-optics route described in `AGENTS.md` and `docs/PRESENTATION_CURRENT.md`.

The processing schematic is deliberately labelled as motivation only: the current optical model does not infer nonlinear material response, ablation, or written-waveguide morphology.

## Poster use

The poster itself keeps the audited black-background simulation figures for B0/V1/V3, axicon decentre, rounded-tip response, and inverse correction. These poster figures are used only where a white-background explanatory schematic is clearer for a general audience.
