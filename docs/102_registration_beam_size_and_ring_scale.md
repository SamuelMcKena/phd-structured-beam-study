# Input beam radius and downstream ring radius

The 50-pixel and 250-pixel cases have Gaussian amplitude radii of 0.4 mm and
2 mm on SLM1. This is an input field parameter, not a downstream ring radius.
A Gaussian fit to the saved immediate-SLM1 fields recovers those two widths;
the FWHM intensity diameters are 0.471 mm and 2.355 mm. A finite 6 mm saved
crop biases the large-beam second moment, so the audit uses a log-quadratic
Gaussian fit rather than equating that cropped moment with the full radius.

Both fields propagate independently. The central 300 um downstream crop uses
the same physical axes and approximately 0.992 um sampling. It is never
rescaled to enforce matching rings. For an ideal vortex Bessel field the
first bright ring is at `jprime_L1 / kr`: approximately 21.14 um for L=20
under the conditional axicon prescription. This relation does not contain
the incident Gaussian width. Finite beam envelopes and axial behaviour differ.
Similar central rings therefore do not prove that input beam widths match.

The reduced incident beam can be realised using a fivefold beam-reducing
telescope. Aperture clipping is a different illumination model. To make the
central Bessel ring itself smaller, change charge/cone angle or introduce
appropriate downstream demagnification; reducing input width is not enough.

Reproduce the audit after extracting both evidence archives into DATA:

```sh
PYTHONPATH=. python tools/audit_registration_beam_scale.py --data DATA --out FIGURES
python -m pytest tests/test_registration_beam_scale.py tests/test_registration_interpanel.py -q
python tools/revise_registration_summary_scale.py --source ORIGINAL_SUMMARY.pdf --figures FIGURES --out REVISED_SUMMARY.pdf
```

PDF revision requires ReportLab, Pillow and pypdf, plus the DejaVu Sans fonts.
The plot audit writes three standalone PNGs, a width CSV and a manifest.
These are re-audited and re-plotted existing independent simulations; the
registration fidelity and image-shape conclusions are not changed.
