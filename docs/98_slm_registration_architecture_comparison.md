# SLM registration architecture comparison

## Purpose

This study asks a narrower question than the general system-error campaign:

How sensitive is the two-SLM vortex-Bessel route to where the optical field
lands inside the physical 8 um SLM pixel unit cell, and does moving vortex
generation from SLM1 to SLM2 reduce that sensitivity?

Only pixel registration is varied. Beam-to-hologram centring remains nominal.

## Phase-allocation architectures

Two routes are compared with the same downstream 4F selection and physical
axicon.

### Architecture A - upstream vortex

~~~
Gaussian
-> SLM1: vortex + carrier/blaze
-> SLM2: correction + carrier/blaze
-> explicit common 4F + selected order
-> physical axicon
-> free-space Bessel region
~~~

### Architecture B - downstream vortex

~~~
Gaussian
-> SLM1: correction + carrier/blaze
-> SLM2: vortex + carrier/blaze
-> explicit common 4F + selected order
-> physical axicon
-> free-space Bessel region
~~~

Both SLM1 and SLM2 carry the carrier/blaze in both architectures. This is a
hard study contract, not an optional plotting convention.

In the scalar effective-channel model, the two displayed linear ramps are
sequential phase terms. The common-4F selected-order centre is therefore
computed from their summed scalar carrier, and the summed carrier is removed
after the 4F image plane. This is the internally consistent scalar analogue of
the two blazed masks. The absolute laboratory carrier sign/orientation remains
a hardware-coordinate convention to verify experimentally.

The current repository does not contain a validated spatial correction map.
Therefore the default architecture comparison sets the correction term exactly
to zero and labels it flat_zero_physics_isolation. A user-supplied spatial
correction command/map may be supplied for a second pass. No nominal or
invented correction is allowed to masquerade as measured bench correction.

> Quantitative report results from the sampled-4F route are superseded by
> [the independent audited study](100_slm_registration_definitive.md).
> Both blazes are required but do not by themselves repair numerical propagation.
> The identity-transfer flat-correction model is role-exchange symmetric and
> cannot test an architecture winner. Historical hypothesis language below
> must be read in that scope.

## What registration means

For panel i define beam centre b_i, hologram origin h_i and physical
pixel-lattice origin g_i.

Ordinary decentre is b_i - h_i. The registration variable of interest is the
position of h_i relative to g_i modulo one pixel.

The study uses the same compensated construction as the audited registration
branch:

~~~
panel_translation = -d
pattern_offset     = +d
~~~

so beam-to-hologram decentre remains exactly zero while beam/hologram-to-lattice
registration changes.

A separate beam-only decentre is not part of the production matrix. Existing
system-error tooling already covers ordinary hologram/beam decentre.

## Why two-dimensional registration is required

An x-only sweep is insufficient because the vortex centre may lie near a pixel
centre, an edge, or a corner. Representative runs therefore include a complete
pixel unit-cell map:

~~~
rho_x, rho_y = {0, 1/8, ..., 7/8} pixel
~~~

for each SLM independently.

The endpoint at one full pitch is omitted because it is physically periodic
with zero registration.

## Independent and relative SLM registration

The study separates four one-dimensional registration degrees of freedom:

- SLM1 only,
- SLM2 only,
- common mode,
- differential mode.

It also provides a direct SLM1-offset x SLM2-offset map for representative
charge/beam-size cases. This map is the key diagnostic for the architecture
hypothesis: architecture B removes an upstream pixelated vortex from SLM1, so
the relative two-panel sensitivity should change if that mechanism is important.

## Directions

Production sweeps support:

- x,
- y,
- diagonal.

This is required because both panels carry x-directed carrier/blaze terms,
which introduce a preferred axis. A raw x-direction panel ranking must not be
generalised to morphology until y and diagonal behaviour are checked.

## Beam-size and charge matrix

The intended core matrix is:

~~~
charge L = 0, 1, 3, 5, 10, 20
beam radius = 25, 50, 100, 250 pixels
registration = 0, 1/8, 1/4, 3/8, 1/2 pixel
~~~

The 250-pixel radius is the canonical approximately 2 mm bench beam.

Smaller radii are mechanism-revealing stress cases. Very small configurations
that cross the Bessel-forming boundary are to be labelled stress tests rather
than representative bench beams.

## Diagnostic planes

For representative XY-profile runs the route can retain:

1. field immediately after SLM1,
2. field immediately after SLM2,
3. field immediately after the physical +1 iris on the fine grid,
4. field immediately before the physical axicon,
5. field immediately behind the axicon,
6. a matched propagated plane in the Bessel region.

The physical axicon is a pure phase element in this model, so its immediate
post-axicon intensity is identical to its input intensity. The useful
post-axicon comparison is after propagation.

## Metrics

The source registration metrics are reused.

### Laboratory-frame complex-field change

~~~
1 - |<E,E0>|^2 / (<E,E><E0,E0>)
~~~

This includes beam walk.

### Translation-registered morphology change

The test field is Fourier-recentred onto the reference centroid before fidelity
is recomputed. This is the preferred quantity for statements about deformation
rather than total laboratory-frame change.

### Other transverse diagnostics

The existing metric set also records:

- centroid x/y,
- power ratio,
- ring radius,
- ring peak intensity,
- core darkness,
- azimuthal purity,
- ring asymmetry.

### Axial diagnostics

Representative cases additionally evaluate:

- peak ring intensity versus z,
- core darkness versus z,
- ring radius versus z,
- peak-z position,
- Bessel-zone FWHM length,
- retained XY fields at selected z planes.

## Primary outputs

The study runner writes:

~~~
outputs/validation/slm_registration_architecture/<tag>/
    manifest.json
    architecture_sweep.csv
    unit_cell_maps.csv
    interpanel_registration_map.csv
    axial_metrics.csv
    representative_xy_fields.npz
~~~

The plotter writes to:

~~~
outputs/figures/slm_registration_architecture/<tag>/
~~~

including:

- offset curves for both architectures,
- direct B/A sensitivity ratio,
- charge x beam-radius maps,
- B/A architecture-ratio map,
- SLM1 and SLM2 pixel-unit-cell maps,
- SLM1-offset x SLM2-offset maps,
- quantitative post-axicon peak/zone/morphology curves,
- real XY pre-axicon intensity/residual panels,
- real XY propagated intensity/residual panels.

## Architecture hypothesis

The study begins with a falsifiable hypothesis, not a conclusion.

Generating the vortex locally on SLM2 may reduce sensitivity to differential
inter-SLM registration because SLM1 no longer imprints a pixelated high-gradient
vortex phase that subsequently encounters a second independently registered
panel. However, architecture B still retains the local SLM2 vortex-to-pixel
registration error, which should increase with L and decrease with beam radius.

The B/A sensitivity-ratio map is the principal test of this hypothesis.

## Correction-map policy

The code supports a user-supplied spatial correction command. The CLI accepts a
square .npy phase map covering the simulation window and bilinearly samples it
in panel coordinates.

Two report layers are therefore possible:

1. physics isolation: correction OFF, used to isolate vortex ownership;
2. actual correction: correction ON using a supplied measured/fitted map.

The report must keep these two layers separate.

## Claim boundary: SLM1-to-SLM2 propagation

The existing audited registration route contains the accepted explicit 4F model,
but it does not contain a measured physical SLM1-to-SLM2 propagation distance.
The separate component-owned CSLM scaffold has only a diagnostic placeholder
distance and explicitly labels it non-measured.

This architecture study therefore does not invent an inter-SLM distance. It
compares phase ownership and pixel registration on the accepted
effective-channel route. A measured SLM1-to-SLM2 geometry can be added later as
a second realism layer without changing the registration definitions.

## Reproduction

Core architecture sweep and representative XY fields:

~~~
python tools/run_slm_registration_architecture_study.py \
  --tag core \
  --stages sweep,axial,xy \
  --fine-grid-n 2500 \
  --relay-grid-n 1024
~~~

Add two-dimensional pixel-unit-cell and relative-panel maps:

~~~
python tools/run_slm_registration_architecture_study.py \
  --tag full \
  --stages sweep,unit_cell,interpanel,axial,xy \
  --fine-grid-n 2500 \
  --relay-grid-n 1024
~~~

Plot:

~~~
python tools/plot_slm_registration_architecture_study.py --tag full
~~~

For the higher-confidence canonical bench pass, repeat with the existing
three-samples-per-pixel commensurate grid:

~~~
fine_grid_n = 3750
window = 10 mm
~~~

and repeat the representative cases using both area_average and centre_sample
pixel-value conventions.

## Required validation before report conclusions

- phase ownership matches the declared architecture;
- SLM1 and SLM2 carrier/blaze terms exist in both architectures;
- compensated registration leaves beam-to-hologram decentre at zero;
- whole-pitch periodicity remains exact;
- x/y symmetry is recovered when the carrier is removed in a diagnostic test;
- x/y differences are allowed when the SLM2 carrier/blaze is active;
- the full-size zero-registration route remains consistent with the accepted
  registration route;
- the B/A comparison is repeated at increased samples per pixel for headline
  cases;
- any supplied correction map is labelled by provenance and never silently
  substituted by a synthetic map.
