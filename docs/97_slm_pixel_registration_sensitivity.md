# Sub-pixel SLM pixel-lattice registration sensitivity

## The question

The hologram is written onto a discrete LCoS pixel lattice that is fixed in the
laboratory frame. Where the incident beam lands *within* that lattice decides
which part of the continuous phase command each pixel area-averages, and — once
the pixel aperture is resolved — which part of the beam falls on inter-pixel
dead space. The question is how much the beam delivered to the axicon, and the
Bessel field behind it, depends on that registration; whether SLM1 or SLM2
alignment dominates; and whether higher vortex charges are more sensitive.

## Why the accepted Phase 2E sampling cannot answer it

The pixel pitch is 8 um. The accepted Phase 2E source grids run N=1536 over a
10 mm window, which is dx = 6.51 um, or **0.8 computational samples per pixel**.

On such a grid every computational sample falls inside a pixel of its own, so
the circular mean in `pixelate` is taken over a single sample and the operator
degenerates into the identity. The pixel lattice leaves the model entirely, and
a sub-pixel registration sweep returns **exactly zero for every offset** — a
null that is indistinguishable at a glance from a physical finding of no
sensitivity.

This is not a hypothetical. It was observed directly during development: an
end-to-end sweep at N=512 produced a bitwise-identical field at every offset.
`build_registration_route` therefore *refuses* to run a registration sweep below
two samples per pixel unless `allow_unresolved_lattice=True` is passed, and
`blocked_or_data_driven_families()` records why these families must not be added
to the `system_sweep_registry` that drives the existing suite.

## Two registration errors that must not be conflated

`SLMError` now carries two distinct displacements.

| field | meaning | moves the pixel lattice? |
|---|---|---|
| `panel_translation_m` | the physical panel is translated; its pixel lattice, dead-space lattice, active area and the hologram addressed in panel pixel coordinates all move together | yes |
| `pattern_offset_m` | the addressed hologram is shifted electronically relative to its own panel | no |

A pure sub-pixel registration sweep sets, for each selected panel,
`panel_translation_m = -d` together with `pattern_offset_m = +d`. The beam is
never moved. The beam-to-hologram alignment therefore stays exactly nominal
while the beam-to-pixel-lattice phase changes by `d`. Without that
compensation the sweep would instead be an ordinary hologram decentre study,
which the registry already covers at the 25-pixel scale as
`slm1_hologram_offset_x` — a different error at a different scale.

Because the two panels share a plane in this route, translating the panel is
also the *only* way to vary the beam-to-lattice registration independently for
SLM2. That is why the per-panel lattice origin had to be added to
`slm_model.py`; all its new arguments default to the previous behaviour.

## Route and sampling

Resolving sub-pixel registration needs several samples per pixel; the canonical
2 mm beam needs a 10 mm window. Running the whole accepted route at dx = 2 um
over 10 mm would mean N = 5120, and the existing `build_system_route` holds
about 30 full-grid arrays in flight — measured at 2.02 GB peak for N=2048, which
extrapolates to roughly 12.4 GB. That does not fit on the study machine.

The study therefore splits the sampling:

```
fine grid  (N_fine, dx <= 2 um)   beam -> SLM1 -> SLM2/carrier -> L1 -> Fourier plane -> iris
                                             |
                                  band-limited spectral crop
                                             v
relay grid (N_relay, accepted)     -> L2 -> output -> carrier removal -> axicon -> free space
```

The handoff is legitimate because the physical iris band-limits the field: it
passes only spatial frequencies within `iris_radius / (lambda f)` of the
carrier, far inside the fine-grid Nyquist limit. The crop is the exact
counterpart of the existing `fourier_resample_fixed_window`, with the matching
cell-centred origin correction, and it **reports the discarded out-of-band power
for every route** rather than assuming it is negligible.

The isolation arm needs no such split: it uses a small beam, so a 2 mm window at
N=1024 already gives 4.1 samples per pixel.

## Three sampling traps, all of which produce believable wrong numbers

These were found by measurement during development, not anticipated, and each
one silently corrupts the result rather than failing loudly.

**1. An unresolved lattice returns an exact null.** Covered above; guarded.

**2. A grid incommensurate with the pitch manufactures signal.** If the pitch is
not an exact integer number of samples, neighbouring pixels contain different
sample counts, and translating the lattice changes *which* pixels hold the extra
sample. That alone registers as sensitivity. A 10 mm window makes the 8 um pitch
an exact integer m of samples only when `N = 1250 m` (N = 2500, 3750, 5000 for
m = 2, 3, 4); a 2 mm window when `N = 250 m`. On commensurate grids the isolation
arm converges to four significant figures across m = 4, 8, 16; on incommensurate
grids nearby the metric scatters. Routes now report `pitch_commensurate`.

**3. A resolved pixel aperture must never be used at this bench sampling.** A
fill factor of 0.93 implies inter-pixel dead-space lines of only
`(1 - sqrt(FF)) * pitch = 0.285 um`. `resolved_pixel_aperture` guards only that
dx <= pitch/2, which passes at dx = 3.9 um while being fourteen times too coarse
to represent a 0.285 um gap. The resulting mask is aliased, and translating the
lattice changes which samples fall in a gap essentially at random. Measured
consequence: the same configuration reported 1.37e-4 with the resolved aperture
and 8.4e-7 with `throughput_only` — a factor of 160, entirely spurious. A guard
now refuses the resolved aperture unless dx resolves the actual dead space.

The scalar model is not a resigned fallback here; for the bench arm it is the
*correct* reduction, and the reason is analytic. The pixel-aperture lattice is
periodic at the pitch, so it produces spectral replicas spaced `1/p` = 125000
per metre, while the fixed iris passes only `|f - G| <= 2500` per metre. The
nearest replica is **46 times the iris half-width** outside the passed band, so
no replica survives the filter. Within the passed band the aperture contributes
only the flat part of its sinc envelope, which runs 0.9986 -> 0.9925 across the
band: a 0.6% smooth amplitude taper, not a registration-dependent term. Under a
rigid panel translation the value lattice and the aperture lattice move
together, so the aperture adds nothing beyond a sub-micron translation of the
output. Carrying it as `sqrt(FF)` is therefore exact to that 0.6% taper.

This argument is specific to the bench arm, where the iris does the filtering.
It does **not** transfer to the isolation arm, which has no iris; there the
replica orders propagate and the aperture lattice is a genuine contributor that
this study does not model.

A fourth issue is not a trap but a measurement limit: on the bench arm the
registration signal (~1e-6) is smaller than the power the spectral crop discards
(~1e-5). Bench-arm registration metrics are therefore taken on the fine grid
*immediately after the physical iris*, before any decimation, which removes the
crop from the comparison entirely. The decimated field is still used for the
Bessel-region observables, where the quantities are far larger.

## Validation gates

All gates are executed by `tests/test_slm_pixel_registration.py`.

| gate | result |
|---|---|
| lean blocked propagator vs `angular_spectrum_propagate_bl` | bit-identical (0.0e+00) |
| spectral crop vs direct analytic sampling of a band-limited field | 1.0e-15 |
| crop inverts `fourier_resample_fixed_window` | 6.0e-16 |
| `_apply_panel` vs `apply_slm`, both fill-factor models | 5.5e-17 |
| **lean route vs `build_system_route` at matched sampling** | **3.5e-16 to 4.5e-16** |
| iris band-limit: power discarded by the crop | 7.5e-8 (B0), 3.8e-5 (V3) |
| **whole-pitch panel translation, end to end** | **exactly 0.0** |
| half-pitch panel translation, end to end | 9.2e-4 (relative field change) |

Two of these carry the argument. The route-equivalence gate shows the fine/coarse
plumbing *is* the accepted Phase 2E route rather than a new model — at matched
sampling it reproduces it to machine precision. The whole-pitch gate shows the
registration knob moves the pixel lattice and nothing else: a translation of
exactly one pitch maps the lattice onto itself and returns a bitwise-identical
field, which a disguised hologram decentre could not do.

## Scope and provenance

- Registration offsets are geometric sensitivity values, **not** measured panel
  positions.
- The scalar effective-channel convention is used: full charge on SLM1, carrier
  only on SLM2. The documented bench convention splits the charge
  (`ell_SLM1=+10`, `ell_SLM2=-10` for V20). Because SLM2 here carries only a
  linear ramp, this convention bounds the SLM2 result from below, and a
  split-charge run would be required before claiming how the sensitivity
  divides between the panels.
- Absolute hardware claims additionally require the measured grey-to-phase LUT,
  the static panel phase map and a fitted fringing kernel, all still
  data-blocked.
- The isolation arm has no carrier and no iris. It is a mechanism study and is
  not the bench architecture.
- "After the axicon" always means after propagation. With no clear aperture
  bound the axicon is a pure phase element, so the intensity immediately behind
  it is *identical* to the intensity in front of it; a test asserts this so the
  reporting rule cannot quietly lapse.

## Results

Data in `outputs/validation/slm_pixel_registration/`, figures in
`outputs/figures/slm_pixel_registration/`. Metric throughout is the
registration-induced infidelity: `1 - |<E,Eref>|^2 / (<E,E><Eref,Eref>)` between
the field at a given sub-pixel offset and the same route at zero offset. It is
invariant to a global complex scale, so it reports structural change only.

### Charge and beam size are one parameter, not two

The isolation-arm sweep covers charges 1..20 and beam radii 2..50 pixels, 30
combinations. Plotted against the single dimensionless group

    ell * p / w   =   (pixel-unresolved core radius) / (beam radius)

all 25 points fall on one curve. Over the 17 unsaturated points a single power
law fits with **10.9% median and 20.9% worst-case scatter**, across a factor of
20 in charge and 25 in beam radius:

    registration infidelity  ~=  0.65 * (ell p / w)^1.60

The collapse is direct, not just a fit. Three unrelated configurations at
`ell p/w = 0.4` -- (ell 20, 50 px), (ell 10, 25 px), (ell 5, 12.5 px) -- give
0.155, 0.151 and 0.161. At `ell p/w = 0.2`, (ell 1, 5 px), (ell 10, 50 px) and
(ell 5, 25 px) give 0.049, 0.052 and 0.055.

This is the physical answer to "are higher vortex beams more sensitive": yes,
as `ell^1.6`, but only because charge and beam radius enter through their ratio.
A high charge on a large beam and a low charge on a small beam are the same
problem. The mechanism is that the azimuthal phase step across one pixel is
`ell p / r`, so the region a pixel cannot resolve has radius ~ `ell p`; what
matters is how much of the beam sits inside it.

Above `ell p / w ~ 1` the response saturates: the sub-pixel offset stops being a
perturbation and essentially determines the output field. At (ell 20, 5 px) the
infidelity is 0.96 -- two beams differing only in where they land inside a pixel
are almost completely different fields.

### The same parameter also decides whether a Bessel beam forms at all

A charge-`ell` vortex carries transverse momentum `ell / (w k)`; the axicon bends
by `k_r / k`. A Bessel zone forms only when the vortex does not out-diverge the
cone:

    ell / (w k)  <  k_r / k      ->      w  >  ell / k_r      ->      ell p / w  <  k_r p

For the canonical axicon `k_r p = 0.781`. So the **same dimensionless group that
governs registration sensitivity also sets the Bessel-forming boundary**, at
`ell p / w < 0.78`. In beam-radius terms the requirement is `w > ell / k_r`:

| charge | 1 | 3 | 5 | 10 | 20 |
|---|---|---|---|---|---|
| minimum beam radius | 1.3 px | 3.8 px | 6.4 px | 12.8 px | 25.6 px |

This has a useful consequence. Substituting the boundary into the collapse gives
a registration infidelity of `0.655 * 0.781^1.60 = 0.44` there, and the nearest
measured point (charge 20 at 25 px, `ell p/w = 0.80`) reads 0.389. **The
saturated regime lies entirely outside the Bessel-forming regime.** Any
configuration that actually produces a Bessel beam has a worst-case sub-pixel
registration penalty of roughly 0.4 infidelity or less -- large, but never the
total loss of structure seen at `ell p/w > 1`.

It also explains the charge-20, 100-pixel-beam propagation map: at
`ell p/w = 1.6` the vortex diverges at 32.8 mrad against a 16.0 mrad cone, so
the light expands instead of forming a Bessel zone. Axial metrics in that regime
describe an expanding ring, not a Bessel beam, and should not be read as
Bessel-zone properties.

### B0 is exactly zero

Under SLM1 translation, charge 0 gives identically 0.0 at every offset and every
beam radius. A flat phase command pixelates exactly, so there is nothing for the
lattice position to change. This is a useful internal check that the measured
signal is the vortex core and not a numerical artefact of translating a panel.

### Scope of the isolation numbers

These are isolation-arm results: no carrier, no iris. They are the mechanism in
the absence of spatial filtering, and they are *not* bench predictions.

### On the actual bench the effect is negligible

Full route, canonical 2 mm beam (250 pixels), commensurate grid at three
samples per pixel, 108 routes. Registration-induced infidelity, measured
crop-free on the fine grid immediately after the iris:

| charge | SLM1 only | SLM2 only | both, common | both, differential |
|---:|---:|---:|---:|---:|
| 0  | **exactly 0** | 2.1e-6 | 2.1e-6 | 2.4e-6 |
| 1  | 1.4e-7 | 2.3e-6 | 2.3e-6 | 2.5e-6 |
| 3  | 7.2e-7 | 3.3e-6 | 2.7e-6 | 3.2e-6 |
| 5  | 1.4e-6 | 4.2e-6 | 3.1e-6 | 3.9e-6 |
| 10 | 3.1e-6 | 6.6e-6 | 3.7e-6 | 5.5e-6 |
| 20 | 6.1e-6 | 1.08e-5 | 5.6e-6 | 8.3e-6 |

Worst case over the whole matrix is 1.1e-5. Sub-pixel registration is not a
bench problem at the canonical beam size, and no alignment tolerance needs to be
derived from it.

The charge-0/SLM1 entry is identically 0.0 at every offset. That is a live
noise floor *inside the production dataset*: a flat phase command pixelates
exactly, so a panel translation can change nothing, and any non-zero reading
there would have condemned the whole run.

### SLM2 matters more than SLM1, which was not the expectation

SLM1 carries the entire vortex charge and SLM2 carries only a linear carrier
ramp, so the naive expectation is that SLM1 dominates. It does not. SLM2-only
translation is **16x more damaging than SLM1-only at charge 1** and still ~1.8x
at charge 20, and it is non-zero even at charge 0 where SLM1 contributes
identically nothing.

The reason follows from the iris, and it is the same mechanism that makes the
bench insensitive overall. Pixelating the vortex core produces an error
concentrated at small radius, which is *broad* in the Fourier plane and is
therefore thrown away by the 0.772 mm iris. Pixelating the carrier instead
changes the sampling phase of the blazed ramp, which alters the +1 order itself
-- a low-spatial-frequency change that lands inside the iris and survives. The
filter removes the SLM1 error and passes the SLM2 error.

Two practical consequences. Translating both panels together (common mode) is
*less* harmful than translating SLM2 alone at every charge, so the two errors
partially cancel; and the differential case is the worse of the two-panel
configurations, as expected.

This ranking is specific to the effective-channel convention used here. Under
the documented split-charge bench convention (`ell_SLM1=+10`, `ell_SLM2=-10`),
SLM2 would also carry a phase core and would acquire the SLM1 mechanism on top
of the carrier mechanism measured here. These numbers therefore bound the SLM2
contribution from below, not above.

### Does the field behind the axicon change too?

The axicon is a pure phase element with no bound clear aperture, so the
intensity immediately behind it is *identical* to the intensity in front of it
-- a test asserts this. Any consequence appears only after propagation into the
Bessel region. Isolation arm, spread across the offset sweep:

| charge | beam radius | infidelity at the axicon input | peak intensity spread | Bessel zone-length spread |
|---:|---:|---:|---:|---:|
| 0  | any     | 0       | 0       | 0     |
| 3  | 50 px   | 7.8e-3  | 1.8e-5  | 0     |
| 3  | 12.5 px | 8.2e-2  | 7.8e-4  | 9.7%  |
| 3  | 5 px    | 0.33    | 18%     | 0     |
| 20 | 50 px   | 0.156   | 7.3e-4  | 12%   |
| 20 | 12.5 px | 0.74    | 80%     | 180%  |
| 20 | 5 px    | 0.96    | 115%    | 150%  |

The transfer is strongly nonlinear. Below roughly 0.1 infidelity at the axicon
input the Bessel region is untouched, with peak intensity varying by less than
1e-3. Above roughly 0.3 it is grossly affected: at charge 20 with a 12.5-pixel
beam the peak Bessel intensity varies by 80% and the zone length by 180%
depending only on where the beam sits inside a pixel.

The interesting case is charge 20 at 50 pixels, where the input infidelity is a
substantial 0.156 but the peak Bessel intensity is robust to 7e-4 while the zone
length still moves 12%. The axicon plus free propagation act as their own
filter: only radial content near `k_r` builds the Bessel core, so much of the
transverse registration error does not survive, while the *axial* structure
still inherits it. A study that looked only at peak intensity would have
concluded, wrongly, that nothing propagates through.

`z_at_peak` is quantised by the 14-plane axial sampling and should not be read
quantitatively; zone length and peak intensity are the reliable axial measures
here.

### Measured: the iris suppresses the SLM1 error by 600-1800x

The claim that spatial filtering is what makes the bench insensitive is not left
as an argument. The isolation arm was re-run at the canonical 2 mm beam on the
*same* N=4096 commensurate grid as the bench arm, so the two differ only by the
carrier and the Fourier-plane iris. SLM1 translation only:

| charge | no iris (isolation) | full bench | suppression |
|---:|---:|---:|---:|
| 0  | 0        | 0        | -- |
| 1  | 9.86e-5  | 1.41e-7  | 699x |
| 3  | 4.46e-4  | 7.17e-7  | 622x |
| 5  | 1.00e-3  | 1.38e-6  | 728x |
| 10 | 3.29e-3  | 3.08e-6  | 1068x |
| 20 | 1.08e-2  | 6.09e-6  | 1780x |

The suppression *increases* with charge, from about 700x to about 1800x. That
is the signature the mechanism predicts: pixelating a higher-charge core puts
the error at higher spatial frequency, which is precisely where the fixed iris
cuts hardest. The bench is insensitive to sub-pixel registration because it
spatially filters, not merely because the beam is large.

The same filter is why SLM2 outranks SLM1 on the bench: the carrier error is
low-frequency and is not removed.

### The fixed iris is a hard aperture for high charges

An incidental but material finding. Transmission of the +1 order through the
fixed 0.772 mm iris, canonical 2 mm beam:

| charge | 0 | 1 | 3 | 5 | 10 | 20 |
|---|---|---|---|---|---|---|
| iris transmission | 1.000 | 0.998 | 0.982 | 0.951 | 0.820 | **0.452** |

At charge 20 the iris passes less than half the order, because the far field of
a high-charge vortex is broad. This is unrelated to registration and is a
standing property of the bench as configured; it is worth an explicit
throughput/aperture decision before high-charge work.

## Artifacts

Data under `outputs/validation/slm_pixel_registration/<tag>/`, figures under
`outputs/figures/slm_pixel_registration/<tag>/`. 360 routes in total.

| tag | routes | what it is |
|---|---|---|
| `bench` | 108 | full route, canonical 2 mm beam, all four registration DOFs, commensurate m=3 |
| `isolation` | 180 | SLM1 -> axicon, charge x beam-radius sweep, commensurate m=8 |
| `isolation_canonical` | 36 | isolation arm at the canonical 2 mm beam on the bench grid; the iris comparison |
| `after_axicon` | 36 | axial profiles through the Bessel region |

Figures worth going to first:

- `isolation/10_registration_scaling_collapse.png` -- the `ell p / w` collapse.
- `isolation/12_propagation_region_*.png` -- longitudinal x-z maps, ideal vs a
  half-pixel offset vs their difference, at 12.5 px and 50 px for charges 3 and
  20. These span the whole range: at `ell p/w = 0.06` the two are
  indistinguishable, at 0.40 a systematic loss appears along the inner ring
  edge, and at 1.60 no Bessel zone forms at all.
- `isolation/07_` and `08_registration_before/after_axicon_v20.png` --
  transverse profiles with difference maps and domain-coloured phase.
- `bench/01_registration_offset_curves.png` -- per-charge offset curves with all
  four registration degrees of freedom.

Reproduce with `tools/run_slm_pixel_registration_study.py` and plot with
`tools/plot_slm_pixel_registration.py`; `tools/summarise_slm_pixel_registration.py`
prints the headline table for any tag.

## What this does not answer

- **Split-charge convention.** Everything here puts the full charge on SLM1.
  Under the documented bench convention (`ell_SLM1=+10`, `ell_SLM2=-10`) SLM2
  also carries a phase core, so its sensitivity would gain the SLM1 mechanism on
  top of the carrier mechanism measured here.
- **Dead-space lattice on the isolation arm.** The analytic reduction to a
  scalar throughput relies on the iris rejecting the pixel replica orders. The
  isolation arm has no iris, so its replicas propagate and the aperture lattice
  is a genuine unmodelled contributor there.
- **Measured panel data.** LUT, static phase map and fringing kernel remain
  data-blocked, so all offsets are geometric sensitivity values rather than
  predictions for the actual HOLOEYE panels.
- **Bench-arm convergence.** The bench numbers are at three samples per pixel on
  a commensurate grid. They are consistent and far below any level that would
  matter, but they are not demonstrated converged the way the isolation arm is
  (four significant figures across m = 4, 8, 16).

## Next step: scalable angular spectrum

The bench arm currently goes fine grid -> FFT -> iris -> spectral crop -> relay
grid, and the crop discards ~1e-5 of the power while the bench registration
signal is ~1e-6. That is why bench metrics are taken crop-free on the fine grid.

A scalable/chirp-z angular-spectrum propagator would compute the Fourier plane
directly over the ~2 mm region around the +1 order at its own sampling, removing
the crop entirely and shrinking the fine stage's peak memory. That would allow
m = 4 and m = 8 on the bench arm and let the bench number be stated as converged
rather than as a bound.

It would *not* relax the sampling requirement at the SLM itself: representing an
8 um pixel needs dx <= 4 um at the SLM plane, and no propagation method changes
what the input plane must resolve. A replacement propagator must also clear the
same equivalence bar the current one did - reproducing `explicit_4f_relay` to
~1e-12 - because the claim that this route *is* the accepted Phase 2E route
rests on that gate.
