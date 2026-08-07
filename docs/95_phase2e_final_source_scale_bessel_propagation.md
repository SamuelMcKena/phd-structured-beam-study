# Phase 2E - Final Source-Scale Bessel Propagation

**Outcome: PHASE2E-FINAL-A.** Source resolution, z sampling, route semantics and final report
figures validate. Nominal no-additional-aperture B0/V1/V3 figures are authorised.

This closure applies only to source-scale propagation after the common 4F reconstruction. It does
not alter the Phase 2C objective/sample-scale Debye contract or make an objective-focused claim.

## Governed Route

The nominal physical sequence is:

```text
SLM phase-only modulation
-> common 4F selected-order filtering
-> carrier removal and reconstructed field
-> no additional real-space aperture
-> one physical axicon phase
-> band-limited angular-spectrum propagation in air
```

The finite Gaussian envelope and SLM/numerical support remain. The nominal route is therefore not
an infinite-aperture model. It applies one 4F filter, one carrier removal, one axicon, and zero
objective transforms.

Three route identities are preserved:

| route | aperture | report role |
|---|---|---|
| `nominal_no_additional_aperture` | none additional | primary source-scale prediction |
| `soft_aperture_sensitivity` | `exp[-(r/1.8 mm)^8]` | unmeasured sensitivity case |
| `hard_aperture_diagnostic` | unit disk, radius 1.8 mm | diagnostic only; calibration required |

The full route metadata are in `outputs/validation/phase2e_final_propagation/source_scale_route_contract.json`.

## Source-Resolution Gate

The physical source window remained 10 mm. N=3072 was selected, giving dx=3.255208 um,
19.7646 samples per axicon radial phase period, an adjacent radial phase increment of 0.317901 rad,
and 49.152 samples per carrier period.

| N | dx (um) | radial samples/period | worst raw observable difference | worst fixed-bucket difference | worst radius difference | gate |
|---:|---:|---:|---:|---:|---:|---|
| 2048 | 4.882812 | 13.1764 | 3.1427% | 2.8891% | 1.9599% | fail |
| 2560 | 3.906250 | 16.4705 | 1.2128% | 2.1607% | 0.5109% | fail |
| 3072 | 3.255208 | 19.7646 | reference | reference | reference | pass |

N=2560 fails the predeclared 1% raw-observable and fixed-bucket gates. Its worst total-power
difference is 0.00168%, maximum edge fraction is 0.00197%, and maximum power drift is 0.0000197%;
those conservation checks pass but do not rescue its axial-detail failure.

The accepted complex64 production backend was checked against complex128 BL-ASM at B0/V1/V3 and
z=20, 60 and 100 mm. Maximum phase-aligned field L2 difference was 2.714e-6 and maximum intensity
L2 difference was 6.474e-7.

## Z Convergence

Production uses z=0--180 mm with dz=0.25 mm. The representative 0--140 mm comparison against
dz=0.125 mm passed every declared gate.

| case | raw L2 | fixed-power L2 | max radius difference | max boundary shift | ripple-period difference |
|---|---:|---:|---:|---:|---:|
| B0 | 1.568e-5 | 1.214e-5 | 1.528e-4 | 0 mm | 8.921e-4 |
| V1 | 2.696e-5 | 1.509e-5 | 1.700e-5 | 0.125 mm | 8.921e-4 |
| V3 | 6.291e-4 | 2.540e-5 | 6.268e-6 | 0.125 mm | 8.921e-4 |

## Nominal Results

The historical 20--60 mm interval remains a configuration reference only. The geometric estimate
is 0--125.05 mm. Measured intervals and features are:

| case | measured FWHM zone (mm) | strict useful region (mm) | reference radius (um) | median width (um) | median dark-core radius (um) | winding |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 20.00--120.00 | 20.00--120.00 | 11.257 | 22.513 | n/a | 0 |
| V1 | 17.25--119.25 | 17.25--119.25 | 18.671 | 19.496 | 8.943 | 1 |
| V3 | 20.25--119.50 | 20.50--119.25 | 42.882 | 23.701 | 30.305 | 3 |

Across nominal B0/V1/V3, the maximum edge-energy fraction is 4.736e-5 and the maximum propagation
power drift is 4.227e-6.

## Aperture Finding

The soft sensitivity route shortens the strict region to 19.75--94.00 mm (B0), 16.75--94.00 mm
(V1), and 20.25--93.00 mm (V3). The hard diagnostic route gives 23.75--103.50 mm, 20.00--103.50 mm,
and 20.50--103.75 mm respectively.

For B0 over 25--90 mm, detrended ripple RMS is 0.00109 without an additional aperture, 0.00113 for
the soft assumption, and 0.07810 for the hard disk. Peak-to-valley ripple is 0.00448, 0.00470 and
0.34672 respectively. Pronounced axial beading therefore does not survive without the hard stop;
it is a hard-truncation diagnostic, not a nominal experimental prediction.

## Figures And Provenance

The report pack contains 18 PNG/PDF figure pairs under:

```text
outputs/figures/phase2e_final_source_propagation/
```

Primary propagation figures show both the complete simulated +/-5 mm field and a common fixed
+/-0.25 mm detail crop. All primary intensity maps use one global linear scale: no log, dB, gamma,
or per-z normalisation. Native arrays supply all metrics; cubic upsampling is used only to render the
3D surfaces, whose top-down panels use the identical displayed arrays.

The exhaustive paths, plot contracts and hashes are recorded in:

- `00_manifest/final_figure_manifest.json`
- `00_manifest/final_artifact_manifest.json`
- `00_manifest/final_figure_style.json`

Validation tables and the final claim registry are under:

```text
outputs/validation/phase2e_final_propagation/
```

The selected production run used an estimated peak memory of 1.146 GiB. Nine complete route/case
runs took 5,978.96 s; the resolution gate, z-convergence runs and production runs totalled 8,696.75 s.
Upstream Phase 2A/2B/2C hashes are unchanged.

## Remaining Calibration

Report figures are authorised. Exact bench prediction still requires measurement of the physical
stop/aperture presence and radius at the reconstructed-field plane, SLM phase LUT/stroke, exact 4F
iris centre/radius, camera scale and z-stage, and beam/axicon centring.
