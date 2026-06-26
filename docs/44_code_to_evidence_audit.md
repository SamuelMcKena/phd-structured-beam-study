# Stage 9A.2 Code-to-Evidence Audit

This audit maps the current repository from implemented code to claim boundary,
evidence need, and next research or measurement task.  It does not add optical
propagation, 4F modelling, camera modelling, inverse correction, neural
networks, or material-response physics.

## Boundary

```text
fourier_filter_physics_available = False
camera_model_enabled = False
material_model_enabled = False
diagnostic_only = True
final_export_allowed = False
```

Literature support for a principle is not the same thing as validation of this
numerical implementation, calibration of this bench, or demonstration of a
fused-silica process outcome.

## Counts

- Canonical active claims: 8
- Placeholder/assumption claims: 5
- Manufacturer-data blockers: 10
- Bench-data blockers: 21
- Literature/source blockers: 8

## Code And Claim Inventory

| item | path | classification | relevance | affects execution | affects lab decisions |
|---|---|---:|---|---:|---:|
| `dt_cslm_route` | `vbb_study/digital_twin/cslm_route.py` | canonical_active | central scalar CSLM -> future 4F -> physical axicon programme | True | True |
| `dt_4f_readiness_inventory` | `vbb_study/digital_twin/bench_inventory.py` | calibration_infrastructure | prevents fake physical 4F activation | False | True |
| `dt_stage9a_acquisition` | `vbb_study/digital_twin/calibration_acquisition.py` | calibration_infrastructure | first lab-data acquisition path | False | True |
| `dt_stage9a1_carrier_masks` | `vbb_study/digital_twin/slm_calibration_masks.py` | calibration_infrastructure | P0 carrier-to-Fourier-plane mapping | False | True |
| `dt_measured_image_metrics` | `vbb_study/digital_twin/measured_image_metrics.py` | calibration_infrastructure | camera z-stack evidence once acquired | False | True |
| `equations_propagation_holography` | `vbb_study/equations/propagation.py; vbb_study/equations/holography.py` | canonical_active | core numerical/phase primitives | True | True |
| `notebook_full_cockpit` | `notebooks/digital_twin/00_full_beam_to_write_cockpit_MVP.ipynb` | canonical_supporting | human-facing canonical status view | False | True |
| `configs_hardware_and_studies` | `configs/hardware/*.json; configs/studies/*.json` | calibration_infrastructure | evidence capture and readiness gating | False | True |
| `materials_proxy_branch` | `vbb_study/equations/materials.py; vbb_study/config.py; notebooks/materials/*.ipynb; docs/03_materials_application.md` | legacy_retained | not part of immediate fused-silica CSLM calibration | False | False |
| `vector_branch` | `vbb_study/equations/vector_jones.py; vbb_study/publication/vector.py; notebooks/vector/*.ipynb` | experimental_development | excluded from scalar first lab path | False | False |
| `polygonal_discrete_branch` | `vbb_study/equations/polygonal.py; vbb_study/studies/*polygonal*; notebooks/advanced/02_*.ipynb; notebooks/advanced/03_*.ipynb` | experimental_development | not prerequisite for scalar CSLM/4F calibration | False | False |
| `capsule_weld_branch` | `vbb_study/equations/capsule_geometry.py; vbb_study/publication/capsule.py; notebooks/advanced/01_capsule_weld_feature_design.ipynb` | experimental_development | excluded from immediate optical calibration | False | False |
| `generated_outputs` | `outputs/**` | generated_output | evidence/preview outputs only | False | False |

## Physical And Numerical Claim Register

Full structured records are stored in
`configs/evidence/project_claim_registry.json`.

| claim ID | current status | physics status | evidence status | next action |
|---|---|---|---|---|
| `angular_spectrum_or_bl_asm_propagation` | implemented_active | numerically_modelled | needs_verified_source | literature/source verification |
| `finite_sampling_and_aliasing_control` | implemented_active | numerically_modelled | needs_verified_source | observed order locations versus command carrier |
| `phase_only_slm_mask_generation` | implemented_active | physically_modelled | needs_manufacturer_data | phase response calibration, zero/order power split |
| `phase_quantisation_and_grayscale_export` | implemented_active | numerically_modelled | needs_manufacturer_data | grayscale-to-phase response at 1030 nm |
| `command_domain_carrier_grating` | implemented_active | measurement_only | needs_bench_measurement | carrier cycles to Fourier-plane order displacement |
| `pixelated_slm_zero_order_and_unwanted_orders` | planned_future | future_not_implemented | needs_bench_measurement | zero/+1/residual order power fractions |
| `physical_fourier_filtering_future_route` | not_implemented | future_not_implemented | needs_bench_measurement | lens positions, Fourier-stop centre/radius |
| `ideal_selected_order_handoff` | implemented_diagnostic_only | diagnostic_placeholder | assumption_declared | measured Fourier-plane order selection and residual-order rejection |
| `slm1_to_slm2_propagation` | implemented_active | physically_modelled | needs_bench_measurement | actual SLM1-to-SLM2 distance and relay geometry |
| `slm_registration_and_coordinate_frames` | implemented_diagnostic_only | record_only | needs_bench_measurement | SLM2-to-lab transform, Fourier-plane frame |
| `physical_axicon_bessel_conversion` | implemented_benchmark_only | physically_modelled | needs_manufacturer_data | axicon position/orientation, z-stack validation |
| `physical_axicon_aperture_and_decentre` | implemented_benchmark_only | physically_modelled | needs_bench_measurement | beam-to-axicon centring, axicon axial/lateral position |
| `vortex_phase_generation` | implemented_active | physically_modelled | needs_verified_source | measured charge-dependent annular field |
| `energy_and_fluence_accounting` | implemented_active | numerically_modelled | assumption_declared | pulse energy before optics, transmission through actual chain |
| `ring_centre_radius_dark_core_and_uniformity_metrics` | implemented_active | numerically_modelled | needs_bench_measurement | camera z-stack images and scale/orientation calibration |
| `camera_z_stack_acquisition` | implemented_diagnostic_only | measurement_only | needs_bench_measurement | raw camera files, z positions |
| `camera_coordinate_calibration` | implemented_diagnostic_only | measurement_only | needs_bench_measurement | magnification, orientation |
| `multi_plane_phase_retrieval_future` | planned_future | future_not_implemented | needs_verified_source | calibrated multi-plane z-stack |
| `effective_aberration_inference_future` | planned_future | future_not_implemented | needs_verified_source | repeatable calibrated z-stacks |
| `zernike_or_phase_conjugate_correction_future` | planned_future | future_not_implemented | needs_verified_source | identified correction target and validation capture |
| `neural_fast_estimator_future` | not_implemented | future_not_implemented | needs_verified_source | large calibrated dataset with uncertainty labels |
| `fused_silica_application_boundary` | not_implemented | future_not_implemented | needs_bench_measurement | fused-silica sample/process outcomes, microscopy/metrology |
| `legacy_crznse_material_proxy_branch` | implemented_legacy | diagnostic_placeholder | assumption_declared | literature/source verification |
| `vector_beam_branch` | implemented_legacy | numerically_modelled | needs_verified_source | literature/source verification |
| `hexagonal_polygonal_discrete_beam_branch` | implemented_legacy | numerically_modelled | assumption_declared | literature/source verification |
| `capsule_or_weld_feature_geometry_branch` | implemented_legacy | diagnostic_placeholder | assumption_declared | literature/source verification |

## Research Backlog

Full structured records are stored in `configs/evidence/research_backlog.json`.

| research ID | priority | horizon | title | current gap | deliverable |
|---|---|---|---|---|---|
| `P0_SLM_SPEC` | P0 | immediate | Actual SLM make/model, pitch, active area, resolution, orientation | SLM geometry and phase response are placeholders. | manufacturer specification pack plus bench orientation note |
| `P0_SLM_PHASE` | P0 | immediate | SLM phase response and polarisation requirement at 1030 nm | Grayscale command is not yet calibrated to phase. | phase-response calibration plan and first measurement |
| `P0_LENS_GEOMETRY` | P0 | immediate | Lens focal lengths, clear apertures, coatings, and real positions | 4F lens geometry is declaration only. | measured lens table |
| `P0_CARRIER_MAPPING` | P0 | immediate | Carrier command to Fourier-plane order-position mapping | Command cycles are not yet physical Fourier-plane coordinates. | first Fourier-plane carrier sweep dataset |
| `P0_FOURIER_STOP` | P0 | immediate | Fourier-stop geometry, centre, radius, and adjustment convention | Stop model cannot be placed without measured geometry. | stop-centre/radius measurement log |
| `P0_CAMERA` | P0 | immediate | Camera pixel pitch, magnification, orientation, linearity, saturation | Pixel metrics cannot be physical metrics yet. | camera calibration sheet |
| `P0_AXICON` | P0 | immediate | Physical axicon specification, aperture, orientation, and bench position | Benchmark uses demo axicon parameters until measured. | axicon spec and alignment log |
| `P0_INPUT_BEAM` | P0 | immediate | Input beam size, ellipticity, centring, and polarisation at relevant planes | Source field is demo geometry. | beam profiling sheet |
| `P1_BLASM` | P1 | pre_4f_model | BL-ASM/ASM sampling validity for carrier, aperture, and distances | Need source-backed validity criteria. | validated sampling memo |
| `P1_4F_MODEL` | P1 | pre_4f_model | Finite-aperture thin-lens / 4F modelling | No component-owned lens model exists. | 4F model design note |
| `P1_SLM_DIFFRACTION` | P1 | pre_4f_model | Pixelated SLM diffraction and zero-order behaviour | Order power split unknown. | order-efficiency evidence plan |
| `P1_PHASE_QUANT` | P1 | pre_4f_model | Phase quantisation and calibration effects | Quantisation exists but physical response unverified. | phase quantisation sensitivity note |
| `P1_AXICON_TOL` | P1 | pre_4f_model | Axicon tolerance, decentre, tilt, apex quality, and aperture behaviour | Tolerance model needs literature/manufacturer context. | axicon tolerance register |
| `P1_CAMERA_PROFILING` | P1 | pre_4f_model | Camera-based annular beam profiling and z-stack metrology | Metric reliability needs metrology backing. | metric validity memo |
| `P2_PHASE_RETRIEVAL` | P2 | pre_inverse_correction | Multi-plane phase retrieval | No phase retrieval implemented. | algorithm literature review |
| `P2_PHASE_DIVERSITY` | P2 | pre_inverse_correction | Phase-diversity wavefront sensing | No phase-diversity inference implemented. | phase-diversity feasibility note |
| `P2_IDENTIFIABILITY` | P2 | pre_inverse_correction | Effective aberration versus component-root-cause identifiability | Need to know what can be inferred from intensity-only data. | identifiability risk register |
| `P2_ZERNIKE_BASIS` | P2 | pre_inverse_correction | Zernike basis and normalisation | Basis conventions must be locked before correction. | basis convention doc |
| `P2_SLM_CONJ` | P2 | pre_inverse_correction | SLM conjugate-phase correction | No correction map is validated. | correction validation plan |
| `P2_SENSORLESS` | P2 | pre_inverse_correction | Sensorless optimisation | No optimisation loop exists. | sensorless optimisation literature note |
| `P2_NEURAL` | P2 | later_research | Synthetic-to-real training and uncertainty for neural estimators | No dataset or uncertainty model exists. | AI feasibility note |
| `P3_FS_REGIMES` | P3 | pre_fused_silica_pilot | Fused-silica internal modification regimes | No fused-silica process evidence in repo. | fused-silica literature table |
| `P3_FS_INTERFACE` | P3 | pre_fused_silica_pilot | Bessel/vortex-Bessel propagation at the sample interface | No sample-interface model is active. | interface research note |
| `P3_TGV` | P3 | pre_fused_silica_pilot | TGV/channel formation and etching | TGV etch response is not modelled. | TGV evidence plan |
| `P3_WAVEGUIDE` | P3 | pre_fused_silica_pilot | Waveguide-writing regimes and characterisation | Waveguide outcomes are not predicted. | waveguide evidence plan |
| `P3_WELDING` | P3 | pre_fused_silica_pilot | Ultrafast glass welding / symmetric weld-feature conditions | Weld-feature branch is geometry only. | welding evidence plan |
| `LEGACY_CRZNSE` | P3 | later_research | CrZnSe legacy materials proxy work | Retained but excluded from fused-silica decisions. | legacy separation note |
| `LEGACY_VECTOR` | P3 | later_research | Vector beams and Jones modelling | Optional vector branch not needed for scalar campaign. | optional vector review |
| `LEGACY_POLYGONAL` | P3 | later_research | Hexagonal / polygonal / discrete beam studies | Exploratory branch not needed for first lab calibration. | optional beam-shaping review |
| `LEGACY_CAPSULE` | P3 | later_research | Capsule and weld-feature geometry studies | Geometry proxy not material physics. | optional geometry review |

## Fused-Silica / Cr:ZnSe Separation

The immediate scalar CSLM -> future 4F -> physical-axicon calibration campaign
is material-neutral until the fused-silica application profile is populated from
verified literature and bench evidence.  The legacy Cr:ZnSe branch is retained
for historical planning context only.

This branch contains CrZnSe-specific proxy assumptions and is not validated for
fused-silica TGV, waveguide, welding, or modification predictions.

- Material-neutral current path: `vbb_study/digital_twin/cslm_route.py`,
  `bench_inventory.py`, `calibration_acquisition.py`,
  `slm_calibration_masks.py`, and `measured_image_metrics.py`.
- Fused-silica template: `configs/materials/fused_silica_evidence_template.json`
  with null/unknown values only.
- Cr:ZnSe-specific or material-proxy paths: `vbb_study/config.py`,
  `vbb_study/equations/materials.py`, `vbb_study/publication/materials.py`,
  `notebooks/materials/*.ipynb`, `docs/03_materials_application.md`.
- These must not be used for fused-silica decision-making until replaced by
  fused-silica evidence and bench validation.

## Legacy Branches Retained But Excluded From Immediate Lab Path

- `materials_proxy_branch` (legacy_retained): vbb_study/equations/materials.py; vbb_study/config.py; notebooks/materials/*.ipynb; docs/03_materials_application.md -- not part of immediate fused-silica CSLM calibration
- `vector_branch` (experimental_development): vbb_study/equations/vector_jones.py; vbb_study/publication/vector.py; notebooks/vector/*.ipynb -- excluded from scalar first lab path
- `polygonal_discrete_branch` (experimental_development): vbb_study/equations/polygonal.py; vbb_study/studies/*polygonal*; notebooks/advanced/02_*.ipynb; notebooks/advanced/03_*.ipynb -- not prerequisite for scalar CSLM/4F calibration
- `capsule_weld_branch` (experimental_development): vbb_study/equations/capsule_geometry.py; vbb_study/publication/capsule.py; notebooks/advanced/01_capsule_weld_feature_design.ipynb -- excluded from immediate optical calibration

## Evidence Registers

- Literature search plan:
  `configs/evidence/literature_search_plan.json`
- Manufacturer evidence register:
  `configs/evidence/manufacturer_evidence_register.json`
- Bench evidence register:
  `configs/evidence/bench_evidence_register.json`
- Bibliography placeholder:
  `references/structured_beam_methods.bib`

No BibTeX entry is added until DOI/publisher metadata is verified from the
actual source.
