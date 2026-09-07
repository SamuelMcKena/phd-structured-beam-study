# PHASE 2A Error-Injection Plane Registry

**Status:** authoritative plane/operation registry for current laboratory perturbations.

A physical upstream perturbation acts on the **complex field** at its declared plane before downstream propagation. Camera noise and display shifts remain post-propagation diagnostics and cannot stand in for beam tilt, decentre, misalignment, aperture diffraction, or wavefront correction. Module-level evidence authority is additionally enforced by `vbb_study/digital_twin/legacy_error_policy.py`.

| error_id | injection plane | operator | field? | timing | class | routes | status |
|---|---|---|---:|---|---|---|---|
| `input_beam_decentre` | before SLM1 | translate input amplitude `E(x-dx,y-dy)` | true | before | physical | canonical scalar/vector | active |
| `input_tilt` | before SLM1 | multiply by `exp(i(kx*x+ky*y))` | true | before | physical | canonical scalar/vector | active |
| `hologram_offset` | corresponding SLM phase plane | transform commanded mask coordinates before pixelation | true | before | physical | canonical scalar/vector | active |
| `slm_phase_error` | SLM phase plane | measured/declared deterministic phase error `delta_phi(x,y)` | true | before | physical/calibration-limited | canonical scalar/vector | active |
| `slm_quantisation` | SLM phase plane | nearest allowed phase level | true | before | physical sensitivity | `vbb_study.slm_model` | active |
| `slm_fill_factor_throughput` | SLM plane | uniform `sqrt(FF)` amplitude factor | true | before | throughput-only | legacy component-plane compatibility | **restricted: no morphology claim** |
| `slm_pixel_aperture_deadspace` | SLM plane | resolved binary pixel aperture or coherent active/dead-space field | true | before | physical | `vbb_study.slm_model` | canonical for morphology |
| `iris_offset_radius` | 4F Fourier plane | translate/resize hard spectral aperture before inverse relay propagation | true | before | physical | explicit 4F route | active |
| `axicon_decentre` | axicon plane | translate physical conical/apex coordinates | true | before | physical | canonical system route | active |
| `axicon_tilt` | tilted axicon plane | rotated-plane propagation to optic, axicon transmission, rotate back | true | before | scalar physical sensitivity | canonical system route | calibration/model limited for absolute vector refraction |
| `axicon_tip_rounding` | axicon plane | rounded/hyperboloidal sag defect relative to sharp cone | true | before | physical sensitivity | canonical system route | calibration limited by measured apex profile |
| `objective_pupil_clipping` | objective pupil | multiply by physical pupil mask before downstream propagation | true | before | physical | objective/vector route | active |
| `low_order_aberration` | declared pupil/aberrating plane | multiply by `exp(i*2*pi*sum(c_j Z_j))` | true | before | generic wavefront sensitivity | scalar/vector | active; not a surrogate for a named optic without evidence |
| `sample_interface_tilt` | sample surface | tilted dielectric-interface operator | true | before | physical | through-sample/vector route | calibration_limited |
| `camera_noise` | camera plane | seeded detector-noise operator on measured intensity | false | after | diagnostic/detector | observation routes | active; never optical correction evidence |
| `post_processing_display_shift` | display only | translate rendered intensity array | false | after | diagnostic | display routes | active; never physical evidence |

Machine-readable module authority: `vbb_study/digital_twin/legacy_error_policy.py`.

Machine-readable per-error taxonomy: `vbb_study/digital_twin/vortex_system_error_matrix.py`.
