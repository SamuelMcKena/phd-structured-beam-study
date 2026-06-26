# Stage 9A.1B Downstream Carrier/Stop Characterisation Summary

Starting checkpoint: Stage 9A.3 verified bibliography and evidence layers
(`21ad69c`).

Stage 9A.1B adds a dual-mode calibration contract:

- `direct_fourier_plane_access`
- `downstream_focus_empirical`

The current laboratory default is `downstream_focus_empirical`, because the
installed camera is at the downstream final-focus/output plane. Existing Stage
9A.1 command-domain carrier masks remain valid and unchanged.

## Created

- `vbb_study/digital_twin/downstream_carrier_stop.py`
- `configs/studies/cslm_carrier_stop_characterisation_downstream_v1.json`
- `docs/45_downstream_focus_carrier_stop_characterisation.md`
- `STAGE9A1B_DOWNSTREAM_CARRIER_STOP_CHARACTERISATION_SUMMARY.md`
- `LAB_README_DOWNSTREAM_CARRIER_STOP_SESSION.md`
- `tests/test_stage9a1b_downstream_carrier_stop_characterisation.py`
- `outputs/figures/digital_twin/stage9a1b_downstream_carrier_stop_session_overview.png`

## Boundary

empirical downstream response only; not_direct_fourier_plane_calibration; not_physical_4f_model_validation; no camera model; no inverse correction; no AI; no material response; final_export_allowed=False

## First Bench Action

Run the downstream session: SLM1 flat, SLM2 carrier-only, installed downstream
camera fixed, Fourier stop recorded and varied deliberately where possible,
axicon state recorded, all downstream optics logged as fixed bench state.
