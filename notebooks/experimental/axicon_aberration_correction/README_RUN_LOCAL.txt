Q20 NEW ABERRATION-CORRECTION PIPELINE — LOCAL RUN
=================================================

1. Unzip this folder anywhere on the Windows machine that contains Python.
2. Copy your ORIGINAL raw-data folder named exactly:

     z-scan 2 1010

   into this folder. It should contain 72 BMG files:
   z0_1.bmg ... z17_4.bmg (4 repeats at each of 18 z positions).

3. Double-click RUN_BASELINE_RETRIEVAL.bat.

The first run installs numpy/scipy/pandas/matplotlib/pillow/pytest and then
executes run_q20_miao_retrieval.py.

IMPORTANT:
- With q20_hardware_calibration.json still containing null values, the code
  WILL run the raw BMG per-plane retrieval and create diagnostics/fit tables,
  but it will deliberately block a final SLM2 hardware correction.
- Do not invent calibration values just to get a phase mask.
- Fill q20_hardware_calibration.json only with measured bench quantities.
- Q20_LAB_CALIBRATION_PROTOCOL.md explains each required measurement.

Main outputs are written to:

     outputs/miao_full_q20/

Once full calibration is supplied, the pipeline can additionally write:
- retrieved_full_residual_phase_input_plane_rad.npy
- conjugate_correction_input_plane_rad.npy
- radial_phase_rad.npy
- angular_phase_rows_rad.npy
- slm2_correction_phase_rad.npy (only if all hardware gates are satisfied)
- correction_manifest.json

To verify the packaged code itself, double-click RUN_TESTS.bat.
