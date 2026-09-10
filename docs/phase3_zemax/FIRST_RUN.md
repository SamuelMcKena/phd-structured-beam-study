# First run

Run on the Windows workstation that has Ansys Zemax OpticStudio:

```powershell
C:\PhD\.venv2\Scripts\python.exe -m compileall -q vbb_study tools tests
C:\PhD\.venv2\Scripts\python.exe -m pytest tests -q
C:\PhD\.venv2\Scripts\python.exe -m pip install -r requirements-zemax.txt
C:\PhD\.venv2\Scripts\python.exe tools\check_zemax_environment.py
C:\PhD\.venv2\Scripts\python.exe tools\run_zemax_smoke_test.py
C:\PhD\.venv2\Scripts\python.exe -m pytest -m zemax -q
git diff --check
```

Expected preflight return codes are `0` ready, `2` optional Python dependency missing, `3` OpticStudio/ZOS-API unavailable, `4` licence or standalone connection unavailable, and `5` unexpected integration failure.

For a real model, copy `config/zemax_surface_map.example.json` to a local mapping file, fill in the actual surface indices after inspecting the prescription, then begin with `--dry-run`. Export a canonical Python focal reference with `tools\export_phase3_python_reference.py` or let the real runner build it with `--python-reference phase2c-vector-focus --confirm-same-plane` after the surfaces are physically matched.
