# Phase 3A — Zemax / OpticStudio cross-validation

Phase 3A adds **OpticStudio POP as an optional independent validation backend**. It does not replace the canonical Python solver hierarchy and it does not modify accepted Phase 1–2C evidence.

Install the optional environment only on the OpticStudio Windows machine:

```powershell
C:\PhD\.venv2\Scripts\python.exe -m pip install -r requirements-zemax.txt
C:\PhD\.venv2\Scripts\python.exe tools\check_zemax_environment.py
C:\PhD\.venv2\Scripts\python.exe tools\run_zemax_smoke_test.py
```

The repository remains importable and testable without ZOSPy or OpticStudio. A real `.ZOS`/`.ZMX` prescription is treated as read-only evidence: the runner hashes it, makes a work copy, inspects the prescription, and records the source SHA-256. No real-bench validation is claimed until the actual prescription and surface map are supplied.

For B0/V1/V3, the current bridge deliberately blocks arbitrary Python-to-ZBF export unless a documented supported writer is established. A pre-existing, independently generated `.ZBF` may be supplied. This is preferable to reverse engineering a binary format and silently corrupting a field.

The canonical Python side is connected through `python_reference.py`, which adapts the existing Phase 2C vector-Debye benchmark rather than implementing a second Bessel simulator. Use `--python-reference phase2c-vector-focus --confirm-same-plane` only after confirming the mapped Zemax output is that same physical focal plane.
