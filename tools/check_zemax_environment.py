from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vbb_study.integrations.zemax.availability import classify_exit_code, probe_zemax_environment


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight the optional ZOSPy/OpticStudio Phase 3 backend.")
    parser.add_argument("--json", dest="json_path", type=Path, default=None)
    parser.add_argument("--no-connect", action="store_true", help="Only probe imports/assemblies; do not acquire a session.")
    args = parser.parse_args()
    try:
        status = probe_zemax_environment(try_connections=not args.no_connect)
    except Exception as exc:
        print(f"Unexpected Zemax integration failure: {type(exc).__name__}: {exc}")
        return 5
    rows = (("OS", status.platform), ("Python executable", status.python_executable), ("Python version", status.python_version), ("ZOSPy installed", status.zospy_installed), ("ZOSPy version", status.zospy_version or "-"), ("Python.NET loadable", status.pythonnet_loadable), ("OpticStudio detected", status.opticstudio_detected), ("ZOS-API assemblies loadable", status.zosapi_loadable), ("OpticStudio version", status.opticstudio_version or "-"), ("Licence/session available", status.license_available), ("Standalone connection", status.standalone_connection_available), ("Interactive extension", status.interactive_connection_available))
    width = max(len(label) for label, _ in rows)
    for label, value in rows:
        print(f"{label:<{width}} : {value}")
    if status.diagnostic_messages:
        print("\nDiagnostics:")
        for message in status.diagnostic_messages:
            print(f"  - {message}")
    if args.json_path is not None:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(json.dumps(status.to_dict(), indent=2), encoding="utf-8")
    return classify_exit_code(status)


if __name__ == "__main__":
    raise SystemExit(main())
