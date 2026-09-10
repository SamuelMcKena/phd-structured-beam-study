"""Environment probing without making ZOSPy a mandatory dependency."""
from __future__ import annotations

import importlib.util
import platform
import sys
from importlib import import_module, metadata
from typing import Any

from .models import ZemaxAvailability


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def import_zospy() -> Any:
    """Import ZOSPy lazily so the ordinary repository never depends on it."""
    if importlib.util.find_spec("zospy") is None:
        raise ImportError("ZOSPy is not installed. Install requirements-zemax.txt.")
    return import_module("zospy")


def classify_exit_code(status: ZemaxAvailability) -> int:
    if not status.zospy_installed:
        return 2
    if not status.opticstudio_detected or not status.zosapi_loadable:
        return 3
    if not status.license_available or not status.standalone_connection_available:
        return 4
    return 0


def probe_zemax_environment(*, try_connections: bool = True) -> ZemaxAvailability:
    messages: list[str] = []
    system_name = platform.system()
    zospy_version = _package_version("zospy")
    zospy_installed = zospy_version is not None or importlib.util.find_spec("zospy") is not None
    pythonnet_loadable = importlib.util.find_spec("clr") is not None
    if not zospy_installed:
        messages.append("ZOSPy is not installed. Install requirements-zemax.txt.")
        return ZemaxAvailability(platform=system_name, python_version=platform.python_version(), python_executable=sys.executable, zospy_installed=False, zospy_version=None, pythonnet_loadable=pythonnet_loadable, diagnostic_messages=tuple(messages))
    if system_name != "Windows":
        messages.append("OpticStudio/ZOS-API live use requires the Windows OpticStudio host; core Publication_Study remains usable.")
        return ZemaxAvailability(platform=system_name, python_version=platform.python_version(), python_executable=sys.executable, zospy_installed=True, zospy_version=zospy_version, pythonnet_loadable=pythonnet_loadable, diagnostic_messages=tuple(messages))

    zos = None
    opticstudio_detected = False
    zosapi_loadable = False
    opticstudio_version = None
    standalone = False
    interactive = False
    license_available = False
    try:
        zp = import_zospy()
        zos = zp.ZOS()
        opticstudio_detected = True
        zosapi_loadable = getattr(zos, "ZOSAPI", None) is not None
        if not zosapi_loadable:
            messages.append("ZOSPy imported, but ZOS-API assemblies were not exposed by ZOS().")
    except Exception as exc:
        messages.append(f"OpticStudio/ZOS-API detection failed: {type(exc).__name__}: {exc}")
        return ZemaxAvailability(platform=system_name, python_version=platform.python_version(), python_executable=sys.executable, zospy_installed=True, zospy_version=zospy_version, pythonnet_loadable=pythonnet_loadable, diagnostic_messages=tuple(messages))

    if try_connections and zos is not None:
        try:
            oss = zos.connect("standalone")
            standalone = oss is not None
            license_available = standalone
            application = getattr(zos, "Application", None)
            version = getattr(application, "Version", None) if application is not None else None
            if version is None and hasattr(zos, "version"):
                version = getattr(zos, "version")
            opticstudio_version = None if version is None else str(version)
        except Exception as exc:
            messages.append(f"Standalone connection unavailable: {type(exc).__name__}: {exc}")
        finally:
            try:
                zos.disconnect()
            except Exception:
                pass
        try:
            oss = zos.connect("extension")
            interactive = oss is not None
        except Exception as exc:
            messages.append("Interactive extension unavailable (this is normal unless OpticStudio is open with Programming > Interactive Extension enabled): " f"{type(exc).__name__}: {exc}")
        finally:
            try:
                zos.disconnect()
            except Exception:
                pass

    if standalone:
        messages.append("Standalone OpticStudio connection is available.")
    return ZemaxAvailability(platform=system_name, python_version=platform.python_version(), python_executable=sys.executable, zospy_installed=True, zospy_version=zospy_version, pythonnet_loadable=pythonnet_loadable, opticstudio_detected=opticstudio_detected, zosapi_loadable=zosapi_loadable, opticstudio_version=opticstudio_version, license_available=license_available, standalone_connection_available=standalone, interactive_connection_available=interactive, diagnostic_messages=tuple(messages))
