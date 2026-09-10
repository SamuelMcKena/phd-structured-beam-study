"""Deterministic lifetime management for OpticStudio sessions."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .availability import import_zospy
from .errors import ZemaxModelError, ZemaxUnavailableError

ConnectionMode = Literal["standalone", "interactive", "extension"]


@dataclass
class ZemaxSession:
    mode: ConnectionMode = "standalone"
    opticstudio_directory: str | None = None
    _zos: Any = None
    system: Any = None

    def __enter__(self) -> "ZemaxSession":
        try:
            zp = import_zospy()
            kwargs = {}
            if self.opticstudio_directory:
                kwargs["opticstudio_directory"] = self.opticstudio_directory
            self._zos = zp.ZOS(**kwargs)
            connect_mode = "extension" if self.mode in {"interactive", "extension"} else "standalone"
            self.system = self._zos.connect(connect_mode)
        except Exception as exc:
            self._safe_disconnect()
            raise ZemaxUnavailableError(f"Could not establish OpticStudio {self.mode!r} session: {type(exc).__name__}: {exc}") from exc
        return self

    def _safe_disconnect(self) -> None:
        if self._zos is not None:
            try:
                self._zos.disconnect()
            except Exception:
                pass
        self.system = None

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._safe_disconnect()

    @property
    def zospy(self) -> Any:
        return import_zospy()

    def open_model(self, path: str | Path) -> Any:
        if self.system is None:
            raise ZemaxUnavailableError("OpticStudio session is not connected")
        model = Path(path).expanduser().resolve()
        if not model.is_file():
            raise ZemaxModelError(f"Zemax model does not exist: {model}")
        if model.suffix.lower() not in {".zos", ".zmx"}:
            raise ZemaxModelError("Expected an OpticStudio .ZOS or .ZMX prescription")
        try:
            self.system.load(str(model), saveifneeded=False)
        except Exception as exc:
            raise ZemaxModelError(f"OpticStudio could not load {model.name}: {exc}") from exc
        return self.system
