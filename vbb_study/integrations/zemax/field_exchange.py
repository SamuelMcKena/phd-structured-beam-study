"""Audited field exchange. ZBF writing is intentionally not reverse engineered."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from .errors import ZemaxFieldExchangeError
from .models import FieldPlane

ZBF_LONGITUDINAL_COMPONENT_EXCHANGE_SUPPORTED = False
ARBITRARY_PYTHON_TO_ZBF_EXPORT_AVAILABLE = False


def assert_zbf_component_supported(component: str) -> None:
    if component.strip().lower() in {"ez", "z", "longitudinal"}:
        raise ZemaxFieldExchangeError("Traditional ZBF exchange is transverse-only for this governed workflow; it cannot validate Ez.")


def validate_zbf_input(path: str | Path) -> Path:
    beam = Path(path).expanduser().resolve()
    if not beam.is_file() or beam.suffix.lower() != ".zbf":
        raise ZemaxFieldExchangeError(f"Expected an existing .ZBF beam file, got: {beam}")
    return beam


def save_field_npz(path: str | Path, plane: FieldPlane) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = {**dict(plane.metadata), "plane_name": plane.plane_name, "source_solver": plane.source_solver, "longitudinal_component_exchange_supported": False}
    arrays: dict[str, object] = {"x_m": plane.x_m, "y_m": plane.y_m, "wavelength_m": np.asarray(plane.wavelength_m), "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True))}
    for name, value in (("Ex", plane.Ex), ("Ey", plane.Ey), ("Ez", plane.Ez)):
        if value is not None:
            arrays[f"{name}_real"] = np.real(value); arrays[f"{name}_imag"] = np.imag(value)
    if plane.intensity is not None:
        arrays["intensity"] = plane.intensity
    if plane.phase is not None:
        arrays["phase"] = plane.phase
    np.savez_compressed(target, **arrays)
    return target


def load_field_npz(path: str | Path) -> FieldPlane:
    source = Path(path).expanduser().resolve()
    with np.load(source, allow_pickle=False) as data:
        meta = json.loads(str(data["metadata_json"])) if "metadata_json" in data else {}
        def component(name: str) -> np.ndarray | None:
            real, imag = f"{name}_real", f"{name}_imag"
            if real not in data or imag not in data:
                return None
            return np.asarray(data[real]) + 1j * np.asarray(data[imag])
        return FieldPlane(x_m=np.asarray(data["x_m"], dtype=float), y_m=np.asarray(data["y_m"], dtype=float), wavelength_m=float(data["wavelength_m"]), Ex=component("Ex"), Ey=component("Ey"), Ez=component("Ez"), intensity=np.asarray(data["intensity"], dtype=float) if "intensity" in data else None, phase=np.asarray(data["phase"], dtype=float) if "phase" in data else None, plane_name=str(meta.pop("plane_name", "unknown")), source_solver=str(meta.pop("source_solver", "unknown")), metadata=meta)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class StagedZbfInput:
    """Temporary copy of a pre-existing ZBF in OpticStudio's required POP folder."""
    def __init__(self, source: str | Path, pop_directory: str | Path):
        self.source = validate_zbf_input(source)
        self.pop_directory = Path(pop_directory).expanduser().resolve()
        self.source_sha256 = sha256_file(self.source)
        self.staged_path = self.pop_directory / f"phase3_{self.source_sha256[:12]}_{self.source.name}"

    def __enter__(self) -> "StagedZbfInput":
        self.pop_directory.mkdir(parents=True, exist_ok=True)
        if self.staged_path.exists():
            raise ZemaxFieldExchangeError(f"Refusing to overwrite existing OpticStudio POP beam file: {self.staged_path}")
        shutil.copy2(self.source, self.staged_path)
        return self

    @property
    def zemax_filename(self) -> str:
        return self.staged_path.name

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        try:
            self.staged_path.unlink(missing_ok=True)
        except OSError as cleanup_error:
            raise ZemaxFieldExchangeError(f"Could not remove staged POP beam file {self.staged_path}: {cleanup_error}") from cleanup_error
