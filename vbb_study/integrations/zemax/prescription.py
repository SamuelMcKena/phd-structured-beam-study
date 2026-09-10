"""Read-only provenance, work-copy handling, and prescription inspection."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from .errors import ZemaxModelError
from .models import ZemaxModelMetadata


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_working_copy(source: str | Path, work_directory: str | Path) -> tuple[Path, str]:
    src = Path(source).expanduser().resolve()
    if not src.is_file():
        raise ZemaxModelError(f"Zemax model does not exist: {src}")
    if src.suffix.lower() not in {".zos", ".zmx"}:
        raise ZemaxModelError("Expected an OpticStudio .ZOS or .ZMX prescription")
    digest = sha256_file(src)
    work = Path(work_directory).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    dst = work / f"{src.stem}__phase3_working{src.suffix}"
    shutil.copy2(src, dst)
    return dst, digest


def load_surface_map(path: str | Path) -> dict[str, int | None]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ZemaxModelError("surface map must be a JSON object")
    result: dict[str, int | None] = {}
    for key, value in payload.items():
        if value is not None and (not isinstance(value, int) or value < 0):
            raise ZemaxModelError(f"surface map value for {key!r} must be a non-negative integer or null")
        result[str(key)] = value
    return result


def validate_surface_map(surface_map: dict[str, int | None], surface_count: int) -> None:
    for name, index in surface_map.items():
        if index is not None and index >= surface_count:
            raise ZemaxModelError(f"surface map entry {name!r}={index} is outside prescription surface range 0..{surface_count - 1}")


def _safe_value(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name)
    except Exception:
        return default


def inspect_model(system: Any, *, source_path: str | Path, source_sha256: str) -> ZemaxModelMetadata:
    source = Path(source_path).expanduser().resolve()
    mode = str(_safe_value(system, "Mode", _safe_value(_safe_value(system, "SystemData"), "SystemType", "unknown")))
    units_obj = _safe_value(_safe_value(system, "SystemData"), "Units")
    units = str(_safe_value(units_obj, "LensUnits", "unknown"))
    wavelengths: list[dict[str, Any]] = []
    wave_editor = _safe_value(_safe_value(system, "SystemData"), "Wavelengths")
    n_wave = int(_safe_value(wave_editor, "NumberOfWavelengths", 0) or 0)
    for index in range(1, n_wave + 1):
        row = wave_editor.GetWavelength(index)
        wavelength_um = float(_safe_value(row, "Wavelength", float("nan")))
        wavelengths.append({"index": index, "wavelength_um": wavelength_um, "wavelength_m": wavelength_um * 1e-6, "weight": float(_safe_value(row, "Weight", float("nan")))})
    fields: list[dict[str, Any]] = []
    field_editor = _safe_value(_safe_value(system, "SystemData"), "Fields")
    n_fields = int(_safe_value(field_editor, "NumberOfFields", 0) or 0)
    for index in range(1, n_fields + 1):
        row = field_editor.GetField(index)
        fields.append({"index": index, "x": float(_safe_value(row, "X", float("nan"))), "y": float(_safe_value(row, "Y", float("nan"))), "weight": float(_safe_value(row, "Weight", float("nan")))})
    surfaces: list[dict[str, Any]] = []
    apertures: list[dict[str, Any]] = []
    lde = _safe_value(system, "LDE")
    n_surfaces = int(_safe_value(lde, "NumberOfSurfaces", 0) or 0)
    for index in range(n_surfaces):
        surface = lde.GetSurfaceAt(index)
        material = _safe_value(surface, "Material", None)
        if material is None:
            material = _safe_value(_safe_value(surface, "MaterialCell"), "Value", "")
        row = {"index": index, "comment": str(_safe_value(surface, "Comment", "")), "type": str(_safe_value(surface, "TypeName", _safe_value(surface, "Type", "unknown"))), "radius": _safe_value(surface, "Radius", None), "thickness": _safe_value(surface, "Thickness", None), "material": str(material or ""), "semi_diameter": _safe_value(surface, "SemiDiameter", None)}
        surfaces.append(row)
        apertures.append({"index": index, "semi_diameter": row["semi_diameter"]})
    return ZemaxModelMetadata(source_path=str(source), source_sha256=source_sha256, zemax_filename=source.name, sequential_or_nonsequential=mode, system_units=units, wavelength_table=tuple(wavelengths), field_table=tuple(fields), surface_count=n_surfaces, surface_summary=tuple(surfaces), aperture_summary=tuple(apertures), model_modified=False)


def length_unit_to_m(system_units: str) -> float:
    key = system_units.strip().lower().replace(" ", "")
    mapping = {"millimeters": 1e-3, "millimetres": 1e-3, "mm": 1e-3, "centimeters": 1e-2, "centimetres": 1e-2, "cm": 1e-2, "meters": 1.0, "metres": 1.0, "m": 1.0, "inches": 0.0254, "inch": 0.0254, "feet": 0.3048, "foot": 0.3048}
    if key not in mapping:
        raise ZemaxModelError(f"Unsupported/unknown OpticStudio lens unit {system_units!r}; refusing an implicit conversion")
    return mapping[key]
