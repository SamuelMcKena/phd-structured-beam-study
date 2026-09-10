"""ZOSPy Physical Optics Propagation wrapper with explicit settings and units."""
from __future__ import annotations

from typing import Any

import numpy as np

from .availability import import_zospy
from .errors import ZemaxIntegrationError
from .models import PopRequest, PopResult


def _as_dataframe(payload: Any) -> Any:
    data = getattr(payload, "data", payload)
    if hasattr(data, "to_numpy"):
        return data
    if isinstance(data, (list, tuple)) and data:
        item = data[0]
        nested = getattr(item, "data", item)
        if hasattr(nested, "to_numpy"):
            return nested
    raise ZemaxIntegrationError("ZOSPy POP result did not expose a tabular data grid")


def _numeric_axis(values: Any, *, name: str, unit_to_m: float) -> np.ndarray:
    try:
        axis = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ZemaxIntegrationError(f"POP {name} coordinates were not numeric; refusing to invent physical axes") from exc
    if axis.ndim != 1 or axis.size < 2 or not np.all(np.isfinite(axis)):
        raise ZemaxIntegrationError(f"POP {name} coordinates are invalid")
    if np.all(np.diff(axis) < 0):
        axis = axis[::-1]
    if not np.all(np.diff(axis) > 0):
        raise ZemaxIntegrationError(f"POP {name} coordinates are not monotonic")
    return axis * float(unit_to_m)


def run_pop(system: Any, request: PopRequest, *, system_length_unit_to_m: float) -> PopResult:
    request.validate()
    zp = import_zospy()
    if system_length_unit_to_m <= 0:
        raise ValueError("system_length_unit_to_m must be positive")
    beam_parameters = None if request.beam_parameters_m is None else {key: float(value) / system_length_unit_to_m for key, value in request.beam_parameters_m.items()}
    kwargs = dict(wavelength=request.wavelength_index, field=request.field_index, start_surface=request.start_surface, end_surface=request.end_surface, use_polarization=request.polarization, beam_type=request.beam_type, beam_file=request.beam_file, beam_parameters=beam_parameters, x_sampling=request.x_sampling, y_sampling=request.y_sampling, x_width=request.x_width_m / system_length_unit_to_m, y_width=request.y_width_m / system_length_unit_to_m, use_total_power=request.use_total_power, total_power=request.total_power, use_peak_irradiance=request.use_peak_irradiance, peak_irradiance=request.peak_irradiance, data_type=request.data_type, save_output_beam=request.save_output_beam, output_beam_file=request.output_beam_file, save_beam_at_all_surfaces=request.save_beam_at_all_surfaces, auto_calculate_beam_sampling=request.auto_calculate_beam_sampling)
    try:
        analysis = zp.analyses.physicaloptics.PhysicalOpticsPropagation(**kwargs)
        result = analysis.run(system, oncomplete="Close")
        frame = _as_dataframe(result)
    except Exception as exc:
        raise ZemaxIntegrationError(f"OpticStudio POP failed: {type(exc).__name__}: {exc}") from exc
    irradiance = np.asarray(frame.to_numpy(), dtype=float)
    x_raw = np.asarray(frame.columns)
    y_raw = np.asarray(frame.index)
    x = _numeric_axis(x_raw, name="x", unit_to_m=system_length_unit_to_m)
    y = _numeric_axis(y_raw, name="y", unit_to_m=system_length_unit_to_m)
    if np.all(np.diff(np.asarray(frame.index, dtype=float)) < 0):
        irradiance = irradiance[::-1, :]
    if np.all(np.diff(np.asarray(frame.columns, dtype=float)) < 0):
        irradiance = irradiance[:, ::-1]
    return PopResult(x_m=x, y_m=y, irradiance=irradiance, metadata={"analysis": "Zemax OpticStudio Physical Optics Propagation", "validation_backend": "zemax_pop", "canonical_solver_replaced": False, "system_length_unit_to_m": float(system_length_unit_to_m), "pop_request": request.to_dict(), "longitudinal_component_exchange_supported": False}, warnings=["POP irradiance result is not experimental validation and does not validate Ez."])
