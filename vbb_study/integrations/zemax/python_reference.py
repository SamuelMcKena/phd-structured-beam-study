"""Adapters from accepted Phase 2C fields into the Phase 3 comparison contract."""
from __future__ import annotations

from .models import FieldPlane

PHASE3_CASES = ("G0", "B0", "V1", "V3")


def build_phase2c_vector_focus_reference(case_id: str, *, high_resolution: bool = False) -> FieldPlane:
    """Return the existing Phase 2C vector-Debye focal reference without duplicating physics.

    Only compare it with a Zemax output surface representing the same physical plane.
    """
    case = str(case_id).upper()
    if case not in PHASE3_CASES:
        raise ValueError(f"Phase 3 Zemax vortex scope is {PHASE3_CASES}; got {case_id!r}")
    from vbb_study.digital_twin.phase2a_contracts import canonical_hardware_manifest, hardware_value
    from vbb_study.digital_twin.phase2c_objective_interface import Phase2CConfig, run_phase2c_benchmark
    config = Phase2CConfig.high_resolution_hero_preset() if high_resolution else Phase2CConfig.validation_preset()
    benchmark = run_phase2c_benchmark(config, _include_grid_convergence=False)
    objective = benchmark.objective_cases[case]
    vector = objective.vector
    wavelength_m = float(hardware_value(canonical_hardware_manifest(), "wavelength_m"))
    return FieldPlane(x_m=vector.x_m, y_m=vector.y_m, wavelength_m=wavelength_m, Ex=vector.Ex, Ey=vector.Ey, Ez=vector.Ez, intensity=vector.intensity, plane_name="phase2c_matched_objective_focal_plane_z0", source_solver="vector_debye", metadata={"case_id": case, "reference_source": "vbb_study.digital_twin.phase2c_objective_interface.run_phase2c_benchmark", "phase2c_outcome": benchmark.outcome, "mapping_mode": config.mapping_mode, "canonical_solver": "vector_debye", "validation_backend": "none", "same_plane_confirmation_required_before_zemax_comparison": True, "longitudinal_component_present_in_python_reference": True, "longitudinal_component_exchange_supported_by_zbf": False})
