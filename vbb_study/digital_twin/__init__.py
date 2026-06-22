"""
vbb_study.digital_twin
======================

Beam-to-write digital twin — layered simulation package.

Stage status:
    8B  energy_accounting, exposure_bookkeeping  [IMPLEMENTED]
    8C  field_coupling, field_fluence, figures   [IMPLEMENTED]
    8D  3D beam-to-sample visualiser             [planned]
    8E  writing trajectory / dose accumulation   [planned]
    8F+ lab realism, nonlinear, thermal          [planned]
    8G  calibration ingestion                    [planned]

Allowed model statuses at Stage 8B-8C:
    optical_prediction
    energy_accounting_prediction
    fluence_prediction
    exposure_bookkeeping
    diagnostic_preview

Forbidden at Stage 8B-8C:
    fluence_threshold_proxy
    dose_accumulation_proxy
    uncalibrated_material_response_proxy
    calibrated_material_prediction
    experimentally_validated_prediction
"""

from vbb_study.digital_twin.energy_accounting import (
    LaserSource,
    OpticalComponent,
    EnergyLedgerRow,
    EnergyLedger,
    FluenceEstimate,
    PeakIntensityEstimate,
    average_power_w,
    validate_fraction,
    compute_energy_ledger,
    fresnel_normal_incidence_transmission,
    energy_fraction_after_components,
    fluence_from_effective_area_j_cm2,
    scale_intensity_to_fluence_j_cm2,
    peak_fluence_j_cm2,
    peak_intensity_w_cm2,
)
from vbb_study.digital_twin.exposure_bookkeeping import (
    pulse_spacing_um,
    pulses_per_spot,
    dose_per_unit_length_proxy,
    static_exposure_total_energy_uJ,
    line_exposure_summary,
)
from vbb_study.digital_twin.field_coupling import (
    MissingOpticalFieldError,
    InvalidOpticalFieldError,
    UnsupportedSurfaceFieldError,
    OpticalFieldPlane,
    OpticalFieldStack,
    validate_intensity_plane,
    validate_intensity_stack,
    plane_from_arrays,
    stack_from_arrays,
    extract_plane_from_surfacefield,
    extract_stack_from_surfacefield,
)
from vbb_study.digital_twin.field_fluence import (
    FluencePlaneResult,
    FluenceStackResult,
    transverse_integral_um2,
    integrated_energy_uJ_from_fluence,
    scale_plane_to_fluence,
    scale_stack_to_fluence,
    peak_intensity_from_fluence_result,
    field_fluence_summary,
)
from vbb_study.digital_twin.field_figures import (
    plot_stage8c_field_fluence_preview,
    CaveatsRequiredError,
)
from vbb_study.digital_twin.lab_realism_controls import (
    LabStageControl,
    LabStageResult,
    LabRealismReport,
    REQUIRED_STAGE_NAMES,
    ALLOWED_STATUS_LEVELS,
    default_lab_controls,
    validate_future_physics_disabled,
    build_laser_source_from_controls,
    build_energy_components_from_controls,
    build_energy_ledger_from_controls,
    build_exposure_summary_from_controls,
    build_lab_realism_report,
)
from vbb_study.digital_twin.lab_perturbations import (
    ControlClassification,
    PerturbationResult,
    CONTROL_CLASSIFICATIONS,
    AFFECTED_OUTPUTS,
    PHYSICAL_PLACEMENTS,
    IMPLEMENTATION_STAGES,
    stage8c3_default_controls,
    classification_for_control,
    classify_lab_controls,
    classification_rows_for_controls,
    physical_placement_for_control,
    implementation_stage_for_control,
    physical_placement_rows_for_controls,
    enabled_uncoupled_controls,
    apply_lab_perturbations_to_stack,
)
from vbb_study.digital_twin.active_realism_metrics import (
    SensitivityScenario,
    SensitivitySweepResult,
    compute_degradation_metrics,
    metric_delta_table,
    build_stage8c3_sensitivity_scenarios,
    compute_misalignment_sensitivity_sweep,
    plot_baseline_vs_perturbed_comparison,
    plot_misalignment_sensitivity_sweep,
)
from vbb_study.digital_twin.cockpit_dashboard import (
    build_cockpit_summary,
    compute_peak_location_diagnostics,
    choose_display_plane,
    build_warning_flags,
    compute_overall_status,
    build_beam_path_strip,
    plot_integrated_cockpit_dashboard,
    make_interpretation_text,
)
from vbb_study.digital_twin.component_plane_states import (
    ComponentPlaneState,
    PropagatedFieldStack,
    field_power,
)
from vbb_study.digital_twin.component_plane_pipeline import (
    ComponentPlaneConfig,
    ComponentPlaneRun,
    run_component_plane_pipeline,
    WARNING_ONLY_CONTROLS,
)
from vbb_study.digital_twin.component_plane_metrics import (
    ComponentPlaneScenario,
    ComponentPlaneScenarioResult,
    ResponseCurveFamily,
    ResponseCurveResult,
    DIAGNOSTIC_SWEEP_LABEL,
    stack_to_fluence,
    compute_axis_tracking,
    compute_energy_throughput,
    classify_translation_vs_deformation,
    build_component_plane_scenarios,
    run_component_plane_scenario,
    build_response_curve_families,
    run_response_curve,
    plot_component_plane_reality_preview,
)
from vbb_study.digital_twin.component_plane_validation import (
    canonical_free_space_reference,
    zero_control_equivalence,
    compute_energy_audit,
    validate_beam_tilt,
    fov_convergence_check,
)
from vbb_study.digital_twin.component_plane_figures import (
    plot_reference_plane_energy_axis_validation,
    plot_individual_sensitivity_atlas,
    plot_fov_convergence_check,
    plot_annular_axis_tracking_validation,
    plot_individual_response_curves,
    plot_free_space_study_summary,
    DEFAULT_ATLAS_SCENARIOS,
)
from vbb_study.digital_twin.annular_axis_tracking import (
    RAW_PEAK_LABEL,
    AnnularAxisEstimate,
    estimate_annular_axis,
    track_axis_trajectory,
)
from vbb_study.digital_twin.route_aware_axicon import (
    DIAGNOSTIC_GEOMETRY_NOTE,
    PHYSICAL_LOCATIONS,
    REPRESENTED_PHYSICAL_AXICON_LOCATIONS,
    RouteGraphNode,
    RoutePerturbationRecord,
    RouteAwareAxiconConfig,
    RouteAwareAxiconRun,
    AxiconAlignmentSweepFamily,
    AxiconAlignmentSweepResult,
    physical_axicon_route_graph,
    build_route_perturbation_records,
    holographic_slm_route_declarations,
    physical_axicon_transmission,
    run_route_aware_axicon_pipeline,
    build_axicon_alignment_sweep_families,
    run_axicon_alignment_sweep,
    plot_route_aware_axicon_pipeline,
    plot_upstream_vs_post_axicon_tilt_comparison,
    plot_axicon_alignment_sensitivity_atlas,
)

__all__ = [
    # energy accounting
    "LaserSource",
    "OpticalComponent",
    "EnergyLedgerRow",
    "EnergyLedger",
    "FluenceEstimate",
    "PeakIntensityEstimate",
    "average_power_w",
    "validate_fraction",
    "compute_energy_ledger",
    "fresnel_normal_incidence_transmission",
    "energy_fraction_after_components",
    "fluence_from_effective_area_j_cm2",
    "scale_intensity_to_fluence_j_cm2",
    "peak_fluence_j_cm2",
    "peak_intensity_w_cm2",
    # exposure bookkeeping
    "pulse_spacing_um",
    "pulses_per_spot",
    "dose_per_unit_length_proxy",
    "static_exposure_total_energy_uJ",
    "line_exposure_summary",
    # field coupling (Stage 8C)
    "MissingOpticalFieldError",
    "InvalidOpticalFieldError",
    "UnsupportedSurfaceFieldError",
    "OpticalFieldPlane",
    "OpticalFieldStack",
    "validate_intensity_plane",
    "validate_intensity_stack",
    "plane_from_arrays",
    "stack_from_arrays",
    "extract_plane_from_surfacefield",
    "extract_stack_from_surfacefield",
    # field fluence (Stage 8C)
    "FluencePlaneResult",
    "FluenceStackResult",
    "transverse_integral_um2",
    "integrated_energy_uJ_from_fluence",
    "scale_plane_to_fluence",
    "scale_stack_to_fluence",
    "peak_intensity_from_fluence_result",
    "field_fluence_summary",
    # field figures (Stage 8C)
    "plot_stage8c_field_fluence_preview",
    "CaveatsRequiredError",
    # lab realism controls (Stage 8C.1)
    "LabStageControl",
    "LabStageResult",
    "LabRealismReport",
    "REQUIRED_STAGE_NAMES",
    "ALLOWED_STATUS_LEVELS",
    "default_lab_controls",
    "validate_future_physics_disabled",
    "build_laser_source_from_controls",
    "build_energy_components_from_controls",
    "build_energy_ledger_from_controls",
    "build_exposure_summary_from_controls",
    "build_lab_realism_report",
    # active lab realism coupling (Stage 8C.3)
    "ControlClassification",
    "PerturbationResult",
    "CONTROL_CLASSIFICATIONS",
    "AFFECTED_OUTPUTS",
    "PHYSICAL_PLACEMENTS",
    "IMPLEMENTATION_STAGES",
    "stage8c3_default_controls",
    "classification_for_control",
    "classify_lab_controls",
    "classification_rows_for_controls",
    "physical_placement_for_control",
    "implementation_stage_for_control",
    "physical_placement_rows_for_controls",
    "enabled_uncoupled_controls",
    "apply_lab_perturbations_to_stack",
    "compute_degradation_metrics",
    "metric_delta_table",
    "SensitivityScenario",
    "SensitivitySweepResult",
    "build_stage8c3_sensitivity_scenarios",
    "compute_misalignment_sensitivity_sweep",
    "plot_baseline_vs_perturbed_comparison",
    "plot_misalignment_sensitivity_sweep",
    # integrated cockpit dashboard (Stage 8C.1)
    "build_cockpit_summary",
    "compute_peak_location_diagnostics",
    "compute_overall_status",
    "build_beam_path_strip",
    "choose_display_plane",
    "build_warning_flags",
    "plot_integrated_cockpit_dashboard",
    "make_interpretation_text",
    # component-plane physical lab-realism (Stage 8C.3R)
    "ComponentPlaneState",
    "PropagatedFieldStack",
    "field_power",
    "ComponentPlaneConfig",
    "ComponentPlaneRun",
    "run_component_plane_pipeline",
    "WARNING_ONLY_CONTROLS",
    "ComponentPlaneScenario",
    "ComponentPlaneScenarioResult",
    "ResponseCurveFamily",
    "ResponseCurveResult",
    "DIAGNOSTIC_SWEEP_LABEL",
    "stack_to_fluence",
    "compute_axis_tracking",
    "compute_energy_throughput",
    "classify_translation_vs_deformation",
    "build_component_plane_scenarios",
    "run_component_plane_scenario",
    "build_response_curve_families",
    "run_response_curve",
    "plot_component_plane_reality_preview",
    "RAW_PEAK_LABEL",
    "AnnularAxisEstimate",
    "estimate_annular_axis",
    "track_axis_trajectory",
    # route-aware physical axicon alignment (Stage 8C.3R.3)
    "DIAGNOSTIC_GEOMETRY_NOTE",
    "PHYSICAL_LOCATIONS",
    "REPRESENTED_PHYSICAL_AXICON_LOCATIONS",
    "RouteGraphNode",
    "RoutePerturbationRecord",
    "RouteAwareAxiconConfig",
    "RouteAwareAxiconRun",
    "AxiconAlignmentSweepFamily",
    "AxiconAlignmentSweepResult",
    "physical_axicon_route_graph",
    "build_route_perturbation_records",
    "holographic_slm_route_declarations",
    "physical_axicon_transmission",
    "run_route_aware_axicon_pipeline",
    "build_axicon_alignment_sweep_families",
    "run_axicon_alignment_sweep",
    "plot_route_aware_axicon_pipeline",
    "plot_upstream_vs_post_axicon_tilt_comparison",
    "plot_axicon_alignment_sensitivity_atlas",
    # free-space reference-plane validation (Stage 8C.3R.1)
    "canonical_free_space_reference",
    "zero_control_equivalence",
    "compute_energy_audit",
    "validate_beam_tilt",
    "fov_convergence_check",
    "plot_reference_plane_energy_axis_validation",
    "plot_individual_sensitivity_atlas",
    "plot_fov_convergence_check",
    "plot_annular_axis_tracking_validation",
    "plot_individual_response_curves",
    "plot_free_space_study_summary",
    "DEFAULT_ATLAS_SCENARIOS",
]
