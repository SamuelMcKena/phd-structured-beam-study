from __future__ import annotations

"""Conservative handoff from error screening to Miao-style phase retrieval.

A good intensity match is not itself a correction phase.  This module therefore
never converts a fitted scalar error parameter directly into an SLM conjugate
map.  It only decides what kind of next action is physically justified.
"""

from dataclasses import dataclass
from typing import Any, Mapping

from .candidate_models import family_spec
from .matcher import MatchResult


@dataclass(frozen=True)
class MiaoHandoff:
    family: str
    value: float
    match_score: float
    status: str
    correction_map_ready: bool
    next_action: str
    score_gap_to_second: float | None
    score_ratio_to_second: float | None
    metadata: Mapping[str, Any]


def build_miao_handoff(result: MatchResult) -> MiaoHandoff:
    """Classify the best single-error hypothesis for the correction stage.

    The declared L1 Zernike candidates are coherent phase hypotheses, but their
    fitted scalar amplitudes are not substituted for a retrieved phase map.
    Their physical plane must be mapped to the chosen correction SLM before a
    conjugate correction can be applied.
    """
    best = result.best
    spec = family_spec(best.family)
    second = result.scores[1] if len(result.scores) > 1 else None
    gap = None if second is None else float(second.score - best.score)
    ratio = None if second is None or second.score <= 0.0 else float(best.score / second.score)
    policy = spec.miao_handoff_policy

    if policy == "phase_retrieval_candidate_requires_plane_mapping":
        status = "PHASE_RETRIEVAL_CANDIDATE_REQUIRES_PLANE_MAPPING"
        next_action = (
            "Use the measured multi-plane intensity stack in the Miao retrieval to estimate a complex/phase map; "
            "then propagate/map that retrieved phase to the intended correction SLM plane before applying a conjugate. "
            "Do not use the fitted Zernike scalar itself as the correction map."
        )
    elif policy == "amplitude_or_clipping_not_phase_only":
        status = "NOT_MIAO_PHASE_ONLY_AMPLITUDE_OR_CLIPPING"
        next_action = (
            "Correct the physical aperture/alignment mechanism or include it explicitly in the forward model; "
            "a phase-only conjugate cannot restore field removed by clipping."
        )
    elif policy == "calibrate_SLM_response_before_wavefront_retrieval":
        status = "CALIBRATE_SLM_RESPONSE_BEFORE_MIAO"
        next_action = (
            "Measure/fix the SLM voltage-to-phase response and static panel calibration before interpreting residual "
            "intensity as an unknown wavefront phase."
        )
    elif policy == "alignment_parameter_not_direct_conjugate_phase":
        status = "ALIGNMENT_HYPOTHESIS_NOT_DIRECT_CONJUGATE_PHASE"
        next_action = (
            "Treat the inferred decentre as a mechanical/coordinate alignment hypothesis first; rerun the forward "
            "model after alignment before attempting residual phase retrieval."
        )
    elif policy == "surface_geometry_requires_mechanism_specific_model":
        status = "AXICON_GEOMETRY_HYPOTHESIS_REQUIRES_SURFACE_MODEL"
        next_action = (
            "Retain the rounded-tip/surface geometry in the forward model and validate its dimensions independently; "
            "do not replace coherent geometry-induced interference with an arbitrary phase-only correction."
        )
    else:
        status = "MECHANISM_SPECIFIC_REMEDIATION_BEFORE_MIAO"
        next_action = (
            "Resolve or explicitly model the inferred non-wavefront mechanism before using Miao phase retrieval on the residual."
        )

    return MiaoHandoff(
        family=best.family,
        value=float(best.value),
        match_score=float(best.score),
        status=status,
        correction_map_ready=False,
        next_action=next_action,
        score_gap_to_second=gap,
        score_ratio_to_second=ratio,
        metadata={
            "family_title": spec.title,
            "operator_class": spec.operator_class,
            "physical_plane": spec.physical_plane,
            "candidate_provenance": spec.provenance,
            "policy": policy,
            "important_limitation": (
                "single-error intensity matching is a hypothesis screen, not proof of parameter uniqueness or a phase retrieval"
            ),
            "correction_map_policy": "always_false_until_complex_phase_retrieval_and_plane_mapping_are_completed",
        },
    )


__all__ = ["MiaoHandoff", "build_miao_handoff"]
