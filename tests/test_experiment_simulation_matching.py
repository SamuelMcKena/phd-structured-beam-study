from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vbb_study.inference.experimental_stack import IntensityStack, load_experimental_stack
from vbb_study.inference.matcher import CandidateScore, MatchResult, match_error_candidates, score_simulation_stack
from vbb_study.inference.miao_handoff import build_miao_handoff


def _stack(values: np.ndarray, *, z_reference: str = "post_axicon_plane") -> IntensityStack:
    values = np.asarray(values, dtype=float)
    nz, ny, nx = values.shape
    return IntensityStack(
        z_m=np.linspace(0.04, 0.08, nz),
        x_m=np.linspace(-1e-3, 1e-3, nx),
        y_m=np.linspace(-1e-3, 1e-3, ny),
        intensity=values,
        metadata={"z_reference": z_reference},
    )


def test_one_global_gain_recovers_radiometric_scale_exactly() -> None:
    base = np.arange(1.0, 25.0).reshape(2, 3, 4)
    simulation = _stack(base)
    measurement = _stack(2.75 * base)
    score, gain, corr, valid, mean_plane, max_plane, _ = score_simulation_stack(
        measurement, simulation
    )
    assert score < 1e-12
    assert gain == pytest.approx(2.75, rel=1e-12, abs=1e-12)
    assert corr == pytest.approx(1.0, abs=1e-12)
    assert valid == pytest.approx(1.0)
    assert mean_plane < 1e-12
    assert max_plane < 1e-12


def test_per_plane_gain_is_not_silently_fitted() -> None:
    base = np.arange(1.0, 25.0).reshape(2, 3, 4)
    simulation = _stack(base)
    measured = base.copy()
    measured[0] *= 2.0
    measured[1] *= 4.0
    score, gain, *_ = score_simulation_stack(_stack(measured), simulation)
    assert 2.0 < gain < 4.0
    assert score > 0.1


def test_manifest_loader_requires_explicit_coordinate_calibration(tmp_path: Path) -> None:
    np.save(tmp_path / "plane.npy", np.ones((4, 5)))
    manifest = {
        "schema": "vbb-experimental-intensity-stack-v1",
        "case_id": "V1",
        "z_reference": "post_axicon_plane",
        "coordinate_calibration": {"pixel_pitch_m": 10e-6},
        "planes": [{"z_m": 0.05, "path": "plane.npy"}],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="array_centre_m"):
        load_experimental_stack(path)


def test_manifest_loader_preserves_declared_lab_coordinates_and_orientation(tmp_path: Path) -> None:
    first = np.arange(12, dtype=float).reshape(3, 4)
    second = first + 100.0
    np.save(tmp_path / "z1.npy", first)
    np.save(tmp_path / "z2.npy", second)
    manifest = {
        "schema": "vbb-experimental-intensity-stack-v1",
        "case_id": "V3",
        "z_reference": "camera_stage_zero",
        "coordinate_calibration": {
            "pixel_pitch_m": [20e-6, 30e-6],
            "array_centre_m": [0.5e-3, -0.25e-3],
            "orientation": {"flip_x": True},
        },
        "planes": [
            {"z_m": 0.02, "path": "z2.npy"},
            {"z_m": 0.01, "path": "z1.npy"},
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    stack = load_experimental_stack(path)
    assert np.allclose(stack.z_m, [0.01, 0.02])
    assert stack.x_m[0] == pytest.approx(0.5e-3 - 1.5 * 20e-6)
    assert stack.x_m[-1] == pytest.approx(0.5e-3 + 1.5 * 20e-6)
    assert stack.y_m[0] == pytest.approx(-0.25e-3 - 30e-6)
    assert np.array_equal(stack.intensity[0], first[:, ::-1])
    assert stack.metadata["z_reference"] == "camera_stage_zero"


def test_non_axicon_z_reference_requires_explicit_model_offset() -> None:
    measurement = _stack(np.ones((1, 3, 4)), z_reference="camera_stage_zero")
    with pytest.raises(ValueError, match="model_z_offset_m"):
        match_error_candidates(
            measurement,
            case_id="B0",
            grid_n=64,
            families=(),
        )


def _score(family: str, value: float, score: float) -> CandidateScore:
    return CandidateScore(
        family=family,
        value=value,
        score=score,
        global_gain=1.0,
        global_correlation=0.99,
        valid_fraction=1.0,
        mean_plane_nrmse=score,
        max_plane_nrmse=score,
        mean_shape_nrmse=score,
        per_plane=(),
        metadata={},
    )


def test_miao_handoff_never_turns_fit_scalar_into_correction_map() -> None:
    phase_result = MatchResult(
        case_id="V3",
        scores=(
            _score("lens1_trefoil", 0.35, 0.05),
            _score("axicon_decentre_x", 0.5e-3, 0.12),
        ),
        objective="test",
        model_z_offset_m=0.0,
        metadata={},
    )
    handoff = build_miao_handoff(phase_result)
    assert handoff.status == "PHASE_RETRIEVAL_CANDIDATE_REQUIRES_PLANE_MAPPING"
    assert handoff.correction_map_ready is False
    assert "Do not use the fitted Zernike scalar" in handoff.next_action

    aperture_result = MatchResult(
        case_id="V3",
        scores=(
            _score("fourf_iris_offset_x", 0.45e-3, 0.04),
            _score("lens1_trefoil", 0.35, 0.2),
        ),
        objective="test",
        model_z_offset_m=0.0,
        metadata={},
    )
    aperture_handoff = build_miao_handoff(aperture_result)
    assert aperture_handoff.status == "NOT_MIAO_PHASE_ONLY_AMPLITUDE_OR_CLIPPING"
    assert aperture_handoff.correction_map_ready is False
