import numpy as np
import pytest

from vbb_study.integrations.zemax.errors import ZemaxFieldExchangeError, ZemaxModelError
from vbb_study.integrations.zemax.field_exchange import StagedZbfInput, assert_zbf_component_supported
from vbb_study.integrations.zemax.models import FieldPlane, PopRequest
from vbb_study.integrations.zemax.prescription import length_unit_to_m, validate_surface_map


def test_field_plane_transverse_intensity():
    axis = np.linspace(-1e-3, 1e-3, 9); ex = np.ones((9, 9), dtype=complex); plane = FieldPlane(axis, axis, 1030e-9, Ex=ex, Ey=np.zeros_like(ex)); assert np.allclose(plane.transverse_intensity, 1.0)


def test_pop_request_rejects_file_without_zbf_path():
    with pytest.raises(ValueError): PopRequest(model_path=None, beam_type="File").validate()


def test_surface_map_bounds():
    with pytest.raises(ZemaxModelError): validate_surface_map({"sample_surface": 8}, surface_count=8)


def test_zbf_cannot_validate_ez():
    with pytest.raises(ZemaxFieldExchangeError, match="cannot validate Ez"): assert_zbf_component_supported("Ez")


def test_pop_request_serializes_si_beam_parameters():
    request = PopRequest(model_path=None, beam_parameters_m={"Waist X": 0.5e-3}); assert request.to_dict()["beam_parameters_m"]["Waist X"] == 0.5e-3


def test_opticstudio_enum_style_length_units_are_explicitly_supported():
    assert length_unit_to_m("LensUnits.Millimeters") == 1e-3; assert length_unit_to_m("Meters") == 1.0


def test_staged_zbf_input_uses_pop_folder_and_cleans_up(tmp_path):
    source = tmp_path / "input.zbf"; source.write_bytes(b"phase3-test-zbf"); pop_dir = tmp_path / "POP" / "BEAMFILES"
    with StagedZbfInput(source, pop_dir) as staged:
        assert staged.staged_path.parent == pop_dir.resolve(); assert staged.staged_path.exists(); assert staged.zemax_filename.endswith("input.zbf")
    assert not staged.staged_path.exists()
