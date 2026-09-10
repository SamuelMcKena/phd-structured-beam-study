import pytest

from vbb_study.integrations.zemax.python_reference import build_phase2c_vector_focus_reference


def test_h1_is_not_silently_added_to_first_zemax_vortex_scope():
    with pytest.raises(ValueError, match="vortex scope"): build_phase2c_vector_focus_reference("H1")
