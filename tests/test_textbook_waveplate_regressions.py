"""Independent physical invariants found during the textbook audit."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

from vbb_study.vector_arm_chain import hwp45_matrix, qwp45_matrix


@pytest.mark.parametrize('error', [-0.3, 0.0, 0.23])
@pytest.mark.parametrize('matrix', [hwp45_matrix, qwp45_matrix])
def test_lossless_waveplate_is_unitary(matrix, error):
    j = matrix(error)
    np.testing.assert_allclose(j.conj().T @ j, np.eye(2), atol=1e-14)


def test_hwp_retardance_error_leaves_unconverted_component():
    error = 0.2
    out = hwp45_matrix(error) @ np.array([1., 0.])
    assert abs(out[0])**2 == pytest.approx(np.sin(error/2)**2)
    assert abs(out[1])**2 == pytest.approx(np.cos(error/2)**2)

