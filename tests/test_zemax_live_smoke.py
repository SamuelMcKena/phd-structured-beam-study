import importlib.util
import platform

import pytest

from vbb_study.integrations.zemax.availability import probe_zemax_environment


@pytest.mark.zemax
def test_live_zemax_environment_when_available():
    if platform.system() != "Windows" or importlib.util.find_spec("zospy") is None: pytest.skip("Zemax live test requires Windows + ZOSPy")
    status = probe_zemax_environment(try_connections=True)
    if not status.ready: pytest.skip("OpticStudio installation/licence is not available for this live test")
    assert status.standalone_connection_available
