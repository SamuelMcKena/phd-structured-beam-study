from vbb_study.integrations.zemax.availability import classify_exit_code
from vbb_study.integrations.zemax.models import ZemaxAvailability


def _status(**updates):
    base = dict(platform="Windows", python_version="3.13", python_executable="python", zospy_installed=False); base.update(updates); return ZemaxAvailability(**base)


def test_preflight_exit_codes_are_stable():
    assert classify_exit_code(_status()) == 2
    assert classify_exit_code(_status(zospy_installed=True)) == 3
    assert classify_exit_code(_status(zospy_installed=True, opticstudio_detected=True, zosapi_loadable=True)) == 4
    assert classify_exit_code(_status(zospy_installed=True, opticstudio_detected=True, zosapi_loadable=True, license_available=True, standalone_connection_available=True)) == 0
