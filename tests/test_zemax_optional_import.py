def test_optional_package_imports_without_importing_zospy():
    import sys
    sys.modules.pop("zospy", None)
    import vbb_study.integrations.zemax as zemax
    assert zemax.ZBF_LONGITUDINAL_COMPONENT_EXCHANGE_SUPPORTED is False
    assert "zospy" not in sys.modules
