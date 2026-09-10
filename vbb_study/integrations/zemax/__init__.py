"""Optional Ansys Zemax OpticStudio validation backend.

Importing this package never imports ZOSPy eagerly. The canonical Python
solvers remain authoritative; Zemax POP is an independent validation backend.
"""
from .availability import classify_exit_code, probe_zemax_environment
from .comparison import compare_field_planes
from .field_exchange import ARBITRARY_PYTHON_TO_ZBF_EXPORT_AVAILABLE, ZBF_LONGITUDINAL_COMPONENT_EXCHANGE_SUPPORTED, load_field_npz, save_field_npz
from .models import CrossValidationResult, FieldPlane, PopRequest, PopResult, ZemaxAvailability, ZemaxModelMetadata

__all__ = ["ARBITRARY_PYTHON_TO_ZBF_EXPORT_AVAILABLE", "ZBF_LONGITUDINAL_COMPONENT_EXCHANGE_SUPPORTED", "CrossValidationResult", "FieldPlane", "PopRequest", "PopResult", "ZemaxAvailability", "ZemaxModelMetadata", "classify_exit_code", "compare_field_planes", "load_field_npz", "probe_zemax_environment", "save_field_npz"]
