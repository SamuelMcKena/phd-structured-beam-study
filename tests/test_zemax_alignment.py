import numpy as np

from vbb_study.integrations.zemax.alignment import align_field_planes
from vbb_study.integrations.zemax.models import FieldPlane


def _gaussian(x, y, shift=0.0):
    xx, yy = np.meshgrid(x, y, indexing="xy"); return np.exp(-((xx - shift) ** 2 + yy**2) / (2 * (0.25e-3) ** 2))


def test_alignment_resamples_by_physical_coordinates():
    x1 = np.linspace(-1e-3, 1e-3, 101); x2 = np.linspace(-1.2e-3, 1.2e-3, 121); p = FieldPlane(x1, x1, 1030e-9, intensity=_gaussian(x1, x1), plane_name="p"); z = FieldPlane(x2, x2, 1030e-9, intensity=_gaussian(x2, x2), plane_name="z"); pa, za, meta = align_field_planes(p, z, comparison_grid="python"); assert pa.intensity.shape == za.intensity.shape; assert meta["comparison_grid"] == "python"; assert np.max(np.abs(pa.intensity - za.intensity)) < 1e-10
