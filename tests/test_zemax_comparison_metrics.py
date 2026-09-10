import numpy as np

from vbb_study.integrations.zemax.comparison import centroid, complex_vector_overlap, normalized_intensity_correlation, normalized_intensity_l2_error, phase_rms_after_piston


def test_identical_intensity_metrics_are_exact():
    a = np.arange(1, 26, dtype=float).reshape(5, 5); assert np.isclose(normalized_intensity_correlation(a, a), 1.0); assert np.isclose(normalized_intensity_l2_error(a, a), 0.0)


def test_complex_overlap_ignores_global_complex_scale():
    ex = np.ones((8, 8), complex); ey = 1j * np.ones_like(ex); scale = 3.4 * np.exp(1j * 0.8); assert np.isclose(complex_vector_overlap(ex, ey, scale * ex, scale * ey), 1.0)


def test_phase_rms_removes_piston_only():
    first = np.zeros((8, 8)); second = np.full((8, 8), 1.7); assert phase_rms_after_piston(first, second) < 1e-12


def test_centroid_detects_shift():
    x = np.linspace(-1, 1, 201); xx, yy = np.meshgrid(x, x, indexing="xy"); image = np.exp(-((xx - 0.2) ** 2 + yy**2) / 0.05); cx, cy = centroid(image, x, x); assert abs(cx - 0.2) < 1e-4; assert abs(cy) < 1e-12


def test_shift_and_rotation_are_not_registered_away():
    x = np.linspace(-1.0, 1.0, 101); xx, yy = np.meshgrid(x, x, indexing="xy"); base = np.exp(-(xx**2 / 0.04 + yy**2 / 0.25)); shifted = np.exp(-((xx - 0.2) ** 2 / 0.04 + yy**2 / 0.25)); rotated = np.rot90(base); assert normalized_intensity_correlation(base, shifted) < 0.95; assert normalized_intensity_correlation(base, rotated) < 0.95
