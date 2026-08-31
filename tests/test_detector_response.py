import numpy as np

from vbb_study.digital_twin.detector_response import (
    detector_axis_for_display,
    integrate_and_sample_square_pixels,
    interpolate_detector_to_display,
    plane_normalise,
    sample_camera_response,
)


def test_detector_axis_covers_display_and_uses_requested_pitch():
    display = np.linspace(-180e-6, 180e-6, 241)
    pitch = 5.5e-6
    detector = detector_axis_for_display(display, pitch)
    assert detector[0] <= display[0] - 0.5 * pitch
    assert detector[-1] >= display[-1] + 0.5 * pitch
    assert np.allclose(np.diff(detector), pitch)
    assert np.any(np.isclose(detector, 0.0))


def test_square_pixel_integration_preserves_constant_field():
    native = np.linspace(-250e-6, 250e-6, 201)
    detector = np.arange(-150e-6, 150.1e-6, 5.5e-6)
    stack = np.ones((2, native.size, native.size), float)
    sampled = integrate_and_sample_square_pixels(
        stack, native, detector, pixel_pitch_m=5.5e-6, quadrature_n=3,
    )
    assert sampled.shape == (2, detector.size, detector.size)
    assert np.allclose(sampled, 1.0, atol=1e-12)


def test_detector_roundtrip_is_finite_and_shape_preserving():
    native = np.linspace(-300e-6, 300e-6, 257)
    X, Y = np.meshgrid(native, native, indexing="xy")
    image = np.exp(-((X / 45e-6) ** 2 + (Y / 60e-6) ** 2))
    stack = np.stack([image, 0.4 * image])
    display = np.linspace(-180e-6, 180e-6, 241)
    shown, detector = sample_camera_response(
        stack, native, display, pixel_pitch_m=5.5e-6, quadrature_n=3,
    )
    assert shown.shape == (2, display.size, display.size)
    assert detector.ndim == 1
    assert np.isfinite(shown).all()
    normalised = plane_normalise(shown)
    assert np.allclose(normalised.max(axis=(1, 2)), 1.0)


def test_detector_sampling_suppresses_unresolved_checkerboard_contrast():
    native = np.linspace(-220e-6, 220e-6, 441)
    step = native[1] - native[0]
    # Spatial period below the 5.5 um detector pitch.
    xx = np.arange(native.size)
    checker = 0.5 + 0.5 * np.cos(2 * np.pi * xx * step / 3.0e-6)
    image = np.outer(checker, checker)
    display = np.linspace(-160e-6, 160e-6, 321)
    shown, detector = sample_camera_response(
        image[None, ...], native, display,
        pixel_pitch_m=5.5e-6, quadrature_n=5,
    )
    raw_contrast = float(image.std() / image.mean())
    shown_contrast = float(shown[0].std() / shown[0].mean())
    assert shown_contrast < raw_contrast
