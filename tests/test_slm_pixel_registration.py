"""Tests for the sub-pixel SLM pixel-lattice registration study.

The gates here exist because the study's conclusions depend entirely on two
claims: that the new fine/coarse plumbing is the *accepted* route rather than a
new model, and that the registration knob moves the pixel lattice rather than
quietly decentring the hologram.  Both are asserted numerically.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from vbb_study.digital_twin.phase2a_canonical import _panel_from_manifest
from vbb_study.digital_twin.phase2a_contracts import canonical_hardware_manifest
from vbb_study.digital_twin.slm_pixel_registration import (
    RegistrationSampling,
    _apply_panel,
    band_limited_decimate_fixed_window,
    build_registration_route,
    lean_asm_propagate,
    lean_xy_grid,
    registration_panels,
)
from vbb_study.digital_twin.slm_registration_metrics import (
    azimuthal_spectrum,
    complex_fidelity,
    plane_metrics,
    relative_l2_phase_aligned,
)
from vbb_study.digital_twin.vortex_beam_slm_errors import SLMError, actual_slm_phase
from vbb_study.digital_twin.vortex_system_route import (
    SystemErrorConfig,
    build_system_route,
    fourier_resample_fixed_window,
)
from vbb_study.equations.fields import make_xy_grid
from vbb_study.equations.propagation import angular_spectrum_propagate_bl
from vbb_study.slm_model import apply_slm, pixelate, slm_active_aperture

WINDOW_M = 10.0e-3
WAVELENGTH_M = 1.029e-6
PITCH_M = 8.0e-6


def _rel(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b))
                 / max(float(np.linalg.norm(np.asarray(b))), 1e-300))


# --------------------------------------------------------------------------
# lattice plumbing
# --------------------------------------------------------------------------

def test_lattice_offset_default_preserves_legacy_pixelation():
    grid = make_xy_grid(256, WINDOW_M / 256)
    panel = _panel_from_manifest(canonical_hardware_manifest())
    phase = 3.0 * np.arctan2(grid["Y"], grid["X"])
    assert np.array_equal(
        pixelate(phase, grid, panel),
        pixelate(phase, grid, panel, lattice_offset_m=(0.0, 0.0)),
    )


def test_full_pitch_lattice_shift_is_a_pure_translation_of_the_result():
    """Shifting pattern and lattice together by one pitch just translates the output.

    The grid is chosen so that the pitch is an exact integer number of samples,
    which turns the expected translation into an exact array roll.
    """

    n = 1024
    window = 2.048e-3  # dx = 2 um exactly -> 4 samples per 8 um pixel
    grid = make_xy_grid(n, window / n)
    step = int(round(PITCH_M / (window / n)))
    assert step == 4
    panel = _panel_from_manifest(canonical_hardware_manifest())
    base = pixelate(3.0 * np.arctan2(grid["Y"], grid["X"]), grid, panel)
    shifted = pixelate(
        3.0 * np.arctan2(grid["Y"], grid["X"] - PITCH_M),
        grid,
        panel,
        lattice_offset_m=(PITCH_M, 0.0),
    )
    inner = np.abs(grid["X"]) < 0.5e-3
    assert np.allclose(np.roll(base, step, axis=1)[inner], shifted[inner], atol=1e-12)


def test_unresolved_grid_is_refused_rather_than_returning_a_silent_null():
    """dx > pitch makes pixelate an identity, so a sweep there would read as zero."""

    coarse = RegistrationSampling(fine_grid_n=512, relay_grid_n=512, window_m=WINDOW_M)
    with pytest.raises(ValueError, match="does not resolve the SLM pixel lattice"):
        build_registration_route("V3", sampling=coarse, arm="bench",
                                 fill_factor_model="throughput_only")
    # the opt-out exists only to reproduce the unresolved accepted route
    route = build_registration_route("V3", sampling=coarse, arm="bench",
                                     fill_factor_model="throughput_only",
                                     allow_unresolved_lattice=True)
    assert route["metadata"]["pixel_lattice_resolved"] is False


def test_panel_translation_moves_the_active_aperture():
    grid = make_xy_grid(256, 20.0e-3 / 256)
    panel = _panel_from_manifest(canonical_hardware_manifest())
    a0 = slm_active_aperture(grid, panel)
    a1 = slm_active_aperture(grid, panel, lattice_offset_m=(1.0e-3, 0.0))
    assert not np.array_equal(a0, a1)
    assert a0.sum() == pytest.approx(a1.sum(), rel=0.05)


def test_registration_panels_hold_beam_to_hologram_alignment_fixed():
    """Panel moves by -d, hologram is addressed +d back, so the core does not move."""

    d = 3.0e-6
    for dof, expect1, expect2 in (
        ("slm1", -d, 0.0),
        ("slm2", 0.0, -d),
        ("common", -d, -d),
        ("differential", -d, +d),
    ):
        s1, s2 = registration_panels(d, dof=dof, axis="x")
        assert s1.panel_translation_m[0] == pytest.approx(expect1)
        assert s2.panel_translation_m[0] == pytest.approx(expect2)
        for s in (s1, s2):
            # hologram centre = translation + electronic offset = 0
            assert s.panel_translation_m[0] + s.pattern_offset_m[0] == pytest.approx(0.0)
            assert s.panel_translation_m[1] + s.pattern_offset_m[1] == pytest.approx(0.0)


def test_registration_dof_is_rejected_when_unknown():
    with pytest.raises(ValueError):
        registration_panels(1e-6, dof="slm3")


# --------------------------------------------------------------------------
# numerical primitives
# --------------------------------------------------------------------------

def test_lean_grid_matches_reference_coordinates():
    ref = make_xy_grid(128, WINDOW_M / 128)
    lean = lean_xy_grid(128, WINDOW_M / 128)
    for key in ("X", "Y", "FX", "FY", "x"):
        assert np.array_equal(ref[key], lean[key])


def test_lean_asm_matches_reference_propagator():
    n = 256
    ref = make_xy_grid(n, WINDOW_M / n)
    lean = lean_xy_grid(n, WINDOW_M / n)
    rng = np.random.default_rng(5)
    u = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
    u *= np.exp(-(ref["X"] ** 2 + ref["Y"] ** 2) / (2.0e-3) ** 2)
    for z in (0.3, -0.05):
        got = lean_asm_propagate(u, lean, WAVELENGTH_M, z)
        want = angular_spectrum_propagate_bl(u, dict(ref), WAVELENGTH_M, z, 1.0, True, True)
        assert _rel(got, want) < 1e-13


def test_decimation_reproduces_direct_analytic_sampling():
    """A band-limited field decimated must equal that field sampled directly."""

    n_fine, n_coarse = 1024, 256
    modes = ((3, -2), (-5, 1), (7, 4), (0, 0))
    amps = (1.0, 0.5 - 0.3j, -0.2 + 0.8j, 1.5)

    def build(n):
        x = (np.arange(n) - n / 2 + 0.5) * (WINDOW_M / n)
        X, Y = np.meshgrid(x, x, indexing="xy")
        out = np.zeros(X.shape, dtype=complex)
        for (mx, my), a in zip(modes, amps):
            out += a * np.exp(2j * np.pi * (mx * X + my * Y) / WINDOW_M)
        return out

    got, meta = band_limited_decimate_fixed_window(build(n_fine), n_coarse)
    assert _rel(got, build(n_coarse)) < 1e-10
    assert meta["discarded_power_fraction"] < 1e-12


def test_decimation_inverts_the_zero_padding_resampler():
    n_coarse, n_fine = 128, 512
    rng = np.random.default_rng(2)
    x = (np.arange(n_coarse) - n_coarse / 2 + 0.5) * (WINDOW_M / n_coarse)
    X, Y = np.meshgrid(x, x, indexing="xy")
    band = np.zeros((n_coarse, n_coarse), dtype=complex)
    for _ in range(8):
        mx, my = rng.integers(-n_coarse // 8, n_coarse // 8, size=2)
        band += (rng.standard_normal() + 1j * rng.standard_normal()) * np.exp(
            2j * np.pi * (mx * X + my * Y) / WINDOW_M)
    back, _ = band_limited_decimate_fixed_window(
        fourier_resample_fixed_window(band, n_fine), n_coarse)
    assert _rel(back, band) < 1e-10


def test_decimation_rejects_upsampling():
    with pytest.raises(ValueError):
        band_limited_decimate_fixed_window(np.zeros((32, 32), dtype=complex), 64)


def test_apply_panel_matches_apply_slm():
    n = 512
    # dx = 0.5 um, an exact 16 samples per 8 um pitch, fine enough that the
    # widened dead space below is genuinely resolved
    grid = lean_xy_grid(n, 0.256e-3 / n)
    panel = _panel_from_manifest(canonical_hardware_manifest())
    rng = np.random.default_rng(13)
    u = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    cmd = 3.0 * np.arctan2(grid["Y"], grid["X"])
    err = SLMError(panel_translation_m=(-3e-6, 1e-6), pattern_offset_m=(3e-6, -1e-6))
    lat = err.lattice_offset_m
    prepared = pixelate(cmd, grid, panel, lattice_offset_m=lat)
    actual, _ = actual_slm_phase(prepared, grid, error=err, pixel_pitch_m=PITCH_M)
    # A 0.93 fill factor implies 0.285 um dead-space lines, which no affordable
    # grid resolves, so the resolved-aperture branch is exercised on a panel
    # whose dead space this grid genuinely resolves.
    wide = dataclasses.replace(panel, fill_factor=0.25)
    for model, cfg in (("throughput_only", panel),
                       ("resolved_pixel_aperture", wide)):
        got, _ = _apply_panel(u, grid, lambda xp, yp: 3.0 * np.arctan2(yp, xp),
                              panel_cfg=cfg, error=err,
                              pixel_pitch_m=PITCH_M, pixelate_phase=True,
                              fill_factor_model=model)
        want = apply_slm(u, actual, grid, cfg, phase_is_prepared=True,
                         quantise_phase=False, apply_fill_factor=True,
                         apply_carrier=False, fill_factor_model=model,
                         lattice_offset_m=lat)
        assert _rel(got, want.total) < 1e-13


# --------------------------------------------------------------------------
# route equivalence and physical behaviour
# --------------------------------------------------------------------------

@pytest.mark.parametrize("case", ["B0", "V3"])
def test_registration_route_reproduces_accepted_system_route(case):
    """With no decimation the lean route must BE the accepted Phase 2E route."""

    n = 512
    ref = build_system_route(case, grid_n=n, config=SystemErrorConfig(), window_m=WINDOW_M)
    got = build_registration_route(
        case,
        sampling=RegistrationSampling(fine_grid_n=n, relay_grid_n=n, window_m=WINDOW_M),
        arm="bench", pixelate_phase=True, fill_factor_model="throughput_only",
        allow_unresolved_lattice=True)
    assert _rel(got["field_on_axicon_plane"], ref["field_on_axicon_plane"]) < 1e-12
    assert _rel(got["post_axicon"], ref["post_axicon"]) < 1e-12


def _resolved_isolation_route(delta_m: float, *, dof: str = "slm1"):
    """Cheap lattice-resolving route: isolation arm, small window, small beam."""

    sampling = RegistrationSampling(fine_grid_n=1024, relay_grid_n=1024,
                                    window_m=2.0e-3)
    s1, s2 = registration_panels(delta_m, dof=dof, axis="x")
    return build_registration_route(
        "V3", sampling=sampling, slm1=s1, slm2=s2, arm="isolation",
        beam_radius_m=100e-6, fill_factor_model="throughput_only")


def test_whole_pixel_registration_shift_is_a_no_op_end_to_end():
    """The registration effect must be exactly periodic in the pixel pitch."""

    base = _resolved_isolation_route(0.0)
    full = _resolved_isolation_route(PITCH_M)
    assert base["metadata"]["pixel_lattice_resolved"] is True
    assert _rel(full["field_on_axicon_plane"], base["field_on_axicon_plane"]) < 1e-12


def test_sub_pixel_registration_changes_the_field():
    base = _resolved_isolation_route(0.0)
    half = _resolved_isolation_route(0.5 * PITCH_M)
    assert _rel(half["field_on_axicon_plane"], base["field_on_axicon_plane"]) > 1e-6


def test_axicon_is_pure_phase_so_intensity_is_unchanged_across_it():
    """Guards the reporting rule: 'after the axicon' must mean after propagation."""

    sampling = RegistrationSampling(fine_grid_n=512, relay_grid_n=512, window_m=WINDOW_M)
    route = build_registration_route("V1", sampling=sampling, arm="bench",
                                     fill_factor_model="throughput_only",
                                     allow_unresolved_lattice=True)
    assert np.allclose(np.abs(route["post_axicon"]),
                       np.abs(route["field_on_axicon_plane"]), rtol=1e-12, atol=0.0)


def test_isolation_arm_has_no_carrier_or_iris():
    sampling = RegistrationSampling(fine_grid_n=256, relay_grid_n=256,
                                    window_m=2.0e-3)
    route = build_registration_route("V3", sampling=sampling, arm="isolation",
                                     beam_radius_m=100e-6,
                                     fill_factor_model="throughput_only",
                                     allow_unresolved_lattice=True)
    meta = route["metadata"]
    assert meta["fourf"] == "absent_by_construction"
    assert "mechanism study only" in meta["isolation_scope"]


def test_unknown_arm_is_rejected():
    with pytest.raises(ValueError):
        build_registration_route("B0", arm="bench_with_extras")


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def test_fidelity_is_invariant_to_a_global_phase():
    rng = np.random.default_rng(4)
    a = rng.standard_normal((32, 32)) + 1j * rng.standard_normal((32, 32))
    assert complex_fidelity(a, a) == pytest.approx(1.0)
    assert complex_fidelity(a * np.exp(1.3j), a) == pytest.approx(1.0)
    assert relative_l2_phase_aligned(a * np.exp(1.3j), a) == pytest.approx(0.0, abs=1e-12)


def test_azimuthal_spectrum_recovers_a_pure_vortex_charge():
    n = 256
    grid = lean_xy_grid(n, 2.0e-3 / n)
    for ell in (1, 3, 7):
        field = np.exp(1j * ell * np.arctan2(grid["Y"], grid["X"]))
        spec = azimuthal_spectrum(field, grid, 0.4e-3, n_theta=512)
        assert spec[ell] > 0.999


def test_plane_metrics_report_a_dark_core_for_a_vortex():
    n = 256
    grid = lean_xy_grid(n, 2.0e-3 / n)
    r = np.hypot(grid["X"], grid["Y"])
    field = (r / 0.3e-3) * np.exp(-(r / 0.3e-3) ** 2) * np.exp(
        3j * np.arctan2(grid["Y"], grid["X"]))
    m = plane_metrics(field, grid, charge=3)
    assert m["core_darkness"] < 0.2
    assert m["azimuthal_purity"] > 0.99
    assert m["ring_radius_m"] > 0.0


def test_centre_sample_convention_is_exactly_piecewise_constant():
    """Commanding at pixel centres must be constant within each pixel."""

    from vbb_study.digital_twin.slm_pixel_registration import (
        PIXEL_VALUE_MODELS, commanded_pixel_phase, snap_to_pixel_centres)

    n = 512
    window = 1.024e-3  # dx = 2 um exactly -> 4 samples per pixel
    grid = lean_xy_grid(n, window / n)
    panel = _panel_from_manifest(canonical_hardware_manifest())
    err = SLMError()
    phase = commanded_pixel_phase(
        grid, lambda xp, yp: 3.0 * np.arctan2(yp, xp),
        error=err, panel_cfg=panel, pixel_value_model="centre_sample")
    # every 4x4 block of samples lies in one pixel and must share a value
    block = phase[:4 * (n // 4), :4 * (n // 4)].reshape(n // 4, 4, n // 4, 4)
    spread = np.max(np.abs(block - block[:, :1, :, :1]))
    assert spread == 0.0
    assert set(PIXEL_VALUE_MODELS) == {"area_average", "centre_sample"}


def test_centre_sample_snapping_is_lattice_referenced():
    from vbb_study.digital_twin.slm_pixel_registration import snap_to_pixel_centres

    x = np.array([0.0, 3.9e-6, 8.1e-6])
    snapped = snap_to_pixel_centres(x, PITCH_M, 0.0)
    assert np.allclose(snapped, [4e-6, 4e-6, 12e-6])
    # with the lattice origin offset by 2 um the boundaries sit at 2 + 8k um,
    # so x = 0 falls in [-6, 2) um and takes centre -2 um
    shifted = snap_to_pixel_centres(x, PITCH_M, 2e-6)
    assert np.allclose(shifted, [-2e-6, 6e-6, 6e-6])


def test_pixel_value_models_disagree_but_both_are_periodic_in_the_pitch():
    """Both conventions are lattice physics; they differ on how a pixel is valued."""

    sampling = RegistrationSampling(fine_grid_n=1024, relay_grid_n=1024,
                                    window_m=2.0e-3)

    def run(delta, model):
        s1, s2 = registration_panels(delta, dof="slm1", axis="x")
        return build_registration_route(
            "V3", sampling=sampling, slm1=s1, slm2=s2, arm="isolation",
            beam_radius_m=100e-6, fill_factor_model="throughput_only",
            pixel_value_model=model)["field_on_axicon_plane"]

    for model in ("area_average", "centre_sample"):
        assert _rel(run(PITCH_M, model), run(0.0, model)) < 1e-12
    area = run(0.5 * PITCH_M, "area_average")
    centre = run(0.5 * PITCH_M, "centre_sample")
    assert _rel(area, centre) > 1e-9


def test_unresolvable_dead_space_is_refused():
    """A 0.93 fill factor implies 0.285 um gaps that no affordable grid resolves."""

    sampling = RegistrationSampling(fine_grid_n=2500, relay_grid_n=1250,
                                    window_m=WINDOW_M)
    with pytest.raises(ValueError, match="dead space"):
        build_registration_route("V3", sampling=sampling, arm="bench",
                                 fill_factor_model="resolved_pixel_aperture")


def test_route_reports_pitch_commensurability():
    """Non-integer samples per pitch gives pixels unequal sample counts."""

    good = build_registration_route(
        "V3", sampling=RegistrationSampling(fine_grid_n=2500, relay_grid_n=1250,
                                            window_m=WINDOW_M),
        arm="bench", fill_factor_model="throughput_only")
    assert good["metadata"]["pitch_commensurate"] is True
    assert good["metadata"]["samples_per_pixel"] == pytest.approx(2.0)

    odd = build_registration_route(
        "V3", sampling=RegistrationSampling(fine_grid_n=2560, relay_grid_n=1280,
                                            window_m=WINDOW_M),
        arm="bench", fill_factor_model="throughput_only")
    assert odd["metadata"]["pitch_commensurate"] is False
