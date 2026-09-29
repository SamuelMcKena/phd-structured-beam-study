"""Sub-pixel SLM pixel-lattice registration sensitivity for the vortex source route.

Physical question
-----------------
The commanded hologram is written onto a discrete LCoS pixel lattice that is
fixed in the laboratory frame.  Where the incident beam lands *within* that
lattice therefore sets which part of the continuous phase command each pixel
area-averages, and -- with a resolved pixel aperture -- which part of the beam
falls on inter-pixel dead space.  This module varies that registration at
sub-pixel resolution and measures the consequence both at the axicon input
plane and in the Bessel region behind it.

Two distinct arms are provided.

``bench``
    The full documented route ``Gaussian -> SLM1 -> SLM2/carrier -> explicit 4F
    with the fixed +1 iris -> axicon``.  This is the architecture that actually
    exists, and it is the only arm from which bench claims may be made.

``isolation``
    ``Gaussian -> SLM1 -> axicon`` with no carrier and no iris.  This arm exists
    only to expose the pixel-lattice mechanism in the strongly undersampled
    regime (beam radius of a few pixels), which the bench arm cannot reach: once
    the beam radius falls below roughly six pixels its far-field divergence
    exceeds the carrier order separation and the +1-order architecture ceases to
    exist.  Isolation-arm results are a mechanism study, not a bench prediction.

Sampling
--------
Resolving sub-pixel registration requires several computational samples per
8 um SLM pixel, which the accepted Phase 2E source grids (N=1536 over a 10 mm
window, dx=6.5 um) do not provide.  The bench arm therefore runs the SLM and
Fourier-plane stage on a fine grid and hands the *post-iris* field down to the
established relay/axicon sampling.  That handoff is legitimate because the
physical iris band-limits the field: it passes only spatial frequencies within
``iris_radius / (lambda f)`` of the carrier, far inside the fine-grid Nyquist
limit.  The discarded out-of-band power is measured and reported for every
route rather than assumed negligible.

Provenance
----------
Registration offsets here are geometric sensitivity values, not measurements of
the HOLOEYE panels.  Absolute claims additionally require the measured
grey-to-phase LUT, the static panel phase map and a fitted fringing kernel,
which remain data-blocked in the Phase 2E registry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from vbb_study.digital_twin.phase2a_canonical import _panel_from_manifest
from vbb_study.digital_twin.phase2a_contracts import (
    canonical_hardware_manifest,
    hardware_value,
)
from vbb_study.digital_twin.vortex_beam_slm_errors import (
    GaussianBeamError,
    SLMError,
    actual_slm_phase,
    gaussian_input_field,
)
from vbb_study.digital_twin.vortex_explicit_4f import (
    LensError,
    _apply_lens_plane,
    nominal_order_position_m,
    physical_iris,
)
from vbb_study.digital_twin.vortex_system_route import (
    AxiconError,
    _ell,
    physical_axicon_on_own_plane,
)
from vbb_study.slm_model import pixelate, resolved_pixel_aperture, slm_active_aperture

EPS = np.finfo(float).tiny
TWOPI = 2.0 * np.pi

BENCH_WINDOW_M = 10.0e-3
ISOLATION_WINDOW_M = 2.0e-3


# --------------------------------------------------------------------------
# lean grid and lean propagation
# --------------------------------------------------------------------------

def lean_xy_grid(n: int, dx_m: float) -> dict[str, Any]:
    """Square centred grid holding only the arrays this route actually uses.

    ``make_xy_grid`` additionally materialises ``R`` and ``PHI``, which costs two
    further full float64 arrays.  At the fine sampling required here that is
    hundreds of megabytes of dead weight, so this builds the same coordinate
    convention without them.  The cell-centred convention is identical.
    """

    n = int(n)
    x = (np.arange(n) - n / 2 + 0.5) * float(dx_m)
    fx = np.fft.fftshift(np.fft.fftfreq(n, d=float(dx_m)))
    X, Y = np.meshgrid(x, x, indexing="xy")
    FX, FY = np.meshgrid(fx, fx, indexing="xy")
    return {"N": n, "dx": float(dx_m), "x": x, "X": X, "Y": Y, "FX": FX, "FY": FY}


def lean_asm_propagate(
    field: np.ndarray,
    grid: Mapping[str, Any],
    wavelength_m: float,
    distance_m: float,
    *,
    block_rows: int = 256,
) -> np.ndarray:
    """Band-limited angular-spectrum propagation with a blocked transfer function.

    Arithmetically identical to :func:`angular_spectrum_propagate_bl` with
    ``bandlimit=True`` and ``include_evanescent=True``; the transfer function is
    formed one row block at a time and multiplied straight into the spectrum so
    that the full-grid temporaries of the reference implementation are never all
    resident at once.  Equality with the reference is asserted by the test suite.
    """

    if abs(float(distance_m)) < 1e-15:
        return np.asarray(field, dtype=np.complex128)

    spectrum = np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(np.asarray(field, dtype=np.complex128)))
    )
    n = int(grid["N"])
    k = TWOPI / float(wavelength_m)
    lam = float(wavelength_m)
    du = 1.0 / (n * float(grid["dx"]))
    zz = abs(float(distance_m))
    u_lim = 1.0 / (lam * math.sqrt((2.0 * du * zz) ** 2 + 1.0))
    FX = grid["FX"]
    FY = grid["FY"]

    for start in range(0, n, int(block_rows)):
        stop = min(start + int(block_rows), n)
        kx = TWOPI * FX[start:stop]
        ky = TWOPI * FY[start:stop]
        arg = k * k - kx * kx - ky * ky
        prop = arg >= 0.0
        h = np.zeros(arg.shape, dtype=complex)
        h[prop] = np.exp(1j * np.sqrt(arg[prop]) * float(distance_m))
        evan = ~prop
        h[evan] = np.exp(-np.sqrt(np.maximum(-arg[evan], 0.0)) * zz)
        h *= (np.abs(FX[start:stop]) <= u_lim) & (np.abs(FY[start:stop]) <= u_lim)
        spectrum[start:stop] *= h
        del kx, ky, arg, prop, h, evan

    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(spectrum)))


def lean_asm_propagator(
    field: np.ndarray,
    grid: Mapping[str, Any],
    wavelength_m: float,
    *,
    block_rows: int = 256,
):
    """Return a closure propagating one field to any z, reusing its spectrum.

    An axial scan sends the *same* post-axicon field to many planes, so the
    forward transform is identical every time.  Computing it once and reusing it
    removes one full transform per plane, which is the dominant cost of a scan.
    Mirrors the repository's own ``make_bl_asm_propagator`` pattern and returns
    values identical to :func:`lean_asm_propagate`.
    """

    source = np.asarray(field, dtype=np.complex128)
    spectrum0 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(source)))
    n = int(grid["N"])
    k = TWOPI / float(wavelength_m)
    lam = float(wavelength_m)
    du = 1.0 / (n * float(grid["dx"]))
    FX = grid["FX"]
    FY = grid["FY"]

    def propagate(distance_m: float) -> np.ndarray:
        if abs(float(distance_m)) < 1e-15:
            return source.copy()
        zz = abs(float(distance_m))
        u_lim = 1.0 / (lam * math.sqrt((2.0 * du * zz) ** 2 + 1.0))
        spectrum = spectrum0.copy()
        for start in range(0, n, int(block_rows)):
            stop = min(start + int(block_rows), n)
            kx = TWOPI * FX[start:stop]
            ky = TWOPI * FY[start:stop]
            arg = k * k - kx * kx - ky * ky
            prop = arg >= 0.0
            h = np.zeros(arg.shape, dtype=complex)
            h[prop] = np.exp(1j * np.sqrt(arg[prop]) * float(distance_m))
            evan = ~prop
            h[evan] = np.exp(-np.sqrt(np.maximum(-arg[evan], 0.0)) * zz)
            h *= (np.abs(FX[start:stop]) <= u_lim) & (np.abs(FY[start:stop]) <= u_lim)
            spectrum[start:stop] *= h
            del kx, ky, arg, prop, h, evan
        return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(spectrum)))

    return propagate


def band_limited_decimate_fixed_window(
    field: np.ndarray, output_n: int
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reduce grid density without changing the physical window.

    This is the counterpart of
    :func:`vortex_system_route.fourier_resample_fixed_window`: the spectrum is
    cropped rather than zero padded, with the matching cell-centred origin
    correction.  It is information preserving only for a field already band
    limited inside the retained block, so the discarded power fraction is
    returned and must be checked by the caller.
    """

    source = np.asarray(field, dtype=np.complex128)
    if source.ndim != 2 or source.shape[0] != source.shape[1]:
        raise ValueError("field must be a square 2D array")
    input_n = int(source.shape[0])
    output_n = int(output_n)
    if output_n > input_n:
        raise ValueError("output_n must be at most the input grid size")

    if output_n == input_n:
        return source.copy(), {
            "decimation": "identity",
            "input_n": input_n,
            "output_n": output_n,
            "discarded_power_fraction": 0.0,
        }

    spectrum = np.fft.fftshift(np.fft.fft2(source))
    total = float(np.sum(np.abs(spectrum) ** 2))
    start = (input_n - output_n) // 2
    cropped = spectrum[start:start + output_n, start:start + output_n].copy()
    kept = float(np.sum(np.abs(cropped) ** 2))
    del spectrum

    # lean_xy_grid is cell centred, so changing N moves the first coordinate
    # from -L/2 + dx_in/2 to -L/2 + dx_out/2.  Undo that fractional origin shift
    # across the retained band, mirroring the zero-padding helper.
    delta_in_samples = 0.5 * (1.0 - float(output_n) / input_n)
    freq = np.fft.fftshift(np.fft.fftfreq(output_n, d=1.0))
    fy, fx = np.meshgrid(freq, freq, indexing="ij")
    cropped *= np.exp(1j * TWOPI * delta_in_samples * (fx + fy))

    out = np.fft.ifft2(np.fft.ifftshift(cropped)) * (float(output_n) / input_n) ** 2
    return out, {
        "decimation": "band_limited_spectral_crop_fixed_window",
        "input_n": input_n,
        "output_n": output_n,
        "discarded_power_fraction": float(max(0.0, (total - kept) / max(total, EPS))),
    }


# --------------------------------------------------------------------------
# registration configuration
# --------------------------------------------------------------------------

REGISTRATION_DOFS = ("slm1", "slm2", "common", "differential")
REGISTRATION_AXES = ("x", "y", "diagonal")

PIXEL_VALUE_MODELS = ("area_average", "centre_sample")


def snap_to_pixel_centres(
    coord: np.ndarray, pitch_m: float, origin_offset_m: float
) -> np.ndarray:
    """Return the centre of the pixel containing each coordinate."""

    p = float(pitch_m)
    o = float(origin_offset_m)
    return (np.floor((np.asarray(coord, dtype=float) - o) / p) + 0.5) * p + o


def commanded_pixel_phase(
    grid: Mapping[str, Any],
    command,
    *,
    error: SLMError,
    panel_cfg: Any,
    pixel_value_model: str,
) -> np.ndarray:
    """Evaluate one panel's commanded phase under the chosen pixel-value model.

    ``area_average``
        The repository convention: the continuous command is evaluated on the
        computational grid and :func:`pixelate` takes the circular mean over the
        samples inside each pixel.  The accuracy of that mean is a quadrature
        that depends on how many samples fall in a pixel, so it converges only
        as the grid is refined -- and it converges erratically unless the pitch
        is an exact integer number of samples, because otherwise neighbouring
        pixels hold different sample counts and translating the lattice changes
        which pixels hold the extra one.

    ``centre_sample``
        Each pixel is commanded with the ideal phase at its own centre, which is
        how hologram addressing software normally computes a pixel value.  This
        is exact by construction: there is no quadrature, so the result does not
        depend on grid density beyond representing the piecewise-constant
        result.

    The two are different physical statements about how the hologram is written,
    not two approximations to one truth, so the study reports both and treats
    their difference as a modelling uncertainty.
    """

    if pixel_value_model not in PIXEL_VALUE_MODELS:
        raise ValueError(f"pixel_value_model must be one of {PIXEL_VALUE_MODELS}")

    ox, oy = error.lattice_offset_m
    X = np.asarray(grid["X"], dtype=float)
    Y = np.asarray(grid["Y"], dtype=float)
    if pixel_value_model == "centre_sample":
        # The pixel lattice is physical, so snapping happens in laboratory
        # coordinates; the hologram transform is applied afterwards.
        X = snap_to_pixel_centres(X, panel_cfg.pitch_m, ox)
        Y = snap_to_pixel_centres(Y, panel_cfg.pitch_m, oy)

    cx = float(error.panel_translation_m[0]) + float(error.pattern_offset_m[0])
    cy = float(error.panel_translation_m[1]) + float(error.pattern_offset_m[1])
    xr = X - cx
    yr = Y - cy
    c = math.cos(float(error.pattern_rotation_rad))
    s = math.sin(float(error.pattern_rotation_rad))
    xp = (c * xr + s * yr) / float(error.pattern_scale_x)
    yp = (-s * xr + c * yr) / float(error.pattern_scale_y)
    return np.asarray(command(xp, yp), dtype=float)


def _axis_vector(axis: str) -> tuple[float, float]:
    if axis == "x":
        return (1.0, 0.0)
    if axis == "y":
        return (0.0, 1.0)
    if axis == "diagonal":
        return (1.0, 1.0)
    raise ValueError(f"unsupported registration axis {axis!r}")


def registration_panels(
    delta_m: float, *, dof: str, axis: str = "x"
) -> tuple[SLMError, SLMError]:
    """Return (slm1, slm2) errors for a pure sub-pixel lattice registration.

    The beam is never moved.  Each selected panel is translated by ``-delta``
    while its hologram is addressed ``+delta`` back, so the beam-to-hologram
    alignment stays exactly nominal and only the beam-to-pixel-lattice phase
    changes.  Without that compensation the sweep would be an ordinary hologram
    decentre study, which the Phase 2E registry already covers at the 25-pixel
    scale under ``slm1_hologram_offset_x``.
    """

    if dof not in REGISTRATION_DOFS:
        raise ValueError(f"dof must be one of {REGISTRATION_DOFS}; got {dof!r}")
    ux, uy = _axis_vector(axis)
    d = float(delta_m)

    def panel(sign: float) -> SLMError:
        tx, ty = -sign * d * ux, -sign * d * uy
        return SLMError(panel_translation_m=(tx, ty), pattern_offset_m=(-tx, -ty))

    idle = SLMError()
    if dof == "slm1":
        return panel(1.0), idle
    if dof == "slm2":
        return idle, panel(1.0)
    if dof == "common":
        return panel(1.0), panel(1.0)
    return panel(1.0), panel(-1.0)


# --------------------------------------------------------------------------
# route
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class RegistrationSampling:
    """Grid densities for one registration route."""

    fine_grid_n: int = 4096
    relay_grid_n: int = 2048
    window_m: float = BENCH_WINDOW_M

    def validate(self) -> None:
        if self.relay_grid_n > self.fine_grid_n:
            raise ValueError("relay_grid_n must not exceed fine_grid_n")
        if min(self.fine_grid_n, self.relay_grid_n) < 64:
            raise ValueError("grid sizes are too small to be meaningful")

    @property
    def fine_dx_m(self) -> float:
        return float(self.window_m) / int(self.fine_grid_n)

    def samples_per_pixel(self, pitch_m: float) -> float:
        return float(pitch_m) / self.fine_dx_m


def canonical_registration_hardware() -> dict[str, Any]:
    """Canonical bench constants used by the registration route."""

    m = canonical_hardware_manifest()
    return {
        "panel": _panel_from_manifest(m),
        "wavelength_m": float(hardware_value(m, "wavelength_m")),
        "beam_radius_on_slm_m": float(hardware_value(m, "beam_radius_on_slm_m")),
        "pixel_pitch_m": float(hardware_value(m, "slm_pixel_pitch_m")),
        "carrier_cpm": float(hardware_value(m, "carrier_frequency_cpm")),
        "fourf_focal_length_m": float(hardware_value(m, "fourf_focal_length_m")),
        "fourier_iris_radius_m": float(hardware_value(m, "fourier_iris_radius_m")),
        "axicon_base_angle_rad": math.radians(
            float(hardware_value(m, "axicon_base_angle_deg"))
        ),
        "axicon_refractive_index": float(hardware_value(m, "axicon_refractive_index")),
        "axicon_external_index": float(
            hardware_value(m, "axicon_external_medium_index")
        ),
    }


def _apply_panel(
    field: np.ndarray,
    grid: Mapping[str, Any],
    command,
    *,
    panel_cfg: Any,
    error: SLMError,
    pixel_pitch_m: float,
    pixelate_phase: bool,
    fill_factor_model: str,
    pixel_value_model: str = "area_average",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply one panel without materialising the full SLMApplication ledger.

    Mirrors :func:`vbb_study.slm_model.apply_slm` for the two fill-factor models
    whose loss is carried inside the field, but does not build the separate
    modulated/unmodulated/total arrays, which at fine sampling would triple the
    peak footprint.  Equality with ``apply_slm`` is asserted by the test suite.
    """

    lattice = error.lattice_offset_m
    if not pixelate_phase:
        commanded = commanded_pixel_phase(
            grid, command, error=error, panel_cfg=panel_cfg,
            pixel_value_model="area_average",  # continuous: no snapping either way
        )
        pixelation = "continuous_ideal_no_pixel_lattice"
    elif pixel_value_model == "centre_sample":
        # already piecewise constant on the lattice; pixelate would be a no-op
        commanded = commanded_pixel_phase(
            grid, command, error=error, panel_cfg=panel_cfg,
            pixel_value_model="centre_sample",
        )
        pixelation = "commanded_at_pixel_centres_on_physical_lattice"
    else:
        commanded = pixelate(
            commanded_pixel_phase(
                grid, command, error=error, panel_cfg=panel_cfg,
                pixel_value_model="area_average",
            ),
            grid, panel_cfg, lattice_offset_m=lattice,
        )
        pixelation = "area_averaged_onto_physical_pixel_lattice"

    actual, phase_meta = actual_slm_phase(
        commanded,
        grid,
        error=error,
        pixel_pitch_m=float(pixel_pitch_m),
        lut_phase_rad=None,
        static_phase_map_rad=None,
    )
    del commanded

    out = np.asarray(field, dtype=np.complex128) * np.exp(1j * actual)
    del actual

    aperture = slm_active_aperture(grid, panel_cfg, lattice_offset_m=lattice)
    out *= aperture
    del aperture

    ff = float(panel_cfg.fill_factor)
    if fill_factor_model == "throughput_only":
        out *= math.sqrt(ff)
        ff_status = "scalar_throughput_no_dead_space_lattice"
    elif fill_factor_model == "resolved_pixel_aperture":
        # The binary mask's own guard only asks for two samples per pixel, which
        # is far too weak here: at fill factor 0.93 the dead-space lines are
        # (1 - sqrt(FF)) * pitch wide -- 0.285 um for an 8 um pixel.  Sampling
        # coarser than that produces an aliased mask whose gaps catch samples
        # more or less at random as the lattice is translated, which shows up as
        # a large but entirely spurious registration signal.
        dead_space_m = (1.0 - math.sqrt(max(ff, 0.0))) * float(panel_cfg.pitch_m)
        if float(grid["dx"]) > 0.5 * dead_space_m:
            raise ValueError(
                f"grid dx = {float(grid['dx']) * 1e6:.3f} um cannot resolve the "
                f"{dead_space_m * 1e6:.3f} um inter-pixel dead space implied by fill "
                f"factor {ff:.3f}; the resolved pixel aperture would alias and "
                "manufacture a spurious registration signal.  Use "
                "fill_factor_model='throughput_only' and report the dead-space "
                "lattice as unresolved at this bench scale."
            )
        if not pixelate_phase:
            # The continuous reference must differ from the pixelated route by the
            # lattice ALONE.  A binary pixel aperture transmits a power fraction
            # equal to the fill factor, so the reference takes the same mean
            # throughput as a scalar and simply carries no lattice.
            out *= math.sqrt(ff)
            ff_status = "mean_throughput_matched_scalar_no_lattice"
        else:
            mask = resolved_pixel_aperture(
                grid, panel_cfg, lattice_offset_m=lattice
            )
            out *= mask
            del mask
            ff_status = "binary_dead_space_lattice_registered_to_panel"
    else:
        raise ValueError(
            "registration route supports 'throughput_only' or 'resolved_pixel_aperture'"
        )

    return out, {
        **phase_meta,
        "pixelation": pixelation,
        "pixel_value_model": pixel_value_model if pixelate_phase else "not_applicable",
        "fill_factor": ff,
        "fill_factor_model": fill_factor_model,
        "fill_factor_status": ff_status,
        "pixel_lattice_offset_m": lattice,
        "pattern_offset_m": tuple(map(float, error.pattern_offset_m)),
        "panel_translation_m": tuple(map(float, error.panel_translation_m)),
    }


def build_registration_route(
    case_id: str,
    *,
    sampling: RegistrationSampling = RegistrationSampling(),
    slm1: SLMError = SLMError(),
    slm2: SLMError = SLMError(),
    beam: GaussianBeamError = GaussianBeamError(),
    arm: str = "bench",
    pixelate_phase: bool = True,
    fill_factor_model: str = "resolved_pixel_aperture",
    axicon: AxiconError = AxiconError(),
    beam_radius_m: float | None = None,
    allow_unresolved_lattice: bool = False,
    pixel_value_model: str = "area_average",
    keep_fine_post_iris: bool = False,
) -> dict[str, Any]:
    """Build one registration route up to and including the axicon.

    Returns the field on the axicon input plane and immediately behind the
    axicon, both on the relay/axicon grid.  The axicon is a pure phase element
    when no clear aperture is bound, so the post-axicon *intensity* equals the
    incident intensity by construction; the Bessel-region consequence appears
    only after propagation, which the caller performs.

    A registration sweep on a grid coarser than two samples per pixel is refused
    unless ``allow_unresolved_lattice`` is set.  This is not defensive
    pedantry: when ``dx > pitch`` every computational sample falls in a pixel of
    its own, :func:`pixelate` degenerates into the identity, the pixel lattice
    leaves the model entirely, and a registration sweep returns *exactly* zero
    for every offset.  That null is indistinguishable at a glance from a
    physical finding of no sensitivity, so it must not be reachable by accident.
    The accepted Phase 2E source grids (dx=6.5 um against an 8 um pitch) are on
    the wrong side of this line, which is why this study needs its own sampling.
    """

    sampling.validate()
    beam.validate()
    slm1.validate()
    slm2.validate()
    axicon.validate()
    if arm not in ("bench", "isolation"):
        raise ValueError(f"arm must be 'bench' or 'isolation'; got {arm!r}")
    if pixel_value_model not in PIXEL_VALUE_MODELS:
        raise ValueError(f"pixel_value_model must be one of {PIXEL_VALUE_MODELS}")

    hw = canonical_registration_hardware()
    panel_cfg = hw["panel"]
    lam = hw["wavelength_m"]
    pitch = hw["pixel_pitch_m"]
    w0 = hw["beam_radius_on_slm_m"] if beam_radius_m is None else float(beam_radius_m)
    ell = _ell(case_id)

    samples_per_pixel = sampling.samples_per_pixel(pitch)
    lattice_moved = any(
        float(v) != 0.0
        for panel_error in (slm1, slm2)
        for v in panel_error.panel_translation_m
    )
    if pixelate_phase and samples_per_pixel < 2.0 and not allow_unresolved_lattice:
        raise ValueError(
            "this grid does not resolve the SLM pixel lattice "
            f"({samples_per_pixel:.2f} samples per {pitch * 1e6:.1f} um pixel; at least 2 "
            "are required).  A registration sweep here returns exactly zero for "
            "every offset because pixelate() degenerates into the identity.  Refine "
            "the grid, or pass allow_unresolved_lattice=True if you are deliberately "
            "reproducing the unresolved accepted route."
        )
    if lattice_moved and samples_per_pixel < 2.0 and not allow_unresolved_lattice:
        raise ValueError(
            "a panel translation was requested on a grid that cannot resolve the "
            "pixel lattice; the result would be an artificial null"
        )

    fine = lean_xy_grid(sampling.fine_grid_n, sampling.fine_dx_m)
    field, beam_meta = gaussian_input_field(
        fine, wavelength_m=lam, canonical_radius_m=w0, error=beam
    )

    field, slm1_meta = _apply_panel(
        field,
        fine,
        lambda xp, yp: float(ell) * np.arctan2(yp, xp),
        panel_cfg=panel_cfg,
        error=slm1,
        pixel_pitch_m=pitch,
        pixelate_phase=pixelate_phase,
        fill_factor_model=fill_factor_model,
        pixel_value_model=pixel_value_model,
    )

    meta: dict[str, Any] = {
        "route_id": "slm_pixel_registration_route_v1",
        "arm": arm,
        "case_id": case_id,
        "vortex_charge": ell,
        "wavelength_m": lam,
        "beam_radius_on_slm_m": w0,
        "beam_radius_in_pixels": w0 / pitch,
        "pixel_pitch_m": pitch,
        "fine_grid_n": int(sampling.fine_grid_n),
        "fine_dx_m": sampling.fine_dx_m,
        "samples_per_pixel": samples_per_pixel,
        "pixel_lattice_resolved": bool(samples_per_pixel >= 2.0),
        "pitch_commensurate": bool(
            abs(samples_per_pixel - round(samples_per_pixel)) < 1e-9
        ),
        "window_m": float(sampling.window_m),
        "beam": beam_meta,
        "slm1": slm1_meta,
        "pixelate_phase": bool(pixelate_phase),
        "pixel_value_model": pixel_value_model,
        "charge_convention": "full charge on SLM1; SLM2 carries the carrier only",
    }

    if arm == "isolation":
        coarse = fine
        meta.update(
            {
                "relay_grid_n": int(sampling.fine_grid_n),
                "relay_dx_m": sampling.fine_dx_m,
                "slm2": "not_used_in_isolation_arm",
                "fourf": "absent_by_construction",
                "decimation": {
                    "decimation": "identity",
                    "discarded_power_fraction": 0.0,
                },
                "isolation_scope": (
                    "mechanism study only; no carrier and no iris, so this is not "
                    "the bench architecture and must not be reported as a bench "
                    "prediction"
                ),
            }
        )
        field_on_axicon = field
        fine_post_iris = field if keep_fine_post_iris else None
    else:
        carrier = hw["carrier_cpm"]
        field, slm2_meta = _apply_panel(
            field,
            fine,
            lambda xp, yp: TWOPI * carrier * xp,
            panel_cfg=panel_cfg,
            error=slm2,
            pixel_pitch_m=pitch,
            pixelate_phase=pixelate_phase,
            fill_factor_model=fill_factor_model,
            pixel_value_model=pixel_value_model,
        )

        f4f = hw["fourf_focal_length_m"]
        field = lean_asm_propagate(field, fine, lam, f4f)
        field, lens1_meta = _apply_lens_plane(
            field,
            fine,
            wavelength_m=lam,
            focal_length_m=f4f,
            error=LensError(),
            opd_map_m=None,
        )
        field = lean_asm_propagate(field, fine, lam, f4f)

        centre = nominal_order_position_m(
            wavelength_m=lam, focal_length_m=f4f, carrier_cpm=carrier
        )
        iris = physical_iris(
            fine, radius_m=hw["fourier_iris_radius_m"], centre_m=centre
        )
        pre_iris_power = float(np.sum(np.abs(field) ** 2))
        field = field * iris
        selected_fraction = float(np.sum(np.abs(field) ** 2)) / max(pre_iris_power, EPS)
        del iris

        fine_post_iris = field.copy() if keep_fine_post_iris else None
        field, decim_meta = band_limited_decimate_fixed_window(
            field, sampling.relay_grid_n
        )
        coarse = lean_xy_grid(
            sampling.relay_grid_n, float(sampling.window_m) / sampling.relay_grid_n
        )

        field = lean_asm_propagate(field, coarse, lam, f4f)
        field, lens2_meta = _apply_lens_plane(
            field,
            coarse,
            wavelength_m=lam,
            focal_length_m=f4f,
            error=LensError(),
            opd_map_m=None,
        )
        field = lean_asm_propagate(field, coarse, lam, f4f)
        # A unity-magnification 4F relay inverts the image, so the input +G
        # carrier arrives as -G; +G removes it in the selected-order frame.
        field = field * np.exp(+1j * TWOPI * carrier * coarse["X"])

        meta.update(
            {
                "relay_grid_n": int(sampling.relay_grid_n),
                "relay_dx_m": float(coarse["dx"]),
                "slm2": slm2_meta,
                "fourf": {
                    "model": "explicit_4f_ASM_nominal_lenses_fixed_plus_one_iris",
                    "nominal_focal_length_m": f4f,
                    "nominal_carrier_cpm": carrier,
                    "iris_centre_m": tuple(map(float, centre)),
                    "iris_radius_m": hw["fourier_iris_radius_m"],
                    "iris_selected_power_fraction": selected_fraction,
                    "lens1": lens1_meta,
                    "lens2": lens2_meta,
                },
                "decimation": decim_meta,
                "selected_order_carrier_removal": "plus_G_after_4F_image_inversion",
            }
        )
        field_on_axicon = field

    axicon_t, axicon_meta = physical_axicon_on_own_plane(
        coarse,
        wavelength_m=lam,
        base_angle_rad=hw["axicon_base_angle_rad"],
        refractive_index=hw["axicon_refractive_index"],
        external_index=hw["axicon_external_index"],
        error=axicon,
        surface_height_error_m=None,
    )
    post_axicon = field_on_axicon * axicon_t
    del axicon_t

    radial_period_m = TWOPI / max(float(axicon_meta["exact_kr_m_inv"]), EPS)
    meta["axicon"] = axicon_meta
    meta["samples_per_axicon_radial_period"] = radial_period_m / float(coarse["dx"])
    meta["calibration_policy"] = {
        "SLM_LUT_and_static_maps": (
            "measured arrays required for absolute hardware claims"
        ),
        "SLM_fringing": "kernel must be fitted to actual panel diffraction data",
        "registration_offsets": (
            "geometric sensitivity values, not measured panel positions"
        ),
    }

    result = {
        "grid": coarse,
        "field_on_axicon_plane": np.asarray(field_on_axicon, dtype=np.complex128),
        "post_axicon": np.asarray(post_axicon, dtype=np.complex128),
        "metadata": meta,
    }
    if keep_fine_post_iris and fine_post_iris is not None:
        # The field immediately after the physical iris, still on the fine grid.
        # Comparing registration states here removes the spectral crop from the
        # comparison entirely, which matters on the bench arm where the
        # registration signal is smaller than the power the crop discards.
        result["fine_post_iris"] = np.asarray(fine_post_iris, dtype=np.complex128)
        result["fine_grid"] = fine
    return result
