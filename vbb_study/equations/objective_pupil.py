"""Objective-pupil and first-order-filter equations.

This module owns the compact scalar formulas for:

* objective pupil radius from NA and focal length,
* focal-plane pixel size from SLM pixel pitch and demagnification,
* SLM-plane to pupil-plane magnification in a 4f relay,
* Fourier-plane ring radius for a Bessel field (where the filter sits),
* Gaussian beam fill fraction through a hard pupil aperture,
* first-order filter inner and outer radius estimates.

What this module does NOT own:

* the full propagation pipeline from SLM to focal plane
  (``bessel_twin_core`` owns that),
* the ``ObjectiveMap`` coordinate-plane bookkeeping
  (``vbb_study.vbb_planes`` owns that),
* aberration phase maps
  (``equations.interface`` owns the low-order formulae).

Coordinate convention:
  All radii are at the plane implied by the argument name.
  ``f_eff_m`` is the effective focal length of the objective at the
  design tube-lens spacing, not the nominal manufacturer value.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from vbb_study import vbb_planes

EPS = 1.0e-30


# ---------------------------------------------------------------------------
# Pupil size and pixel scale
# ---------------------------------------------------------------------------


def pupil_radius_m(f_eff_m: float, NA: float, n_medium: float = 1.0) -> float:
    """Return the objective back-pupil radius in metres.

    The scalar ideal-lens formula is ``R_pupil = f_eff * NA / n``.

    Parameters
    ----------
    f_eff_m:
        Effective focal length of the objective at the working wavelength, m.
    NA:
        Numerical aperture (in the medium).
    n_medium:
        Refractive index of the immersion medium (1.0 for air).
    """

    return float(f_eff_m) * float(NA) / max(float(n_medium), EPS)


def focal_plane_pixel_size_m(slm_pixel_pitch_m: float, demag: float) -> float:
    """Return the effective pixel size in the focal/surface plane in metres.

    The demagnification ``M = f_obj / f_relay`` converts the SLM pixel pitch
    to an equivalent size in the focused plane.

    Parameters
    ----------
    slm_pixel_pitch_m:
        SLM pixel pitch, m.
    demag:
        Demagnification factor ``sample_size / slm_size`` (< 1 for a
        focusing relay).
    """

    return float(slm_pixel_pitch_m) * abs(float(demag))


def slm_to_pupil_magnification(f_relay_m: float, f_tube_m: float) -> float:
    """Return the SLM-plane to objective-back-pupil magnification.

    In a standard 4f SLM relay with focal lengths ``f_relay`` (beam-shaping
    lens) and ``f_tube`` (tube lens that images the SLM onto the pupil), the
    transverse magnification is ``M = f_tube / f_relay``.  Values > 1 mean
    the beam is expanded at the pupil; < 1 means it is contracted.

    Parameters
    ----------
    f_relay_m:
        Focal length of the relay lens closest to the SLM, m.
    f_tube_m:
        Focal length of the lens that images onto the objective back pupil, m.
    """

    return float(f_tube_m) / max(float(f_relay_m), EPS)


# ---------------------------------------------------------------------------
# Fourier-plane geometry (where the first-order filter lives)
# ---------------------------------------------------------------------------


def fourier_plane_ring_radius_m(
    kr_m_inv: float,
    f_lens_m: float,
    wavelength_m: float,
) -> float:
    """Return the Bessel-ring radius in the Fourier plane of a lens.

    A lens of focal length ``f`` maps the transverse spatial frequency
    ``k_r / (2*pi)`` (cycles/m) to a radial position
    ``r = wavelength * f * k_r / (2*pi) = f * k_r / k0`` in its back focal
    plane.  This is where the
    Bessel ring appears, and where the first-order filter must be centred.

    Parameters
    ----------
    kr_m_inv:
        Transverse wavevector (axicon ring spatial frequency), rad/m.
    f_lens_m:
        Focal length of the Fourier-transform lens, m.
    wavelength_m:
        Vacuum wavelength, m. It is explicit because this function returns a
        physical distance rather than a spatial-frequency coordinate.
    """

    return float(wavelength_m) * float(kr_m_inv) * float(f_lens_m) / (2.0 * math.pi)


def fourier_plane_carrier_separation_m(
    carrier_cpm: float,
    f_lens_m: float,
    wavelength_m: float,
) -> float:
    """Return the lateral shift of the blaze carrier in the Fourier plane.

    A carrier grating with frequency ``carrier_cpm`` (cycles/m) shifts the
    first diffraction order by ``wavelength * f * carrier_cpm`` in the
    Fourier plane.
    The filter must be positioned at this offset to capture the first order.

    Parameters
    ----------
    carrier_cpm:
        Blaze carrier spatial frequency, cycles/m.
    f_lens_m:
        Focal length of the Fourier-transform lens, m.
    wavelength_m:
        Vacuum wavelength, m.
    """

    return float(wavelength_m) * float(carrier_cpm) * float(f_lens_m)


# ---------------------------------------------------------------------------
# First-order filter radius estimates
# ---------------------------------------------------------------------------


def first_order_filter_inner_radius_m(
    carrier_cpm: float,
    f_lens_m: float,
    kr_m_inv: float,
    wavelength_m: float,
    *,
    safety_factor: float = 0.5,
) -> float:
    """Estimate the minimum safe inner radius of the first-order filter.

    The filter must exclude the DC order (at the Fourier-plane origin) and
    any higher Bessel rings.  The minimum inner radius is chosen to sit
    between the carrier-shifted ring and the DC spot.

    Parameters
    ----------
    carrier_cpm:
        Blaze carrier spatial frequency, cycles/m.
    f_lens_m:
        Focal length of the Fourier-transform lens, m.
    kr_m_inv:
        Bessel-ring transverse wavevector, rad/m.
    safety_factor:
        Fraction of the carrier separation used as inner clearance (0–1).
    """

    ring_r = fourier_plane_ring_radius_m(kr_m_inv, f_lens_m, wavelength_m)
    carrier_shift = fourier_plane_carrier_separation_m(carrier_cpm, f_lens_m, wavelength_m)
    # The first-order ring sits at carrier_shift ± ring_r.
    # Inner edge clears the DC spot; set to carrier_shift - ring_r with margin.
    inner = max(carrier_shift - ring_r * (1.0 + float(safety_factor)), 0.0)
    return float(inner)


def first_order_filter_outer_radius_m(
    carrier_cpm: float,
    f_lens_m: float,
    kr_m_inv: float,
    wavelength_m: float,
    *,
    safety_factor: float = 0.5,
) -> float:
    """Estimate the maximum safe outer radius of the first-order filter.

    The filter must exclude the second-order ring (at twice the carrier
    shift).  The outer radius is set between the ring and the second order.

    Parameters
    ----------
    safety_factor:
        Fraction of the ring-to-second-order gap retained as outer margin (0–1).
    """

    ring_r = fourier_plane_ring_radius_m(kr_m_inv, f_lens_m, wavelength_m)
    carrier_shift = fourier_plane_carrier_separation_m(carrier_cpm, f_lens_m, wavelength_m)
    # Second-order ring sits at 2*carrier_shift.  Outer edge should be less.
    outer = carrier_shift + ring_r * (1.0 + float(safety_factor))
    outer = min(outer, 2.0 * carrier_shift - ring_r * float(safety_factor))
    return float(max(outer, ring_r))


# ---------------------------------------------------------------------------
# Gaussian beam fill and clipping
# ---------------------------------------------------------------------------


def gaussian_pupil_fill_fraction(beam_1e_radius_m: float, pupil_radius_m_val: float) -> float:
    """Return the power fraction of a Gaussian beam passing through a circular pupil.

    The Gaussian beam has ``1/e`` field amplitude radius ``w0``
    (equivalently, ``1/e^2`` intensity radius ``w0``).  The fraction of
    total power within a hard aperture of radius ``R_pupil`` is::

        P_through / P_total = 1 - exp(-2 * R_pupil^2 / w0^2)

    Parameters
    ----------
    beam_1e_radius_m:
        Gaussian beam ``1/e`` amplitude radius at the pupil plane, m.
    pupil_radius_m_val:
        Pupil hard aperture radius, m.
    """

    w = max(float(beam_1e_radius_m), EPS)
    R = float(pupil_radius_m_val)
    return float(1.0 - math.exp(-2.0 * R**2 / w**2))


def gaussian_clipping_power_fraction(beam_1e_radius_m: float, pupil_radius_m_val: float) -> float:
    """Return the power fraction clipped (lost) at a circular pupil.

    Complement of :func:`gaussian_pupil_fill_fraction`.
    """

    return 1.0 - gaussian_pupil_fill_fraction(beam_1e_radius_m, pupil_radius_m_val)


def pupil_fill_ratio(beam_1e_radius_m: float, pupil_radius_m_val: float) -> float:
    """Return the dimensionless fill ratio ``w0 / R_pupil``.

    A fill ratio close to 1 means the beam nearly fills the pupil.
    Ratios > 1 indicate the beam overfills the aperture (significant clipping).
    """

    return float(beam_1e_radius_m) / max(float(pupil_radius_m_val), EPS)


def objective_map_from_design_inputs(
    laser: Any,
    target: Any,
    material: Any,
    beam_radius_on_slm_m: float | None = None,
):
    """Return the explicit SLM/free-space -> sample transverse map.

    The current inverse design chooses the sample Bessel-Gauss waist from the
    requested sample-plane zone length, then maps the SLM Gaussian radius onto
    that waist.  This helper keeps that cross-plane conversion named and
    auditable while preserving the legacy numerical value exactly.
    """

    D = max(float(target.target_core_diameter_m), EPS)
    L = max(float(target.target_bessel_length_m), EPS)
    w_slm = float(laser.beam_radius_on_slm_m if beam_radius_on_slm_m is None else beam_radius_on_slm_m)
    k_medium = laser.k0 * float(material.refractive_index)
    # Compatibility contract: D is the equivalent ell=0 first-zero diameter,
    # not the measured bright-ring diameter for vortex beams.
    kr_sample = 2.0 * 2.405 / D
    w0_sample = L * kr_sample / max(k_medium, EPS)
    return vbb_planes.objective_map_from_waists(
        pre_objective_radius_m=w_slm,
        sample_radius_m=w0_sample,
        n_sample=float(material.refractive_index),
        source="compute_design_from_targets:w0_sample/beam_radius_on_slm",
    )


def headline_length_tags(config: Any) -> dict[str, Any]:
    """Return JSON-friendly plane tags for headline length quantities."""

    return dict(vbb_planes.headline_lengths_jsonable(config))
