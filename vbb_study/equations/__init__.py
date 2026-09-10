"""Formula-level helpers for the vortex-Bessel-beam study.

This package keeps the equations I repeatedly use in notebooks and tables
grouped by topic so they can be audited against the written derivations.

Submodules
----------
scalar_bessel
    Bessel-Gauss and conical axicon field definitions, core radii, ring
    radii, non-diffracting length.

propagation
    Medium wavelength and wavenumber, ASM longitudinal wavenumber,
    Matsushima band-limit mask, Nyquist margins, discrete power.

fields
    Engine-compatible scalar grids, centered FFTs, phase wrapping and
    quantisation, Gaussian amplitudes, and transverse wavevector helpers.

holography
    Axicon phase, spiral phase plate, combined phase, signum flip, blaze
    carrier, phase quantisation, greyscale conversion, fill-factor mask.

objective_pupil
    Objective pupil radius from NA, focal-plane pixel size, SLM-to-pupil
    magnification, Fourier-plane ring position, first-order filter radii,
    Gaussian pupil fill fraction.

calibrated_objective
    Measured objective-pupil amplitude/OPD/mask transfer around the accepted
    vector Debye/Richards-Wolf solver.

dispersion
    Explicit wavelength-dependent Sellmeier material models and constant-index
    comparison models. No material is selected implicitly.

vector_surface_refraction
    Exact local vector Snell/Fresnel surface-boundary operator for arbitrary
    surface normals. Curved-surface spatial remapping remains a separate solver
    responsibility.

metrics
    Engine-compatible radial metrics and strict Bessel-region adapters.

interface
    Snell's law for cone angles, transverse wavevector conservation,
    non-diffracting length in sample, interface aberration pupil phase,
    piston/defocus/spherical Zernike fit.

vector_jones
    Jones vectors for linear/circular/radial/azimuthal polarisation,
    Stokes parameters, Müller matrix helpers.

materials
    Fluence, incubation proxy, line-fluence proxy, material threshold
    planning formulas.

capsule_geometry
    Capsule target masks, overlap scores, accepted-depth and proxy geometry
    formulas.

polygonal
    Regular polygon radial profile, discrete N-fold plane-wave field,
    axial interference period, polygon area/perimeter, symmetry metric.
"""

from __future__ import annotations

from . import (
    calibrated_objective,
    capsule_geometry,
    dispersion,
    fields,
    holography,
    interface,
    materials,
    metrics,
    objective_pupil,
    polygonal,
    propagation,
    scalar_bessel,
    vector_jones,
    vector_surface_refraction,
)

__all__ = [
    "calibrated_objective",
    "capsule_geometry",
    "dispersion",
    "fields",
    "holography",
    "interface",
    "materials",
    "metrics",
    "objective_pupil",
    "polygonal",
    "propagation",
    "scalar_bessel",
    "vector_jones",
    "vector_surface_refraction",
]
