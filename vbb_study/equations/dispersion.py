"""Auditable refractive-index dispersion models for broadband optics.

The optical solvers use SI units.  Sellmeier coefficients conventionally use
wavelength in micrometres, so the conversion is explicit here and nowhere else.
No material is selected implicitly: callers must choose a named model or build
one from measured/manufacturer coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class SellmeierMaterial:
    """Three-term Sellmeier material.

    ``B`` is dimensionless and ``C_um2`` is in micrometre squared.  The model is

        n(lambda)^2 = 1 + sum_j B_j * lambda_um^2/(lambda_um^2 - C_j).

    ``valid_wavelength_m`` is a governance bound, not a numerical fit action:
    values outside it raise rather than extrapolate silently.
    """

    name: str
    B: tuple[float, float, float]
    C_um2: tuple[float, float, float]
    valid_wavelength_m: tuple[float, float]
    source: str

    def refractive_index(self, wavelength_m: np.ndarray | float) -> np.ndarray:
        lam_m = np.asarray(wavelength_m, dtype=float)
        if np.any(~np.isfinite(lam_m)) or np.any(lam_m <= 0.0):
            raise ValueError("wavelength_m must be finite and positive")
        lo, hi = map(float, self.valid_wavelength_m)
        if np.any((lam_m < lo) | (lam_m > hi)):
            raise ValueError(
                f"{self.name} Sellmeier model is only authorised on "
                f"[{lo:.6g}, {hi:.6g}] m"
            )
        lam_um2 = (lam_m * 1.0e6) ** 2
        n2 = np.ones_like(lam_um2, dtype=float)
        for b, c in zip(self.B, self.C_um2):
            denominator = lam_um2 - float(c)
            if np.any(np.isclose(denominator, 0.0, rtol=0.0, atol=1e-15)):
                raise ValueError("wavelength lies on a Sellmeier pole")
            n2 += float(b) * lam_um2 / denominator
        if np.any(n2 <= 0.0):
            raise FloatingPointError("Sellmeier evaluation produced n^2 <= 0")
        return np.sqrt(n2)

    def group_index(self, wavelength_m: np.ndarray | float) -> np.ndarray:
        """Return n_g = n - lambda dn/dlambda using the analytic derivative."""

        lam_m = np.asarray(wavelength_m, dtype=float)
        n = self.refractive_index(lam_m)
        lam_um = lam_m * 1.0e6
        lam_um2 = lam_um**2
        d_n2_d_lam_um = np.zeros_like(lam_um, dtype=float)
        for b, c in zip(self.B, self.C_um2):
            d_n2_d_lam_um += -2.0 * float(b) * float(c) * lam_um / (lam_um2 - float(c)) ** 2
        dn_dlam_m = (0.5 * d_n2_d_lam_um / n) * 1.0e6
        return n - lam_m * dn_dlam_m

    def group_velocity_dispersion_s2_per_m(self, wavelength_m: float) -> float:
        """Return beta2=d^2k/domega^2 by a stable centred frequency derivative.

        A small fractional frequency step is used only to differentiate the
        analytic Sellmeier law; this is not a fitted or experimental parameter.
        """

        c0 = 299_792_458.0
        lam = float(wavelength_m)
        self.refractive_index(lam)  # validates range first
        omega0 = 2.0 * np.pi * c0 / lam
        h = omega0 * 1.0e-5

        def beta(omega: float) -> float:
            wl = 2.0 * np.pi * c0 / float(omega)
            return float(self.refractive_index(wl)) * float(omega) / c0

        return float((beta(omega0 + h) - 2.0 * beta(omega0) + beta(omega0 - h)) / (h * h))


@dataclass(frozen=True)
class ConstantIndexMaterial:
    """Explicit constant-index model for controlled comparisons only."""

    name: str
    index: float
    source: str = "user_supplied_constant_index"

    def refractive_index(self, wavelength_m: np.ndarray | float) -> np.ndarray:
        wl = np.asarray(wavelength_m, dtype=float)
        if np.any(~np.isfinite(wl)) or np.any(wl <= 0.0):
            raise ValueError("wavelength_m must be finite and positive")
        if not np.isfinite(self.index) or self.index <= 0.0:
            raise ValueError("constant refractive index must be finite and positive")
        return np.full_like(wl, float(self.index), dtype=float)


# Malitson fused-silica coefficients.  These are intentionally named rather
# than made a global default so a different sample/axicon glass cannot inherit
# fused-silica dispersion accidentally.
FUSED_SILICA_MALITSON = SellmeierMaterial(
    name="fused_silica_Malitson",
    B=(0.6961663, 0.4079426, 0.8974794),
    C_um2=(0.0684043**2, 0.1162414**2, 9.896161**2),
    valid_wavelength_m=(0.21e-6, 3.71e-6),
    source="Malitson 1965 fused-silica Sellmeier coefficients",
)


def refractive_index_table(
    material: SellmeierMaterial | ConstantIndexMaterial,
    wavelengths_m: Iterable[float],
) -> list[dict[str, float | str]]:
    wavelengths = np.asarray(tuple(wavelengths_m), dtype=float)
    indices = material.refractive_index(wavelengths)
    return [
        {
            "material": material.name,
            "wavelength_m": float(wl),
            "refractive_index": float(n),
            "source": material.source,
        }
        for wl, n in zip(wavelengths, indices)
    ]


__all__ = [
    "ConstantIndexMaterial",
    "FUSED_SILICA_MALITSON",
    "SellmeierMaterial",
    "refractive_index_table",
]
