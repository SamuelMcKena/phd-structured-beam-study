"""Vector Bessel source modes for the existing three-component vector ASM.

Convention: exp(+i*k_z*z-i*omega*t). Radial and azimuthal modes are exact
cylindrical vector-Bessel families; Cartesian states are explicitly labelled
seeds and should pass through the repository spectral transversality projector.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping
import numpy as np
from scipy import special

Polarisation = Literal["radial", "azimuthal", "linear_x", "linear_y", "circular_plus", "circular_minus"]

@dataclass(frozen=True)
class VectorBesselComponents:
    ex: np.ndarray
    ey: np.ndarray
    ez: np.ndarray
    x: np.ndarray
    y: np.ndarray
    k_t: float
    k_z: float
    polarisation: Polarisation
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def intensity(self):
        return np.abs(self.ex)**2 + np.abs(self.ey)**2 + np.abs(self.ez)**2

    @property
    def longitudinal_fraction(self):
        total = float(np.sum(self.intensity))
        return float(np.sum(np.abs(self.ez)**2) / max(total, np.finfo(float).tiny))

    def to_vector_field(self, *, wavelength_m: float, medium_index: float = 1.0):
        from vbb_study.vector_field import VectorField
        dx = float(np.median(np.diff(self.x))); dy = float(np.median(np.diff(self.y)))
        fx = np.fft.fftshift(np.fft.fftfreq(self.x.size, d=dx))
        fy = np.fft.fftshift(np.fft.fftfreq(self.y.size, d=dy))
        FX, FY = np.meshgrid(fx, fy, indexing="xy")
        return VectorField(ex=self.ex, ey=self.ey, ez=self.ez,
            grid={"x": self.x, "y": self.y, "dx": dx, "dy": dy, "FX": FX, "FY": FY},
            wavelength_m=float(wavelength_m), medium_index=float(medium_index),
            metadata={**dict(self.metadata), "vector_bessel_polarisation": self.polarisation})

def dispersion_kz(k: float, k_t: float) -> float:
    if k <= 0 or k_t < 0 or k_t >= k:
        raise ValueError("require k>0 and 0<=k_t<k")
    return float(np.sqrt(k*k-k_t*k_t))

def _grid(x, y):
    x=np.asarray(x,float); y=np.asarray(y,float); X,Y=np.meshgrid(x,y,indexing="xy")
    return x,y,np.hypot(X,Y),np.arctan2(Y,X)

def cylindrical_vector_bessel(x, y, *, k_t: float, k_z: float,
        polarisation: Literal["radial","azimuthal"], amplitude: complex=1+0j, z: float=0.0):
    if k_t < 0 or k_z <= 0: raise ValueError("require k_t>=0 and k_z>0")
    x,y,r,phi=_grid(x,y); a=complex(amplitude)*np.exp(1j*k_z*z); j1=special.jv(1,k_t*r)
    if polarisation == "radial":
        er=a*j1; ex=er*np.cos(phi); ey=er*np.sin(phi)
        ez=1j*(k_t/k_z)*a*special.jv(0,k_t*r)
    elif polarisation == "azimuthal":
        ep=a*j1; ex=-ep*np.sin(phi); ey=ep*np.cos(phi); ez=np.zeros_like(ex,dtype=complex)
    else: raise ValueError("polarisation must be radial or azimuthal")
    return VectorBesselComponents(ex,ey,ez,x,y,float(k_t),float(k_z),polarisation,
        {"exact_cylindrical_vector_mode":True,"phase_convention":"exp(+i*kz*z-i*omega*t)"})

def projected_seed_bessel(x, y, *, k_t: float, k_z: float,
        polarisation: Literal["linear_x","linear_y","circular_plus","circular_minus"],
        order: int=0, amplitude: complex=1+0j, z: float=0.0):
    x,y,r,phi=_grid(x,y); s=complex(amplitude)*special.jv(order,k_t*r)*np.exp(1j*order*phi+1j*k_z*z)
    if polarisation=="linear_x": ex,ey=s,np.zeros_like(s)
    elif polarisation=="linear_y": ex,ey=np.zeros_like(s),s
    elif polarisation=="circular_plus": ex,ey=s/np.sqrt(2),1j*s/np.sqrt(2)
    elif polarisation=="circular_minus": ex,ey=s/np.sqrt(2),-1j*s/np.sqrt(2)
    else: raise ValueError("unsupported seed polarisation")
    return VectorBesselComponents(ex,ey,np.zeros_like(s,dtype=complex),x,y,float(k_t),float(k_z),polarisation,
        {"exact_cylindrical_vector_mode":False,"requires_spectral_projection":True,"order":int(order)})

def analytic_radial_divergence_residual(r, *, k_t: float, k_z: float, amplitude: complex=1+0j):
    r=np.asarray(r,float); j0=special.jv(0,k_t*r); a=complex(amplitude)
    return a*k_t*j0 + 1j*k_z*(1j*k_t/k_z)*a*j0

__all__=["Polarisation","VectorBesselComponents","dispersion_kz","cylindrical_vector_bessel","projected_seed_bessel","analytic_radial_divergence_residual"]
