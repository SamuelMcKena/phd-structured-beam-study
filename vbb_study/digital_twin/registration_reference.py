"""Independent pixel-integrated ideal-4F registration reference.

Flat-correction effective-channel baseline. Does not invent an inter-panel
transfer. Pixel phase is commanded at centres; Gaussian illumination is
integrated over the intersections of the two physical pixel grids. Rectangular
subcells have analytic sinc transforms, so an eighth-pitch displacement is not
rounded to a computational raster. Quadrature subdivides illumination only,
never the commanded pixel phase. Ideal 4F selection is performed in SOURCE
frequency space, avoiding coarsening a physical Fourier-plane field.
"""
from dataclasses import dataclass
import math
import numpy as np
from scipy import fft
from scipy.special import erf


@dataclass(frozen=True)
class ReferenceParameters:
    pitch_m: float = 8e-6
    wavelength_m: float = 1029e-9
    carrier_cpm: float = 6250.
    filter_radius_cpm: float = 2500.
    focal_length_m: float = .150  # equal-f idealisation of user 300 mm separation
    axicon_base_angle_deg: float = 20.
    refractive_index: float = 1.458
    phase_levels: int = 256
    fill_factor: float = .93
    active_width_m: float = .01536
    active_height_m: float = .00864
    cone_kr_override_m_inv: float | None = None

    def validate(self):
        if min(self.pitch_m, self.wavelength_m, self.filter_radius_cpm,
               self.focal_length_m, self.active_width_m, self.active_height_m) <= 0:
            raise ValueError("lengths, frequency bandwidth and wavelength must be positive")
        if self.phase_levels < 2 or not 0 < self.fill_factor <= 1:
            raise ValueError("invalid panel response")

    @property
    def beta(self):
        a = math.radians(self.axicon_base_angle_deg)
        return math.asin(self.refractive_index * math.sin(a)) - a

    @property
    def kr(self):
        if self.cone_kr_override_m_inv is not None:
            return float(self.cone_kr_override_m_inv)
        return 2 * math.pi / self.wavelength_m * math.sin(self.beta)


def periodic_rho(rho):
    return tuple(float(round(float(v) % 1., 12) % 1.) for v in rho)


def pixel_centres(x, rho, p):
    g = -periodic_rho((rho,))[0] * p
    return (np.floor((np.asarray(x) - g) / p) + .5) * p + g


def panel_phase(x, y, charge, rho, params, carrier=True):
    xc = pixel_centres(x, rho[0], params.pitch_m)
    yc = pixel_centres(y, rho[1], params.pitch_m)
    phase = charge * np.arctan2(yc, xc)
    if carrier:
        phase = phase + 2 * np.pi * params.carrier_cpm * xc
    step = 2 * np.pi / params.phase_levels
    return (np.floor(np.mod(phase, 2*np.pi) / step + .5) % params.phase_levels) * step


def intervals(r1, r2, p, q):
    edges = sorted(set([0., p] + [round((-r % 1.) * p, 16)
                                  for r in (r1, r2) if (-r % 1.) > 1e-10]))
    result = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        for j in range(q):
            width = (hi-lo)/q
            result.append((lo+(j+.5)*width, width))
    return result


def integrated_spectrum(charge, w_px, rho1=(0.,0.), rho2=(0.,0.), *,
                        architecture="upstream_vortex", params=ReferenceParameters(),
                        quadrature=1, n_pixels=None, carrier=True, spectral_replicas=0):
    """Fourier integral of the two panel product on a pixel-commensurate window.

    Returned S has units of area for unit incident amplitude. The base window is extended by explicitly requested physical replica
    windows when a diagnostic iris requires them. Sinc and origin factors use
    the true physical frequencies; omitted orders are not claimed.
    """
    params.validate()
    if architecture not in ("upstream_vortex", "downstream_vortex"):
        raise ValueError("unknown architecture")
    if w_px <= 0 or quadrature < 1:
        raise ValueError("positive beam radius and quadrature required")
    rho1, rho2 = periodic_rho(rho1), periodic_rho(rho2)
    p = params.pitch_m
    n = int(n_pixels or max(240, 20*math.ceil(6*w_px/20)))
    if n % 20:
        raise ValueError("n_pixels must be a multiple of 20 for exact carrier bins")
    W = n*p
    x = (np.arange(n)-n/2)*p
    replicas=int(spectral_replicas)
    if replicas < 0:
        raise ValueError('spectral_replicas must be non-negative')
    nf=n*(2*replicas+1)
    f = np.arange(-nf//2,nf//2)/W
    S = np.zeros((nf,nf), complex)
    l1, l2 = (charge,0) if architecture == "upstream_vortex" else (0,charge)
    for ox, wx in intervals(rho1[0],rho2[0],p,quadrature):
        for oy, wy in intervals(rho1[1],rho2[1],p,quadrature):
            X, Y = np.meshgrid(x+ox,x+oy)
            A = np.exp(-(X*X+Y*Y)/(w_px*p)**2)
            # Each actual panel aperture moves with its own lattice.
            for rho in (rho1,rho2):
                A *= ((np.abs(X+rho[0]*p) <= params.active_width_m/2) &
                      (np.abs(Y+rho[1]*p) <= params.active_height_m/2))
            phase = panel_phase(X,Y,l1,rho1,params,carrier)
            phase += panel_phase(X,Y,l2,rho2,params,carrier)
            a = params.fill_factor*A*np.exp(1j*phase)
            sf = fft.fftshift(fft.fft2(fft.ifftshift(a),workers=2))
            if replicas:
                # The lattice sum repeats, but the physical top-hat transform
                # and subcell-origin phase do not. This is NOT an aliased FFT
                # extrapolation: sinc and origin factors use the true f below.
                sf=np.tile(sf,(2*replicas+1,2*replicas+1))
            kx = wx*np.sinc(f*wx)*np.exp(-2j*np.pi*f*ox)
            ky = wy*np.sinc(f*wy)*np.exp(-2j*np.pi*f*oy)
            S += sf*ky[:,None]*kx[None,:]
    return S, f, W


def selected_coefficients(S, f, W, *, params=ReferenceParameters(), carrier=True):
    """Exact ideal 4F filter in source spatial frequency coordinates.

    Image parity is unfolded identically in A/B. A centred radial axicon is
    parity invariant, so this does not change relative registration conclusions.
    """
    centre = 2*params.carrier_cpm if carrier else 0.
    half = int(math.ceil(params.filter_radius_cpm*W))
    ix = int(np.argmin(abs(f-centre)))
    iy = len(f)//2
    if ix-half < 0 or ix+half >= len(f):
        raise ValueError("selected order plus iris exceeds source-frequency window")
    ff = np.arange(-half,half+1)/W
    FX,FY=np.meshgrid(ff,ff)
    C = S[iy-half:iy+half+1,ix-half:ix+half+1].copy()/W**2
    C *= FX*FX+FY*FY <= params.filter_radius_cpm**2
    return C


def synthesize(C, W, n):
    """Reconstruct selected envelope on a cell-centred periodic square."""
    m=C.shape[0]; h=m//2
    if n < m+2:
        raise ValueError("output Nyquist is too small")
    spectrum=np.zeros((n,n),np.complex64)
    f=np.arange(-h,h+1)/W
    # FFT samples are at -W/2; shift to cell centres without interpolation.
    phase=np.exp(2j*np.pi*f*W/(2*n))
    spectrum[n//2-h:n//2+h+1,n//2-h:n//2+h+1]=C*phase[:,None]*phase[None,:]*n**2
    return fft.fftshift(fft.ifft2(fft.ifftshift(spectrum),workers=2))


def propagate_axicon(C, W, z_values, *, params=ReferenceParameters(), dx_target=2e-6,
                     crop_half_m=150e-6, propagation_window_m=None):
    """Scalar exact-ASM cone reference. Returns cropped fields and power drift.

    The high-angle axicon is an ideal effective cone phase, not a thick-surface
    vector or calibrated model. Propagation has no objective or material interface.
    """
    Wout=float(propagation_window_m or W)
    n=fft.next_fast_len(int(math.ceil(Wout/dx_target)))
    if n % 2: n=fft.next_fast_len(n+1)
    dx=Wout/n; x=(np.arange(n)-n/2+.5)*dx
    r=np.hypot(x[:,None],x[None,:])
    if Wout==W:
        E=synthesize(C,W,n)
    else:
        h=C.shape[0]//2; ff=np.arange(-h,h+1)/W
        basis=np.exp(2j*np.pi*x[:,None]*ff[None,:]).astype(np.complex64)
        E=(basis@C.astype(np.complex64))@basis.T
    E *= np.exp(-1j*params.kr*r).astype(np.complex64)
    del r
    power0=float(np.sum(np.abs(E)**2,dtype=np.float64)*dx**2)
    full_power=float(np.sum(abs(C)**2)*W**2)
    spectrum=fft.fft2(fft.ifftshift(E),workers=2)
    del E
    f=fft.fftfreq(n,dx)
    s2=(params.wavelength_m*f[:,None])**2+(params.wavelength_m*f[None,:])**2
    kz=(2*np.pi/params.wavelength_m)*(np.sqrt(np.maximum(1-s2,0))-1)
    valid=s2 <= 1
    sl=np.flatnonzero(abs(x)<=crop_half_m)
    crops=[]; rows=[]
    for z in z_values:
        H=np.exp(1j*kz*z).astype(np.complex64)*valid
        out=fft.fftshift(fft.ifft2(spectrum*H,workers=2))
        power=float(np.sum(np.abs(out)**2,dtype=np.float64)*dx**2)
        crops.append(out[np.ix_(sl,sl)].copy())
        border=max(1,int(.02*n))
        I=np.abs(out)**2
        edge=float((I[:border].sum()+I[-border:].sum()+I[border:-border,:border].sum()+I[border:-border,-border:].sum())/I.sum())
        rows.append(dict(z_m=float(z),power_ratio=power/power0,
                         propagation_power_drift_fraction=abs(power/power0-1),
                         boundary_power_fraction=edge,global_peak=float(I.max()),
                         input_window_retained_power_fraction=power0/full_power,
                         n=n,dx_m=dx))
    return np.asarray(crops),x[sl],rows


def spectrum_fidelity(C, reference):
    C=np.asarray(C,dtype=np.complex128)
    reference=np.asarray(reference,dtype=np.complex128)
    den=np.vdot(C,C).real*np.vdot(reference,reference).real
    return float(np.clip(abs(np.vdot(reference,C))**2/max(den,1e-300),0,1))


def local_fidelity(charge, w_px, rho1=(0.,0.), rho2=(0.,0.), *,
                   params=ReferenceParameters(), carrier=True):
    """Analytic Gaussian-weighted overlap after the two phase-only panels.

    Partition the union of both lattices and the reference lattice. The
    command product is constant in each rectangle and |A|² has separable erf
    integrals. No sub-pixel raster approximation is used. The finite integration
    domain is chosen inside the panel; lost Gaussian tails are below 1e-8 for
    the stress radii and below 2e-5 at the clipped bench aperture.
    """
    p=params.pitch_m; w=w_px*p
    n=int(max(240,20*math.ceil(6*w_px/20)))
    base=(np.arange(n)-n/2)*p
    rho1,rho2=periodic_rho(rho1),periodic_rho(rho2)
    def split(rs):
        return sorted(set([0.,p]+[round((-r%1)*p,16) for r in rs if (-r%1)>1e-10]))
    ex=split([0,rho1[0],rho2[0]]); ey=split([0,rho1[1],rho2[1]])
    inner=0j; total=0.
    for lx,hx in zip(ex[:-1],ex[1:]):
        xm=base+(lx+hx)/2
        wx=erf(np.sqrt(2)*(base+hx)/w)-erf(np.sqrt(2)*(base+lx)/w)
        for ly,hy in zip(ey[:-1],ey[1:]):
            ym=base+(ly+hy)/2
            wy=erf(np.sqrt(2)*(base+hy)/w)-erf(np.sqrt(2)*(base+ly)/w)
            X,Y=np.meshgrid(xm,ym)
            d=panel_phase(X,Y,charge,rho1,params,carrier)+panel_phase(X,Y,0,rho2,params,carrier)
            d-=panel_phase(X,Y,charge,(0,0),params,carrier)+panel_phase(X,Y,0,(0,0),params,carrier)
            weight=wy[:,None]*wx[None,:]
            weight *= (abs(X)<params.active_width_m/2)&(abs(Y)<params.active_height_m/2)
            inner+=np.sum(weight*np.exp(1j*d)); total+=weight.sum()
    return float(np.clip(abs(inner/total)**2,0,1))
