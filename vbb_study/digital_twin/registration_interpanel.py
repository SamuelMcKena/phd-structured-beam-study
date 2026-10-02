"""Conditional free-space transfer between two independently registered SLMs.

The 4F aperture is downstream of BOTH panels. The intermediate frequency
window is numerical, not a physical iris. The fixed nominal carrier ray is
the coordinate origin, never a measured per-state centroid.
"""
from dataclasses import dataclass
import math
import numpy as np
from scipy import fft
from scipy.special import erf
from .registration_reference import (ReferenceParameters,panel_phase,periodic_rho,
                                    integrated_spectrum,selected_coefficients)


@dataclass(frozen=True)
class InterpanelParameters:
    distance_m: float=.2
    source_window_m: float=.04
    panel2_window_m: float=.016
    numerical_bandwidth_cpm: float=60000.
    source_quadrature: int=2
    panel2_quadrature: int=2

    def validate(self,p):
        p.validate()
        if self.distance_m<0 or min(self.source_window_m,self.panel2_window_m,self.numerical_bandwidth_cpm)<=0:
            raise ValueError('invalid distance or window')
        if min(self.source_quadrature,self.panel2_quadrature)<1:
            raise ValueError('invalid quadrature')
        for W in [self.source_window_m,self.panel2_window_m]:
            n=W/p.pitch_m
            if abs(n/20-round(n/20))>1e-8:
                raise ValueError('windows must contain whole blaze periods')
        if self.panel2_window_m<max(p.active_width_m,p.active_height_m):
            raise ValueError('window must include physical aperture')


def fourier_line(values,f0,df,x0,dx,m,chunk=64):
    """Uniform Fourier sum by chunked Bluestein; fractional origins are exact."""
    a=np.asarray(values,dtype=np.complex64);n=a.shape[-1]
    k=np.arange(n,dtype=float);j=np.arange(m,dtype=float);alpha=df*dx
    length=fft.next_fast_len(n+m-1)
    pre=np.exp(2j*np.pi*k*df*x0+1j*np.pi*alpha*k*k).astype(np.complex64)
    t=np.arange(-(n-1),m,dtype=float)
    kernel=fft.fft(np.exp(-1j*np.pi*alpha*t*t).astype(np.complex64),length)
    post=np.exp(1j*np.pi*alpha*j*j+2j*np.pi*f0*(x0+j*dx)).astype(np.complex64)
    flat=a.reshape(-1,n);out=np.empty((len(flat),m),np.complex64)
    for start in range(0,len(flat),chunk):
        conv=fft.ifft(fft.fft(flat[start:start+chunk]*pre,length,axis=-1,workers=2)*kernel,axis=-1,workers=2)
        out[start:start+chunk]=conv[:,n-1:n-1+m]*post
    return out.reshape(a.shape[:-1]+(m,))


def nominal_walkoff(p,z):
    sine=p.wavelength_m*p.carrier_cpm
    if abs(sine)>=1:raise ValueError('nonpropagating carrier')
    return z*sine/math.sqrt(1-sine*sine)


def incident_amplitude(x,y,w):
    return np.exp(-(np.asarray(x)**2+np.asarray(y)**2)/w**2)


def input_coefficients(L,w_px,rho,p,t):
    t.validate(p);W=t.source_window_m;pitch=p.pitch_m;n=int(round(W/pitch))
    ni=min(n,int(max(240,20*math.ceil(6*w_px/20))));r=periodic_rho(rho);q=t.source_quadrature
    h=int(math.ceil(t.numerical_bandwidth_cpm*W));nu=np.arange(-h,h+1)/W
    centre=int(round(p.carrier_cpm*W));ix=(np.arange(-h,h+1)+centre+n//2)%n
    iy=(np.arange(-h,h+1)+n//2)%n;fx=nu+p.carrier_cpm
    base=(np.arange(ni)-ni/2)*pitch;C=np.zeros((len(nu),len(nu)),np.complex64);d=pitch/q
    for sx in range(q):
        ox=(sx+.5)*d-r[0]*pitch;x=base+ox
        for sy in range(q):
            oy=(sy+.5)*d-r[1]*pitch;y=base+oy
            X,Y=np.meshgrid(x,y);A=incident_amplitude(X,Y,w_px*pitch)
            A*=((abs(X+r[0]*pitch)<=p.active_width_m/2)&(abs(Y+r[1]*pitch)<=p.active_height_m/2))
            sample=(math.sqrt(p.fill_factor)*A*np.exp(1j*panel_phase(X,Y,L,r,p))).astype(np.complex64)
            padded=np.zeros((n,n),np.complex64);lo=(n-ni)//2;padded[lo:lo+ni,lo:lo+ni]=sample
            sf=fft.fftshift(fft.fft2(fft.ifftshift(padded),workers=2))
            kx=(d*np.sinc(fx*d)*np.exp(-2j*np.pi*fx*ox)).astype(np.complex64)
            ky=(d*np.sinc(nu*d)*np.exp(-2j*np.pi*nu*oy)).astype(np.complex64)
            C+=sf[np.ix_(iy,ix)]*ky[:,None]*kx[None,:]/W**2
            del padded,sf
    C*=nu[:,None]**2+nu[None,:]**2<=t.numerical_bandwidth_cpm**2
    w=w_px*pitch
    def integral(half,delta):
        return math.sqrt(math.pi)*w/(2*math.sqrt(2))*(erf(math.sqrt(2)*(half-delta)/w)-erf(math.sqrt(2)*(-half-delta)/w))
    power=p.fill_factor*integral(p.active_width_m/2,r[0]*pitch)*integral(p.active_height_m/2,r[1]*pitch)
    return C,nu,power


def transfer_coefficients(C,nu,p,t):
    lam=p.wavelength_m;fx=nu+p.carrier_cpm
    s2=(lam*fx[None,:])**2+(lam*nu[:,None])**2
    if np.any(s2>=1):raise ValueError('evanescent window')
    k=2*np.pi/lam;kz=k*np.sqrt(1-s2);kz0=k*math.sqrt(1-(lam*p.carrier_cpm)**2)
    walk=nominal_walkoff(p,t.distance_m)
    H=np.exp(1j*(t.distance_m*(kz-kz0)+2*np.pi*nu[None,:]*walk)).astype(np.complex64)
    P=C*H;drift=float(abs(np.sum(abs(P)**2,dtype=float)/np.sum(abs(C)**2,dtype=float)-1))
    return P,dict(propagation_power_drift_fraction=drift,nominal_walkoff_m=walk)


def panel2_plane(P,nu,L,rho,p,t):
    r=periodic_rho(rho);W=t.panel2_window_m;n=int(round(W/p.pitch_m))*t.panel2_quadrature;d=W/n
    x=(np.arange(n)-n/2+.5)*d-r[0]*p.pitch_m;y=(np.arange(n)-n/2+.5)*d-r[1]*p.pitch_m
    df=float(nu[1]-nu[0]);tmp=fourier_line(P,float(nu[0]),df,float(x[0]),d,n)
    incoming=fourier_line(tmp.T,float(nu[0]),df,float(y[0]),d,n).T;del tmp
    aperture=((abs(x[None,:]+r[0]*p.pitch_m)<=p.active_width_m/2)&(abs(y[:,None]+r[1]*p.pitch_m)<=p.active_height_m/2))
    I=abs(incoming)**2*aperture;total=float(I.sum(dtype=float));power=total*d*d
    centroid=dict(slm2_incident_centroid_x_m=float(np.sum(I.sum(axis=0,dtype=float)*x)/total),
                  slm2_incident_centroid_y_m=float(np.sum(I.sum(axis=1,dtype=float)*y)/total))
    del I
    phase=panel_phase(x[None,:],y[:,None],L,r,p);command=np.exp(1j*phase).astype(np.complex64);del phase
    outgoing=incoming*command;outgoing*=math.sqrt(p.fill_factor);outgoing*=aperture
    return incoming,outgoing,x,y,power,centroid


def select_panel2(E,rho,p,t):
    W=t.panel2_window_m;n=E.shape[0];d=W/n;h=int(math.ceil(p.filter_radius_cpm*W))
    centre=int(round(p.carrier_cpm*W))
    if n//2+centre+h>=n or n//2-h<0:raise ValueError('iris exceeds Nyquist')
    s=fft.fftshift(fft.fft2(fft.ifftshift(E),workers=2))*d*d
    f=np.arange(-h,h+1)/W;r=periodic_rho(rho)
    C=s[n//2-h:n//2+h+1,n//2+centre-h:n//2+centre+h+1]/W**2
    C*=np.exp(-2j*np.pi*(f+p.carrier_cpm)*(d/2-r[0]*p.pitch_m))[None,:]
    C*=np.exp(-2j*np.pi*f*(d/2-r[1]*p.pitch_m))[:,None]
    C*=f[:,None]**2+f[None,:]**2<=p.filter_radius_cpm**2
    C*=np.sinc((f+p.carrier_cpm)*d)[None,:]*np.sinc(f*d)[:,None]
    return C.astype(np.complex64)


def interpanel_route(L,w_px,rho1=(0,0),rho2=(0,0),*,architecture='upstream_vortex',
                     hardware=ReferenceParameters(),transfer=InterpanelParameters(),source_cache=None,retain_planes=False):
    p=hardware;t=transfer;t.validate(p)
    if architecture not in ['upstream_vortex','downstream_vortex']:raise ValueError('unknown architecture')
    if t.distance_m==0:
        s,f,W=integrated_spectrum(L,w_px,rho1,rho2,architecture=architecture,params=p,quadrature=t.source_quadrature)
        return dict(coefficients=selected_coefficients(s,f,W,params=p),window_m=W,
                    metadata=dict(route='exact identity-transfer control',distance_m=0,propagation_power_drift_fraction=0))
    l1,l2=(L,0) if architecture=='upstream_vortex' else (0,L)
    key=(l1,w_px,periodic_rho(rho1),p,t)
    if source_cache is not None and key in source_cache:P,nu,pin,meta=source_cache[key]
    else:
        C,nu,pin=input_coefficients(l1,w_px,rho1,p,t);band=float(np.sum(abs(C)**2,dtype=float)*t.source_window_m**2)
        P,meta=transfer_coefficients(C,nu,p,t);meta.update(band_power=band,source_numerical_band_retained_fraction=float(band/pin))
        if source_cache is not None:
            source_cache.clear();source_cache[key]=(P,nu,pin,meta)
    incoming,outgoing,x,y,intercepted,centroid=panel2_plane(P,nu,l2,rho2,p,t)
    C=select_panel2(outgoing,rho2,p,t)
    meta=dict(meta,**centroid,distance_m=t.distance_m,physical_interpanel_iris=False,
        numerical_bandwidth_cpm=t.numerical_bandwidth_cpm,selected_power=float(np.sum(abs(C)**2,dtype=float)*t.panel2_window_m**2),
        slm2_intercepted_fraction_of_band_power=intercepted/meta['band_power'],incident_power_after_slm1=float(pin),
        imposed_nominal_axis_decentre_m=[0,0],coordinate_frame='fixed nominal +1 ray; not per-state centroid',
        slm1_role='vortex + blaze' if l1 else 'flat correction + blaze',slm2_role='vortex + blaze' if l2 else 'flat correction + blaze')
    result=dict(coefficients=C,window_m=t.panel2_window_m,metadata=meta)
    if retain_planes:
        step=t.panel2_quadrature;sx=np.flatnonzero(abs(x[::step])<.003)*step;sy=np.flatnonzero(abs(y[::step])<.003)*step
        result.update(incoming_slm2=incoming[np.ix_(sy,sx)],after_slm2=outgoing[np.ix_(sy,sx)],slm2_x=x[sx],slm2_y=y[sy])
    return result
