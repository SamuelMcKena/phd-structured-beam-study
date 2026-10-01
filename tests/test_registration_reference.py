"""Independent numerical/physical gates for the definitive registration study."""
from dataclasses import replace
import numpy as np
import pytest
from scipy import fft
from vbb_study.digital_twin.registration_reference import (
    ReferenceParameters, integrated_spectrum, selected_coefficients, synthesize,
    spectrum_fidelity, panel_phase, local_fidelity, propagate_axicon)
from vbb_study.digital_twin.slm_registration_architectures import pure_registration_error


def coeff(L=20,w=50,r1=(0,0),r2=(0,0),**kw):
    s,f,W=integrated_spectrum(L,w,r1,r2,**kw)
    return selected_coefficients(s,f,W,params=kw.get('params',ReferenceParameters()),
                                 carrier=kw.get('carrier',True)),W


def test_registration_centre_invariance_and_whole_pitch_periodicity():
    for d in [0,1e-6,2e-6,3e-6,4e-6,8e-6]:
        e=pure_registration_error(d,-d)
        assert np.array(e.panel_translation_m)+e.pattern_offset_m == pytest.approx([0,0])
    a,W=coeff(r1=(.25,.5)); b,_=coeff(r1=(1.25,-.5))
    np.testing.assert_allclose(a,b,rtol=0,atol=1e-17)


def test_zero_shift_is_direct_two_blaze_baseline():
    p=ReferenceParameters(); x,y=np.meshgrid(np.arange(10)*p.pitch_m,np.arange(10)*p.pitch_m)
    a=panel_phase(x,y,0,(0,0),p)
    assert np.ptp(a)>1  # both commands are independently evaluated, not one shared blaze
    assert 1/(p.carrier_cpm*p.pitch_m)==pytest.approx(20)
    A,_=coeff(L=0); B,_=coeff(L=0,architecture='downstream_vortex')
    np.testing.assert_array_equal(A,B)


@pytest.mark.parametrize('r1,r2',[((.25,.5),(0,0)),((0,0),(.5,.25)),((.5,.5),(.5,.5)),((.25,.25),(-.25,-.25))])
def test_A_B_equivalent_physical_roles_are_identical(r1,r2):
    a,_=coeff(r1=r1,r2=r2)
    b,_=coeff(r1=r2,r2=r1,architecture='downstream_vortex')
    np.testing.assert_allclose(a,b,atol=1e-17,rtol=1e-12)


def test_no_carrier_rotational_symmetry_and_carrier_anisotropy():
    x=1-local_fidelity(20,50,(.5,0),carrier=False)
    y=1-local_fidelity(20,50,(0,.5),carrier=False)
    assert x==pytest.approx(y,abs=1e-12)
    x=1-local_fidelity(20,50,(.5,0))
    y=1-local_fidelity(20,50,(0,.5))
    assert abs(x-y)>1e-6


def test_selected_order_window_gate():
    p=replace(ReferenceParameters(),carrier_cpm=60000)
    s,f,W=integrated_spectrum(1,25,params=p)
    with pytest.raises(ValueError,match='exceeds'):
        selected_coefficients(s,f,W,params=p)


def test_pixel_integral_matches_independent_resolved_raster():
    p=ReferenceParameters(); npix=240; W=npix*p.pitch_m
    C,_=coeff(r1=(.375,.125),quadrature=2,n_pixels=npix)
    errors=[]
    for spp in [4,8,16]:
        n=npix*spp; dx=p.pitch_m/spp
        x=(np.arange(n)-n/2+.5)*dx
        X,Y=np.meshgrid(x,x)
        phase=panel_phase(X,Y,20,(.375,.125),p)+panel_phase(X,Y,0,(0,0),p)
        E=p.fill_factor*np.exp(-(X*X+Y*Y)/(50*p.pitch_m)**2)*np.exp(1j*phase)
        S=fft.fftshift(fft.fft2(fft.ifftshift(E),workers=2))*dx**2
        f=fft.fftshift(fft.fftfreq(n,dx))
        # Correct the midpoint coordinate origin in the discrete Fourier integral.
        ph=np.exp(-2j*np.pi*f*dx/2)
        S*=ph[:,None]*ph[None,:]
        Cr=selected_coefficients(S,f,W)
        errors.append(np.linalg.norm(Cr-C)/np.linalg.norm(C))
    assert errors[-1]<.003
    assert errors[-1]<errors[1]


def test_quadrature_convergence():
    a,_=coeff(r1=(.375,.125),quadrature=1)
    b,_=coeff(r1=(.375,.125),quadrature=2)
    c,_=coeff(r1=(.375,.125),quadrature=4)
    assert np.linalg.norm(c-b)<np.linalg.norm(b-a)
    assert 1-spectrum_fidelity(c,b)<1e-6


def test_synthesis_parseval_and_phase_only_axicon():
    C,W=coeff(L=3,w=25)
    E=synthesize(C,W,256)
    assert np.mean(abs(E)**2)==pytest.approx(np.sum(abs(C)**2),rel=2e-6)
    x=(np.arange(256)-128+.5)*W/256
    ax=np.exp(-1j*ReferenceParameters().kr*np.hypot(x[:,None],x[None,:]))
    np.testing.assert_allclose(abs(E*ax)**2,abs(E)**2,rtol=2e-6,atol=1e-12)


def test_propagation_power_and_cone_sampling_gate():
    C,W=coeff(L=3,w=25)
    crops,x,rows=propagate_axicon(C,W,[.001],dx_target=1.5e-6)
    assert rows[0]['propagation_power_drift_fraction']<1e-5
    assert rows[0]['boundary_power_fraction']<.01
    assert 2*np.pi/ReferenceParameters().kr/rows[0]['dx_m']>3
