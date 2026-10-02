from dataclasses import replace
import numpy as np
import pytest
from vbb_study.digital_twin.registration_interpanel import *
from vbb_study.digital_twin.registration_reference import spectrum_fidelity

@pytest.mark.parametrize('x0',[-.001003,.000007])
def test_fractional_fourier_sum(x0):
    rng=np.random.default_rng(12);a=rng.normal(size=(4,13))+1j*rng.normal(size=(4,13))
    f0,df,dx,m=-600,31.25,4e-6,27
    direct=a@np.exp(2j*np.pi*(f0+np.arange(13)*df)[:,None]*(x0+np.arange(m)*dx)[None,:])
    np.testing.assert_allclose(fourier_line(a,f0,df,x0,dx,m),direct,atol=3e-6,rtol=3e-6)

def test_unitary_and_nominal_walkoff():
    p=ReferenceParameters();t=InterpanelParameters();f=np.arange(-10,11)*100
    _,m=transfer_coefficients(np.ones((21,21)),f,p,t)
    assert m['propagation_power_drift_fraction']<1e-6
    assert nominal_walkoff(p,.2)==pytest.approx(.0012862766011032404)
    a,_=transfer_coefficients(np.ones((3,3)),np.array([-1.,0,1]),p,t)
    assert np.angle(a[1,2]/a[1,0])==pytest.approx(0,abs=1e-8)

def test_zero_distance_role_exchange():
    t=replace(InterpanelParameters(),distance_m=0)
    a=interpanel_route(10,25,(.25,.5),transfer=t)
    b=interpanel_route(10,25,rho2=(1.25,-.5),architecture='downstream_vortex',transfer=t)
    np.testing.assert_allclose(a['coefficients'],b['coefficients'],atol=1e-17)

def cheap():
    return InterpanelParameters(source_window_m=.016,numerical_bandwidth_cpm=4000,source_quadrature=1,panel2_quadrature=1)

def test_periodic_centring_and_both_blazes():
    a=interpanel_route(1,50,(.25,.5),(.5,.125),transfer=cheap())
    b=interpanel_route(1,50,(1.25,-.5),(-.5,1.125),transfer=cheap())
    np.testing.assert_array_equal(a['coefficients'],b['coefficients'])
    assert a['metadata']['imposed_nominal_axis_decentre_m']==[0,0]
    assert not a['metadata']['physical_interpanel_iris']
    for panel in [1,2]:assert 'blaze' in a['metadata'][f'slm{panel}_role']

def test_finite_transfer_does_not_commute():
    a=interpanel_route(1,50,transfer=cheap())['coefficients']
    b=interpanel_route(1,50,architecture='downstream_vortex',transfer=cheap())['coefficients']
    assert spectrum_fidelity(a,b)<.999

def test_artificial_no_carrier_square_aperture_rotation():
    p=replace(ReferenceParameters(),carrier_cpm=0,active_width_m=.00864)
    a=interpanel_route(3,50,(.5,0),hardware=p,transfer=cheap())['coefficients']
    b=interpanel_route(3,50,(0,.5),hardware=p,transfer=cheap())['coefficients']
    assert spectrum_fidelity(np.rot90(a),b)>.999999

def test_overlap_accumulates_in_double_precision():
    a=np.ones((100,100),np.complex64);b=a.copy();b[:,50:]*=np.exp(.001j)
    assert 1-spectrum_fidelity(a,b)==pytest.approx(np.sin(.0005)**2,rel=1e-6)

def test_beam_does_not_follow_lattice():
    x=np.arange(-100,101)*1e-6
    for rho in [0,.125,.25,.375,.5]:
        g=-rho*8e-6;physical_x=(x-g)+g
        assert physical_x[np.argmax(incident_amplitude(physical_x,0,400e-6))]==pytest.approx(0,abs=1e-20)

def test_bad_aperture_window_rejected():
    with pytest.raises(ValueError,match='physical aperture'):
        replace(InterpanelParameters(),panel2_window_m=.008).validate(ReferenceParameters())
