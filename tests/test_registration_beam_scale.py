"""Physical width checks independent of the displayed downstream crop."""
import numpy as np
import pytest
from tools.audit_registration_beam_scale import fit_gaussian_radius
from vbb_study.digital_twin.registration_interpanel import incident_amplitude
from vbb_study.digital_twin.registration_reference import ReferenceParameters, panel_phase

@pytest.mark.parametrize('w_px',[50,250])
def test_actual_input_radius_survives_phase_command_and_finite_crop(w_px):
    p=ReferenceParameters();w=w_px*p.pitch_m
    x=(np.arange(1500)-750+.5)*4e-6
    X,Y=np.meshgrid(x,x);a=incident_amplitude(X,Y,w)
    for d in [0,.125,.25,.375,.5]:
        e=a*np.exp(1j*panel_phase(X,Y,20,(d,0),p))
        assert fit_gaussian_radius(x,e)==pytest.approx(w,rel=1e-10)
        np.testing.assert_allclose(abs(e)**2,a*a,atol=1e-15)

def test_input_radius_scales_fivefold_not_downstream_bessel_ring():
    p=ReferenceParameters()
    assert (250*p.pitch_m)/(50*p.pitch_m)==pytest.approx(5)
    # The ideal Bessel transverse scale depends on cone wavevector and charge,
    # not on the Gaussian envelope width. Width changes its finite envelope.
    from scipy.special import jnp_zeros
    r=jnp_zeros(20,1)[0]/p.kr
    assert r==pytest.approx(21.13994670528523e-6)
