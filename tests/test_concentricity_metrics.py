import numpy as np

from vbb_study.digital_twin.concentricity_metrics import ring_metrics


def _rings(axis):
    X,Y=np.meshgrid(axis,axis,indexing='xy'); R=np.hypot(X,Y); th=np.arctan2(Y,X)
    base=np.exp(-0.5*((R-45.0)/2.2)**2)+0.35*np.exp(-0.5*((R-58.0)/1.8)**2)+0.20*np.exp(-0.5*((R-69.0)/1.7)**2)
    return base,th


def test_circular_field_scores_better_than_cross_modulated_field():
    axis=np.linspace(-120.0,120.0,241)
    target,th=_rings(axis)
    cross=target*(1.0+0.45*np.cos(4.0*th))
    good=ring_metrics(target,target,axis)
    bad=ring_metrics(cross,target,axis)
    assert good['mean_ring_intensity_cv'] < 1e-3
    assert bad['mean_ring_intensity_cv'] > good['mean_ring_intensity_cv'] + 0.1
    assert bad['mean_angular_harmonic_energy'] > good['mean_angular_harmonic_energy'] + 0.1


def test_radially_warped_ring_reports_radius_variation():
    axis=np.linspace(-120.0,120.0,241)
    target,th=_rings(axis)
    X,Y=np.meshgrid(axis,axis,indexing='xy'); R=np.hypot(X,Y)
    warped_r=R-2.0*np.cos(2.0*th)
    warped=np.exp(-0.5*((warped_r-45.0)/2.2)**2)+0.35*np.exp(-0.5*((warped_r-58.0)/1.8)**2)+0.20*np.exp(-0.5*((warped_r-69.0)/1.7)**2)
    good=ring_metrics(target,target,axis)
    bad=ring_metrics(warped,target,axis)
    assert bad['mean_ring_radius_std_um'] > good['mean_ring_radius_std_um'] + 0.5
