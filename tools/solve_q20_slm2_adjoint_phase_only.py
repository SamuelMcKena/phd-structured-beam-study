"""Adjoint phase-only SLM2 solve for the real-data-supported q=20 residual.

The detector-aware real-BMG inverse recovered a low-order residual phase at the
selected-order field immediately before the axicon.  Low-dimensional SLM2
bases reproduced the detector-domain target well, but finite-iris amplitude
coupling left residual optical asymmetry and could perturb an inner q=20 phase
loop.  This solver removes that artificial basis restriction.

We optimise one phase value per computational SLM2 sample while keeping the
incident SLM2 amplitude fixed.  Every iteration propagates through the exact
carrier + explicit 4F + fixed +1 iris model.  The gradient is obtained with the
adjoint of that same relay.  The complex target at the axicon input is the
nominal selected-order field multiplied by the conjugate of the held-out-
supported recovered residual.  The final phase is validated through the full
4096-sample axicon/propagation route and the measured 5.5 um detector model.

This remains a model-space precompensation candidate: native SLM2 coordinate
registration and the 1030-nm LUT are not independently calibrated.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "notebooks" / "experimental" / "axicon_aberration_correction"
TOOLS = ROOT / "tools"
for p in (ROOT, EXP, TOOLS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from real_bmg_digital_twin_correction import AxiconError, FourFError, FIT_WINDOW_M, PIXEL_M, Q, SystemErrorConfig
from optimize_q20_slm2_detector_closure_v2 import phase_basis, phase_from_coefficients
from vbb_study.digital_twin.detector_response import sample_camera_response
from vbb_study.digital_twin.phase2a_contracts import canonical_hardware_manifest, hardware_value
from vbb_study.digital_twin.residual_phase_fit import angular_phase_from_coefficients
from vbb_study.digital_twin.vortex_continuous_propagation import build_fixed_support_spectrum, native_field_at_z
from vbb_study.digital_twin.vortex_explicit_4f import _propagate, explicit_4f_relay, thin_lens_transmission
from vbb_study.digital_twin.vortex_system_route import build_multirate_system_route, build_system_route
from vbb_study.viz_fields import phase_winding

EPS = np.finfo(float).tiny
TWOPI = 2.0*np.pi
THERMAL = "inferno"
RELAY_N = 1024
PROD_N = 4096
DISPLAY_AXIS_UM = np.linspace(-180.0, 180.0, 241)
MAX_ITER = 14


def manifest_values():
    m = canonical_hardware_manifest()
    return {
        "wavelength": float(hardware_value(m, "wavelength_m")),
        "carrier": float(hardware_value(m, "carrier_frequency_cpm")),
        "f": float(hardware_value(m, "fourf_focal_length_m")),
        "iris_radius": float(hardware_value(m, "fourier_iris_radius_m")),
    }


def forward_selected(u_slm2: np.ndarray, grid: dict, vals: dict, config: SystemErrorConfig):
    relay = explicit_4f_relay(
        u_slm2, grid,
        wavelength_m=vals["wavelength"], nominal_focal_length_m=vals["f"],
        nominal_iris_radius_m=vals["iris_radius"], nominal_carrier_cpm=vals["carrier"],
        error=config.fourf,
    )
    X = np.asarray(grid["X"], float)
    selected = np.asarray(relay["output"], complex)*np.exp(1j*TWOPI*vals["carrier"]*X)
    return selected, relay


def adjoint_selected(residual_selected: np.ndarray, grid: dict, vals: dict, iris_mask: np.ndarray):
    """Adjoint of nominal parallel-plane 4F + selected-order carrier removal."""
    X = np.asarray(grid["X"], float)
    r = np.asarray(residual_selected, complex)*np.exp(-1j*TWOPI*vals["carrier"]*X)
    f = vals["f"]; wl = vals["wavelength"]
    # A = P_f L2 P_f M_iris P_f L1 P_f; use A^H in reverse order.
    r = _propagate(r, grid, wl, -f)
    L2, _ = thin_lens_transmission(grid, wavelength_m=wl, focal_length_m=f)
    r = r*np.conj(L2)
    r = _propagate(r, grid, wl, -f)
    r = r*np.asarray(iris_mask, float)
    r = _propagate(r, grid, wl, -f)
    L1, _ = thin_lens_transmission(grid, wavelength_m=wl, focal_length_m=f)
    r = r*np.conj(L1)
    r = _propagate(r, grid, wl, -f)
    return np.asarray(r, complex)


def weighted_objective(y: np.ndarray, target: np.ndarray, weight: np.ndarray):
    r = np.asarray(y)-np.asarray(target)
    den = max(float(np.sum(weight*np.abs(target)**2)), EPS)
    return float(np.sum(weight*np.abs(r)**2)/den)


def overlap(a: np.ndarray, b: np.ndarray, w: np.ndarray):
    num = abs(np.sum(w*np.conj(a)*b))
    den = np.sqrt(np.sum(w*np.abs(a)**2)*np.sum(w*np.abs(b)**2))
    return float(num/max(float(den), EPS))


def wrapped_smooth_phase(phi: np.ndarray, sigma: float = 0.42):
    z = ndimage.gaussian_filter(np.exp(1j*np.asarray(phi,float)).real, sigma=sigma, mode="nearest") + 1j*ndimage.gaussian_filter(np.exp(1j*np.asarray(phi,float)).imag, sigma=sigma, mode="nearest")
    return np.angle(z)


def relay_solve(config: SystemErrorConfig, residual: dict, seed: dict):
    vals = manifest_values()
    base = build_system_route(f"V{Q}", grid_n=RELAY_N, config=config, window_m=FIT_WINDOW_M)
    grid = base["grid"]
    u0 = np.asarray(base["post_slm2"], complex)
    nominal = np.asarray(base["post_4f_selected_order"], complex)
    X = np.asarray(grid["X"], float); Y = np.asarray(grid["Y"], float)
    theta = np.arctan2(Y, X)
    err = angular_phase_from_coefficients(theta, np.asarray(residual["coefficients_rad"],float), modes=tuple(residual["angular_modes"]))
    target = nominal*np.exp(-1j*err)

    # Start from the best compact v2 phase; the adjoint solve is free to leave it.
    basis, names = phase_basis(grid)
    lookup = dict(zip(seed["basis_names"], seed["coefficients_rad"]))
    coeff = np.asarray([float(lookup.get(n,0.0)) for n in names])
    phi = phase_from_coefficients(basis, coeff)
    del basis

    amp2 = np.abs(target)**2
    amp2n = amp2/max(float(amp2.max()), EPS)
    R = np.hypot(X, Y)
    support = (R <= 2.7e-3) & (np.abs(u0) >= 2e-3*np.max(np.abs(u0)))
    weight = np.where(support, 0.04 + 0.96*amp2n, 0.0)

    # Iris is independent of phase, so obtain it once.
    _, relay0 = forward_selected(u0, grid, vals, config)
    iris = np.asarray(relay0["iris_mask"], float)
    history=[]
    for it in range(MAX_ITER):
        u = u0*np.exp(1j*phi)
        y, _ = forward_selected(u, grid, vals, config)
        obj = weighted_objective(y,target,weight)
        ov = overlap(y,target,weight)
        res = weight*(y-target)
        g_u = adjoint_selected(res, grid, vals, iris)
        grad = -2.0*np.imag(np.conj(g_u)*u)
        grad = np.where(support, grad, 0.0)
        # Smooth only the search direction; the phase itself remains a full actuator map.
        grad = ndimage.gaussian_filter(grad, sigma=0.55, mode="nearest")
        rms = float(np.sqrt(np.mean(grad[support]**2)))
        if not np.isfinite(rms) or rms < 1e-14:
            history.append({"iteration":it,"objective":obj,"overlap":ov,"stopped":"zero_gradient"}); break
        direction = -grad/rms
        chosen=None
        # The RMS phase move is controlled explicitly in radians.
        for step in (0.22,0.12,0.06,0.03):
            trial_phi = np.angle(np.exp(1j*(phi + step*direction)))
            # suppress checkerboard-scale phase without removing physically useful structure
            trial_phi = wrapped_smooth_phase(trial_phi, sigma=0.32)
            trial_u = u0*np.exp(1j*trial_phi)
            trial_y,_ = forward_selected(trial_u,grid,vals,config)
            trial_obj=weighted_objective(trial_y,target,weight)
            if chosen is None or trial_obj < chosen[0]: chosen=(trial_obj,trial_phi,trial_y,step)
        if chosen[0] >= obj*(1.0-2e-5):
            history.append({"iteration":it+1,"objective":obj,"overlap":ov,"accepted":False}); break
        phi=chosen[1]
        history.append({"iteration":it+1,"objective_before":obj,"objective_after":chosen[0],"overlap_before":ov,"overlap_after":overlap(chosen[2],target,weight),"step_rms_rad":chosen[3]})
        del u,y,res,g_u,grad,direction; gc.collect()

    solved_u=u0*np.exp(1j*phi); solved,_=forward_selected(solved_u,grid,vals,config)
    closed=solved*np.exp(1j*err)
    # Remove one global piston for phase-RMS reporting only.
    phase_delta=np.angle(closed*np.conj(nominal)); piston=np.angle(np.sum(weight*np.exp(1j*phase_delta)))
    phase_delta=np.angle(np.exp(1j*(phase_delta-piston)))
    amp_ratio=np.abs(closed)/np.maximum(np.abs(nominal),EPS)
    relay_summary={
        "relay_grid_n":RELAY_N,
        "initial_v2_objective":None if not history else history[0].get("objective_before",history[0].get("objective")),
        "final_objective":weighted_objective(solved,target,weight),
        "target_field_overlap":overlap(solved,target,weight),
        "closure_field_overlap":overlap(closed,nominal,weight),
        "closure_phase_rms_rad":float(np.sqrt(np.sum(weight*phase_delta**2)/max(float(np.sum(weight)),EPS))),
        "closure_amplitude_ratio_rms_from_unity":float(np.sqrt(np.sum(weight*(amp_ratio-1.0)**2)/max(float(np.sum(weight)),EPS))),
        "iterations":history,
    }
    return phi, relay_summary


def crop_optical(field, x_m):
    I=np.abs(np.asarray(field))**2; x=np.asarray(x_m,float)
    pix=np.interp(DISPLAY_AXIS_UM*1e-6,x,np.arange(len(x))); yy,xx=np.meshgrid(pix,pix,indexing="ij")
    out=ndimage.map_coordinates(I,[yy,xx],order=1,mode="constant",cval=0.0)
    return out/max(float(out.max()),EPS)


def crop_detector(field,x_m):
    shown,_=sample_camera_response((np.abs(np.asarray(field))**2)[None],np.asarray(x_m,float),DISPLAY_AXIS_UM*1e-6,pixel_pitch_m=PIXEL_M,quadrature_n=3)
    out=shown[0]; return out/max(float(out.max()),EPS)


def image_metrics(a,b):
    X,Y=np.meshgrid(DISPLAY_AXIS_UM,DISPLAY_AXIS_UM,indexing="xy"); roi=np.hypot(X,Y)<=145
    av=np.asarray(a)[roi]; bv=np.asarray(b)[roi]
    return float(np.corrcoef(av,bv)[0,1]),float(np.sqrt(np.mean((av-bv)**2)))


def mirror_metrics(im):
    X,Y=np.meshgrid(DISPLAY_AXIS_UM,DISPLAY_AXIS_UM,indexing="xy"); R=np.hypot(X,Y); m=(R>=20)&(R<=140)
    return float(np.sqrt(np.mean((im[m]-im[:,::-1][m])**2))),float(np.sqrt(np.mean((im[m]-im[::-1,:][m])**2)))


def radial(im,dr=1.5,rmax=150):
    X,Y=np.meshgrid(DISPLAY_AXIS_UM,DISPLAY_AXIS_UM,indexing="xy"); R=np.hypot(X,Y); edges=np.arange(0,rmax+dr,dr); ids=np.digitize(R.ravel(),edges)-1; good=(ids>=0)&(ids<len(edges)-1)
    s=np.bincount(ids[good],weights=np.asarray(im).ravel()[good],minlength=len(edges)-1); n=np.bincount(ids[good],minlength=len(edges)-1)
    return .5*(edges[:-1]+edges[1:]),s/np.maximum(n,1)


def production_validate(config,residual,phi,out):
    vals=manifest_values(); src=EXP/"outputs"/"digital_twin_correction"; zscan=pd.read_csv(src/"full_route_z_registration_scan.csv"); z0=float(zscan.loc[zscan.selected.astype(bool),"value"].iloc[0]); zrel=np.arange(-17.,1.); zabs=(z0+zrel)*1e-3
    nom=build_multirate_system_route(f"V{Q}",relay_grid_n=RELAY_N,propagation_grid_n=PROD_N,window_m=FIT_WINDOW_M,config=config)
    x=np.asarray(nom["grid"]["x"],float); X,Y=np.meshgrid(x,x,indexing="xy"); theta=np.arctan2(Y,X)
    err=angular_phase_from_coefficients(theta,np.asarray(residual["coefficients_rad"],float),modes=tuple(residual["angular_modes"]))
    cor=build_multirate_system_route(f"V{Q}",relay_grid_n=RELAY_N,propagation_grid_n=PROD_N,window_m=FIT_WINDOW_M,config=config,slm2_static_phase_map_rad=phi,axicon_input_phase_map_rad=err)
    pnom=build_fixed_support_spectrum(nom["post_axicon"],nom["grid"],wavelength_m=vals["wavelength"],z_max_m=max(abs(zabs)),minimum_retained_spectral_power=.99)
    pcorr=build_fixed_support_spectrum(cor["post_axicon"],cor["grid"],wavelength_m=vals["wavelength"],z_max_m=max(abs(zabs)),minimum_retained_spectral_power=.99)
    rows=[]; on=[]; oc=[]; dn=[]; dc=[]
    for iz,zz in enumerate(zabs):
        fn=native_field_at_z(pnom,float(zz)); fc=native_field_at_z(pcorr,float(zz)); a=crop_optical(fn,x); b=crop_optical(fc,x); ad=crop_detector(fn,x); bd=crop_detector(fc,x)
        ro,eo=image_metrics(b,a); rd,ed=image_metrics(bd,ad); mx,my=mirror_metrics(b)
        rows.append(dict(z_relative_mm=zrel[iz],optical_r=ro,optical_nrmse=eo,detector_r=rd,detector_nrmse=ed,optical_xmirror_rmse=mx,optical_ymirror_rmse=my))
        on.append(a); oc.append(b); dn.append(ad); dc.append(bd); del fn,fc
    df=pd.DataFrame(rows); df.to_csv(out/"adjoint_4096_metrics_vs_z.csv",index=False)
    on=np.stack(on); oc=np.stack(oc); dn=np.stack(dn); dc=np.stack(dc)
    wnom={}; wcorr={}
    for rmm in (1.0,1.1,1.2,1.3,1.4,1.5):
        k=f"radius_{rmm:.1f}_mm"; wnom[k]=float(phase_winding(nom["post_axicon"],nom["grid"],rmm*1e-3,n_phi=720)); wcorr[k]=float(phase_winding(cor["post_axicon"],cor["grid"],rmm*1e-3,n_phi=720))
    summary={"production_grid_n":PROD_N,"dx_um":FIT_WINDOW_M/PROD_N*1e6,"samples_per_axicon_period":float(nom["metadata"]["samples_per_axicon_radial_phase_period"]),"mean_optical_r":float(df.optical_r.mean()),"mean_optical_nrmse":float(df.optical_nrmse.mean()),"mean_detector_r":float(df.detector_r.mean()),"mean_detector_nrmse":float(df.detector_nrmse.mean()),"mean_optical_xmirror_rmse":float(df.optical_xmirror_rmse.mean()),"mean_optical_ymirror_rmse":float(df.optical_ymirror_rmse.mean()),"winding_nominal":wnom,"winding_corrected":wcorr}
    np.savez_compressed(out/"adjoint_4096_display_arrays.npz",axis_um=DISPLAY_AXIS_UM,z_relative_mm=zrel,optical_nominal=on.astype(np.float32),optical_corrected=oc.astype(np.float32),detector_nominal=dn.astype(np.float32),detector_corrected=dc.astype(np.float32))
    rep=int(np.argmin(abs(zrel+10))); ext=[DISPLAY_AXIS_UM[0],DISPLAY_AXIS_UM[-1],DISPLAY_AXIS_UM[0],DISPLAY_AXIS_UM[-1]]; fig,axs=plt.subplots(2,4,figsize=(17,8.5),constrained_layout=True)
    stacks=((on,"nominal optical field"),(oc,"adjoint-corrected optical field"),(dn,"nominal predicted BeamGage"),(dc,"corrected predicted BeamGage"))
    for j,(st,title) in enumerate(stacks):
        axs[0,j].imshow(st[rep],origin="lower",extent=ext,cmap=THERMAL,vmin=0,vmax=1); axs[0,j].set(title=title,xlabel="x (um)",ylabel="y (um)",aspect="equal")
        ref=on if j<2 else dn; rr,p0=radial(ref[rep]); _,p1=radial(st[rep]); axs[1,j].plot(rr,p0,lw=1.7,label="nominal"); axs[1,j].plot(rr,p1,"--",lw=1.4,label=title); axs[1,j].set(xlim=(0,140),xlabel="radius (um)",ylabel="azimuthal mean intensity"); axs[1,j].grid(alpha=.2); axs[1,j].legend(fontsize=7)
    fig.suptitle("q=20 full phase-only SLM2 solve: 4096 optical field and predicted 5.5 um camera response"); fig.savefig(out/"25_q20_adjoint_slm2_4096_validation.png",dpi=600,bbox_inches="tight"); fig.savefig(out/"25_q20_adjoint_slm2_4096_validation.pdf",bbox_inches="tight"); plt.close(fig)
    return summary


def main():
    out=ROOT/"outputs"/"validation"/"q20_slm2_adjoint_phase_only"; out.mkdir(parents=True,exist_ok=True)
    src=EXP/"outputs"/"digital_twin_correction"; residual=json.loads((EXP/"candidates"/"q20_detector_aware_axicon_residual_candidate.json").read_text()); seed=json.loads((EXP/"candidates"/"q20_detector_domain_slm2_v2_candidate.json").read_text()); rs=json.loads((src/"run_summary.json").read_text()); scale=float(rs["effective_axicon"]["effective_scale_relative_to_repository_2deg_assumption"])
    config=SystemErrorConfig(fourf=FourFError(iris_radius_scale=1.),axicon=AxiconError(base_angle_scale=scale))
    phi,relay_summary=relay_solve(config,residual,seed); np.save(out/"model_space_slm2_phase_adjoint_rad.npy",phi.astype(np.float32)); prod=production_validate(config,residual,phi,out)
    summary={"status":"full_phase_only_adjoint_model_candidate","relay_solve":relay_summary,"production_validation":prod,"hardware_ready":False,"hardware_blockers":residual.get("hardware_blockers",[])}; (out/"q20_slm2_adjoint_summary.json").write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))
if __name__=="__main__": main()
