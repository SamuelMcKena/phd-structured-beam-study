"""Production-grid validation of the best q=20 SLM2 model candidate.

The inverse is fitted on a tractable grid, but the measured axicon phase period
is under-resolved at N=2048.  This script reconstructs the v2 SLM2 candidate
from its compact coefficient record and validates it at N=4096 on the same
10-mm physical window.  It reports the underlying optical field separately
from the predicted 5.5-um BeamGage response.
"""
from __future__ import annotations
import gc,json,sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
ROOT=Path(__file__).resolve().parents[1]; EXP=ROOT/"notebooks"/"experimental"/"axicon_aberration_correction"; TOOLS=ROOT/"tools"
for p in (ROOT,EXP,TOOLS):
    if str(p) not in sys.path: sys.path.insert(0,str(p))
from real_bmg_digital_twin_correction import AxiconError,FourFError,FIT_WINDOW_M,PIXEL_M,Q,RELAY_N,SystemErrorConfig
from optimize_q20_slm2_detector_closure_v2 import phase_basis,phase_from_coefficients
from vbb_study.digital_twin.detector_response import sample_camera_response
from vbb_study.digital_twin.residual_phase_fit import angular_phase_from_coefficients
from vbb_study.digital_twin.vortex_continuous_propagation import build_fixed_support_spectrum,native_field_at_z
from vbb_study.digital_twin.vortex_system_route import build_multirate_system_route
from vbb_study.digital_twin.phase2a_contracts import canonical_hardware_manifest,hardware_value
from vbb_study.viz_fields import phase_winding
EPS=np.finfo(float).tiny; N=4096; AXIS_UM=np.linspace(-180.,180.,241); THERMAL="inferno"

def norm(a):
    a=np.maximum(np.asarray(a,float),0); return a/max(float(a.max()),EPS)

def native_crop(field,x):
    I=np.abs(np.asarray(field))**2; x=np.asarray(x,float); pix=np.interp(AXIS_UM*1e-6,x,np.arange(len(x))); yy,xx=np.meshgrid(pix,pix,indexing="ij")
    return norm(ndimage.map_coordinates(I,[yy,xx],order=1,mode="constant",cval=0))

def detector_crop(field,x):
    shown,_=sample_camera_response((np.abs(np.asarray(field))**2)[None],np.asarray(x,float),AXIS_UM*1e-6,pixel_pitch_m=PIXEL_M,quadrature_n=3)
    return norm(shown[0])

def metrics(a,b):
    X,Y=np.meshgrid(AXIS_UM,AXIS_UM,indexing="xy"); roi=np.hypot(X,Y)<=145; av=a[roi]; bv=b[roi]
    return float(np.corrcoef(av,bv)[0,1]),float(np.sqrt(np.mean((av-bv)**2)))

def radial(im,dr=1.5,rmax=150):
    X,Y=np.meshgrid(AXIS_UM,AXIS_UM,indexing="xy"); R=np.hypot(X,Y); edges=np.arange(0,rmax+dr,dr); ids=np.digitize(R.ravel(),edges)-1; good=(ids>=0)&(ids<len(edges)-1)
    s=np.bincount(ids[good],weights=im.ravel()[good],minlength=len(edges)-1); n=np.bincount(ids[good],minlength=len(edges)-1); return .5*(edges[:-1]+edges[1:]),s/np.maximum(n,1)

def mirror_metrics(im):
    X,Y=np.meshgrid(AXIS_UM,AXIS_UM,indexing="xy"); m=(np.hypot(X,Y)>=20)&(np.hypot(X,Y)<=140)
    return float(np.sqrt(np.mean((im[m]-im[:,::-1][m])**2))),float(np.sqrt(np.mean((im[m]-im[::-1,:][m])**2)))

def main():
    out=ROOT/"outputs"/"validation"/"q20_slm2_4096_production"; out.mkdir(parents=True,exist_ok=True)
    src=EXP/"outputs"/"digital_twin_correction"; residual=json.loads((EXP/"candidates"/"q20_detector_aware_axicon_residual_candidate.json").read_text()); seed=json.loads((EXP/"candidates"/"q20_detector_domain_slm2_v2_candidate.json").read_text())
    rs=json.loads((src/"run_summary.json").read_text()); scale=float(rs["effective_axicon"]["effective_scale_relative_to_repository_2deg_assumption"]); zscan=pd.read_csv(src/"full_route_z_registration_scan.csv"); z0=float(zscan.loc[zscan.selected.astype(bool),"value"].iloc[0]); zrel=np.arange(-17.,1.); zabs=(z0+zrel)*1e-3
    cfg=SystemErrorConfig(fourf=FourFError(iris_radius_scale=1.),axicon=AxiconError(base_angle_scale=scale)); wavelength=float(hardware_value(canonical_hardware_manifest(),"wavelength_m"))
    # Relay-grid SLM2 phase from compact v2 coefficients.
    relay_only=build_multirate_system_route(f"V{Q}",relay_grid_n=RELAY_N,propagation_grid_n=RELAY_N,window_m=FIT_WINDOW_M,config=cfg)
    basis,names=phase_basis(relay_only["relay_route"]["grid"]); lookup=dict(zip(seed["basis_names"],seed["coefficients_rad"])); coeff=np.asarray([float(lookup.get(n,0.)) for n in names]); slm2=phase_from_coefficients(basis,coeff); del relay_only,basis; gc.collect()
    # Fine-grid positive residual diagnosed from BMG.
    nom=build_multirate_system_route(f"V{Q}",relay_grid_n=RELAY_N,propagation_grid_n=N,window_m=FIT_WINDOW_M,config=cfg)
    x=np.asarray(nom["grid"]["x"],float); X,Y=np.meshgrid(x,x,indexing="xy"); theta=np.arctan2(Y,X); err=angular_phase_from_coefficients(theta,np.asarray(residual["coefficients_rad"],float),modes=tuple(residual["angular_modes"]))
    cor=build_multirate_system_route(f"V{Q}",relay_grid_n=RELAY_N,propagation_grid_n=N,window_m=FIT_WINDOW_M,config=cfg,slm2_static_phase_map_rad=slm2,axicon_input_phase_map_rad=err)
    pnom=build_fixed_support_spectrum(nom["post_axicon"],nom["grid"],wavelength_m=wavelength,z_max_m=max(abs(zabs)),minimum_retained_spectral_power=.99); pcorr=build_fixed_support_spectrum(cor["post_axicon"],cor["grid"],wavelength_m=wavelength,z_max_m=max(abs(zabs)),minimum_retained_spectral_power=.99)
    optical_nom=[]; optical_corr=[]; det_nom=[]; det_corr=[]; rows=[]
    for i,zz in enumerate(zabs):
        fn=native_field_at_z(pnom,float(zz)); fc=native_field_at_z(pcorr,float(zz)); on=native_crop(fn,x); oc=native_crop(fc,x); dn=detector_crop(fn,x); dc=detector_crop(fc,x); optical_nom.append(on); optical_corr.append(oc); det_nom.append(dn); det_corr.append(dc)
        ro,eo=metrics(oc,on); rd,ed=metrics(dc,dn); mx,my=mirror_metrics(oc); rows.append(dict(z_relative_mm=zrel[i],optical_r=ro,optical_nrmse=eo,detector_r=rd,detector_nrmse=ed,optical_xmirror_rmse=mx,optical_ymirror_rmse=my)); del fn,fc
    optical_nom=np.stack(optical_nom); optical_corr=np.stack(optical_corr); det_nom=np.stack(det_nom); det_corr=np.stack(det_corr); df=pd.DataFrame(rows); df.to_csv(out/"q20_slm2_4096_metrics_vs_z.csv",index=False)
    # Topological test on converged field.
    wnom={}; wcorr={}
    for rmm in (1.0,1.1,1.2,1.3,1.4,1.5):
        key=f"radius_{rmm:.1f}_mm"; wnom[key]=float(phase_winding(nom["post_axicon"],nom["grid"],rmm*1e-3,n_phi=720)); wcorr[key]=float(phase_winding(cor["post_axicon"],cor["grid"],rmm*1e-3,n_phi=720))
    summary=dict(grid_n=N,dx_um=FIT_WINDOW_M/N*1e6,samples_per_axicon_period=float(nom["metadata"]["samples_per_axicon_radial_phase_period"]),mean_optical_r=float(df.optical_r.mean()),mean_optical_nrmse=float(df.optical_nrmse.mean()),mean_detector_r=float(df.detector_r.mean()),mean_detector_nrmse=float(df.detector_nrmse.mean()),mean_optical_xmirror_rmse=float(df.optical_xmirror_rmse.mean()),mean_optical_ymirror_rmse=float(df.optical_ymirror_rmse.mean()),winding_nominal=wnom,winding_corrected=wcorr)
    (out/"q20_slm2_4096_summary.json").write_text(json.dumps(summary,indent=2))
    np.savez_compressed(out/"q20_slm2_4096_display_arrays.npz",axis_um=AXIS_UM,z_relative_mm=zrel,optical_nominal=optical_nom.astype(np.float32),optical_corrected=optical_corr.astype(np.float32),detector_nominal=det_nom.astype(np.float32),detector_corrected=det_corr.astype(np.float32))
    rep=int(np.argmin(abs(zrel+10))); ext=[AXIS_UM[0],AXIS_UM[-1],AXIS_UM[0],AXIS_UM[-1]]; fig,axs=plt.subplots(2,4,figsize=(17,8.5),constrained_layout=True)
    stacks=((optical_nom,"nominal optical field"),(optical_corr,"SLM2-corrected optical field"),(det_nom,"nominal after 5.5 um detector"),(det_corr,"corrected predicted BeamGage"))
    for j,(st,title) in enumerate(stacks):
        axs[0,j].imshow(st[rep],origin="lower",extent=ext,cmap=THERMAL,vmin=0,vmax=1); axs[0,j].set(title=title,xlabel="x (um)",ylabel="y (um)",aspect="equal")
        rr,pn=radial(optical_nom[rep] if j<2 else det_nom[rep]); _,pp=radial(st[rep]); axs[1,j].plot(rr,pn,lw=1.6,label="nominal"); axs[1,j].plot(rr,pp,"--",lw=1.4,label=title); axs[1,j].set(xlim=(0,140),xlabel="radius (um)",ylabel="azimuthal mean intensity"); axs[1,j].grid(alpha=.2); axs[1,j].legend(fontsize=7)
    fig.suptitle("q=20 production validation at N=4096: optical rings versus predicted camera appearance"); fig.savefig(out/"24_q20_slm2_4096_optical_and_detector.png",dpi=600,bbox_inches="tight"); fig.savefig(out/"24_q20_slm2_4096_optical_and_detector.pdf",bbox_inches="tight"); plt.close(fig)
    print(json.dumps(summary,indent=2))
if __name__=="__main__": main()
