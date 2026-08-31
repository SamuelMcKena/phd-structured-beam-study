"""High-resolution q=20 radial-symmetry audit at fixed physical parameters."""
from __future__ import annotations
import gc, json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage

ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/"notebooks"/"experimental"/"axicon_aberration_correction"
for p in (ROOT,EXP):
    if str(p) not in sys.path: sys.path.insert(0,str(p))
from real_bmg_digital_twin_correction import AxiconError,FourFError,FIT_WINDOW_M,PIXEL_M,Q,RELAY_N,SystemErrorConfig
from vbb_study.digital_twin.detector_response import sample_camera_response
from vbb_study.digital_twin.vortex_continuous_propagation import build_fixed_support_spectrum,native_field_at_z
from vbb_study.digital_twin.vortex_system_route import build_multirate_system_route
from vbb_study.digital_twin.phase2a_contracts import canonical_hardware_manifest,hardware_value

EPS=np.finfo(float).tiny; THERMAL="inferno"; GRID_NS=(2048,3072,4096); REP_Z_REL_MM=-10.0
AXIS_UM=np.linspace(-180.,180.,241)

def norm2d(a):
    a=np.maximum(np.asarray(a,float),0); return a/max(float(a.max()),EPS)

def native_crop(field,x_m):
    I=np.abs(np.asarray(field))**2; x=np.asarray(x_m,float)
    pix=np.interp(AXIS_UM*1e-6,x,np.arange(len(x))); yy,xx=np.meshgrid(pix,pix,indexing="ij")
    return norm2d(ndimage.map_coordinates(I,[yy,xx],order=1,mode="constant",cval=0))

def detector_crop(field,x_m):
    I=(np.abs(np.asarray(field))**2)[None,...]
    shown,_=sample_camera_response(I,np.asarray(x_m,float),AXIS_UM*1e-6,pixel_pitch_m=PIXEL_M,quadrature_n=3)
    return norm2d(shown[0])

def radial_profile(im,dr=1.5,rmax=170):
    X,Y=np.meshgrid(AXIS_UM,AXIS_UM,indexing="xy"); R=np.hypot(X,Y)
    edges=np.arange(0.,rmax+dr,dr); ids=np.digitize(R.ravel(),edges)-1; good=(ids>=0)&(ids<len(edges)-1)
    s=np.bincount(ids[good],weights=np.asarray(im).ravel()[good],minlength=len(edges)-1); n=np.bincount(ids[good],minlength=len(edges)-1)
    return .5*(edges[:-1]+edges[1:]),s/np.maximum(n,1)

def metrics(im):
    X,Y=np.meshgrid(AXIS_UM,AXIS_UM,indexing="xy"); R=np.hypot(X,Y); rr,p=radial_profile(im)
    rad=np.interp(R.ravel(),rr,p,left=p[0],right=p[-1]).reshape(R.shape); m=(R>=20)&(R<=140)
    nrmse=float(np.sqrt(np.mean((im[m]-rad[m])**2))/max(np.sqrt(np.mean(rad[m]**2)),EPS))
    return dict(axisym_nrmse=nrmse,xmirror_rmse=float(np.sqrt(np.mean((im[m]-im[:,::-1][m])**2))),ymirror_rmse=float(np.sqrt(np.mean((im[m]-im[::-1,:][m])**2))))

def main():
    out=ROOT/"outputs"/"validation"/"q20_highres_radial_symmetry"; out.mkdir(parents=True,exist_ok=True)
    src=EXP/"outputs"/"digital_twin_correction"; rs=json.loads((src/"run_summary.json").read_text()); scale=float(rs["effective_axicon"]["effective_scale_relative_to_repository_2deg_assumption"])
    zscan=pd.read_csv(src/"full_route_z_registration_scan.csv"); z0=float(zscan.loc[zscan.selected.astype(bool),"value"].iloc[0]); z_abs=(z0+REP_Z_REL_MM)*1e-3
    cfg=SystemErrorConfig(fourf=FourFError(iris_radius_scale=1.),axicon=AxiconError(base_angle_scale=scale)); wavelength=float(hardware_value(canonical_hardware_manifest(),"wavelength_m"))
    rows=[]; natives=[]; detectors=[]; profiles=[]; ref=None
    for N in GRID_NS:
        rt=build_multirate_system_route(f"V{Q}",relay_grid_n=RELAY_N,propagation_grid_n=N,window_m=FIT_WINDOW_M,config=cfg)
        prop=build_fixed_support_spectrum(rt["post_axicon"],rt["grid"],wavelength_m=wavelength,z_max_m=max(abs(z_abs),1e-6),minimum_retained_spectral_power=.99)
        field=native_field_at_z(prop,z_abs); nat=native_crop(field,rt["grid"]["x"]); det=detector_crop(field,rt["grid"]["x"])
        if ref is None: ref=nat.copy()
        mn,md=metrics(nat),metrics(det); rr,pr=radial_profile(nat)
        rows.append(dict(grid_n=N,dx_um=FIT_WINDOW_M/N*1e6,samples_per_axicon_period=float(rt["metadata"]["samples_per_axicon_radial_phase_period"]),retained_spectral_power=float(prop.retained_spectral_power_fraction),native_corr_to_2048=float(np.corrcoef(ref.ravel(),nat.ravel())[0,1]),**{f"native_{k}":v for k,v in mn.items()},**{f"detector_{k}":v for k,v in md.items()}))
        natives.append(nat); detectors.append(det); profiles.append((rr,pr)); del rt,prop,field; gc.collect()
    pd.DataFrame(rows).to_csv(out/"q20_highres_radial_symmetry_metrics.csv",index=False); (out/"q20_highres_radial_symmetry_summary.json").write_text(json.dumps({"z_relative_mm":REP_Z_REL_MM,"metrics":rows},indent=2))
    ext=[AXIS_UM[0],AXIS_UM[-1],AXIS_UM[0],AXIS_UM[-1]]; fig,axs=plt.subplots(2,3,figsize=(14,9),constrained_layout=True)
    for j,N in enumerate(GRID_NS):
        axs[0,j].imshow(natives[j],origin="lower",extent=ext,cmap=THERMAL,vmin=0,vmax=1); axs[0,j].set(title=f"optical N={N}, {rows[j]['samples_per_axicon_period']:.2f} samp/period",xlabel="x (um)",ylabel="y (um)",aspect="equal")
        axs[1,j].imshow(detectors[j],origin="lower",extent=ext,cmap=THERMAL,vmin=0,vmax=1); axs[1,j].set(title=f"5.5 um detector, axisym NRMSE={rows[j]['detector_axisym_nrmse']:.3f}",xlabel="x (um)",ylabel="y (um)",aspect="equal")
    fig.suptitle("q=20 nominal radial symmetry versus axicon-plane sampling"); fig.savefig(out/"22_q20_radial_symmetry_sampling_audit.png",dpi=600,bbox_inches="tight"); fig.savefig(out/"22_q20_radial_symmetry_sampling_audit.pdf",bbox_inches="tight"); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5),constrained_layout=True)
    for (rr,pr),N in zip(profiles,GRID_NS): ax.plot(rr,pr,label=f"N={N}")
    ax.set(xlim=(0,140),xlabel="radius (um)",ylabel="azimuthal mean intensity",title="Underlying optical radial profiles"); ax.grid(alpha=.2); ax.legend(); fig.savefig(out/"23_q20_radial_profiles_sampling_audit.png",dpi=600,bbox_inches="tight"); fig.savefig(out/"23_q20_radial_profiles_sampling_audit.pdf",bbox_inches="tight"); plt.close(fig)
    print(pd.DataFrame(rows).to_string(index=False))
if __name__=="__main__": main()
