"""Audit whether q=20 cross-like sidelobes are a propagation-grid artefact.

The real-data inverse has so far used FIT_N=2048 for tractability.  With the
measured axicon k_perp this is only about 2.7 samples per conical phase period.
This script holds every optical parameter fixed and renders the same nominal
q=20 field at 2048, 3072, and 4096 samples on the same 10 mm window.  It reports
native optical-field radial symmetry separately from the 5.5 um detector
response, so detector integration cannot be confused with propagation aliasing.
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
for p in (ROOT, EXP):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from real_bmg_digital_twin_correction import AxiconError, FourFError, FIT_WINDOW_M, PIXEL_M, Q, RELAY_N, SystemErrorConfig
from vbb_study.digital_twin.detector_response import plane_normalise, sample_camera_response
from vbb_study.digital_twin.vortex_continuous_propagation import build_fixed_support_spectrum, native_field_at_z
from vbb_study.digital_twin.vortex_system_route import build_multirate_system_route
from vbb_study.digital_twin.phase2a_contracts import canonical_hardware_manifest, hardware_value

EPS = np.finfo(float).tiny
THERMAL = "inferno"
GRID_NS = (2048, 3072, 4096)
REP_Z_REL_MM = -10.0
AXIS_UM = np.linspace(-180.0, 180.0, 241)


def norm2d(a):
    a=np.maximum(np.asarray(a,float),0.0)
    return a/max(float(np.max(a)),EPS)


def sample_native_intensity(field, x_m, axis_um):
    I=np.abs(np.asarray(field))**2
    coord=np.asarray(x_m,float)
    # direct bilinear sampling of the optical intensity, no detector integration
    pix=np.interp(np.asarray(axis_um)*1e-6,coord,np.arange(len(coord)))
    yy,xx=np.meshgrid(pix,pix,indexing="ij")
    return norm2d(ndimage.map_coordinates(I,[yy,xx],order=1,mode="constant",cval=0.0))


def detector_intensity(field,x_m,axis_um):
    shown,_=sample_camera_response(
        np.abs(np.asarray(field))**2,
        np.asarray(x_m,float),
        np.asarray(axis_um,float)*1e-6,
        pixel_pitch_m=PIXEL_M,
        quadrature_n=3,
        input_is_intensity=True,
    )
    return norm2d(shown)


def radial_profile(image,axis_um,dr=1.5,rmax=150.0):
    X,Y=np.meshgrid(axis_um,axis_um,indexing="xy"); R=np.hypot(X,Y)
    edges=np.arange(0.0,rmax+dr,dr); ids=np.digitize(R.ravel(),edges)-1
    good=(ids>=0)&(ids<len(edges)-1)
    sums=np.bincount(ids[good],weights=np.asarray(image).ravel()[good],minlength=len(edges)-1)
    counts=np.bincount(ids[good],minlength=len(edges)-1)
    return .5*(edges[:-1]+edges[1:]),sums/np.maximum(counts,1)


def axisymmetry_metrics(image,axis_um):
    X,Y=np.meshgrid(axis_um,axis_um,indexing="xy"); R=np.hypot(X,Y)
    rr,p=radial_profile(image,axis_um,dr=1.5,rmax=170.0)
    expected=np.interp(R.ravel(),rr,p,left=p[0],right=p[-1]).reshape(R.shape)
    ann=(R>=20)&(R<=140)
    residual=float(np.sqrt(np.mean((image[ann]-expected[ann])**2)))
    denom=max(float(np.sqrt(np.mean(expected[ann]**2))),EPS)
    radial_nrmse=residual/denom
    xmirror=float(np.sqrt(np.mean((image[ann]-image[:,::-1][ann])**2)))
    ymirror=float(np.sqrt(np.mean((image[ann]-image[::-1,:][ann])**2)))
    return {"axisymmetric_reconstruction_nrmse":radial_nrmse,"x_mirror_rmse":xmirror,"y_mirror_rmse":ymirror}


def main():
    out=ROOT/"outputs"/"validation"/"q20_highres_radial_symmetry"
    out.mkdir(parents=True,exist_ok=True)
    source=EXP/"outputs"/"digital_twin_correction"
    summary=json.loads((source/"run_summary.json").read_text(encoding="utf-8"))
    scale=float(summary["effective_axicon"]["effective_scale_relative_to_repository_2deg_assumption"])
    zscan=pd.read_csv(source/"full_route_z_registration_scan.csv")
    z0=float(zscan.loc[zscan.selected.astype(bool),"value"].iloc[0])
    z_abs=(z0+REP_Z_REL_MM)*1e-3
    config=SystemErrorConfig(fourf=FourFError(iris_radius_scale=1.0),axicon=AxiconError(base_angle_scale=scale))
    manifest=canonical_hardware_manifest(); wavelength=float(hardware_value(manifest,"wavelength_m"))

    rows=[]; images_native=[]; images_detector=[]; profiles=[]
    reference_native=None
    for n in GRID_NS:
        route=build_multirate_system_route(f"V{Q}",relay_grid_n=RELAY_N,propagation_grid_n=n,window_m=FIT_WINDOW_M,config=config)
        spp=float(route["metadata"]["samples_per_axicon_radial_phase_period"])
        prop=build_fixed_support_spectrum(route["post_axicon"],route["grid"],wavelength_m=wavelength,z_max_m=max(abs(z_abs),1e-6),minimum_retained_spectral_power=0.99)
        field=native_field_at_z(prop,z_abs)
        native=sample_native_intensity(field,np.asarray(route["grid"]["x"],float),AXIS_UM)
        detector=detector_intensity(field,np.asarray(route["grid"]["x"],float),AXIS_UM)
        mn=axisymmetry_metrics(native,AXIS_UM); md=axisymmetry_metrics(detector,AXIS_UM)
        if reference_native is None: reference_native=native
        corr_to_2048=float(np.corrcoef(reference_native.ravel(),native.ravel())[0,1])
        rr,pr=radial_profile(native,AXIS_UM)
        rows.append({"grid_n":n,"dx_um":FIT_WINDOW_M/n*1e6,"samples_per_axicon_period":spp,"retained_spectral_power":prop.retained_spectral_power_fraction,"native_corr_to_2048":corr_to_2048,**{f"native_{k}":v for k,v in mn.items()},**{f"detector_{k}":v for k,v in md.items()}})
        images_native.append(native); images_detector.append(detector); profiles.append((rr,pr))
        del route,prop,field; gc.collect()

    df=pd.DataFrame(rows); df.to_csv(out/"q20_highres_radial_symmetry_metrics.csv",index=False)
    (out/"q20_highres_radial_symmetry_summary.json").write_text(json.dumps({"representative_z_relative_mm":REP_Z_REL_MM,"metrics":rows},indent=2),encoding="utf-8")

    fig,axs=plt.subplots(2,3,figsize=(14,9),constrained_layout=True)
    extent=[AXIS_UM[0],AXIS_UM[-1],AXIS_UM[0],AXIS_UM[-1]]
    for j,n in enumerate(GRID_NS):
        axs[0,j].imshow(images_native[j],origin="lower",extent=extent,cmap=THERMAL,vmin=0,vmax=1,interpolation="nearest")
        axs[0,j].set(title=f"optical field, N={n}\n{rows[j]['samples_per_axicon_period']:.2f} samples/axicon period",xlabel="x (um)",ylabel="y (um)",aspect="equal")
        axs[1,j].imshow(images_detector[j],origin="lower",extent=extent,cmap=THERMAL,vmin=0,vmax=1,interpolation="nearest")
        axs[1,j].set(title=f"after 5.5 um detector response\naxisym NRMSE={rows[j]['detector_axisymmetric_reconstruction_nrmse']:.3f}",xlabel="x (um)",ylabel="y (um)",aspect="equal")
    fig.suptitle("q=20 nominal radial symmetry versus axicon-plane sampling")
    fig.savefig(out/"22_q20_radial_symmetry_sampling_audit.png",dpi=600,bbox_inches="tight"); fig.savefig(out/"22_q20_radial_symmetry_sampling_audit.pdf",bbox_inches="tight"); plt.close(fig)

    fig,ax=plt.subplots(figsize=(8,5),constrained_layout=True)
    for (rr,pr),n in zip(profiles,GRID_NS): ax.plot(rr,pr,label=f"N={n}")
    ax.set(xlim=(0,140),xlabel="radius (um)",ylabel="azimuthal mean intensity",title="Underlying optical radial profiles"); ax.grid(alpha=.2); ax.legend()
    fig.savefig(out/"23_q20_radial_profiles_sampling_audit.png",dpi=600,bbox_inches="tight"); fig.savefig(out/"23_q20_radial_profiles_sampling_audit.pdf",bbox_inches="tight"); plt.close(fig)
    print(df.to_string(index=False))

if __name__=="__main__": main()
