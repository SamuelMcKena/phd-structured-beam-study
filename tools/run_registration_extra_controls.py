#!/usr/bin/env python3
"""Reproducible no-carrier scaling and severe-stress presentation control."""
import argparse,json
from dataclasses import replace
from pathlib import Path
import numpy as np
from vbb_study.digital_twin.registration_reference import *
from tools.run_registration_reference_study import image_metrics,write_csv

def run(out):
    out.mkdir(parents=True,exist_ok=True);p=ReferenceParameters();rows=[]
    for w in [25,50,100,150,250]:
        for L in [0,1,3,5,10,20]:
            rows.append(dict(L=L,w_px=w,eta=L/w,infidelity=1-local_fidelity(L,w,(.5,0),carrier=False)))
    write_csv(out/'no_carrier_scaling.csv',rows)
    w=25;L=20;B=5*L/(2*np.pi*w*p.pitch_m);pp=replace(p,filter_radius_cpm=B)
    def get(d):
        s,f,W=integrated_spectrum(L,w,(d,0),params=pp,quadrature=2,spectral_replicas=1)
        return selected_coefficients(s,f,W,params=pp),W
    C,W=get(0);z=np.linspace(.0002,.003,25)
    Es,x,ch=propagate_axicon(C,W,z,dx_target=1.25e-6)
    k=int(np.argmax(np.max(abs(Es)**2,axis=(1,2))));ref=Es[k];files={'x':x,'zstar':z[k]};rows=[]
    for d in [0,.125,.25,.375,.5]:
        C,W=get(d);E,x,ch=propagate_axicon(C,W,[z[k]],dx_target=1.25e-6)
        files[f'd{d:g}']=E[0];rows.append(dict(L=L,w_px=w,offset_px=d,filter_radius_cpm=B,iris_policy='adapted',z_m=float(z[k]),**image_metrics(E[0],ref,x,L)))
    np.savez_compressed(out/'extreme_stress.npz',**files);write_csv(out/'extreme_stress.csv',rows)
    # Severe headline: double spatial sampling with the same quadrature and plane.
    rows=[]
    for d in [0,.5]:
        C,W=get(d);E,x,ch=propagate_axicon(C,W,[z[k]],dx_target=.8e-6)
        if d==0:Rfine=E[0]
        check=ch[0].copy();check['global_propagation_power_ratio']=check.pop('power_ratio')
        rows.append(dict(L=L,w_px=w,offset_px=d,**check,**image_metrics(E[0],Rfine,x,L)))
    write_csv(out/'extreme_stress_convergence.csv',rows)
    # Fitted cone control, explicitly uncalibrated, on the adjusted-iris bench field.
    pp=replace(p,filter_radius_cpm=5*20/(2*np.pi*250*p.pitch_m),cone_kr_override_m_inv=531199.8037877748)
    def fitted(d):
        s,f,W=integrated_spectrum(20,250,(d,0),quadrature=2,params=pp)
        return selected_coefficients(s,f,W,params=pp),W
    C,W=fitted(0);z=np.linspace(.001,.024,25)
    Es,x,ch=propagate_axicon(C,W,z,params=pp,propagation_window_m=.008)
    k=int(np.argmax(np.max(abs(Es)**2,axis=(1,2))));R=Es[k]
    C,W=fitted(.5);E,x,ch=propagate_axicon(C,W,[z[k]],params=pp,propagation_window_m=.008)
    result=dict(kr_m_inv=pp.kr,cone_status='archived modal-fit value, not independent hardware calibration',
                z_m=float(z[k]),**image_metrics(E[0],R,x,20),checks=ch)
    (out/'fitted_cone_control.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--out',type=Path,default=Path('outputs/validation/registration_iris_adjusted'))
    run(a.parse_args().out)
