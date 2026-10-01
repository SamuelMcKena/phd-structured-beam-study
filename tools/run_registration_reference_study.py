#!/usr/bin/env python3
"""Staged definitive registration study; source data, never figure-derived metrics."""
import argparse, csv, hashlib, json, subprocess, time
from pathlib import Path
from dataclasses import asdict,replace
import numpy as np
from scipy import fft
from vbb_study.digital_twin.registration_reference import *
from vbb_study.digital_twin.slm_registration_metrics import (
    complex_fidelity, translation_registered_fidelity, vortex_structure)

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'outputs/validation/registration_iris_adjusted'


def write_csv(path,rows):
    if not rows: return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def grid(x):
    X,Y=np.meshgrid(x,x)
    return dict(N=len(x),dx=float(x[1]-x[0]),x=x,X=X,Y=Y)


def image_metrics(E,R,x,L):
    I=abs(E)**2; J=abs(R)**2; g=grid(x)
    raw=complex_fidelity(E,R)
    reg=translation_registered_fidelity(E,R,g)
    p=I.sum(); pr=J.sum();cx=float((I*g['X']).sum()/p);cy=float((I*g['Y']).sum()/p)
    # Absolute intensity residual and power-normalised shape residual are distinct.
    residual=float(np.sqrt(np.sum((I-J)**2)/np.sum(J**2)))
    shape=float(np.sqrt(np.sum((I/p-J/pr)**2)/np.sum((J/pr)**2)))
    cov=np.array([[((g['X']-cx)**2*I).sum(),((g['X']-cx)*(g['Y']-cy)*I).sum()],
                  [((g['X']-cx)*(g['Y']-cy)*I).sum(),((g['Y']-cy)**2*I).sum()]])/p
    ev=np.linalg.eigvalsh(cov)
    struct=vortex_structure(E,g,charge=L,r_max_m=max(abs(x))*.8)
    return dict(raw_infidelity=max(0.,1-raw),centroid_registered_infidelity=reg['infidelity'],
        centroid_x_m=cx,centroid_y_m=cy,centroid_displacement_m=reg['centroid_separation_m'],
        power_ratio=float(p/pr),peak_ratio=float(I.max()/J.max()),
        residual_l2=residual,power_normalised_residual_l2=shape,
        ellipticity=float(np.sqrt(ev[0]/ev[-1])),**struct)


def run(out, stages, iris_policy='fixed'):
    out.mkdir(parents=True,exist_ok=True);p=ReferenceParameters();rows=[];cache={}
    offsets=[0,.125,.25,.375,.5]
    def route(L,w,r1=(0,0),r2=(0,0),q=1,architecture='upstream_vortex',band=None):
        if band is None:
            band=max(2500.,5*L/(2*np.pi*w*p.pitch_m)) if iris_policy=='adapted' else 2500.
        key=(L,w,tuple(r1),tuple(r2),q,architecture,band)
        if key not in cache:
            pp=replace(p,filter_radius_cpm=band)
            replicas=1 if 2*p.carrier_cpm+band>=.5/p.pitch_m else 0
            S,f,W=integrated_spectrum(L,w,r1,r2,quadrature=q,architecture=architecture,params=pp,spectral_replicas=replicas)
            C=selected_coefficients(S,f,W,params=pp)
            cache[key]=(C,W)
        return cache[key]
    def pre_metrics(C,C0,W,L):
        n=max(512,fft.next_fast_len(C.shape[0]+4));E=synthesize(C,W,n);R=synthesize(C0,W,n)
        x=(np.arange(n)-n/2+.5)*W/n
        return image_metrics(E,R,x,L)
    if 'sweep' in stages:
        for w in [25,50,100,150,250]:
            for L in [0,1,3,5,10,20]:
                C0,W=route(L,w)
                pin=p.fill_factor**2*np.pi*(w*p.pitch_m)**2/2
                for axis in ['x','y','diagonal']:
                    for d in offsets:
                        rho=(d,0) if axis=='x' else ((0,d) if axis=='y' else (d,d))
                        C,W=route(L,w,rho)
                        row=dict(L=L,w_px=w,eta=L/w,axis=axis,offset_px=d,
                            filter_radius_cpm=max(2500.,5*L/(2*np.pi*w*p.pitch_m)) if iris_policy=='adapted' else 2500.,
                            local_infidelity=1-local_fidelity(L,w,rho),
                            selected_fraction=float((abs(C)**2).sum()*W**2/pin),
                            window_m=W,**pre_metrics(C,C0,W,L))
                        rows.append(row)
                print('sweep',w,L,flush=True)
                write_csv(out/'sweep.csv',rows)
    if 'unit_cell' in stages:
        rows=[]
        for w in [50,250]:
            C0,W=route(20,w)
            for y in np.arange(8)/8:
                for x in np.arange(8)/8:
                    C,W=route(20,w,(x,y))
                    rows.append(dict(L=20,w_px=w,rho_x=x,rho_y=y,raw_infidelity=1-spectrum_fidelity(C,C0),
                                power_ratio=float(np.sum(abs(C)**2)/np.sum(abs(C0)**2))))
            write_csv(out/'unit_cell.csv',rows);print('unit cell',w,flush=True)
    if 'roles' in stages:
        rows=[]
        for w in [50,250]:
            for L in [10,20]:
                C0,W=route(L,w)
                for axis in ['x','y','diagonal']:
                    for d in [0,.125,.25,.375,.5]:
                        rho=(d,0) if axis=='x' else ((0,d) if axis=='y' else (d,d))
                        for mode in ['vortex_owner','blaze_owner','common','differential']:
                            r1=rho if mode!='blaze_owner' else (0,0)
                            r2=rho if mode in ['blaze_owner','common'] else (tuple(-v for v in rho) if mode=='differential' else (0,0))
                            C,W=route(L,w,r1,r2)
                            B,W=route(L,w,r2,r1,architecture='downstream_vortex')
                            rows.append(dict(L=L,w_px=w,axis=axis,offset_px=d,mode=mode,
                                A_shifted_physical_panel='SLM2' if mode=='blaze_owner' else 'SLM1',
                                B_shifted_physical_panel='SLM1' if mode=='blaze_owner' else 'SLM2',
                                max_AB_coefficient_difference=float(np.max(abs(C-B))),
                                **pre_metrics(C,C0,W,L)))
                write_csv(out/'roles.csv',rows);print('roles',w,L,flush=True)
    if 'convergence' in stages:
        rows=[]
        for w in [50,250]:
            for L in [10,20]:
                for q in [1,2,4]:
                    C0,W=route(L,w,q=q)
                    for axis in ['x','y','diagonal']:
                        rho=(.5,0) if axis=='x' else ((0,.5) if axis=='y' else (.5,.5))
                        C,W=route(L,w,rho,q=q)
                        rows.append(dict(L=L,w_px=w,q=q,axis=axis,raw_infidelity=1-spectrum_fidelity(C,C0),
                             power_ratio=float(np.sum(abs(C)**2)/np.sum(abs(C0)**2))))
                write_csv(out/'convergence.csv',rows);print('convergence',w,L,flush=True)
        rows=[]
        for L in [10,20]:
            for band in [1500,2500,4000,5000]:
                for axis in ['x','y','diagonal']:
                    rho=(.5,0) if axis=='x' else ((0,.5) if axis=='y' else (.5,.5))
                    C0,W=route(L,250,band=band);C,W=route(L,250,rho,band=band)
                    rows.append(dict(L=L,w_px=250,bandwidth_cpm=band,axis=axis,
                                 raw_infidelity=1-spectrum_fidelity(C,C0),
                                 power_ratio=float(np.sum(abs(C)**2)/np.sum(abs(C0)**2))))
        write_csv(out/'bandwidth_sensitivity.csv',rows)
    if 'fields' in stages:
        files={};planes=[];axial=[];post=[];conv=[];finepost=[]
        for w in [50,250]:
            for L in [10,20]:
                C0,W=route(L,w,q=2)
                Wprop=.008 if w==250 else W
                # Conditional nominal 20-degree cone; sample 0..16/7 mm for bench/stress.
                z=np.linspace(.0002,.016 if w==250 else .007,25)
                E0,x,checks0=propagate_axicon(C0,W,z,propagation_window_m=Wprop)
                regional_peak=np.max(abs(E0)**2,axis=(1,2));k=int(np.argmax(regional_peak));zstar=z[k]
                print('fields baseline',w,L,'zstar mm',zstar*1e3,flush=True)
                pre0=synthesize(C0,W,512);xp=(np.arange(512)-256+.5)*W/512
                files[f'w{w}_L{L}_pre_x']=xp;files[f'w{w}_L{L}_post_x']=x
                files[f'w{w}_L{L}_z']=z;files[f'w{w}_L{L}_zstar']=np.asarray(zstar)
                for d in offsets:
                    C,W=route(L,w,(d,0),q=2)
                    pre=synthesize(C,W,512)
                    if d==0: E=E0;checks=checks0
                    else:
                        zs=z if d==.5 else [zstar]
                        E,x,checks=propagate_axicon(C,W,zs,propagation_window_m=Wprop)
                    chosen=E[k] if d in [0,.5] else E[0]
                    files[f'w{w}_L{L}_d{d:g}_pre']=pre
                    files[f'w{w}_L{L}_d{d:g}_post']=chosen
                    files[f'w{w}_L{L}_d{d:g}_C']=C
                    files[f'w{w}_L{L}_W']=np.asarray(W)
                    post.append(dict(L=L,w_px=w,axis='x',offset_px=d,z_m=float(zstar),
                                **image_metrics(chosen,E0[k],x,L)))
                    for check in checks:
                        if check['propagation_power_drift_fraction']>.05:
                            raise RuntimeError('propagation numerical power gate failed')
                        planes.append(dict(L=L,w_px=w,offset_px=d,**check))
                    if d in [0,.5]:
                        for j,zz in enumerate(z):
                            m=image_metrics(E[j],E0[j],x,L)
                            axial.append(dict(L=L,w_px=w,offset_px=d,z_m=float(zz),
                                    roi_peak=float(np.max(abs(E[j])**2)),**m))
                # Increased cone sampling at the exact same physical plane.
                for d in offsets:
                    C,W=route(L,w,(d,0),q=4)
                    Ef,xf,cf=propagate_axicon(C,W,[zstar],dx_target=1.25e-6,propagation_window_m=Wprop)
                    files[f'w{w}_L{L}_d{d:g}_post_fine']=Ef[0]
                    files[f'w{w}_L{L}_post_fine_x']=xf
                    if d==0: Ef0=Ef[0].copy()
                    finepost.append(dict(L=L,w_px=w,axis='x',offset_px=d,z_m=float(zstar),
                                         **image_metrics(Ef[0],Ef0,xf,L)))
                    conv.append(dict(L=L,w_px=w,offset_px=d,
                                     roi_peak=float(np.max(abs(Ef[0])**2)),**cf[0]))
                # y and diagonal propagated headline controls.
                for axis,rho in [('y',(0,.5)),('diagonal',(.5,.5))]:
                    C,W=route(L,w,rho,q=2)
                    E,x,checks=propagate_axicon(C,W,[zstar],propagation_window_m=Wprop)
                    files[f'w{w}_L{L}_{axis}_half_post']=E[0]
                    post.append(dict(L=L,w_px=w,axis=axis,offset_px=.5,z_m=float(zstar),
                                 **image_metrics(E[0],E0[k],x,L)))
                np.savez_compressed(out/'fields.npz',**files)
                write_csv(out/'post_metrics.csv',post);write_csv(out/'axial.csv',axial)
                write_csv(out/'propagation_checks.csv',planes);write_csv(out/'post_convergence.csv',conv)
                write_csv(out/'post_fine_metrics.csv',finepost)
                print('fields done',w,L,flush=True)
        # Window convergence at the bench L20 headline; original frequency window retained.
        rows=[]
        zstar=float(files['w250_L20_zstar'])
        for d in [0,.5]:
            C,W=route(20,250,(d,0),q=2)
            E,x,ch=propagate_axicon(C,W,[zstar],propagation_window_m=W)
            rows.append(dict(L=20,w_px=250,offset_px=d,roi_peak=float(np.max(abs(E)**2)),**ch[0]))
        write_csv(out/'window_convergence.csv',rows)
    if True:  # Refresh evidence provenance after every completed stage.
        manifest=dict(route='pixel_integrated_ideal_4F_effective_channel',parameters=asdict(p),
            correction='FLAT-CORRECTION PHYSICS-ISOLATION BASELINE',
            interpanel_transfer='unknown; identity assumed for local reference only',
            mapping_mode='fixed_physical_optics',
            hardware_maturity='user-reported pitch, beam size and blaze; uncalibrated relay/iris/cone',
            focal_length_status='equal-f illustrative 150 mm; lens separation 300 mm does not fix individual f',
            axicon_status='conditional 20-degree BASE-angle ideal cone; part number/orientation/index unverified',
            filter_status=('case-adjusted diagnostic iris, fixed within each registration comparison' if iris_policy=='adapted' else '2500 cycles/m fixed control; bandwidth sensitivity supplied'),
            iris_policy=iris_policy,
            adapted_bandwidth_rule='max(2500, 5L/(2 pi w)); fixed across all registration states within each L,w comparison',
            wide_iris_scope='ideal selected-channel diagnostic; broad aperture may admit physical zero/unwanted orders absent in throughput-only model',
            pixel_command='centre_sample consistent with inspected GUI; ideal 256-level LUT',
            illumination_quadrature='sweep q=1; headline fields q=2; convergence q=4',
            pixel_phase_integration='analytic top-hat rectangle sinc, exact lattice intersections',
            scalar_boundary='relative free-space morphology only; no vector/objective/material claims',
            beta_rad=p.beta,cone_radial_period_m=2*np.pi/p.kr,
            baseline_git_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            source_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [
                ROOT/'vbb_study/digital_twin/registration_reference.py',Path(__file__)]},
            files={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in out.iterdir() if f.is_file() and f.name!='manifest.json'})
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--out',type=Path,default=DEFAULT)
    a.add_argument('--stages',default='sweep,unit_cell,roles,convergence,fields')
    a.add_argument('--iris-policy',choices=['fixed','adapted'],default='adapted');args=a.parse_args()
    run(args.out,args.stages.split(','),args.iris_policy)
