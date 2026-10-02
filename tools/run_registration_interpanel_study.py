#!/usr/bin/env python3
"""Resumable staged 200 mm study. Large arrays remain in ignored outputs."""
import argparse,json,hashlib,subprocess
from pathlib import Path
from dataclasses import replace,asdict
import numpy as np
from scipy.ndimage import map_coordinates
from vbb_study.digital_twin.registration_interpanel import *
from vbb_study.digital_twin.registration_reference import synthesize,propagate_axicon,spectrum_fidelity,local_fidelity
from tools.run_registration_reference_study import image_metrics,write_csv

ROOT=Path(__file__).resolve().parents[1]
CASES=[(20,50),(10,250),(20,250)]
ARCHS=['upstream_vortex','downstream_vortex']
OFFSETS=[0,.125,.25,.375,.5]

def parameters(L,w):
    p=ReferenceParameters()
    return replace(p,filter_radius_cpm=max(2500,5*L/(2*np.pi*w*p.pitch_m)))

def filename(L,w,arch,panel):return f'fields_L{L}_w{w}_{arch}_panel{panel}.npz'

def offsets(d,panel):return ((d,0),(0,0)) if panel==1 else ((0,0),(d,0))

def diagnostic_z(L,w,arch):
    if w==50:return .012 if arch=='upstream_vortex' else .001
    return (.006 if L==10 else .012) if arch=='upstream_vortex' else (.008 if L==10 else .004)

def run(out,stage):
    out.mkdir(parents=True,exist_ok=True);t=InterpanelParameters();rows=[];ledger=[]
    if stage=='local':
        # Independently reproduce the complete corrected two-blaze local layer.
        from tools.run_registration_reference_study import run as local_run
        from scipy.stats import linregress
        local=out/'local_reference';local_run(local,['sweep','unit_cell'],'adapted')
        scaling=[]
        for w in [25,50,100,150,250]:
            for L in [1,3,5,10,20]:
                scaling.append(dict(L=L,w_px=w,eta=L/w,
                    infidelity=1-local_fidelity(L,w,(.5,0),carrier=False)))
        write_csv(local/'no_carrier_scaling.csv',scaling)
        fit=linregress(np.log([r['eta'] for r in scaling]),np.log([r['infidelity'] for r in scaling]))
        (local/'scaling_fit.json').write_text(json.dumps(dict(prefactor=float(np.exp(fit.intercept)),
            exponent=float(fit.slope),log_R_squared=float(fit.rvalue**2),
            scope='no-carrier local diagnostic only; not a law for two-blaze selection or 200 mm propagation'),indent=2))
        return
    for L,w in CASES:
        p=parameters(L,w);W=t.panel2_window_m
        for arch in ARCHS:
            owner=1 if arch=='upstream_vortex' else 2;z=diagnostic_z(L,w,arch);cache={}
            if stage=='sweep':
                b=interpanel_route(L,w,architecture=arch,hardware=p,transfer=t,source_cache=cache)
                c0=b['coefficients'];n=2048;x=(np.arange(n)-n/2+.5)*W/n;E0=synthesize(c0,W,n)
                post0,px,cl=propagate_axicon(c0,W,[z],params=p,propagation_window_m=.008)
                for panel in [1,2]:
                    fields=[];posts=[];cs=[]
                    for d in OFFSETS:
                        r1,r2=offsets(d,panel)
                        r=b if d==0 else interpanel_route(L,w,r1,r2,architecture=arch,hardware=p,transfer=t,source_cache=cache)
                        c=r['coefficients'];E=synthesize(c,W,n);cs.append(c)
                        fields.append(E[np.ix_(abs(x)<.003,abs(x)<.003)])
                        row=dict(L=L,w_px=w,architecture=arch,panel=panel,role='vortex + blaze' if panel==owner else 'flat correction + blaze',
                                 offset_px=d,selected_infidelity=1-spectrum_fidelity(c,c0),
                                 **{'pre_'+k:v for k,v in image_metrics(E,E0,x,L).items()})
                        ledger.append(dict(L=L,w_px=w,architecture=arch,panel=panel,offset_px=d,**r['metadata']))
                        if panel==owner:
                            post,px,checks=propagate_axicon(c,W,[z],params=p,propagation_window_m=.008)
                            posts.append(post[0]);row.update(z_m=z,**{'post_'+k:v for k,v in image_metrics(post[0],post0[0],px,L).items()})
                            ledger.append(dict(L=L,w_px=w,architecture=arch,panel=panel,offset_px=d,diagnostic='post axicon',**checks[0]))
                        rows.append(row);write_csv(out/'metrics.csv',rows);write_csv(out/'power_ledger.csv',ledger)
                        print('sweep',L,w,arch,panel,d,flush=True)
                    np.savez_compressed(out/filename(L,w,arch,panel),pre=np.array(fields),pre_x=x[abs(x)<.003],
                         post=np.array(posts),post_x=px,coefficients=np.array(cs),window_m=W,z_m=z,offsets=OFFSETS)
            else:
                a=np.load(out/filename(L,w,arch,owner));ref=a['coefficients'][[0,-1]]
                if stage=='controls':
                    configs=[('primary',t),('band80k',replace(t,numerical_bandwidth_cpm=80000)),
                            ('window48mm',replace(t,source_window_m=.048)),('source_q4',replace(t,source_quadrature=4)),
                            ('panel_q4',replace(t,panel2_quadrature=4))]
                    for name,tt in configs:
                        cc=[]
                        for d in [0,.5]:
                            r1,r2=offsets(d,owner)
                            r=interpanel_route(L,w,r1,r2,architecture=arch,hardware=p,transfer=tt,source_cache=cache)
                            cc.append(r['coefficients']);ledger.append(dict(L=L,w_px=w,architecture=arch,control=name,offset_px=d,**r['metadata']))
                        rows.append(dict(L=L,w_px=w,architecture=arch,control=name,
                            selected_infidelity=1-spectrum_fidelity(cc[1],cc[0]),
                            baseline_coefficient_error=float(np.linalg.norm(cc[0]-ref[0])/np.linalg.norm(ref[0])),
                            shifted_coefficient_error=float(np.linalg.norm(cc[1]-ref[1])/np.linalg.norm(ref[1]))))
                        np.savez_compressed(out/f'control_L{L}_w{w}_{arch}_{name}.npz',coefficients=np.array(cc))
                        write_csv(out/'convergence.csv',rows);write_csv(out/'convergence_ledger.csv',ledger)
                        print('control',L,w,arch,name,flush=True)
                elif stage=='postcontrols':
                    fine_control=np.load(out/f'control_L{L}_w{w}_{arch}_panel_q4.npz')['coefficients']
                    for name,cc,ww,dx in [('primary',ref,.008,2e-6),('window12mm',ref,.012,2e-6),
                                        ('dx1um',ref,.008,1e-6),('panel_q4',fine_control,.008,2e-6)]:
                        ef=[]
                        for c in cc:
                            e,xx,ch=propagate_axicon(c,W,[z],params=p,propagation_window_m=ww,dx_target=dx)
                            ef.append(e[0]);ledger.append(dict(L=L,w_px=w,architecture=arch,control=name,**ch[0]))
                        rows.append(dict(L=L,w_px=w,architecture=arch,control=name,z_m=z,**image_metrics(ef[1],ef[0],xx,L)))
                        if name=='dx1um':np.savez_compressed(out/f'fine_post_L{L}_w{w}_{arch}.npz',fields=np.array(ef),x=xx,z_m=z)
                        write_csv(out/'post_convergence.csv',rows);write_csv(out/'post_convergence_power.csv',ledger)
                        print('post control',L,w,arch,name,flush=True)
                elif stage=='extra':
                    for mode,rho in [('y',(0,.5)),('diagonal',(.5,.5)),('common',(.5,.5)),('differential',(.25,.25))]:
                        if mode=='common':r1=r2=rho
                        elif mode=='differential':r1=rho;r2=tuple(-v for v in rho)
                        else:r1,r2=(rho,(0,0)) if owner==1 else ((0,0),rho)
                        r=interpanel_route(L,w,r1,r2,architecture=arch,hardware=p,transfer=t,source_cache=cache)
                        rows.append(dict(L=L,w_px=w,architecture=arch,diagnostic=mode,
                                         selected_infidelity=1-spectrum_fidelity(r['coefficients'],ref[0]),**r['metadata']))
                    planes=[]
                    for d in [0,.5]:
                        r1,r2=offsets(d,owner)
                        rr=interpanel_route(L,w,r1,r2,architecture=arch,hardware=p,transfer=t,source_cache=cache,retain_planes=True)
                        planes.append([rr['incoming_slm2'],rr['after_slm2']])
                    np.savez_compressed(out/f'actual_panel_L{L}_w{w}_{arch}.npz',fields=np.array(planes),x=rr['slm2_x'],y=rr['slm2_y'])
                    zs=np.array([.001,.004,.008,.012,.016,.020,.025]);ee=[]
                    for c in ref:
                        ef,xx,ch=propagate_axicon(c,W,zs,params=p,propagation_window_m=.012)
                        ee.append(ef);ledger.extend(dict(L=L,w_px=w,architecture=arch,diagnostic='axial',**q) for q in ch)
                    ar=[]
                    for zz,e,h in zip(zs,*ee):
                        ar.append(dict(L=L,w_px=w,architecture=arch,z_m=zz,baseline_roi_peak=float(np.max(abs(e)**2)),**image_metrics(h,e,xx,L)))
                    write_csv(out/f'axial_L{L}_w{w}_{arch}.csv',ar)
                    np.savez_compressed(out/f'axial_L{L}_w{w}_{arch}.npz',fields=np.array(ee),x=xx,z_m=zs)
                    fine=np.load(out/f'fine_post_L{L}_w{w}_{arch}.npz');phi=np.arange(720)*2*np.pi/720;xt=fine['x']
                    for d,E in zip([0,.5],fine['fields']):
                        for radius in [30e-6,50e-6,80e-6,120e-6]:
                            coords=np.array([(radius*np.sin(phi)-xt[0])/(xt[1]-xt[0]),(radius*np.cos(phi)-xt[0])/(xt[1]-xt[0])])
                            e=map_coordinates(E.real,coords,order=1)+1j*map_coordinates(E.imag,coords,order=1)
                            rows.append(dict(L=L,w_px=w,architecture=arch,diagnostic='scalar E contour winding',offset_px=d,
                                contour_radius_m=radius,winding=float(np.angle(np.roll(e,-1)*np.conj(e)).sum()/(2*np.pi)),
                                contour_min_intensity_over_peak=float(np.min(abs(e)**2)/np.max(abs(E)**2))))
                    if L==20 and w==250:
                        for distance in [.15,.25]:
                            cc=[]
                            for d in [0,.5]:
                                r1,r2=offsets(d,owner)
                                rr=interpanel_route(L,w,r1,r2,architecture=arch,hardware=p,
                                    transfer=replace(t,distance_m=distance),source_cache=cache)
                                cc.append(rr['coefficients'])
                            rows.append(dict(L=L,w_px=w,architecture=arch,diagnostic='illustrative distance scenario',
                                distance_m=distance,selected_infidelity=1-spectrum_fidelity(cc[1],cc[0])))
                    write_csv(out/'additional_diagnostics.csv',rows);write_csv(out/'additional_power.csv',ledger)
                    print('extra',L,w,arch,flush=True)
    paths=[Path(__file__),ROOT/'vbb_study/digital_twin/registration_interpanel.py',ROOT/'vbb_study/digital_twin/registration_reference.py']
    manifest=dict(stage=stage,transfer=asdict(t),hardware=asdict(ReferenceParameters()),
        geometry='conditional approximate user-reported 200 mm free space; aligned same-sign x carriers',
        mapping_mode='fixed_physical_optics',relay_location='after BOTH SLMs; ideal unit-magnification unfolded image',
        correction='FLAT-CORRECTION PHYSICS-ISOLATION BASELINE',physical_interpanel_iris=False,
        iris='adapted per L,w; fixed across states and architectures; actual order purity uncalibrated',
        axicon='conditional ideal 20 degree BASE angle cone, index 1.458; exact optic and axicon plane unmeasured',
        solver_scope='scalar broad morphology only; not vector focal detail',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_sha256={str(q.relative_to(ROOT)):hashlib.sha256(q.read_bytes()).hexdigest() for q in paths})
    (out/f'manifest_{stage}.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--stage',required=True,choices=['local','sweep','controls','postcontrols','extra'])
    a.add_argument('--output',type=Path,default=ROOT/'outputs/validation/registration_interpanel_200mm')
    args=a.parse_args();run(args.output,args.stage)
