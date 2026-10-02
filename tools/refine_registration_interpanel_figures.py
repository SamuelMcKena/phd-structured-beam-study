#!/usr/bin/env python3
"""Refine all five diagnostic XY states at 1 um without rerunning panel transfer."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from tools.run_registration_interpanel_study import CASES,ARCHS,parameters,filename
from tools.run_registration_reference_study import image_metrics,write_csv
from vbb_study.digital_twin.registration_reference import propagate_axicon,panel_phase

def export_immediate_slm1(data):
    for L,w in CASES:
        for arch in ARCHS:
            panel=1 if arch=='upstream_vortex' else 2
            p=parameters(L,w);xi=(np.arange(1500)-750+.5)*p.pitch_m/2
            X,Y=np.meshgrid(xi,xi);A=np.sqrt(p.fill_factor)*np.exp(-(X*X+Y*Y)/(w*p.pitch_m)**2)
            immediate=[]
            for d in [0,.5]:
                rho=(d,0) if panel==1 else (0,0)
                immediate.append((A*np.exp(1j*panel_phase(X,Y,L if panel==1 else 0,rho,p))).astype(np.complex64))
            np.savez_compressed(data/f'immediate_slm1_L{L}_w{w}_{arch}.npz',fields=np.array(immediate),x=xi,
                vortex_owner_offset_px=[0,.5],dx_m=p.pitch_m/2,description='Fixed physical coordinates; exact pointwise pixel phase and Gaussian amplitude')

def run(data):
    export_immediate_slm1(data)
    rows=[];ledger=[]
    for L,w in CASES:
        for arch in ARCHS:
            panel=1 if arch=='upstream_vortex' else 2
            source=data/filename(L,w,arch,panel);a=np.load(source);fine=np.load(data/f'fine_post_L{L}_w{w}_{arch}.npz')
            E=[]
            for i,d in enumerate(a['offsets']):
                if i in [0,4]:e=fine['fields'][0 if i==0 else 1];x=fine['x']
                else:
                    fields,x,checks=propagate_axicon(a['coefficients'][i],float(a['window_m']),[float(a['z_m'])],
                        params=parameters(L,w),propagation_window_m=.008,dx_target=1e-6)
                    e=fields[0];ledger.extend(dict(L=L,w_px=w,architecture=arch,offset_px=float(d),**c) for c in checks)
                E.append(e)
                rows.append(dict(L=L,w_px=w,architecture=arch,offset_px=float(d),z_m=float(a['z_m']),
                    **image_metrics(e,E[0],x,L)))
            np.savez_compressed(data/f'fine_sequence_L{L}_w{w}_{arch}.npz',fields=np.array(E),x=x,z_m=a['z_m'],offsets=a['offsets'])
            write_csv(data/'fine_sequence_metrics.csv',rows);write_csv(data/'fine_sequence_power.csv',ledger)
            print('fine sequence',L,w,arch,flush=True)
    inputs=sorted(data.glob('fields_*.npz'))+sorted(data.glob('fine_post_*.npz'))
    hashes={}
    for p in inputs:
        with p.open('rb') as f:hashes[p.name]=hashlib.file_digest(f,'sha256').hexdigest()
    (data/'manifest_fine_sequences.json').write_text(json.dumps(dict(dx_target_m=1e-6,propagation_window_m=.008,
        crop_width_m=.0003,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_sha256=hashes,scope='same physical planes and panel coefficients; only downstream sampling refined'),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);a=p.parse_args();run(a.data)
