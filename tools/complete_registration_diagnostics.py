#!/usr/bin/env python3
"""Derive topology, radial and axial checks from the retained complex fields."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from vbb_study.digital_twin.slm_registration_metrics import _sample_ring,radial_profile
from tools.run_registration_reference_study import grid,write_csv

def run(p):
    f=np.load(p/'fields.npz');rows=[]
    for w in [50,250]:
        for L in [10,20]:
            x=f[f'w{w}_L{L}_post_fine_x'];g=grid(x)
            for d in [0,.125,.25,.375,.5]:
                E=f[f'w{w}_L{L}_d{d:g}_post_fine'];I=abs(E)**2
                r,prof=radial_profile(I,g,n_bins=120,r_max_m=100e-6);k=int(np.argmax(prof));rr=r[k]
                ring=_sample_ring(E,g,rr,1024)
                winding=np.sum(np.angle(np.roll(ring,-1)*ring.conj()))/(2*np.pi)
                left=k;right=k
                while left>0 and prof[left-1]>=prof[k]/2:left-=1
                while right<len(prof)-1 and prof[right+1]>=prof[k]/2:right+=1
                width=r[right]-r[left]+r[1]-r[0];R=np.hypot(g['X'],g['Y'])
                rows.append(dict(w_px=w,L=L,offset_px=d,winding=float(winding),contour_radius_m=float(rr),
                    radial_fwhm_m=float(width),core_to_peak=float(np.mean(I[R<.3*rr])/I.max()) if np.any(R<.3*rr) else float('nan'),
                    energy_inside_first_ring_fraction=float(I[R<rr+width].sum()/I.sum()),
                    energy_normalisation='retained +/-150 um ROI, not total propagated power',
                    radial_width_status='sampled diagnostic, no sub-grid fit or calibrated dimension'))
    write_csv(p/'topology_and_ring.csv',rows)
    a=pd.read_csv(p/'axial.csv');zones=[];checks=pd.read_csv(p/'propagation_checks.csv')
    for (w,L,d),b in a.groupby(['w_px','L','offset_px']):
        peak=b.roi_peak.to_numpy();zz=b.z_m.to_numpy();idx=int(np.argmax(peak));left=idx;right=idx
        while left>0 and peak[left-1]>=peak.max()/2:left-=1
        while right<len(peak)-1 and peak[right+1]>=peak.max()/2:right+=1
        ch=checks[(checks.w_px==w)&(checks.L==L)&(checks.offset_px==d)]
        zones.append(dict(w_px=w,L=L,offset_px=d,z_peak_m=float(zz[idx]),
            start_sample_m=float(zz[left]),end_sample_m=float(zz[right]),
            sample_bracket_length_m=float(zz[right]-zz[left]),z_step_m=float(zz[1]-zz[0]),
            boundary_censored=left==0 or right==len(zz)-1,
            max_boundary_power_fraction=float(ch.boundary_power_fraction.max()),
            definition='connected half-maximum interval of ROI transverse peak; not propagation invariance proof'))
    write_csv(p/'axial_zones.csv',zones)
    # Every derived item must remain traceable to the source complex fields.
    m=json.loads((p/'manifest.json').read_text())
    m['files']={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in p.iterdir() if f.is_file() and f.name!='manifest.json'}
    (p/'manifest.json').write_text(json.dumps(m,indent=2))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--data',type=Path,default=Path('outputs/validation/registration_iris_adjusted'));run(a.parse_args().data)
