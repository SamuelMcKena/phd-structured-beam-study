#!/usr/bin/env python3
"""Audit physical input radii separately from the downstream axicon ring scale.

Reuses independently simulated 50/250-pixel fields, without image resizing or
changing propagation. Fitted input width avoids bias from the saved XY crop.
"""
import argparse, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import jnp_zeros
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from vbb_study.digital_twin.registration_reference import ReferenceParameters
from tools.plot_registration_interpanel_study import save


def fit_gaussian_radius(x, field):
    """Fit I(x, y_nearest_zero) = const exp(-2 x²/w²) inside half peak.

    A finite saved crop biases a second moment for the large beam. This
    log-quadratic fit recovers the stated Gaussian parameter without claiming
    that a cropped second moment is a full-field width.
    """
    x=np.asarray(x); I=abs(field[len(x)//2])**2
    use=I>I.max()*.5
    slope,_=np.polyfit(x[use]**2,np.log(I[use]),1)
    if slope>=0:raise ValueError('not a decaying centred Gaussian')
    return float(np.sqrt(-2/slope))


def run(data, out):
    out.mkdir(parents=True,exist_ok=True)
    p=ReferenceParameters();rows=[];inputs={}
    for w in [50,250]:
        name=f'immediate_slm1_L20_w{w}_upstream_vortex.npz'
        a=np.load(data/name);x=a['x'];fitted=fit_gaussian_radius(x,a['fields'][0])
        expected=w*p.pitch_m
        if abs(fitted/expected-1)>1e-5:raise AssertionError('input beam radius disagrees with model')
        invariant=float(np.max(abs(abs(a['fields'][1])**2-abs(a['fields'][0])**2))/np.max(abs(a['fields'][0])**2))
        if invariant>1e-6:raise AssertionError('phase-only registration altered incident intensity')
        rows.append(dict(w_px=w,expected_radius_m=expected,fitted_radius_m=fitted,
            input_intensity_FWHM_diameter_m=fitted*np.sqrt(2*np.log(2)),
            immediate_intensity_relative_max_difference=invariant))
        inputs[name]=hashlib.sha256((data/name).read_bytes()).hexdigest()
    if abs(rows[1]['fitted_radius_m']/rows[0]['fitted_radius_m']-5)>1e-4:
        raise AssertionError('input radii are not fivefold different')
    pd.DataFrame(rows).to_csv(out/'beam_scale_audit.csv',index=False)
    plt.rcParams.update({'font.size':14,'axes.labelsize':14,'savefig.dpi':300})
    fig,axs=plt.subplots(1,3,figsize=(12,4.6),layout='constrained')
    for ax,w in zip(axs[:2],[50,250]):
        a=np.load(data/f'immediate_slm1_L20_w{w}_upstream_vortex.npz');x=a['x']*1e3
        I=abs(a['fields'][0])**2;ax.imshow(I/I.max(),origin='lower',extent=[x[0],x[-1],x[0],x[-1]],cmap='inferno',vmin=0,vmax=1,interpolation='bilinear')
        ax.add_patch(Circle((0,0),w*p.pitch_m*1e3,fill=False,color='cyan',lw=1.5))
        ax.set(xlabel='x on SLM1 (mm)',ylabel='y on SLM1 (mm)',title=f'{w} px radius = {w*p.pitch_m*1e3:g} mm')
        xx=a['x'];I=abs(a['fields'][0][len(xx)//2])**2
        axs[2].plot(xx*1e3,I/I.max(),label=f'{w} px',lw=2)
    axs[2].axhline(np.exp(-2),ls=':',color='grey',label='1/e² intensity')
    axs[2].set(xlabel='x on SLM1 (mm)',ylabel='Intensity / own peak',title='Same physical x axis',xlim=(-3,3));axs[2].legend();axs[2].grid(alpha=.2)
    fig.suptitle('Actual input footprints | L = 20 | Equal millimetre axes\nCyan circle: 1/e² intensity radius; phase-only vortex leaves this intensity unchanged',fontsize=16)
    save(fig,out/'01_input_radius_verified.png')
    for plane in ['pre','post']:
        arrays=[np.load(data/(f'fields_L20_w{w}_upstream_vortex_panel1.npz' if plane=='pre' else f'fine_post_L20_w{w}_upstream_vortex.npz')) for w in [50,250]]
        vals=[abs(a['pre'][[0,-1]] if plane=='pre' else a['fields'])**2 for a in arrays]
        residual=[(v[1]-v[0])/v[0].max() for v in vals];limit=max(np.max(abs(r)) for r in residual)
        fig,axs=plt.subplots(2,3,figsize=(11,7.4),layout='constrained')
        for i,(a,v,r,w) in enumerate(zip(arrays,vals,residual,[50,250])):
            factor=1e3 if plane=='pre' else 1e6;unit='mm' if plane=='pre' else 'µm'
            x=a['pre_x' if plane=='pre' else 'x']*factor
            for j in range(3):
                im=axs[i,j].imshow(r if j==2 else v[j]/v[0].max(),origin='lower',extent=[x[0],x[-1],x[0],x[-1]],
                    cmap='RdBu_r' if j==2 else 'inferno',vmin=-limit if j==2 else 0,vmax=limit if j==2 else 1.05,interpolation='bilinear')
                axs[i,j].set_title(['Zero registration','Half-pixel x shift','Signed difference'][j])
                axs[i,j].set_xlabel(f'x at this plane ({unit})')
                axs[i,j].set_ylabel(f'SLM input w = {w*p.pitch_m*1e3:g} mm\ny ({unit})')
            fig.colorbar(im,ax=list(axs[i,:]),shrink=.8,pad=.02,label='Difference / row baseline peak')
        where='BEFORE AXICON | millimetre-scale selected field' if plane=='pre' else '12 mm BEYOND AXICON | micrometre-scale central ring crop'
        fig.suptitle(where+'\nL = 20; A; both blazes; flat correction; conditional 200 mm gap',fontsize=16)
        save(fig,out/('02_before_axicon.png' if plane=='pre' else '03_downstream_ring_crop.png'))
    manifest=dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),input_sha256=inputs,
        ideal_bessel_first_ring_radius_m=float(jnp_zeros(20,1)[0]/p.kr),kr_m_inverse=p.kr,
        ideal_relation='I(r) proportional to J_L(kr*r)^2; first ring r=jprime_L1/kr',
        interpretation='Input Gaussian radius and propagated Bessel central ring radius are distinct; no cosmetic resizing',
        propagation='Existing independent fields, exact scalar ASM; refined dx = 8 mm / 8064 = 0.992063 um',
        scope='Audits scale and re-plots existing evidence; no new architecture-ranking claim')
    (out/'beam_scale_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(pd.DataFrame(rows).to_string(index=False));print('Ideal L=20 ring radius (um):',manifest['ideal_bessel_first_ring_radius_m']*1e6)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();run(args.data,args.out)
