#!/usr/bin/env python3
"""Publication XY maps with shared crops, reference-normalised scales and residuals."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
plt.rcParams.update({'font.size':16,'axes.titlesize':16,'axes.labelsize':15,'savefig.dpi':300})

def save(fig,path):fig.savefig(path,bbox_inches='tight',facecolor='white');plt.close(fig)

def run(data,out):
    out.mkdir(parents=True,exist_ok=True)
    from tools.plot_registration_reference_study import geometry
    geometry(out,architectures=False)
    fig,axs=plt.subplots(2,1,figsize=(10,5),layout='constrained')
    for ax,arch in zip(axs,['A','B']):
        ax.set_xlim(0,10);ax.set_ylim(0,2);ax.axis('off')
        texts=['SLM1\nVortex + blaze','SLM2\nFlat correction + blaze'] if arch=='A' else ['SLM1\nFlat correction + blaze','SLM2\nVortex + blaze']
        for x,label in zip([.3,4.7,7.8],texts+['4F\nOrder selection']):
            ax.add_patch(Rectangle((x,.55),1.9,1.05,facecolor='#eef2f6',edgecolor='#334155'))
            ax.text(x+.95,1.075,label,ha='center',va='center',fontsize=12)
        ax.annotate('',(4.6,1.1),(2.3,1.1),arrowprops=dict(arrowstyle='->',lw=2))
        ax.text(3.4,1.65,'~200 mm free space',ha='center',fontsize=12)
        ax.annotate('',(7.7,1.1),(6.7,1.1),arrowprops=dict(arrowstyle='->',lw=2))
        ax.text(.3,1.85,'Architecture '+arch,fontweight='bold')
    save(fig,out/'01_geometry_200mm.png')
    for path in sorted(data.glob('fields_*.npz')):
        a=np.load(path);stem=path.stem;panel=int(stem[-1]);arch='A' if 'upstream' in stem else 'B'
        if panel!=(1 if arch=='A' else 2):continue
        L=int(stem.split('_')[1][1:]);w=int(stem.split('_')[2][1:])
        for plane in ['pre','post']:
            x=a[plane+'_x'];I=abs(a[plane])**2;ref=I[0];peak=ref.max();factor=1e3 if plane=='pre' else 1e6
            unit='mm' if plane=='pre' else r'$\mu$m'
            for residual in [False,True]:
                vals=(I-ref)/peak if residual else I/peak;limit=max(abs(vals.min()),abs(vals.max())) if residual else max(1.,vals.max())
                fig,axs=plt.subplots(2,3,figsize=(10,6.8),layout='constrained')
                for i,ax in enumerate(axs.flat[:5]):
                    im=ax.imshow(vals[i],origin='lower',extent=[x[0]*factor,x[-1]*factor,x[0]*factor,x[-1]*factor],
                        cmap='RdBu_r' if residual else 'inferno',vmin=-limit if residual else 0,vmax=limit,interpolation='bilinear')
                    ax.set_title(r'$\Delta x = '+f'{a["offsets"][i]:g}'+r'\,p$');ax.set_xlabel('x ('+unit+')');ax.set_ylabel('y ('+unit+')')
                axs[1,2].axis('off');axs[1,2].text(.02,.85,'Both SLMs blazed\n20 pixels per period\n\n8 µm lattice\nFlat correction baseline\n\nConditional 200 mm gap\nShared crop and scale',va='top',fontsize=12,linespacing=1.5)
                fig.colorbar(im,ax=list(axs.flat[:5]),shrink=.8,pad=.02,label='Signed residual / reference peak' if residual else 'Intensity / reference peak')
                where='Before physical axicon' if plane=='pre' else f'Propagated {1000*float(a["z_m"]):g} mm downstream of axicon'
                fig.suptitle(f'{"STRESS TEST" if w==50 else "BENCH RADIUS"}  |  L = {L}, w = {w} px  |  Architecture {arch}\n'+where,fontsize=16)
                save(fig,out/(stem+'_'+plane+('_residual' if residual else '_intensity')+'.png'))
    m=pd.read_csv(data/'metrics.csv');fig,axs=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
    for ax,w in zip(axs,[50,250]):
        for (L,arch,panel),g in m[m.w_px==w].groupby(['L','architecture','panel']):
            owner=panel==(1 if arch=='upstream_vortex' else 2)
            ax.semilogy(g.offset_px.iloc[1:],g.selected_infidelity.iloc[1:],marker='o',ls='-' if owner else '--',
                label=f'L={L}, {"A" if arch=="upstream_vortex" else "B"}, '+('vortex panel' if owner else 'flat panel'))
        ax.set_xlabel(r'Registration $\Delta x/p$');ax.set_ylabel('Selected-field infidelity');ax.grid(alpha=.25)
        ax.set_title('Stress radius 50 px' if w==50 else 'Bench radius 250 px');ax.legend(fontsize=9)
    save(fig,out/'02_sensitivity.png')
    # Presentation comparison uses SAME architecture, SAME propagated distance
    # (12 mm), SAME crop, and a SHARED signed residual scale across beam sizes.
    paths=[data/f'fine_post_L20_w{w}_upstream_vortex.npz' for w in [50,250]]
    if all(p.exists() for p in paths):
        maps=[np.load(p) for p in paths];vals=[abs(a['fields'])**2 for a in maps]
        res=[(v[1]-v[0])/v[0].max() for v in vals];lim=max(np.max(abs(r)) for r in res)
        fig,axs=plt.subplots(2,3,figsize=(11,7.6),layout='constrained')
        for i,(a,v,r,w) in enumerate(zip(maps,vals,res,[50,250])):
            x=a['x']*1e6
            for j in range(3):
                image=r if j==2 else v[j]/v[0].max()
                im=axs[i,j].imshow(image,origin='lower',extent=[x[0],x[-1],x[0],x[-1]],interpolation='bilinear',
                    cmap='RdBu_r' if j==2 else 'inferno',vmin=-lim if j==2 else 0,vmax=lim if j==2 else 1.05)
                axs[i,j].set_title(['Zero registration','Half-pixel shift','Signed residual'][j]);axs[i,j].set_xlabel(r'x ($\mu$m)')
                axs[i,j].set_ylabel(('Stress 50 px' if w==50 else 'Bench 250 px')+'\n'+r'y ($\mu$m)')
            fig.colorbar(im,ax=list(axs[i,:]),shrink=.8,pad=.02,label='Residual / row reference peak')
        fig.suptitle('L = 20  |  Architecture A  |  12 mm downstream of axicon\nConditional 200 mm gap; flat correction; both SLMs blazed',fontsize=16)
        save(fig,out/'presentation_01_stress_vs_bench_200mm.png')
    fig,axs=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for (L,arch),g in m[(m.w_px==250)&(m.role=='vortex + blaze')].groupby(['L','architecture']):
        label=f'L={L}, {"A" if arch=="upstream_vortex" else "B"}, z={1000*g.z_m.iloc[0]:g} mm'
        axs[0].plot(g.offset_px,100*g.selected_infidelity,'o-',label=label)
        axs[1].plot(g.offset_px,100*g.post_power_normalised_residual_l2,'o-',label=label)
    for ax in axs:ax.set_xlabel(r'Registration $\Delta x/p$');ax.grid(alpha=.25);ax.legend(fontsize=10)
    axs[0].set_ylabel('Selected-field infidelity (%)');axs[1].set_ylabel('Propagated shape residual (%)')
    fig.suptitle('Bench radius 2 mm  |  Conditional 200 mm gap  |  Flat correction',fontsize=16)
    save(fig,out/'presentation_02_bench_sensitivity_200mm.png')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.data,a.output)
