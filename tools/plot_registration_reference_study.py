#!/usr/bin/env python3
"""Publication figures from the independently audited numerical evidence."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Circle,FancyArrowPatch
from scipy.stats import linregress

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'outputs/validation/registration_iris_adjusted'
OUT=ROOT/'outputs/figures/registration_definitive'
plt.rcParams.update({'font.size':14,'axes.titlesize':16,'axes.labelsize':14,
                    'xtick.labelsize':12,'ytick.labelsize':12,'legend.fontsize':12,
                    'savefig.dpi':300,'font.family':'DejaVu Sans','axes.spines.top':False,
                    'axes.spines.right':False,'figure.facecolor':'white'})
COLORS={25:'#803da4',50:'#d34a32',100:'#e39723',150:'#247b65',250:'#2864a0'}


def save(fig,path):
    fig.savefig(path.with_suffix('.png'),bbox_inches='tight',facecolor='white')
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight',facecolor='white')
    plt.close(fig)


def geometry(out):
    fig,axs=plt.subplots(2,4,figsize=(13.5,7.3),layout='constrained')
    cases=[(0,0),(.125,0),(.25,0),(.375,0),(.5,0),(0,.5),(.5,.5)]
    for ax,(x,y) in zip(axs.flat,cases):
        # This is a local magnification, not a tiny Gaussian footprint.
        ax.set_facecolor('#edf5fc')
        for a in np.arange(-3,4)-x: ax.axvline(a,color='#4f6174',lw=1.5)
        for a in np.arange(-3,4)-y: ax.axhline(a,color='#4f6174',lw=1.5)
        ax.add_patch(Circle((0,0),.23,fill=False,color='#b33436',lw=2.5))
        ax.plot(0,0,'+',ms=17,mew=2.7,color='#b33436')
        ax.set(xlim=(-1.4,1.4),ylim=(-1.4,1.4),aspect='equal',xlabel=r'$x/p$',ylabel=r'$y/p$')
        ax.set_title(rf'$\rho_x={x:g},\ \rho_y={y:g}$')
        ax.set_xticks([-1,0,1]);ax.set_yticks([-1,0,1])
        if x or y: ax.annotate('',xy=(-.8-x,-.9-y),xytext=(-.8,-.9),arrowprops={'arrowstyle':'->','color':'#2864a0','lw':2})
    ax=axs.flat[-1];ax.set_axis_off()
    ax.text(0,.93,'Fixed beam and hologram',weight='bold',fontsize=16)
    ax.text(0,.78,r'$b=h=0$ in every panel',fontsize=16)
    ax.text(0,.58,'The physical grid moves\nby −ρp underneath them.',linespacing=1.6)
    ax.text(0,.32,'Red: common optical centre\nBlue: illuminated footprint\nGrey: physical pixel boundaries',linespacing=1.6)
    ax.text(0,.05,'Central few pixels of a broad beam;\nfootprint edge is outside this zoom.',fontsize=11)
    save(fig,out/'01_registration_geometry')
    fig,ax=plt.subplots(figsize=(12.8,5.4));ax.set(xlim=(0,13),ylim=(0,5));ax.axis('off')
    for y,label,texts in [(3.6,'A — upstream vortex',['Vortex + blaze','Correction + blaze']),
                           (1.2,'B — downstream vortex',['Correction + blaze','Vortex + blaze'])]:
        ax.text(.1,y+.75,label,weight='bold',fontsize=17)
        for x,name,content in [(1,'SLM1',texts[0]),(4.5,'SLM2',texts[1]),(7.7,'Common 4F','Order selection'),(10.5,'Axicon','Propagate to z > 0')]:
            ax.add_patch(Rectangle((x,y-.5),2.35,1.1,facecolor='#eff3f7',edgecolor='#4f6174',lw=1.5))
            ax.text(x+1.175,y+.22,name,ha='center',weight='bold',fontsize=15)
            ax.text(x+1.175,y-.16,content,ha='center',fontsize=12)
        for x0,x1 in [(3.35,4.5),(6.85,7.7),(10.05,10.5)]:
            ax.annotate('',xy=(x1,y),xytext=(x0,y),arrowprops={'arrowstyle':'->','lw':1.7})
        ax.text(3.9,y-.8,r'$T_{12}$ unknown',ha='center',fontsize=12)
    ax.text(.1,.05,'Both panels: 20-pixel blaze. Flat correction in this study. Distances are not to scale.',fontsize=12)
    save(fig,out/'02_architectures')


def xy_panels(f,w,plane,residual,out,adapted=False):
    L=20;ds=[0,.125,.25,.375,.5];kind='post_fine' if plane=='post' else 'pre'
    x=f[f'w{w}_L{L}_{"post_fine_x" if plane=="post" else "pre_x"}']*1e6
    Es=[f[f'w{w}_L{L}_d{d:g}_{kind}'] for d in ds]
    # Match physical crop and absolute normalisation throughout each sequence.
    half=(70 if w==250 else 100) if plane=='post' else (2600 if w==250 else 900)
    ids=np.flatnonzero(abs(x)<=half);xc=x[ids];Is=[abs(E[np.ix_(ids,ids)])**2 for E in Es]
    peak=float(Is[0].max());arrs=[(I-Is[0])/peak if residual else I/peak for I in Is]
    vmax=max(float(np.max(abs(a))) for a in arrs) if residual else max(float(np.max(a)) for a in arrs)
    if residual: vmax=max(vmax,1e-7)
    fig=plt.figure(figsize=(11.8,7.6),layout='constrained');gs=fig.add_gridspec(2,3)
    for j,(d,a) in enumerate(zip(ds,arrs)):
        ax=fig.add_subplot(gs[j//3,j%3])
        im=ax.imshow(a,origin='lower',extent=[xc[0],xc[-1],xc[0],xc[-1]],
                     cmap='RdBu_r' if residual else 'inferno',vmin=-vmax if residual else 0,
                     vmax=vmax,interpolation='bilinear',aspect='equal')
        ax.set_title(rf'$\Delta x={d:g}\,p$');ax.set_xlabel('x / μm');ax.set_ylabel('y / μm')
        ax.tick_params(labelsize=12)
    ax=fig.add_subplot(gs[1,2]);ax.axis('off')
    label='BENCH SIZE' if w==250 else ('REGISTRATION STRESS CASE' if adapted else 'FILTERING STRESS CASE')
    ax.text(0,.9,label,color=COLORS[w],weight='bold',fontsize=15)
    ax.text(0,.76,rf'$L=20,\ w={w}\ \mathrm{{px}}$',fontsize=18)
    if plane=='post':
        ax.text(0,.58,f'Propagated z = {float(f[f"w{w}_L20_zstar"])*1e3:.3f} mm\nConditional 20° cone model',linespacing=1.5,fontsize=13)
    else: ax.text(0,.58,'Immediately before axicon\nAfter ideal 4F order selection',linespacing=1.5,fontsize=13)
    ax.text(0,.29,'Scale fixed to the zero-shift peak.\nNo per-panel renormalisation.',linespacing=1.5,fontsize=12)
    if w==50:
        ax.text(0,.06,'Iris adjusted for the vortex.\nSame iris at every offset.' if adapted else 'The intended L = 20 vortex is\nalready rejected by this iris.',color=COLORS[w],fontsize=12)
    cb=fig.colorbar(im,ax=fig.axes[:5],fraction=.024,pad=.025)
    cb.set_label(r'$(I-I_0)/I_{0,\max}$' if residual else r'$I/I_{0,\max}$',fontsize=15)
    name=f'{"stress" if w==50 else "bench"}_{plane}_{"residual" if residual else "intensity"}'
    save(fig,out/name)


def quantitative(data,out):
    d=pd.read_csv(data/'sweep.csv');q=d[(d.offset_px==.5)&(d.axis=='x')]
    fig,axs=plt.subplots(1,2,figsize=(12.3,4.8),layout='constrained')
    for w,a in q.groupby('w_px'):
        for ax,key in zip(axs,['local_infidelity','raw_infidelity']):
            ax.semilogy(a.L,np.maximum(a[key],1e-12),'o-',color=COLORS[w],label=f'w = {w} px',lw=2)
    axs[0].set_title('Immediately after the two SLMs')
    axs[1].set_title('After order selection / before axicon')
    for ax in axs:ax.set(xlabel='Topological charge L',ylabel='Raw complex-field infidelity');ax.grid(alpha=.2);ax.set_xticks([0,1,3,5,10,20])
    axs[0].legend(ncol=2);save(fig,out/'03_charge_and_beam_size')
    fig,axs=plt.subplots(1,2,figsize=(12.3,4.8),layout='constrained')
    for axis,ls in [('x','-'),('y','--'),('diagonal',':')]:
        a=d[(d.L==20)&(d.offset_px==.5)&(d.axis==axis)]
        axs[0].semilogy(a.w_px,a.raw_infidelity,'o'+ls,lw=2,label=axis)
        axs[1].plot(a.w_px,a.power_ratio,'o'+ls,lw=2,label=axis)
    axs[0].set(xlabel='Beam radius / pixels',ylabel='Selected-field infidelity',title='Direction depends on the carrier')
    axs[1].set(xlabel='Beam radius / pixels',ylabel='Selected power / zero-shift power',title='Power loss and shape change differ')
    for ax in axs:ax.grid(alpha=.2);ax.legend();ax.axvline(250,color='#2864a0',alpha=.25,lw=5)
    save(fig,out/'04_direction_and_power')
    u=pd.read_csv(data/'unit_cell.csv');fig,axs=plt.subplots(2,2,figsize=(10,8),layout='constrained')
    for col,w in enumerate([50,250]):
        a=u[u.w_px==w]
        for row,key in enumerate(['raw_infidelity','power_ratio']):
            z=a.pivot(index='rho_y',columns='rho_x',values=key).to_numpy()
            im=axs[row,col].imshow(z,origin='lower',extent=(-.0625,.9375,-.0625,.9375),cmap='magma' if row==0 else 'viridis',interpolation='nearest')
            axs[row,col].set(xlabel=r'$\rho_x$ / pixels',ylabel=r'$\rho_y$ / pixels',title=f'L = 20, w = {w} px')
            axs[row,col].set_xticks([0,.25,.5,.75]);axs[row,col].set_yticks([0,.25,.5,.75])
            fig.colorbar(im,ax=axs[row,col],label='Infidelity' if row==0 else 'Power ratio',fraction=.046)
    save(fig,out/'05_unit_cell')
    s=pd.read_csv(data/'no_carrier_scaling.csv');s=s[s.L>0].copy()
    fit=linregress(np.log(s.eta),np.log(s.infidelity))
    result=dict(prefactor=float(np.exp(fit.intercept)),exponent=float(fit.slope),log_R_squared=float(fit.rvalue**2))
    (data/'scaling_fit.json').write_text(json.dumps(result,indent=2))
    fig,axs=plt.subplots(1,2,figsize=(12.3,4.8),layout='constrained')
    for w,a in s.groupby('w_px'):axs[0].loglog(a.eta,a.infidelity,'o',color=COLORS[w],label=f'{w} px')
    xx=np.geomspace(s.eta.min(),s.eta.max(),100);axs[0].loglog(xx,np.exp(fit.intercept)*xx**fit.slope,'k--',lw=1.5,label=f'Fit exponent {fit.slope:.2f}')
    axs[0].set(xlabel=r'$\eta=L p/w=L/w_{\mathrm{px}}$',ylabel='Local infidelity without carrier',title='Vortex-only scaling diagnostic');axs[0].legend(ncol=2)
    for w,a in q[q.L>0].groupby('w_px'):axs[1].loglog(a.eta,a.raw_infidelity,'o-',color=COLORS[w],label=f'{w} px')
    axs[1].set(xlabel=r'$\eta=L/w_{\mathrm{px}}$',ylabel='Selected-field infidelity',title='Two-blaze filtered result does not collapse')
    for ax in axs:ax.grid(alpha=.2,which='both')
    save(fig,out/'06_scaling')
    c=pd.read_csv(data/'convergence.csv');fig,axs=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
    for (L,w),a in c[c.axis=='x'].groupby(['L','w_px']):
        axs[0].plot(a.q,a.raw_infidelity,'o-',label=f'L={L}, w={w} px')
        axs[1].plot(a.q,a.power_ratio,'o-',label=f'L={L}, w={w} px')
    axs[0].set_yscale('log');axs[0].set_ylabel('Selected-field infidelity');axs[1].set_ylabel('Selected power ratio')
    for ax in axs:ax.set(xlabel='Illumination subdivisions per interval',xticks=[1,2,4]);ax.grid(alpha=.2)
    axs[1].legend();save(fig,out/'07_convergence')
    a=pd.read_csv(data/'axial.csv');fig,axs=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for ax,w in zip(axs,[50,250]):
        for L,color in [(10,'#2864a0'),(20,'#d34a32')]:
            b=a[(a.w_px==w)&(a.L==L)];norm=b[b.offset_px==0].roi_peak.max()
            for d,ls in [(0,'-'),(.5,'--')]:
                cc=b[b.offset_px==d];ax.plot(cc.z_m*1e3,cc.roi_peak/norm,ls,color=color,lw=2,label=f'L={L}, Δx={d:g}p')
        ax.set(title=f'w = {w} px',xlabel='Distance downstream of axicon / mm',ylabel='ROI peak / zero-shift axial maximum');ax.grid(alpha=.2);ax.legend(fontsize=10)
    save(fig,out/'08_axial_evolution')
    roles=pd.read_csv(data/'roles.csv');fig,axs=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for ax,w in zip(axs,[50,250]):
        b=roles[(roles.L==20)&(roles.w_px==w)&(roles.axis=='x')]
        for mode,cc in b.groupby('mode'): ax.semilogy(cc.offset_px,np.maximum(cc.raw_infidelity,1e-12),'o-',lw=2,label=mode.replace('_',' '))
        ax.set(title=f'L = 20, w = {w} px',xlabel='x registration / pixels',ylabel='Selected-field infidelity');ax.grid(alpha=.2);ax.legend(fontsize=10)
    save(fig,out/'09_panel_roles')
    b=pd.read_csv(data/'post_fine_metrics.csv');fig,axs=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for L,color in [(10,'#2864a0'),(20,'#d34a32')]:
        c=b[(b.w_px==250)&(b.L==L)]
        axs[0].plot(c.offset_px,(c.peak_ratio-1)*100,'o-',color=color,lw=2,label=f'L = {L}')
        axs[1].plot(c.offset_px,c.power_normalised_residual_l2*100,'o-',color=color,lw=2,label=f'L = {L}')
    axs[0].set(xlabel='x registration / pixels',ylabel='Propagated peak change / %',title='Bench size: small intensity change')
    axs[1].set(xlabel='x registration / pixels',ylabel='Power-normalised intensity residual / %',title='Bench size: small shape change')
    for ax in axs:ax.grid(alpha=.2);ax.legend()
    save(fig,out/'10_bench_answer')


def presentation(f,out,adapted=False):
    ds=[0,.375,.5];fig,axs=plt.subplots(2,3,figsize=(13.4,8.1),layout='constrained')
    for row,w in enumerate([50,250]):
        x=f[f'w{w}_L20_post_fine_x']*1e6;half=70;ids=np.flatnonzero(abs(x)<=half);xx=x[ids]
        Is=[abs(f[f'w{w}_L20_d{d:g}_post_fine'][np.ix_(ids,ids)])**2 for d in ds]
        norm=float(Is[0].max())
        for col,(d,I) in enumerate(zip(ds,Is)):
            ax=axs[row,col];im=ax.imshow(I/norm,origin='lower',extent=[xx[0],xx[-1],xx[0],xx[-1]],cmap='inferno',vmin=0,vmax=1.04,interpolation='bilinear')
            ax.set(title=rf'$\Delta x={d:g}p$',xlabel='x / μm',ylabel='y / μm')
            if col==0: ax.text(.02,.98,f'w = {w} px',transform=ax.transAxes,va='top',color='white',weight='bold')
        fig.colorbar(im,ax=axs[row,:],fraction=.025,pad=.02,label='I / zero-shift peak')
    fig.suptitle('L = 20: registration stress case versus 2 mm beam' if adapted else 'L = 20: filtering stress case versus 2 mm beam',fontsize=22)
    fig.supxlabel('Iris adjusted for each beam size, then held fixed throughout its registration sweep.' if adapted else 'Top: intended vortex already rejected by the iris. Bottom: bench-size conditional cone model.',fontsize=12)
    save(fig,out/'presentation_01_stress_vs_bench')
    # Separate residual figure makes the weak bench changes inspectable.
    fig,axs=plt.subplots(1,3,figsize=(13,4.7),layout='constrained');w=250
    x=f['w250_L20_post_fine_x']*1e6;ids=np.flatnonzero(abs(x)<=70);xx=x[ids]
    I0=abs(f['w250_L20_d0_post_fine'][np.ix_(ids,ids)])**2
    I=abs(f['w250_L20_d0.5_post_fine'][np.ix_(ids,ids)])**2
    arrs=[I0/I0.max(),I/I0.max(),(I-I0)/I0.max()]
    for ax,a,title in zip(axs,arrs,['Zero shift','Half-pixel x shift','Signed intensity residual']):
        residual=title.startswith('Signed');lim=max(abs(a.min()),abs(a.max())) if residual else 1.04
        im=ax.imshow(a,origin='lower',extent=[xx[0],xx[-1],xx[0],xx[-1]],cmap='RdBu_r' if residual else 'inferno',vmin=-lim if residual else 0,vmax=lim,interpolation='bilinear')
        ax.set(title=title,xlabel='x / μm',ylabel='y / μm');fig.colorbar(im,ax=ax,fraction=.046,pad=.03)
    fig.suptitle('L = 20, w = 250 px: robust propagated ring',fontsize=21)
    save(fig,out/'presentation_02_bench_ring')


def extreme_presentation(data,f,out):
    e=np.load(data/'extreme_stress.npz');fig,axs=plt.subplots(2,3,figsize=(13.3,8),layout='constrained')
    for row,(x,Es,label) in enumerate([
        (e['x']*1e6,[e[f'd{d:g}'] for d in [0,.25,.5]],'w = 25 px: severe stress'),
        (f['w250_L20_post_fine_x']*1e6,[f[f'w250_L20_d{d:g}_post_fine'] for d in [0,.25,.5]],'w = 250 px: bench size')]):
        ids=np.flatnonzero(abs(x)<=70);xx=x[ids];Is=[abs(E[np.ix_(ids,ids)])**2 for E in Es];norm=Is[0].max()
        for col,(d,I) in enumerate(zip([0,.25,.5],Is)):
            ax=axs[row,col];im=ax.imshow(I/norm,origin='lower',extent=[xx[0],xx[-1],xx[0],xx[-1]],cmap='inferno',vmin=0,vmax=1.15,interpolation='bilinear')
            ax.set(title=rf'$\Delta x={d:g}p$',xlabel='x / μm',ylabel='y / μm')
        axs[row,0].text(.03,.97,label,transform=axs[row,0].transAxes,color='white',fontsize=12,va='top')
        fig.colorbar(im,ax=axs[row,:],fraction=.026,pad=.02,label='I / zero-shift peak')
    fig.suptitle('L = 20: what substantial registration sensitivity looks like',fontsize=21)
    fig.supxlabel('Adjusted iris, fixed within each sweep. Top beam radius 0.2 mm; bottom 2 mm. Conditional 20° cone.',fontsize=12)
    save(fig,out/'presentation_04_severe_stress_vs_bench')


def run(data,out):
    out.mkdir(parents=True,exist_ok=True);geometry(out);quantitative(data,out)
    adapted=json.loads((data/'manifest.json').read_text()).get('iris_policy')=='adapted'
    with np.load(data/'fields.npz') as f:
        for w in [50,250]:
            for plane in ['pre','post']:
                for residual in [False,True]:xy_panels(f,w,plane,residual,out,adapted)
        presentation(f,out,adapted)
        if adapted and (data/'extreme_stress.npz').exists():extreme_presentation(data,f,out)
    # A concise presentation graph without additional modelling assumptions.
    import shutil
    shutil.copyfile(out/'10_bench_answer.png',out/'presentation_03_bench_graphs.png')


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--data',type=Path,default=DATA);a.add_argument('--out',type=Path,default=OUT);args=a.parse_args();run(args.data,args.out)
