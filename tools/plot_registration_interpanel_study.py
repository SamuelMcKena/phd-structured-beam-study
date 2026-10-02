#!/usr/bin/env python3
"""Publication XY maps with shared crops, reference-normalised scales and residuals."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
plt.rcParams.update({'font.size':16,'axes.titlesize':16,'axes.labelsize':15,'savefig.dpi':300})

def save(fig,path):
    from PIL import Image
    from io import BytesIO
    temp=path.with_name(path.stem+'.tmp'+path.suffix);buffer=BytesIO()
    fig.savefig(buffer,format='png',bbox_inches='tight',facecolor='white');plt.close(fig)
    payload=buffer.getvalue()
    with Image.open(BytesIO(payload)) as check:check.verify()
    temp.write_bytes(payload)
    if temp.stat().st_size!=len(payload):raise IOError('Incomplete figure write')
    with Image.open(temp) as check:check.verify()
    temp.replace(path)

def local_figures(data,out):
    local=data/'local_reference'
    if not all((local/name).exists() for name in ['sweep.csv','no_carrier_scaling.csv','scaling_fit.json','unit_cell.csv']):return
    d=pd.read_csv(local/'sweep.csv');q=d[(d.offset_px==.5)&(d.axis=='x')]
    fig,axs=plt.subplots(1,2,figsize=(11,4.7),layout='constrained')
    for w,g in q.groupby('w_px'):
        for ax,key in zip(axs,['local_infidelity','raw_infidelity']):
            ax.semilogy(g.L,np.maximum(g[key],1e-12),'o-',label=f'w = {w} px')
    for ax in axs:ax.set_xlabel('Charge L');ax.set_ylabel('Complex-field infidelity');ax.set_xticks([0,1,3,5,10,20]);ax.grid(alpha=.2)
    axs[0].set_title('Local two-panel phase product');axs[1].set_title('After ideal order selection');axs[1].legend(fontsize=11,ncol=2)
    fig.suptitle('Local identity-transfer baseline  |  Both SLMs blazed  |  Half-pixel x shift',fontsize=16)
    save(fig,out/'03_local_charge_radius.png')
    s=pd.read_csv(local/'no_carrier_scaling.csv');fit=json.loads((local/'scaling_fit.json').read_text())
    fig,axs=plt.subplots(1,2,figsize=(11,4.7),layout='constrained')
    for w,g in s.groupby('w_px'):axs[0].loglog(g.eta,g.infidelity,'o',label=f'{w} px')
    xx=np.geomspace(s.eta.min(),s.eta.max(),100)
    axs[0].loglog(xx,fit['prefactor']*xx**fit['exponent'],'k--',label=f"Fit exponent {fit['exponent']:.2f}")
    for w,g in q[q.L>0].groupby('w_px'):axs[1].loglog(g.eta,g.raw_infidelity,'o-',label=f'{w} px')
    for ax in axs:ax.set_xlabel(r'$\eta=L p/w$');ax.set_ylabel('Complex-field infidelity');ax.grid(alpha=.2,which='both')
    axs[0].set_title('Vortex-only local diagnostic');axs[1].set_title('Two-blaze selected channel');axs[0].legend(fontsize=10,ncol=2)
    fig.suptitle('Scaling is diagnostic not a universal propagated-field law',fontsize=16)
    save(fig,out/'04_local_scaling.png')
    u=pd.read_csv(local/'unit_cell.csv');fig,axs=plt.subplots(1,2,figsize=(10,4.8),layout='constrained')
    for ax,w in zip(axs,[50,250]):
        g=u[u.w_px==w];im=ax.imshow(g.pivot(index='rho_y',columns='rho_x',values='raw_infidelity'),origin='lower',
            extent=(-.0625,.9375,-.0625,.9375),cmap='magma',interpolation='nearest')
        ax.set_title(f'L = 20, w = {w} px');ax.set_xlabel(r'$\rho_x$ / pixels');ax.set_ylabel(r'$\rho_y$ / pixels')
        fig.colorbar(im,ax=ax,label='Selected-field infidelity',shrink=.8)
    fig.suptitle('Local unit cell  |  Two-blaze identity-transfer baseline',fontsize=16)
    save(fig,out/'05_local_unit_cell.png')

def run(data,out):
    out.mkdir(parents=True,exist_ok=True)
    local_figures(data,out)
    from tools.plot_registration_reference_study import geometry
    geometry(out,architectures=False)
    fig,axs=plt.subplots(2,1,figsize=(10,5),layout='constrained')
    for ax,arch in zip(axs,['A','B']):
        ax.set_xlim(0,10);ax.set_ylim(0,2);ax.axis('off')
        texts=['SLM1\nVortex + blaze','SLM2\nFlat correction\n+ blaze'] if arch=='A' else ['SLM1\nFlat correction\n+ blaze','SLM2\nVortex + blaze']
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
            fine=data/f'fine_sequence_L{L}_w{w}_{"upstream_vortex" if arch=="A" else "downstream_vortex"}.npz'
            if plane=='post' and fine.exists():
                refined=np.load(fine);x=refined['x'];I=abs(refined['fields'])**2
            else:x=a[plane+'_x'];I=abs(a[plane])**2
            ref=I[0];peak=ref.max();factor=1e3 if plane=='pre' else 1e6
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
    fine_metrics=pd.read_csv(data/'fine_sequence_metrics.csv') if (data/'fine_sequence_metrics.csv').exists() else None
    for (L,arch),g in m[(m.w_px==250)&(m.role=='vortex + blaze')].groupby(['L','architecture']):
        label=f'L={L}, {"A" if arch=="upstream_vortex" else "B"}, z={1000*g.z_m.iloc[0]:g} mm'
        axs[0].plot(g.offset_px,100*g.selected_infidelity,'o-',label=label)
        h=fine_metrics[(fine_metrics.L==L)&(fine_metrics.w_px==250)&(fine_metrics.architecture==arch)] if fine_metrics is not None else None
        axs[1].plot(h.offset_px if h is not None else g.offset_px,
            100*h.power_normalised_residual_l2 if h is not None else 100*g.post_power_normalised_residual_l2,'o-',label=label)
    for ax in axs:ax.set_xlabel(r'Registration $\Delta x/p$');ax.grid(alpha=.25);ax.legend(fontsize=10)
    axs[0].set_ylabel('Selected-field infidelity (%)');axs[1].set_ylabel('Propagated shape residual (%)')
    fig.suptitle('Bench radius 2 mm  |  Conditional 200 mm gap  |  Flat correction',fontsize=16)
    save(fig,out/'presentation_02_bench_sensitivity_200mm.png')
    files=sorted(data.glob('axial_L*.csv'))
    if files:
        fig,axs=plt.subplots(2,2,figsize=(11,7.5),layout='constrained');context=[]
        for path in files:
            g=pd.read_csv(path);r=g.iloc[0];arch='A' if r.architecture=='upstream_vortex' else 'B'
            field=np.load(path.with_suffix('.npz'));roi=np.sum(abs(field['fields'][0])**2,axis=(1,2));fraction=roi/roi.max()
            col=0 if r.w_px==50 else 1;label=f'L={int(r.L)}, {arch}'
            line=axs[0,col].semilogy(1000*g.z_m,fraction,'o-',label=label)[0]
            axs[1,col].semilogy(1000*g.z_m,100*g.power_normalised_residual_l2,'o-',color=line.get_color(),markerfacecolor='white',label=label)
            bright=fraction>=.01
            axs[1,col].scatter(1000*g.z_m[bright],100*g.power_normalised_residual_l2[bright],color=line.get_color(),zorder=3)
            for i,row in enumerate(g.to_dict('records')):context.append(dict(**row,baseline_roi_power_fraction_of_axial_max=float(fraction[i])))
        pd.DataFrame(context).to_csv(data/'axial_context.csv',index=False)
        for ax in axs.flat:ax.set_xlabel('Distance beyond axicon (mm)');ax.grid(alpha=.2,which='both');ax.legend(fontsize=10)
        for ax in axs[0]:ax.set_ylabel('ROI power / route axial max');ax.axhline(.01,color='grey',ls=':',lw=1)
        for ax in axs[1]:ax.set_ylabel('Half-pixel shape residual (%)')
        axs[0,0].set_title('Stress radius 50 px');axs[0,1].set_title('Bench radius 250 px')
        fig.suptitle('Common physical z planes  |  Fixed 300 µm XY metric region',fontsize=16)
        save(fig,out/'06_axial_registration.png')
    path=data/'actual_panel_L20_w50_upstream_vortex.npz'
    if path.exists():
        a=np.load(path);E=a['fields'];extent=[1000*a['x'][0],1000*a['x'][-1],1000*a['y'][0],1000*a['y'][-1]]
        fig,axs=plt.subplots(2,2,figsize=(9,7.6),layout='constrained');peak=np.max(abs(E[0,0])**2)
        for i,d in enumerate([0,.5]):
            im=axs[i,0].imshow(abs(E[i,0])**2/peak,origin='lower',extent=extent,cmap='inferno',vmin=0,vmax=1.05,interpolation='bilinear')
            ph=axs[i,1].imshow(np.angle(E[i,1]),origin='lower',extent=extent,cmap='twilight',vmin=-np.pi,vmax=np.pi,interpolation='nearest')
            for ax,title in zip(axs[i],[f'Incident on SLM2, Δx={d:g}p',f'After SLM2 phase, Δx={d:g}p']):
                ax.set_title(title);ax.set_xlabel('x (mm)');ax.set_ylabel('y (mm)')
        fig.colorbar(im,ax=list(axs[:,0]),shrink=.75,label='Intensity / baseline peak')
        fig.colorbar(ph,ax=list(axs[:,1]),shrink=.75,label='Wrapped envelope phase (rad)')
        fig.suptitle('Stress L = 20, w = 50 px  |  Architecture A\nActual SLM2 plane before downstream order selection',fontsize=16)
        save(fig,out/'07_actual_slm2_plane.png')
    path=data/'immediate_slm1_L20_w50_upstream_vortex.npz'
    if path.exists():
        a=np.load(path);ids=np.flatnonzero(abs(a['x'])<.001);x=1000*a['x'][ids]
        E=a['fields'][:,ids][:,:,ids];I=abs(E[0])**2;I/=I.max()
        fig,axs=plt.subplots(1,3,figsize=(11,3.9),layout='constrained')
        im=axs[0].imshow(I,origin='lower',extent=[x[0],x[-1],x[0],x[-1]],cmap='inferno',vmin=0,vmax=1,interpolation='bilinear')
        axs[0].set_title('SLM1 intensity')
        for ax,e,d in zip(axs[1:],E,[0,.5]):
            phase=np.ma.masked_where(I<1e-3,np.angle(e))
            ph=ax.imshow(phase,origin='lower',extent=[x[0],x[-1],x[0],x[-1]],cmap='twilight',vmin=-np.pi,vmax=np.pi,interpolation='nearest')
            ax.set_title(f'Phase, Δx={d:g}p')
        for ax in axs:ax.set_xlabel('x (mm)');ax.set_ylabel('y (mm)')
        fig.colorbar(im,ax=axs[0],shrink=.8,label='I / peak')
        fig.colorbar(ph,ax=list(axs[1:]),shrink=.8,label='Wrapped phase (rad)')
        fig.suptitle('L = 20, w = 50 px  |  Phase-only vortex + blaze  |  Before propagation',fontsize=15)
        save(fig,out/'08_immediate_slm1.png')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.data,a.output)
