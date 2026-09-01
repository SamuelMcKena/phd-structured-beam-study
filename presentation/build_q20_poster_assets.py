from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

p = argparse.ArgumentParser(description='Build q20 poster assets from accepted workflow artifacts.')
p.add_argument('--detector-dir', type=Path, required=True)
p.add_argument('--topology-dir', type=Path, required=True)
p.add_argument('--out', type=Path, default=Path('presentation/generated/q20_poster_assets'))
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)

D = np.load(a.detector_dir / 'detector_aware_residual_stacks.npz')
Q = np.load(a.topology_dir / 'q20_v2_selected_4096_display_arrays.npz')
MASK = np.load(a.topology_dir / 'model_space_slm2_phase_v2_scaled_rad.npy')
SWEEP = pd.read_csv(a.topology_dir / 'q20_v2_strength_sweep_3072.csv')
SUMMARY = json.loads((a.topology_dir / 'q20_v2_strength_selected_summary.json').read_text())

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.labelsize':12,'xtick.labelsize':10,'ytick.labelsize':10})
axis, z = D['axis_um'], D['z_relative_mm']
qaxis, qz = Q['axis_um'], Q['z_relative_mm']
rep = int(np.argmin(np.abs(z + 10.0)))
qrep = int(np.argmin(np.abs(qz + 10.0)))
meas = D['measured_beam_frame']
fit = D['axicon_input_positive_error']
nom, corr = Q['optical_nominal'], Q['optical_corrected']


def save_map(arr, axis_um, name, cmap='inferno', norm=None, vmin=0, vmax=1, cbar=None):
    fig, ax = plt.subplots(figsize=(5.1,4.45))
    kw = dict(origin='lower', extent=[axis_um[0],axis_um[-1],axis_um[0],axis_um[-1]], cmap=cmap, interpolation='nearest')
    if norm is not None: kw['norm'] = norm
    else: kw.update(vmin=vmin, vmax=vmax)
    im = ax.imshow(arr, **kw)
    ax.set(xlabel='x (µm)', ylabel='y (µm)'); ax.set_aspect('equal')
    if cbar: fig.colorbar(im, ax=ax, fraction=.046, pad=.04, label=cbar)
    fig.tight_layout(pad=.25); fig.savefig(a.out/name, dpi=300, bbox_inches='tight', facecolor='white'); plt.close(fig)

save_map(meas[rep], axis, 'measured_q20_xy.png')
save_map(fit[rep], axis, 'reconstructed_q20_xy.png')
save_map(nom[qrep], qaxis, 'nominal_q20_optical.png')
save_map(corr[qrep], qaxis, 'corrected_q20_optical.png')

mid = len(axis)//2
fig, ax = plt.subplots(figsize=(6.8,3.55)); im=ax.imshow(meas[:,mid,:],origin='lower',aspect='auto',extent=[axis[0],axis[-1],z[0],z[-1]],cmap='inferno',vmin=0,vmax=1,interpolation='nearest'); ax.set(xlabel='x (µm)',ylabel='relative z (mm)'); fig.colorbar(im,ax=ax,fraction=.035,pad=.02,label='normalized intensity'); fig.tight_layout(pad=.3); fig.savefig(a.out/'measured_q20_xz.png',dpi=300,bbox_inches='tight',facecolor='white'); plt.close(fig)

qmid = len(qaxis)//2
fig, ax = plt.subplots(figsize=(6.8,3.55)); im=ax.imshow(corr[:,qmid,:],origin='lower',aspect='auto',extent=[qaxis[0],qaxis[-1],qz[0],qz[-1]],cmap='inferno',vmin=0,vmax=1,interpolation='nearest'); ax.set(xlabel='x (µm)',ylabel='relative z (mm)'); fig.colorbar(im,ax=ax,fraction=.035,pad=.02,label='normalized optical intensity'); fig.tight_layout(pad=.3); fig.savefig(a.out/'corrected_q20_optical_xz.png',dpi=300,bbox_inches='tight',facecolor='white'); plt.close(fig)

rdiff = fit[rep] - meas[rep]; lim=float(np.quantile(np.abs(rdiff),.995)); save_map(rdiff,axis,'reconstruction_difference.png','coolwarm',TwoSlopeNorm(vcenter=0,vmin=-lim,vmax=lim),cbar='model - measurement')
delta = corr[qrep] - nom[qrep]; dlim=float(np.quantile(np.abs(delta),.997)); save_map(delta,qaxis,'corrected_minus_nominal.png','coolwarm',TwoSlopeNorm(vcenter=0,vmin=-dlim,vmax=dlim),cbar='corrected - nominal intensity')

coeff=D['axicon_input_coefficients_rad']; u=np.linspace(-1,1,501); X,Y=np.meshgrid(u,u); th=np.arctan2(Y,X); R=np.hypot(X,Y); phase=np.zeros_like(th)
for i,m in enumerate((1,2,3)): phase += coeff[2*i]*np.cos(m*th) + coeff[2*i+1]*np.sin(m*th)
phase=np.where(R<=1,phase,np.nan); plim=max(abs(np.nanmin(phase)),abs(np.nanmax(phase)))
fig,ax=plt.subplots(figsize=(5,4.45)); im=ax.imshow(phase,origin='lower',extent=[-1,1,-1,1],cmap='twilight',vmin=-plim,vmax=plim,interpolation='bilinear'); ax.set(xlabel='normalized x',ylabel='normalized y'); ax.set_aspect('equal'); fig.colorbar(im,ax=ax,fraction=.046,pad=.04,label='retrieved phase (rad)'); fig.tight_layout(pad=.25); fig.savefig(a.out/'retrieved_axicon_residual_phase.png',dpi=300,bbox_inches='tight',facecolor='white'); plt.close(fig)

mlim=max(abs(float(MASK.min())),abs(float(MASK.max()))); fig,ax=plt.subplots(figsize=(5,4.45)); im=ax.imshow(MASK,origin='lower',cmap='twilight',vmin=-mlim,vmax=mlim,interpolation='nearest'); ax.set(xlabel='SLM2 x (model pixels)',ylabel='SLM2 y (model pixels)'); ax.set_aspect('equal'); fig.colorbar(im,ax=ax,fraction=.046,pad=.04,label='added phase (rad)'); fig.tight_layout(pad=.25); fig.savefig(a.out/'slm2_correction_phase.png',dpi=300,bbox_inches='tight',facecolor='white'); plt.close(fig)

Xq,Yq=np.meshgrid(qaxis,qaxis); Rq=np.hypot(Xq,Yq); rb=np.linspace(0,145,146); rc=0.5*(rb[:-1]+rb[1:])
def radial(img):
    v=np.array([np.mean(img[(Rq>=lo)&(Rq<hi)]) for lo,hi in zip(rb[:-1],rb[1:])]); return v/np.nanmax(v)
fig,ax=plt.subplots(figsize=(6.1,3.7)); ax.plot(rc,radial(nom[qrep]),lw=2.2,label='nominal q=20 target'); ax.plot(rc,radial(corr[qrep]),lw=2.2,ls='--',label='predicted corrected optical field'); ax.set(xlim=(0,140),ylim=(0,1.05),xlabel='radius (µm)',ylabel='azimuthal mean intensity'); ax.grid(alpha=.25); ax.legend(frameon=False,fontsize=10); fig.tight_layout(pad=.35); fig.savefig(a.out/'radial_profiles_q20.png',dpi=300,bbox_inches='tight',facecolor='white'); plt.close(fig)

fig,ax=plt.subplots(figsize=(6.1,3.8)); ax.plot(SWEEP.alpha,SWEEP.mean_optical_r,marker='o',label='optical-field r'); ax.plot(SWEEP.alpha,SWEEP.mean_detector_r,marker='s',ls='--',label='predicted detector r'); ok=SWEEP[SWEEP.topology_q20_all_contours.astype(bool)]; bad=SWEEP[~SWEEP.topology_q20_all_contours.astype(bool)]; ax.scatter(ok.alpha,ok.mean_optical_r,s=105,facecolors='none',edgecolors='black',linewidths=2,label='q=20 preserved'); [ax.text(r.alpha,r.mean_optical_r-.004,'×',ha='center',va='top',fontsize=15) for _,r in bad.iterrows()]; ax.axvline(.40,ls=':',lw=1.5); ax.set(xlabel='correction strength α',ylabel='mean correlation',ylim=(.82,.97)); ax.grid(alpha=.25); ax.legend(frameon=False,fontsize=9,loc='lower right'); fig.tight_layout(pad=.35); fig.savefig(a.out/'correction_strength_topology.png',dpi=300,bbox_inches='tight',facecolor='white'); plt.close(fig)

radii=np.array([1.0,1.1,1.2,1.3,1.4,1.5]); fig,ax=plt.subplots(figsize=(6.1,3.1)); ax.plot(radii,np.full_like(radii,20.),marker='o',lw=2.2); ax.axhline(20,ls='--',lw=1); ax.set(xlabel='winding contour radius (mm)',ylabel='winding number',ylim=(18.8,21.2)); ax.set_yticks([19,20,21]); ax.grid(alpha=.25); fig.tight_layout(pad=.35); fig.savefig(a.out/'winding_validation.png',dpi=300,bbox_inches='tight',facecolor='white'); plt.close(fig)

for arr,name,cmap,nrm in [(nom[qrep],'nominal_q20_tile.png','inferno',None),(corr[qrep],'corrected_q20_tile.png','inferno',None),(delta,'delta_q20_tile.png','coolwarm',TwoSlopeNorm(vcenter=0,vmin=-dlim,vmax=dlim))]:
    fig=plt.figure(figsize=(4,4)); ax=fig.add_axes([0,0,1,1]); kw=dict(origin='lower',cmap=cmap,interpolation='nearest'); kw.update(norm=nrm) if nrm else kw.update(vmin=0,vmax=1); ax.imshow(arr,**kw); ax.axis('off'); fig.savefig(a.out/name,dpi=300,bbox_inches='tight',pad_inches=0); plt.close(fig)

print(f'Built {len(list(a.out.glob("*.png")))} poster assets in {a.out}')
