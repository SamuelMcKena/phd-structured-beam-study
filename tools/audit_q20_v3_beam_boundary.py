"""Train-only boundary audit for the q=20 v3 beam-radius nuisance.

The accepted v3 diagnostic selected beam_radius_scale=1.25 at the upper edge of
its screening grid.  This script does not silently promote a larger beam.  It
asks whether the train-only objective continues to improve beyond 1.25 when
absolute z is re-optimised locally for every tested beam scale, while keeping the
v3 iris scale fixed.  Odd z planes are scored only after the train optimum is
frozen and never enter the selection.

The output is an identifiability audit.  If the optimum remains on the new
boundary, the correct scientific conclusion is that the input-beam radius is a
model-bound nuisance that requires an independent bench measurement, not that
an arbitrarily large fitted beam should be accepted.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
TOOLS=ROOT/'tools'; EXP=ROOT/'notebooks'/'experimental'/'axicon_aberration_correction'
for p in (ROOT,TOOLS,EXP):
    if str(p) not in sys.path: sys.path.insert(0,str(p))
import fit_q20_detector_aware_model_v2 as v2  # noqa
import fit_q20_detector_aware_model_v3 as v3  # noqa

BEAM_SCALES=np.asarray([1.10,1.20,1.25,1.30,1.40,1.50,1.60],float)
Z0_VALUES=np.asarray([32.5,33.5,34.5,35.5,36.5],float)
IRIS_SCALE=1.05

def run(source_dir:Path,out:Path):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    base=v2.build_context(Path(source_dir)); n=len(base['z_rel'])
    train=np.arange(0,n,2,dtype=int); held=np.arange(1,n,2,dtype=int)
    rows=[]
    for bs in BEAM_SCALES:
        cfg=v3.config_with(base['config'],beam_scale=float(bs),iris_scale=IRIS_SCALE)
        ctx=v3.route_context(base,cfg)
        for z0 in Z0_VALUES:
            m=v3.evaluate_context(ctx,float(z0),train)
            rows.append({'beam_radius_scale':float(bs),'z0_mm':float(z0),
                         'train_r':m['mean_pearson_r'],'train_nrmse':m['mean_nrmse'],
                         'train_max_nrmse':m['max_nrmse'],'train_objective':m['objective']})
    df=pd.DataFrame(rows)
    per=[]
    for bs,g in df.groupby('beam_radius_scale',sort=True):
        r=g.loc[g.train_objective.idxmin()].to_dict(); per.append(r)
    best_by_beam=pd.DataFrame(per).sort_values('beam_radius_scale').reset_index(drop=True)
    best=best_by_beam.loc[best_by_beam.train_objective.idxmin()]
    bs=float(best.beam_radius_scale); z0=float(best.z0_mm)
    cfg=v3.config_with(base['config'],beam_scale=bs,iris_scale=IRIS_SCALE); ctx=v3.route_context(base,cfg)
    held_metrics=v3.evaluate_context(ctx,z0,held)
    boundary=bool(np.isclose(bs,float(BEAM_SCALES.max())) or np.isclose(bs,float(BEAM_SCALES.min())))
    # Monotonic improvement at the high end is an even stronger non-identifiability flag.
    tail=best_by_beam.tail(3).train_objective.to_numpy(float)
    decreasing_tail=bool(np.all(np.diff(tail)<0))
    result={
      'study':'q20 v3 train-only beam-radius boundary/identifiability audit',
      'beam_radius_scales':BEAM_SCALES.tolist(),'z0_values_mm':Z0_VALUES.tolist(),
      'fixed_iris_radius_scale':IRIS_SCALE,
      'selection_uses_heldout':False,
      'selected_train_only':{'beam_radius_scale':bs,'model_beam_radius_mm':2.0*bs,'z0_mm':z0,
          'train_pearson_r':float(best.train_r),'train_nrmse':float(best.train_nrmse),'train_objective':float(best.train_objective)},
      'heldout_after_freeze':{'mean_pearson_r':held_metrics['mean_pearson_r'],'mean_nrmse':held_metrics['mean_nrmse'],'max_nrmse':held_metrics['max_nrmse']},
      'selected_on_tested_boundary':boundary,'objective_still_decreasing_at_high_end':decreasing_tail,
      'decision':('Do not promote fitted beam radius: nuisance remains boundary-limited/non-identifiable; independently measure the bench beam radius.' if boundary or decreasing_tail else 'Boundary resolved inside the extended scan; candidate may be reconsidered in a separate fully re-fit model step.'),
      'hardware_ready':False,
    }
    df.to_csv(out/'beam_z_grid.csv',index=False); best_by_beam.to_csv(out/'best_z_per_beam.csv',index=False)
    (out/'beam_boundary_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    fig,axs=plt.subplots(1,2,figsize=(10.5,4.3),constrained_layout=True)
    axs[0].plot(best_by_beam.beam_radius_scale,best_by_beam.train_objective,'o-',lw=2)
    axs[0].axvline(1.25,ls='--',lw=1.3,label='v3 selected edge')
    axs[0].set(xlabel='Gaussian radius scale',ylabel='best train-only objective',title='Does the v3 optimum continue beyond 1.25?'); axs[0].grid(alpha=.2); axs[0].legend(frameon=False)
    axs[1].plot(best_by_beam.beam_radius_scale,best_by_beam.z0_mm,'o-',lw=2)
    axs[1].set(xlabel='Gaussian radius scale',ylabel='best z0 (mm)',title='Beam-radius / z registration coupling'); axs[1].grid(alpha=.2)
    fig.savefig(out/'beam_boundary_audit.png',dpi=400,bbox_inches='tight'); fig.savefig(out/'beam_boundary_audit.pdf',bbox_inches='tight'); plt.close(fig)
    print(json.dumps(result,indent=2)); return result

def main():
    p=argparse.ArgumentParser(); p.add_argument('--source-dir',type=Path,default=EXP/'outputs'/'digital_twin_correction'); p.add_argument('--out',type=Path,default=ROOT/'outputs'/'validation'/'q20_v3_beam_boundary_audit'); a=p.parse_args(); run(a.source_dir,a.out)
if __name__=='__main__': main()
