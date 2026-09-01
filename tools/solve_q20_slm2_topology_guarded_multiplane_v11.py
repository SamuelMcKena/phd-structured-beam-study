"""q=20 correction v11: topology-guarded continuation of the v10 multi-plane solve.

v10 finally put the *propagated multi-ring train* inside the correction loop, but
its tested command strengths (0.60--1.00) all split the inner high-order vortex:
the 1.0--1.3 mm winding contours no longer enclosed q=20.  This script treats
that failure as a continuation problem rather than another image-fit problem.

The v10 multi-plane command is regenerated without refitting the frozen
Miao-initialised diagnostic residual.  We then scan two physically interpretable
regularisers on INNER validation planes only:

  1. global command strength alpha, now extending down to 0.05;
  2. a smooth radial SLM2 guard which suppresses the correction in the central
     part of the illuminated SLM aperture and restores it outside a transition.

Every candidate is topology-tested *before* its propagated detector stack is
scored.  Only q=20-preserving candidates can be selected.  Among them, selection
uses the multi-ring visible-concentricity objective (ring intensity CV, angular
harmonic energy and radial wobble), never the experimental angular pattern.

The legacy odd planes are inspected only after alpha/guard selection is frozen.
All corrected outputs remain numerical model-space predictions, not corrected
BeamGage measurements or hardware-ready SLM masks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
EXP = ROOT / "notebooks" / "experimental" / "axicon_aberration_correction"
for p in (ROOT, TOOLS, EXP):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import solve_q20_slm2_hybrid_miao_concentric_v5 as v5  # noqa: E402
import solve_q20_slm2_ifta_circular_v9 as v9  # noqa: E402
import solve_q20_slm2_multiplane_circular_v10 as v10  # noqa: E402
from real_bmg_digital_twin_correction import FIT_WINDOW_M, Q, RELAY_N  # noqa: E402
from vbb_study.digital_twin.vortex_system_route import build_system_route  # noqa: E402

EPS = np.finfo(float).tiny
THERMAL = "inferno"
PROD_N = 4096
ALPHAS = np.asarray([0.00,0.05,0.08,0.12,0.16,0.20,0.25,0.30,0.35,0.40,0.50,0.60],float)
# (inner radius, outer radius) of SLM-plane correction transition in metres.
# None means no spatial guard.  These are MODEL-SPACE continuation parameters,
# not claims about a measured SLM/axicon conjugate radius.
GUARDS = [
    ("none", None, None),
    ("g0p8_1p2",0.8e-3,1.2e-3),
    ("g1p0_1p4",1.0e-3,1.4e-3),
    ("g1p2_1p6",1.2e-3,1.6e-3),
    ("g1p4_1p8",1.4e-3,1.8e-3),
]


def savefig(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"),dpi=500,bbox_inches="tight",facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"),bbox_inches="tight",facecolor="white")
    plt.close(fig)


def smoothstep01(x: np.ndarray) -> np.ndarray:
    t=np.clip(np.asarray(x,float),0.0,1.0)
    return t*t*(3.0-2.0*t)


def guard_map(grid: dict, inner_m: float|None, outer_m: float|None) -> np.ndarray:
    if inner_m is None or outer_m is None:
        return np.ones_like(np.asarray(grid["R"],float))
    if not float(outer_m) > float(inner_m):
        raise ValueError("guard outer radius must exceed inner radius")
    R=np.asarray(grid["R"],float)
    return smoothstep01((R-float(inner_m))/(float(outer_m)-float(inner_m)))


def candidate_command(base_command: np.ndarray, grid: dict, alpha: float, inner_m, outer_m) -> np.ndarray:
    return float(alpha)*np.asarray(base_command,float)*guard_map(grid,inner_m,outer_m)


def validation_baseline(config,z_abs,target_det,pcoef,acoef):
    route=v10.slm2_command_route(config,v10.OPT_N,np.zeros((RELAY_N,RELAY_N),float),pcoef,acoef)
    wd,ok=v10.topology(route)
    det=v5.detector_stack(route,z_abs)
    score,multi=v10.multiring_objective(det,target_det,v10.VALID_PLANES)
    return {"score":float(score),"multiring":multi,"winding":wd,"topology_q20_all_contours":bool(ok)}


def guarded_sweep(config,z_abs,target_det,base_command,pcoef,acoef,out:Path):
    base_route=build_system_route(f"V{Q}",grid_n=RELAY_N,config=config,window_m=FIT_WINDOW_M)
    grid=base_route["grid"]
    baseline=validation_baseline(config,z_abs,target_det,pcoef,acoef)
    rows=[]
    for gname,rin,rout in GUARDS:
        for alpha in ALPHAS:
            cmd=candidate_command(base_command,grid,float(alpha),rin,rout)
            route=v10.slm2_command_route(config,v10.OPT_N,cmd,pcoef,acoef)
            wd,ok=v10.topology(route)
            row={
                "guard":gname,"guard_inner_m":None if rin is None else float(rin),
                "guard_outer_m":None if rout is None else float(rout),"alpha":float(alpha),
                "topology_q20_all_contours":bool(ok),"winding":wd,
                "validation_multiring_score":None,"validation_multiring":None,
                "validation_score_reduction_fraction":None,
            }
            # Topology is the hard first gate.  Skip expensive 18-plane scoring
            # for candidates which have already split the vortex.
            if ok:
                det=v5.detector_stack(route,z_abs)
                score,multi=v10.multiring_objective(det,target_det,v10.VALID_PLANES)
                row["validation_multiring_score"]=float(score)
                row["validation_multiring"]=multi
                row["validation_score_reduction_fraction"]=float(1.0-score/max(baseline["score"],EPS))
            rows.append(row)
            print(json.dumps(row,indent=2))
    (out/"v11_topology_guard_sweep.json").write_text(json.dumps({"baseline":baseline,"rows":rows},indent=2)+"\n",encoding="utf-8")
    passing=[r for r in rows if r["topology_q20_all_contours"] and r["alpha"]>0.0 and r["validation_multiring_score"] is not None]
    improving=[r for r in passing if float(r["validation_multiring_score"]) < float(baseline["score"])]
    selected=min(improving,key=lambda r:r["validation_multiring_score"]) if improving else (min(passing,key=lambda r:r["validation_multiring_score"]) if passing else None)

    # Compact audit plot: score vs alpha for topology-preserving points only.
    fig,ax=plt.subplots(figsize=(8.8,5.8),constrained_layout=True)
    for gname,_,_ in GUARDS:
        rr=[r for r in rows if r["guard"]==gname and r["validation_multiring_score"] is not None]
        if rr:
            ax.plot([r["alpha"] for r in rr],[r["validation_multiring_score"] for r in rr],marker="o",label=gname)
    ax.axhline(baseline["score"],ls="--",lw=1.2,label="uncorrected baseline")
    ax.set_xlabel("global correction strength alpha"); ax.set_ylabel("validation multi-ring score (lower is better)")
    ax.set_title("v11 topology-preserving continuation sweep"); ax.grid(alpha=0.25); ax.legend(fontsize=8,ncol=2)
    savefig(fig,out/"v11_topology_guard_sweep")
    return baseline,rows,selected,grid


def production(config,z_abs,z_rel,target_det,command,pcoef,acoef,out:Path):
    positive=v5.build_route(config,PROD_N,slm2_phase=None,pcoef=pcoef,acoef=acoef)
    corrected=v10.slm2_command_route(config,PROD_N,command,pcoef,acoef)
    pdet=v5.detector_stack(positive,z_abs); cdet=v5.detector_stack(corrected,z_abs)
    popt=v5.optical_stack(positive,z_abs); copt=v5.optical_stack(corrected,z_abs)
    wp,_=v10.topology(positive); wc,top_ok=v10.topology(corrected)
    groups={"train":v10.TRAIN_PLANES,"inner_validation":v10.VALID_PLANES,"legacy_heldout":v10.LEGACY_HELD,"all_planes":np.arange(len(z_rel),dtype=int)}
    metrics={}
    for name,ids in groups.items():
        ps,pm=v10.multiring_objective(pdet,target_det,ids); cs,cm=v10.multiring_objective(cdet,target_det,ids)
        metrics[name]={"detector_positive_multiring_score":float(ps),"detector_corrected_multiring_score":float(cs),"detector_positive_multiring":pm,"detector_corrected_multiring":cm,"detector_positive":v5.concentric_metrics(pdet,target_det,ids),"detector_corrected":v5.concentric_metrics(cdet,target_det,ids)}
    np.save(out/"model_space_slm2_command_topology_guarded_v11_rad.npy",np.asarray(command,np.float32))
    np.savez_compressed(out/"v11_4096_display_arrays.npz",axis_um=v5.AXIS_UM,z_relative_mm=z_rel,concentric_target=target_det.astype(np.float32),detector_positive=pdet.astype(np.float32),detector_corrected=cdet.astype(np.float32),optical_positive=popt.astype(np.float32),optical_corrected=copt.astype(np.float32))
    ext=[v5.AXIS_UM[0],v5.AXIS_UM[-1],v5.AXIS_UM[0],v5.AXIS_UM[-1]]; ids=[1,5,9,13,17]
    fig,axs=plt.subplots(3,len(ids),figsize=(15.5,8.8),constrained_layout=True)
    for col,iz in enumerate(ids):
        for row,(stack,label) in enumerate(((pdet,"diagnosed model"),(target_det,"concentric target"),(cdet,"v11 corrected"))):
            axs[row,col].imshow(stack[iz],origin="lower",extent=ext,cmap=THERMAL,vmin=0,vmax=1,interpolation="nearest"); axs[row,col].set_aspect("equal"); axs[row,col].set_xticks([]); axs[row,col].set_yticks([])
            if row==0: axs[row,col].set_title(f"z = {z_rel[iz]:.0f} mm")
            if col==0: axs[row,col].set_ylabel(label,fontweight="bold")
    fig.suptitle("q=20 v11: topology-guarded multi-plane correction",fontsize=14,fontweight="bold"); savefig(fig,out/"poster_v11_multiplane_detector")
    return {"production_grid_n":PROD_N,"topology_q20_all_contours":bool(top_ok),"winding_positive":wp,"winding_corrected":wc,"metrics":metrics}


def run(source_dir:Path,candidate_json:Path,residual_json:Path,out:Path)->dict:
    source_dir=Path(source_dir); out=Path(out); out.mkdir(parents=True,exist_ok=True)
    config,candidate=v9.config_from_files(candidate_json,source_dir); pcoef,acoef,_=v9.frozen_residual(residual_json)
    summary0=json.loads((source_dir/"run_summary.json").read_text(encoding="utf-8")); z_rel=np.asarray(summary0["data"]["z_relative_mm"],float); z0=float(candidate["physical_nuisance"]["selected_z0_mm"]); z_abs=(z0+z_rel)*1e-3
    _,target_amp,_,_=v10.nominal_target(config,z_abs)
    base_command,history,v9_history,_,prop_meta,best=v10.optimise_multiplane(config,z_abs,target_amp,pcoef,acoef,out/"v10_regenerated")
    target_det=v10.evaluation_target(config,z_abs)
    baseline,rows,selected,relay_grid=guarded_sweep(config,z_abs,target_det,base_command,pcoef,acoef,out)
    if selected is None:
        result={"status":"q20_topology_guarded_multiplane_v11_no_topology_preserving_nonzero_candidate","baseline":baseline,"selected":None,"acceptance":{"passes_concentricity_gate":False},"hardware_ready":False,"evidence_boundary":"numerical model-space correction only"}
        (out/"v11_summary.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8"); print(json.dumps(result,indent=2)); return result
    command=candidate_command(base_command,relay_grid,selected["alpha"],selected["guard_inner_m"],selected["guard_outer_m"])
    prod=production(config,z_abs,z_rel,target_det,command,pcoef,acoef,out)
    held=prod["metrics"]["legacy_heldout"]; pm=held["detector_positive_multiring"]; cm=held["detector_corrected_multiring"]
    reductions={
        "multiring_score":float(1.0-held["detector_corrected_multiring_score"]/max(held["detector_positive_multiring_score"],EPS)),
        "ring_intensity_cv":float(1.0-cm["mean_ring_intensity_cv"]/max(pm["mean_ring_intensity_cv"],EPS)),
        "angular_harmonic_energy":float(1.0-cm["mean_angular_harmonic_energy"]/max(pm["mean_angular_harmonic_energy"],EPS)),
        "ring_radius_std":float(1.0-cm["mean_ring_radius_std_um"]/max(pm["mean_ring_radius_std_um"],EPS)),
    }
    acceptance={
        "q20_topology_preserved":bool(prod["topology_q20_all_contours"]),
        "legacy_heldout_reductions":reductions,
        "passes_concentricity_gate":bool(prod["topology_q20_all_contours"] and reductions["multiring_score"]>=0.25 and reductions["ring_intensity_cv"]>=0.18 and reductions["angular_harmonic_energy"]>=0.18 and reductions["ring_radius_std"]>=0.10),
    }
    result={
        "status":"q20_topology_guarded_multiplane_candidate_v11",
        "frozen_residual_source":Path(residual_json).name,
        "selected_on_inner_validation_only":selected,
        "validation_baseline":baseline,
        "production_validation":prod,"acceptance":acceptance,
        "target_policy":"explicitly concentric multi-plane radial target; measured angular structure is not rewarded",
        "topology_policy":"hard q=20 gate on every 1.0--1.5 mm contour before score-based selection",
        "guard_policy":"model-space SLM radial continuation parameter only; no bench conjugacy radius is claimed",
        "v10_best_iteration":best,"v10_optimisation_history":history,"v9_initialiser_history":v9_history,"propagation_support":prop_meta,
        "hardware_ready":False,
        "evidence_boundary":"corrected fields are numerical model-space predictions only; no corrected BeamGage image is experimental evidence",
    }
    (out/"v11_summary.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8"); print(json.dumps(result,indent=2)); return result


def main()->None:
    p=argparse.ArgumentParser(); p.add_argument("--source-dir",type=Path,default=EXP/"outputs"/"digital_twin_correction"); p.add_argument("--candidate-json",type=Path,default=EXP/"candidates"/"q20_detector_aware_model_v3_candidate.json"); p.add_argument("--residual-json",type=Path,default=EXP/"candidates"/"q20_miao_initialized_complex_residual_v1.json"); p.add_argument("--out",type=Path,default=ROOT/"outputs"/"validation"/"q20_slm2_topology_guarded_multiplane_v11"); a=p.parse_args(); run(a.source_dir,a.candidate_json,a.residual_json,a.out)

if __name__=="__main__": main()
