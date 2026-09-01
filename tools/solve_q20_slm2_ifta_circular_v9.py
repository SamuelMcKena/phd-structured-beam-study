"""q=20 correction v9: finite-4F iterative Fourier phase-only SLM2 synthesis.

This solver abandons the low-order intensity Jacobian used by v5/v6 and the
one-shot local complex encoding used by v7.  It treats the SLM2 -> finite 4F
+1-order relay as a linear wavefront-synthesis operator and iterates between
its input and output planes, Gerchberg-Saxton / IFTA style.

The desired field *after* the frozen detector-supported residual is an explicit
circular q=20 field obtained by radial complex projection of the nominal
selected-order field.  Therefore the desired selected-order field *before* the
residual is that circular field divided by the frozen complex residual.  At
each iteration the finite-4F output is relaxed toward this target, adjoint-
propagated back to SLM2, and the known incident amplitude is re-imposed while
retaining only phase.  The final phase is re-run through the normal pixelated
SLM + explicit finite-iris model and through the q=20 topology audit.

The frozen residual is never re-fit here.  Corrected images are numerical model
predictions only, not post-correction BeamGage measurements.
"""
from __future__ import annotations

import argparse
import json
import math
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
import solve_q20_slm2_iterative_circular_v8 as v8  # noqa: E402
from real_bmg_digital_twin_correction import FIT_WINDOW_M, Q, RELAY_N  # noqa: E402
from vbb_study.digital_twin.phase2a_contracts import canonical_hardware_manifest, hardware_value  # noqa: E402
from vbb_study.digital_twin.vortex_explicit_4f import (  # noqa: E402
    _propagate,
    explicit_4f_relay,
    physical_iris,
    thin_lens_transmission,
)
from vbb_study.digital_twin.vortex_system_route import build_multirate_system_route, build_system_route  # noqa: E402
from vbb_study.viz_fields import phase_winding  # noqa: E402

EPS = np.finfo(float).tiny
TWOPI = 2.0 * np.pi
THERMAL = "inferno"
PROD_N = 4096
IFTA_ITERS = 24
RELAXATION = (0.45, 0.60, 0.72, 0.82)
STATIC_GUARDS = (1.00, 0.85, 0.70, 0.55)


def savefig(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=500, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def wrap_pm_pi(a: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * np.asarray(a, float)))


def frozen_residual(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return np.asarray(d["phase_coefficients_rad"], float), np.asarray(d["log_amplitude_coefficients"], float), d


def config_from_files(candidate_path: Path, source_dir: Path):
    candidate = json.loads(Path(candidate_path).read_text(encoding="utf-8"))
    return v5.v4.config_from_candidate(candidate, source_dir), candidate


def assert_adjoint_supported(config) -> None:
    f = config.fourf
    if abs(float(f.lens1_axial_shift_m)) > 1e-15 or abs(float(f.lens2_axial_shift_m)) > 1e-15:
        raise RuntimeError("v9 adjoint currently requires zero 4F lens axial shifts")
    for le in (f.lens1, f.lens2):
        if tuple(map(float, le.decentre_m)) != (0.0, 0.0) or tuple(map(float, le.tilt_rad)) != (0.0, 0.0):
            raise RuntimeError("v9 adjoint currently requires parallel centred 4F lenses")
        if le.clear_radius_m is not None:
            raise RuntimeError("v9 adjoint currently requires no finite lens clear aperture")


def relay_constants(config, grid):
    manifest = canonical_hardware_manifest()
    wavelength = float(hardware_value(manifest, "wavelength_m"))
    carrier = float(hardware_value(manifest, "carrier_frequency_cpm"))
    f0 = float(hardware_value(manifest, "fourf_focal_length_m"))
    iris0 = float(hardware_value(manifest, "fourier_iris_radius_m"))
    f1 = f0 * float(config.fourf.lens1.focal_length_scale)
    f2 = f0 * float(config.fourf.lens2.focal_length_scale)
    l1, _ = thin_lens_transmission(grid, wavelength_m=wavelength, focal_length_m=f1)
    l2, _ = thin_lens_transmission(grid, wavelength_m=wavelength, focal_length_m=f2)
    from vbb_study.digital_twin.vortex_explicit_4f import nominal_order_position_m
    c0 = nominal_order_position_m(wavelength_m=wavelength, focal_length_m=f0, carrier_cpm=carrier)
    centre = (c0[0] + float(config.fourf.iris_offset_m[0]), c0[1] + float(config.fourf.iris_offset_m[1]))
    iris = physical_iris(grid, radius_m=iris0 * float(config.fourf.iris_radius_scale), centre_m=centre)
    return wavelength, carrier, f0, l1, l2, iris


def relay_forward(field_slm: np.ndarray, grid: dict, config) -> np.ndarray:
    wavelength, carrier, f0, _, _, _ = relay_constants(config, grid)
    r = explicit_4f_relay(
        np.asarray(field_slm, complex), grid,
        wavelength_m=wavelength,
        nominal_focal_length_m=f0,
        nominal_iris_radius_m=float(hardware_value(canonical_hardware_manifest(), "fourier_iris_radius_m")),
        nominal_carrier_cpm=carrier,
        error=config.fourf,
    )
    X = np.asarray(grid["X"], float)
    return np.asarray(r["output"], complex) * np.exp(1j * TWOPI * carrier * X)


def relay_adjoint(selected_order: np.ndarray, grid: dict, config) -> np.ndarray:
    wavelength, carrier, f0, l1, l2, iris = relay_constants(config, grid)
    X = np.asarray(grid["X"], float)
    u = np.asarray(selected_order, complex) * np.exp(-1j * TWOPI * carrier * X)
    u = _propagate(u, grid, wavelength, -f0)
    u = u * np.conj(l2)
    u = _propagate(u, grid, wavelength, -f0)
    u = u * iris
    u = _propagate(u, grid, wavelength, -f0)
    u = u * np.conj(l1)
    u = _propagate(u, grid, wavelength, -f0)
    return np.asarray(u, complex)


def fit_global_complex_scale(actual: np.ndarray, target: np.ndarray, mask: np.ndarray) -> complex:
    a = np.asarray(actual, complex)[mask].ravel(); t = np.asarray(target, complex)[mask].ravel()
    return np.vdot(a, t) / max(float(np.vdot(a, a).real), EPS)


def field_metrics(actual_after_residual: np.ndarray, circular_target: np.ndarray, grid: dict) -> dict:
    X = np.asarray(grid["X"], float); Y = np.asarray(grid["Y"], float); R = np.hypot(X,Y)
    amp_t = np.abs(circular_target)
    mask = (R >= 0.75e-3) & (R <= 3.0e-3) & (amp_t >= 0.04 * float(np.max(amp_t)))
    a = np.asarray(actual_after_residual, complex); t = np.asarray(circular_target, complex)
    c = fit_global_complex_scale(a,t,mask); ac = c*a
    ov = float(abs(np.vdot(ac[mask].ravel(),t[mask].ravel())) / max(np.linalg.norm(ac[mask].ravel())*np.linalg.norm(t[mask].ravel()),EPS))
    cn = float(np.sqrt(np.mean(np.abs(ac[mask]-t[mask])**2)) / max(np.sqrt(np.mean(np.abs(t[mask])**2)),EPS))
    an = np.abs(ac[mask]); tn = np.abs(t[mask]); amp = float(np.sqrt(np.mean((an-tn)**2)) / max(np.sqrt(np.mean(tn**2)),EPS))
    return {"complex_overlap":ov,"relative_complex_nrmse":cn,"relative_amplitude_nrmse":amp,"global_complex_scale_real":float(c.real),"global_complex_scale_imag":float(c.imag)}


def desired_fields(config, pcoef: np.ndarray, acoef: np.ndarray):
    base = build_system_route(f"V{Q}", grid_n=RELAY_N, config=config, window_m=FIT_WINDOW_M)
    grid = base["grid"]
    nominal = np.asarray(base["field_on_axicon_plane"], complex)
    circular = v8.radial_complex_projection(nominal, grid)
    phase_err, amp_err = v5.residual_maps(grid,pcoef,acoef)
    E = np.maximum(np.asarray(amp_err,float),1e-4) * np.exp(1j*np.asarray(phase_err,float))
    desired_pre = circular / E
    # Match desired selected-order power to the nominal selected-order power.
    desired_pre *= np.linalg.norm(nominal.ravel()) / max(np.linalg.norm(desired_pre.ravel()),EPS)
    return base, circular, np.asarray(desired_pre,complex), E


def ifta_solve(config, pcoef: np.ndarray, acoef: np.ndarray, out: Path):
    assert_adjoint_supported(config)
    base,circular,target_pre,E = desired_fields(config,pcoef,acoef)
    grid=base["grid"]; post1=np.asarray(base["post_slm1"],complex); u=np.asarray(base["post_slm2"],complex)
    X=np.asarray(grid["X"],float); Y=np.asarray(grid["Y"],float); R=np.hypot(X,Y)
    support=(R<=3.05e-3) & (np.abs(target_pre)>=0.025*float(np.max(np.abs(target_pre))))
    history=[]
    for it in range(IFTA_ITERS):
        selected=relay_forward(u,grid,config)
        after=selected*E
        before=field_metrics(after,circular,grid)
        beta=RELAXATION[min(it//6,len(RELAXATION)-1)]
        # Remove one arbitrary complex scale before enforcing the target.
        c=fit_global_complex_scale(selected,target_pre,support)
        aligned=c*selected
        constrained=aligned.copy()
        constrained[support]=(1.0-beta)*aligned[support]+beta*target_pre[support]
        # Outside the controlled region keep the current field (mixed-region freedom).
        back=relay_adjoint(constrained/max(abs(c),1e-9),grid,config)
        u=np.abs(post1)*np.exp(1j*np.angle(back))
        selected2=relay_forward(u,grid,config); after2=selected2*E
        afterm=field_metrics(after2,circular,grid)
        history.append({"iteration":it+1,"beta":float(beta),"before":before,"after":afterm})
        print(json.dumps(history[-1],indent=2))
    # Static phase relative to the accepted canonical SLM2 carrier field.
    ratio=u/np.where(np.abs(post1)>1e-12*np.max(np.abs(post1)),post1,1.0+0j)
    canonical=np.asarray(base["post_slm2"],complex)/np.where(np.abs(post1)>1e-12*np.max(np.abs(post1)),post1,1.0+0j)
    static=wrap_pm_pi(np.angle(ratio)-np.angle(canonical))
    np.save(out/"ifta_v9_full_static_phase_rad.npy",static.astype(np.float32))
    np.save(out/"ifta_v9_circular_selected_order_target.npy",circular.astype(np.complex64))
    np.save(out/"ifta_v9_desired_pre_residual_selected_order.npy",target_pre.astype(np.complex64))
    return static,history,circular


def static_guard(grid: dict, phase: np.ndarray, strength: float) -> np.ndarray:
    # Full phase away from the q=20 winding audit; taper only if topology requires it.
    R=np.hypot(np.asarray(grid["X"],float),np.asarray(grid["Y"],float))
    g=v8.smoothstep01((R-1.45e-3)/(1.85e-3-1.45e-3))
    return np.asarray(phase,float) * ((1.0-float(strength)) + float(strength)*g)


def corrected_route(config,N:int,static_phase:np.ndarray,pcoef:np.ndarray,acoef:np.ndarray):
    return build_multirate_system_route(
        f"V{Q}",relay_grid_n=RELAY_N,propagation_grid_n=int(N),window_m=FIT_WINDOW_M,config=config,
        slm2_static_phase_map_rad=np.asarray(static_phase,float),
        axicon_input_phase_map_rad=v5.residual_maps(build_multirate_system_route(f"V{Q}",relay_grid_n=RELAY_N,propagation_grid_n=int(N),window_m=FIT_WINDOW_M,config=config)["grid"],pcoef,acoef)[0],
        axicon_input_amplitude_map=v5.residual_maps(build_multirate_system_route(f"V{Q}",relay_grid_n=RELAY_N,propagation_grid_n=int(N),window_m=FIT_WINDOW_M,config=config)["grid"],pcoef,acoef)[1],
    )


def build_corrected(config,N:int,static_phase:np.ndarray,pcoef:np.ndarray,acoef:np.ndarray):
    nominal=build_multirate_system_route(f"V{Q}",relay_grid_n=RELAY_N,propagation_grid_n=int(N),window_m=FIT_WINDOW_M,config=config,slm2_static_phase_map_rad=np.asarray(static_phase,float))
    ph,am=v5.residual_maps(nominal["grid"],pcoef,acoef)
    return build_multirate_system_route(f"V{Q}",relay_grid_n=RELAY_N,propagation_grid_n=int(N),window_m=FIT_WINDOW_M,config=config,slm2_static_phase_map_rad=np.asarray(static_phase,float),axicon_input_phase_map_rad=ph,axicon_input_amplitude_map=am)


def target_stack(config,z_abs):
    nom=build_multirate_system_route(f"V{Q}",relay_grid_n=RELAY_N,propagation_grid_n=2048,window_m=FIT_WINDOW_M,config=config)
    s=v5.detector_stack(nom,z_abs)
    return np.stack([v5.radial_symmetrise_plane(p,v5.AXIS_UM) for p in s])


def evaluate_guard(guard_strength,config,z_abs,target,static,pcoef,acoef):
    base=build_system_route(f"V{Q}",grid_n=RELAY_N,config=config,window_m=FIT_WINDOW_M)
    ph=static_guard(base["grid"],static,float(guard_strength))
    pos=v5.build_route(config,2048,slm2_phase=None,pcoef=pcoef,acoef=acoef)
    cor=build_corrected(config,2048,ph,pcoef,acoef)
    pdet=v5.detector_stack(pos,z_abs); cdet=v5.detector_stack(cor,z_abs)
    pm=v5.concentric_metrics(pdet,target,v5.INNER_VALID); cm=v5.concentric_metrics(cdet,target,v5.INNER_VALID)
    wd,ok=v5.winding(cor)
    return {"guard_strength":float(guard_strength),"topology_q20_all_contours":bool(ok),"winding":wd,"positive":pm,"corrected":cm,"corrected_objective":v5.objective(cm),"principal_ring_cv_reduction_fraction":float(1.0-cm["mean_principal_ring_azimuth_cv"]/max(pm["mean_principal_ring_azimuth_cv"],EPS)),"mirror_rmse_reduction_fraction":float(1.0-cm["mirror_rmse"]/max(pm["mirror_rmse"],EPS))}


def production(guard_strength,config,z_abs,z_rel,target,static,pcoef,acoef,out):
    base=build_system_route(f"V{Q}",grid_n=RELAY_N,config=config,window_m=FIT_WINDOW_M)
    phase=static_guard(base["grid"],static,float(guard_strength))
    pos=v5.build_route(config,PROD_N,slm2_phase=None,pcoef=pcoef,acoef=acoef)
    cor=build_corrected(config,PROD_N,phase,pcoef,acoef)
    pdet=v5.detector_stack(pos,z_abs); cdet=v5.detector_stack(cor,z_abs); popt=v5.optical_stack(pos,z_abs); copt=v5.optical_stack(cor,z_abs)
    wc,ok=v5.winding(cor)
    groups={"inner_train":v5.INNER_TRAIN,"inner_validation":v5.INNER_VALID,"legacy_heldout":v5.LEGACY_HELD,"all_planes":np.arange(len(z_rel),dtype=int)}
    metrics={}
    for name,ids in groups.items():
        metrics[name]={"detector_positive":v5.concentric_metrics(pdet,target,ids),"detector_corrected":v5.concentric_metrics(cdet,target,ids),"optical_positive":v5.concentric_metrics(popt,target,ids),"optical_corrected":v5.concentric_metrics(copt,target,ids)}
    np.save(out/"model_space_slm2_static_phase_ifta_v9_rad.npy",phase.astype(np.float32))
    np.savez_compressed(out/"ifta_v9_4096_display_arrays.npz",axis_um=v5.AXIS_UM,z_relative_mm=z_rel,concentric_target=target.astype(np.float32),detector_positive=pdet.astype(np.float32),detector_corrected=cdet.astype(np.float32),optical_positive=popt.astype(np.float32),optical_corrected=copt.astype(np.float32))
    ext=[v5.AXIS_UM[0],v5.AXIS_UM[-1],v5.AXIS_UM[0],v5.AXIS_UM[-1]]; ids=[1,5,9,13,17]
    fig,axs=plt.subplots(3,len(ids),figsize=(15.5,8.8),constrained_layout=True)
    for col,iz in enumerate(ids):
        for row,(stack,label) in enumerate(((pdet,"diagnosed model"),(target,"explicit concentric target"),(cdet,"IFTA corrected"))):
            axs[row,col].imshow(stack[iz],origin="lower",extent=ext,cmap=THERMAL,vmin=0,vmax=1,interpolation="nearest"); axs[row,col].set_aspect("equal"); axs[row,col].set_xticks([]); axs[row,col].set_yticks([])
            if row==0: axs[row,col].set_title(f"z = {z_rel[iz]:.0f} mm")
            if col==0: axs[row,col].set_ylabel(label,fontweight="bold")
    fig.suptitle("q=20 v9: finite-4F iterative Fourier phase-only correction",fontsize=14,fontweight="bold"); savefig(fig,out/"poster_ifta_v9_multiplane")
    ids2=[5,11,17]; fig,axs=plt.subplots(2,3,figsize=(11.2,7.0),constrained_layout=True)
    for col,iz in enumerate(ids2):
        for row,stack in enumerate((popt,copt)):
            axs[row,col].imshow(stack[iz],origin="lower",extent=ext,cmap=THERMAL,vmin=0,vmax=1,interpolation="nearest"); axs[row,col].set_aspect("equal"); axs[row,col].set_xticks([]); axs[row,col].set_yticks([])
            if row==0: axs[row,col].set_title(f"z = {z_rel[iz]:.0f} mm")
            if col==0: axs[row,col].set_ylabel("diagnosed" if row==0 else "IFTA corrected",fontweight="bold")
    fig.suptitle("4096-grid optical field: IFTA before/after",fontsize=13,fontweight="bold"); savefig(fig,out/"poster_ifta_v9_optical_before_after")
    return {"production_grid_n":PROD_N,"guard_strength":float(guard_strength),"topology_q20_all_contours":bool(ok),"winding_corrected":wc,"metrics":metrics}


def run(source_dir:Path,candidate_json:Path,residual_json:Path,out:Path)->dict:
    source_dir=Path(source_dir); out=Path(out); out.mkdir(parents=True,exist_ok=True)
    config,candidate=config_from_files(candidate_json,source_dir); pcoef,acoef,frozen=frozen_residual(residual_json)
    static,history,circular=ifta_solve(config,pcoef,acoef,out)
    summary0=json.loads((source_dir/"run_summary.json").read_text(encoding="utf-8")); z_rel=np.asarray(summary0["data"]["z_relative_mm"],float); z0=float(candidate["physical_nuisance"]["selected_z0_mm"]); z_abs=(z0+z_rel)*1e-3
    target=target_stack(config,z_abs)
    sweep=[]
    for g in STATIC_GUARDS:
        s=evaluate_guard(g,config,z_abs,target,static,pcoef,acoef); sweep.append(s); print(json.dumps(s,indent=2))
    (out/"ifta_v9_guard_sweep.json").write_text(json.dumps(sweep,indent=2)+"\n",encoding="utf-8")
    passing=[s for s in sweep if s["topology_q20_all_contours"]]
    selected=min(passing,key=lambda s:s["corrected_objective"]) if passing else min(sweep,key=lambda s:s["corrected_objective"])
    prod=production(selected["guard_strength"],config,z_abs,z_rel,target,static,pcoef,acoef,out)
    held=prod["metrics"]["legacy_heldout"]; pos=held["detector_positive"]; cor=held["detector_corrected"]
    cvred=float(1.0-cor["mean_principal_ring_azimuth_cv"]/max(pos["mean_principal_ring_azimuth_cv"],EPS)); mirred=float(1.0-cor["mirror_rmse"]/max(pos["mirror_rmse"],EPS))
    acceptance={"q20_topology_preserved":bool(prod["topology_q20_all_contours"]),"legacy_heldout_principal_ring_cv_reduction_fraction":cvred,"legacy_heldout_mirror_rmse_reduction_fraction":mirred,"legacy_heldout_radial_profile_corr":float(cor["mean_radial_profile_corr"]),"legacy_heldout_target_r":float(cor["mean_r"]),"legacy_heldout_target_nrmse":float(cor["mean_nrmse"]),"passes_concentricity_gate":bool(prod["topology_q20_all_contours"] and cvred>=0.30 and mirred>=0.25 and cor["mean_radial_profile_corr"]>=0.92)}
    result={"status":"q20_finite_4f_ifta_circular_candidate_v9","frozen_residual_source":str(Path(residual_json).name),"ifta_iterations":IFTA_ITERS,"ifta_history":history,"selected_validation_guard":selected,"production_validation":prod,"acceptance":acceptance,"target_policy":"explicit circular q20 complex selected-order target plus explicitly circular nominal detector target; no measured angular distortion is rewarded","hardware_ready":False,"evidence_boundary":"corrected fields are numerical model-space predictions only; no corrected BeamGage frame is experimental evidence","literature_basis":["Gerchberg-Saxton / iterative Fourier transform phase-only synthesis","J. S. Liu and M. R. Taghizadeh, Opt. Lett. 27, 1463-1465 (2002), doi:10.1364/OL.27.001463"]}
    (out/"ifta_v9_summary.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8"); print(json.dumps(result,indent=2)); return result


def main()->None:
    p=argparse.ArgumentParser(); p.add_argument("--source-dir",type=Path,default=EXP/"outputs"/"digital_twin_correction"); p.add_argument("--candidate-json",type=Path,default=EXP/"candidates"/"q20_detector_aware_model_v3_candidate.json"); p.add_argument("--residual-json",type=Path,default=EXP/"candidates"/"q20_miao_initialized_complex_residual_v1.json"); p.add_argument("--out",type=Path,default=ROOT/"outputs"/"validation"/"q20_slm2_ifta_circular_v9"); a=p.parse_args(); run(a.source_dir,a.candidate_json,a.residual_json,a.out)

if __name__=="__main__": main()
