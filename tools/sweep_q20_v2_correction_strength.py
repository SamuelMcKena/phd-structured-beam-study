"""Find the strongest v2 SLM2 correction that preserves q=20 topology.

The 4096 production check showed that the compact v2 phase correction gives the
best intensity closure so far but changes the innermost winding contour from 20
to 18. The later full adjoint solve was worse. This script therefore tests a
simple physically meaningful regularisation: scale the *same* v2 additive SLM2
phase map by alpha, choose the best alpha that preserves q=20 on every tested
contour, then validate that candidate on the converged N=4096 optical grid and
through the explicit 5.5 um BeamGage pixel response.

This remains a model-space candidate, not a hardware-ready phase map: the bench
SLM2 conjugacy/coordinate map and 1030-nm LUT are still uncalibrated.
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "notebooks" / "experimental" / "axicon_aberration_correction"
TOOLS = ROOT / "tools"
for p in (ROOT, EXP, TOOLS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from real_bmg_digital_twin_correction import (
    AxiconError, FourFError, FIT_WINDOW_M, PIXEL_M, Q, RELAY_N, SystemErrorConfig,
)
from optimize_q20_slm2_detector_closure_v2 import phase_basis, phase_from_coefficients
from vbb_study.digital_twin.detector_response import sample_camera_response
from vbb_study.digital_twin.residual_phase_fit import angular_phase_from_coefficients
from vbb_study.digital_twin.vortex_continuous_propagation import (
    build_fixed_support_spectrum, native_field_at_z,
)
from vbb_study.digital_twin.vortex_system_route import build_multirate_system_route
from vbb_study.digital_twin.phase2a_contracts import canonical_hardware_manifest, hardware_value
from vbb_study.viz_fields import phase_winding

EPS = np.finfo(float).tiny
SWEEP_N = 3072
PROD_N = 4096
AXIS_UM = np.linspace(-180.0, 180.0, 241)
# The coarse pass showed alpha=0.40 preserves q=20 while alpha=0.60 already
# creates an 18-charge inner contour. Resolve that transition directly.
ALPHAS = np.asarray([0.40, 0.45, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60], dtype=float)
WINDING_RADII_MM = (1.0, 1.1, 1.2, 1.3, 1.4, 1.5)
THERMAL = "inferno"


def norm(a):
    a = np.maximum(np.asarray(a, float), 0.0)
    return a / max(float(a.max()), EPS)


def native_crop(field, x):
    I = np.abs(np.asarray(field)) ** 2
    x = np.asarray(x, float)
    pix = np.interp(AXIS_UM * 1e-6, x, np.arange(len(x)))
    yy, xx = np.meshgrid(pix, pix, indexing="ij")
    return norm(ndimage.map_coordinates(I, [yy, xx], order=1, mode="constant", cval=0.0))


def detector_crop(field, x):
    shown, _ = sample_camera_response(
        (np.abs(np.asarray(field)) ** 2)[None], np.asarray(x, float), AXIS_UM * 1e-6,
        pixel_pitch_m=PIXEL_M, quadrature_n=3,
    )
    return norm(shown[0])


def metrics(a, b):
    X, Y = np.meshgrid(AXIS_UM, AXIS_UM, indexing="xy")
    roi = np.hypot(X, Y) <= 145.0
    av, bv = np.asarray(a)[roi], np.asarray(b)[roi]
    return float(np.corrcoef(av, bv)[0, 1]), float(np.sqrt(np.mean((av - bv) ** 2)))


def mirror_metrics(im):
    X, Y = np.meshgrid(AXIS_UM, AXIS_UM, indexing="xy")
    R = np.hypot(X, Y)
    m = (R >= 20.0) & (R <= 140.0)
    return (
        float(np.sqrt(np.mean((im[m] - im[:, ::-1][m]) ** 2))),
        float(np.sqrt(np.mean((im[m] - im[::-1, :][m]) ** 2))),
    )


def radial(im, dr=1.5, rmax=150.0):
    X, Y = np.meshgrid(AXIS_UM, AXIS_UM, indexing="xy")
    R = np.hypot(X, Y)
    edges = np.arange(0.0, rmax + dr, dr)
    ids = np.digitize(R.ravel(), edges) - 1
    good = (ids >= 0) & (ids < len(edges) - 1)
    sums = np.bincount(ids[good], weights=np.asarray(im).ravel()[good], minlength=len(edges) - 1)
    counts = np.bincount(ids[good], minlength=len(edges) - 1)
    return 0.5 * (edges[:-1] + edges[1:]), sums / np.maximum(counts, 1)


def load_problem():
    src = EXP / "outputs" / "digital_twin_correction"
    residual = json.loads((EXP / "candidates" / "q20_detector_aware_axicon_residual_candidate.json").read_text())
    seed = json.loads((EXP / "candidates" / "q20_detector_domain_slm2_v2_candidate.json").read_text())
    rs = json.loads((src / "run_summary.json").read_text())
    scale = float(rs["effective_axicon"]["effective_scale_relative_to_repository_2deg_assumption"])
    zscan = pd.read_csv(src / "full_route_z_registration_scan.csv")
    z0 = float(zscan.loc[zscan.selected.astype(bool), "value"].iloc[0])
    zrel = np.arange(-17.0, 1.0)
    zabs = (z0 + zrel) * 1e-3
    cfg = SystemErrorConfig(
        fourf=FourFError(iris_radius_scale=1.0),
        axicon=AxiconError(base_angle_scale=scale),
    )
    wavelength = float(hardware_value(canonical_hardware_manifest(), "wavelength_m"))

    relay_only = build_multirate_system_route(
        f"V{Q}", relay_grid_n=RELAY_N, propagation_grid_n=RELAY_N,
        window_m=FIT_WINDOW_M, config=cfg,
    )
    basis, names = phase_basis(relay_only["relay_route"]["grid"])
    lookup = dict(zip(seed["basis_names"], seed["coefficients_rad"]))
    coeff = np.asarray([float(lookup.get(n, 0.0)) for n in names])
    slm2 = phase_from_coefficients(basis, coeff)
    del relay_only, basis
    gc.collect()
    return residual, cfg, wavelength, zrel, zabs, slm2


def fine_residual(grid, residual):
    x = np.asarray(grid["x"], float)
    X, Y = np.meshgrid(x, x, indexing="xy")
    theta = np.arctan2(Y, X)
    return angular_phase_from_coefficients(
        theta, np.asarray(residual["coefficients_rad"], float),
        modes=tuple(residual["angular_modes"]),
    )


def winding_dict(route):
    out = {}
    vals = []
    for rmm in WINDING_RADII_MM:
        w = float(phase_winding(route["post_axicon"], route["grid"], rmm * 1e-3, n_phi=720))
        out[f"radius_{rmm:.1f}_mm"] = w
        vals.append(w)
    return out, bool(np.all(np.abs(np.asarray(vals) - float(Q)) <= 0.25))


def evaluate_alpha(alpha, *, N, residual, cfg, wavelength, zrel, zabs, slm2, nominal_route, nominal_prop, nominal_optical=None, nominal_detector=None):
    x = np.asarray(nominal_route["grid"]["x"], float)
    err = fine_residual(nominal_route["grid"], residual)
    cor = build_multirate_system_route(
        f"V{Q}", relay_grid_n=RELAY_N, propagation_grid_n=N,
        window_m=FIT_WINDOW_M, config=cfg,
        slm2_static_phase_map_rad=float(alpha) * slm2,
        axicon_input_phase_map_rad=err,
    )
    pcorr = build_fixed_support_spectrum(
        cor["post_axicon"], cor["grid"], wavelength_m=wavelength,
        z_max_m=max(abs(zabs)), minimum_retained_spectral_power=0.99,
    )
    rows = []
    optical_corr, detector_corr = [], []
    if nominal_optical is None or nominal_detector is None:
        nominal_optical, nominal_detector = [], []
        make_nominal = True
    else:
        make_nominal = False
    for i, zz in enumerate(zabs):
        fc = native_field_at_z(pcorr, float(zz))
        oc = native_crop(fc, x)
        dc = detector_crop(fc, x)
        if make_nominal:
            fn = native_field_at_z(nominal_prop, float(zz))
            on = native_crop(fn, x)
            dn = detector_crop(fn, x)
            nominal_optical.append(on)
            nominal_detector.append(dn)
            del fn
        else:
            on = nominal_optical[i]
            dn = nominal_detector[i]
        ro, eo = metrics(oc, on)
        rd, ed = metrics(dc, dn)
        mx, my = mirror_metrics(oc)
        rows.append(dict(
            alpha=float(alpha), z_relative_mm=float(zrel[i]),
            optical_r=ro, optical_nrmse=eo, detector_r=rd, detector_nrmse=ed,
            optical_xmirror_rmse=mx, optical_ymirror_rmse=my,
        ))
        optical_corr.append(oc)
        detector_corr.append(dc)
        del fc
    wd, topology_ok = winding_dict(cor)
    df = pd.DataFrame(rows)
    summary = dict(
        alpha=float(alpha), grid_n=int(N), topology_q20_all_contours=topology_ok,
        mean_optical_r=float(df.optical_r.mean()),
        mean_optical_nrmse=float(df.optical_nrmse.mean()),
        mean_detector_r=float(df.detector_r.mean()),
        mean_detector_nrmse=float(df.detector_nrmse.mean()),
        mean_optical_xmirror_rmse=float(df.optical_xmirror_rmse.mean()),
        mean_optical_ymirror_rmse=float(df.optical_ymirror_rmse.mean()),
        winding_corrected=wd,
    )
    del cor, pcorr, err
    gc.collect()
    return summary, df, nominal_optical, nominal_detector, np.stack(optical_corr), np.stack(detector_corr)


def main():
    out = ROOT / "outputs" / "validation" / "q20_v2_strength_sweep"
    out.mkdir(parents=True, exist_ok=True)
    residual, cfg, wavelength, zrel, zabs, slm2 = load_problem()

    nominal = build_multirate_system_route(
        f"V{Q}", relay_grid_n=RELAY_N, propagation_grid_n=SWEEP_N,
        window_m=FIT_WINDOW_M, config=cfg,
    )
    pnom = build_fixed_support_spectrum(
        nominal["post_axicon"], nominal["grid"], wavelength_m=wavelength,
        z_max_m=max(abs(zabs)), minimum_retained_spectral_power=0.99,
    )
    sweep_summaries = []
    sweep_rows = []
    nom_opt = nom_det = None
    for alpha in ALPHAS:
        s, df, nom_opt, nom_det, _, _ = evaluate_alpha(
            alpha, N=SWEEP_N, residual=residual, cfg=cfg, wavelength=wavelength,
            zrel=zrel, zabs=zabs, slm2=slm2, nominal_route=nominal,
            nominal_prop=pnom, nominal_optical=nom_opt, nominal_detector=nom_det,
        )
        sweep_summaries.append(s)
        sweep_rows.append(df)
        print(json.dumps(s))
    sweep_df = pd.DataFrame([{k: v for k, v in s.items() if k != "winding_corrected"} for s in sweep_summaries])
    sweep_df.to_csv(out / "q20_v2_strength_sweep_3072.csv", index=False)
    pd.concat(sweep_rows, ignore_index=True).to_csv(out / "q20_v2_strength_sweep_3072_vs_z.csv", index=False)
    (out / "q20_v2_strength_sweep_3072.json").write_text(json.dumps(sweep_summaries, indent=2))

    passing = [s for s in sweep_summaries if s["topology_q20_all_contours"]]
    if not passing:
        selected = min(sweep_summaries, key=lambda s: abs(s["alpha"]))
        selection_status = "no_topology_preserving_candidate_in_refined_range"
    else:
        selected = max(passing, key=lambda s: (s["mean_optical_r"], s["mean_detector_r"]))
        selection_status = "best_topology_preserving_optical_closure"
    alpha_star = float(selected["alpha"])

    del nominal, pnom, nom_opt, nom_det
    gc.collect()

    nominal4 = build_multirate_system_route(
        f"V{Q}", relay_grid_n=RELAY_N, propagation_grid_n=PROD_N,
        window_m=FIT_WINDOW_M, config=cfg,
    )
    pnom4 = build_fixed_support_spectrum(
        nominal4["post_axicon"], nominal4["grid"], wavelength_m=wavelength,
        z_max_m=max(abs(zabs)), minimum_retained_spectral_power=0.99,
    )
    prod, prod_df, nom4_opt, nom4_det, cor4_opt, cor4_det = evaluate_alpha(
        alpha_star, N=PROD_N, residual=residual, cfg=cfg, wavelength=wavelength,
        zrel=zrel, zabs=zabs, slm2=slm2, nominal_route=nominal4, nominal_prop=pnom4,
    )
    prod_df.to_csv(out / "q20_v2_selected_4096_vs_z.csv", index=False)
    np.save(out / "model_space_slm2_phase_v2_scaled_rad.npy", alpha_star * slm2)
    np.savez_compressed(
        out / "q20_v2_selected_4096_display_arrays.npz",
        axis_um=AXIS_UM, z_relative_mm=zrel,
        optical_nominal=np.asarray(nom4_opt, dtype=np.float32),
        optical_corrected=cor4_opt.astype(np.float32),
        detector_nominal=np.asarray(nom4_det, dtype=np.float32),
        detector_corrected=cor4_det.astype(np.float32),
    )

    result = dict(
        status="topology_regularised_v2_model_candidate",
        selection_status=selection_status,
        selected_alpha=alpha_star,
        sweep_grid_n=SWEEP_N,
        production_grid_n=PROD_N,
        production_validation=prod,
        hardware_ready=False,
        hardware_blockers=[
            "SLM2 to axicon/input-plane coordinate transform is not bench-calibrated",
            "SLM2 conjugacy/parity/rotation/scale are not independently measured",
            "SLM2 1030-nm phase LUT/stroke is not calibrated",
            "candidate has not yet been tested on a post-SLM measured z-stack",
        ],
    )
    (out / "q20_v2_strength_selected_summary.json").write_text(json.dumps(result, indent=2))

    rep = int(np.argmin(abs(zrel + 10.0)))
    fig, axs = plt.subplots(2, 3, figsize=(15.5, 9.0), constrained_layout=True)
    good = sweep_df["topology_q20_all_contours"].astype(bool).to_numpy()
    axs[0, 0].plot(sweep_df.alpha, sweep_df.mean_optical_r, "o-", label="optical r")
    axs[0, 0].plot(sweep_df.alpha, sweep_df.mean_detector_r, "s--", label="detector r")
    axs[0, 0].axvline(alpha_star, ls=":", lw=1.5, label=f"selected alpha={alpha_star:.2f}")
    if np.any(good):
        axs[0, 0].scatter(
            sweep_df.alpha[good], sweep_df.mean_optical_r[good], s=65,
            facecolors="none", edgecolors="k", label="q=20 all contours",
        )
    axs[0, 0].set(
        xlabel="v2 correction strength alpha", ylabel="mean correlation",
        ylim=(0, 1.02), title="Correction-strength sweep",
    )
    axs[0, 0].grid(alpha=.25)
    axs[0, 0].legend(fontsize=8)

    axs[1, 0].plot(sweep_df.alpha, sweep_df.mean_optical_nrmse, "o-", label="optical NRMSE")
    axs[1, 0].plot(sweep_df.alpha, sweep_df.mean_detector_nrmse, "s--", label="detector NRMSE")
    axs[1, 0].axvline(alpha_star, ls=":", lw=1.5)
    axs[1, 0].set(
        xlabel="v2 correction strength alpha", ylabel="mean NRMSE", title="Intensity error",
    )
    axs[1, 0].grid(alpha=.25)
    axs[1, 0].legend(fontsize=8)

    ext = [AXIS_UM[0], AXIS_UM[-1], AXIS_UM[0], AXIS_UM[-1]]
    panels = [
        (np.asarray(nom4_opt)[rep], "Nominal optical field"),
        (cor4_opt[rep], f"Selected optical field (alpha={alpha_star:.2f})"),
        (cor4_det[rep], "Selected predicted BeamGage"),
    ]
    for j, (im, title) in enumerate(panels):
        ax = axs[0, 1 + j] if j < 2 else axs[1, 2]
        ax.imshow(im, origin="lower", extent=ext, cmap=THERMAL, vmin=0, vmax=1)
        ax.set(title=title, xlabel="x (um)", ylabel="y (um)", aspect="equal")

    rr, pn = radial(np.asarray(nom4_opt)[rep])
    _, pc = radial(cor4_opt[rep])
    _, pdet = radial(cor4_det[rep])
    axs[1, 1].plot(rr, pn, lw=1.7, label="nominal optical")
    axs[1, 1].plot(rr, pc, "--", lw=1.5, label="selected optical")
    axs[1, 1].plot(rr, pdet, ":", lw=1.5, label="predicted BeamGage")
    axs[1, 1].set(
        xlim=(0, 140), xlabel="radius (um)", ylabel="azimuthal mean intensity",
        title="Representative radial profile",
    )
    axs[1, 1].grid(alpha=.25)
    axs[1, 1].legend(fontsize=8)

    fig.suptitle(
        "q=20 v2 SLM2 correction-strength regularisation: preserve topology before maximising closure"
    )
    fig.savefig(out / "26_q20_v2_topology_regularised_4096.png", dpi=600, bbox_inches="tight")
    fig.savefig(out / "26_q20_v2_topology_regularised_4096.pdf", bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
