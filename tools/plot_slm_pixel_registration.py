"""Figures for the sub-pixel SLM pixel-lattice registration study.

Encoding choices, stated because they are deliberate:

* The sub-pixel offset ``delta`` is an *ordered magnitude*, so offset series use a
  single-hue sequential ramp rather than categorical hues.
* The registration degrees of freedom are *identities*, so they use the study's
  fixed categorical order (blue, vermillion, green, orange -- the Okabe-Ito
  subset that clears adjacent-pair CVD separation), always with a distinct
  marker as secondary encoding so identity never rests on colour alone.
* Signed field differences use the diverging map with a neutral midpoint and
  symmetric limits, so zero difference reads as zero.
* Vortex phase is domain coloured, hue for phase and value for ``|U|/max``.  A
  bare cyclic colormap shows random colour in the dark core where the phase is
  numerically meaningless; VIZ_AUDIT records that defect against an earlier
  figure and it is not repeated here.
* Metrics spanning decades are plotted on log axes, and no figure uses two y
  scales.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

DOF_COLOURS = {
    "slm1": "#0072B2",
    "slm2": "#D55E00",
    "common": "#009E73",
    "differential": "#E69F00",
}
DOF_MARKERS = {"slm1": "o", "slm2": "s", "common": "^", "differential": "D"}
DOF_LABELS = {
    "slm1": "SLM1 only",
    "slm2": "SLM2 only",
    "common": "both, common",
    "differential": "both, differential",
}


def _read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _f(row: dict[str, Any], key: str) -> float:
    v = row.get(key, "")
    if v in ("", None, "nan"):
        return float("nan")
    try:
        return float(v)
    except ValueError:
        return float("nan")


def _domain_colour(field: np.ndarray) -> np.ndarray:
    """Hue = phase, value = |U| / max.  Keeps the dark core visually dark."""

    from matplotlib.colors import hsv_to_rgb

    amp = np.abs(field)
    peak = float(np.max(amp)) or 1.0
    hsv = np.zeros(field.shape + (3,), dtype=float)
    hsv[..., 0] = (np.angle(field) + np.pi) / (2.0 * np.pi)
    hsv[..., 1] = 1.0
    hsv[..., 2] = np.clip(amp / peak, 0.0, 1.0) ** 0.5
    return hsv_to_rgb(hsv)


def _crop(arr: np.ndarray, x: np.ndarray, half_m: float) -> tuple[np.ndarray, np.ndarray]:
    keep = np.abs(x) <= float(half_m)
    return x[keep], np.asarray(arr)[np.ix_(keep, keep)]


# --------------------------------------------------------------------------
# curve figures
# --------------------------------------------------------------------------

def figure_offset_curves(rows, out_dir, style, metric, ylabel, name):
    import matplotlib.pyplot as plt

    charges = sorted({int(_f(r, "charge")) for r in rows if r["series"] in ("sweep", "registered")})
    if not charges:
        return None
    ncol = min(3, len(charges))
    nrow = int(math.ceil(len(charges) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.2 * nrow),
                             squeeze=False, sharex=True)
    for i, ell in enumerate(charges):
        ax = axes[i // ncol][i % ncol]
        for dof, colour in DOF_COLOURS.items():
            pts = [(_f(r, "delta_px"), _f(r, metric)) for r in rows
                   if r["series"] == "sweep" and int(_f(r, "charge")) == ell and r["dof"] == dof]
            zero = [(0.0, _f(r, metric)) for r in rows
                    if r["series"] == "registered" and int(_f(r, "charge")) == ell]
            pts = sorted(set(zero + pts))
            pts = [(a, b) for a, b in pts if np.isfinite(b)]
            if len(pts) < 2:
                continue
            ax.plot([p[0] for p in pts], [max(p[1], 1e-18) for p in pts],
                    marker=DOF_MARKERS[dof], color=colour, lw=1.6, ms=5,
                    label=DOF_LABELS[dof])
        ax.set_yscale("log")
        ax.set_title(f"$\\ell$ = {ell}", fontsize=10)
        ax.grid(True, which="both", alpha=0.25, lw=0.5)
        if i % ncol == 0:
            ax.set_ylabel(ylabel)
        if i // ncol == nrow - 1:
            ax.set_xlabel("beam offset within the pixel  $\\delta / p$")
    for j in range(len(charges), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False,
                   bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Sub-pixel registration sensitivity by vortex charge", fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    return style.save_figure(
        fig, out_dir / name,
        "Change in the complex field at the axicon input plane as the incident beam is "
        "stepped across one SLM pixel, relative to the same route registered at zero "
        "offset. One panel per vortex charge; colour and marker identify which panel "
        "was translated. Log vertical axis; dimensionless infidelity 1 - |<E,Eref>|^2 / "
        "(<E,E><Eref,Eref>).",
        metadata={"metric": metric, "encoding": "categorical dof + log y"},
    )


def figure_charge_scaling(summary, out_dir, style, name):
    import matplotlib.pyplot as plt

    metric = "before_infidelity_vs_registered0__peak_to_peak"
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    any_series = False
    for dof, colour in DOF_COLOURS.items():
        pts = sorted((int(_f(r, "charge")), _f(r, metric)) for r in summary
                     if r["dof"] == dof)
        pts = [(e, v) for e, v in pts if e > 0 and np.isfinite(v) and v > 0]
        if len(pts) < 2:
            continue
        any_series = True
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=DOF_MARKERS[dof],
                color=colour, lw=1.6, ms=6, label=DOF_LABELS[dof])
    if not any_series:
        plt.close(fig)
        return None
    # reference slopes anchored on the first plotted point
    ref = sorted((int(_f(r, "charge")), _f(r, metric)) for r in summary
                 if r["dof"] == "slm1" and int(_f(r, "charge")) > 0
                 and np.isfinite(_f(r, metric)) and _f(r, metric) > 0)
    if ref:
        e0, v0 = ref[0]
        ells = np.array([e for e, _ in ref], dtype=float)
        for power, ls in ((1.0, ":"), (2.0, "--")):
            ax.plot(ells, v0 * (ells / e0) ** power, ls=ls, color="0.45", lw=1.2,
                    label=f"$\\ell^{{{int(power)}}}$ guide")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("vortex charge  $\\ell$")
    ax.set_ylabel("registration-induced infidelity, peak to peak")
    ax.set_title("Higher charges are more sensitive to where the beam sits in a pixel")
    ax.grid(True, which="both", alpha=0.25, lw=0.5)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return style.save_figure(
        fig, out_dir / name,
        "Peak-to-peak variation of the axicon-plane field infidelity across a full "
        "sub-pixel offset sweep, against vortex charge, for each registration degree "
        "of freedom. Grey guides are pure power laws anchored on the lowest plotted "
        "charge and are not fits. Both axes logarithmic; infidelity dimensionless.",
        metadata={"metric": metric},
    )


def figure_beam_size(summary, out_dir, style, name):
    import matplotlib.pyplot as plt

    metric = "before_infidelity_vs_registered0__peak_to_peak"
    groups = defaultdict(list)
    for r in summary:
        if r["dof"] != "slm1":
            continue
        groups[int(_f(r, "charge"))].append((_f(r, "beam_radius_px"), _f(r, metric)))
    groups = {k: sorted(v) for k, v in groups.items() if len(v) > 1}
    if not groups:
        return None
    cmap = plt.get_cmap("viridis")
    charges = sorted(groups)
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for i, ell in enumerate(charges):
        pts = [(a, b) for a, b in groups[ell] if np.isfinite(b) and b > 0]
        if len(pts) < 2:
            continue
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=5, lw=1.6,
                color=cmap(0.12 + 0.76 * i / max(len(charges) - 1, 1)),
                label=f"$\\ell$ = {ell}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("beam radius on the SLM  [pixels]   (smaller to the right)")
    ax.set_ylabel("registration-induced infidelity, peak to peak")
    ax.set_title("Registration sensitivity as the beam shrinks toward the pixel pitch")
    ax.grid(True, which="both", alpha=0.25, lw=0.5)
    ax.legend(frameon=False, fontsize=9, title="charge", title_fontsize=9)
    fig.tight_layout()
    return style.save_figure(
        fig, out_dir / name,
        "Peak-to-peak registration-induced infidelity against beam radius expressed in "
        "SLM pixels, for SLM1 translation. The horizontal axis is reversed so that the "
        "strongly undersampled regime lies to the right. Both axes logarithmic. The "
        "Fourier-plane iris transmission for each beam radius is reported separately "
        "because shrinking the beam also fills the iris, which is a confounding change.",
        metadata={"metric": metric},
    )


def figure_iris_confound(rows, out_dir, style, name):
    import matplotlib.pyplot as plt

    pts = sorted({(_f(r, "beam_radius_px"), _f(r, "iris_selected_fraction"))
                  for r in rows if np.isfinite(_f(r, "iris_selected_fraction"))})
    pts = [(a, b) for a, b in pts if np.isfinite(a)]
    if len(pts) < 2:
        return None
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=5, lw=1.6,
            color="#0072B2")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("beam radius on the SLM  [pixels]   (smaller to the right)")
    ax.set_ylabel("iris transmission of the +1 order")
    ax.set_title("The beam-size axis is confounded by iris clipping")
    ax.grid(True, which="both", alpha=0.25, lw=0.5)
    fig.tight_layout()
    return style.save_figure(
        fig, out_dir / name,
        "Fraction of the +1 diffraction order transmitted by the fixed Fourier-plane "
        "iris against beam radius in SLM pixels. Shown on its own axes rather than "
        "overlaid on the sensitivity curve, because it is a separate mechanism that "
        "grows over the same sweep and would be misread as part of the pixel effect.",
    )


def figure_convergence(summary, out_dir, style, name):
    import matplotlib.pyplot as plt

    metric = "before_infidelity_vs_registered0__peak_to_peak"
    groups = defaultdict(list)
    for r in summary:
        groups[int(_f(r, "charge"))].append((_f(r, "samples_per_pixel"), _f(r, metric)))
    groups = {k: sorted(v) for k, v in groups.items() if len({a for a, _ in v}) > 1}
    if not groups:
        return None
    cmap = plt.get_cmap("viridis")
    charges = sorted(groups)
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for i, ell in enumerate(charges):
        pts = [(a, b) for a, b in groups[ell] if np.isfinite(b) and b > 0]
        if len(pts) < 2:
            continue
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=5, lw=1.6,
                color=cmap(0.12 + 0.76 * i / max(len(charges) - 1, 1)),
                label=f"$\\ell$ = {ell}")
    ax.axvline(2.0, color="0.45", ls="--", lw=1.2)
    ax.annotate("lattice unresolved below 2 samples/pixel:\nsweep returns an exact null",
                xy=(2.0, 0.5), xycoords=("data", "axes fraction"),
                xytext=(6, 0), textcoords="offset points", fontsize=8, color="0.3",
                va="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("computational samples per SLM pixel")
    ax.set_ylabel("registration-induced infidelity, peak to peak")
    ax.set_title("Sampling convergence of the registration metric")
    ax.grid(True, which="both", alpha=0.25, lw=0.5)
    ax.legend(frameon=False, fontsize=9, title="charge", title_fontsize=9)
    fig.tight_layout()
    return style.save_figure(
        fig, out_dir / name,
        "Convergence of the registration-induced infidelity against the number of "
        "computational samples per 8 um SLM pixel. Below two samples per pixel the "
        "pixel-averaging operator degenerates into the identity and the sweep returns "
        "exactly zero, so that region cannot be plotted on a log axis and is marked "
        "instead. Both axes logarithmic.",
    )


def figure_scaling_collapse(rows, out_dir, style, name):
    """The money plot: does one dimensionless group govern the sensitivity?"""

    import matplotlib.pyplot as plt

    metric = ("postiris_infidelity_vs_registered0"
              if any(_f(r, "postiris_infidelity_vs_registered0")
                     == _f(r, "postiris_infidelity_vs_registered0") for r in rows)
              else "before_infidelity_vs_registered0")
    groups = defaultdict(list)
    for r in rows:
        if r["series"] == "continuous":
            continue
        v = _f(r, metric)
        ell = int(_f(r, "charge"))
        w = _f(r, "beam_radius_px")
        if not np.isfinite(v) or ell <= 0 or not np.isfinite(w) or w <= 0:
            continue
        groups[(ell, w)].append(v)
    if len(groups) < 4:
        return None

    pts = sorted((ell / w, max(v), ell, w) for (ell, w), v in groups.items())
    x = np.array([p[0] for p in pts])
    y = np.array([p[1] for p in pts])
    charges = sorted({p[2] for p in pts})
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    for i, ell in enumerate(charges):
        sel = [(a, b) for a, b, e, _ in pts if e == ell]
        ax.plot([p[0] for p in sel], [p[1] for p in sel], linestyle="none",
                marker="o", ms=7, mew=0.8, mec="white",
                color=cmap(0.10 + 0.80 * i / max(len(charges) - 1, 1)),
                label=f"$\\ell$ = {ell}")

    unsat = (y < 0.5) & (x > 0)
    if unsat.sum() >= 3:
        coef = np.polyfit(np.log(x[unsat]), np.log(y[unsat]), 1)
        xs = np.logspace(np.log10(x[unsat].min()), np.log10(x[unsat].max()), 50)
        ax.plot(xs, np.exp(coef[1]) * xs ** coef[0], color="0.35", lw=1.4, ls="--",
                label=f"$({np.exp(coef[1]):.2f})\\,(\\ell p/w)^{{{coef[0]:.2f}}}$")
    ax.axhspan(0.5, 1.2, color="0.85", alpha=0.45, zorder=0)
    ax.annotate("saturated: the offset\nsets the output field",
                xy=(0.03, 0.90), xycoords="axes fraction", ha="left",
                va="top", fontsize=8, color="0.3")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(top=1.4)
    ax.set_xlabel("$\\ell\\,p\\,/\\,w$   (unresolved core radius / beam radius)")
    ax.set_ylabel("registration-induced infidelity")
    ax.set_title("Charge and beam size collapse onto one parameter")
    ax.grid(True, which="both", alpha=0.25, lw=0.5)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    return style.save_figure(
        fig, out_dir / name,
        "Registration-induced infidelity against the dimensionless group "
        "l*p/w, the ratio of the pixel-unresolved vortex core radius to the beam "
        "radius. Points span a factor of 20 in charge and 25 in beam radius; they "
        "fall on a single power law, so charge and beam size are not independent "
        "axes but one combined parameter. The shaded band marks saturation, where "
        "the sub-pixel offset essentially determines the output field. Both axes "
        "logarithmic; the dashed line is a fit to the unsaturated points only.",
        metadata={"metric": metric, "group": "ell*pitch/beam_radius"},
    )


def figure_after_axicon(rows, out_dir, style, name):
    import matplotlib.pyplot as plt

    have = [r for r in rows if np.isfinite(_f(r, "after_peak_intensity"))]
    if not have:
        return None
    charges = sorted({int(_f(r, "charge")) for r in have})
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    for ax, (metric, label) in zip(axes, (
        ("after_peak_intensity", "Bessel-zone peak intensity, relative spread"),
        ("after_bessel_zone_fwhm_m", "Bessel-zone length, relative spread"),
    )):
        for dof, colour in DOF_COLOURS.items():
            xs, ys = [], []
            for ell in charges:
                vals = [_f(r, metric) for r in have
                        if int(_f(r, "charge")) == ell
                        and (r["dof"] == dof or r["series"] == "registered")]
                vals = [v for v in vals if np.isfinite(v)]
                if len(vals) < 2:
                    continue
                mean = float(np.mean(vals))
                if mean == 0.0:
                    continue
                xs.append(ell)
                ys.append(float(np.max(vals) - np.min(vals)) / abs(mean))
            if len(xs) > 1:
                ax.plot(xs, ys, marker=DOF_MARKERS[dof], color=colour, lw=1.6, ms=5,
                        label=DOF_LABELS[dof])
        ax.set_yscale("log")
        ax.set_xlabel("vortex charge  $\\ell$")
        ax.set_ylabel(label)
        ax.grid(True, which="both", alpha=0.25, lw=0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False,
                   bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Consequence in the Bessel region behind the axicon", fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    return style.save_figure(
        fig, out_dir / name,
        "Relative spread across the sub-pixel offset sweep of two Bessel-region "
        "observables: the peak transverse intensity and the axial zone length. Peak "
        "transverse intensity is used rather than the on-axis value because a charge "
        "l > 0 beam is dark on axis by construction. Log vertical axes.",
    )


def figure_axial_curves(axial_rows, out_dir, style, name):
    """Bessel-region axial profiles, one curve per sub-pixel offset.

    This is the direct answer to whether the field *behind* the axicon changes:
    if the curves for different offsets lie on top of one another the Bessel
    region is insensitive, and if they separate it is not.
    """

    import matplotlib.pyplot as plt

    if not axial_rows:
        return None
    keys = sorted({(int(_f(r, "charge")), _f(r, "beam_radius_px"))
                   for r in axial_rows})
    keys = [k for k in keys if np.isfinite(k[1])]
    if not keys:
        return None
    charges = sorted({k[0] for k in keys})
    beams = sorted({k[1] for k in keys}, reverse=True)
    fig, axes = plt.subplots(len(charges), len(beams),
                             figsize=(3.5 * len(beams), 2.7 * len(charges)),
                             squeeze=False, sharex="col")
    cmap = plt.get_cmap("viridis")
    offsets = sorted({_f(r, "delta_px") for r in axial_rows
                      if np.isfinite(_f(r, "delta_px"))})

    for i, ell in enumerate(charges):
        for j, w in enumerate(beams):
            ax = axes[i][j]
            for k, d in enumerate(offsets):
                pts = sorted((_f(r, "z_m"), _f(r, "peak_intensity"))
                             for r in axial_rows
                             if int(_f(r, "charge")) == ell
                             and _f(r, "beam_radius_px") == w
                             and _f(r, "delta_px") == d)
                pts = [(a, b) for a, b in pts if np.isfinite(a) and np.isfinite(b)]
                if len(pts) < 2:
                    continue
                ax.plot([p[0] * 1e3 for p in pts], [p[1] for p in pts], lw=1.6,
                        color=cmap(0.12 + 0.76 * k / max(len(offsets) - 1, 1)),
                        label=f"$\\delta/p$ = {d:g}")
            ax.grid(True, alpha=0.25, lw=0.5)
            if i == 0:
                ax.set_title(f"w = {w:.0f} px", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"$\\ell$ = {ell}\npeak intensity [a.u.]", fontsize=9)
            if i == len(charges) - 1:
                ax.set_xlabel("z behind the axicon [mm]")
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(labels),
                   frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Does the field behind the axicon change with sub-pixel registration?",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    return style.save_figure(
        fig, out_dir / name,
        "Peak transverse intensity against propagation distance behind the axicon, "
        "one curve per sub-pixel beam offset, for each charge and beam radius. Peak "
        "transverse intensity replaces the on-axis value because a charge l > 0 beam "
        "is dark on axis by construction. Curves that overlie one another mean the "
        "Bessel region is insensitive to registration at that condition; curves that "
        "separate mean it is not. Linear axes; intensity in arbitrary units common "
        "to each panel.",
    )


# --------------------------------------------------------------------------
# field figures
# --------------------------------------------------------------------------

def figure_fields(case, out_dir, style, *, fine_n, relay_n, delta_frac, arm,
                  beam_radius_m, fill_model, crop_before_m, crop_after_m):
    import matplotlib.pyplot as plt

    from vbb_study.digital_twin.slm_pixel_registration import (
        RegistrationSampling, build_registration_route, registration_panels)
    from vbb_study.digital_twin.slm_registration_metrics import (
        axial_profile, complex_fidelity)

    pitch = 8e-6
    window = 10e-3 if arm == "bench" else 2e-3
    sampling = RegistrationSampling(fine_grid_n=fine_n,
                                    relay_grid_n=relay_n if arm == "bench" else fine_n,
                                    window_m=window)
    charge = 0 if case == "B0" else int(case[1:])

    def build(delta, pixelated):
        s1, s2 = registration_panels(delta, dof="slm1", axis="x")
        return build_registration_route(
            case, sampling=sampling, slm1=s1, slm2=s2, arm=arm,
            pixelate_phase=pixelated, fill_factor_model=fill_model,
            beam_radius_m=beam_radius_m)

    cont = build(0.0, False)
    shifted = build(delta_frac * pitch, True)
    grid = cont["grid"]
    x = np.asarray(grid["x"], dtype=float)

    paths = []
    for tag, key, crop_m, zlabel in (
        ("before_axicon", "field_on_axicon_plane", crop_before_m, "axicon input plane"),
        ("after_axicon", None, crop_after_m, "Bessel region"),
    ):
        if key is None:
            lam = float(cont["metadata"]["wavelength_m"])
            kr = float(cont["metadata"]["axicon"]["exact_kr_m_inv"])
            k = 2 * np.pi / lam
            sb = min(0.999, kr / k)
            zmax = (beam_radius_m or cont["metadata"]["beam_radius_on_slm_m"]) / (
                sb / math.sqrt(1 - sb * sb))
            zs = np.linspace(0.05 * zmax, 1.0 * zmax, 16)
            pa = axial_profile(cont["post_axicon"], grid, wavelength_m=lam,
                               z_values_m=zs, charge=charge)
            z_pick = float(pa["z_at_peak_m"])
            from vbb_study.digital_twin.slm_pixel_registration import lean_asm_propagate
            a = lean_asm_propagate(cont["post_axicon"], grid, lam, z_pick)
            b = lean_asm_propagate(shifted["post_axicon"], grid, lam, z_pick)
            zlabel = f"Bessel region, z = {z_pick * 1e3:.1f} mm"
        else:
            a = cont[key]
            b = shifted[key]

        xa, Ia = _crop(np.abs(a) ** 2, x, crop_m)
        _, Ib = _crop(np.abs(b) ** 2, x, crop_m)
        _, Ba = _crop(a, x, crop_m)
        ext = [xa[0] * 1e3, xa[-1] * 1e3, xa[0] * 1e3, xa[-1] * 1e3]
        peak = max(float(Ia.max()), float(Ib.max())) or 1.0
        diff = (Ib - Ia) / peak
        lim = float(np.max(np.abs(diff))) or 1.0

        # constrained layout, not tight_layout: a colorbar is added below and
        # tight_layout refuses to run once one exists
        fig, axes = plt.subplots(1, 4, figsize=(15.0, 3.8), layout="constrained")
        axes[0].imshow(Ia / peak, extent=ext, origin="lower", cmap="inferno",
                       vmin=0, vmax=1)
        axes[0].set_title("ideal: continuous phase\n(no pixel lattice)", fontsize=9)
        axes[1].imshow(Ib / peak, extent=ext, origin="lower", cmap="inferno",
                       vmin=0, vmax=1)
        axes[1].set_title(f"pixelated, beam at $\\delta$ = {delta_frac:g}p", fontsize=9)
        im = axes[2].imshow(diff, extent=ext, origin="lower", cmap="RdBu_r",
                            vmin=-lim, vmax=lim)
        axes[2].set_title(f"difference (peak {lim:.2e})", fontsize=9)
        fig.colorbar(im, ax=axes[2], fraction=0.046)
        axes[3].imshow(_domain_colour(Ba), extent=ext, origin="lower")
        axes[3].set_title("phase, domain coloured\n(hue = phase, value = |U|)", fontsize=9)
        for ax in axes:
            ax.set_xlabel("x [mm]")
        axes[0].set_ylabel("y [mm]")
        fid = complex_fidelity(b, a)
        fig.suptitle(f"{case} ({zlabel}) -- pixelation infidelity {1 - fid:.2e}",
                     fontsize=11)
        paths.append(style.save_figure(
            fig, out_dir / f"{'07' if key else '08'}_registration_{tag}_{case.lower()}.png",
            f"{case} field at the {zlabel} for the ideal continuous-phase route, the "
            f"pixelated route with the beam offset {delta_frac:g} of a pixel, their "
            "normalised intensity difference on a diverging scale with symmetric limits "
            "about zero, and the domain-coloured complex field. Intensities are "
            "normalised to the larger peak of the two panels; the difference annotation "
            "gives its own peak magnitude.",
            metadata={"case": case, "delta_fraction_of_pitch": delta_frac,
                      "fine_grid_n": fine_n, "arm": arm},
        ))
    return paths



def figure_propagation_region(case, out_dir, style, *, fine_n, arm, window_m,
                              beam_radius_m, fill_model, n_z, half_width_m,
                              delta_frac):
    """Longitudinal x-z map of the Bessel region, ideal vs worst registration.

    The transverse panels say what the beam looks like at one plane; this says
    what the whole propagation region looks like, which is where an axial
    change shows up.  The worst case is the beam sitting half a pixel from where
    the ideal, unpixelated route puts it.
    """

    import matplotlib.pyplot as plt

    from vbb_study.digital_twin.slm_pixel_registration import (
        RegistrationSampling, build_registration_route, lean_asm_propagator,
        registration_panels)
    from vbb_study.digital_twin.slm_registration_metrics import (
        complex_fidelity, vortex_structure)

    pitch = 8e-6
    charge = 0 if case == "B0" else int(case[1:])
    sampling = RegistrationSampling(fine_grid_n=fine_n, relay_grid_n=fine_n,
                                    window_m=window_m)

    def build(delta, pixelated):
        s1, s2 = registration_panels(delta, dof="slm1", axis="x")
        return build_registration_route(
            case, sampling=sampling, slm1=s1, slm2=s2, arm=arm,
            pixelate_phase=pixelated, fill_factor_model=fill_model,
            beam_radius_m=beam_radius_m)

    ideal = build(0.0, False)
    worst = build(delta_frac * pitch, True)
    grid = ideal["grid"]
    lam = float(ideal["metadata"]["wavelength_m"])
    kr = float(ideal["metadata"]["axicon"]["exact_kr_m_inv"])
    k = 2 * np.pi / lam
    sb = min(0.999, kr / k)
    z_max = beam_radius_m / (sb / np.sqrt(1 - sb * sb))
    zs = np.linspace(0.02 * z_max, 1.05 * z_max, int(n_z))

    x = np.asarray(grid["x"], dtype=float)
    keep = np.abs(x) <= half_width_m
    n = int(grid["N"])
    rows = (n // 2 - 1, n // 2)   # cell-centred grid: average the two central rows

    def xz_map(route):
        prop = lean_asm_propagator(route["post_axicon"], grid, lam)
        out = np.zeros((int(n_z), int(keep.sum())), dtype=float)
        for i, z in enumerate(zs):
            u = prop(z)
            out[i] = 0.5 * (np.abs(u[rows[0]]) ** 2 + np.abs(u[rows[1]]) ** 2)[keep]
            del u
        return out

    a = xz_map(ideal)
    b = xz_map(worst)
    peak = max(float(a.max()), float(b.max())) or 1.0
    a /= peak
    b /= peak
    diff = b - a
    lim = float(np.max(np.abs(diff))) or 1.0
    ext = [x[keep][0] * 1e6, x[keep][-1] * 1e6, zs[0] * 1e3, zs[-1] * 1e3]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), layout="constrained")
    for ax, img, ttl, kw in (
        (axes[0], a, "ideal: continuous phase\n(no pixel lattice)",
         dict(cmap="inferno", vmin=0, vmax=1)),
        (axes[1], b, f"worst case: beam {delta_frac:g} pixel off\n(pixelated)",
         dict(cmap="inferno", vmin=0, vmax=1)),
        (axes[2], diff, f"difference (peak {lim:.2f})",
         dict(cmap="RdBu_r", vmin=-lim, vmax=lim)),
    ):
        im = ax.imshow(img, extent=ext, origin="lower", aspect="auto", **kw)
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel("x [um]")
        fig.colorbar(im, ax=ax, fraction=0.046)
    axes[0].set_ylabel("z behind the axicon [mm]")

    w_px = beam_radius_m / pitch
    fid = complex_fidelity(worst["post_axicon"], ideal["post_axicon"])
    fig.suptitle(
        f"{case}, beam radius {w_px:.1f} px  ($\\ell p/w$ = {charge / max(w_px, 1e-9):.2f})"
        f"  --  axicon-plane infidelity {1 - fid:.2e}", fontsize=12)
    return style.save_figure(
        fig, out_dir / f"12_propagation_region_{case.lower()}_{w_px:.0f}px.png",
        f"Longitudinal x-z intensity through the Bessel region behind the axicon for "
        f"{case} at a beam radius of {w_px:.1f} SLM pixels, comparing the ideal "
        f"continuous-phase route against the same route pixelated with the beam "
        f"displaced {delta_frac:g} of a pixel, and their difference on a diverging "
        f"scale with symmetric limits about zero. Both intensity panels share the "
        f"same normalisation, so they are directly comparable. Horizontal axis is "
        f"the transverse cut through y = 0; vertical axis is propagation distance.",
        metadata={"case": case, "beam_radius_px": w_px, "arm": arm,
                  "delta_fraction_of_pitch": delta_frac},
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics-root", type=Path,
                   default=Path("outputs/validation/slm_pixel_registration"))
    p.add_argument("--tag", default="registration")
    p.add_argument("--figure-root", type=Path,
                   default=Path("outputs/figures/slm_pixel_registration"))
    p.add_argument("--mode", default="curves",
                   choices=("curves", "fields", "propagation", "all"))
    p.add_argument("--prop-cases", nargs="+", default=["V20"])
    p.add_argument("--prop-beam-radii-m", nargs="+", type=float,
                   default=[100e-6])
    p.add_argument("--prop-n-z", type=int, default=60)
    p.add_argument("--prop-half-width-m", type=float, default=150e-6)
    p.add_argument("--prop-delta-frac", type=float, default=0.5)
    p.add_argument("--field-cases", nargs="+", default=["V20"])
    p.add_argument("--field-fine-n", type=int, default=4096)
    p.add_argument("--field-relay-n", type=int, default=2048)
    p.add_argument("--field-delta-frac", type=float, default=0.5)
    p.add_argument("--field-arm", default="bench", choices=("bench", "isolation"))
    p.add_argument("--field-beam-radius-m", type=float, default=-1.0)
    p.add_argument("--field-fill-model", default="resolved_pixel_aperture")
    p.add_argument("--crop-before-m", type=float, default=3.0e-3)
    p.add_argument("--crop-after-m", type=float, default=0.3e-3)
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    from vbb_study import vbb_style as style
    style.apply_style()

    src = args.metrics_root / args.tag
    out = args.figure_root / args.tag
    out.mkdir(parents=True, exist_ok=True)

    written: list[Any] = []
    if args.mode in ("curves", "all"):
        rows = _read(src / "registration_metrics.csv")
        summary = _read(src / "registration_summary.csv")
        axial_rows = _read(src / "registration_axial.csv")
        if not rows:
            print(f"no metrics at {src}; run the study first")
        else:
            written += [x for x in (
                figure_offset_curves(rows, out, style,
                                     "before_infidelity_vs_registered0",
                                     "infidelity vs zero-offset route",
                                     "01_registration_offset_curves.png"),
                figure_offset_curves(rows, out, style,
                                     "before_infidelity_vs_continuous",
                                     "infidelity vs continuous ideal",
                                     "02_registration_pixelation_penalty.png"),
                figure_charge_scaling(summary, out, style,
                                      "03_registration_charge_scaling.png"),
                figure_beam_size(summary, out, style,
                                 "04_registration_beam_size_scaling.png"),
                figure_iris_confound(rows, out, style,
                                     "05_registration_iris_confound.png"),
                figure_convergence(summary, out, style,
                                   "06_registration_sampling_convergence.png"),
                figure_scaling_collapse(rows, out, style,
                                        "10_registration_scaling_collapse.png"),
                figure_after_axicon(rows, out, style,
                                    "09_registration_bessel_consequence.png"),
                figure_axial_curves(axial_rows, out, style,
                                    "11_registration_axial_profiles.png"),
            ) if x]

    if args.mode in ("fields", "all"):
        for case in args.field_cases:
            written += figure_fields(
                case, out, style,
                fine_n=args.field_fine_n, relay_n=args.field_relay_n,
                delta_frac=args.field_delta_frac, arm=args.field_arm,
                beam_radius_m=None if args.field_beam_radius_m <= 0
                else args.field_beam_radius_m,
                fill_model=args.field_fill_model,
                crop_before_m=args.crop_before_m, crop_after_m=args.crop_after_m)

    if args.mode in ("propagation", "all"):
        for case in args.prop_cases:
            for w in args.prop_beam_radii_m:
                written.append(figure_propagation_region(
                    case, out, style,
                    fine_n=args.field_fine_n, arm=args.field_arm,
                    window_m=(2.048e-3 if args.field_arm == "isolation"
                              else 0.010922666666666667),
                    beam_radius_m=float(w), fill_model=args.field_fill_model,
                    n_z=args.prop_n_z, half_width_m=args.prop_half_width_m,
                    delta_frac=args.prop_delta_frac))

    for path in written:
        print(f"wrote {path}")
    print(f"{len(written)} figures in {out}")


if __name__ == "__main__":
    main()
