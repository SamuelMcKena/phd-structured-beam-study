"""Miao-style analytic Bessel-modal benchmark for the real q=20 BeamGage stack.

This is an *adapted benchmark*, not a claim of reproducing every detail of Miao
et al.  It implements the core intensity-only model of Eq. (3)--(4) in

    B. Miao, L. Feder, J. E. Shrock, H. M. Milchberg,
    "Phase front retrieval and correction of Bessel beams",
    Opt. Express 30, 11360--11371 (2022), doi:10.1364/OE.454796.

For a q-th order Bessel beam the focal field is represented as

    U(r,phi,z) = A(z) sum_n (-i)^n c_n^(q) J_n(k_perp r) exp(-i n phi).

Miao retrieves the complex modal coefficients independently from measured
intensity profiles and maps each z plane to an annulus of the axicon/input
aperture.  Here we use the same analytic modal representation as a deliberately
simpler baseline against the bench-matched digital twin:

* real BeamGage data and the exact observation-frame registration produced by
  ``fit_q20_detector_aware_model_v2.py`` are used;
* only even-index z planes are fitted;
* a compact set of modes centred on the ideal n=-q term is retrieved from
  intensity alone;
* coefficient trajectories are interpolated in z and scored on the untouched
  odd-index planes;
* no finite SLM/4F/axicon route is present in this analytic baseline and no SLM
  correction mask is emitted.

The held-out construction is an adaptation introduced solely to compare
predictive generalisation with the digital-twin inverse.  It is more stringent
than the plane-by-plane reconstruction test in the original paper and must not
be described as Miao's own validation protocol.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from scipy.special import jv

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "outputs" / "validation" / "q20_detector_aware_model_v2"
DEFAULT_OUT = ROOT / "outputs" / "validation" / "q20_miao_bessel_modal_benchmark"
Q = 20
THERMAL = "inferno"
EPS = np.finfo(float).tiny


def plane_normalise(stack: np.ndarray) -> np.ndarray:
    a = np.asarray(stack, float)
    if a.ndim == 2:
        return a / max(float(np.max(a)), EPS)
    m = np.max(a, axis=(-2, -1), keepdims=True)
    return a / np.maximum(m, EPS)


def score(predicted: np.ndarray, measured: np.ndarray, axis_um: np.ndarray) -> dict:
    X, Y = np.meshgrid(axis_um, axis_um, indexing="xy")
    roi = np.hypot(X, Y) <= 145.0
    p, m = plane_normalise(predicted), plane_normalise(measured)
    rows = []
    for iz in range(len(p)):
        aa, bb = p[iz][roi], m[iz][roi]
        rows.append({
            "pearson_r": float(np.corrcoef(aa, bb)[0, 1]),
            "nrmse": float(np.sqrt(np.mean((aa - bb) ** 2))),
        })
    return {
        "mean_pearson_r": float(np.mean([r["pearson_r"] for r in rows])),
        "mean_nrmse": float(np.mean([r["nrmse"] for r in rows])),
        "max_nrmse": float(np.max([r["nrmse"] for r in rows])),
        "per_plane": rows,
    }


def radial_profile(image: np.ndarray, axis_um: np.ndarray, dr_um: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    X, Y = np.meshgrid(axis_um, axis_um, indexing="xy")
    R = np.hypot(X, Y)
    edges = np.arange(0.0, min(150.0, float(np.max(R))) + dr_um, dr_um)
    centres = 0.5 * (edges[:-1] + edges[1:])
    ids = np.digitize(R.ravel(), edges) - 1
    vals = np.asarray(image, float).ravel()
    prof = np.zeros_like(centres)
    for i in range(len(centres)):
        use = ids == i
        prof[i] = float(np.mean(vals[use])) if np.any(use) else np.nan
    return centres, prof


def first_jq_intensity_peak(q: int) -> float:
    # The first bright annulus of J_q is just beyond x=q for q>0.
    result = minimize_scalar(lambda x: -float(jv(q, x) ** 2), bounds=(q, q + 8.0), method="bounded")
    return float(result.x)


def estimate_k_perp(measured: np.ndarray, axis_um: np.ndarray, train: np.ndarray) -> dict:
    avg = np.mean(plane_normalise(measured[train]), axis=0)
    r, p = radial_profile(avg, axis_um)
    valid = (r >= 20.0) & (r <= 110.0) & np.isfinite(p)
    rv, pv = r[valid], p[valid]
    # Smooth only for locating the principal annulus; no smoothing enters the fit.
    kernel = np.ones(5, float) / 5.0
    smooth = np.convolve(pv, kernel, mode="same")
    r_peak_um = float(rv[int(np.argmax(smooth))])
    x_peak = first_jq_intensity_peak(Q)
    k0 = x_peak / (r_peak_um * 1e-6)

    # Refine k_perp against the azimuthally averaged train morphology.  Allow a
    # slowly varying Gaussian envelope because a finite-aperture Bessel beam is
    # not the infinite J_q solution used by this analytic baseline.
    use = valid
    target = p[use]
    target = target / max(float(np.nanmax(target)), EPS)
    rr_m = r[use] * 1e-6

    def objective(logk: float) -> float:
        k = float(np.exp(logk))
        model = jv(Q, k * rr_m) ** 2
        model /= max(float(np.max(model)), EPS)
        # Optimal non-negative affine background/contrast for radial comparison.
        A = np.column_stack([model, np.ones_like(model)])
        coef, *_ = np.linalg.lstsq(A, target, rcond=None)
        fitted = A @ coef
        return float(np.mean((fitted - target) ** 2))

    fit = minimize_scalar(
        objective,
        bounds=(np.log(0.72 * k0), np.log(1.35 * k0)),
        method="bounded",
        options={"xatol": 2e-4},
    )
    return {
        "ring_peak_radius_um": r_peak_um,
        "first_Jq_peak_argument": x_peak,
        "initial_k_perp_m_inv": float(k0),
        "selected_k_perp_m_inv": float(np.exp(fit.x)),
        "radial_objective": float(fit.fun),
    }


def modal_orders(q: int, half_width: int) -> np.ndarray:
    # Ideal q vortex corresponds to n=-q in Miao's c_n^(q) convention.
    return np.arange(-q - half_width, -q + half_width + 1, dtype=int)


def modal_basis(axis_um: np.ndarray, k_perp: float, orders: np.ndarray, stride: int = 2) -> dict:
    axis = np.asarray(axis_um, float)[::stride]
    X, Y = np.meshgrid(axis * 1e-6, axis * 1e-6, indexing="xy")
    R = np.hypot(X, Y)
    theta = np.arctan2(Y, X)
    roi = (R >= 10e-6) & (R <= 145e-6)
    fields = []
    for n in orders:
        # Integer powers of -i are exact up to floating-point roundoff.
        fields.append(((-1j) ** int(n)) * jv(int(n), k_perp * R) * np.exp(-1j * int(n) * theta))
    return {"axis_um": axis, "R": R, "roi": roi, "fields": np.asarray(fields, complex)}


def unpack_relative_coefficients(params: np.ndarray, orders: np.ndarray, q: int = Q) -> np.ndarray:
    centre = int(np.where(orders == -q)[0][0])
    coeff = np.zeros(len(orders), complex)
    coeff[centre] = 1.0 + 0.0j
    cursor = 0
    for j in range(len(orders)):
        if j == centre:
            continue
        coeff[j] = float(params[cursor]) + 1j * float(params[cursor + 1])
        cursor += 2
    return coeff


def pack_relative_coefficients(coeff: np.ndarray, orders: np.ndarray, q: int = Q) -> np.ndarray:
    centre = int(np.where(orders == -q)[0][0])
    out = []
    c = np.asarray(coeff, complex)
    # Remove the unobservable global phase and amplitude by pinning c_-q=1.
    anchor = c[centre]
    if abs(anchor) < 1e-8:
        anchor = 1.0 + 0j
    c = c / anchor
    for j in range(len(orders)):
        if j == centre:
            continue
        out.extend([float(c[j].real), float(c[j].imag)])
    return np.asarray(out, float)


def render_from_coefficients(coeff: np.ndarray, basis: dict) -> np.ndarray:
    field = np.tensordot(np.asarray(coeff, complex), basis["fields"], axes=(0, 0))
    return plane_normalise(np.abs(field) ** 2)


def fit_plane(target_full: np.ndarray, basis: dict, x0: np.ndarray) -> tuple[np.ndarray, dict]:
    target = plane_normalise(np.asarray(target_full, float)[::2, ::2])
    roi = basis["roi"]
    t = target[roi]
    weight = np.sqrt(0.18 + 0.82 * np.sqrt(np.clip(t, 0.0, 1.0)))

    def residual(params: np.ndarray) -> np.ndarray:
        coeff = unpack_relative_coefficients(params, ORDERS_CACHE)
        pred = render_from_coefficients(coeff, basis)[roi]
        return weight * (pred - t)

    fit = least_squares(
        residual,
        np.asarray(x0, float),
        bounds=(-1.5, 1.5),
        method="trf",
        loss="soft_l1",
        f_scale=0.08,
        max_nfev=100,
        xtol=2e-6,
        ftol=2e-6,
        gtol=2e-6,
        verbose=0,
    )
    coeff = unpack_relative_coefficients(fit.x, ORDERS_CACHE)
    pred = render_from_coefficients(coeff, basis)
    err = float(np.sqrt(np.mean((pred[roi] - target[roi]) ** 2)))
    return coeff, {
        "success": bool(fit.success),
        "nfev": int(fit.nfev),
        "cost": float(fit.cost),
        "fit_roi_nrmse": err,
    }


def interpolate_coefficients(z_train: np.ndarray, coeff_train: np.ndarray, z_all: np.ndarray) -> np.ndarray:
    zt = np.asarray(z_train, float)
    za = np.asarray(z_all, float)
    c = np.asarray(coeff_train, complex)
    out = np.empty((len(za), c.shape[1]), complex)
    for j in range(c.shape[1]):
        out[:, j] = np.interp(za, zt, c[:, j].real) + 1j * np.interp(za, zt, c[:, j].imag)
    return out


def reconstruct_annular_phase(coeff: np.ndarray, orders: np.ndarray, theta: np.ndarray, q: int = Q) -> np.ndarray:
    # Inverse angular Fourier series implied by c_n^(q)=int E~ exp[i(n+q)theta] dtheta.
    harmonic = np.zeros_like(theta, dtype=complex)
    for c, n in zip(np.asarray(coeff, complex), orders):
        harmonic += c * np.exp(-1j * (int(n) + int(q)) * theta)
    return np.angle(harmonic)


def savefig(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=450, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run(source: Path, out: Path, half_width: int = 5) -> dict:
    source, out = Path(source), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    arrays = np.load(source / "model_v2_stacks.npz")
    measured = plane_normalise(np.asarray(arrays["measured_beam_frame"], float))
    axis_um = np.asarray(arrays["axis_um"], float)
    z_rel = np.asarray(arrays["z_relative_mm"], float)
    model_v2_summary = json.loads((source / "model_v2_summary.json").read_text(encoding="utf-8"))

    nplane = len(z_rel)
    train = np.arange(0, nplane, 2, dtype=int)
    held = np.arange(1, nplane, 2, dtype=int)
    kfit = estimate_k_perp(measured, axis_um, train)
    orders = modal_orders(Q, int(half_width))
    global ORDERS_CACHE
    ORDERS_CACHE = orders
    basis = modal_basis(axis_um, kfit["selected_k_perp_m_inv"], orders, stride=2)

    centre = int(np.where(orders == -Q)[0][0])
    initial_coeff = np.zeros(len(orders), complex)
    initial_coeff[centre] = 1.0
    x0 = pack_relative_coefficients(initial_coeff, orders)
    fitted_coeff = []
    fit_records = []
    # Sequential warm starts suppress coefficient branch jumps without using any
    # held-out image.  Only the previous TRAIN plane initializes the next fit.
    for iz in train:
        coeff, record = fit_plane(measured[iz], basis, x0)
        fitted_coeff.append(coeff)
        record.update({"plane_index": int(iz), "z_relative_mm": float(z_rel[iz])})
        fit_records.append(record)
        x0 = pack_relative_coefficients(coeff, orders)
    fitted_coeff = np.asarray(fitted_coeff, complex)

    coeff_all = interpolate_coefficients(z_rel[train], fitted_coeff, z_rel)
    predicted_small = np.asarray([render_from_coefficients(c, basis) for c in coeff_all], float)
    # Score on the same 241x241 camera coordinate frame as model-v2 by linear
    # interpolation from the deliberately reduced analytic-fit grid.
    from scipy.ndimage import zoom
    factor = measured.shape[-1] / predicted_small.shape[-1]
    predicted = np.asarray([zoom(p, factor, order=1, prefilter=False) for p in predicted_small], float)
    predicted = predicted[:, : measured.shape[-2], : measured.shape[-1]]
    predicted = plane_normalise(predicted)

    held_metrics = score(predicted[held], measured[held], axis_um)
    train_metrics = score(predicted[train], measured[train], axis_um)

    theta = np.linspace(-np.pi, np.pi, 361, endpoint=True)
    annular_phase_train = np.asarray([reconstruct_annular_phase(c, orders, theta) for c in fitted_coeff])
    mean_modal_magnitude = np.mean(np.abs(fitted_coeff), axis=0)
    mean_modal_magnitude /= max(float(mean_modal_magnitude[centre]), EPS)

    v2res = model_v2_summary["residual_model"]
    summary = {
        "study": "Miao-style analytic Bessel-modal benchmark adapted to q=20 with held-out z interpolation",
        "paper": "B. Miao et al., Opt. Express 30, 11360-11371 (2022), doi:10.1364/OE.454796",
        "implementation_scope": {
            "implemented": [
                "Eq. (3)-style Bessel modal field with complex c_n^(q)",
                "intensity-only nonlinear fitting on real measured planes",
                "ideal mode centred at n=-q for q=20",
                "annular input-phase reconstruction from fitted angular Fourier coefficients",
            ],
            "adaptation_for_fair_comparison": "fit even z planes only; interpolate complex coefficient trajectories and score untouched odd z planes",
            "not_in_analytic_baseline": [
                "explicit SLM carrier",
                "finite 4F iris",
                "refractive axicon propagation route",
                "camera pixel-area integration",
                "SLM2 correction solve",
            ],
            "not_claimed": "This is not a verbatim reproduction of Miao's Tensorlab/nonlinear-conjugate-gradient implementation or their experimental validation protocol.",
        },
        "q": Q,
        "mode_orders_n": orders.tolist(),
        "mode_half_width_about_minus_q": int(half_width),
        "k_perp_estimation": kfit,
        "data_split": {"train_indices": train.tolist(), "heldout_indices": held.tolist(), "heldout_used_in_fit": False},
        "train_metrics": train_metrics,
        "heldout_metrics": held_metrics,
        "fit_records": fit_records,
        "digital_twin_v2_heldout_reference": {
            "nominal": v2res["baseline_heldout"],
            "phase_only": v2res["phase_only_same_phase_heldout"],
            "phase_plus_amplitude": v2res["complex_heldout"],
        },
        "scientific_boundary": "Analytic benchmark only. It retrieves an intensity-consistent modal description; it does not constitute an SLM2 hardware correction or a corrected BeamGage measurement.",
    }
    (out / "miao_benchmark_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        out / "miao_benchmark_arrays.npz",
        measured=measured.astype(np.float32),
        predicted=predicted.astype(np.float32),
        axis_um=axis_um,
        z_relative_mm=z_rel,
        train_indices=train,
        heldout_indices=held,
        orders_n=orders,
        fitted_coefficients_train=fitted_coeff,
        interpolated_coefficients_all=coeff_all,
        theta_rad=theta,
        annular_phase_train=annular_phase_train.astype(np.float32),
    )

    # Figure 1: only held-out planes, so the visual cannot hide interpolation failures.
    ids = held[[0, 2, 4, 6, 8]]
    extent = [axis_um[0], axis_um[-1], axis_um[0], axis_um[-1]]
    fig, axs = plt.subplots(2, len(ids), figsize=(14.0, 5.7), constrained_layout=True)
    for col, iz in enumerate(ids):
        for row, stack in enumerate((measured, predicted)):
            axs[row, col].imshow(stack[iz], origin="lower", extent=extent, cmap=THERMAL, vmin=0, vmax=1, interpolation="nearest")
            axs[row, col].set_aspect("equal"); axs[row, col].set_xticks([]); axs[row, col].set_yticks([])
            if row == 0:
                axs[row, col].set_title(f"held-out z = {z_rel[iz]:.0f} mm", fontsize=12)
            if col == 0:
                axs[row, col].set_ylabel("MEASURED" if row == 0 else "MIAO-STYLE", fontsize=12, fontweight="bold")
    fig.suptitle("Intensity-only analytic Bessel-modal benchmark: untouched odd z planes", fontsize=16, fontweight="bold")
    savefig(fig, out / "01_miao_heldout_morphology")

    # Figure 2: direct held-out comparison against the bench-matched inverse.
    v2_nom = v2res["baseline_heldout"]["per_plane"]
    v2_phase = v2res["phase_only_same_phase_heldout"]["per_plane"]
    v2_complex = v2res["complex_heldout"]["per_plane"]
    fig, axs = plt.subplots(2, 1, figsize=(8.7, 7.0), sharex=True, constrained_layout=True)
    series = [
        ("Miao-style analytic", held_metrics["per_plane"]),
        ("digital twin nominal", v2_nom),
        ("digital twin phase", v2_phase),
        ("digital twin phase + amp", v2_complex),
    ]
    for label, rows in series:
        axs[0].plot(z_rel[held], [r["pearson_r"] for r in rows], "o-", lw=1.7, label=label)
        axs[1].plot(z_rel[held], [r["nrmse"] for r in rows], "o-", lw=1.7, label=label)
    axs[0].set(ylabel="Pearson r", title="Held-out z generalisation: analytic modal baseline vs bench-matched digital twin")
    axs[1].set(xlabel="relative z (mm)", ylabel="NRMSE")
    for ax in axs:
        ax.grid(alpha=.23); ax.legend(frameon=False, fontsize=9, ncol=2)
    savefig(fig, out / "02_miao_vs_digital_twin_heldout")

    # Figure 3: modal content and annular phase retrieved only from train planes.
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    offsets = orders + Q
    axs[0].bar(offsets, mean_modal_magnitude)
    axs[0].set(xlabel="angular offset m = n + q", ylabel="mean |c_n| / |c_-q|", title="Retrieved modal content (train planes)")
    im = axs[1].imshow(
        annular_phase_train,
        origin="lower",
        aspect="auto",
        extent=[-180, 180, z_rel[train][0], z_rel[train][-1]],
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi,
    )
    axs[1].set(xlabel="azimuth angle (deg)", ylabel="train-plane relative z (mm)", title="Miao-style annular phase reconstruction")
    fig.colorbar(im, ax=axs[1], label="phase (rad)", fraction=.047)
    savefig(fig, out / "03_miao_modal_and_annular_phase")

    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--half-width", type=int, default=5)
    args = parser.parse_args()
    if args.half_width < 2 or args.half_width > 10:
        raise SystemExit("--half-width must be between 2 and 10")
    run(args.source, args.out, half_width=args.half_width)


ORDERS_CACHE = modal_orders(Q, 5)

if __name__ == "__main__":
    main()
