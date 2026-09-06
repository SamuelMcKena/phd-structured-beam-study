"""Miao-style intensity-only phase-front retrieval for Bessel/vortex z-scans.

Reusable port of the mature experimental implementation. It follows the two-loop
structure in B. Miao et al., Optics Express 30, 11360-11371 (2022): per-plane
k_perp fitting, adaptive complex Bessel-mode fitting, stationary-phase annulus
mapping, radial phase reconstruction and explicit hardware safety gates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import ndimage, optimize, special

EPS = 1e-12
TWOPI = 2.0 * np.pi


@dataclass(frozen=True)
class PlaneRetrieval:
    z_index: int
    z_relative_m: float
    center_y_px: float
    center_x_px: float
    k_perp_m_inv: float
    aberration_order_max: int
    fit_cost: float
    fit_corr: float
    fit_nrmse: float
    m_values: np.ndarray
    coeffs: np.ndarray
    theta_rad: np.ndarray
    angular_field: np.ndarray


@dataclass(frozen=True)
class FullApertureRetrieval:
    rho_m: np.ndarray
    radial_phase_gradient_rad_per_m: np.ndarray
    radial_phase_rad: np.ndarray
    theta_rad: np.ndarray
    angular_phase_rows_rad: np.ndarray
    total_phase_rows_rad: np.ndarray
    angular_amplitude_rows: np.ndarray
    k_perp_nominal_m_inv: float
    branch: Literal["direct", "conjugate", "unresolved"]
    branch_score_direct: float | None
    branch_score_conjugate: float | None


def _normalise(a: np.ndarray) -> np.ndarray:
    arr = np.asarray(a, float)
    return arr / max(float(np.max(arr)), EPS)


def sample_polar(image: np.ndarray, center_yx: tuple[float, float],
                 radii_px: np.ndarray, theta: np.ndarray) -> np.ndarray:
    cy, cx = map(float, center_yx)
    rr, tt = np.meshgrid(np.asarray(radii_px, float), np.asarray(theta, float), indexing="ij")
    return ndimage.map_coordinates(
        np.asarray(image, float),
        [cy + rr * np.sin(tt), cx + rr * np.cos(tt)],
        order=1, mode="constant", cval=0.0,
    )


def modal_basis(q: int, m_values: np.ndarray, k_perp_m_inv: float,
                r_m: np.ndarray, phi_rad: np.ndarray) -> np.ndarray:
    """Miao Eq. (3) basis; aberration harmonic m maps to Bessel n=m-q."""
    cols = []
    for m in np.asarray(m_values, int):
        n = int(m) - int(q)
        cols.append(((-1j) ** n) * special.jv(n, float(k_perp_m_inv) * r_m)
                    * np.exp(-1j * n * phi_rad))
    return np.column_stack(cols)


def angular_field_from_coefficients(coeffs: np.ndarray, m_values: np.ndarray,
                                    theta_rad: np.ndarray) -> np.ndarray:
    """Input angular field after removing the programmed q*theta term."""
    g = np.zeros(np.asarray(theta_rad).shape, complex)
    for c, m in zip(np.asarray(coeffs, complex), np.asarray(m_values, int)):
        g += c * np.exp(-1j * int(m) * theta_rad)
    return g


def _pack(c: np.ndarray, m_values: np.ndarray) -> np.ndarray:
    i0 = int(np.where(np.asarray(m_values) == 0)[0][0])
    out = [np.log(max(float(np.real(c[i0])), 1e-8))]
    for i in range(len(c)):
        if i != i0:
            out.extend([float(np.real(c[i])), float(np.imag(c[i]))])
    return np.asarray(out)


def _unpack(x: np.ndarray, m_values: np.ndarray) -> np.ndarray:
    i0 = int(np.where(np.asarray(m_values) == 0)[0][0])
    c = np.zeros(len(m_values), complex)
    c[i0] = np.exp(x[0])
    k = 1
    for i in range(len(c)):
        if i != i0:
            c[i] = x[k] + 1j * x[k + 1]
            k += 2
    return c


def fit_coefficients(B: np.ndarray, measured: np.ndarray, weights: np.ndarray,
                     m_values: np.ndarray, maxiter: int = 160,
                     reg: float = 2e-4) -> tuple[np.ndarray, float]:
    y = _normalise(measured)
    w = np.asarray(weights, float)
    den = max(float(np.sum(w * y * y)), EPS)
    i0 = int(np.where(np.asarray(m_values) == 0)[0][0])
    p0 = np.abs(B[:, i0]) ** 2
    a2 = float(np.sum(w * p0 * y) / max(np.sum(w * p0 * p0), EPS))
    c0 = np.zeros(B.shape[1], complex)
    c0[i0] = np.sqrt(max(a2, 1e-8))
    x0 = _pack(c0, m_values)

    def fg(x: np.ndarray) -> tuple[float, np.ndarray]:
        c = _unpack(x, m_values)
        u = B @ c
        pred = np.abs(u) ** 2
        residual = pred - y
        loss = float(np.sum(w * residual * residual) / den)
        z = (w * residual) * np.conj(u)
        ga = 4.0 * np.real(z @ B) / den
        gb = -4.0 * np.imag(z @ B) / den
        idx = np.arange(len(c)) != i0
        c0r = max(float(np.real(c[i0])), 1e-12)
        reg_term = reg * float(np.sum(np.abs(c[idx]) ** 2)) / (c0r * c0r)
        loss += reg_term
        ga[idx] += 2.0 * reg * np.real(c[idx]) / (c0r * c0r)
        gb[idx] += 2.0 * reg * np.imag(c[idx]) / (c0r * c0r)
        grad = np.empty_like(x)
        grad[0] = ga[i0] * c0r - 2.0 * reg_term
        k = 1
        for j in range(len(c)):
            if j != i0:
                grad[k] = ga[j]
                grad[k + 1] = gb[j]
                k += 2
        return loss, grad

    res = optimize.minimize(
        lambda x: fg(x)[0], x0, jac=lambda x: fg(x)[1], method="L-BFGS-B",
        options={"maxiter": int(maxiter), "ftol": 1e-12, "gtol": 1e-7, "maxls": 40},
    )
    return _unpack(res.x, m_values), float(res.fun)


def _fit_arrays(image: np.ndarray, center_yx: tuple[float, float], pixel_pitch_m: float,
                rmax_um: float, n_r: int, n_theta: int):
    rmax_px = min(float(rmax_um) * 1e-6 / float(pixel_pitch_m), 0.47 * min(image.shape))
    radii_px = np.linspace(1.5, rmax_px, int(n_r))
    theta = np.linspace(0.0, TWOPI, int(n_theta), endpoint=False)
    polar = sample_polar(image, center_yx, radii_px, theta)
    rr, tt = np.meshgrid(radii_px * float(pixel_pitch_m), theta, indexing="ij")
    y = polar.ravel()
    rflat, pflat = rr.ravel(), tt.ravel()
    w = rflat / max(float(np.max(rflat)), EPS)
    w *= 0.25 + 0.75 * np.sqrt(np.clip(y / max(float(np.max(y)), EPS), 0.0, 1.0))
    return theta, y, rflat, pflat, w


def optimise_k_perp_ideal_mode(image: np.ndarray, center_yx: tuple[float, float],
                               pixel_pitch_m: float, q: int, k_perp_seed_m_inv: float,
                               search_fraction: float = 0.18, rmax_um: float = 220.0,
                               n_r: int = 30, n_theta: int = 48) -> float:
    """Miao first loop: fit k_perp using only the ideal m=0 mode."""
    _, y, r, phi, w = _fit_arrays(image, center_yx, pixel_pitch_m, rmax_um, n_r, n_theta)
    yn = _normalise(y)
    den = max(float(np.sum(w * yn * yn)), EPS)

    def objective(kp: float) -> float:
        b = modal_basis(q, np.asarray([0]), float(kp), r, phi)[:, 0]
        p = np.abs(b) ** 2
        scale = float(np.sum(w * p * yn) / max(np.sum(w * p * p), EPS))
        return float(np.sum(w * (scale * p - yn) ** 2) / den)

    seed = abs(float(k_perp_seed_m_inv))
    res = optimize.minimize_scalar(
        objective,
        bounds=(seed * (1.0 - search_fraction), seed * (1.0 + search_fraction)),
        method="bounded",
        options={"xatol": max(seed * 1e-7, 1e-3)},
    )
    if not res.success:
        raise RuntimeError(f"k_perp optimisation failed: {res.message}")
    return float(res.x)


def fit_plane_adaptive(image: np.ndarray, z_index: int, z_relative_m: float,
                       center_yx: tuple[float, float], pixel_pitch_m: float, q: int,
                       k_perp_seed_m_inv: float, max_aberration_order: int = 30,
                       order_step: int = 2, cost_threshold: float = 0.05,
                       min_fractional_improvement: float = 0.01,
                       rmax_um: float = 220.0, n_r: int = 48,
                       n_theta: int = 96) -> PlaneRetrieval:
    """Per-plane k_perp fit followed by increasing modal order."""
    kp = optimise_k_perp_ideal_mode(
        image, center_yx, pixel_pitch_m, q, k_perp_seed_m_inv, rmax_um=rmax_um
    )
    _, y, r, phi, w = _fit_arrays(image, center_yx, pixel_pitch_m, rmax_um, n_r, n_theta)
    yn = _normalise(y)
    chosen = None
    previous = np.inf
    stale = 0
    for order in range(2, int(max_aberration_order) + 1, int(order_step)):
        m_values = np.arange(-order, order + 1, dtype=int)
        B = modal_basis(q, m_values, kp, r, phi)
        coeffs, cost = fit_coefficients(B, y, w, m_values)
        pred = _normalise(np.abs(B @ coeffs) ** 2)
        corr = float(np.corrcoef(yn, pred)[0, 1])
        nrmse = float(np.sqrt(np.mean((yn - pred) ** 2)) / max(float(np.sqrt(np.mean(yn ** 2))), EPS))
        chosen = (order, m_values, coeffs, cost, corr, nrmse)
        if cost <= cost_threshold:
            break
        if np.isfinite(previous):
            improvement = (previous - cost) / max(previous, EPS)
            stale = stale + 1 if improvement < min_fractional_improvement else 0
            if stale >= 2:
                break
        previous = cost
    if chosen is None:
        raise RuntimeError("adaptive fit did not evaluate any modal order")
    order, m_values, coeffs, cost, corr, nrmse = chosen
    theta = np.linspace(0.0, TWOPI, 720, endpoint=False)
    g = angular_field_from_coefficients(coeffs, m_values, theta)
    return PlaneRetrieval(
        int(z_index), float(z_relative_m), float(center_yx[0]), float(center_yx[1]),
        kp, int(order), cost, corr, nrmse, m_values, coeffs, theta, g,
    )


def remove_row_piston(angular_fields: np.ndarray) -> np.ndarray:
    g = np.asarray(angular_fields, complex)
    u = g / np.maximum(np.abs(g), EPS)
    out = np.empty_like(u)
    for i, row in enumerate(u):
        mean = np.mean(row)
        if abs(mean) < 1e-8:
            mean = row[0]
        out[i] = row * np.exp(-1j * np.angle(mean))
    return np.angle(out)


def _integrate_radial_phase(rho: np.ndarray, gradient: np.ndarray) -> np.ndarray:
    rho = np.asarray(rho, float)
    gradient = np.asarray(gradient, float)
    if np.any(np.diff(rho) <= 0):
        raise ValueError("rho must be strictly increasing")
    phase = np.zeros_like(rho)
    phase[1:] = np.cumsum(0.5 * (gradient[1:] + gradient[:-1]) * np.diff(rho))
    return phase


def resolve_conjugate_branch(angular_fields: np.ndarray,
                             reference_intensity_rows: np.ndarray | None = None,
                             min_score_margin: float = 0.03):
    """Resolve the U/U* branch only when an independent reference is supplied."""
    if reference_intensity_rows is None:
        return "unresolved", None, None
    direct = np.abs(np.asarray(angular_fields, complex)) ** 2
    ref = np.asarray(reference_intensity_rows, float)
    if ref.shape != direct.shape:
        raise ValueError("reference intensity rows must match angular field shape")
    conjugate = np.roll(direct, direct.shape[1] // 2, axis=1)

    def corr(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])

    sd, sc = corr(direct, ref), corr(conjugate, ref)
    if abs(sd - sc) < min_score_margin:
        return "unresolved", sd, sc
    return ("direct" if sd > sc else "conjugate"), sd, sc


def assemble_full_aperture(retrievals: list[PlaneRetrieval], z_absolute_m: np.ndarray,
                           wavelength_m: float, k_perp_nominal_m_inv: float | None = None,
                           reference_intensity_rows: np.ndarray | None = None) -> FullApertureRetrieval:
    """Assemble radial + angular phase on the sampled input annuli."""
    if len(retrievals) < 2:
        raise ValueError("at least two planes are required")
    z = np.asarray(z_absolute_m, float)
    if z.shape != (len(retrievals),) or np.any(z <= 0):
        raise ValueError("z_absolute_m must contain one positive distance per plane")
    k = TWOPI / float(wavelength_m)
    kp = np.asarray([r.k_perp_m_inv for r in retrievals], float)
    rho = z * kp / k
    order = np.argsort(rho)
    rho, kp = rho[order], kp[order]
    fields = np.stack([retrievals[i].angular_field for i in order])
    theta = retrievals[order[0]].theta_rad
    nominal = float(np.median(kp) if k_perp_nominal_m_inv is None else k_perp_nominal_m_inv)
    radial_gradient = nominal - kp
    radial_phase = _integrate_radial_phase(rho, radial_gradient)
    angular_phase = remove_row_piston(fields)
    branch, sd, sc = resolve_conjugate_branch(fields, reference_intensity_rows)
    if branch == "conjugate":
        half = angular_phase.shape[1] // 2
        angular_phase = -np.roll(angular_phase, half, axis=1)
        fields = np.conj(np.roll(fields, half, axis=1))
    total = np.angle(np.exp(1j * (angular_phase + radial_phase[:, None])))
    return FullApertureRetrieval(
        rho, radial_gradient, radial_phase, theta, angular_phase, total,
        np.abs(fields), nominal, branch, sd, sc,
    )


def interpolate_to_cartesian(full: FullApertureRetrieval, grid_size: int = 512,
                             padding_fraction: float = 0.05):
    """Interpolate wrapped phase through unit phasors, never phase values directly."""
    rho = np.asarray(full.rho_m, float)
    extent = float(rho[-1] + max(0.0, padding_fraction) * (rho[-1] - rho[0]))
    axis = np.linspace(-extent, extent, int(grid_size))
    X, Y = np.meshgrid(axis, axis, indexing="xy")
    R, TH = np.hypot(X, Y), np.mod(np.arctan2(Y, X), TWOPI)
    rows = np.exp(1j * full.total_phase_rows_rad)
    rows = np.concatenate([rows, rows[:, :1]], axis=1)
    rcoord = np.interp(R, rho, np.arange(len(rho), dtype=float))
    tcoord = TH / TWOPI * full.total_phase_rows_rad.shape[1]
    real = ndimage.map_coordinates(rows.real, [rcoord, tcoord], order=1, mode="nearest")
    imag = ndimage.map_coordinates(rows.imag, [rcoord, tcoord], order=1, mode="nearest")
    residual = np.angle(real + 1j * imag)
    valid = (R >= rho[0]) & (R <= rho[-1])
    residual[~valid] = np.nan
    correction = np.full_like(residual, np.nan)
    correction[valid] = np.angle(np.exp(-1j * residual[valid]))
    return {
        "x_m": axis, "y_m": axis, "residual_phase_rad": residual,
        "conjugate_correction_phase_rad": correction, "valid": valid,
    }


def map_input_phase_to_slm2(cartesian: dict, slm_shape: tuple[int, int],
                            input_plane_m_per_slm_pixel: float,
                            slm_center_yx_px: tuple[float, float], rotation_deg: float,
                            parity_x: int, parity_y: int) -> np.ndarray:
    """Map input-plane correction to SLM2 with measured scale/rotation/parity."""
    if parity_x not in (-1, 1) or parity_y not in (-1, 1):
        raise ValueError("parity must be +/-1")
    phase = np.asarray(cartesian["conjugate_correction_phase_rad"], float)
    x_axis, y_axis = np.asarray(cartesian["x_m"]), np.asarray(cartesian["y_m"])
    u = np.zeros_like(phase, complex)
    good = np.isfinite(phase)
    u[good] = np.exp(1j * phase[good])
    ny, nx = map(int, slm_shape)
    cy, cx = map(float, slm_center_yx_px)
    yy, xx = np.indices((ny, nx), dtype=float)
    xs = (xx - cx) * float(input_plane_m_per_slm_pixel) * int(parity_x)
    ys = (yy - cy) * float(input_plane_m_per_slm_pixel) * int(parity_y)
    a = np.deg2rad(float(rotation_deg))
    xin = np.cos(a) * xs - np.sin(a) * ys
    yin = np.sin(a) * xs + np.cos(a) * ys
    xcoord = (xin - x_axis[0]) / (x_axis[1] - x_axis[0])
    ycoord = (yin - y_axis[0]) / (y_axis[1] - y_axis[0])
    real = ndimage.map_coordinates(u.real, [ycoord, xcoord], order=1, mode="constant", cval=0)
    imag = ndimage.map_coordinates(u.imag, [ycoord, xcoord], order=1, mode="constant", cval=0)
    mag = np.hypot(real, imag)
    out = np.full((ny, nx), np.nan)
    ok = mag > 0.25
    out[ok] = np.angle(real[ok] + 1j * imag[ok])
    return out


def correction_manifest(full: FullApertureRetrieval, *, absolute_z_calibrated: bool,
                        camera_to_slm_calibrated: bool, slm_lut_calibrated: bool,
                        independent_validation_done: bool = False) -> dict:
    """Block hardware claims/application until the required evidence exists."""
    pretrial: list[str] = []
    if full.branch == "unresolved":
        pretrial.append("conjugate/180-degree retrieval branch is unresolved")
    if not absolute_z_calibrated:
        pretrial.append("absolute camera-z to axicon/input distance is not calibrated")
    if not camera_to_slm_calibrated:
        pretrial.append("camera/input-plane to SLM2 scale/rotation/parity/centre is not calibrated")
    if not slm_lut_calibrated:
        pretrial.append("SLM2 1030-nm phase LUT/stroke is not calibrated")
    final = list(pretrial)
    if not independent_validation_done:
        final.append("candidate has not passed a new independent measured z-stack")
    return {
        "method": "Miao-style per-plane k_perp + adaptive complex Bessel modal retrieval",
        "programmed_vortex_in_correction": False,
        "radial_phase_recovered_from_k_perp_gradient": True,
        "branch": full.branch,
        "application_ready_for_low_gain_trial": len(pretrial) == 0,
        "pretrial_blockers": pretrial,
        "hardware_ready": len(final) == 0,
        "hardware_blockers": final,
    }


__all__ = [
    "PlaneRetrieval", "FullApertureRetrieval", "sample_polar", "modal_basis",
    "angular_field_from_coefficients", "fit_coefficients", "optimise_k_perp_ideal_mode",
    "fit_plane_adaptive", "remove_row_piston", "resolve_conjugate_branch",
    "assemble_full_aperture", "interpolate_to_cartesian", "map_input_phase_to_slm2",
    "correction_manifest",
]
