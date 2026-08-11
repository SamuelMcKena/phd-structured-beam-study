"""First-principles vector boundary-field model for a tilted refractive axicon.

The macroscopic axicon is treated as two real dielectric surfaces.  A coherent
vector field is sampled on the tilted flat entrance surface with an analytically
tracked carrier, its *local electromagnetic Poynting direction* defines the ray
/eikonal direction, and the field is transported through both interfaces by
vector Snell law and local s/p Fresnel coefficients.  Finite optical path,
ray-tube Jacobian, Poynting-flux conservation and a fixed laboratory output plane
then produce a regular Ex/Ey/Ez boundary field for vector ASM / Debye propagation.

This is a Maxwell-consistent vector geometrical-optics/eikonal construction for
a macroscopic refractive optic; it is not a full-volume FDTD/FEM solution.  The
code deliberately rejects folded ray maps, total internal reflection where a
transmitted boundary field is requested, unresolved transverse wavevectors and
large interpolation-flux corrections.

References
----------
G. Yun, K. Crabtree and R. A. Chipman, Applied Optics 50, 2855-2865 (2011).
J. Kim et al., JOSA A 35, 526-535 (2018).
Z. Bin and L. Zhu, Applied Optics 37, 2563-2568 (1998).
A. Thaning, Z. Jaroszewicz and A. T. Friberg, Applied Optics 42, 9-17 (2003).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np
from scipy.ndimage import map_coordinates

from vbb_study.digital_twin.vortex_refractive_axicon import RefractiveAxiconGeometry
from vbb_study.digital_twin.vortex_rotated_plane import rotation_matrix
from vbb_study.digital_twin.vortex_rotated_plane_baseband import rotate_baseband_angular_spectrum
from vbb_study.equations.fields import fft2c, ifft2c, make_xy_grid
from vbb_study.vector_field import VectorField, propagate_vector_asm, spectral_transversality_residual


EPS = np.finfo(float).tiny
TWOPI = 2.0 * np.pi


@dataclass(frozen=True)
class VectorRefractiveAxiconRayBundle:
    entrance_x_m: np.ndarray
    entrance_y_m: np.ndarray
    valid: np.ndarray
    incident_local: np.ndarray
    internal_local: np.ndarray
    exit_point_local_m: np.ndarray
    exit_normal_local: np.ndarray
    outgoing_local: np.ndarray
    exit_point_lab_m: np.ndarray
    outgoing_lab: np.ndarray
    reference_x_lab_m: np.ndarray
    reference_y_lab_m: np.ndarray
    reference_z_lab_m: float
    reference_distance_m: np.ndarray
    internal_distance_m: np.ndarray
    optical_path_m: np.ndarray
    mapping_jacobian_signed: np.ndarray
    input_normal_flux_density_au: np.ndarray
    fresnel_power_surface1: np.ndarray
    fresnel_power_surface2: np.ndarray
    fresnel_reflectance_surface1: np.ndarray
    fresnel_reflectance_surface2: np.ndarray
    entrance_reference_phase_rad: np.ndarray
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class VectorRefractiveAxiconResult:
    field: VectorField
    entrance_surface_envelope_lab: VectorField
    geometry_bundle: VectorRefractiveAxiconRayBundle
    coverage_mask: np.ndarray
    outgoing_direction_lab: np.ndarray
    metadata: Mapping[str, Any]

    @property
    def reference_x_lab_m(self) -> np.ndarray:
        return self.geometry_bundle.reference_x_lab_m

    @property
    def reference_y_lab_m(self) -> np.ndarray:
        return self.geometry_bundle.reference_y_lab_m

    @property
    def reference_z_lab_m(self) -> float:
        return float(self.geometry_bundle.reference_z_lab_m)

    @property
    def optical_path_m(self) -> np.ndarray:
        return self.geometry_bundle.optical_path_m


def _strict_unit(v: np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=float)
    if arr.shape[-1] != 3:
        raise ValueError("vector arrays must have final dimension 3")
    norm = np.linalg.norm(arr, axis=-1, keepdims=True)
    if np.any(norm <= np.finfo(float).eps):
        raise ValueError("zero-length vector")
    return arr / norm


def _safe_unit(v: np.ndarray, fallback: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(v, dtype=float)
    fb = np.asarray(fallback, dtype=float)
    norm = np.linalg.norm(arr, axis=-1, keepdims=True)
    valid = np.squeeze(norm > np.finfo(float).eps, axis=-1)
    safe = np.where(valid[..., None], arr, np.broadcast_to(fb, arr.shape))
    return _strict_unit(safe), valid


def _broadcast_vector_inputs(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    converted = [np.asarray(value) for value in arrays]
    for value in converted:
        if value.shape[-1] != 3:
            raise ValueError("all vector inputs must have final dimension 3")
    lead_shape = np.broadcast_shapes(*(value.shape[:-1] for value in converted))
    target = lead_shape + (3,)
    return tuple(np.broadcast_to(value, target) for value in converted)


def _vector_spectral_centroid_cpm(field: VectorField) -> tuple[float, float]:
    weight = (
        np.abs(fft2c(field.ex)) ** 2
        + np.abs(fft2c(field.ey)) ** 2
        + np.abs(fft2c(field.ez)) ** 2
    )
    total = float(np.sum(weight))
    if total <= EPS:
        return 0.0, 0.0
    FX = np.asarray(field.grid["FX"], dtype=float)
    FY = np.asarray(field.grid["FY"], dtype=float)
    return float(np.sum(weight * FX) / total), float(np.sum(weight * FY) / total)


def _spectral_project_and_poynting_on_tilted_plane(
    ex_env: np.ndarray,
    ey_env: np.ndarray,
    ez_env: np.ndarray,
    grid: Mapping[str, Any],
    *,
    vacuum_wavelength_m: float,
    medium_index: float,
    carrier_local_cpm: tuple[float, float],
    rotation_local_to_lab: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Return transverse E envelope and local time-averaged-flux direction data.

    The omitted analytic carrier is inserted in the absolute spectral wavevector,
    so many-degree entrance-plane rotations do not alias a global optical carrier
    onto the sampled grid.  H is reconstructed spectrally from
    ``H * Z0 = n (khat x E)``; the common Z0 factor is intentionally omitted.
    """

    ax = fft2c(np.asarray(ex_env, dtype=np.complex128))
    ay = fft2c(np.asarray(ey_env, dtype=np.complex128))
    az = fft2c(np.asarray(ez_env, dtype=np.complex128))
    spectrum = np.stack([ax, ay, az], axis=-1)

    FX = np.asarray(grid["FX"], dtype=float)
    FY = np.asarray(grid["FY"], dtype=float)
    fc_x, fc_y = map(float, carrier_local_cpm)
    f_total = float(medium_index) / float(vacuum_wavelength_m)
    fx = fc_x + FX
    fy = fc_y + FY
    transverse_sq = fx * fx + fy * fy
    propagating = transverse_sq < f_total * f_total
    fz = np.sqrt(np.maximum(f_total * f_total - transverse_sq, 0.0))
    k_local = np.stack([fx, fy, fz], axis=-1) / max(f_total, EPS)
    k_lab = k_local @ np.asarray(rotation_local_to_lab, dtype=float).T

    dot = np.sum(spectrum * k_lab, axis=-1, keepdims=True)
    projected = np.where(propagating[..., None], spectrum - dot * k_lab, 0.0j)
    h_spectrum = float(medium_index) * np.cross(k_lab, projected)

    e_env = np.stack(
        [ifft2c(projected[..., 0]), ifft2c(projected[..., 1]), ifft2c(projected[..., 2])],
        axis=-1,
    )
    h_env = np.stack(
        [ifft2c(h_spectrum[..., 0]), ifft2c(h_spectrum[..., 1]), ifft2c(h_spectrum[..., 2])],
        axis=-1,
    )
    poynting = np.real(np.cross(e_env, np.conj(h_env)))

    numerator = float(np.sum(np.abs(np.sum(projected * k_lab, axis=-1)) ** 2))
    denominator = max(float(np.sum(np.abs(projected) ** 2)), EPS)
    source_energy = max(float(np.sum(np.abs(spectrum) ** 2)), EPS)
    retained = float(np.sum(np.abs(projected) ** 2) / source_energy)
    return np.asarray(e_env, dtype=np.complex128), np.asarray(poynting, dtype=float), {
        "spectral_transversality_power_ratio": numerator / denominator,
        "transverse_projection_spectral_energy_retained": retained,
        "propagating_spectral_fraction": float(np.mean(propagating)),
    }


def sample_vector_field_on_tilted_entrance(
    field: VectorField,
    *,
    tilt_x_rad: float,
    tilt_y_rad: float,
    spectral_center_cpm: tuple[float, float] | None = None,
) -> tuple[VectorField, np.ndarray, np.ndarray, dict[str, Any]]:
    """Sample E on the physical entrance plane and compute its local Poynting field.

    The returned :class:`VectorField` is the *baseband envelope* in fixed lab
    Cartesian components; its metadata contains the analytically tracked local
    carrier.  The third return value is the lab-basis Poynting vector (up to the
    common factor ``1/(2 Z0)``).
    """

    if field.medium_index <= 0.0:
        raise ValueError("field medium index must be positive")
    projected_source = propagate_vector_asm(field, 0.0)
    fsx, fsy = (
        _vector_spectral_centroid_cpm(projected_source)
        if spectral_center_cpm is None
        else tuple(map(float, spectral_center_cpm))
    )
    f_total = float(field.medium_index) / float(field.wavelength_m)
    if fsx * fsx + fsy * fsy >= f_total * f_total:
        raise ValueError("source vector carrier is non-propagating")

    X = np.asarray(projected_source.grid["X"], dtype=float)
    Y = np.asarray(projected_source.grid["Y"], dtype=float)
    source_carrier = np.exp(-1j * TWOPI * (fsx * X + fsy * Y))
    source_env = [component * source_carrier for component in (projected_source.ex, projected_source.ey, projected_source.ez)]

    effective_wavelength = float(field.wavelength_m) / float(field.medium_index)
    output_env: list[np.ndarray] = []
    transform_meta: list[Mapping[str, Any]] = []
    for component in source_env:
        mapped, meta = rotate_baseband_angular_spectrum(
            component,
            projected_source.grid,
            wavelength_m=effective_wavelength,
            tilt_x_rad=float(tilt_x_rad),
            tilt_y_rad=float(tilt_y_rad),
            source_spectral_center_cpm=(fsx, fsy),
        )
        output_env.append(np.asarray(mapped, dtype=np.complex128))
        transform_meta.append(dict(meta))

    destination_carrier = tuple(map(float, transform_meta[0]["destination_spectral_center_cpm"]))
    for meta in transform_meta[1:]:
        if not np.allclose(meta["destination_spectral_center_cpm"], destination_carrier, rtol=0.0, atol=1e-9):
            raise RuntimeError("vector components acquired inconsistent tilted-plane carriers")

    rotation = rotation_matrix(float(tilt_x_rad), float(tilt_y_rad))
    e_env, poynting_lab, maxwell_meta = _spectral_project_and_poynting_on_tilted_plane(
        output_env[0],
        output_env[1],
        output_env[2],
        projected_source.grid,
        vacuum_wavelength_m=field.wavelength_m,
        medium_index=field.medium_index,
        carrier_local_cpm=destination_carrier,
        rotation_local_to_lab=rotation,
    )
    envelope = VectorField(
        ex=e_env[..., 0],
        ey=e_env[..., 1],
        ez=e_env[..., 2],
        grid=projected_source.grid,
        wavelength_m=field.wavelength_m,
        medium_index=field.medium_index,
        metadata={
            **dict(field.metadata),
            "stage": "tilted_axicon_entrance_baseband_envelope",
            "field_components_basis": "fixed_lab_xyz",
            "surface_coordinates": "axicon_local_xy",
            "analytic_carrier_local_cpm": list(destination_carrier),
            "carrier_representation": "analytic_unsampled",
        },
    )
    return envelope, np.asarray(destination_carrier, dtype=float), poynting_lab, {
        "source_transversality_residual": float(spectral_transversality_residual(projected_source)),
        "source_spectral_center_cpm": [float(fsx), float(fsy)],
        "destination_spectral_center_local_cpm": list(destination_carrier),
        "component_rotated_plane_metadata": transform_meta,
        **maxwell_meta,
    }


def _refract_masked(
    direction1: np.ndarray,
    normal12: np.ndarray,
    *,
    n1: float,
    n2: float,
) -> tuple[np.ndarray, np.ndarray]:
    d1, normal = _broadcast_vector_inputs(np.asarray(direction1, dtype=float), np.asarray(normal12, dtype=float))
    d1 = _strict_unit(d1)
    normal = _strict_unit(normal)
    cos_i = np.sum(d1 * normal, axis=-1, keepdims=True)
    tangent1 = d1 - cos_i * normal
    tangent2 = (float(n1) / float(n2)) * tangent1
    radicand = 1.0 - np.sum(tangent2 * tangent2, axis=-1, keepdims=True)
    valid = (np.squeeze(cos_i > 0.0, axis=-1) & np.squeeze(radicand >= -1.0e-12, axis=-1))
    transmitted = tangent2 + np.sqrt(np.maximum(radicand, 0.0)) * normal
    transmitted, nonzero = _safe_unit(transmitted, normal)
    valid &= nonzero
    return transmitted, valid


def _fallback_s_axis(direction: np.ndarray) -> np.ndarray:
    d = _strict_unit(direction)
    trial_x = np.broadcast_to(np.asarray([1.0, 0.0, 0.0]), d.shape)
    trial_y = np.broadcast_to(np.asarray([0.0, 1.0, 0.0]), d.shape)
    trial = np.where((np.abs(d[..., 0]) < 0.9)[..., None], trial_x, trial_y)
    return _strict_unit(np.cross(d, trial))


def fresnel_transmit_vector_3d(
    electric1: np.ndarray,
    direction1: np.ndarray,
    direction2: np.ndarray,
    normal12: np.ndarray,
    *,
    n1: float,
    n2: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transmit a complex 3-D E vector and return mixed-polarization T and R."""

    n1f = float(n1)
    n2f = float(n2)
    if n1f <= 0.0 or n2f <= 0.0:
        raise ValueError("refractive indices must be positive")
    e_raw, d1_raw, d2_raw, n_raw = _broadcast_vector_inputs(
        np.asarray(electric1, dtype=np.complex128),
        np.asarray(direction1, dtype=float),
        np.asarray(direction2, dtype=float),
        np.asarray(normal12, dtype=float),
    )
    d1 = _strict_unit(d1_raw)
    d2 = _strict_unit(d2_raw)
    normal = _strict_unit(n_raw)
    e1 = e_raw - np.sum(e_raw * d1, axis=-1, keepdims=True) * d1

    cos_i = np.sum(d1 * normal, axis=-1)
    cos_t = np.sum(d2 * normal, axis=-1)
    if np.any(cos_i <= 0.0) or np.any(cos_t <= 0.0):
        raise ValueError("Fresnel transmission requires forward rays")

    s_raw = np.cross(normal, d1)
    s_norm = np.linalg.norm(s_raw, axis=-1)
    near_normal = s_norm < 1.0e-12
    s_axis = np.where(near_normal[..., None], _fallback_s_axis(d1), s_raw)
    s_axis = _strict_unit(s_axis)
    p1 = _strict_unit(np.cross(d1, s_axis))
    p2 = _strict_unit(np.cross(d2, s_axis))

    ts = 2.0 * n1f * cos_i / (n1f * cos_i + n2f * cos_t)
    tp = 2.0 * n1f * cos_i / (n2f * cos_i + n1f * cos_t)
    rs = (n1f * cos_i - n2f * cos_t) / (n1f * cos_i + n2f * cos_t)
    rp = (n2f * cos_i - n1f * cos_t) / (n2f * cos_i + n1f * cos_t)
    es = np.sum(e1 * s_axis, axis=-1)
    ep = np.sum(e1 * p1, axis=-1)
    transmitted = ts[..., None] * es[..., None] * s_axis + tp[..., None] * ep[..., None] * p2

    if np.any(near_normal):
        t0 = 2.0 * n1f / (n1f + n2f)
        projected2 = e1 - np.sum(e1 * d2, axis=-1, keepdims=True) * d2
        transmitted = np.where(near_normal[..., None], t0 * projected2, transmitted)

    ein2 = np.sum(np.abs(e1) ** 2, axis=-1)
    eout2 = np.sum(np.abs(transmitted) ** 2, axis=-1)
    transmission = n2f * cos_t * eout2 / np.maximum(n1f * cos_i * ein2, EPS)
    reflection = (np.abs(rs * es) ** 2 + np.abs(rp * ep) ** 2) / np.maximum(ein2, EPS)
    if np.any(near_normal):
        r0 = (n1f - n2f) / (n1f + n2f)
        reflection = np.where(near_normal, abs(r0) ** 2, reflection)
    dark = ein2 <= 100.0 * EPS
    transmission = np.where(dark, 0.0, transmission)
    reflection = np.where(dark, 0.0, reflection)
    return transmitted, np.asarray(transmission, dtype=float), np.asarray(reflection, dtype=float)


def _cone_intersection_distance(
    x0: np.ndarray,
    y0: np.ndarray,
    direction: np.ndarray,
    *,
    base_angle_rad: float,
    centre_thickness_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    d = np.asarray(direction, dtype=float)
    sx, sy, sz = d[..., 0], d[..., 1], d[..., 2]
    x = np.asarray(x0, dtype=float)
    y = np.asarray(y0, dtype=float)
    m = math.tan(float(base_angle_rad))
    tc = float(centre_thickness_m)
    a = sz * sz - m * m * (sx * sx + sy * sy)
    b = -2.0 * (tc * sz + m * m * (x * sx + y * sy))
    c = tc * tc - m * m * (x * x + y * y)
    disc = b * b - 4.0 * a * c
    linear = np.abs(a) < 1.0e-14
    safe_a = np.where(linear, 1.0, a)
    root_disc = np.sqrt(np.maximum(disc, 0.0))
    r1 = (-b - root_disc) / (2.0 * safe_a)
    r2 = (-b + root_disc) / (2.0 * safe_a)
    linear_root = -c / np.where(np.abs(b) > 1.0e-14, b, np.nan)
    r1 = np.where(linear, linear_root, r1)
    r2 = np.where(linear, np.nan, r2)
    positive1 = r1 > 0.0
    positive2 = r2 > 0.0
    root = np.where(positive1 & positive2, np.minimum(r1, r2), np.where(positive1, r1, np.where(positive2, r2, np.nan)))
    valid = np.isfinite(root) & (root > 0.0) & (disc >= -1.0e-12)
    return np.asarray(root, dtype=float), np.asarray(valid, dtype=bool)


def _mapping_jacobian_signed(xr: np.ndarray, yr: np.ndarray, *, dx_m: float, dy_m: float) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    dX_dy, dX_dx = np.gradient(np.asarray(xr, dtype=float), float(dy_m), float(dx_m))
    dY_dy, dY_dx = np.gradient(np.asarray(yr, dtype=float), float(dy_m), float(dx_m))
    det = dX_dx * dY_dy - dX_dy * dY_dx
    return det, {"dX_dx": dX_dx, "dX_dy": dX_dy, "dY_dx": dY_dx, "dY_dy": dY_dy}


def _grid_with_lab_origin(n: int, window_m: float, centre_m: tuple[float, float]) -> dict[str, Any]:
    grid = dict(make_xy_grid(int(n), float(window_m) / int(n)))
    cx, cy = map(float, centre_m)
    original = np.asarray(grid["x"], dtype=float)
    grid["x"] = original + cx
    grid["y"] = original + cy
    grid["X"] = np.asarray(grid["X"], dtype=float) + cx
    grid["Y"] = np.asarray(grid["Y"], dtype=float) + cy
    grid["R"] = np.hypot(grid["X"], grid["Y"])
    grid["PHI"] = np.arctan2(grid["Y"], grid["X"])
    return grid


def _sample_regular(values: np.ndarray, iy: np.ndarray, ix: np.ndarray, *, order: int) -> np.ndarray:
    arr = np.asarray(values)
    query = np.vstack([iy.ravel(), ix.ravel()])
    if np.iscomplexobj(arr):
        re = map_coordinates(arr.real, query, order=order, mode="constant", cval=0.0, prefilter=order > 1)
        im = map_coordinates(arr.imag, query, order=order, mode="constant", cval=0.0, prefilter=order > 1)
        return (re + 1j * im).reshape(ix.shape)
    values_out = map_coordinates(arr.astype(float), query, order=order, mode="constant", cval=np.nan, prefilter=order > 1)
    return values_out.reshape(ix.shape)


def _inverse_remap(
    ray_field: np.ndarray,
    valid: np.ndarray,
    x_in: np.ndarray,
    y_in: np.ndarray,
    x_ref: np.ndarray,
    y_ref: np.ndarray,
    output_grid: Mapping[str, Any],
    *,
    iterations: int = 8,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    X = np.asarray(x_in, dtype=float)
    Y = np.asarray(y_in, dtype=float)
    XR = np.asarray(x_ref, dtype=float)
    YR = np.asarray(y_ref, dtype=float)
    good_ray = np.asarray(valid, dtype=bool)
    x_axis = np.asarray(X[X.shape[0] // 2], dtype=float)
    y_axis = np.asarray(Y[:, Y.shape[1] // 2], dtype=float)
    dx = float(np.median(np.diff(x_axis)))
    dy = float(np.median(np.diff(y_axis)))

    ids = np.flatnonzero(good_ray.ravel())
    if ids.size < 64:
        raise ValueError("too few valid rays for vector field remapping")
    ids_fit = ids[:: max(1, ids.size // 4096)]
    A = np.column_stack([X.ravel()[ids_fit], Y.ravel()[ids_fit], np.ones(ids_fit.size)])
    cx, *_ = np.linalg.lstsq(A, XR.ravel()[ids_fit], rcond=None)
    cy, *_ = np.linalg.lstsq(A, YR.ravel()[ids_fit], rcond=None)
    affine = np.asarray([[cx[0], cx[1]], [cy[0], cy[1]]], dtype=float)
    if abs(float(np.linalg.det(affine))) < 1.0e-12:
        raise ValueError("ray-map affine inverse seed is singular")
    inv_affine = np.linalg.inv(affine)
    offset = np.asarray([cx[2], cy[2]])

    XT = np.asarray(output_grid["X"], dtype=float)
    YT = np.asarray(output_grid["Y"], dtype=float)
    initial = np.stack([XT - offset[0], YT - offset[1]], axis=-1) @ inv_affine.T
    xq, yq = initial[..., 0], initial[..., 1]
    _det, terms = _mapping_jacobian_signed(XR, YR, dx_m=dx, dy_m=dy)
    active = np.ones_like(XT, dtype=bool)
    residual = np.full_like(XT, np.inf)
    for _ in range(int(iterations)):
        ix = (xq - x_axis[0]) / dx
        iy = (yq - y_axis[0]) / dy
        xr = _sample_regular(XR, iy, ix, order=1)
        yr = _sample_regular(YR, iy, ix, order=1)
        dXdx = _sample_regular(terms["dX_dx"], iy, ix, order=1)
        dXdy = _sample_regular(terms["dX_dy"], iy, ix, order=1)
        dYdx = _sample_regular(terms["dY_dx"], iy, ix, order=1)
        dYdy = _sample_regular(terms["dY_dy"], iy, ix, order=1)
        rx, ry = xr - XT, yr - YT
        det = dXdx * dYdy - dXdy * dYdx
        step_ok = np.isfinite(rx) & np.isfinite(ry) & np.isfinite(det) & (np.abs(det) > 1.0e-10)
        active &= step_ok
        safe = np.where(step_ok, det, 1.0)
        xq = np.where(step_ok, xq - (dYdy * rx - dXdy * ry) / safe, xq)
        yq = np.where(step_ok, yq - (-dYdx * rx + dXdx * ry) / safe, yq)
        residual = np.hypot(rx, ry)

    ix = (xq - x_axis[0]) / dx
    iy = (yq - y_axis[0]) / dy
    ray_valid_sample = _sample_regular(good_ray.astype(float), iy, ix, order=1)
    tolerance = max(0.20 * float(output_grid["dx"]), 1.0e-10)
    coverage = (
        active
        & np.isfinite(ray_valid_sample)
        & (ray_valid_sample > 0.95)
        & np.isfinite(residual)
        & (residual <= tolerance)
        & (ix >= 0.0) & (ix <= X.shape[1] - 1.0)
        & (iy >= 0.0) & (iy <= X.shape[0] - 1.0)
    )
    components = []
    for component in range(3):
        sampled = _sample_regular(np.asarray(ray_field)[..., component], iy, ix, order=3)
        components.append(np.where(coverage, sampled, 0.0j))
    result = np.stack(components, axis=-1)
    rv = residual[coverage]
    return result, coverage, {
        "inverse_mapping_coverage_fraction": float(np.mean(coverage)),
        "inverse_mapping_median_residual_m": float(np.median(rv)) if rv.size else float("nan"),
        "inverse_mapping_p95_residual_m": float(np.percentile(rv, 95.0)) if rv.size else float("nan"),
        "inverse_mapping_max_residual_m": float(np.max(rv)) if rv.size else float("nan"),
        "inverse_mapping_tolerance_m": float(tolerance),
    }


def spectral_normal_flux_au(field: VectorField) -> float:
    """Plane-integrated +z Poynting flux, omitting only the common ``1/(2Z0)``."""

    ax, ay, az = fft2c(field.ex), fft2c(field.ey), fft2c(field.ez)
    FX = np.asarray(field.grid["FX"], dtype=float)
    FY = np.asarray(field.grid["FY"], dtype=float)
    f_total = float(field.medium_index) / float(field.wavelength_m)
    fz = np.sqrt(np.maximum(f_total * f_total - FX * FX - FY * FY, 0.0))
    propagating = FX * FX + FY * FY < f_total * f_total
    cos_theta = np.where(propagating, fz / max(f_total, EPS), 0.0)
    energy = np.abs(ax) ** 2 + np.abs(ay) ** 2 + np.abs(az) ** 2
    ny, nx = field.ex.shape
    dx = float(field.grid["dx"])
    dy = float(field.grid.get("dy", dx))
    return float(field.medium_index) * dx * dy / float(nx * ny) * float(np.sum(cos_theta * energy))


def _dominant_unwrapped_phase(electric: np.ndarray) -> tuple[np.ndarray, int]:
    energy = [float(np.sum(np.abs(electric[..., i]) ** 2)) for i in range(3)]
    index = int(np.argmax(energy))
    phase = np.angle(electric[..., index])
    return np.unwrap(np.unwrap(phase, axis=1), axis=0), index


def build_tilted_vector_refractive_axicon_field(
    upstream_field: VectorField,
    *,
    geometry: RefractiveAxiconGeometry,
    tilt_x_rad: float,
    tilt_y_rad: float,
    axicon_decentre_m: tuple[float, float] = (0.0, 0.0),
    reference_gap_m: float = 0.25e-3,
    output_n: int | None = None,
    output_window_m: float | None = None,
    output_center_lab_m: tuple[float, float] = (0.0, 0.0),
    apex_exclusion_radius_m: float = 0.0,
    minimum_mapping_jacobian: float = 1.0e-5,
    maximum_nyquist_fraction: float = 0.90,
    maximum_interpolation_flux_correction_fraction: float = 0.10,
    maximum_local_nontransverse_power_fraction: float = 0.02,
) -> VectorRefractiveAxiconResult:
    """Construct the transmitted vector boundary field on a fixed lab z-plane."""

    geometry.validate()
    if not np.isclose(upstream_field.medium_index, geometry.external_index, rtol=0.0, atol=1.0e-12):
        raise ValueError("upstream field medium must equal the axicon external medium")

    envelope, carrier_local, poynting_lab, surface_meta = sample_vector_field_on_tilted_entrance(
        upstream_field,
        tilt_x_rad=float(tilt_x_rad),
        tilt_y_rad=float(tilt_y_rad),
    )
    grid = envelope.grid
    X = np.asarray(grid["X"], dtype=float)
    Y = np.asarray(grid["Y"], dtype=float)
    dx = float(grid["dx"])
    dy = float(grid.get("dy", dx))
    rotation = rotation_matrix(float(tilt_x_rad), float(tilt_y_rad))
    dec_x, dec_y = map(float, axicon_decentre_m)
    X_rel, Y_rel = X - dec_x, Y - dec_y

    e_env_lab = np.stack([envelope.ex, envelope.ey, envelope.ez], axis=-1)
    carrier_phase_rad = TWOPI * (float(carrier_local[0]) * X + float(carrier_local[1]) * Y)
    e_lab = e_env_lab * np.exp(1j * carrier_phase_rad)[..., None]
    e_local = e_lab @ rotation
    poynting_local = np.asarray(poynting_lab, dtype=float) @ rotation
    incident_local, nonzero_flux = _safe_unit(poynting_local, np.asarray([0.0, 0.0, 1.0]))
    entrance_forward = incident_local[..., 2] > 0.0

    surface_intensity = np.sum(np.abs(e_local) ** 2, axis=-1)
    bright = surface_intensity > 1.0e-12 * max(float(np.max(surface_intensity)), EPS)
    e_transverse = e_local - np.sum(e_local * incident_local, axis=-1, keepdims=True) * incident_local
    transverse_intensity = np.sum(np.abs(e_transverse) ** 2, axis=-1)
    nontransverse_fraction = np.where(
        surface_intensity > EPS,
        np.maximum(surface_intensity - transverse_intensity, 0.0) / np.maximum(surface_intensity, EPS),
        0.0,
    )
    bright_values = nontransverse_fraction[bright]
    p95_nontransverse = float(np.percentile(bright_values, 95.0)) if bright_values.size else 1.0
    if p95_nontransverse > float(maximum_local_nontransverse_power_fraction):
        raise ValueError(
            "local plane-wave/eikonal approximation is not valid for this entrance field: "
            f"p95 non-transverse power fraction={p95_nontransverse:.4g}"
        )

    flat_normal = np.asarray([0.0, 0.0, 1.0])
    internal_local, snell1_valid = _refract_masked(
        incident_local,
        flat_normal,
        n1=geometry.external_index,
        n2=geometry.refractive_index,
    )
    safe_e = np.where((nonzero_flux & entrance_forward & bright)[..., None], e_transverse, 0.0j)
    e_internal, T1, R1 = fresnel_transmit_vector_3d(
        safe_e,
        incident_local,
        internal_local,
        flat_normal,
        n1=geometry.external_index,
        n2=geometry.refractive_index,
    )

    internal_distance, hit_cone = _cone_intersection_distance(
        X_rel,
        Y_rel,
        internal_local,
        base_angle_rad=geometry.base_angle_rad,
        centre_thickness_m=geometry.centre_thickness_m,
    )
    safe_distance = np.where(hit_cone, internal_distance, 0.0)
    exit_x = X_rel + safe_distance * internal_local[..., 0]
    exit_y = Y_rel + safe_distance * internal_local[..., 1]
    exit_z = safe_distance * internal_local[..., 2]
    exit_radius = np.hypot(exit_x, exit_y)
    phi = np.arctan2(exit_y, exit_x)
    sg = math.sin(geometry.base_angle_rad)
    cg = math.cos(geometry.base_angle_rad)
    cone_normal = np.stack([sg * np.cos(phi), sg * np.sin(phi), np.full_like(phi, cg)], axis=-1)
    outgoing_local, snell2_valid = _refract_masked(
        internal_local,
        cone_normal,
        n1=geometry.refractive_index,
        n2=geometry.external_index,
    )
    e_exit_local, T2, R2 = fresnel_transmit_vector_3d(
        np.where(hit_cone[..., None], e_internal, 0.0j),
        internal_local,
        outgoing_local,
        cone_normal,
        n1=geometry.refractive_index,
        n2=geometry.external_index,
    )

    local_valid = (
        nonzero_flux
        & entrance_forward
        & bright
        & snell1_valid
        & hit_cone
        & snell2_valid
        & (exit_radius <= geometry.clear_radius_m)
        & (exit_radius >= float(apex_exclusion_radius_m))
    )
    exit_local = np.stack([exit_x, exit_y, exit_z], axis=-1)
    tangent_offset_lab = rotation @ np.asarray([dec_x, dec_y, 0.0])
    exit_lab = exit_local @ rotation.T + tangent_offset_lab
    outgoing_lab = outgoing_local @ rotation.T
    local_valid &= outgoing_lab[..., 2] > 0.0
    if np.count_nonzero(local_valid) < 64:
        raise ValueError("too few transmitted rays remain after physical axicon gates")

    gap = max(float(reference_gap_m), 8.0 * max(dx, dy))
    z_reference = float(np.max(exit_lab[..., 2][local_valid])) + gap
    t_ref = (z_reference - exit_lab[..., 2]) / np.where(outgoing_lab[..., 2] > 0.0, outgoing_lab[..., 2], np.nan)
    local_valid &= np.isfinite(t_ref) & (t_ref > 0.0)
    x_ref = exit_lab[..., 0] + t_ref * outgoing_lab[..., 0]
    y_ref = exit_lab[..., 1] + t_ref * outgoing_lab[..., 1]
    jac_signed, _terms = _mapping_jacobian_signed(x_ref, y_ref, dx_m=dx, dy_m=dy)
    jac_abs = np.abs(jac_signed)
    local_valid &= np.isfinite(jac_abs) & (jac_abs > float(minimum_mapping_jacobian))

    interior = local_valid.copy()
    if min(interior.shape) > 8:
        interior[:3, :] = False
        interior[-3:, :] = False
        interior[:, :3] = False
        interior[:, -3:] = False
    signs = jac_signed[interior]
    positive_fraction = float(np.mean(signs > 0.0)) if signs.size else 0.0
    negative_fraction = float(np.mean(signs < 0.0)) if signs.size else 0.0
    if positive_fraction > 0.02 and negative_fraction > 0.02:
        raise ValueError("ray map folds before the fixed laboratory reference plane")

    cos_in = incident_local[..., 2]
    input_flux_density = float(geometry.external_index) * cos_in * transverse_intensity
    cumulative_T = T1 * T2
    output_polarization_norm = np.sqrt(np.sum(np.abs(e_exit_local) ** 2, axis=-1))
    output_unit_local = np.divide(
        e_exit_local,
        output_polarization_norm[..., None],
        out=np.zeros_like(e_exit_local),
        where=output_polarization_norm[..., None] > 100.0 * EPS,
    )
    output_unit_lab = output_unit_local @ rotation.T
    desired_output_intensity = (
        input_flux_density * cumulative_T
        / np.maximum(float(geometry.external_index) * outgoing_lab[..., 2] * jac_abs, EPS)
    )
    opl = geometry.refractive_index * safe_distance + geometry.external_index * t_ref
    opl_reference = float(np.median(opl[local_valid]))
    propagation_phase = np.exp(1j * (TWOPI / upstream_field.wavelength_m) * (opl - opl_reference))
    ray_field = (
        output_unit_lab
        * np.sqrt(np.maximum(desired_output_intensity, 0.0))[..., None]
        * propagation_phase[..., None]
    )
    ray_field = np.where(local_valid[..., None], ray_field, 0.0j)

    phase_for_eikonal, dominant_component = _dominant_unwrapped_phase(e_lab)

    interface1_error = np.abs(T1 + R1 - 1.0)
    interface2_error = np.abs(T2 + R2 - 1.0)
    fresnel1_max = float(np.max(interface1_error[local_valid]))
    fresnel2_max = float(np.max(interface2_error[local_valid]))

    all_transmitted_flux = float(np.sum(input_flux_density[local_valid] * cumulative_T[local_valid]) * dx * dy)
    reconstructed_ray_flux = float(
        np.sum(
            float(geometry.external_index)
            * np.sum(np.abs(ray_field[local_valid]) ** 2, axis=-1)
            * outgoing_lab[..., 2][local_valid]
            * jac_abs[local_valid]
        ) * dx * dy
    )

    n_out = int(upstream_field.ex.shape[0] if output_n is None else output_n)
    if n_out < 64:
        raise ValueError("output_n must be at least 64")
    input_window = max(float(np.ptp(X) + dx), float(np.ptp(Y) + dy))
    mapped_span = max(float(np.ptp(x_ref[local_valid])), float(np.ptp(y_ref[local_valid])))
    window = float(output_window_m) if output_window_m is not None else max(input_window, 1.05 * mapped_span)
    if window <= 0.0:
        raise ValueError("output_window_m must be positive")
    output_grid = _grid_with_lab_origin(n_out, window, output_center_lab_m)
    x_axis_out = np.asarray(output_grid["x"], dtype=float)
    y_axis_out = np.asarray(output_grid["y"], dtype=float)
    in_output_window = (
        local_valid
        & (x_ref >= x_axis_out[0]) & (x_ref <= x_axis_out[-1])
        & (y_ref >= y_axis_out[0]) & (y_ref <= y_axis_out[-1])
    )
    window_transmitted_flux = float(np.sum(input_flux_density[in_output_window] * cumulative_T[in_output_window]) * dx * dy)
    if window_transmitted_flux <= EPS:
        raise ValueError("fixed laboratory output window misses the transmitted field")

    f_total_out = geometry.external_index / upstream_field.wavelength_m
    required_fx = f_total_out * float(np.max(np.abs(outgoing_lab[..., 0][in_output_window])))
    required_fy = f_total_out * float(np.max(np.abs(outgoing_lab[..., 1][in_output_window])))
    nyquist = 0.5 / float(output_grid["dx"])
    required_nyquist_fraction = max(required_fx, required_fy) / max(nyquist, EPS)
    if required_nyquist_fraction > float(maximum_nyquist_fraction):
        raise ValueError(
            "output sampling cannot represent the refracted vector wavevectors: "
            f"required/nyquist={required_nyquist_fraction:.3f} > {maximum_nyquist_fraction:.3f}; "
            "increase output_n or reduce output_window_m"
        )

    remapped, coverage, remap_meta = _inverse_remap(
        ray_field,
        in_output_window,
        X,
        Y,
        x_ref,
        y_ref,
        output_grid,
    )
    raw = VectorField(
        ex=remapped[..., 0], ey=remapped[..., 1], ez=remapped[..., 2],
        grid=output_grid, wavelength_m=upstream_field.wavelength_m,
        medium_index=geometry.external_index,
        metadata={"stage": "vector_refractive_axicon_fixed_lab_plane_pre_projection"},
    )
    pre_projection_residual = float(spectral_transversality_residual(raw))
    projected = propagate_vector_asm(raw, 0.0)
    raster_flux_before = spectral_normal_flux_au(projected)
    correction_power_ratio = window_transmitted_flux / max(raster_flux_before, EPS)
    if abs(correction_power_ratio - 1.0) > float(maximum_interpolation_flux_correction_fraction):
        raise ValueError(
            "irregular-to-regular vector remapping loses too much normal flux: "
            f"power correction ratio={correction_power_ratio:.6f}"
        )
    global_scale = math.sqrt(correction_power_ratio)
    final_field = VectorField(
        ex=projected.ex * global_scale,
        ey=projected.ey * global_scale,
        ez=projected.ez * global_scale,
        grid=output_grid,
        wavelength_m=projected.wavelength_m,
        medium_index=projected.medium_index,
        metadata={
            "stage": "vector_refractive_axicon_fixed_lab_reference_plane",
            "reference_z_lab_m": z_reference,
            "reference_grid_center_lab_m": list(map(float, output_center_lab_m)),
        },
    )
    final_flux = spectral_normal_flux_au(final_field)
    final_transversality = float(spectral_transversality_residual(final_field))
    spectrum = np.abs(fft2c(final_field.ex)) ** 2 + np.abs(fft2c(final_field.ey)) ** 2 + np.abs(fft2c(final_field.ez)) ** 2
    edge = max(2, int(0.04 * n_out))
    edge_mask = np.zeros_like(spectrum, dtype=bool)
    edge_mask[:edge, :] = edge_mask[-edge:, :] = True
    edge_mask[:, :edge] = True
    edge_mask[:, -edge:] = True
    spectral_edge_fraction = float(np.sum(spectrum[edge_mask]) / max(float(np.sum(spectrum)), EPS))

    bundle = VectorRefractiveAxiconRayBundle(
        entrance_x_m=X,
        entrance_y_m=Y,
        valid=np.asarray(local_valid, dtype=bool),
        incident_local=np.asarray(incident_local, dtype=float),
        internal_local=np.asarray(internal_local, dtype=float),
        exit_point_local_m=np.asarray(exit_local, dtype=float),
        exit_normal_local=np.asarray(cone_normal, dtype=float),
        outgoing_local=np.asarray(outgoing_local, dtype=float),
        exit_point_lab_m=np.asarray(exit_lab, dtype=float),
        outgoing_lab=np.asarray(outgoing_lab, dtype=float),
        reference_x_lab_m=np.asarray(x_ref, dtype=float),
        reference_y_lab_m=np.asarray(y_ref, dtype=float),
        reference_z_lab_m=float(z_reference),
        reference_distance_m=np.asarray(t_ref, dtype=float),
        internal_distance_m=np.asarray(safe_distance, dtype=float),
        optical_path_m=np.asarray(opl, dtype=float),
        mapping_jacobian_signed=np.asarray(jac_signed, dtype=float),
        input_normal_flux_density_au=np.asarray(input_flux_density, dtype=float),
        fresnel_power_surface1=np.asarray(T1, dtype=float),
        fresnel_power_surface2=np.asarray(T2, dtype=float),
        fresnel_reflectance_surface1=np.asarray(R1, dtype=float),
        fresnel_reflectance_surface2=np.asarray(R2, dtype=float),
        entrance_reference_phase_rad=np.asarray(phase_for_eikonal, dtype=float),
        metadata={"dominant_phase_component_index": dominant_component},
    )
    return VectorRefractiveAxiconResult(
        field=final_field,
        entrance_surface_envelope_lab=envelope,
        geometry_bundle=bundle,
        coverage_mask=np.asarray(coverage, dtype=bool),
        outgoing_direction_lab=np.asarray(outgoing_lab, dtype=float),
        metadata={
            "outcome": "VECTOR-TWO-SURFACE-REFRACTIVE-AXICON-LAB-PLANE",
            "model_class": "local_Poynting_vector_eikonal_plus_two_surface_Snell_Fresnel_plus_vector_wave_boundary",
            "surface_order": "flat_entrance_then_conical_exit",
            "tilt_x_rad": float(tilt_x_rad),
            "tilt_y_rad": float(tilt_y_rad),
            "axicon_decentre_m": [dec_x, dec_y],
            "reference_z_lab_m": float(z_reference),
            "output_n": n_out,
            "output_window_m": window,
            "output_dx_m": float(output_grid["dx"]),
            "valid_ray_fraction": float(np.mean(local_valid)),
            "output_window_ray_fraction": float(np.count_nonzero(in_output_window) / max(np.count_nonzero(local_valid), 1)),
            "coverage_fraction": float(np.mean(coverage)),
            "mapping_positive_fraction": positive_fraction,
            "mapping_negative_fraction": negative_fraction,
            "p95_local_nontransverse_power_fraction": p95_nontransverse,
            "interface1_max_abs_R_plus_T_minus_1": fresnel1_max,
            "interface2_max_abs_R_plus_T_minus_1": fresnel2_max,
            "all_transmitted_flux_au": all_transmitted_flux,
            "window_transmitted_flux_au": window_transmitted_flux,
            "window_capture_fraction": window_transmitted_flux / max(all_transmitted_flux, EPS),
            "ray_flux_closure_ratio": reconstructed_ray_flux / max(all_transmitted_flux, EPS),
            "raster_flux_before_global_closure_au": raster_flux_before,
            "interpolation_flux_power_correction_ratio": correction_power_ratio,
            "interpolation_global_amplitude_correction": global_scale,
            "final_spectral_normal_flux_au": final_flux,
            "final_flux_closure_ratio": final_flux / max(window_transmitted_flux, EPS),
            "pre_projection_transversality_residual": pre_projection_residual,
            "final_transversality_residual": final_transversality,
            "required_nyquist_fraction": required_nyquist_fraction,
            "spectral_edge_power_fraction": spectral_edge_fraction,
            "opl_reference_m": opl_reference,
            "entrance_surface_sampling": surface_meta,
            **remap_meta,
            "physics_limit": (
                "macroscopic vector eikonal/Fresnel boundary-field model; multiple internal reflections, "
                "coating stacks, measured surface figure, full-volume Maxwell scattering and nonlinear material response are separate"
            ),
            "report_figures_authorised": False,
        },
    )


def lab_reference_eikonal_direction_consistency(
    result: VectorRefractiveAxiconResult,
    *,
    wavelength_m: float,
    external_index: float,
    trim_pixels: int = 4,
) -> dict[str, float]:
    """Check Fermat/eikonal phase gradients against the independently traced rays."""

    b = result.geometry_bundle
    X, Y = np.asarray(b.entrance_x_m), np.asarray(b.entrance_y_m)
    dx = float(np.median(np.diff(X[X.shape[0] // 2])))
    dy = float(np.median(np.diff(Y[:, Y.shape[1] // 2])))
    phase = np.asarray(b.entrance_reference_phase_rad) + (TWOPI / float(wavelength_m)) * np.asarray(b.optical_path_m)
    dphase_dy, dphase_dx = np.gradient(phase, dy, dx)
    det, terms = _mapping_jacobian_signed(b.reference_x_lab_m, b.reference_y_lab_m, dx_m=dx, dy_m=dy)
    gx = (terms["dY_dy"] * dphase_dx - terms["dX_dy"] * dphase_dy) / np.where(np.abs(det) > 1e-15, det, np.nan)
    gy = (-terms["dY_dx"] * dphase_dx + terms["dX_dx"] * dphase_dy) / np.where(np.abs(det) > 1e-15, det, np.nan)
    k_ext = TWOPI * float(external_index) / float(wavelength_m)
    expected_x = k_ext * np.asarray(b.outgoing_lab[..., 0])
    expected_y = k_ext * np.asarray(b.outgoing_lab[..., 1])
    scale = np.maximum(np.hypot(expected_x, expected_y), 0.01 * k_ext)
    error = np.hypot(gx - expected_x, gy - expected_y) / scale
    valid = np.asarray(b.valid) & np.isfinite(error) & (np.abs(det) > 1e-8)
    trim = int(trim_pixels)
    if trim > 0:
        interior = np.zeros_like(valid)
        interior[trim:-trim, trim:-trim] = True
        valid &= interior
    values = error[valid]
    if values.size < 32:
        raise ValueError("too few valid rays for eikonal direction consistency")
    return {
        "median_relative_direction_error": float(np.median(values)),
        "p95_relative_direction_error": float(np.percentile(values, 95.0)),
        "max_relative_direction_error": float(np.max(values)),
    }


__all__ = [
    "VectorRefractiveAxiconRayBundle",
    "VectorRefractiveAxiconResult",
    "build_tilted_vector_refractive_axicon_field",
    "fresnel_transmit_vector_3d",
    "lab_reference_eikonal_direction_consistency",
    "sample_vector_field_on_tilted_entrance",
    "spectral_normal_flux_au",
]
