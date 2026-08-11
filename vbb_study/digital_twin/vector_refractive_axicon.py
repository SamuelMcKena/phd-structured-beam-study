"""Vector two-surface refractive axicon field solver.

This module removes the rigid-tilt block for *vector* axicon physics without
reusing the rejected scalar thin-phase surrogate.  It combines:

1. carrier-aware vector sampling onto the physical tilted entrance plane;
2. the validated two-surface plano-conical ray geometry;
3. vector Snell refraction at both dielectric interfaces;
4. local s/p Fresnel transmission of the full complex electric vector;
5. finite optical-path/eikonal phase;
6. ray-tube/Poynting-flux amplitude transport through the mapping Jacobian;
7. inverse remapping onto one fixed laboratory z-plane;
8. spectral transverse projection and a final global flux-closure correction;
9. vector angular-spectrum compatibility downstream.

The solver is a high-frequency vector-eikonal boundary-field construction for a
macroscopic refractive optic, followed by wave propagation.  It is not claimed
to be a full-volume FDTD/FEM solution of the glass axicon.  Absolute bench use
requires calibrated geometry/material values and adequate sampling.

References
----------
G. Yun, K. Crabtree and R. A. Chipman, Applied Optics 50, 2855-2865 (2011):
three-dimensional polarization ray tracing with refraction/diattenuation.
J. Kim et al., JOSA A 35, 526-535 (2018): vectorial diffraction using traced ray
and electromagnetic-field vectors to construct boundary fields.
Z. Bin and L. Zhu, Applied Optics 37, 2563-2568 (1998): oblique axicon
illumination, theory and experiment.
A. Thaning, Z. Jaroszewicz and A. T. Friberg, Applied Optics 42, 9-17 (2003):
oblique axicon caustics verified by diffraction simulation and experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np
from scipy.ndimage import map_coordinates

from vbb_study.digital_twin.vortex_refractive_axicon import (
    RefractiveAxiconBundle,
    RefractiveAxiconGeometry,
    trace_refractive_axicon_bundle,
)
from vbb_study.digital_twin.vortex_rotated_plane import (
    lab_to_tilted_plane,
    rotation_matrix,
)
from vbb_study.equations.fields import fft2c, make_xy_grid
from vbb_study.vector_field import (
    VectorField,
    propagate_vector_asm,
    spectral_transversality_residual,
)


EPS = np.finfo(float).tiny
TWOPI = 2.0 * np.pi


@dataclass(frozen=True)
class VectorRefractiveAxiconResult:
    """Regular vector boundary field immediately downstream of the real axicon."""

    field: VectorField
    entrance_surface_field_lab: VectorField
    geometry_bundle: RefractiveAxiconBundle
    coverage_mask: np.ndarray
    reference_x_lab_m: np.ndarray
    reference_y_lab_m: np.ndarray
    reference_z_lab_m: float
    reference_distance_m: np.ndarray
    optical_path_m: np.ndarray
    mapping_jacobian_signed: np.ndarray
    fresnel_power_surface1: np.ndarray
    fresnel_power_surface2: np.ndarray
    fresnel_reflectance_surface1: np.ndarray
    fresnel_reflectance_surface2: np.ndarray
    outgoing_direction_lab: np.ndarray
    metadata: Mapping[str, Any]


def _normalise_real(v: np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=float)
    norm = np.linalg.norm(arr, axis=-1, keepdims=True)
    if np.any(norm <= np.finfo(float).eps):
        raise ValueError("zero-length real vector")
    return arr / norm


def _vector_spectral_centroid_cpm(field: VectorField) -> tuple[float, float]:
    spectra = [fft2c(field.ex), fft2c(field.ey), fft2c(field.ez)]
    weight = sum(np.abs(spec) ** 2 for spec in spectra)
    total = float(np.sum(weight))
    if total <= EPS:
        return 0.0, 0.0
    FX = np.asarray(field.grid["FX"], dtype=float)
    FY = np.asarray(field.grid["FY"], dtype=float)
    return (
        float(np.sum(weight * FX) / total),
        float(np.sum(weight * FY) / total),
    )


def sample_vector_field_on_tilted_entrance(
    field: VectorField,
    *,
    tilt_x_rad: float,
    tilt_y_rad: float,
    spectral_center_cpm: tuple[float, float] | None = None,
) -> tuple[VectorField, np.ndarray, dict[str, Any]]:
    """Sample a lab-plane vector field on the physical tilted entrance plane.

    The three Cartesian field components are transformed with one *shared*
    carrier centre.  Components remain expressed in the fixed laboratory basis;
    only the sampled plane changes.  The source field is first projected onto
    the transverse Maxwell subspace so an artificial ``Ez=0`` paraxial input
    cannot survive as a non-transverse off-axis field.
    """

    projected = propagate_vector_asm(field, 0.0)
    fsx, fsy = (
        _vector_spectral_centroid_cpm(projected)
        if spectral_center_cpm is None
        else tuple(map(float, spectral_center_cpm))
    )
    spatial_frequency = float(field.medium_index) / float(field.wavelength_m)
    if fsx * fsx + fsy * fsy >= spatial_frequency * spatial_frequency:
        raise ValueError("vector spectral centre is non-propagating")
    fsz = math.sqrt(max(spatial_frequency * spatial_frequency - fsx * fsx - fsy * fsy, 0.0))
    incident_direction_lab = np.asarray([fsx, fsy, fsz], dtype=float) / spatial_frequency
    incident_direction_lab = _normalise_real(incident_direction_lab)

    # rotate_angular_spectrum is written in terms of wavelength in the sampled
    # medium.  VectorField.wavelength_m is the vacuum wavelength.
    effective_wavelength = float(field.wavelength_m) / float(field.medium_index)
    components: list[np.ndarray] = []
    component_meta: list[Mapping[str, Any]] = []
    for component in (projected.ex, projected.ey, projected.ez):
        mapped, meta = lab_to_tilted_plane(
            component,
            projected.grid,
            wavelength_m=effective_wavelength,
            tilt_x_rad=float(tilt_x_rad),
            tilt_y_rad=float(tilt_y_rad),
            spectral_center_cpm=(fsx, fsy),
        )
        components.append(np.asarray(mapped, dtype=np.complex128))
        component_meta.append(dict(meta))

    sampled = VectorField(
        ex=components[0],
        ey=components[1],
        ez=components[2],
        grid=projected.grid,
        wavelength_m=field.wavelength_m,
        medium_index=field.medium_index,
        metadata={
            **dict(field.metadata),
            "stage": "physical_tilted_axicon_entrance_surface",
            "components_basis": "fixed_lab_xyz",
            "surface_coordinates": "axicon_local_xy",
            "tilt_x_rad": float(tilt_x_rad),
            "tilt_y_rad": float(tilt_y_rad),
            "shared_source_spectral_center_cpm": [float(fsx), float(fsy)],
        },
    )
    return sampled, incident_direction_lab, {
        "source_transversality_residual": float(spectral_transversality_residual(projected)),
        "incident_direction_lab": incident_direction_lab.tolist(),
        "shared_source_spectral_center_cpm": [float(fsx), float(fsy)],
        "component_plane_transform_metadata": component_meta,
    }


def _fallback_s_axis(direction: np.ndarray) -> np.ndarray:
    d = _normalise_real(direction)
    trial_x = np.broadcast_to(np.asarray([1.0, 0.0, 0.0]), d.shape)
    trial_y = np.broadcast_to(np.asarray([0.0, 1.0, 0.0]), d.shape)
    trial = np.where((np.abs(d[..., 0]) < 0.9)[..., None], trial_x, trial_y)
    return _normalise_real(np.cross(d, trial))


def fresnel_transmit_vector_3d(
    electric1: np.ndarray,
    direction1: np.ndarray,
    direction2: np.ndarray,
    normal12: np.ndarray,
    *,
    n1: float,
    n2: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vector Fresnel transmission plus T/R power coefficients.

    ``normal12`` points from medium 1 toward medium 2.  The complex electric
    vector is decomposed into the local s/p basis at each ray, transmitted with
    the nonmagnetic Fresnel electric-field amplitudes, then reconstructed using
    the transmitted p direction.  Returned T and R are power ratios for the
    actual mixed polarization state, so ``T + R = 1`` for lossless media.
    """

    n1f = float(n1)
    n2f = float(n2)
    if n1f <= 0.0 or n2f <= 0.0:
        raise ValueError("refractive indices must be positive")

    d1 = _normalise_real(np.asarray(direction1, dtype=float))
    d2 = _normalise_real(np.asarray(direction2, dtype=float))
    normal = _normalise_real(np.asarray(normal12, dtype=float))
    d1, d2, normal = np.broadcast_arrays(d1, d2, normal)
    e1 = np.broadcast_to(np.asarray(electric1, dtype=np.complex128), d1.shape)
    e1 = e1 - np.sum(e1 * d1, axis=-1, keepdims=True) * d1

    cos_i = np.sum(d1 * normal, axis=-1)
    cos_t = np.sum(d2 * normal, axis=-1)
    if np.any(cos_i <= 0.0) or np.any(cos_t <= 0.0):
        raise ValueError("Fresnel transmission requires forward-going rays")

    s_raw = np.cross(normal, d1)
    s_norm = np.linalg.norm(s_raw, axis=-1)
    near_normal = s_norm < 1.0e-12
    s_axis = np.where(near_normal[..., None], _fallback_s_axis(d1), s_raw)
    s_axis = _normalise_real(s_axis)
    p1 = _normalise_real(np.cross(d1, s_axis))
    p2 = _normalise_real(np.cross(d2, s_axis))

    ts = (2.0 * n1f * cos_i) / (n1f * cos_i + n2f * cos_t)
    tp = (2.0 * n1f * cos_i) / (n2f * cos_i + n1f * cos_t)
    rs = (n1f * cos_i - n2f * cos_t) / (n1f * cos_i + n2f * cos_t)
    rp = (n2f * cos_i - n1f * cos_t) / (n2f * cos_i + n1f * cos_t)

    es = np.sum(e1 * s_axis, axis=-1)
    ep = np.sum(e1 * p1, axis=-1)
    transmitted = ts[..., None] * es[..., None] * s_axis + tp[..., None] * ep[..., None] * p2

    # At exact normal incidence the plane of incidence is degenerate.  Fresnel
    # amplitudes are polarization independent, so use the direct transverse map.
    if np.any(near_normal):
        t0 = 2.0 * n1f / (n1f + n2f)
        projected2 = e1 - np.sum(e1 * d2, axis=-1, keepdims=True) * d2
        transmitted = np.where(near_normal[..., None], t0 * projected2, transmitted)

    ein2 = np.sum(np.abs(e1) ** 2, axis=-1)
    eout2 = np.sum(np.abs(transmitted) ** 2, axis=-1)
    denominator = np.maximum(n1f * cos_i * ein2, EPS)
    trans_power = n2f * cos_t * eout2 / denominator

    reflected2 = np.abs(rs * es) ** 2 + np.abs(rp * ep) ** 2
    reflect_power = reflected2 / np.maximum(ein2, EPS)
    if np.any(near_normal):
        r0 = (n1f - n2f) / (n1f + n2f)
        reflect_power = np.where(near_normal, abs(r0) ** 2, reflect_power)

    zero_input = ein2 <= 100.0 * EPS
    trans_power = np.where(zero_input, 0.0, trans_power)
    reflect_power = np.where(zero_input, 0.0, reflect_power)
    return (
        np.asarray(transmitted, dtype=np.complex128),
        np.asarray(trans_power, dtype=float),
        np.asarray(reflect_power, dtype=float),
    )


def _mapping_jacobian_signed(
    x_reference: np.ndarray,
    y_reference: np.ndarray,
    *,
    dx_m: float,
    dy_m: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    dX_dy, dX_dx = np.gradient(np.asarray(x_reference, dtype=float), float(dy_m), float(dx_m))
    dY_dy, dY_dx = np.gradient(np.asarray(y_reference, dtype=float), float(dy_m), float(dx_m))
    det = dX_dx * dY_dy - dX_dy * dY_dx
    return np.asarray(det, dtype=float), {
        "dX_dx": dX_dx,
        "dX_dy": dX_dy,
        "dY_dx": dY_dx,
        "dY_dy": dY_dy,
    }


def _shifted_output_grid(n: int, window_m: float, centre_x_m: float, centre_y_m: float) -> dict[str, Any]:
    grid = make_xy_grid(int(n), float(window_m) / int(n))
    grid = dict(grid)
    grid["x"] = np.asarray(grid["x"], dtype=float) + float(centre_x_m)
    grid["y"] = np.asarray(grid.get("y", grid["x"] - float(centre_x_m)), dtype=float) + float(centre_y_m)
    grid["X"] = np.asarray(grid["X"], dtype=float) + float(centre_x_m)
    grid["Y"] = np.asarray(grid["Y"], dtype=float) + float(centre_y_m)
    grid["R"] = np.hypot(grid["X"], grid["Y"])
    grid["PHI"] = np.arctan2(grid["Y"], grid["X"])
    return grid


def _sample_regular(values: np.ndarray, iy: np.ndarray, ix: np.ndarray, *, order: int) -> np.ndarray:
    arr = np.asarray(values)
    coords = np.vstack([iy.ravel(), ix.ravel()])
    if np.iscomplexobj(arr):
        real = map_coordinates(arr.real, coords, order=order, mode="constant", cval=0.0, prefilter=order > 1)
        imag = map_coordinates(arr.imag, coords, order=order, mode="constant", cval=0.0, prefilter=order > 1)
        return (real + 1j * imag).reshape(ix.shape)
    return map_coordinates(arr, coords, order=order, mode="constant", cval=np.nan, prefilter=order > 1).reshape(ix.shape)


def _inverse_remap_vector_field(
    ray_field_lab: np.ndarray,
    valid_ray: np.ndarray,
    entrance_x: np.ndarray,
    entrance_y: np.ndarray,
    reference_x: np.ndarray,
    reference_y: np.ndarray,
    output_grid: Mapping[str, Any],
    *,
    max_iterations: int = 8,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Invert a smooth one-to-one ray map and sample the complex vector field.

    A sparse affine least-squares fit provides the initial inverse guess.  A
    Newton solve then uses the full sampled mapping and its spatial derivatives.
    This scales to production grids without constructing a multi-million-point
    Delaunay triangulation.
    """

    X = np.asarray(entrance_x, dtype=float)
    Y = np.asarray(entrance_y, dtype=float)
    XR = np.asarray(reference_x, dtype=float)
    YR = np.asarray(reference_y, dtype=float)
    valid = np.asarray(valid_ray, dtype=bool)
    if X.shape != Y.shape or X.shape != XR.shape or X.shape != YR.shape:
        raise ValueError("ray-map arrays must share one regular input shape")
    if np.asarray(ray_field_lab).shape != X.shape + (3,):
        raise ValueError("ray_field_lab must have shape input_shape + (3,)")

    x_axis = np.asarray(X[X.shape[0] // 2], dtype=float)
    y_axis = np.asarray(Y[:, Y.shape[1] // 2], dtype=float)
    dx = float(np.median(np.diff(x_axis)))
    dy = float(np.median(np.diff(y_axis)))
    if dx <= 0.0 or dy <= 0.0:
        raise ValueError("input mapping coordinates must increase")

    ids = np.flatnonzero(valid.ravel())
    if ids.size < 64:
        raise ValueError("too few valid rays to invert mapping")
    stride = max(1, ids.size // 4096)
    ids_fit = ids[::stride]
    A = np.column_stack([X.ravel()[ids_fit], Y.ravel()[ids_fit], np.ones(ids_fit.size)])
    coef_x, *_ = np.linalg.lstsq(A, XR.ravel()[ids_fit], rcond=None)
    coef_y, *_ = np.linalg.lstsq(A, YR.ravel()[ids_fit], rcond=None)
    matrix = np.asarray([[coef_x[0], coef_x[1]], [coef_y[0], coef_y[1]]], dtype=float)
    if abs(float(np.linalg.det(matrix))) < 1.0e-12:
        raise ValueError("ray-map affine seed is singular")
    inv_matrix = np.linalg.inv(matrix)
    offset = np.asarray([coef_x[2], coef_y[2]], dtype=float)

    XT = np.asarray(output_grid["X"], dtype=float)
    YT = np.asarray(output_grid["Y"], dtype=float)
    target = np.stack([XT - offset[0], YT - offset[1]], axis=-1)
    guess = target @ inv_matrix.T
    xq = guess[..., 0]
    yq = guess[..., 1]

    det_map, derivatives = _mapping_jacobian_signed(XR, YR, dx_m=dx, dy_m=dy)
    _ = det_map
    finite = np.ones_like(XT, dtype=bool)
    residual = np.full_like(XT, np.inf, dtype=float)
    for _iteration in range(int(max_iterations)):
        ix = (xq - x_axis[0]) / dx
        iy = (yq - y_axis[0]) / dy
        xr = _sample_regular(XR, iy, ix, order=1)
        yr = _sample_regular(YR, iy, ix, order=1)
        dXdx = _sample_regular(derivatives["dX_dx"], iy, ix, order=1)
        dXdy = _sample_regular(derivatives["dX_dy"], iy, ix, order=1)
        dYdx = _sample_regular(derivatives["dY_dx"], iy, ix, order=1)
        dYdy = _sample_regular(derivatives["dY_dy"], iy, ix, order=1)
        rx = xr - XT
        ry = yr - YT
        det = dXdx * dYdy - dXdy * dYdx
        good = (
            np.isfinite(rx)
            & np.isfinite(ry)
            & np.isfinite(det)
            & (np.abs(det) > 1.0e-10)
        )
        finite &= good
        safe_det = np.where(good, det, 1.0)
        delta_x = (dYdy * rx - dXdy * ry) / safe_det
        delta_y = (-dYdx * rx + dXdx * ry) / safe_det
        xq = np.where(good, xq - delta_x, xq)
        yq = np.where(good, yq - delta_y, yq)
        residual = np.hypot(rx, ry)

    ix = (xq - x_axis[0]) / dx
    iy = (yq - y_axis[0]) / dy
    valid_sample = _sample_regular(valid.astype(float), iy, ix, order=1)
    dx_out = float(output_grid["dx"])
    tolerance = max(0.15 * dx_out, 1.0e-10)
    coverage = (
        finite
        & np.isfinite(valid_sample)
        & (valid_sample > 0.95)
        & np.isfinite(residual)
        & (residual <= tolerance)
        & (ix >= 0.0)
        & (ix <= X.shape[1] - 1.0)
        & (iy >= 0.0)
        & (iy <= X.shape[0] - 1.0)
    )

    components = []
    for index in range(3):
        sampled = _sample_regular(np.asarray(ray_field_lab)[..., index], iy, ix, order=3)
        components.append(np.where(coverage, sampled, 0.0j))
    field = np.stack(components, axis=-1)
    residual_values = residual[coverage]
    return field, coverage, {
        "inverse_mapping_coverage_fraction": float(np.mean(coverage)),
        "inverse_mapping_median_residual_m": float(np.median(residual_values)) if residual_values.size else float("nan"),
        "inverse_mapping_p95_residual_m": float(np.percentile(residual_values, 95.0)) if residual_values.size else float("nan"),
        "inverse_mapping_max_residual_m": float(np.max(residual_values)) if residual_values.size else float("nan"),
        "inverse_mapping_tolerance_m": float(tolerance),
    }


def spectral_normal_flux_au(field: VectorField) -> float:
    """Plane-integrated +z Poynting flux up to the common factor 1/(2 Z0)."""

    ax = fft2c(field.ex)
    ay = fft2c(field.ey)
    az = fft2c(field.ez)
    FX = np.asarray(field.grid["FX"], dtype=float)
    FY = np.asarray(field.grid["FY"], dtype=float)
    spatial_frequency = float(field.medium_index) / float(field.wavelength_m)
    transverse_sq = FX * FX + FY * FY
    propagating = transverse_sq < spatial_frequency * spatial_frequency
    fz = np.sqrt(np.maximum(spatial_frequency * spatial_frequency - transverse_sq, 0.0))
    cos_theta = np.where(propagating, fz / max(spatial_frequency, EPS), 0.0)
    spectral_energy = np.abs(ax) ** 2 + np.abs(ay) ** 2 + np.abs(az) ** 2
    ny, nx = field.ex.shape
    dx = float(field.grid["dx"])
    dy = float(field.grid.get("dy", dx))
    parseval_area = dx * dy / float(nx * ny)
    return float(field.medium_index) * parseval_area * float(np.sum(cos_theta * spectral_energy))


def build_tilted_vector_refractive_axicon_field(
    upstream_field: VectorField,
    *,
    geometry: RefractiveAxiconGeometry,
    tilt_x_rad: float,
    tilt_y_rad: float,
    axicon_decentre_m: tuple[float, float] = (0.0, 0.0),
    reference_gap_m: float = 0.5e-3,
    output_n: int | None = None,
    output_window_m: float | None = None,
    apex_exclusion_radius_m: float = 0.0,
    minimum_mapping_jacobian: float = 1.0e-5,
    maximum_nyquist_fraction: float = 0.90,
) -> VectorRefractiveAxiconResult:
    """Trace and remap a spatial vector field through a rigidly tilted axicon.

    The returned :class:`VectorField` lies on a fixed *laboratory* z-plane.  Its
    x/y coordinates retain physical beam steering/decentre; no beam-following
    recentering is applied.
    """

    geometry.validate()
    if not np.isclose(float(upstream_field.medium_index), float(geometry.external_index), rtol=0.0, atol=1e-12):
        raise ValueError("upstream field medium_index must equal axicon external_index")

    surface, incident_direction_lab, surface_meta = sample_vector_field_on_tilted_entrance(
        upstream_field,
        tilt_x_rad=float(tilt_x_rad),
        tilt_y_rad=float(tilt_y_rad),
    )
    X = np.asarray(surface.grid["X"], dtype=float)
    Y = np.asarray(surface.grid["Y"], dtype=float)
    dx = float(surface.grid["dx"])
    dy = float(surface.grid.get("dy", dx))
    dec_x, dec_y = map(float, axicon_decentre_m)
    X_rel = X - dec_x
    Y_rel = Y - dec_y

    bundle = trace_refractive_axicon_bundle(
        X_rel,
        Y_rel,
        geometry=geometry,
        tilt_x_rad=float(tilt_x_rad),
        tilt_y_rad=float(tilt_y_rad),
        incident_direction_lab=incident_direction_lab,
        reference_gap_m=max(float(reference_gap_m), 10.0 * max(dx, dy)),
        apex_exclusion_radius_m=float(apex_exclusion_radius_m),
    )
    rotation = rotation_matrix(float(tilt_x_rad), float(tilt_y_rad))
    tangent_offset_lab = rotation @ np.asarray([dec_x, dec_y, 0.0], dtype=float)
    exit_lab = np.asarray(bundle.exit_point_lab_m, dtype=float) + tangent_offset_lab
    outgoing_lab = np.asarray(bundle.outgoing_lab, dtype=float)
    valid = np.asarray(bundle.valid, dtype=bool).copy()

    z_reference = float(np.max(exit_lab[..., 2][valid])) + max(float(reference_gap_m), 10.0 * max(dx, dy))
    denominator = outgoing_lab[..., 2]
    t_reference = (z_reference - exit_lab[..., 2]) / np.where(np.abs(denominator) > EPS, denominator, np.nan)
    valid &= np.isfinite(t_reference) & (t_reference > 0.0) & (denominator > 0.0)
    reference_x = exit_lab[..., 0] + t_reference * outgoing_lab[..., 0]
    reference_y = exit_lab[..., 1] + t_reference * outgoing_lab[..., 1]

    jac_signed, _derivatives = _mapping_jacobian_signed(reference_x, reference_y, dx_m=dx, dy_m=dy)
    jac_abs = np.abs(jac_signed)
    interior = valid.copy()
    if min(interior.shape) > 6:
        interior[:2, :] = False
        interior[-2:, :] = False
        interior[:, :2] = False
        interior[:, -2:] = False
    sign_values = jac_signed[interior & np.isfinite(jac_signed) & (jac_abs > float(minimum_mapping_jacobian))]
    positive_fraction = float(np.mean(sign_values > 0.0)) if sign_values.size else 0.0
    negative_fraction = float(np.mean(sign_values < 0.0)) if sign_values.size else 0.0
    if positive_fraction > 0.02 and negative_fraction > 0.02:
        raise ValueError("ray mapping folds before the chosen lab reference plane; move the reference plane closer to the optic")
    valid &= np.isfinite(jac_abs) & (jac_abs > float(minimum_mapping_jacobian))

    # Full complex vector Fresnel transport.  Surface samples are stored in lab
    # components, so rotate them into the axicon-local basis for both interfaces.
    e_lab = np.stack([surface.ex, surface.ey, surface.ez], axis=-1)
    e_local = e_lab @ rotation
    flat_normal = np.asarray([0.0, 0.0, 1.0], dtype=float)
    d_local = np.asarray(bundle.incident_local, dtype=float)
    d_internal = np.asarray(bundle.internal_local, dtype=float)
    e_internal, T1, R1 = fresnel_transmit_vector_3d(
        e_local,
        d_local,
        d_internal,
        flat_normal,
        n1=float(geometry.external_index),
        n2=float(geometry.refractive_index),
    )
    internal_bundle = np.broadcast_to(d_internal, np.asarray(bundle.outgoing_local).shape)
    e_out_local, T2, R2 = fresnel_transmit_vector_3d(
        e_internal,
        internal_bundle,
        np.asarray(bundle.outgoing_local, dtype=float),
        np.asarray(bundle.exit_normal_local, dtype=float),
        n1=float(geometry.refractive_index),
        n2=float(geometry.external_index),
    )
    e_out_lab = e_out_local @ rotation.T

    input_intensity = np.sum(np.abs(e_lab) ** 2, axis=-1)
    cumulative_T = np.asarray(T1 * T2, dtype=float)
    cos_in = float(bundle.input_normal_cosine)
    cos_out = np.asarray(outgoing_lab[..., 2], dtype=float)
    output_norm = np.sqrt(np.sum(np.abs(e_out_lab) ** 2, axis=-1))
    unit_polarization = np.divide(
        e_out_lab,
        output_norm[..., None],
        out=np.zeros_like(e_out_lab),
        where=output_norm[..., None] > 100.0 * EPS,
    )
    desired_intensity = (
        input_intensity
        * cos_in
        * cumulative_T
        / np.maximum(cos_out * jac_abs, EPS)
    )
    opl = (
        float(geometry.refractive_index) * np.asarray(bundle.internal_distance_m, dtype=float)
        + float(geometry.external_index) * np.asarray(t_reference, dtype=float)
    )
    opl_reference = float(np.median(opl[valid]))
    propagation_phase = np.exp(
        1j * (TWOPI / float(upstream_field.wavelength_m)) * (opl - opl_reference)
    )
    ray_field_lab = (
        unit_polarization
        * np.sqrt(np.maximum(desired_intensity, 0.0))[..., None]
        * propagation_phase[..., None]
    )
    ray_field_lab = np.where(valid[..., None], ray_field_lab, 0.0j)

    interface1_energy_error = np.abs((T1 + R1) - 1.0)
    interface2_energy_error = np.abs((T2 + R2) - 1.0)
    illuminated = valid & (input_intensity > 1.0e-12 * max(float(np.max(input_intensity)), EPS))
    fresnel1_max_error = float(np.max(interface1_energy_error[illuminated])) if np.any(illuminated) else 0.0
    fresnel2_max_error = float(np.max(interface2_energy_error[illuminated])) if np.any(illuminated) else 0.0

    input_flux = (
        float(geometry.external_index)
        * cos_in
        * float(np.sum(input_intensity[valid]))
        * dx
        * dy
    )
    expected_transmitted_flux = (
        float(geometry.external_index)
        * cos_in
        * float(np.sum(input_intensity[valid] * cumulative_T[valid]))
        * dx
        * dy
    )
    reconstructed_ray_flux = (
        float(geometry.external_index)
        * float(np.sum(np.sum(np.abs(ray_field_lab[valid]) ** 2, axis=-1) * cos_out[valid] * jac_abs[valid]))
        * dx
        * dy
    )

    x_values = reference_x[valid]
    y_values = reference_y[valid]
    centre_x = 0.5 * (float(np.min(x_values)) + float(np.max(x_values)))
    centre_y = 0.5 * (float(np.min(y_values)) + float(np.max(y_values)))
    span = max(float(np.ptp(x_values)), float(np.ptp(y_values)))
    input_window = max(float(np.ptp(X) + dx), float(np.ptp(Y) + dy))
    window = float(output_window_m) if output_window_m is not None else max(input_window, 1.04 * span)
    n_out = int(upstream_field.ex.shape[0] if output_n is None else output_n)
    if n_out < 64 or window <= 0.0:
        raise ValueError("output_n must be >=64 and output_window_m positive")
    output_grid = _shifted_output_grid(n_out, window, centre_x, centre_y)

    spatial_frequency = float(geometry.external_index) / float(upstream_field.wavelength_m)
    required_fx = spatial_frequency * float(np.max(np.abs(outgoing_lab[..., 0][valid])))
    required_fy = spatial_frequency * float(np.max(np.abs(outgoing_lab[..., 1][valid])))
    nyquist = 0.5 / float(output_grid["dx"])
    ray_nyquist_fraction = max(required_fx, required_fy) / max(nyquist, EPS)
    if ray_nyquist_fraction > float(maximum_nyquist_fraction):
        raise ValueError(
            "output sampling cannot represent the refracted vector wavevectors: "
            f"required/nyquist={ray_nyquist_fraction:.3f} > {maximum_nyquist_fraction:.3f}; "
            "increase output_n or reduce output_window_m"
        )

    remapped, coverage, remap_meta = _inverse_remap_vector_field(
        ray_field_lab,
        valid,
        X,
        Y,
        reference_x,
        reference_y,
        output_grid,
    )
    raw = VectorField(
        ex=remapped[..., 0],
        ey=remapped[..., 1],
        ez=remapped[..., 2],
        grid=output_grid,
        wavelength_m=upstream_field.wavelength_m,
        medium_index=geometry.external_index,
        metadata={
            "stage": "vector_refractive_axicon_fixed_lab_reference_plane_pre_projection",
            "reference_z_lab_m": z_reference,
        },
    )
    pre_projection_residual = float(spectral_transversality_residual(raw))
    projected = propagate_vector_asm(raw, 0.0)
    flux_before_global_closure = spectral_normal_flux_au(projected)
    if flux_before_global_closure <= EPS or expected_transmitted_flux <= EPS:
        raise ValueError("zero vector flux after refractive axicon remapping")
    global_scale = math.sqrt(expected_transmitted_flux / flux_before_global_closure)
    final_field = VectorField(
        ex=projected.ex * global_scale,
        ey=projected.ey * global_scale,
        ez=projected.ez * global_scale,
        grid=output_grid,
        wavelength_m=projected.wavelength_m,
        medium_index=projected.medium_index,
        metadata={
            **dict(projected.metadata),
            "stage": "vector_refractive_axicon_fixed_lab_reference_plane",
            "reference_z_lab_m": z_reference,
            "reference_centre_lab_m": [centre_x, centre_y, z_reference],
        },
    )
    final_flux = spectral_normal_flux_au(final_field)
    final_transversality = float(spectral_transversality_residual(final_field))

    spectrum = np.abs(fft2c(final_field.ex)) ** 2 + np.abs(fft2c(final_field.ey)) ** 2 + np.abs(fft2c(final_field.ez)) ** 2
    edge = max(2, int(0.04 * n_out))
    edge_mask = np.zeros_like(spectrum, dtype=bool)
    edge_mask[:edge, :] = True
    edge_mask[-edge:, :] = True
    edge_mask[:, :edge] = True
    edge_mask[:, -edge:] = True
    spectral_edge_fraction = float(np.sum(spectrum[edge_mask]) / max(float(np.sum(spectrum)), EPS))

    return VectorRefractiveAxiconResult(
        field=final_field,
        entrance_surface_field_lab=surface,
        geometry_bundle=bundle,
        coverage_mask=np.asarray(coverage, dtype=bool),
        reference_x_lab_m=np.asarray(reference_x, dtype=float),
        reference_y_lab_m=np.asarray(reference_y, dtype=float),
        reference_z_lab_m=float(z_reference),
        reference_distance_m=np.asarray(t_reference, dtype=float),
        optical_path_m=np.asarray(opl, dtype=float),
        mapping_jacobian_signed=np.asarray(jac_signed, dtype=float),
        fresnel_power_surface1=np.asarray(T1, dtype=float),
        fresnel_power_surface2=np.asarray(T2, dtype=float),
        fresnel_reflectance_surface1=np.asarray(R1, dtype=float),
        fresnel_reflectance_surface2=np.asarray(R2, dtype=float),
        outgoing_direction_lab=np.asarray(outgoing_lab, dtype=float),
        metadata={
            "outcome": "VECTOR-TWO-SURFACE-REFRACTIVE-AXICON-LAB-PLANE",
            "model_class": "vector_eikonal_surface_trace_plus_vector_wave_boundary_field",
            "surface_order": "flat_entrance_then_conical_exit",
            "tilt_x_rad": float(tilt_x_rad),
            "tilt_y_rad": float(tilt_y_rad),
            "axicon_decentre_m": [dec_x, dec_y],
            "reference_z_lab_m": float(z_reference),
            "reference_centre_lab_m": [centre_x, centre_y, z_reference],
            "output_n": int(n_out),
            "output_window_m": float(window),
            "output_dx_m": float(output_grid["dx"]),
            "valid_ray_fraction": float(np.mean(valid)),
            "coverage_fraction": float(np.mean(coverage)),
            "mapping_positive_fraction": positive_fraction,
            "mapping_negative_fraction": negative_fraction,
            "minimum_mapping_jacobian": float(minimum_mapping_jacobian),
            "ray_required_nyquist_fraction": float(ray_nyquist_fraction),
            "spectral_edge_power_fraction": spectral_edge_fraction,
            "interface1_max_abs_R_plus_T_minus_1": fresnel1_max_error,
            "interface2_max_abs_R_plus_T_minus_1": fresnel2_max_error,
            "input_flux_au": float(input_flux),
            "expected_transmitted_flux_au": float(expected_transmitted_flux),
            "reconstructed_ray_flux_au": float(reconstructed_ray_flux),
            "ray_flux_closure_ratio": float(reconstructed_ray_flux / max(expected_transmitted_flux, EPS)),
            "interpolated_projected_flux_before_global_closure_au": float(flux_before_global_closure),
            "interpolation_global_amplitude_correction": float(global_scale),
            "final_spectral_normal_flux_au": float(final_flux),
            "final_flux_closure_ratio": float(final_flux / max(expected_transmitted_flux, EPS)),
            "pre_projection_transversality_residual": pre_projection_residual,
            "final_transversality_residual": final_transversality,
            "opl_reference_m": opl_reference,
            "entrance_surface_sampling": surface_meta,
            **remap_meta,
            "physics_limit": (
                "macroscopic vector eikonal/Fresnel boundary-field model; full-volume Maxwell scattering, "
                "multiple internal reflections, coatings, measured surface figure and nonlinear glass response are separate"
            ),
            "report_figures_authorised": False,
        },
    )


def lab_reference_eikonal_direction_consistency(
    result: VectorRefractiveAxiconResult,
    *,
    wavelength_m: float,
    external_index: float,
    trim_pixels: int = 3,
) -> dict[str, float]:
    """Verify that finite OPL gradients reproduce traced transverse wavevectors."""

    bundle = result.geometry_bundle
    X = np.asarray(bundle.entrance_x_m, dtype=float)
    Y = np.asarray(bundle.entrance_y_m, dtype=float)
    dx = float(np.median(np.diff(X[X.shape[0] // 2])))
    dy = float(np.median(np.diff(Y[:, Y.shape[1] // 2])))
    k0 = TWOPI / float(wavelength_m)
    kext = k0 * float(external_index)
    rotation = rotation_matrix(
        float(result.metadata["tilt_x_rad"]),
        float(result.metadata["tilt_y_rad"]),
    )
    incident_lab = rotation @ np.asarray(bundle.incident_local, dtype=float)
    incident_local_phase = kext * (
        float(bundle.incident_local[0]) * X + float(bundle.incident_local[1]) * Y
    )
    phase = incident_local_phase + k0 * np.asarray(result.optical_path_m, dtype=float)
    dphase_dy, dphase_dx = np.gradient(phase, dy, dx)

    det, terms = _mapping_jacobian_signed(
        result.reference_x_lab_m,
        result.reference_y_lab_m,
        dx_m=dx,
        dy_m=dy,
    )
    gx = (
        terms["dY_dy"] * dphase_dx - terms["dX_dy"] * dphase_dy
    ) / np.where(np.abs(det) > 1.0e-15, det, np.nan)
    gy = (
        -terms["dY_dx"] * dphase_dx + terms["dX_dx"] * dphase_dy
    ) / np.where(np.abs(det) > 1.0e-15, det, np.nan)
    expected_x = kext * np.asarray(result.outgoing_direction_lab[..., 0], dtype=float)
    expected_y = kext * np.asarray(result.outgoing_direction_lab[..., 1], dtype=float)
    scale = np.maximum(np.hypot(expected_x, expected_y), 0.01 * kext)
    error = np.hypot(gx - expected_x, gy - expected_y) / scale

    valid = np.asarray(bundle.valid, dtype=bool) & np.isfinite(error) & (np.abs(det) > 1.0e-8)
    trim = int(trim_pixels)
    if trim > 0:
        interior = np.zeros_like(valid)
        interior[trim:-trim, trim:-trim] = True
        valid &= interior
    values = error[valid]
    if values.size < 32:
        raise ValueError("too few rays for lab-plane eikonal direction check")
    return {
        "median_relative_direction_error": float(np.median(values)),
        "p95_relative_direction_error": float(np.percentile(values, 95.0)),
        "max_relative_direction_error": float(np.max(values)),
        "incident_lab_x": float(incident_lab[0]),
        "incident_lab_y": float(incident_lab[1]),
        "incident_lab_z": float(incident_lab[2]),
    }


__all__ = [
    "VectorRefractiveAxiconResult",
    "build_tilted_vector_refractive_axicon_field",
    "fresnel_transmit_vector_3d",
    "lab_reference_eikonal_direction_consistency",
    "sample_vector_field_on_tilted_entrance",
    "spectral_normal_flux_au",
]
