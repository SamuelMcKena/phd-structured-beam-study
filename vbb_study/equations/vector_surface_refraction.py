"""Vector Snell/Fresnel boundary operator for arbitrary local surface normals.

This is the missing boundary-condition primitive needed for surface-by-surface
refractive optics.  It evaluates exact vector Snell refraction and local s/p
Fresnel transmission for each incident direction and surface normal.

Important scope: this module is a *surface boundary operator*.  It does not by
itself remap a sampled wave field between non-parallel curved surfaces.  A full
thick axicon/objective surface solver must combine this operator with physical
surface intersection/remapping or an independent prescription solver such as
OpticStudio.  The distinction is deliberate so a local Fresnel calculation is
not mislabeled as a complete curved-surface wave solver.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EPS = np.finfo(float).tiny


def _normalise(v: np.ndarray) -> np.ndarray:
    values = np.asarray(v, dtype=float)
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(norm <= EPS) or np.any(~np.isfinite(norm)):
        raise ValueError("direction/normal vectors must be finite and non-zero")
    return values / norm


@dataclass(frozen=True)
class SurfaceRefractionResult:
    transmitted_direction: np.ndarray
    reflected_direction: np.ndarray
    s_hat: np.ndarray
    p_incident_hat: np.ndarray
    p_transmitted_hat: np.ndarray
    t_s: np.ndarray
    t_p: np.ndarray
    r_s: np.ndarray
    r_p: np.ndarray
    T_s: np.ndarray
    T_p: np.ndarray
    R_s: np.ndarray
    R_p: np.ndarray
    total_internal_reflection: np.ndarray
    cos_theta_incident: np.ndarray
    cos_theta_transmitted: np.ndarray


def refract_vectors(
    incident_direction: np.ndarray,
    surface_normal_into_incident_medium: np.ndarray,
    *,
    n_incident: float,
    n_transmitted: float,
) -> SurfaceRefractionResult:
    """Exact vector Snell/Fresnel refraction at an arbitrary lossless interface.

    The supplied normal points from the transmitted medium into the incident
    medium, so a normally incident +z ray at an interface into material uses a
    normal of -z.  ``incident_direction`` points in the direction of energy
    propagation toward the interface.
    """

    n1 = float(n_incident)
    n2 = float(n_transmitted)
    if not np.isfinite(n1) or not np.isfinite(n2) or n1 <= 0.0 or n2 <= 0.0:
        raise ValueError("refractive indices must be finite and positive")
    ki = _normalise(np.asarray(incident_direction, dtype=float))
    normal = _normalise(np.asarray(surface_normal_into_incident_medium, dtype=float))
    ki, normal = np.broadcast_arrays(ki, normal)
    if ki.shape[-1] != 3:
        raise ValueError("incident_direction and surface normal must end in length 3")

    cos_i = -np.sum(ki * normal, axis=-1)
    if np.any(cos_i < -1e-12):
        raise ValueError("surface normal orientation is inconsistent with incident direction")
    cos_i = np.clip(cos_i, 0.0, 1.0)
    eta = n1 / n2
    sin2_t = eta * eta * np.maximum(1.0 - cos_i * cos_i, 0.0)
    tir = sin2_t > 1.0
    cos_t = np.sqrt(np.maximum(1.0 - sin2_t, 0.0))

    kt = eta * ki + (eta * cos_i - cos_t)[..., None] * normal
    kt = np.where(tir[..., None], np.nan, kt)
    valid = ~tir
    if np.any(valid):
        kt_valid = _normalise(kt[valid])
        kt = kt.copy()
        kt[valid] = kt_valid
    kr = ki + 2.0 * cos_i[..., None] * normal
    kr = _normalise(kr)

    s = np.cross(ki, normal)
    s_norm = np.linalg.norm(s, axis=-1, keepdims=True)
    # At exact normal incidence there is no unique s axis.  Choose a stable
    # transverse basis orthogonal to ki rather than allowing numerical noise to
    # define polarisation.
    normal_incidence = s_norm[..., 0] <= 1e-12
    if np.any(normal_incidence):
        trial = np.zeros_like(ki)
        trial[..., 0] = 1.0
        near_x = np.abs(np.sum(trial * ki, axis=-1)) > 0.9
        trial[near_x] = np.array([0.0, 1.0, 0.0])
        replacement = np.cross(ki, trial)
        s = np.where(normal_incidence[..., None], replacement, s)
    s = _normalise(s)
    p_i = _normalise(np.cross(s, ki))
    p_t = np.full_like(kt, np.nan)
    if np.any(valid):
        p_t[valid] = _normalise(np.cross(s[valid], kt[valid]))

    denom_s = n1 * cos_i + n2 * cos_t
    denom_p = n2 * cos_i + n1 * cos_t
    t_s = np.divide(2.0 * n1 * cos_i, denom_s, out=np.zeros_like(cos_i), where=denom_s != 0.0)
    t_p = np.divide(2.0 * n1 * cos_i, denom_p, out=np.zeros_like(cos_i), where=denom_p != 0.0)
    r_s = np.divide(n1 * cos_i - n2 * cos_t, denom_s, out=np.ones_like(cos_i), where=denom_s != 0.0)
    r_p = np.divide(n2 * cos_i - n1 * cos_t, denom_p, out=np.ones_like(cos_i), where=denom_p != 0.0)

    flux = np.divide(n2 * cos_t, n1 * np.maximum(cos_i, EPS))
    T_s = flux * t_s * t_s
    T_p = flux * t_p * t_p
    R_s = r_s * r_s
    R_p = r_p * r_p
    t_s = np.where(tir, 0.0, t_s)
    t_p = np.where(tir, 0.0, t_p)
    T_s = np.where(tir, 0.0, T_s)
    T_p = np.where(tir, 0.0, T_p)
    R_s = np.where(tir, 1.0, R_s)
    R_p = np.where(tir, 1.0, R_p)

    return SurfaceRefractionResult(
        transmitted_direction=kt,
        reflected_direction=kr,
        s_hat=s,
        p_incident_hat=p_i,
        p_transmitted_hat=p_t,
        t_s=t_s,
        t_p=t_p,
        r_s=r_s,
        r_p=r_p,
        T_s=T_s,
        T_p=T_p,
        R_s=R_s,
        R_p=R_p,
        total_internal_reflection=tir,
        cos_theta_incident=cos_i,
        cos_theta_transmitted=cos_t,
    )


def transmit_vector_field_local(
    electric_field: np.ndarray,
    incident_direction: np.ndarray,
    surface_normal_into_incident_medium: np.ndarray,
    *,
    n_incident: float,
    n_transmitted: float,
) -> tuple[np.ndarray, SurfaceRefractionResult]:
    """Apply local s/p Fresnel transmission to a vector electric field.

    ``electric_field[...,3]`` is projected into the incident s/p basis and
    reconstructed in the transmitted s/p basis.  This is exact for each local
    plane-wave boundary interaction, but spatial propagation/remapping across a
    curved surface is intentionally outside this function.
    """

    e = np.asarray(electric_field, dtype=np.complex128)
    if e.shape[-1] != 3 or np.any(~np.isfinite(e)):
        raise ValueError("electric_field must be finite and end in length 3")
    result = refract_vectors(
        incident_direction,
        surface_normal_into_incident_medium,
        n_incident=n_incident,
        n_transmitted=n_transmitted,
    )
    e, s, p_i, p_t = np.broadcast_arrays(e, result.s_hat, result.p_incident_hat, result.p_transmitted_hat)
    es = np.sum(e * s, axis=-1)
    ep = np.sum(e * p_i, axis=-1)
    transmitted = result.t_s[..., None] * es[..., None] * s + result.t_p[..., None] * ep[..., None] * p_t
    transmitted = np.where(result.total_internal_reflection[..., None], 0.0 + 0.0j, transmitted)
    return np.asarray(transmitted, dtype=np.complex128), result


__all__ = [
    "SurfaceRefractionResult",
    "refract_vectors",
    "transmit_vector_field_local",
]
