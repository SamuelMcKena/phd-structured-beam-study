"""Research-grounded system-error taxonomy for the vortex-Bessel bench.

This module is governance/validation infrastructure.  It prevents a generic
phase perturbation from being presented as the physical response of a specific
misaligned component.  Each error is tied to the plane where it occurs, the
physical model required, a literature/independent validation target, and a
report-readiness status.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal

Status = Literal[
    "implemented_screening",
    "implemented_needs_validation",
    "calibration_required",
    "solver_required",
    "separate_downstream_branch",
]


@dataclass(frozen=True)
class SystemErrorSpec:
    error_id: str
    family: str
    physical_plane: str
    physical_parameter: str
    required_model: str
    expected_signature: str
    validation_target: str
    calibration_blocker: str
    status: Status
    report_ready: bool = False


SYSTEM_ERRORS: tuple[SystemErrorSpec, ...] = (
    # Input beam state -------------------------------------------------------
    SystemErrorSpec(
        "IN-POINT", "input_beam", "before_SLM1", "input pointing angle",
        "oblique Gaussian/plane-wave direction cosines propagated through SLM carrier and fixed 4F iris",
        "Bessel axis steering at small angle; +1-order displacement and eventual iris clipping",
        "grating momentum law plus oblique-axicon diffraction benchmark",
        "measured input pointing angle", "implemented_needs_validation",
    ),
    SystemErrorSpec(
        "IN-DEC", "input_beam", "before_SLM1", "lateral beam decentre",
        "translated Gaussian amplitude with nominal propagation direction; full downstream route rebuilt",
        "asymmetric conical illumination; for small beam/axicon displacement the focal line remains approximately parallel to the optical axis",
        "transversal axicon-misalignment literature benchmark",
        "measured beam centre relative to axicon apex", "implemented_needs_validation",
    ),
    SystemErrorSpec(
        "IN-RAD", "input_beam", "before_SLM1", "beam radius",
        "Gaussian beam parameter changed before the SLMs",
        "changed Bessel-zone length and axial envelope; possible transition toward lens-like real-axicon behaviour for small beams",
        "Gaussian-illuminated axicon calculations and real-axicon beam-size dependence",
        "measured 1/e field-amplitude radius at the SLM/axicon planes", "implemented_needs_validation",
    ),
    SystemErrorSpec(
        "IN-CURV", "input_beam", "before_SLM1", "wavefront curvature / collimation error",
        "Gaussian q-parameter or explicit quadratic phase from measured radius of curvature",
        "shifted axial intensity envelope and altered formation region",
        "published axicon divergence/wavefront-curvature dependence plus independent Fresnel propagation",
        "measured wavefront curvature (Shack-Hartmann)", "solver_required",
    ),
    SystemErrorSpec(
        "IN-ELLIP", "input_beam", "before_SLM1", "ellipticity / astigmatic Gaussian",
        "elliptical Gaussian with independent wx, wy and optional independent qx, qy",
        "broken cylindrical symmetry, unequal ring radii/side lobes and astigmatic Bessel structure",
        "independent Fresnel/ASM parity and zero-ellipticity recovery",
        "measured wx, wy and beam orientation", "solver_required",
    ),
    SystemErrorSpec(
        "IN-JITTER", "input_beam", "before_SLM1", "shot-to-shot pointing/decentre jitter",
        "ensemble/Monte-Carlo propagation over measured angular and translational distributions",
        "time-averaged blur, fluctuating peak/ring position and reduced contrast",
        "Monte-Carlo convergence and measured camera time-series statistics",
        "measured pointing/decentre jitter distribution", "calibration_required",
    ),

    # SLM / hologram --------------------------------------------------------
    SystemErrorSpec(
        "SLM-LUT", "slm", "SLM1_and_SLM2", "phase LUT / phase stroke error",
        "measured grey-to-phase transfer applied pixelwise before quantisation",
        "wrong vortex/carrier phase depth, order leakage and morphology degradation",
        "binary/blazed-grating diffraction calibration against measured LUT",
        "measured NIR-149 phase LUT/stroke for each panel", "calibration_required",
    ),
    SystemErrorSpec(
        "SLM-SPNU", "slm", "SLM1_and_SLM2", "spatial phase-response nonuniformity and panel flatness",
        "measured static wavefront map plus spatially varying grey-to-phase response",
        "coma/astigmatism/low-order distortion and spatially varying diffraction efficiency",
        "Shack-Hartmann/interferometric residual wavefront after compensation",
        "measured wavefront/response map for each SLM", "calibration_required",
    ),
    SystemErrorSpec(
        "SLM-FRINGE", "slm", "SLM1_and_SLM2", "LCOS fringing-field pixel crosstalk",
        "measured flyback-width or direction-dependent convolution/subpixel model",
        "blurred sharp phase edges and reduced diffraction efficiency",
        "measured diffraction efficiency of blazed/binary gratings versus model",
        "panel-specific fringing response", "calibration_required",
    ),
    SystemErrorSpec(
        "SLM-PIX", "slm", "SLM1_and_SLM2", "pixel aperture/fill factor/quantisation",
        "resolved pixel aperture where numerically sampled; otherwise explicit throughput plus analytic sinc/order model",
        "sinc envelope, order redistribution, zero-order leakage and phase-quantisation efficiency loss",
        "analytic pixelated-hologram diffraction-order efficiencies",
        "manufacturer fill factor and confirmed panel geometry", "implemented_screening",
    ),
    SystemErrorSpec(
        "SLM-REG", "slm", "SLM1_relative_to_SLM2", "relative hologram decentre/rotation/magnification",
        "coordinate transform of the commanded phase on the physical panel before modulation",
        "vortex-core displacement, phase discontinuity mismatch and altered selected-order field",
        "identity-transform recovery plus known imposed translation/rotation tests",
        "camera/registration calibration between panels", "solver_required",
    ),
    SystemErrorSpec(
        "SLM-INC", "slm", "SLM reflection geometry", "incidence-angle-dependent phase response",
        "angle-dependent measured LCOS phase-response curve, not a scalar universal LUT",
        "reduced/changed phase modulation depth at sufficiently oblique incidence",
        "measured phase response versus incidence angle",
        "actual SLM incidence angle and panel-specific angular calibration", "calibration_required",
    ),

    # 4F spatial filter -----------------------------------------------------
    SystemErrorSpec(
        "4F-IRIS-X", "fourf", "Fourier_plane", "iris lateral offset",
        "fixed physical circular aperture translated in Fourier coordinates",
        "asymmetric spectrum clipping, centroid shift and order-dependent morphology change",
        "direct spectrum/aperture overlap and zero-offset recovery",
        "measured iris position", "implemented_needs_validation",
    ),
    SystemErrorSpec(
        "4F-IRIS-R", "fourf", "Fourier_plane", "iris radius / pinhole opening",
        "physical Fourier-plane aperture with calibrated radius",
        "too-large iris passes unwanted orders/zero order; too-small iris clips desired spatial bandwidth",
        "measured Fourier-plane image plus passed-power/order-purity benchmark",
        "measured iris diameter and Fourier-plane scale", "calibration_required",
    ),
    SystemErrorSpec(
        "4F-LENS-AB", "fourf", "4F_lenses", "Fourier-lens aberration",
        "explicit lens pupil/OPD or measured wavefront, propagated through the actual 4F distances",
        "distorted Fourier pattern and reconstructed beam even when hologram itself is correct",
        "measured lens/system wavefront and published Fourier-transform-lens aberration behaviour",
        "lens prescription or measured 4F wavefront", "solver_required",
    ),
    SystemErrorSpec(
        "4F-DESPACE", "fourf", "4F_lenses", "lens longitudinal spacing/focus error",
        "explicit Fresnel propagation + thin/known lens phases at actual distances",
        "Fourier plane no longer coincident with iris; residual quadratic phase and imperfect reconstruction",
        "ABCD/Fresnel zero-error recovery and controlled despace benchmark",
        "measured lens separations/focal lengths", "solver_required",
    ),
    SystemErrorSpec(
        "4F-DEC-TILT", "fourf", "4F_lenses", "lens decentre/tilt",
        "explicit decentered/tilted lens pupil and phase at its physical plane",
        "coma/astigmatism, order displacement and asymmetric spatial filtering",
        "paraxial chief-ray prediction plus independent diffraction/ray-trace comparison",
        "measured lens centres/tilts", "solver_required",
    ),

    # Axicon ---------------------------------------------------------------
    SystemErrorSpec(
        "AX-DEC", "axicon", "axicon", "beam-apex lateral decentre",
        "translate the physical axicon sag/clear aperture relative to the beam; do not add steering phase",
        "asymmetric illumination; small displacement should preserve an approximately parallel focal line",
        "Dufour et al. transverse-misalignment benchmark",
        "measured apex and beam centres plus clear aperture", "implemented_needs_validation",
    ),
    SystemErrorSpec(
        "AX-TILT", "axicon", "axicon", "rigid-body axicon tilt / oblique illumination",
        "rotated-plane angular-spectrum propagation plus refractive/conical transmission; full Snell/Fresnel for quantitative large-angle work",
        "broadened/astigmatic focal segment and astroid-like caustics as obliquity grows",
        "Zhao-Li and Thaning-Jaroszewicz-Friberg oblique-illumination benchmarks",
        "measured axicon tilt and exact surface-angle convention", "solver_required",
    ),
    SystemErrorSpec(
        "AX-TIP-ROUND", "axicon", "axicon", "rounded/hyperboloidal apex",
        "measured or hyperboloidal sag inserted into Fresnel/ASM propagation",
        "strong axial modulation from interference of lens-like central and conical contributions",
        "independent radial Fresnel/Hankel solver and real-axicon literature oscillation scale",
        "measured apex profile/rounding parameter", "implemented_needs_validation",
    ),
    SystemErrorSpec(
        "AX-TIP-FLAT", "axicon", "axicon", "flat/blunt truncation",
        "explicit central flat plus conical flank",
        "central-wave interference and axial modulation; B0 more sensitive than vortex orders",
        "Bessel-vortex blunt-tip literature order-sensitivity benchmark",
        "measured flat-tip radius/profile", "implemented_needs_validation",
    ),
    SystemErrorSpec(
        "AX-ANGLE", "axicon", "axicon", "base-angle manufacturing error",
        "exact refractive cone-angle/Snell relation for quantitative geometry",
        "changed transverse kr, ring/core scale and Bessel-zone geometry",
        "exact Snell versus shallow-angle limit and measured cone angle",
        "manufacturer/metrology angle and convention", "implemented_needs_validation",
    ),
    SystemErrorSpec(
        "AX-N", "axicon", "axicon", "refractive-index uncertainty/dispersion",
        "wavelength-dependent material index in exact refractive geometry",
        "changed cone deflection and kr",
        "Sellmeier/manufacturer index or measured material",
        "actual axicon glass and index at 1029 nm", "calibration_required",
    ),
    SystemErrorSpec(
        "AX-AP", "axicon", "axicon", "finite clear aperture / chipped edge / clipping",
        "physical aperture multiplied with sag transmission at the same translated/tilted optic coordinates",
        "finite-zone shortening and hard-edge diffraction/ripple if significantly illuminated",
        "window-convergence plus measured-aperture sensitivity",
        "actual clear aperture and edge condition", "calibration_required",
    ),
    SystemErrorSpec(
        "AX-FIG", "axicon", "axicon", "surface figure / local slope error",
        "measured surface-height map converted to OPD; Zernikes only as labelled generic sensitivity when no map exists",
        "local wavefront distortion, asymmetry and axial/ring nonuniformity",
        "metrology-map replay and zero-map recovery",
        "surface metrology", "calibration_required",
    ),

    # Generic upstream aberration -----------------------------------------
    SystemErrorSpec(
        "WF-ZERN", "wavefront", "declared_plane", "defocus/astigmatism/coma/spherical",
        "RMS-normalised Zernike OPD at an explicitly declared plane",
        "controlled generic sensitivity only; not automatically attributed to a specific optic",
        "analytic orthogonality/RMS checks plus Shack-Hartmann comparison when measured",
        "measured system wavefront if used as a lab claim", "implemented_screening",
    ),

    # Downstream objective/sample -----------------------------------------
    SystemErrorSpec(
        "OBJ-ALIGN", "objective_sample", "objective", "objective decentre/tilt/despace",
        "vector Debye/objective-pupil model or optical prescription/ray-wave model",
        "focal displacement, coma/astigmatism and changed longitudinal field",
        "Phase2C vector reference plus objective tolerance benchmark",
        "objective/relay alignment and calibration", "separate_downstream_branch",
    ),
    SystemErrorSpec(
        "SAMPLE-TILT", "objective_sample", "sample_interface", "sample/interface tilt",
        "vector angular-spectrum/Fresnel transmission through a tilted dielectric interface",
        "aberrated focal structure and lateral/axial displacement in material",
        "tilted-interface Bessel-beam literature and vector transversality/energy checks",
        "sample tilt and material index", "separate_downstream_branch",
    ),
    SystemErrorSpec(
        "LASER-NOISE", "laser", "source", "pulse-energy / wavelength / phase noise",
        "ensemble propagation with measured distributions; nonlinear material response only after calibration",
        "intensity/fluence variability; wavelength shifts kr and filter/order geometry",
        "measured laser statistics and Monte-Carlo convergence",
        "PHAROS pulse-energy/wavelength/time-series measurements", "calibration_required",
    ),
)


def system_error_rows() -> list[dict[str, object]]:
    return [asdict(spec) for spec in SYSTEM_ERRORS]


def report_ready_errors() -> tuple[str, ...]:
    return tuple(spec.error_id for spec in SYSTEM_ERRORS if spec.report_ready)


def pending_errors() -> tuple[str, ...]:
    return tuple(spec.error_id for spec in SYSTEM_ERRORS if not spec.report_ready)
