"""Runner for q20 v10 with an explicitly verified centered resampling adjoint."""
from __future__ import annotations

import numpy as np

import solve_q20_slm2_multiplane_circular_v10 as v10


def centred_fixed_window_resample_adjoint(fine_field: np.ndarray, input_n: int) -> np.ndarray:
    """Exact discrete adjoint of fourier_resample_fixed_window.

    The forward map is a centred FFT, fractional half-sample phase, centred
    spectral zero padding, centred IFFT, and (M/N)^2 scale.  With NumPy's FFT
    normalisation those factors cancel in the adjoint, leaving a centred FFT,
    crop, conjugate half-sample phase, and centred IFFT.
    """
    fine = np.asarray(fine_field, complex)
    output_n = int(fine.shape[0]); input_n = int(input_n)
    if fine.shape != (output_n, output_n) or output_n < input_n:
        raise ValueError("fine field must be square and at least input_n")
    if output_n == input_n:
        return fine.copy()
    spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(fine)))
    start = (output_n - input_n) // 2
    cropped = spectrum[start:start+input_n, start:start+input_n].copy()
    delta_samples = 0.5 * (float(input_n) / output_n - 1.0)
    freq = np.fft.fftshift(np.fft.fftfreq(input_n, d=1.0))
    fy, fx = np.meshgrid(freq, freq, indexing="ij")
    phase = np.exp(1j * v10.TWOPI * delta_samples * (fx + fy))
    cropped *= np.conj(phase)
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(cropped)))


def verify_adjoint() -> None:
    rng = np.random.default_rng(240903)
    n, m = 32, 64
    x = rng.normal(size=(n,n)) + 1j*rng.normal(size=(n,n))
    y = rng.normal(size=(m,m)) + 1j*rng.normal(size=(m,m))
    Ax = v10.fourier_resample_fixed_window(x, m)
    Aty = centred_fixed_window_resample_adjoint(y, n)
    lhs = np.vdot(Ax, y)
    rhs = np.vdot(x, Aty)
    rel = abs(lhs-rhs) / max(abs(lhs), abs(rhs), 1e-12)
    print({"resample_adjoint_relative_inner_product_error": float(rel)})
    if rel > 2e-10:
        raise RuntimeError(f"fixed-window resampling adjoint failed: {rel}")


v10.resample_adjoint = centred_fixed_window_resample_adjoint

if __name__ == "__main__":
    verify_adjoint()
    v10.main()
