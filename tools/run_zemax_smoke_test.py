from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from vbb_study.integrations.zemax.connection import ZemaxSession
from vbb_study.integrations.zemax.models import PopRequest
from vbb_study.integrations.zemax.pop import run_pop

OUTPUT = Path("outputs/validation/phase3_zemax/smoke")


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    request = PopRequest(model_path=None, wavelength_index=1, field_index=1, start_surface=1, end_surface="Image", x_sampling=64, y_sampling=64, x_width_m=5e-3, y_width_m=5e-3, beam_type="GaussianWaist", beam_parameters_m={"Waist X": 0.5e-3, "Waist Y": 0.5e-3, "Decenter X": 0.0, "Decenter Y": 0.0}, use_total_power=True, total_power=1.0, use_peak_irradiance=False)
    with ZemaxSession(mode="standalone") as session:
        oss = session.system
        oss.new(saveifneeded=False)
        oss.make_sequential()
        oss.SystemData.Wavelengths.GetWavelength(1).Wavelength = 1.030
        stop = oss.LDE.GetSurfaceAt(1)
        stop.Thickness = 50.0
        result = run_pop(oss, request, system_length_unit_to_m=1e-3)
    finite = bool(np.all(np.isfinite(result.irradiance))); nonnegative = bool(np.all(result.irradiance >= 0.0)); nonzero = bool(float(np.sum(result.irradiance)) > 0.0); shape_ok = result.irradiance.shape == (result.y_m.size, result.x_m.size); passed = finite and nonnegative and nonzero and shape_ok
    np.savez_compressed(OUTPUT / "smoke_pop_result.npz", x_m=result.x_m, y_m=result.y_m, irradiance=result.irradiance)
    payload = {"label": "software_integration_smoke_test", "physical_validation": False, "experimental_validation": False, "passed": passed, "checks": {"finite": finite, "nonnegative": nonnegative, "nonzero": nonzero, "shape_ok": shape_ok}, "request": asdict(request), "metadata": result.metadata, "warnings": result.warnings}
    (OUTPUT / "smoke_test.json").write_text(json.dumps(payload, indent=2), encoding="utf-8"); print(json.dumps(payload, indent=2)); return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
