"""Build the executable real-BMG q=20 digital-twin comparison notebook."""
from pathlib import Path

import nbformat as nbf


HERE = Path(__file__).resolve().parent
TARGET = HERE / "Bessel_zscan_digital_twin_correction.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


nb = nbf.v4.new_notebook()
nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb["metadata"]["language_info"] = {"name": "python", "version": "3.13"}
nb["cells"] = [
    md("""
# Real q=20 BMG digital-twin correction audit

This notebook uses the actual 18-plane × 4-repeat BeamGage acquisition in one
fixed camera coordinate system. It compares the measured stack, a Miao
input-plane inverse, a complete SLM/4F/axicon screen, and a calibrated
finite-energy q=20 target.

The corrected fields are model predictions, not post-correction measurements.
The nominal route passes its morphology and sampling gates. The inferred
residual gives only a small held-out improvement, so it remains a screening
result and hardware SLM2 export is blocked.
"""),
    code("""
from pathlib import Path
import json, sys
import pandas as pd
from IPython.display import display, Image, Markdown

EXP = None
for base in (Path.cwd(), *Path.cwd().parents):
    candidate = base / "notebooks" / "experimental" / "axicon_aberration_correction"
    if (candidate / "real_bmg_digital_twin_correction.py").exists():
        EXP = candidate.resolve(); break
if EXP is None:
    raise FileNotFoundError("Open this notebook from the Publication_Study checkout")
ROOT = EXP.parents[2]
sys.path[:0] = [str(EXP), str(ROOT)]
import real_bmg_digital_twin_correction as bridge

DATA_DIR = bridge.find_bmg_dir()
groups = bridge.inventory(DATA_DIR)
OUT = EXP / "outputs" / "digital_twin_correction"
print("BMG directory:", DATA_DIR)
print("planes:", len(groups), "frames:", sum(map(len, groups.values())))
print("z relative (mm):", bridge.Z_REL_MM.tolist())
print("z hexapod (mm):", (bridge.Z_REL_MM + 6).tolist())
"""),
    md("""
## Reproduction switch

Set `RUN_PIPELINE=True` to repeat the expensive real-BMG inverse and the
4096-point convergence spot check. It defaults to false so reopening this
executed notebook does not silently repeat the fit.
"""),
    code("""
RUN_PIPELINE = False
if RUN_PIPELINE:
    bridge.run(DATA_DIR, OUT, recompute=True)
required = [OUT/"run_summary.json", OUT/"rerender_arrays.npz",
            OUT/"method_comparison_metrics_vs_z.csv"]
missing = [path for path in required if not path.exists()]
if missing:
    raise FileNotFoundError(f"Set RUN_PIPELINE=True; missing outputs: {missing}")
summary = json.loads((OUT/"run_summary.json").read_text(encoding="utf-8"))
print("cached real-data run loaded:", OUT)
"""),
    md("## 1. Real measured data — fixed camera coordinates"),
    code("""
display(Image(filename=str(OUT/"01_measured_BMG_contact_sheet.png"), width=1300))
display(Image(filename=str(OUT/"02_measured_XZ_YZ_fixed_coordinates.png"), width=1200))
display(pd.DataFrame([summary["trajectory"]]).T.rename(columns={0:"value"}))
display(Image(filename=str(OUT/"05_measured_beam_path.png"), width=750))
"""),
    md("""
The dark-core path has a 5.42 mrad fitted magnitude. Camera-stage runout has
not been independently measured, so the walk is preserved but cannot be
attributed uniquely to the axicon or input beam.
"""),
    md("## 2. Measured radial scale and effective axicon"),
    code("""
display(Image(filename=str(OUT/"measured_k_perp_vs_z.png"), width=1050))
display(Image(filename=str(OUT/"measured_radial_profiles_and_fit.png"), width=1250))
a = summary["effective_axicon"]
pd.DataFrame([{
 "real-stack robust k_perp (1/m)": a["measured_effective_k_perp_m_inv"],
 "Miao median k_perp (1/m)": summary["measured_k_perp_calibration"]["miao_median_m_inv"],
 "effective internal base angle (deg)": a["effective_internal_base_angle_deg"],
 "full cone departure (deg)": a["corresponding_full_cone_departure_deg"],
 "manufacturer convention resolved": a["manufacturer_convention_resolved"],
}])
"""),
    md("""
Direct radial fits to all real planes give 482.74 mm⁻¹ (robust MAD
6.87 mm⁻¹), while the Miao global branch gives 530.17 mm⁻¹. The measured-ring
value maps to a 9.738° internal base angle and 19.475° full cone departure in
this model. Its proximity to the nominal “20°” is an inference, not a
datasheet conversion.
"""),
    md("## 3. Hard nominal-model gate before inversion"),
    code("""
display(Image(filename=str(OUT/"measured_vs_nominal_before_fitting.png"), width=1250))
pd.DataFrame([summary["nominal_morphology_gate"]])
"""),
    md("""
This camera-scale figure is the required pre-inverse gate. The calibrated
finite-energy route produces a dark annular q=20 field, matches the ring scale,
and preserves +20 source winding on the illuminated annulus. The earlier
full-window view showed the preceding conical collapse and was not the
measured-region comparison.
"""),
    md("## 4. Miao and physical-parameter screens"),
    code("""
miao = summary["miao"]
b = summary["miao_single_phase_backcheck"]
display(pd.DataFrame([{
 "Miao path reliable": miao["k_perp_path_reliable"],
 "Miao mean modal fit r": miao["mean_global_fit_correlation"],
 "original Miao error recreates lab": b["supports_retrieved_phase_as_cause_of_measured_distortion"],
 "Miao hardware ready": miao["hardware_ready"],
}]))
p = summary["physical_screening"]
display(pd.DataFrame([
 {"quantity":"4F iris radius scale", **p["4f_iris_radius"]},
 {"quantity":"effective axicon lateral displacement", **p["effective_axicon_lateral_displacement"]},
]))
display(pd.read_csv(OUT/"physical_parameter_objective_scans.csv"))
"""),
    md("""
The iris minimum is broad: 1.0, 1.15 and 1.3 are within 5%, so it is not a
precise hardware measurement. Both tested displacement routes prefer zero;
camera-stage runout remains a confounder. The Miao phase stays on its
reconstructed input/axicon plane and is never relabelled as an SLM2 mask.
"""),
    md("## 5. Held-out residual and inverse-error back-check"),
    code("""
r = summary["full_route_residual"]
display(pd.DataFrame([{
 "train r before": r["train_before"]["mean_pearson_r"],
 "train r after": r["train_after"]["mean_pearson_r"],
 "held-out r before": r["heldout_before"]["mean_pearson_r"],
 "held-out r after": r["heldout_after"]["mean_pearson_r"],
 "held-out RMSE before": r["heldout_before"]["mean_rmse"],
 "held-out RMSE after": r["heldout_after"]["mean_rmse"],
 "held-out improves both": r["heldout_improved_both_metrics"],
}]))
display(pd.DataFrame([summary["full_route_validation"]]))
display(Image(filename=str(OUT/"08_error_reconstruction_backcheck.png"), width=1400))
"""),
    md("""
Held-out planes improve only slightly. Applying the positive fitted error to
the ideal route raises mean real-stack correlation by about 0.024 and reduces
NRMSE by about 0.0036. Neither inferred error visually reproduces the measured
hourglass closely, so this is screening evidence rather than a resolved
aberration map.
"""),
    md("## 6. High-resolution measured / corrected / ideal comparison"),
    code("""
display(Image(filename=str(OUT/"03_measured_miao_digital_twin_target_comparison.png"), width=1550))
display(Image(filename=str(OUT/"04_metrics_vs_z.png"), width=1200))
display(Image(filename=str(OUT/"09_measured_corrected_ideal_profile_comparisons.png"), width=1450))
display(pd.read_csv(OUT/"method_summary_metrics.csv"))
"""),
    md("""
All numerical methods use the same downstream calibrated finite-energy twin.
The Miao-only output can remain close to ideal without proving that its inverse
explains the real error; the preceding positive-error back-check is therefore
the decisive companion figure. Dark-core metrics follow each beam centre and
do not erase the fixed-camera propagation walk.
"""),
    md("## 7. Sampling audit"),
    code("""pd.read_csv(OUT/"sampling_convergence.csv")"""),
    md("""
The relay/source window stays fixed at 10 mm. The final 3072 grid has four
samples per effective axicon phase period and agrees with the independent
4096-point spot check at mean correlation 0.981. The lower-resolution rows
show why the original single-grid result was unreliable.
"""),
    md("## 8. Numerical SLM2 correction layer — not hardware-ready"),
    code("""display(Image(filename=str(OUT/"06_predicted_SLM2_correction_phase.png"), width=1200))"""),
    md("""
The left panel is the correction-only additive numerical SLM2 layer and has no
q=20 vortex. The right panel adds an illustrative 20-pixel carrier. Neither is
a native hardware export: SLM2 conjugacy, native scale/centre/parity, real
carrier, branch sign and the 1030-nm phase LUT remain uncalibrated.
"""),
    md("## 9. Measured and predicted-correction 3D XZ mesh"),
    code("""display(Image(filename=str(OUT/"07_measured_vs_corrected_3D_mesh.png"), width=1200))"""),
    md("## Final report and safety gates"),
    code("""
display(Markdown((OUT/"SUMMARY.md").read_text(encoding="utf-8")))
assert summary["hardware_ready"] is False
assert summary["nominal_morphology_gate"]["nominal_morphology_gate_pass"] is True
assert summary["sampling"]["quantitative_high_angle_claim_allowed"] is True
assert summary["full_route_validation"]["decision"] == "MODEL_SCREENING_SUPPORTED_HARDWARE_BLOCKED"
print("Gates passed: coherent nominal model and converged figures; no hardware-ready SLM phase claimed.")
"""),
]

nbf.write(nb, TARGET)
print(TARGET)
