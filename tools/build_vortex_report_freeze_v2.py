"""Build the v2 vortex report-freeze metadata without executing or rewriting optical simulations.

This builder is governance-only. It reads and hashes accepted evidence that already exists on disk.
It never runs an optical solver, never regenerates Phase 2A/2B/2C/2E arrays or figures, and never
modifies the historical v1 freeze under ``outputs/validation/report_freeze/``.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
FREEZE_ROOT = ROOT / "outputs" / "validation" / "report_freeze_v2"
HISTORICAL_FREEZE_ROOT = ROOT / "outputs" / "validation" / "report_freeze"
CLAIM_SOURCE = ROOT / "docs" / "reporting" / "VORTEX_CLAIM_TO_EVIDENCE.csv"

PHASE2E_VALIDATION = ROOT / "outputs" / "validation" / "phase2e_final_propagation"
PHASE2E_FIGURES = ROOT / "outputs" / "figures" / "phase2e_final_source_propagation"
PHASE2E_FIGURE_MANIFEST = PHASE2E_FIGURES / "00_manifest" / "final_figure_manifest.json"

FREEZE_NAME = "vortex_bessel_report_final_v2_governance"
HISTORICAL_FREEZE_NAME = "historical_vortex_report_freeze_v1"
EXPECTED_PHYSICS_OUTCOME = "PHASE2E-FINAL-A"
EXPECTED_SOURCE_SCALE_GRID_N = 3072
EXPECTED_SOURCE_SCALE_Z_STEP_M = 0.00025
EXPECTED_PHASE2E_FIGURE_PAIRS = 18

NOMINAL_ROUTE_ID = "nominal_no_additional_aperture"
SOFT_ROUTE_ID = "soft_aperture_sensitivity"
HARD_ROUTE_ID = "hard_aperture_diagnostic"

CONFIGURED_INTERVAL_M = [0.02, 0.06]

# Final quantitative source-scale axial statements. These must be evidenced by Phase 2E, never by
# Phase 2B axial figures.
SOURCE_SCALE_FINAL_AXIAL_CLAIM_IDS = ("VTX-C26", "VTX-C27", "VTX-C28")

# Phase 2C keeps the objective/sample-scale vector focal contract.
PHASE2C_VECTOR_FOCAL_CLAIM_IDS = (
    "VTX-C03", "VTX-C04", "VTX-C05", "VTX-C06", "VTX-C07", "VTX-C08", "VTX-C09", "VTX-C10",
)

PHASE2E_CLAIM_IDS = tuple(f"VTX-C{index}" for index in range(23, 35))

EXPERIMENTAL_CALIBRATION_BLOCKERS = (
    "beam radius",
    "SLM phase LUT/stroke",
    "exact 4F iris centre/radius",
    "physical stop/aperture presence",
    "axicon centring/geometry",
    "camera scale",
    "z-stage calibration",
    "objective/relay calibration where relevant",
    "energy/transmission calibration for fluence",
)

ROUTE_CALIBRATION_FLAG_MEANING = (
    "calibration_required=false on nominal_no_additional_aperture means only that no additional "
    "aperture calibration is required to define that numerical route. It does not mean "
    "experimentally calibrated, absolute physical scale verified, bench validated, or fluence "
    "calibrated."
)

SCOPE_BOUNDARY = (
    "Numerical and fixed-bench nominal vortex-Bessel evidence only, covering source-scale Phase 2E "
    "propagation and objective/sample-scale Phase 2C focusing as separate scales; no experimental "
    "validation and no nonlinear material-modification claim."
)

CORE_EVIDENCE = (
    ".gitignore",
    "README.md",
    "AGENTS.md",
    "pyproject.toml",
    "requirements-report.txt",
    "docs/00_theory.md",
    "docs/01_conventions.md",
    "docs/04_model_limitations.md",
    "docs/88_phase1_critical_physics_repairs.md",
    "docs/88_phase1_fourier_geometry_impact_audit.md",
    "docs/89_phase1r_regeneration_recovery_and_reconciliation.md",
    "docs/90_phase2a_canonical_lab_realism.md",
    "docs/91_phase2b_visual_diagnostics_and_beam_volume_maps.md",
    "docs/92_phase2c_vectorial_objective_and_interface.md",
    "docs/93_phase2d_experimental_calibration_bridge.md",
    "docs/95_phase2e_final_source_scale_bessel_propagation.md",
    "docs/95_phase2e_propagation_forensic_repair.md",
    "docs/calibration/01_beam_radius_measurement.md",
    "docs/calibration/02_pulse_energy_and_transmission.md",
    "docs/calibration/03_slm_phase_lut_and_stroke.md",
    "docs/calibration/04_fourier_plane_and_iris.md",
    "docs/calibration/05_objective_relay_and_camera_scale.md",
    "docs/calibration/06_axicon_alignment_and_clear_aperture.md",
    "docs/calibration/07_material_index_and_interface_state.md",
    "docs/calibration/08_complete_calibration_acceptance.md",
    "docs/reporting/REPOSITORY_MAP.md",
    "docs/reporting/VORTEX_REPORT_EVIDENCE_INDEX.md",
    "docs/reporting/VORTEX_CLAIM_TO_EVIDENCE.csv",
    "docs/reporting/REPORT_SCOPE_AND_MATURITY.md",
    "docs/reporting/SOFTWARE_AND_REPRODUCIBILITY.md",
    "docs/reporting/VORTEX_FIGURE_AND_TABLE_PLAN.md",
    "tools/build_vortex_report_freeze.py",
    "tools/build_vortex_report_freeze_v2.py",
    "tools/prepare_phase1r_lab_notebooks.py",
    "tools/reconcile_phase1r_artifacts.py",
    "tools/run_phase1r_realistic_window_recovery.py",
    "tools/run_phase2a_canonical.py",
    "tools/run_phase2b_visual_diagnostics.py",
    "tools/run_phase2c_objective_interface.py",
    "tools/run_phase2e_final_source_propagation.py",
    "tools/build_phase2e_final_figure_pack.py",
    "vbb_study/equations/fields.py",
    "vbb_study/equations/scalar_bessel.py",
    "vbb_study/equations/propagation.py",
    "vbb_study/equations/holography.py",
    "vbb_study/equations/metrics.py",
    "vbb_study/equations/vector_debye.py",
    "vbb_study/equations/vector_fresnel_interface.py",
    "vbb_study/vbb_axicon.py",
    "vbb_study/digital_twin/phase2a_contracts.py",
    "vbb_study/digital_twin/phase2a_canonical.py",
    "vbb_study/digital_twin/phase2b_visual_cases.py",
    "vbb_study/digital_twin/phase2b_figures.py",
    "vbb_study/digital_twin/phase2c_objective_interface.py",
    "vbb_study/digital_twin/phase2b_visual_diagnostics.py",
    "vbb_study/digital_twin/phase2c_figures.py",
    "vbb_study/digital_twin/phase2d_canonical_pipeline.py",
    "vbb_study/digital_twin/phase2d_figures.py",
    "vbb_study/digital_twin/phase2e_final_source_propagation.py",
    "vbb_study/digital_twin/phase2e_final_source_metrics.py",
    "vbb_study/digital_twin/phase2e_final_source_figures.py",
    "vbb_study/digital_twin/phase2e_final_figure_style.py",
    "vbb_study/calibration/__init__.py",
    "vbb_study/calibration/io.py",
    "vbb_study/calibration/schema.py",
    "vbb_study/calibration/uncertainty.py",
    "vbb_study/calibration/validation.py",
    "vbb_study/solver_policy.py",
    "tools/run_phase2d_calibration_bridge.py",
    "tools/validate_calibration_bundle.py",
    "tests/test_phase1_critical_physics_repairs.py",
    "tests/test_phase1r_reconciliation.py",
    "tests/test_phase2a_canonical_lab_realism.py",
    "tests/test_phase2b_visual_diagnostics.py",
    "tests/test_phase2c_objective_interface.py",
    "tests/test_phase2d_calibration_bridge.py",
    "tests/test_phase2e_source_sampling_repair.py",
    "tests/test_phase2e_final_source_propagation.py",
    "tests/test_vortex_report_freeze.py",
    "tests/test_vortex_report_freeze_v2.py",
    "tests/test_slm2_preserve_vortex.py",
    "tests/test_slm2_preserve_vortex_end_to_end.py",
    "outputs/validation/phase1_critical_repairs/phase1_repair_summary.json",
    "outputs/validation/phase1_critical_repairs/phase1_claim_status.csv",
    "outputs/validation/phase1_reconciliation/phase1r_regeneration_manifest.json",
    "outputs/validation/phase1_reconciliation/phase1r_final_claim_registry.csv",
    "outputs/validation/phase1_reconciliation/phase1r_selected_convergence_results.json",
    "outputs/validation/phase1_reconciliation/phase1r_convergence_runs.csv",
    "outputs/validation/phase2a/canonical_hardware_manifest.json",
    "outputs/validation/phase2a/canonical_case_summary.csv",
    "outputs/validation/phase2a/canonical_power_ledgers.csv",
    "outputs/validation/phase2a/slm_model_comparison.csv",
    "outputs/validation/phase2a/phase2a_claim_registry.csv",
    "outputs/validation/phase2a/phase2a_outcome_report.json",
    "outputs/figures/phase2b_visual_diagnostics/00_manifests/figure_provenance.csv",
    "outputs/figures/phase2b_visual_diagnostics/00_manifests/figure_provenance.json",
    "outputs/figures/phase2b_visual_diagnostics/00_manifests/phase2b_final_manifest.json",
    "outputs/figures/phase2b_visual_diagnostics/08_summary_tables/phase2b_case_summary.csv",
    "outputs/figures/phase2b_visual_diagnostics/08_summary_tables/phase2b_endpoint_reproduction_audit.csv",
    "outputs/figures/phase2b_visual_diagnostics/09_final_reports/phase2b_outcome_report.json",
    "outputs/validation/phase2c/phase2c_objective_benchmark.csv",
    "outputs/validation/phase2c/phase2c_case_summary.csv",
    "outputs/validation/phase2c/phase2c_interface_benchmark.csv",
    "outputs/validation/phase2c/phase2c_quadrature_convergence.csv",
    "outputs/validation/phase2c/phase2c_solver_validation.json",
    "outputs/validation/phase2c/phase2c_claim_registry.csv",
    "outputs/validation/phase2c/phase2c_outcome_report.json",
    "outputs/validation/phase2c/phase2c_figure_manifest.json",
    "outputs/validation/phase2d/calibration_readiness.json",
    "outputs/validation/phase2d/calibration_dependency_graph.csv",
    "outputs/validation/phase2d/canonical_case_summary.csv",
    "outputs/validation/phase2d/claim_maturity_registry.csv",
    "outputs/validation/phase2d/phase2d_claim_registry.csv",
    "outputs/validation/phase2d/phase2d_manifest.json",
    "outputs/validation/phase2d/phase2d_outcome_report.json",
    "outputs/validation/phase2d/solver_claim_policy.csv",
    "outputs/validation/phase2d/uncertainty_validation.csv",
    "outputs/validation/phase2e_final_propagation/final_resolution_gate.csv",
    "outputs/validation/phase2e_final_propagation/final_resolution_gate.json",
    "outputs/validation/phase2e_final_propagation/production_backend_validation.csv",
    "outputs/validation/phase2e_final_propagation/production_backend_validation.json",
    "outputs/validation/phase2e_final_propagation/z_step_convergence.csv",
    "outputs/validation/phase2e_final_propagation/z_step_convergence.json",
    "outputs/validation/phase2e_final_propagation/source_scale_route_contract.json",
    "outputs/validation/phase2e_final_propagation/final_case_summary.csv",
    "outputs/validation/phase2e_final_propagation/final_zone_summary.csv",
    "outputs/validation/phase2e_final_propagation/final_aperture_comparison.csv",
    "outputs/validation/phase2e_final_propagation/final_sampling_convergence.csv",
    "outputs/validation/phase2e_final_propagation/final_claim_impact_registry.csv",
    "outputs/validation/phase2e_final_propagation/final_outcome_report.json",
    "outputs/validation/phase2e_final_propagation/upstream_hash_status.json",
    "outputs/figures/phase2e_final_source_propagation/00_manifest/final_figure_manifest.json",
    "outputs/figures/phase2e_final_source_propagation/00_manifest/final_artifact_manifest.json",
    "outputs/figures/phase2e_final_source_propagation/00_manifest/final_figure_style.json",
    "calibration/templates/beam_radius_measurement_template.csv",
    "calibration/templates/camera_scale_template.csv",
    "calibration/templates/canonical_lab_calibration_template.json",
    "calibration/templates/energy_transmission_template.csv",
    "calibration/templates/material_interface_template.csv",
    "calibration/templates/objective_relay_template.csv",
    "calibration/templates/slm_phase_lut_template.csv",
)

TEST_COMMANDS = (
    "python -m pytest tests/test_phase1_critical_physics_repairs.py -q",
    "python -m pytest tests/test_phase1r_reconciliation.py -q",
    "python -m pytest tests/test_phase2a_canonical_lab_realism.py -q",
    "python -m pytest tests/test_phase2b_visual_diagnostics.py -q",
    "python -m pytest tests/test_phase2c_objective_interface.py -q",
    "python -m pytest tests/test_phase2e_source_sampling_repair.py -q",
    "python -m pytest tests/test_phase2e_final_source_propagation.py -q",
    "python -m pytest tests/test_slm2_preserve_vortex.py tests/test_slm2_preserve_vortex_end_to_end.py -q",
    "python -m pytest tests/test_vortex_report_freeze_v2.py -q",
    "python -m pytest --collect-only tests -q",
    "python -m compileall -q vbb_study tools tests",
    "git diff --check",
)

# Selections matching any of these are forbidden in the v2 freeze.
FORBIDDEN_SELECTION_PATTERNS = (
    ("phase2e_build_log", re.compile(r"^outputs/.*phase2e.*\.(?:stdout|stderr)\.log$")),
    ("build_log", re.compile(r"\.log$")),
    ("patch", re.compile(r"^patches/|\.(?:patch|diff)$")),
    ("temporary_or_cache", re.compile(r"(?:^|/)(?:__pycache__|\.tmp[^/]*|\.pytest_cache)(?:/|$)|\.pyc$")),
    ("phase2d_synthetic_bundle", re.compile(r"^outputs/validation/phase2d/synthetic_bundles/")),
    ("smoke_cache", re.compile(r"(?:^|/)smoke_cache/")),
)

SECRET_PATTERNS = {
    "aws_access_key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "github_token": re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"),
    "openai_style_key": re.compile(r"sk-[A-Za-z0-9]{20,}"),
    "private_key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "assigned_secret": re.compile(
        r"(?i)(api[_-]?key|password|auth[_-]?token|access[_-]?token|client[_-]?secret)"
        r"\s*[:=]\s*[\"'][^\"']{8,}[\"']"
    ),
}

TEXT_SUFFIXES = {
    ".py", ".md", ".txt", ".json", ".jsonl", ".csv", ".toml", ".yml", ".yaml",
    ".tex", ".bib", ".ipynb", ".ps1", ".sh", ".ini", ".cfg",
}

GITHUB_SIZE_LIMIT_BYTES = 90 * 1024 * 1024


def _run(*args: str) -> str:
    completed = subprocess.run(
        args,
        cwd=ROOT,
        check=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    return completed.stdout.strip()


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: list[str]) -> Path:
    materialised = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialised)
    return path


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return path


def _split_paths(value: str) -> list[str]:
    return [part.strip().replace("\\", "/") for part in str(value).split(";") if part.strip()]


def _classify(path: str, size: int) -> str:
    lower = path.lower()
    parts = lower.split("/")
    suffix = Path(path).suffix.lower()
    if any(part.startswith((".tmp", ".pytest")) for part in parts) or "__pycache__" in parts or suffix == ".pyc" or "vbb_study.egg-info" in lower:
        return "temporary/cache"
    if lower.startswith("vbb_study - copy/") or ("/" not in path and suffix == ".ipynb"):
        return "duplicate legacy notebook"
    if lower.startswith("patches/") or suffix in {".patch", ".diff"}:
        return "patch"
    if suffix == ".log":
        return "build log"
    if suffix == ".zip" or size > GITHUB_SIZE_LIMIT_BYTES:
        return "large binary"
    if lower.startswith("archive/") or "superseded" in lower or "old_" in lower:
        return "historical or superseded diagnostics"
    if lower.startswith("outputs/figures/phase2e_final_source_propagation/"):
        return "phase2e final source-scale figures"
    if lower.startswith("outputs/validation/phase2e_final_propagation/"):
        return "phase2e final source-scale governed outputs"
    if lower.startswith("calibration/templates/"):
        return "calibration templates"
    if lower.startswith("tests/"):
        return "tests"
    if lower.startswith("tools/") or ("/" not in path and lower.startswith("run_")):
        return "tools"
    if lower.startswith("vbb_study/") or suffix == ".py" or path in {"pyproject.toml", "requirements.txt", "requirements-report.txt"}:
        return "source code"
    if lower.startswith("outputs/validation/") or lower.startswith("baselines"):
        return "governed numerical outputs"
    if lower.startswith("report/evidence_pack/") or lower.startswith("docs/reporting/"):
        return "report evidence"
    if suffix in {".png", ".jpg", ".jpeg", ".svg", ".pdf", ".html"} or lower.startswith("outputs/figures/"):
        return "figures"
    if lower.startswith("outputs/") and suffix in {".csv", ".json", ".jsonl", ".npy", ".npz"}:
        return "governed numerical outputs"
    if lower.startswith("docs/") or lower.startswith("report/") or suffix in {".md", ".tex", ".bib"}:
        return "reports and documentation"
    return "unknown"


def _github_action(category: str, path: str, size: int, selected: set[str]) -> str:
    if path in selected:
        return "selected_report_evidence"
    if category in {"temporary/cache", "duplicate legacy notebook", "large binary", "build log"}:
        return "exclude_from_standalone_git"
    if size >= GITHUB_SIZE_LIMIT_BYTES:
        return "exclude_from_standalone_git"
    if category in {"historical or superseded diagnostics", "patch", "unknown"}:
        return "retain_or_review_not_report_evidence"
    return "retain_if_part_of_curated_export"


def _phase2e_figure_files() -> tuple[list[str], list[dict[str, str]]]:
    """Return every PNG/PDF path referenced by the accepted Phase 2E figure manifest."""
    manifest = _read_json(PHASE2E_FIGURE_MANIFEST)
    paths: list[str] = []
    index: list[dict[str, str]] = []
    for row in manifest:
        png = str(row["png_path"]).replace("\\", "/")
        pdf = str(row["pdf_path"]).replace("\\", "/")
        paths.extend((png, pdf))
        index.append({
            "figure_id": str(row["figure_id"]),
            "case_id": str(row["case_id"]),
            "route_id": str(row["route_id"]),
            "report_role": str(row["report_role"]),
            "png_path": png,
            "pdf_path": pdf,
            "manifest_sha256": str(row["sha256"]),
        })
    if len(index) != EXPECTED_PHASE2E_FIGURE_PAIRS:
        raise ValueError(
            f"expected {EXPECTED_PHASE2E_FIGURE_PAIRS} Phase 2E figure pairs, found {len(index)}"
        )
    return paths, index


def _selected_evidence(claims: list[dict[str, str]]) -> set[str]:
    selected = set(CORE_EVIDENCE)
    for row in claims:
        for column in ("source_code_path", "data_path", "figure_path", "test_path"):
            selected.update(_split_paths(row.get(column, "")))
    for case in ("g0", "b0", "v1", "v3"):
        for folder, suffix in (
            ("02_xy_planes", "xy_landmarks_and_sas_focus.png"),
            ("03_xz_yz_slices", "xz_yz_and_power.png"),
            ("04_profiles", "native_profiles.png"),
        ):
            selected.add(f"outputs/figures/phase2b_visual_diagnostics/{folder}/{case}_{suffix}")
        for folder, suffix in (
            ("objective", "scalar_vs_debye.png"),
            ("interface", "scalar_vs_vector_fresnel.png"),
            ("profiles", "objective_profiles.png"),
            ("components", "debye_components.png"),
        ):
            selected.add(f"outputs/figures/phase2c/{folder}/{case}_{suffix}")
    for case in ("b0", "v1", "v3"):
        selected.add(f"outputs/figures/phase2b_visual_diagnostics/05_3d_maps/{case}_3d_intensity_surface.png")
    phase2e_figures, _ = _phase2e_figure_files()
    selected.update(phase2e_figures)
    return selected


def _assert_no_forbidden_selection(selected: set[str]) -> None:
    violations = [
        {"path": path, "reason": reason}
        for path in sorted(selected)
        for reason, pattern in FORBIDDEN_SELECTION_PATTERNS
        if pattern.search(path)
    ]
    if violations:
        rendered = "\n".join(f"{item['path']} ({item['reason']})" for item in violations)
        raise ValueError("forbidden material selected into the v2 freeze:\n" + rendered)


def _secret_scan(files: list[Path]) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    for path in files:
        if path.suffix.lower() not in TEXT_SUFFIXES or path.stat().st_size > 20 * 1024 * 1024:
            continue
        if any(part.startswith((".tmp", ".pytest")) for part in path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for pattern_id, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                findings.append({"path": _relative(path), "pattern_id": pattern_id})
    return findings


def _resolve_reference(source: Path, reference: str, kind: str) -> tuple[str, bool, str]:
    ref = reference.strip().strip("<>").split("#", 1)[0].split("?", 1)[0]
    if not ref or re.match(r"^[a-zA-Z]+://", ref) or ref.startswith(("#", "mailto:")):
        return "", True, "external_or_anchor"
    ref = ref.replace("\\", "/")
    if "*" in ref or "{" in ref or "}" in ref or "<" in ref or ">" in ref:
        return ref, True, "pattern_not_resolved"
    if ref.startswith("Publication_Study/"):
        ref = ref[len("Publication_Study/"):]
    first = ref.split("/", 1)[0]
    if first in {"outputs", "docs", "report", "calibration", "vbb_study", "tests", "tools", "notebooks", "archive", "references", "reference_kernels"}:
        target = ROOT / ref
    else:
        target = source.parent / ref
    candidates = [target]
    if kind == "latex_graphic" and not target.suffix:
        candidates.extend(target.with_suffix(suffix) for suffix in (".pdf", ".png", ".jpg", ".jpeg"))
    if kind == "latex_bibliography" and not target.suffix:
        candidates.append(target.with_suffix(".bib"))
    existing = next((candidate for candidate in candidates if candidate.exists()), None)
    resolved = existing or target
    try:
        resolved_text = _relative(resolved.resolve())
    except ValueError:
        resolved_text = str(resolved.resolve())
    return resolved_text, existing is not None, "resolved" if existing is not None else "missing"


def _reference_audit(files: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    md_link = re.compile(r"\]\(([^)]+)\)")
    inline_path = re.compile(
        r"`((?:Publication_Study/)?(?:outputs|docs|report|calibration|vbb_study|tests|tools)/[^`\n]+)`"
    )
    latex_patterns = (
        ("latex_graphic", re.compile(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}")),
        ("latex_input", re.compile(r"\\(?:input|include)\{([^}]+)\}")),
        ("latex_bibliography", re.compile(r"\\bibliography\{([^}]+)\}")),
    )
    for source in files:
        if source.suffix.lower() not in {".md", ".tex"} or source.stat().st_size > 10 * 1024 * 1024:
            continue
        text = source.read_text(encoding="utf-8", errors="ignore")
        found: list[tuple[str, str]] = [("markdown_link", value) for value in md_link.findall(text)]
        found.extend(("inline_repository_path", value) for value in inline_path.findall(text))
        for kind, pattern in latex_patterns:
            found.extend((kind, value) for value in pattern.findall(text))
        seen: set[tuple[str, str]] = set()
        for kind, reference in found:
            key = (kind, reference)
            if key in seen:
                continue
            seen.add(key)
            resolved, exists, notes = _resolve_reference(source, reference, kind)
            rows.append({
                "source_path": _relative(source),
                "reference_kind": kind,
                "reference": reference,
                "resolved_path": resolved,
                "exists": exists,
                "scope": "report_ready" if _relative(source).startswith("docs/reporting/") or source.name in {"README.md", "AGENTS.md"} else "historical_or_other",
                "notes": notes,
            })
    return rows


def _package_versions() -> dict[str, str | None]:
    names = (
        "numpy", "scipy", "pandas", "matplotlib", "plotly", "nbformat", "pillow", "ipython",
        "pytest", "papermill", "ipykernel", "jupyter", "colorcet", "cmocean",
    )
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _phase2e_state() -> dict[str, Any]:
    """Read the accepted Phase 2E outcome and verify it against the declared v2 contract."""
    outcome = _read_json(PHASE2E_VALIDATION / "final_outcome_report.json")
    routes = _read_json(PHASE2E_VALIDATION / "source_scale_route_contract.json")
    zones = _read_csv(PHASE2E_VALIDATION / "final_zone_summary.csv")
    cases = _read_csv(PHASE2E_VALIDATION / "final_case_summary.csv")
    apertures = _read_csv(PHASE2E_VALIDATION / "final_aperture_comparison.csv")
    z_gate = _read_json(PHASE2E_VALIDATION / "z_step_convergence.json")
    resolution = _read_json(PHASE2E_VALIDATION / "final_resolution_gate.json")

    if outcome["outcome"] != EXPECTED_PHYSICS_OUTCOME:
        raise ValueError(f"unexpected Phase 2E outcome {outcome['outcome']!r}")
    if outcome["report_figures_authorised"] is not True:
        raise ValueError("Phase 2E report figures are not authorised")
    if int(outcome["selected_production_grid_n"]) != EXPECTED_SOURCE_SCALE_GRID_N:
        raise ValueError("unexpected Phase 2E production grid")
    if float(outcome["selected_z_step_m"]) != EXPECTED_SOURCE_SCALE_Z_STEP_M:
        raise ValueError("unexpected Phase 2E z step")
    if int(resolution["selected_production_grid_n"]) != EXPECTED_SOURCE_SCALE_GRID_N:
        raise ValueError("resolution gate disagrees with the selected production grid")
    if z_gate["status"] != "passed":
        raise ValueError("Phase 2E z-step convergence did not pass")

    grid_values = {int(row["grid_n"]) for row in cases}
    if grid_values != {EXPECTED_SOURCE_SCALE_GRID_N}:
        raise ValueError(f"final case summary grids are {sorted(grid_values)}")
    step_values = {float(row["z_step_m"]) for row in cases}
    if step_values != {EXPECTED_SOURCE_SCALE_Z_STEP_M}:
        raise ValueError(f"final case summary z steps are {sorted(step_values)}")

    if routes[NOMINAL_ROUTE_ID]["report_eligibility"] != "primary":
        raise ValueError("nominal route is not report primary")
    if routes[NOMINAL_ROUTE_ID]["aperture_application_count"] != 0:
        raise ValueError("nominal route applies an additional aperture")
    if routes[HARD_ROUTE_ID]["report_eligibility"] != "diagnostic_only":
        raise ValueError("hard aperture route is not diagnostic only")
    if routes[SOFT_ROUTE_ID]["report_eligibility"] != "sensitivity_only":
        raise ValueError("soft aperture route is not sensitivity only")

    configured = [row for row in zones if row["zone_definition"] == "configured_nominal_interval"]
    if not configured or any(row["is_measured"] != "False" for row in configured):
        raise ValueError("configured nominal interval is not marked unmeasured")
    configured_bounds = {(float(row["start_m"]), float(row["stop_m"])) for row in configured}
    if configured_bounds != {(CONFIGURED_INTERVAL_M[0], CONFIGURED_INTERVAL_M[1])}:
        raise ValueError(f"unexpected configured interval bounds {sorted(configured_bounds)}")

    hard_rows = [row for row in apertures if row["route_id"] == HARD_ROUTE_ID]
    if not hard_rows or any(row["report_eligibility"] != "diagnostic_only" for row in hard_rows):
        raise ValueError("hard aperture comparison rows are not diagnostic only")

    def _zone(case: str, definition: str) -> list[float]:
        row = next(r for r in zones if r["case_id"] == case and r["zone_definition"] == definition)
        return [float(row["start_m"]), float(row["stop_m"])]

    measured = {}
    for row in cases:
        case = row["case_id"]
        measured[case] = {
            "measured_FWHM_axial_zone_m": _zone(case, "measured_FWHM_axial_zone"),
            "measured_strict_useful_region_m": _zone(case, "measured_strict_useful_region"),
            "reference_feature_radius_m": float(row["reference_feature_radius_m"]),
            "median_strict_feature_width_m": float(row["median_strict_feature_width_m"]),
            "median_strict_dark_core_radius_m": (
                float(row["median_strict_dark_core_radius_m"])
                if row["median_strict_dark_core_radius_m"] else None
            ),
            "requested_winding": int(row["requested_winding"]),
            "maximum_edge_energy_fraction": float(row["maximum_edge_energy_fraction"]),
            "maximum_propagation_power_drift_fraction": float(row["maximum_propagation_power_drift_fraction"]),
        }

    return {
        "physics_outcome": outcome["outcome"],
        "phase2e_report_figures_authorised": bool(outcome["report_figures_authorised"]),
        "source_scale_production_grid_n": EXPECTED_SOURCE_SCALE_GRID_N,
        "source_scale_z_step_m": EXPECTED_SOURCE_SCALE_Z_STEP_M,
        "source_scale_window_m": 0.01,
        "nominal_route_id": NOMINAL_ROUTE_ID,
        "soft_route_id": SOFT_ROUTE_ID,
        "hard_route_id": HARD_ROUTE_ID,
        "route_report_eligibility": {
            key: routes[key]["report_eligibility"] for key in (NOMINAL_ROUTE_ID, SOFT_ROUTE_ID, HARD_ROUTE_ID)
        },
        "measured_source_scale_results": measured,
        "upstream_hash_status": outcome.get("upstream_hash_status"),
    }


def _governance_block(claims: list[dict[str, str]]) -> dict[str, Any]:
    """Assemble and verify the report-governance statements carried by the v2 freeze."""
    by_id = {row["claim_id"]: row for row in claims}

    missing = [claim_id for claim_id in PHASE2E_CLAIM_IDS if claim_id not in by_id]
    if missing:
        raise ValueError(f"missing Phase 2E claim rows: {missing}")

    # Final source-scale axial statements must rest on Phase 2E, never on Phase 2B axial figures.
    for claim_id in SOURCE_SCALE_FINAL_AXIAL_CLAIM_IDS:
        row = by_id[claim_id]
        evidence = _split_paths(row["data_path"]) + _split_paths(row["figure_path"])
        if not any("phase2e_final" in item for item in evidence):
            raise ValueError(f"{claim_id} does not cite Phase 2E evidence")
        if any("phase2b" in item for item in evidence):
            raise ValueError(f"{claim_id} still cites Phase 2B as source-scale axial evidence")

    # Phase 2C keeps the objective/sample-scale vector focal contract.
    for claim_id in PHASE2C_VECTOR_FOCAL_CLAIM_IDS:
        row = by_id.get(claim_id)
        if row is None:
            raise ValueError(f"Phase 2C focal claim {claim_id} was removed")
        if "phase2c" not in row["data_path"]:
            raise ValueError(f"Phase 2C focal claim {claim_id} lost its Phase 2C evidence")

    c11 = by_id["VTX-C11"]
    c11_text = f"{c11['claim_text']} {c11['notes']}".lower()
    if "configuration reference" not in c11_text:
        raise ValueError("VTX-C11 is not narrowed to a configuration reference")
    if "not a measured bessel zone" not in c11_text:
        raise ValueError("VTX-C11 does not deny the measured-zone reading")
    if c11["status"] != "narrowed":
        raise ValueError("VTX-C11 status is not narrowed")

    return {
        "historical_freeze_v1_path": _relative(HISTORICAL_FREEZE_ROOT),
        "historical_freeze_v1_name": HISTORICAL_FREEZE_NAME,
        "historical_freeze_v1_immutable": True,
        "historical_freeze_v1_note": (
            "Retained unchanged as pre-Phase-2E provenance. Its recorded hashes describe the "
            "governance documents as they stood at v1 and no longer match the current working "
            "copies of those documents."
        ),
        "phase2e_claim_ids": list(PHASE2E_CLAIM_IDS),
        "source_scale_final_axial_claim_ids": list(SOURCE_SCALE_FINAL_AXIAL_CLAIM_IDS),
        "source_scale_final_axial_evidence_phase": "phase2e",
        "phase2b_source_scale_axial_role": (
            "historical earlier accepted visual diagnostics; superseded for final quantitative "
            "source-scale axial detail"
        ),
        "phase2b_globally_invalid": False,
        "phase2b_retains_other_visualisation_contracts": True,
        "phase2c_vector_focal_authoritative": True,
        "phase2c_vector_focal_claim_ids": list(PHASE2C_VECTOR_FOCAL_CLAIM_IDS),
        "phase2c_authoritative_for": [
            "vector Debye focal results",
            "scalar/vector focal comparison",
            "longitudinal field",
            "vector Fresnel interface",
            "quantitative focal peak-location claims",
        ],
        "configured_interval_m": list(CONFIGURED_INTERVAL_M),
        "configured_interval_is_measured_zone": False,
        "configured_interval_is_final_source_scale_prediction": False,
        "configured_interval_role": "historical_configuration_reference_only",
        "hard_aperture_role": "diagnostic_only",
        "hard_aperture_is_nominal_prediction": False,
        "soft_aperture_role": "sensitivity_only",
        "nominal_route_is_report_primary": True,
        "route_calibration_flag_meaning": ROUTE_CALIBRATION_FLAG_MEANING,
        "route_calibration_flag_does_not_mean": [
            "experimentally calibrated",
            "absolute physical scale verified",
            "bench validated",
            "fluence calibrated",
        ],
        "experimental_calibration_blockers": list(EXPERIMENTAL_CALIBRATION_BLOCKERS),
        "scale_separation": {
            "source_scale": "tens of mm axial propagation; tens of um transverse Bessel/ring scale",
            "source_scale_phase": "phase2e",
            "objective_sample_scale": "Phase 2C Debye focal plane; approximately micron transverse scale",
            "objective_sample_scale_phase": "phase2c",
        },
    }


def build_freeze() -> dict[str, Any]:
    if not HISTORICAL_FREEZE_ROOT.is_dir():
        raise FileNotFoundError(
            f"historical v1 freeze is missing at {_relative(HISTORICAL_FREEZE_ROOT)}"
        )

    claims = _read_csv(CLAIM_SOURCE)
    phase2e = _phase2e_state()
    governance = _governance_block(claims)
    _, phase2e_figure_index = _phase2e_figure_files()

    selected = _selected_evidence(claims)
    _assert_no_forbidden_selection(selected)
    missing_selected = sorted(path for path in selected if not (ROOT / path).is_file())
    if missing_selected:
        raise FileNotFoundError("selected report evidence is missing:\n" + "\n".join(missing_selected))

    tracked = set(_run("git", "ls-files", "--", ".").splitlines())
    untracked = set(_run("git", "ls-files", "--others", "--exclude-standard", "--", ".").splitlines())
    ignored = set(_run("git", "ls-files", "--others", "-i", "--exclude-standard", "--", ".").splitlines())

    evidence_rows: list[dict[str, Any]] = []
    for relative in sorted(selected):
        path = ROOT / relative
        size = path.stat().st_size
        category = _classify(relative, size)
        phase2e_selected = relative.startswith((
            "outputs/validation/phase2e_final_propagation/",
            "outputs/figures/phase2e_final_source_propagation/",
        )) or "phase2e_final" in relative or "phase2e_source_sampling" in relative
        evidence_rows.append({
            "path": relative,
            "category": category,
            "role": "vortex_report_evidence",
            "phase2e_final_evidence": phase2e_selected,
            "sha256": _sha256(path),
            "size_bytes": size,
            "git_ignored": relative in ignored,
            "notes": "Referenced in claim matrix or core v2 report-freeze contract.",
        })

    oversized = [row for row in evidence_rows if int(row["size_bytes"]) > GITHUB_SIZE_LIMIT_BYTES]
    if oversized:
        raise ValueError(
            "selected evidence exceeds the GitHub size limit:\n"
            + "\n".join(f"{row['path']} ({row['size_bytes']} bytes)" for row in oversized)
        )

    FREEZE_ROOT.mkdir(parents=True, exist_ok=True)
    files_path = _write_csv(
        FREEZE_ROOT / "vortex_report_files.csv",
        evidence_rows,
        ["path", "category", "role", "phase2e_final_evidence", "sha256", "size_bytes", "git_ignored", "notes"],
    )
    claims_path = _write_csv(
        FREEZE_ROOT / "vortex_report_claims.csv",
        claims,
        list(claims[0]),
    )
    figure_index_path = _write_csv(
        FREEZE_ROOT / "phase2e_figure_index.csv",
        phase2e_figure_index,
        ["figure_id", "case_id", "route_id", "report_role", "png_path", "pdf_path", "manifest_sha256"],
    )

    disk_files = [path for path in ROOT.rglob("*") if path.is_file()]
    inventory_rows: list[dict[str, Any]] = []
    for path in disk_files:
        relative = _relative(path)
        size = path.stat().st_size
        category = _classify(relative, size)
        if relative in tracked:
            git_status = "tracked"
        elif relative in untracked:
            git_status = "untracked"
        elif relative in ignored:
            git_status = "ignored"
        else:
            git_status = "unclassified_worktree_file"
        inventory_rows.append({
            "path": relative,
            "category": category,
            "size_bytes": size,
            "git_status": git_status,
            "github_action": _github_action(category, relative, size, selected),
        })

    inventory_path = _write_csv(
        FREEZE_ROOT / "repository_inventory.csv",
        inventory_rows,
        ["path", "category", "size_bytes", "git_status", "github_action"],
    )
    audited_files = [
        path for path in disk_files
        if _relative(path) in tracked or _relative(path) in untracked
    ]
    reference_rows = _reference_audit(audited_files)
    reference_path = _write_csv(
        FREEZE_ROOT / "path_reference_audit.csv",
        reference_rows,
        ["source_path", "reference_kind", "reference", "resolved_path", "exists", "scope", "notes"],
    )
    secrets = _secret_scan(audited_files)
    large_files = [
        {"path": row["path"], "size_bytes": row["size_bytes"]}
        for row in inventory_rows if int(row["size_bytes"]) > GITHUB_SIZE_LIMIT_BYTES
    ]
    broken = [row for row in reference_rows if not row["exists"]]
    report_broken = [row for row in broken if row["scope"] == "report_ready"]
    category_counts = dict(sorted(Counter(str(row["category"]) for row in inventory_rows).items()))
    status_counts = dict(sorted(Counter(str(row["git_status"]) for row in inventory_rows).items()))
    blockers_path = ROOT / "outputs" / "validation" / "phase2d" / "phase2d_outcome_report.json"
    blockers = []
    if blockers_path.is_file():
        blockers = _read_json(blockers_path).get("important_missing_measurements", [])

    phase2e_rows = [row for row in evidence_rows if row["phase2e_final_evidence"]]
    phase2e_figure_rows = [
        row for row in phase2e_rows
        if row["path"].startswith("outputs/figures/phase2e_final_source_propagation/")
        and Path(row["path"]).suffix.lower() in {".png", ".pdf"}
    ]

    manifest = {
        "schema_version": "2.0.0",
        "freeze_name": FREEZE_NAME,
        "supersedes": HISTORICAL_FREEZE_NAME,
        "hash_algorithm": "SHA-256",
        "repository_path": str(ROOT),
        "parent_git_repository": str(ROOT.parent),
        "current_branch": _run("git", "branch", "--show-current"),
        "current_commit_before_report_preparation": _run("git", "rev-parse", "HEAD"),
        "working_tree_commit_created": False,
        "accepted_numerical_results_regenerated": False,
        "optical_solver_executed": False,
        "propagation_run_executed": False,
        "physics_outcome": phase2e["physics_outcome"],
        "phase2e_report_figures_authorised": phase2e["phase2e_report_figures_authorised"],
        "source_scale_production_grid_n": phase2e["source_scale_production_grid_n"],
        "source_scale_z_step_m": phase2e["source_scale_z_step_m"],
        "experimental_validation": False,
        "experimental_calibration_required": True,
        "nonlinear_material_model": False,
        "python": {
            "version": sys.version,
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "package_versions": _package_versions(),
        "test_commands": list(TEST_COMMANDS),
        "numerical_precision": {
            "phase2a_propagated_intensity_stack": "float32 storage",
            "phase2a_power_and_export_metrics": "Python/NumPy floating point",
            "phase2c_vector_working_arrays": "complex128/float64 unless source states otherwise",
            "phase2e_production_backend": "complex64 BL-ASM validated against complex128",
            "metrics_source": "governed CSV/JSON/native arrays; never display pixels",
        },
        "canonical_hardware_manifest_path": "outputs/validation/phase2a/canonical_hardware_manifest.json",
        "solver_policy_provenance": [
            "vbb_study/solver_policy.py",
            "outputs/validation/phase2c/phase2c_claim_registry.csv",
        ],
        "phase2e_state": phase2e,
        "governance": governance,
        "unresolved_calibration_blockers": blockers,
        "selected_evidence_count": len(evidence_rows),
        "selected_evidence_bytes": sum(int(row["size_bytes"]) for row in evidence_rows),
        "phase2e_selected_evidence_count": len(phase2e_rows),
        "phase2e_selected_figure_file_count": len(phase2e_figure_rows),
        "phase2e_figure_pair_count": len(phase2e_figure_index),
        "phase2e_figure_ids": [row["figure_id"] for row in phase2e_figure_index],
        "selected_evidence": evidence_rows,
        "claims_path": _relative(claims_path),
        "claims_sha256": _sha256(claims_path),
        "files_index_path": _relative(files_path),
        "files_index_sha256": _sha256(files_path),
        "phase2e_figure_index_path": _relative(figure_index_path),
        "phase2e_figure_index_sha256": _sha256(figure_index_path),
        "repository_inventory_path": _relative(inventory_path),
        "path_reference_audit_path": _relative(reference_path),
        "audit": {
            "disk_file_count": len(inventory_rows),
            "disk_bytes": sum(int(row["size_bytes"]) for row in inventory_rows),
            "category_counts": category_counts,
            "git_status_counts": status_counts,
            "files_over_90mb": large_files,
            "secret_signature_findings": secrets,
            "broken_reference_count": len(broken),
            "report_ready_broken_reference_count": len(report_broken),
            "report_ready_broken_references": report_broken,
        },
        "standalone_export": {
            "raw_directory_copy_ready": False,
            "curated_git_export_ready": not secrets and not missing_selected and not report_broken,
            "required_exclusions": [
                "temporary/cache trees",
                "root duplicate notebooks",
                "vbb_study - Copy",
                "bulk ZIP archives",
                "phase2e build logs",
                "patches",
                "phase2d synthetic bundles",
                "files above GitHub size limits",
            ],
            "ignored_selected_figures_require_force_add": any(row["git_ignored"] for row in evidence_rows),
        },
        "scope_boundary": SCOPE_BOUNDARY,
    }
    manifest_path = _write_json(FREEZE_ROOT / "vortex_report_manifest.json", manifest)
    return {
        "manifest": str(manifest_path),
        "freeze_name": FREEZE_NAME,
        "physics_outcome": manifest["physics_outcome"],
        "selected_evidence_count": len(evidence_rows),
        "selected_evidence_bytes": manifest["selected_evidence_bytes"],
        "phase2e_selected_evidence_count": len(phase2e_rows),
        "phase2e_selected_figure_file_count": len(phase2e_figure_rows),
        "inventory_file_count": len(inventory_rows),
        "secret_findings": len(secrets),
        "large_files_over_90mb": len(large_files),
        "broken_references": len(broken),
        "report_ready_broken_references": len(report_broken),
        "curated_git_export_ready": manifest["standalone_export"]["curated_git_export_ready"],
    }


def main() -> int:
    result = build_freeze()
    print(json.dumps(result, indent=2))
    return 0 if result["secret_findings"] == 0 and result["report_ready_broken_references"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
