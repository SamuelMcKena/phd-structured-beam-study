"""Governance tests for the v2 vortex report freeze.

These tests read frozen metadata and accepted evidence from disk. They never execute an optical
solver and never regenerate Phase 2A/2B/2C/2E numerical outputs or figures.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "outputs" / "validation" / "report_freeze_v2"
HISTORICAL_FREEZE = ROOT / "outputs" / "validation" / "report_freeze"
PHASE2E_VALIDATION = ROOT / "outputs" / "validation" / "phase2e_final_propagation"
PHASE2E_FIGURES = ROOT / "outputs" / "figures" / "phase2e_final_source_propagation"

GITHUB_SIZE_LIMIT_BYTES = 90 * 1024 * 1024
EXPECTED_PHASE2E_FIGURE_PAIRS = 18

# Pinned identity of the immutable v1 freeze. Its own manifest calls itself "vortex_report_first";
# the v2 governance designation for it is "historical_vortex_report_freeze_v1". Both must hold.
HISTORICAL_FREEZE_INTERNAL_NAME = "vortex_report_first"
HISTORICAL_FREEZE_GOVERNANCE_NAME = "historical_vortex_report_freeze_v1"
HISTORICAL_FREEZE_FILES = {
    "path_reference_audit.csv",
    "repository_inventory.csv",
    "vortex_report_claims.csv",
    "vortex_report_files.csv",
    "vortex_report_manifest.json",
}

CLAIM_COLUMNS = [
    "claim_id",
    "claim_text",
    "beam_case",
    "solver",
    "mapping_mode",
    "maturity",
    "status",
    "quantitative_valid",
    "calibration_required",
    "source_code_path",
    "data_path",
    "figure_path",
    "test_path",
    "notes",
]

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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _split_paths(value: str) -> list[str]:
    return [part.strip().replace("\\", "/") for part in str(value).split(";") if part.strip()]


@pytest.fixture(scope="module")
def manifest() -> dict:
    return _read_json(FREEZE / "vortex_report_manifest.json")


@pytest.fixture(scope="module")
def evidence() -> list[dict[str, str]]:
    return _read_csv(FREEZE / "vortex_report_files.csv")


@pytest.fixture(scope="module")
def claims() -> list[dict[str, str]]:
    return _read_csv(FREEZE / "vortex_report_claims.csv")


# 1. Old report_freeze remains present and unchanged.
def test_01_historical_v1_freeze_is_present_and_unchanged() -> None:
    assert HISTORICAL_FREEZE.is_dir(), "historical v1 freeze directory was removed"
    present = {path.name for path in HISTORICAL_FREEZE.iterdir() if path.is_file()}
    assert HISTORICAL_FREEZE_FILES <= present, "historical v1 freeze lost files"

    historical = _read_json(HISTORICAL_FREEZE / "vortex_report_manifest.json")
    assert historical["freeze_name"] == HISTORICAL_FREEZE_INTERNAL_NAME

    # The v1 manifest must still describe its own frozen copies bit for bit.
    assert _sha256(ROOT / historical["claims_path"]) == historical["claims_sha256"]
    assert _sha256(ROOT / historical["files_index_path"]) == historical["files_index_sha256"]
    assert historical["selected_evidence_count"] == len(
        _read_csv(HISTORICAL_FREEZE / "vortex_report_files.csv")
    )

    # And git must report no working-tree modification to the historical freeze.
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", "outputs/validation/report_freeze/"],
        cwd=ROOT, text=True, capture_output=True, check=True,
    ).stdout.strip()
    assert status == "", f"historical v1 freeze was modified in place:\n{status}"


# 2. New report_freeze_v2 exists.
def test_02_v2_freeze_exists_with_expected_identity(manifest: dict) -> None:
    assert FREEZE.is_dir()
    for name in (
        "vortex_report_manifest.json",
        "vortex_report_files.csv",
        "vortex_report_claims.csv",
        "phase2e_figure_index.csv",
        "repository_inventory.csv",
        "path_reference_audit.csv",
    ):
        assert (FREEZE / name).is_file(), f"missing v2 freeze artifact {name}"
    assert manifest["freeze_name"] == "vortex_bessel_report_final_v2_governance"
    assert manifest["supersedes"] == HISTORICAL_FREEZE_GOVERNANCE_NAME
    assert manifest["governance"]["historical_freeze_v1_name"] == HISTORICAL_FREEZE_GOVERNANCE_NAME
    assert manifest["governance"]["historical_freeze_v1_immutable"] is True
    assert FREEZE != HISTORICAL_FREEZE


# 3. Every v2 claim evidence path exists.
def test_03_every_v2_claim_evidence_path_exists(claims: list[dict[str, str]]) -> None:
    with (FREEZE / "vortex_report_claims.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        assert csv.DictReader(handle).fieldnames == CLAIM_COLUMNS
    ids = [row["claim_id"] for row in claims]
    assert ids and len(ids) == len(set(ids))
    for row in claims:
        for column in ("source_code_path", "data_path", "figure_path", "test_path"):
            for value in _split_paths(row[column]):
                assert (ROOT / value).is_file(), f"{row['claim_id']}: missing {column} {value}"


# 4. Every selected v2 evidence SHA matches disk.
def test_04_selected_evidence_hashes_match_disk(evidence: list[dict[str, str]], manifest: dict) -> None:
    assert evidence
    for row in evidence:
        path = ROOT / row["path"]
        assert path.is_file(), f"selected evidence missing: {row['path']}"
        assert _sha256(path) == row["sha256"], f"hash drift: {row['path']}"
    assert manifest["selected_evidence_count"] == len(evidence)
    assert _sha256(ROOT / manifest["files_index_path"]) == manifest["files_index_sha256"]
    assert _sha256(ROOT / manifest["claims_path"]) == manifest["claims_sha256"]


# 5. Final Phase 2E evidence is selected.
def test_05_final_phase2e_evidence_is_selected(evidence: list[dict[str, str]]) -> None:
    selected = {row["path"] for row in evidence}
    required = {
        "docs/95_phase2e_final_source_scale_bessel_propagation.md",
        "docs/95_phase2e_propagation_forensic_repair.md",
        "vbb_study/digital_twin/phase2e_final_source_propagation.py",
        "vbb_study/digital_twin/phase2e_final_source_metrics.py",
        "vbb_study/digital_twin/phase2e_final_source_figures.py",
        "vbb_study/digital_twin/phase2e_final_figure_style.py",
        "tools/run_phase2e_final_source_propagation.py",
        "tools/build_phase2e_final_figure_pack.py",
        "tests/test_phase2e_source_sampling_repair.py",
        "tests/test_phase2e_final_source_propagation.py",
        "outputs/figures/phase2e_final_source_propagation/00_manifest/final_figure_manifest.json",
        "outputs/figures/phase2e_final_source_propagation/00_manifest/final_artifact_manifest.json",
        "outputs/figures/phase2e_final_source_propagation/00_manifest/final_figure_style.json",
    }
    required |= {
        f"outputs/validation/phase2e_final_propagation/{name}"
        for name in (
            "final_resolution_gate.csv", "final_resolution_gate.json",
            "production_backend_validation.csv", "production_backend_validation.json",
            "z_step_convergence.csv", "z_step_convergence.json",
            "source_scale_route_contract.json", "final_case_summary.csv",
            "final_zone_summary.csv", "final_aperture_comparison.csv",
            "final_sampling_convergence.csv", "final_claim_impact_registry.csv",
            "final_outcome_report.json", "upstream_hash_status.json",
        )
    }
    assert required <= selected, f"missing Phase 2E evidence: {sorted(required - selected)}"

    # Excluded material must never appear in the freeze.
    forbidden = [
        path for path in selected
        if path.endswith(".log")
        or path.startswith("patches/")
        or path.endswith((".patch", ".diff"))
        or "__pycache__" in path
        or "/smoke_cache/" in path
        or path.startswith("outputs/validation/phase2d/synthetic_bundles/")
    ]
    assert forbidden == [], f"forbidden material selected: {forbidden}"


# 6. All 18 Phase 2E figure IDs are represented.
def test_06_all_phase2e_figure_ids_are_represented(manifest: dict) -> None:
    source = _read_json(PHASE2E_FIGURES / "00_manifest" / "final_figure_manifest.json")
    source_ids = [row["figure_id"] for row in source]
    assert len(source_ids) == EXPECTED_PHASE2E_FIGURE_PAIRS
    assert len(set(source_ids)) == EXPECTED_PHASE2E_FIGURE_PAIRS

    frozen = _read_csv(FREEZE / "phase2e_figure_index.csv")
    assert {row["figure_id"] for row in frozen} == set(source_ids)
    assert manifest["phase2e_figure_pair_count"] == EXPECTED_PHASE2E_FIGURE_PAIRS
    assert set(manifest["phase2e_figure_ids"]) == set(source_ids)


# 7. Final Phase 2E figure artifact paths exist.
def test_07_phase2e_figure_artifact_paths_exist(evidence: list[dict[str, str]]) -> None:
    source = _read_json(PHASE2E_FIGURES / "00_manifest" / "final_figure_manifest.json")
    selected = {row["path"] for row in evidence}
    expected: set[str] = set()
    for row in source:
        for key in ("png_path", "pdf_path"):
            value = str(row[key]).replace("\\", "/")
            assert (ROOT / value).is_file(), f"missing figure artifact {value}"
            expected.add(value)
    assert len(expected) == 2 * EXPECTED_PHASE2E_FIGURE_PAIRS
    assert expected <= selected, f"unselected figure artifacts: {sorted(expected - selected)}"

    figure_files = [
        row["path"] for row in evidence
        if row["path"].startswith("outputs/figures/phase2e_final_source_propagation/")
        and Path(row["path"]).suffix.lower() in {".png", ".pdf"}
    ]
    assert len(figure_files) == 2 * EXPECTED_PHASE2E_FIGURE_PAIRS


# 8. Phase 2E final outcome is PHASE2E-FINAL-A.
def test_08_phase2e_final_outcome_is_recorded(manifest: dict) -> None:
    outcome = _read_json(PHASE2E_VALIDATION / "final_outcome_report.json")
    assert outcome["outcome"] == "PHASE2E-FINAL-A"
    assert manifest["physics_outcome"] == "PHASE2E-FINAL-A"
    assert manifest["phase2e_state"]["physics_outcome"] == "PHASE2E-FINAL-A"


# 9. report_figures_authorised is true.
def test_09_report_figures_are_authorised(manifest: dict) -> None:
    outcome = _read_json(PHASE2E_VALIDATION / "final_outcome_report.json")
    assert outcome["report_figures_authorised"] is True
    assert manifest["phase2e_report_figures_authorised"] is True


# 10. Production grid is N=3072.
def test_10_production_grid_and_z_step_are_pinned(manifest: dict) -> None:
    assert manifest["source_scale_production_grid_n"] == 3072
    assert manifest["source_scale_z_step_m"] == 0.00025
    cases = _read_csv(PHASE2E_VALIDATION / "final_case_summary.csv")
    assert {int(row["grid_n"]) for row in cases} == {3072}
    assert {float(row["z_step_m"]) for row in cases} == {0.00025}


# 11. Source-scale axial claims no longer use Phase 2B as final quantitative evidence.
def test_11_source_scale_axial_claims_do_not_rest_on_phase2b(
    claims: list[dict[str, str]], manifest: dict
) -> None:
    by_id = {row["claim_id"]: row for row in claims}
    axial_ids = manifest["governance"]["source_scale_final_axial_claim_ids"]
    assert axial_ids, "no final source-scale axial claims were declared"
    for claim_id in axial_ids:
        row = by_id[claim_id]
        cited = _split_paths(row["data_path"]) + _split_paths(row["figure_path"])
        assert any("phase2e_final" in item for item in cited), f"{claim_id} lacks Phase 2E evidence"
        assert not any("phase2b" in item for item in cited), (
            f"{claim_id} still cites Phase 2B as final source-scale axial evidence"
        )
    assert manifest["governance"]["source_scale_final_axial_evidence_phase"] == "phase2e"
    # Phase 2B is narrowed for this role, not globally invalidated.
    assert manifest["governance"]["phase2b_globally_invalid"] is False
    assert manifest["governance"]["phase2b_retains_other_visualisation_contracts"] is True


# 12. Phase 2C remains authoritative for vector focal claims.
def test_12_phase2c_remains_authoritative_for_vector_focal_claims(
    claims: list[dict[str, str]], manifest: dict
) -> None:
    governance = manifest["governance"]
    assert governance["phase2c_vector_focal_authoritative"] is True
    by_id = {row["claim_id"]: row for row in claims}
    for claim_id in governance["phase2c_vector_focal_claim_ids"]:
        assert claim_id in by_id, f"Phase 2C focal claim {claim_id} was removed"
        assert "phase2c" in by_id[claim_id]["data_path"], (
            f"Phase 2C focal claim {claim_id} lost its Phase 2C evidence"
        )
    for topic in ("vector Debye focal results", "longitudinal field", "vector Fresnel interface"):
        assert topic in governance["phase2c_authoritative_for"]


# 13. Configured 20-60 mm interval is explicitly not a measured zone.
def test_13_configured_interval_is_not_a_measured_zone(
    claims: list[dict[str, str]], manifest: dict
) -> None:
    governance = manifest["governance"]
    assert governance["configured_interval_m"] == [0.02, 0.06]
    assert governance["configured_interval_is_measured_zone"] is False
    assert governance["configured_interval_is_final_source_scale_prediction"] is False

    rows = _read_csv(PHASE2E_VALIDATION / "final_zone_summary.csv")
    configured = [row for row in rows if row["zone_definition"] == "configured_nominal_interval"]
    assert configured
    assert all(row["is_measured"] == "False" for row in configured)
    assert {(float(row["start_m"]), float(row["stop_m"])) for row in configured} == {(0.02, 0.06)}

    c11 = {row["claim_id"]: row for row in claims}["VTX-C11"]
    text = f"{c11['claim_text']} {c11['notes']}".lower()
    assert c11["status"] == "narrowed"
    assert "configuration reference" in text
    assert "not a measured bessel zone" in text
    assert "not the final source-scale axial prediction" in text


# 14. Hard aperture is diagnostic only.
def test_14_hard_aperture_is_diagnostic_only(manifest: dict) -> None:
    governance = manifest["governance"]
    assert governance["hard_aperture_role"] == "diagnostic_only"
    assert governance["hard_aperture_is_nominal_prediction"] is False

    routes = _read_json(PHASE2E_VALIDATION / "source_scale_route_contract.json")
    assert routes["hard_aperture_diagnostic"]["report_eligibility"] == "diagnostic_only"
    assert routes["hard_aperture_diagnostic"]["calibration_required"] is True

    rows = _read_csv(PHASE2E_VALIDATION / "final_aperture_comparison.csv")
    hard = [row for row in rows if row["route_id"] == "hard_aperture_diagnostic"]
    assert hard and all(row["report_eligibility"] == "diagnostic_only" for row in hard)


# 15. Nominal no-additional-aperture route is report primary.
def test_15_nominal_route_is_report_primary(manifest: dict) -> None:
    assert manifest["governance"]["nominal_route_is_report_primary"] is True
    routes = _read_json(PHASE2E_VALIDATION / "source_scale_route_contract.json")
    nominal = routes["nominal_no_additional_aperture"]
    assert nominal["report_eligibility"] == "primary"
    assert nominal["aperture_application_count"] == 0
    assert routes["soft_aperture_sensitivity"]["report_eligibility"] == "sensitivity_only"
    assert manifest["phase2e_state"]["route_report_eligibility"] == {
        "nominal_no_additional_aperture": "primary",
        "soft_aperture_sensitivity": "sensitivity_only",
        "hard_aperture_diagnostic": "diagnostic_only",
    }


# 16. Experimental calibration remains required.
def test_16_experimental_calibration_remains_required(manifest: dict) -> None:
    assert manifest["experimental_calibration_required"] is True
    governance = manifest["governance"]
    blockers = " ".join(governance["experimental_calibration_blockers"]).lower()
    for token in (
        "beam radius", "slm phase lut", "4f iris", "stop/aperture presence",
        "axicon centring", "camera scale", "z-stage", "objective/relay", "energy/transmission",
    ):
        assert token in blockers, f"missing calibration blocker: {token}"

    # A route-level false flag must never be read as experimental calibration.
    meaning = governance["route_calibration_flag_meaning"].lower()
    assert "no additional aperture calibration" in meaning
    denials = [item.lower() for item in governance["route_calibration_flag_does_not_mean"]]
    assert "experimentally calibrated" in denials
    assert "bench validated" in denials
    assert "fluence calibrated" in denials

    scope = (ROOT / "docs" / "reporting" / "REPORT_SCOPE_AND_MATURITY.md").read_text(encoding="utf-8")
    assert "experimental_calibration_required = true" in scope.lower()


# 17. No experimental validation claim is made.
def test_17_no_experimental_validation_claim(manifest: dict, claims: list[dict[str, str]]) -> None:
    assert manifest["experimental_validation"] is False
    assert "no experimental validation" in manifest["scope_boundary"].lower()
    for row in claims:
        assert row["maturity"] != "experimentally_validated_prediction", (
            f"{row['claim_id']} claims experimental validation"
        )
    doc = (ROOT / "docs" / "95_phase2e_final_source_scale_bessel_propagation.md").read_text(encoding="utf-8")
    assert "experimental" in doc.lower()
    assert "PHASE2E-FINAL-A" in doc


# 18. No nonlinear material-modification claim is made.
def test_18_no_nonlinear_material_modification_claim(manifest: dict) -> None:
    assert manifest["nonlinear_material_model"] is False
    assert "material-modification claim" in manifest["scope_boundary"].lower()
    readme = (ROOT / "README.md").read_text(encoding="utf-8").lower()
    assert "material modification are not predicted" in readme
    scope = (ROOT / "docs" / "reporting" / "REPORT_SCOPE_AND_MATURITY.md").read_text(encoding="utf-8").lower()
    assert "no nonlinear response" in scope


# 19. No selected file exceeds GitHub limits.
def test_19_no_selected_file_exceeds_github_limits(
    evidence: list[dict[str, str]], manifest: dict
) -> None:
    oversized = [row["path"] for row in evidence if int(row["size_bytes"]) > GITHUB_SIZE_LIMIT_BYTES]
    assert oversized == [], f"oversized selected evidence: {oversized}"
    for row in evidence:
        assert (ROOT / row["path"]).stat().st_size == int(row["size_bytes"])
    assert manifest["audit"]["files_over_90mb"] == [] or all(
        entry["path"] not in {row["path"] for row in evidence}
        for entry in manifest["audit"]["files_over_90mb"]
    )


# 20. No selected secret signatures are found.
def test_20_no_selected_secret_signatures(evidence: list[dict[str, str]], manifest: dict) -> None:
    assert manifest["audit"]["secret_signature_findings"] == []
    findings: list[str] = []
    for row in evidence:
        path = ROOT / row["path"]
        if path.suffix.lower() not in TEXT_SUFFIXES or path.stat().st_size > 20 * 1024 * 1024:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern_id, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                findings.append(f"{row['path']}:{pattern_id}")
    assert findings == [], f"secret signatures in selected evidence: {findings}"


def test_21_freeze_records_no_solver_execution(manifest: dict) -> None:
    assert manifest["accepted_numerical_results_regenerated"] is False
    assert manifest["optical_solver_executed"] is False
    assert manifest["propagation_run_executed"] is False


def test_22_report_ready_references_are_resolved() -> None:
    broken = [
        row for row in _read_csv(FREEZE / "path_reference_audit.csv")
        if row["scope"] == "report_ready" and row["exists"].lower() == "false"
    ]
    assert broken == [], f"broken report-ready references: {broken}"


def test_23_phase2e_claim_ids_are_present_and_scale_separated(
    claims: list[dict[str, str]], manifest: dict
) -> None:
    by_id = {row["claim_id"]: row for row in claims}
    for claim_id in manifest["governance"]["phase2e_claim_ids"]:
        assert claim_id in by_id, f"missing Phase 2E claim {claim_id}"
    separation = manifest["governance"]["scale_separation"]
    assert separation["source_scale_phase"] == "phase2e"
    assert separation["objective_sample_scale_phase"] == "phase2c"
    assert "tens of mm" in separation["source_scale"]
    assert "micron" in separation["objective_sample_scale"]
