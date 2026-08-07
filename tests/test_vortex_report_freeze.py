from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "outputs" / "validation" / "report_freeze"

# The v1 freeze is an immutable historical metadata snapshot. Its recorded hashes remain untouched
# and describe the repository as it stood when v1 was taken. Current governance documents may
# legitimately evolve in later report freezes -- the v2 governance refresh rewrote the six paths
# below to record Phase 2E. This test therefore distinguishes authorised governance evolution from
# unexpected evidence drift: the six declared paths may differ from their v1 hashes, and nothing
# else may. Physics source, numerical outputs, figures, calibration templates and tests must still
# match v1 exactly.
HISTORICAL_GOVERNANCE_ALLOWED_TO_EVOLVE = {
    "README.md",
    "docs/reporting/VORTEX_CLAIM_TO_EVIDENCE.csv",
    "docs/reporting/VORTEX_REPORT_EVIDENCE_INDEX.md",
    "docs/reporting/VORTEX_FIGURE_AND_TABLE_PLAN.md",
    "docs/reporting/REPORT_SCOPE_AND_MATURITY.md",
    "docs/reporting/SOFTWARE_AND_REPRODUCIBILITY.md",
}

# This module is itself v1 selected evidence, so narrowing the reconciliation below necessarily
# drifts its own recorded hash. That is a self-reference, not a governance-document change, and it
# is held separately so HISTORICAL_GOVERNANCE_ALLOWED_TO_EVOLVE stays exactly the six documents the
# v2 refresh rewrote. Every other test file must still match its v1 hash.
HISTORICAL_SELF_REFERENTIAL_EVIDENCE = {
    "tests/test_vortex_report_freeze.py",
}

HISTORICAL_EVIDENCE_ALLOWED_TO_DRIFT = (
    HISTORICAL_GOVERNANCE_ALLOWED_TO_EVOLVE | HISTORICAL_SELF_REFERENTIAL_EVIDENCE
)

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


def _csv(name: str) -> list[dict[str, str]]:
    with (FREEZE / name).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_frozen_claim_schema_is_exact_and_ids_are_unique() -> None:
    path = FREEZE / "vortex_report_claims.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        assert reader.fieldnames == CLAIM_COLUMNS
    ids = [row["claim_id"] for row in rows]
    assert ids
    assert len(ids) == len(set(ids))


def test_each_claim_evidence_path_exists() -> None:
    for row in _csv("vortex_report_claims.csv"):
        for column in ("source_code_path", "data_path", "figure_path", "test_path"):
            for value in filter(None, (part.strip() for part in row[column].split(";"))):
                assert (ROOT / value).is_file(), f"{row['claim_id']}: missing {column} {value}"


def test_every_selected_evidence_path_still_exists() -> None:
    rows = _csv("vortex_report_files.csv")
    assert rows
    missing = [row["path"] for row in rows if not (ROOT / row["path"]).is_file()]
    assert missing == [], f"v1 selected evidence has been deleted or moved: {missing}"


def test_declared_governance_evolution_set_matches_frozen_evidence() -> None:
    """The declared sets must name real v1 selected-evidence paths, not stale or misspelt ones."""
    selected = {row["path"] for row in _csv("vortex_report_files.csv")}
    unknown = sorted(HISTORICAL_EVIDENCE_ALLOWED_TO_DRIFT - selected)
    assert unknown == [], f"declared drift-allowed paths are not v1 selected evidence: {unknown}"
    assert len(HISTORICAL_GOVERNANCE_ALLOWED_TO_EVOLVE) == 6
    # The self-referential exemption must never be used to excuse a governance document.
    assert not (HISTORICAL_SELF_REFERENTIAL_EVIDENCE & HISTORICAL_GOVERNANCE_ALLOWED_TO_EVOLVE)


def test_selected_evidence_hashes_match_disk_outside_authorised_governance_evolution() -> None:
    """Reconcile the immutable v1 hashes against the current working tree.

    v1 recorded hashes are never rewritten. Only the six declared governance documents, plus this
    self-referential test module, are authorised to differ from them; every other selected file
    must still match v1 byte for byte.
    """
    rows = _csv("vortex_report_files.csv")
    assert rows

    drifted: set[str] = set()
    for row in rows:
        path = ROOT / row["path"]
        assert path.is_file(), f"v1 selected evidence is missing: {row['path']}"
        if _sha256(path) != row["sha256"]:
            drifted.add(row["path"])

    # Any drift outside the declared sets is a real regression: physics source, numerical outputs,
    # figures, calibration templates and every other test must still match the historical snapshot.
    unexpected = sorted(drifted - HISTORICAL_EVIDENCE_ALLOWED_TO_DRIFT)
    assert unexpected == [], (
        "unauthorised drift in v1 selected evidence (not an authorised governance document): "
        f"{unexpected}"
    )

    # The declared sets are exact, not an upper bound: each named path is expected to have been
    # rewritten by the v2 governance refresh.
    no_longer_drifting = sorted(HISTORICAL_EVIDENCE_ALLOWED_TO_DRIFT - drifted)
    assert no_longer_drifting == [], (
        "declared drift-allowed paths unexpectedly still match their v1 hashes; update the "
        f"declared sets if this is intended: {no_longer_drifting}"
    )

    assert drifted == HISTORICAL_EVIDENCE_ALLOWED_TO_DRIFT
    # The six authorised governance drifts are exactly the documents the v2 refresh rewrote.
    assert drifted - HISTORICAL_SELF_REFERENTIAL_EVIDENCE == HISTORICAL_GOVERNANCE_ALLOWED_TO_EVOLVE


def test_frozen_v1_hashes_are_never_rewritten_in_place() -> None:
    """The v1 freeze metadata itself stays internally consistent and untouched."""
    manifest = json.loads((FREEZE / "vortex_report_manifest.json").read_text(encoding="utf-8"))
    assert manifest["freeze_name"] == "vortex_report_first"
    assert _sha256(FREEZE / "vortex_report_files.csv") == manifest["files_index_sha256"]
    assert _sha256(FREEZE / "vortex_report_claims.csv") == manifest["claims_sha256"]


def test_manifest_indexes_match_their_frozen_files() -> None:
    manifest = json.loads((FREEZE / "vortex_report_manifest.json").read_text(encoding="utf-8"))
    claims = ROOT / manifest["claims_path"]
    files = ROOT / manifest["files_index_path"]
    assert _sha256(claims) == manifest["claims_sha256"]
    assert _sha256(files) == manifest["files_index_sha256"]
    assert manifest["selected_evidence_count"] == len(_csv("vortex_report_files.csv"))


def test_freeze_has_no_selected_large_binary_or_secret_signature() -> None:
    assert all(int(row["size_bytes"]) <= 90 * 1024 * 1024 for row in _csv("vortex_report_files.csv"))
    manifest = json.loads((FREEZE / "vortex_report_manifest.json").read_text(encoding="utf-8"))
    assert manifest["audit"]["secret_signature_findings"] == []


def test_report_ready_references_are_resolved() -> None:
    broken = [
        row for row in _csv("path_reference_audit.csv")
        if row["scope"] == "report_ready" and row["exists"].lower() == "false"
    ]
    assert broken == []


def test_scope_forbids_experimental_and_material_claims() -> None:
    manifest = json.loads((FREEZE / "vortex_report_manifest.json").read_text(encoding="utf-8"))
    scope = manifest["scope_boundary"].lower()
    assert "no experimental validation" in scope
    assert "no" in scope and "material-modification claim" in scope


def test_phase_outcomes_are_reported_without_promotion() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for outcome in ("PHASE1-B", "PHASE1R-B", "PHASE2A-B", "PHASE2B-A", "PHASE2C-B"):
        assert outcome in readme
