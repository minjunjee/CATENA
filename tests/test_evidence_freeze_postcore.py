from __future__ import annotations

import json
from pathlib import Path

import pytest

from catena.core.io import file_sha256
from catena.eval.evidence_freeze import freeze_evidence, resolve_report


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _contract(root: Path) -> dict[str, object]:
    run = root / "experiment" / "run-1"
    anchor = root / "CLAIM.json"
    return {
        "claim": {
            "experiment_id": "experiment",
            "run_id": "run-1",
            "claim_disposition": "INCONCLUSIVE",
            "files": {
                "report.json": file_sha256(run / "report.json"),
                "run_manifest.json": file_sha256(run / "run_manifest.json"),
            },
            "expected_report_fields": {
                "status": "PASS",
                "claim_gate.supported": False,
            },
            "anchors": [
                {
                    "path": "CLAIM.json",
                    "sha256": file_sha256(anchor),
                    "expected_fields": {"claim_status": "INCONCLUSIVE"},
                }
            ],
        }
    }


def test_pinned_freeze_separates_execution_from_claim(tmp_path: Path) -> None:
    run = tmp_path / "experiment" / "run-1"
    _write_json(run / "report.json", {"status": "PASS", "claim_gate": {"supported": False}})
    _write_json(run / "run_manifest.json", {"experiment_id": "experiment"})
    _write_json(tmp_path / "CLAIM.json", {"claim_status": "INCONCLUSIVE"})

    registry = freeze_evidence(
        artifact_root=tmp_path,
        evidence_contract=_contract(tmp_path),
    )

    assert registry["core_registry_complete"] is True
    item = registry["evidence"]["claim"]
    assert item["execution_status"] == "PASS"
    assert item["claim_disposition"] == "INCONCLUSIVE"
    assert item["valid"] is True


def test_pinned_freeze_rejects_hash_drift(tmp_path: Path) -> None:
    run = tmp_path / "experiment" / "run-1"
    _write_json(run / "report.json", {"status": "PASS", "claim_gate": {"supported": False}})
    _write_json(run / "run_manifest.json", {"experiment_id": "experiment"})
    _write_json(tmp_path / "CLAIM.json", {"claim_status": "INCONCLUSIVE"})
    contract = _contract(tmp_path)
    _write_json(run / "report.json", {"status": "PASS", "claim_gate": {"supported": True}})

    registry = freeze_evidence(artifact_root=tmp_path, evidence_contract=contract)

    assert registry["core_registry_complete"] is False
    assert registry["evidence"]["claim"]["valid"] is False
    assert "hash mismatch" in registry["evidence"]["claim"]["error"]


def test_resolve_report_anchors_relative_latest_to_experiment(tmp_path: Path) -> None:
    run = tmp_path / "experiment" / "run-1"
    _write_json(run / "report.json", {"status": "PASS"})
    _write_json(tmp_path / "experiment" / "latest.json", {"run_dir": "run-1"})

    report_path, report = resolve_report(tmp_path, "experiment")

    assert report_path == run / "report.json"
    assert report["status"] == "PASS"


def test_resolve_report_rejects_pointer_escape(tmp_path: Path) -> None:
    _write_json(tmp_path / "outside" / "report.json", {"status": "PASS"})
    _write_json(tmp_path / "experiment" / "latest.json", {"run_dir": "../outside"})

    with pytest.raises(ValueError, match="escapes experiment root"):
        resolve_report(tmp_path, "experiment")


def test_validate_only_never_marks_registry_complete(tmp_path: Path) -> None:
    run = tmp_path / "experiment" / "run-1"
    _write_json(run / "report.json", {"status": "PASS", "claim_gate": {"supported": False}})
    _write_json(run / "run_manifest.json", {"experiment_id": "experiment"})
    _write_json(tmp_path / "CLAIM.json", {"claim_status": "INCONCLUSIVE"})

    registry = freeze_evidence(
        artifact_root=tmp_path,
        evidence_contract=_contract(tmp_path),
        validate_only=True,
    )

    assert registry["mode"] == "CONTRACT_VALIDATION_ONLY"
    assert registry["core_registry_complete"] is False
