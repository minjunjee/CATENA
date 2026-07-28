from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from catena.core.io import file_sha256
from catena.eval.postcore_evidence_freeze import (
    REQUIRED_RECORDS,
    SCOPE_FLAGS,
    freeze_postcore_evidence,
    validate_postcore_evidence_contract,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _scope_mapping(values: tuple[bool, ...]) -> dict[str, bool]:
    return dict(zip(SCOPE_FLAGS, values, strict=True))


def _fixture_contract(
    repo_root: Path,
    artifact_root: Path,
) -> dict[str, dict[str, Any]]:
    anchor_payloads = {
        protocol["freeze_anchor"]: {
            "freeze_anchor": protocol["freeze_anchor"],
        }
        for protocol in REQUIRED_RECORDS.values()
    }
    for relative, payload in anchor_payloads.items():
        _write_json(artifact_root / relative, payload)

    repo_lock = repo_root / "docs/E13A_R1_SEQUENCE_CALIBRATION_LOCK.json"
    _write_json(repo_lock, {"original": "locked"})

    contract: dict[str, dict[str, Any]] = {}
    for name, protocol in REQUIRED_RECORDS.items():
        experiment_id = str(protocol["experiment_id"])
        run_id = str(protocol["run_id"])
        run_dir = artifact_root / experiment_id / run_id
        report = run_dir / "report.json"
        manifest = run_dir / "run_manifest.json"
        _write_json(report, {"status": "PASS"})
        _write_json(
            manifest,
            {
                "experiment_id": experiment_id,
                "run_id": run_id,
            },
        )
        anchor_path = artifact_root / str(protocol["freeze_anchor"])
        specification: dict[str, Any] = {
            "experiment_id": experiment_id,
            "run_id": run_id,
            "exact_run_path": f"{experiment_id}/{run_id}",
            "record_role": protocol["record_role"],
            "claim_disposition": protocol["claim_disposition"],
            "evidence_tier": "CONTROLLED_REFERENCE",
            "scope_flags": _scope_mapping(protocol["scope_flags"]),
            "files": {
                "report.json": file_sha256(report),
                "run_manifest.json": file_sha256(manifest),
            },
            "expected_report_fields": {"status": "PASS"},
            "expected_manifest_fields": {
                "experiment_id": experiment_id,
                "run_id": run_id,
            },
            "anchors": [
                {
                    "path": protocol["freeze_anchor"],
                    "sha256": file_sha256(anchor_path),
                    "expected_fields": {
                        "freeze_anchor": protocol["freeze_anchor"],
                    },
                }
            ],
        }
        if protocol.get("repo_anchor") is not None:
            specification["repo_anchors"] = [
                {
                    "path": protocol["repo_anchor"],
                    "sha256": file_sha256(repo_lock),
                    "expected_fields": {"original": "locked"},
                }
            ]
        contract[name] = specification
    return contract


def test_dry_run_resolves_all_exact_records_and_keeps_scopes(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    artifact_root = tmp_path / "artifacts"
    contract = _fixture_contract(repo_root, artifact_root)

    registry = freeze_postcore_evidence(
        repo_root=repo_root,
        artifact_root=artifact_root,
        evidence_contract=contract,
        dry_run=True,
    )

    assert registry["mode"] == "VALIDATED_DRY_RUN_NO_CANONICAL_FREEZE"
    assert registry["postcore_registry_complete"] is True
    assert registry["canonical_freeze_written"] is False
    assert len(registry["evidence"]) == 11
    assert registry["scope_index"]["structured_sequence_claim_eligible"] == [
        "e13c_r1"
    ]
    assert registry["scope_index"]["official_operator_claim_eligible"] == []
    assert registry["scope_index"]["language_model_claim_eligible"] == []
    assert registry["scope_index"]["agent_claim_eligible"] == []
    assert registry["disposition_groups"]["e10_rank_scaling"] == {
        "original": {
            "record": "e10_original",
            "claim_disposition": "NOT_OPENED",
            "valid": True,
        },
        "prospective_repair": {
            "record": "e10b_prospective_repair",
            "claim_disposition": "SUPPORTED",
            "valid": True,
        },
    }


def test_manifest_hash_drift_invalidates_only_affected_record(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    artifact_root = tmp_path / "artifacts"
    contract = _fixture_contract(repo_root, artifact_root)
    record = REQUIRED_RECORDS["e12_canonical"]
    manifest = (
        artifact_root
        / str(record["experiment_id"])
        / str(record["run_id"])
        / "run_manifest.json"
    )
    _write_json(manifest, {"experiment_id": "tampered"})

    registry = freeze_postcore_evidence(
        repo_root=repo_root,
        artifact_root=artifact_root,
        evidence_contract=contract,
        dry_run=True,
    )

    assert registry["postcore_registry_complete"] is False
    assert registry["evidence"]["e12_canonical"]["valid"] is False
    assert "hash mismatch" in registry["evidence"]["e12_canonical"]["error"]
    assert registry["evidence"]["e13c_r1"]["valid"] is True


def test_contract_rejects_noncanonical_e12_run(tmp_path: Path) -> None:
    contract = _fixture_contract(
        tmp_path / "repo",
        tmp_path / "artifacts",
    )
    contract["e12_canonical"]["run_id"] = "20260727T182449.721061Z"

    with pytest.raises(ValueError, match="identity mismatch"):
        validate_postcore_evidence_contract(contract)


def test_contract_rejects_scope_or_disposition_drift(tmp_path: Path) -> None:
    contract = _fixture_contract(
        tmp_path / "repo",
        tmp_path / "artifacts",
    )
    scope_drift = copy.deepcopy(contract)
    scope_drift["e15_canonical_dry_gate"]["scope_flags"][
        "official_operator_claim_eligible"
    ] = True
    with pytest.raises(ValueError, match="scope mismatch"):
        validate_postcore_evidence_contract(scope_drift)

    disposition_drift = copy.deepcopy(contract)
    disposition_drift["e10_original"]["claim_disposition"] = "SUPPORTED"
    with pytest.raises(ValueError, match="claim disposition mismatch"):
        validate_postcore_evidence_contract(disposition_drift)


def test_repo_anchor_hash_drift_is_detected(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    artifact_root = tmp_path / "artifacts"
    contract = _fixture_contract(repo_root, artifact_root)
    _write_json(
        repo_root / "docs/E13A_R1_SEQUENCE_CALIBRATION_LOCK.json",
        {"original": "tampered"},
    )

    registry = freeze_postcore_evidence(
        repo_root=repo_root,
        artifact_root=artifact_root,
        evidence_contract=contract,
        dry_run=True,
    )

    assert registry["postcore_registry_complete"] is False
    original = registry["evidence"]["e13a_original"]
    assert original["valid"] is False
    assert "repository anchor hash mismatch" in original["error"]
