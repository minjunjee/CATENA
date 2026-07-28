from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import freeze_e18_sequence_lattice as freeze


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )


def test_write_exclusive_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "freeze.json"
    payload = {"schema_version": 1}
    freeze._write_exclusive(output, payload)
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    with pytest.raises(FileExistsError):
        freeze._write_exclusive(output, payload)


def test_validate_freeze_rejects_wrong_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = {
        "schema_version": 2,
        "experiment_family": "E18",
        "aggregate_experiment_id": freeze.summaries.AGGREGATE_EXPERIMENT_ID,
        "immutable": True,
        "frozen_at_utc": "2026-07-28T00:00:00Z",
    }
    monkeypatch.setattr(
        freeze,
        "build_freeze_payload",
        lambda **_: payload,
    )
    with pytest.raises(RuntimeError, match="schema_version"):
        freeze.validate_freeze(
            payload,
            repo_root=tmp_path,
            artifact_root=tmp_path,
        )


def test_validate_freeze_detects_nonreproducible_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = {
        "schema_version": 1,
        "experiment_family": "E18",
        "aggregate_experiment_id": freeze.summaries.AGGREGATE_EXPERIMENT_ID,
        "immutable": True,
        "frozen_at_utc": "2026-07-28T00:00:00Z",
        "claim_status": "SUPPORTED",
    }
    rebuilt = dict(payload)
    rebuilt["claim_status"] = "NOT_OPENED"
    monkeypatch.setattr(
        freeze,
        "build_freeze_payload",
        lambda **_: rebuilt,
    )
    with pytest.raises(RuntimeError, match="does not reproduce"):
        freeze.validate_freeze(
            payload,
            repo_root=tmp_path,
            artifact_root=tmp_path,
        )


def _self_consistent_payload(tmp_path: Path) -> dict:
    run_id = "20260728T130000.000000Z"
    supported = True
    return {
        "schema_version": 1,
        "frozen_at_utc": "2026-07-28T13:00:02Z",
        "experiment_family": "E18",
        "aggregate_experiment_id": freeze.summaries.AGGREGATE_EXPERIMENT_ID,
        "run_id": run_id,
        "run_dir": str(
            (
                tmp_path
                / freeze.summaries.AGGREGATE_EXPERIMENT_ID
                / run_id
            ).resolve()
        ),
        "execution_status": "PASS",
        "claim_status": "SUPPORTED",
        "evidence_tier": "CONTROLLED_REFERENCE",
        "scientific_evidence": False,
        "summary": {},
        "contrasts": {},
        "registered_report_claim_gate": {
            "supported": supported,
            "allowed_claim": "verbatim registered report wording",
            "conditions": {"registered_condition": supported},
        },
        "audited_claim_gate": freeze._audited_claim_gate(supported),
        "source_manifest": {
            "run_mode": "MAIN",
            "source_fingerprint": "a" * 64,
            "source_fingerprint_phase": "PRE_RUN",
        },
        "hashes": {
            key: "b" * 64 for key in freeze.FROZEN_HASH_KEYS
        },
        "claim_boundary": freeze._claim_boundary(supported),
        "immutable": True,
    }


def test_validate_freeze_enforces_controlled_reference_boundary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = _self_consistent_payload(tmp_path)
    payload["scientific_evidence"] = True
    monkeypatch.setattr(
        freeze,
        "build_freeze_payload",
        lambda **_: payload,
    )
    with pytest.raises(RuntimeError, match="claim/evidence boundary"):
        freeze.validate_freeze(
            payload,
            repo_root=tmp_path,
            artifact_root=tmp_path,
        )


def test_validate_freeze_enforces_complete_hash_inventory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = _self_consistent_payload(tmp_path)
    payload["hashes"].pop("source_run_provenance.jsonl")
    monkeypatch.setattr(
        freeze,
        "build_freeze_payload",
        lambda **_: payload,
    )
    with pytest.raises(RuntimeError, match="hash inventory"):
        freeze.validate_freeze(
            payload,
            repo_root=tmp_path,
            artifact_root=tmp_path,
        )


def test_audited_gate_narrows_registered_report_wording() -> None:
    gate = freeze._audited_claim_gate(True)
    assert gate["claim_eligible"] is True
    assert "registered-grid mean" in gate["primary_estimand"]
    assert "5/5 paired seeds" in gate["stress_interpretation"]
    assert gate["stress_separate_sesoi_registered"] is False
    assert "relative adjacent non-inferiority" in (
        gate["guardrail_interpretation"]
    )
    assert gate["input_boundary"] == {
        "oracle_erase_write_addresses": True,
        "oracle_candidates": True,
        "explicit_oracle_demand_descriptors": True,
        "model_visible_verified_event_bit": True,
    }
    assert "every grid cell" in gate["forbidden_claim"]
    assert "registered stress SESOI" in gate["forbidden_claim"]
    assert "absolute or accurate preservation" in gate["forbidden_claim"]


def test_claim_boundary_explicitly_closes_overclaims() -> None:
    boundary = freeze._claim_boundary(True)
    assert boundary["controlled_sequence_lattice_claim_eligible"] is True
    assert boundary["registered_grid_mean_claim_eligible"] is True
    assert boundary["stress_five_of_five_direction_claim_eligible"] is True
    assert boundary["explicit_oracle_demand_descriptors"] is True
    assert boundary["model_visible_verified_event_bit"] is True
    assert boundary["every_cell_or_uniform_persistence_claim_eligible"] is False
    assert boundary["stress_sesoi_claim_eligible"] is False
    assert boundary["absolute_accurate_preservation_claim_eligible"] is False
    assert boundary["semantic_demand_or_relevance_inference_claim_eligible"] is False


def test_validate_freeze_rejects_audited_claim_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = _self_consistent_payload(tmp_path)
    payload["audited_claim_gate"]["stress_separate_sesoi_registered"] = True
    monkeypatch.setattr(
        freeze,
        "build_freeze_payload",
        lambda **_: payload,
    )
    with pytest.raises(RuntimeError, match="claim/evidence boundary"):
        freeze.validate_freeze(
            payload,
            repo_root=tmp_path,
            artifact_root=tmp_path,
        )


def test_registered_report_claim_gate_is_preserved_separately(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = _self_consistent_payload(tmp_path)
    registered = dict(payload["registered_report_claim_gate"])
    monkeypatch.setattr(
        freeze,
        "build_freeze_payload",
        lambda **_: payload,
    )
    freeze.validate_freeze(
        payload,
        repo_root=tmp_path,
        artifact_root=tmp_path,
    )
    assert payload["registered_report_claim_gate"] == registered
    assert (
        payload["registered_report_claim_gate"]["allowed_claim"]
        == "verbatim registered report wording"
    )
    assert "allowed_claim_if_supported" in payload["audited_claim_gate"]


def test_build_payload_preserves_registered_gate_and_adds_audit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    artifacts = tmp_path / "artifacts"
    run_id = "20260728T130000.000000Z"
    run_dir = (
        artifacts / freeze.summaries.AGGREGATE_EXPERIMENT_ID / run_id
    )
    registered = {
        "supported": True,
        "conditions": {"all_registered_conditions": True},
        "allowed_claim": "original report wording remains immutable",
        "forbidden_claim": "original report boundary",
    }
    report = {
        "status": "PASS",
        "evidence_tier": "CONTROLLED_REFERENCE",
        "scientific_evidence": False,
        "summary": {"source_runs": 25},
        "contrasts": {"magnitude_factorization": {"passed": True}},
        "claim_gate": registered,
    }
    manifest = {
        "run_mode": "MAIN",
        "source_fingerprint": {"sha256": "a" * 64, "files": 10},
        "source_fingerprint_phase": "RUN_START",
    }
    _write_json(run_dir / "report.json", report)
    _write_json(run_dir / "run_manifest.json", manifest)
    for name in (
        "sequence_control_lattice_paired_metrics.jsonl",
        "sequence_control_lattice_active_path_metrics.jsonl",
        "source_run_provenance.jsonl",
    ):
        (run_dir / name).write_text("{}\n", encoding="utf-8")
    summary_content = "# frozen aggregate summary\n"
    (run_dir / freeze.SUMMARY_FILENAME).write_text(
        summary_content,
        encoding="utf-8",
    )
    lock_path = root / "docs/E18_SEQUENCE_CONTROL_LATTICE_LOCK.json"
    _write_json(lock_path, {"schema_version": 1})
    for relative in (
        "configs/e18a_sequence_control_lattice.yaml",
        "configs/e18b_sequence_control_lattice_aggregate.yaml",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("experiment_id: fixture\n", encoding="utf-8")
    lock_sha256 = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    action = SimpleNamespace(
        kind="AGGREGATE",
        run_dir=run_dir,
        content=summary_content,
    )
    plan = SimpleNamespace(
        aggregate_available=True,
        actions=(action,),
        protocol_lock_sha256=lock_sha256,
    )
    monkeypatch.setattr(
        freeze.summaries,
        "build_summary_plan",
        lambda **_: plan,
    )

    payload = freeze.build_freeze_payload(
        repo_root=root,
        artifact_root=artifacts,
        frozen_at_utc="2026-07-28T13:00:02Z",
    )
    assert payload["registered_report_claim_gate"] == registered
    assert payload["audited_claim_gate"] == freeze._audited_claim_gate(True)
    assert payload["claim_boundary"] == freeze._claim_boundary(True)
    assert "claim_gate" not in payload
    assert payload["hashes"]["report.json"] == hashlib.sha256(
        (run_dir / "report.json").read_bytes()
    ).hexdigest()
