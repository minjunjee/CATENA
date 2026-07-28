from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

from catena.core.provenance_v61 import source_tree_fingerprint
from tools.audit_postcore_artifacts import (
    ExperimentAudit,
    FlatFreezeSpec,
    _audit_source_rows,
    _canonical_sha256,
    _source_provenance_projection,
    audit_e13a_r1_status_amendment,
    audit_e13b_live,
    audit_e13bc_freeze,
    audit_e14_freeze,
    audit_flat_freeze,
    checkpoint_manifest_sha256,
    file_sha256,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_flat_fixture(
    tmp_path: Path,
    *,
    claim_status: str = "NOT_OPENED",
    document_token: str = "full_claim_open: false",
) -> tuple[Path, Path, FlatFreezeSpec]:
    repo = tmp_path / "repo"
    artifacts = tmp_path / "artifacts"
    run = artifacts / "example" / "run-1"
    checkpoint = run / "checkpoints/model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    _write_json(
        run / "report.json",
        {"status": "PASS", "claim_gate": {"supported": claim_status == "SUPPORTED"}},
    )
    _write_json(
        run / "run_manifest.json",
        {"experiment_id": "example", "run_id": "run-1"},
    )
    row = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
    }
    (run / "metrics.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    result_doc = repo / "docs/result.md"
    result_doc.parent.mkdir(parents=True)
    result_doc.write_text(f"run-1\nexecution_status: PASS\n{document_token}\n", encoding="utf-8")
    freeze = {
        "experiment_id": "example",
        "run_id": "run-1",
        "run_dir": str(run),
        "execution_status": "PASS",
        "claim_status": claim_status,
        "hashes": {
            "report.json": file_sha256(run / "report.json"),
            "run_manifest.json": file_sha256(run / "run_manifest.json"),
            "metrics.jsonl": file_sha256(run / "metrics.jsonl"),
            "checkpoint_manifest_1": checkpoint_manifest_sha256(
                run,
                [checkpoint],
                scheme="path_nul_hash",
            ),
            "result_markdown": file_sha256(result_doc),
        },
    }
    _write_json(artifacts / "EXAMPLE_FREEZE.json", freeze)
    spec = FlatFreezeSpec(
        name="example",
        freeze_name="EXAMPLE_FREEZE.json",
        result_doc="docs/result.md",
        aliases={"result_markdown": "docs/result.md"},
        document_tokens=("run-1", "execution_status: PASS", document_token),
        checkpoint_metrics="metrics.jsonl",
    )
    return repo, artifacts, spec


def test_flat_audit_separates_execution_and_claim_and_is_read_only(tmp_path: Path) -> None:
    repo, artifacts, spec = _make_flat_fixture(tmp_path)
    before = {
        path: path.read_bytes()
        for path in sorted(tmp_path.rglob("*"))
        if path.is_file()
    }

    audit = audit_flat_freeze(repo_root=repo, artifact_root=artifacts, spec=spec)

    after = {
        path: path.read_bytes()
        for path in sorted(tmp_path.rglob("*"))
        if path.is_file()
    }
    assert audit.status == "PASS"
    assert audit.execution_status == "PASS"
    assert audit.claim_status == "NOT_OPENED"
    assert before == after


def test_flat_audit_detects_checkpoint_tampering(tmp_path: Path) -> None:
    repo, artifacts, spec = _make_flat_fixture(tmp_path)
    checkpoint = artifacts / "example/run-1/checkpoints/model.pt"
    checkpoint.write_bytes(b"tampered")

    audit = audit_flat_freeze(repo_root=repo, artifact_root=artifacts, spec=spec)

    assert audit.status == "FAIL"
    assert any(
        check.name == "hash:checkpoint_manifest_1" and check.status == "FAIL"
        for check in audit.checks
    )
    assert any(
        check.name == "checkpoint_metric_hashes" and check.status == "FAIL"
        for check in audit.checks
    )


def test_missing_freeze_is_not_complete_not_failure(tmp_path: Path) -> None:
    spec = FlatFreezeSpec(
        name="pending",
        freeze_name="PENDING.json",
        result_doc="docs/pending.md",
        aliases={},
        document_tokens=(),
    )

    audit = audit_flat_freeze(
        repo_root=tmp_path / "repo",
        artifact_root=tmp_path / "artifacts",
        spec=spec,
    )

    assert audit.status == "NOT_COMPLETE"
    assert all(check.status != "FAIL" for check in audit.checks)


def test_document_status_mismatch_is_reported(tmp_path: Path) -> None:
    repo, artifacts, spec = _make_flat_fixture(
        tmp_path,
        document_token="repaired_dependency_eligible: false",
    )
    document = repo / "docs/result.md"
    document.write_text(
        "run-1\nexecution_status: PASS\nrepaired_dependency_eligible: true\n",
        encoding="utf-8",
    )
    freeze_path = artifacts / "EXAMPLE_FREEZE.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze["hashes"]["result_markdown"] = file_sha256(document)
    _write_json(freeze_path, freeze)

    audit = audit_flat_freeze(repo_root=repo, artifact_root=artifacts, spec=spec)

    assert audit.status == "FAIL"
    assert any(
        check.name == "document_token:repaired_dependency_eligible: false"
        and check.status == "FAIL"
        for check in audit.checks
    )


def test_checkpoint_manifest_schemes_are_explicit_and_distinct(tmp_path: Path) -> None:
    run = tmp_path / "run"
    first = run / "checkpoints/a.pt"
    second = run / "checkpoints/b.pt"
    first.parent.mkdir(parents=True)
    first.write_bytes(b"a")
    second.write_bytes(b"b")

    path_nul_hash = checkpoint_manifest_sha256(
        run,
        [second, first],
        scheme="path_nul_hash",
    )
    legacy_sha_path = checkpoint_manifest_sha256(
        run,
        [second, first],
        scheme="legacy_sha_path",
    )

    assert path_nul_hash == checkpoint_manifest_sha256(
        run,
        [first, second],
        scheme="path_nul_hash",
    )
    assert legacy_sha_path == checkpoint_manifest_sha256(
        run,
        [first, second],
        scheme="legacy_sha_path",
    )
    assert path_nul_hash != legacy_sha_path


def test_e13b_live_tolerates_valid_run_start_only_directory(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    artifacts = tmp_path / "artifacts"
    config = {
        "experiment_id": "e13b_r1_transactional_sequence_memory",
        "seeds": [101],
    }
    config_path = repo / "configs/e13b.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    run = artifacts / "e13b_r1_transactional_sequence_memory/run-1"
    run.mkdir(parents=True)
    resolved = run / "config.resolved.yaml"
    resolved.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    (run / "environment.json").write_text("{}\n", encoding="utf-8")
    source = source_tree_fingerprint(repo).as_dict()
    _write_json(
        run / "run_manifest.json",
        {
            "schema_version": 2,
            "experiment_id": "e13b_r1_transactional_sequence_memory",
            "run_id": "run-1",
            "run_mode": "MAIN",
            "source_fingerprint_phase": "RUN_START",
            "source_fingerprint": source,
            "config_path": str(config_path),
            "config_file_sha256": file_sha256(config_path),
            "resolved_config_artifact_sha256": file_sha256(resolved),
            "resolved_config_sha256": _canonical_sha256(config),
            "config": config,
        },
    )

    audit = audit_e13b_live(repo_root=repo, artifact_root=artifacts)

    assert audit.status == "NOT_COMPLETE"
    assert all(check.status != "FAIL" for check in audit.checks)


def test_e13b_completed_checkpoint_payload_is_cross_checked(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    artifacts = tmp_path / "artifacts"
    config = {
        "experiment_id": "e13b_r1_transactional_sequence_memory",
        "seeds": [101],
    }
    config_path = repo / "configs/e13b.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    run = artifacts / "e13b_r1_transactional_sequence_memory/run-1"
    checkpoint = run / "checkpoints/tied_seed101.pt"
    checkpoint.parent.mkdir(parents=True)
    torch.save(
        {
            "model": {"weight": torch.tensor([1.0])},
            "variant": "tied",
            "seed": 101,
            "config": config,
            "model_class": "TransactionalSequenceMemoryV2",
        },
        checkpoint,
    )
    resolved = run / "config.resolved.yaml"
    resolved.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    (run / "environment.json").write_text("{}\n", encoding="utf-8")
    row = {
        "variant": "tied",
        "seed": 101,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
    }
    (run / "sequence_main_metrics.jsonl").write_text(
        json.dumps(row) + "\n",
        encoding="utf-8",
    )
    _write_json(
        run / "report.json",
        {
            "status": "PASS",
            "run_mode": "MAIN",
            "variant": "tied",
            "seed": 101,
            "rows": 1,
            "claim_gate": {"status": "PENDING_AGGREGATE"},
        },
    )
    source = source_tree_fingerprint(repo).as_dict()
    _write_json(
        run / "run_manifest.json",
        {
            "schema_version": 2,
            "experiment_id": "e13b_r1_transactional_sequence_memory",
            "run_id": "run-1",
            "run_mode": "MAIN",
            "source_fingerprint_phase": "RUN_START",
            "source_fingerprint": source,
            "config_path": str(config_path),
            "config_file_sha256": file_sha256(config_path),
            "resolved_config_artifact_sha256": file_sha256(resolved),
            "resolved_config_sha256": _canonical_sha256(config),
            "config": config,
            "completed_at_utc": "2026-07-27T00:00:00+00:00",
            "report_sha256": file_sha256(run / "report.json"),
        },
    )

    audit = audit_e13b_live(repo_root=repo, artifact_root=artifacts)

    assert audit.status == "PASS"
    assert any(
        check.name == "run-1:checkpoint_payload" and check.status == "PASS"
        for check in audit.checks
    )


def _make_e13a_r1_amendment_fixture(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    artifacts = tmp_path / "artifacts"
    original_doc = repo / "docs/E13A_SEQUENCE_CALIBRATION_RESULT_KO.md"
    amendment_doc = repo / "docs/E13A_R1_RESULT_STATUS_AMENDMENT_KO.md"
    original_doc.parent.mkdir(parents=True)
    original_doc.write_text("PASS / GO_FOR_E13B\n", encoding="utf-8")
    original_doc_sha = file_sha256(original_doc)
    amendment_doc.write_text(
        "\n".join(
            (
                original_doc_sha,
                "ff1f13a6955719ada91120891404cbdb43e57d24c4e522f48e007f822e56dd4e",
                "GO_FOR_ORIGINAL_HARD_MASKED_PIPELINE_ONLY",
                "repaired_e13b_dependency_eligible: false",
                "20260727T190642.222102Z",
                "GO_FOR_E13B_R1",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    original_freeze_path = artifacts / "E13A_R1_POSTCORE_ARTIFACT_FREEZE_V1.json"
    _write_json(
        original_freeze_path,
        {"hashes": {"result_markdown": original_doc_sha}},
    )
    r1_run = (
        artifacts
        / "e13a_r1_sequence_floor_throughput"
        / "20260727T183609.755945Z"
    )
    r2_run = (
        artifacts
        / "e13a_r2_sequence_floor_throughput"
        / "20260727T190642.222102Z"
    )
    _write_json(
        r1_run / "report.json",
        {"status": "PASS", "claim_gate": {"go_for_e13b": True}},
    )
    _write_json(
        r1_run / "run_manifest.json",
        {"report_sha256": file_sha256(r1_run / "report.json")},
    )
    _write_json(
        r2_run / "report.json",
        {"status": "PASS", "claim_gate": {"go_for_e13b_r1": True}},
    )
    _write_json(
        r2_run / "run_manifest.json",
        {"report_sha256": file_sha256(r2_run / "report.json")},
    )
    paths = {
        "amendment_markdown": str(amendment_doc),
        "original_result_markdown": str(original_doc),
        "original_e13a_r1_freeze": str(original_freeze_path),
        "e13a_r1_report": str(r1_run / "report.json"),
        "e13a_r1_run_manifest": str(r1_run / "run_manifest.json"),
        "e13a_r2_report": str(r2_run / "report.json"),
        "e13a_r2_run_manifest": str(r2_run / "run_manifest.json"),
    }
    _write_json(
        artifacts / "E13A_R1_RESULT_STATUS_AMENDMENT_FREEZE_V1.json",
        {
            "original_artifacts_immutable": True,
            "immutable": True,
            "e13a_r1": {
                "experiment_id": "e13a_r1_sequence_floor_throughput",
                "run_id": "20260727T183609.755945Z",
                "execution_status": "PASS",
                "calibration_status": "GO_FOR_ORIGINAL_HARD_MASKED_PIPELINE_ONLY",
                "repaired_e13b_dependency_eligible": False,
                "diagnosis": "DISTRACTOR_PATH_STRUCTURALLY_HARD_MASKED",
            },
            "repaired_dependency": {
                "experiment_id": "e13a_r2_sequence_floor_throughput",
                "run_id": "20260727T190642.222102Z",
                "calibration_status": "GO_FOR_E13B_R1",
                "exclusive_for_repaired_e13b_r1": True,
            },
            "paths": paths,
            "hashes": {
                name: file_sha256(Path(path))
                for name, path in paths.items()
            },
        },
    )
    return repo, artifacts


def test_e13a_r1_additive_amendment_preserves_historical_document(
    tmp_path: Path,
) -> None:
    repo, artifacts = _make_e13a_r1_amendment_fixture(tmp_path)
    audit = ExperimentAudit(
        name="E13a-R1",
        freeze_path=None,
        result_doc=None,
    )

    audit_e13a_r1_status_amendment(
        audit,
        repo_root=repo,
        artifact_root=artifacts,
    )

    assert audit.status == "PASS"
    assert any(
        check.name == "status_amendment_r1_disposition"
        and check.status == "PASS"
        for check in audit.checks
    )
    assert any(
        check.name == "status_amendment_repaired_dependency"
        and check.status == "PASS"
        for check in audit.checks
    )


def test_e13a_r1_additive_amendment_detects_original_document_drift(
    tmp_path: Path,
) -> None:
    repo, artifacts = _make_e13a_r1_amendment_fixture(tmp_path)
    (repo / "docs/E13A_SEQUENCE_CALIBRATION_RESULT_KO.md").write_text(
        "mutated\n",
        encoding="utf-8",
    )
    audit = ExperimentAudit(
        name="E13a-R1",
        freeze_path=None,
        result_doc=None,
    )

    audit_e13a_r1_status_amendment(
        audit,
        repo_root=repo,
        artifact_root=artifacts,
    )

    assert audit.status == "FAIL"
    assert any(
        check.name == "status_amendment_hash:original_result_markdown"
        and check.status == "FAIL"
        for check in audit.checks
    )


def test_e13bc_and_e14_missing_freezes_are_not_complete(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    artifacts = tmp_path / "artifacts"

    e13bc = audit_e13bc_freeze(repo_root=repo, artifact_root=artifacts)
    e14 = audit_e14_freeze(repo_root=repo, artifact_root=artifacts)

    assert e13bc.status == "NOT_COMPLETE"
    assert e14.status == "NOT_COMPLETE"
    assert all(check.status != "FAIL" for check in e13bc.checks)
    assert all(check.status != "FAIL" for check in e14.checks)


def test_source_row_audit_derives_run_dir_and_detects_checkpoint_drift(
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "artifacts"
    run = (
        artifacts
        / "e13b_r1_transactional_sequence_memory"
        / "20260727T191308.445971Z"
    )
    checkpoint = run / "checkpoints/dual_seed101.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    _write_json(run / "report.json", {"status": "PASS"})
    _write_json(run / "run_manifest.json", {"run_id": run.name})
    (run / "sequence_main_metrics.jsonl").write_text("{}\n", encoding="utf-8")
    row = {
        "seed": 101,
        "variant": "dual",
        "source_run_id": run.name,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "source_report_sha256": file_sha256(run / "report.json"),
        "source_metrics_sha256": file_sha256(
            run / "sequence_main_metrics.jsonl"
        ),
        "source_manifest_sha256": file_sha256(run / "run_manifest.json"),
    }
    clean = ExperimentAudit(name="source", freeze_path=None, result_doc=None)

    _audit_source_rows(
        clean,
        rows=[row],
        artifact_root=artifacts,
        check_name="source",
    )

    assert clean.status == "PASS"
    checkpoint.write_bytes(b"tampered")
    drifted = ExperimentAudit(name="source", freeze_path=None, result_doc=None)
    _audit_source_rows(
        drifted,
        rows=[row],
        artifact_root=artifacts,
        check_name="source",
    )
    assert drifted.status == "FAIL"
    assert any(
        check.name == "source:(101, 'dual'):hash:checkpoint"
        and check.status == "FAIL"
        for check in drifted.checks
    )


def test_e14_selected_and_sealed_provenance_shapes_compare_equally() -> None:
    run_dir = (
        "/data/minjun_dev/CATENA/artifacts/"
        "e13b_r1_transactional_sequence_memory/run-1"
    )
    selected = {
        "seed": 101,
        "variant": "dual",
        "source_run_id": "run-1",
        "checkpoint_path": f"{run_dir}/checkpoints/dual_seed101.pt",
        "checkpoint_sha256": "a" * 64,
        "source_report_sha256": "b" * 64,
        "source_metrics_sha256": "c" * 64,
        "source_manifest_sha256": "d" * 64,
    }
    sealed = {
        "seed": 101,
        "variant": "dual",
        "source_run_dir": run_dir,
        "checkpoint_path": selected["checkpoint_path"],
        "checkpoint_sha256": selected["checkpoint_sha256"],
        "source_report_path": f"{run_dir}/report.json",
        "source_report_sha256": selected["source_report_sha256"],
        "source_metrics_path": f"{run_dir}/sequence_main_metrics.jsonl",
        "source_metrics_sha256": selected["source_metrics_sha256"],
        "source_manifest_path": f"{run_dir}/run_manifest.json",
        "source_manifest_sha256": selected["source_manifest_sha256"],
    }

    assert _source_provenance_projection(selected) == (
        _source_provenance_projection(sealed)
    )
    sealed["checkpoint_sha256"] = "e" * 64
    assert _source_provenance_projection(selected) != (
        _source_provenance_projection(sealed)
    )
