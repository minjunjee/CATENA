from __future__ import annotations

import hashlib
import json
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from catena.lm.frozen_invariance import (
    FrozenInvarianceError,
    _artifact_inventory,
    _row_aggregate,
    build_frozen_invariance_receipt,
    validate_frozen_invariance_receipt,
    validate_historical_frozen_invariance_receipt,
    verify_frozen_artifacts,
    verify_pre_e26_source,
)


def _git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=root, text=True).strip()


def test_pre_e26_source_accepts_additive_descendant_but_not_base_blob_change(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "catena-test@example.invalid")
    _git(repo, "config", "user.name", "CATENA Test")
    base_file = repo / "base.txt"
    base_file.write_text("frozen\n", encoding="utf-8")
    _git(repo, "add", "base.txt")
    _git(repo, "commit", "-m", "base")
    expected_head = _git(repo, "rev-parse", "HEAD")
    base_bytes = base_file.read_bytes()
    base_sha = hashlib.sha256(base_bytes).hexdigest()
    aggregate = _row_aggregate([{"path": "base.txt", "bytes": len(base_bytes), "sha256": base_sha}])

    (repo / "additive.txt").write_text("new\n", encoding="utf-8")
    _git(repo, "add", "additive.txt")
    _git(repo, "commit", "-m", "additive descendant")
    additive = verify_pre_e26_source(
        live_repo=repo,
        expected_head=expected_head,
        expected_file_count=1,
        expected_aggregate_sha256=aggregate,
    )
    assert additive["passed"] is True
    assert additive["head_matches"] is False
    assert additive["expected_head_is_ancestor"] is True
    assert additive["verification_mode"] == "ANCESTOR_COMMIT_BASE_BLOBS_BYTE_IDENTICAL"

    base_file.write_text("changed\n", encoding="utf-8")
    _git(repo, "add", "base.txt")
    _git(repo, "commit", "-m", "change frozen blob")
    changed = verify_pre_e26_source(
        live_repo=repo,
        expected_head=expected_head,
        expected_file_count=1,
        expected_aggregate_sha256=aggregate,
    )
    assert changed["passed"] is False
    assert changed["changed"] == ["base.txt"]


def test_structured_frozen_receipt_requires_live_source_and_all_2062_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = tmp_path / "baseline.json"
    baseline.write_text("{}\n", encoding="utf-8")
    source = {
        "scope": "all_files_tracked_at_pre_e26_base_commit",
        "expected_head": "a" * 40,
        "observed_head": "a" * 40,
        "head_matches": True,
        "expected_head_is_ancestor": True,
        "verification_mode": "ANCESTOR_COMMIT_BASE_BLOBS_BYTE_IDENTICAL",
        "git_clean": True,
        "registered_file_count": 556,
        "expected_files": 556,
        "observed_files": 556,
        "registered_aggregate_sha256": "b" * 64,
        "base_aggregate_sha256": "b" * 64,
        "missing": [],
        "changed": [],
        "passed": True,
    }
    artifacts = {
        "scope": "immutable_E00_through_E25_completed_artifacts",
        "baseline_manifest": str(baseline.resolve()),
        "base_manifest": str(baseline.resolve()),
        "registered_base_file_count": 1329,
        "registered_base_aggregate_sha256": "c" * 64,
        "expected_file_count": 2062,
        "observed_file_count": 2062,
        "registered_file_count": 2062,
        "expected_aggregate_sha256": "d" * 64,
        "observed_aggregate_sha256": "d" * 64,
        "extension_file_count": 733,
        "extension_aggregate_sha256": "e" * 64,
        "excluded_experiment_min": 26,
        "missing": [],
        "unexpected": [],
        "changed": [],
        "missing_namespaces": [],
        "unexpected_namespaces": [],
        "changed_namespaces": [],
        "passed": True,
    }
    monkeypatch.setattr(
        "catena.lm.frozen_invariance.verify_pre_e26_source",
        lambda **_: deepcopy(source),
    )
    monkeypatch.setattr(
        "catena.lm.frozen_invariance.verify_frozen_artifacts",
        lambda **_: deepcopy(artifacts),
    )
    data_lock = {
        "repository": {
            "live_repo": str(tmp_path),
            "expected_live_head": "a" * 40,
            "pre_e26_source_file_count": 556,
            "pre_e26_source_aggregate_sha256": "b" * 64,
            "frozen_artifact_file_count": 1329,
            "frozen_artifact_aggregate_sha256": "c" * 64,
        }
    }
    receipt = build_frozen_invariance_receipt(
        data_lock=data_lock,
        baseline_manifest=baseline,
    )
    assert receipt["passed"] is True
    assert receipt["live_repository"]["expected_files"] == 556
    assert receipt["frozen_artifacts"]["registered_base_file_count"] == 1329
    assert receipt["frozen_artifacts"]["expected_file_count"] == 2062
    assert receipt["frozen_artifacts"]["excluded_experiment_min"] == 26
    assert (
        validate_frozen_invariance_receipt(
            receipt,
            data_lock=data_lock,
        )
        == receipt
    )

    tampered = deepcopy(receipt)
    tampered["frozen_artifacts"]["changed_namespaces"] = ["e25a_official_gate"]
    with pytest.raises(FrozenInvarianceError, match="differs from a live"):
        validate_frozen_invariance_receipt(tampered, data_lock=data_lock)


def _write_completed_lock(tmp_path: Path) -> tuple[Path, Path]:
    artifact_root = tmp_path / "artifacts"
    (artifact_root / "e21_boundary").mkdir(parents=True)
    (artifact_root / "e22a_screen").mkdir()
    (artifact_root / "e25a_terminal_gate").mkdir()
    (artifact_root / "e26_new_run").mkdir()
    (artifact_root / "e21_boundary" / "report.json").write_text("e21\n", encoding="utf-8")
    (artifact_root / "e22a_screen" / "report.json").write_text("e22\n", encoding="utf-8")
    (artifact_root / "e25a_terminal_gate" / "report.json").write_text("e25\n", encoding="utf-8")
    (artifact_root / "e26_new_run" / "report.json").write_text("e26\n", encoding="utf-8")
    (artifact_root / "E26_NEW_STATUS.json").write_text("e26 status\n", encoding="utf-8")
    (artifact_root / "POST_E21_WAVE1_STATUS.json").write_text("status\n", encoding="utf-8")

    lock_root = tmp_path / "locks"
    lock_root.mkdir()
    base_inventory = _artifact_inventory(
        artifact_root,
        minimum_experiment=0,
        maximum_experiment=21,
    )
    base = {
        "scope": "immutable_E00_through_E21_artifacts",
        "artifact_root": str(artifact_root.resolve()),
        "file_count": base_inventory["file_count"],
        "aggregate_sha256": base_inventory["aggregate_sha256"],
        "files": base_inventory["files"],
    }
    base_path = lock_root / "base.json"
    base_path.write_text(json.dumps(base, sort_keys=True) + "\n", encoding="utf-8")

    extension = _artifact_inventory(
        artifact_root,
        minimum_experiment=22,
        maximum_experiment=25,
    )
    namespaces: list[dict[str, Any]] = []
    for namespace in sorted({str(row["path"]).split("/", 1)[0] for row in extension["files"]}):
        rows = [
            row
            for row in extension["files"]
            if row["path"] == namespace or str(row["path"]).startswith(f"{namespace}/")
        ]
        namespaces.append(
            {
                "path": namespace,
                "file_count": len(rows),
                "total_bytes": sum(int(row["bytes"]) for row in rows),
                "aggregate_sha256": _row_aggregate(rows),
            }
        )
    completed = _artifact_inventory(
        artifact_root,
        minimum_experiment=0,
        maximum_experiment=25,
    )
    lock = {
        "schema_version": 1,
        "scope": "immutable_E00_through_E25_completed_artifacts",
        "artifact_root": str(artifact_root.resolve()),
        "file_count": completed["file_count"],
        "total_bytes": completed["total_bytes"],
        "aggregate_sha256": completed["aggregate_sha256"],
        "base_manifest": {
            "path": base_path.name,
            "sha256": hashlib.sha256(base_path.read_bytes()).hexdigest(),
            "scope": "immutable_E00_through_E21_artifacts",
            "file_count": base["file_count"],
            "aggregate_sha256": base["aggregate_sha256"],
        },
        "extension": {
            "experiment_min": 22,
            "experiment_max": 25,
            "include_top_level": ["POST_E21_WAVE1_STATUS.json"],
            "file_count": extension["file_count"],
            "total_bytes": extension["total_bytes"],
            "aggregate_sha256": extension["aggregate_sha256"],
            "namespaces": namespaces,
        },
    }
    lock_path = lock_root / "completed.json"
    lock_path.write_text(json.dumps(lock, sort_keys=True) + "\n", encoding="utf-8")
    return lock_path, artifact_root


def test_completed_evidence_lock_covers_e22_e25_but_excludes_e26(tmp_path: Path) -> None:
    lock_path, artifact_root = _write_completed_lock(tmp_path)
    base = json.loads((lock_path.parent / "base.json").read_text(encoding="utf-8"))
    result = verify_frozen_artifacts(
        baseline_manifest=lock_path,
        expected_file_count=int(base["file_count"]),
        expected_aggregate_sha256=str(base["aggregate_sha256"]),
    )
    assert result["passed"] is True
    assert result["scope"] == "immutable_E00_through_E25_completed_artifacts"
    assert result["extension_file_count"] == 3
    assert result["expected_file_count"] == 4
    assert result["excluded_experiment_min"] == 26

    (artifact_root / "e26_new_run" / "report.json").write_text("changed e26\n", encoding="utf-8")
    (artifact_root / "E26_NEW_STATUS.json").write_text("changed e26 status\n", encoding="utf-8")
    assert (
        verify_frozen_artifacts(
            baseline_manifest=lock_path,
            expected_file_count=int(base["file_count"]),
            expected_aggregate_sha256=str(base["aggregate_sha256"]),
        )["passed"]
        is True
    )

    (artifact_root / "e25a_terminal_gate" / "report.json").write_text(
        "changed e25\n", encoding="utf-8"
    )
    failed = verify_frozen_artifacts(
        baseline_manifest=lock_path,
        expected_file_count=int(base["file_count"]),
        expected_aggregate_sha256=str(base["aggregate_sha256"]),
    )
    assert failed["passed"] is False
    assert failed["changed_namespaces"] == ["e25a_terminal_gate"]


def test_old_e00_e21_manifest_cannot_stand_in_for_completed_evidence(tmp_path: Path) -> None:
    old = tmp_path / "old.json"
    old.write_text(
        json.dumps(
            {
                "scope": "immutable_E00_through_E21_artifacts",
                "artifact_root": str(tmp_path),
                "file_count": 0,
                "aggregate_sha256": "0" * 64,
                "files": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(FrozenInvarianceError, match="E00 through E25"):
        verify_frozen_artifacts(
            baseline_manifest=old,
            expected_file_count=1,
            expected_aggregate_sha256="1" * 64,
        )


def _historical_receipt_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, dict[str, Any], dict[str, Any], Path]:
    repo = tmp_path / "live_repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "catena-test@example.invalid")
    _git(repo, "config", "user.name", "CATENA Test")
    base_file = repo / "base.txt"
    base_file.write_text("frozen base\n", encoding="utf-8")
    _git(repo, "add", "base.txt")
    _git(repo, "commit", "-m", "frozen base")
    expected_head = _git(repo, "rev-parse", "HEAD")
    base_bytes = base_file.read_bytes()
    base_aggregate = _row_aggregate(
        [
            {
                "path": "base.txt",
                "bytes": len(base_bytes),
                "sha256": hashlib.sha256(base_bytes).hexdigest(),
            }
        ]
    )
    completed_lock, artifact_root = _write_completed_lock(tmp_path)
    base_lock = json.loads(
        (completed_lock.parent / "base.json").read_text(encoding="utf-8")
    )
    data_lock: dict[str, Any] = {
        "repository": {
            "live_repo": str(repo.resolve()),
            "expected_live_head": expected_head,
            "pre_e26_source_file_count": 1,
            "pre_e26_source_aggregate_sha256": base_aggregate,
            "frozen_artifact_file_count": int(base_lock["file_count"]),
            "frozen_artifact_aggregate_sha256": str(base_lock["aggregate_sha256"]),
        }
    }
    historical = build_frozen_invariance_receipt(
        data_lock=data_lock,
        baseline_manifest=completed_lock,
    )
    assert historical["passed"] is True
    (repo / "stage3d.txt").write_text("additive descendant\n", encoding="utf-8")
    _git(repo, "add", "stage3d.txt")
    _git(repo, "commit", "-m", "additive stage3d")
    return repo, base_file, data_lock, historical, artifact_root


def test_historical_receipt_allows_only_clean_additive_descendant_head(
    tmp_path: Path,
) -> None:
    repo, _base_file, data_lock, historical, _artifact_root = (
        _historical_receipt_fixture(tmp_path)
    )
    refreshed = validate_historical_frozen_invariance_receipt(
        historical,
        data_lock=data_lock,
    )
    assert refreshed["passed"] is True
    assert refreshed["live_repository"]["observed_head"] == _git(
        repo, "rev-parse", "HEAD"
    )
    assert (
        refreshed["live_repository"]["observed_head"]
        != historical["live_repository"]["observed_head"]
    )


def test_historical_receipt_blocks_dirty_or_changed_frozen_base(
    tmp_path: Path,
) -> None:
    repo, base_file, data_lock, historical, _artifact_root = (
        _historical_receipt_fixture(tmp_path)
    )
    additive = repo / "stage3d.txt"
    additive.write_text("dirty descendant\n", encoding="utf-8")
    with pytest.raises(FrozenInvarianceError):
        validate_historical_frozen_invariance_receipt(
            historical,
            data_lock=data_lock,
        )
    additive.write_text("additive descendant\n", encoding="utf-8")
    base_file.write_text("mutated frozen base\n", encoding="utf-8")
    _git(repo, "add", "base.txt")
    _git(repo, "commit", "-m", "mutate frozen base")
    with pytest.raises(FrozenInvarianceError):
        validate_historical_frozen_invariance_receipt(
            historical,
            data_lock=data_lock,
        )


def test_historical_receipt_blocks_frozen_artifact_drift(tmp_path: Path) -> None:
    _repo, _base_file, data_lock, historical, artifact_root = (
        _historical_receipt_fixture(tmp_path)
    )
    (artifact_root / "e25a_terminal_gate" / "report.json").write_text(
        "mutated frozen e25\n",
        encoding="utf-8",
    )
    with pytest.raises(FrozenInvarianceError):
        validate_historical_frozen_invariance_receipt(
            historical,
            data_lock=data_lock,
        )


def test_historical_receipt_blocks_tampering_before_live_reaudit(tmp_path: Path) -> None:
    _repo, _base_file, data_lock, historical, _artifact_root = (
        _historical_receipt_fixture(tmp_path)
    )
    tampered = deepcopy(historical)
    tampered["frozen_artifacts"]["expected_file_count"] += 1
    with pytest.raises(FrozenInvarianceError, match="canonical SHA-256"):
        validate_historical_frozen_invariance_receipt(
            tampered,
            data_lock=data_lock,
        )
