from __future__ import annotations

import json
from pathlib import Path

import pytest

from catena.core.io import file_sha256, write_json
from catena.post_e21.contracts import (
    PostE21ContractError,
    combined_checkpoint_sha256,
    copy_protocol_snapshot,
    validate_protocol_lock,
    write_required_rows,
)


def _locked_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    config = tmp_path / "configs" / "e99.yaml"
    config.parent.mkdir()
    config.write_text("experiment_id: e99\n", encoding="utf-8")
    source = tmp_path / "experiments" / "e99.py"
    source.parent.mkdir()
    source.write_text("VALUE = 1\n", encoding="utf-8")
    lock = tmp_path / "docs" / "E99_LOCK.json"
    lock.parent.mkdir()
    write_json(
        lock,
        {
            "schema_version": 1,
            "experiment_id": "e99",
            "protocol_frozen_before_main": True,
            "main_execution_started": False,
            "files": {
                "configs/e99.yaml": file_sha256(config),
                "experiments/e99.py": file_sha256(source),
            },
        },
    )
    return config, source, lock


def test_protocol_snapshot_is_exact_and_detects_tampering(tmp_path: Path) -> None:
    config, source, lock = _locked_fixture(tmp_path)
    snapshot = validate_protocol_lock(
        lock_path=lock,
        config_path=config,
        experiment_id="e99",
        repo_root=tmp_path,
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    copied = copy_protocol_snapshot(snapshot=snapshot, run_dir=run_dir)
    assert copied.read_bytes() == lock.read_bytes()

    source.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(PostE21ContractError, match="Locked file changed"):
        validate_protocol_lock(
            lock_path=lock,
            config_path=config,
            experiment_id="e99",
            repo_root=tmp_path,
        )


def test_required_jsonl_layers_are_written_even_when_empty(tmp_path: Path) -> None:
    descriptors = write_required_rows(run_dir=tmp_path, raw_rows=[], seed_rows=[])
    assert descriptors["raw"]["rows"] == 0
    assert descriptors["seed"]["rows"] == 0
    assert (tmp_path / "raw_metrics.jsonl").read_text(encoding="utf-8") == ""
    assert (tmp_path / "seed_metrics.jsonl").read_text(encoding="utf-8") == ""


def test_checkpoint_hash_contract() -> None:
    assert combined_checkpoint_sha256({}) is None
    assert combined_checkpoint_sha256({"seed": "a" * 64}) is not None
    with pytest.raises(PostE21ContractError, match="Invalid checkpoint"):
        combined_checkpoint_sha256({"seed": "mock"})


def test_lock_requires_prospective_flag(tmp_path: Path) -> None:
    config, _, lock = _locked_fixture(tmp_path)
    payload = json.loads(lock.read_text(encoding="utf-8"))
    payload["main_execution_started"] = True
    write_json(lock, payload)
    with pytest.raises(PostE21ContractError, match="precede main"):
        validate_protocol_lock(
            lock_path=lock,
            config_path=config,
            experiment_id="e99",
            repo_root=tmp_path,
        )
