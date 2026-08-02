from __future__ import annotations

import json
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

import jsonschema
import pytest
import yaml

from catena.core.provenance_v61 import (
    sha256_canonical_json,
    sha256_file,
    write_json_strict,
)
from catena.lm.stage3c_data_lock import (
    Stage3CDataLockError,
    build_stage3c_data_lock,
    validate_stage3c_data_lock,
)


def _parent() -> dict[str, Any]:
    return {
        "schema_version": "catena-e26-data-lock-v1",
        "status": "PROSPECTIVE_PRE_E26A_LOCK",
        "scientific_evidence": False,
        "claim_ceiling": "PROTOCOL_IDENTIFIABILITY_ONLY",
        "repository": {
            "live_repo": "/frozen/live",
            "e26_worktree": "/frozen/e26",
            "expected_live_head": "a" * 40,
        },
        "source": {"revision": "locked"},
        "content_partition": {
            "near_duplicate": {
                "flagged_pair_policy": "FAIL_PENDING_MANUAL_AUDIT",
                "estimated_jaccard_flag_threshold": 0.8,
            }
        },
        "tokenizer": {"vocab_size": 16_384},
        "data_root": "/frozen/data-v1",
        "memmaps": {"general_train_tokens_min": 400_000_000},
        "transaction": {"schedule_seed": 260_026},
        "tooling": {"environment": "ISOLATED_FROM_CATENA_V6"},
        "resource_policy": {"max_main_wall_clock_hours": 168},
        "stop_policy": {
            "execute_scientific_e26a": False,
            "execute_e26b": False,
            "execute_e26c_or_later": False,
        },
    }


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")


def _git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=root, text=True).strip()


class _Readiness:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def as_dict(self) -> dict[str, Any]:
        return dict(self.payload)


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, Any], dict[str, Path]]:
    parent_path = tmp_path / "e26_data_lock_v1.yaml"
    _write_yaml(parent_path, _parent())
    protocol_path = tmp_path / "e26_data_lock_v2_zero_tolerance.yaml"
    _write_yaml(
        protocol_path,
        {
            "schema_version": "catena-e26-data-lock-v2-zero-tolerance",
            "original_inputs": {
                "data_lock": {
                    "path": str(parent_path),
                    "sha256": sha256_file(parent_path),
                }
            },
        },
    )
    repair_path = tmp_path / "repair_receipt.json"
    write_json_strict(
        repair_path,
        {
            "disposition": "ZERO_PROTECTED_TRAIN_FLAGS",
            "policy": "EXCLUDE_ALL_FLAGGED_TRAIN_DOCUMENTS",
            "human_labels_used": False,
            "gpu_preflight_started": False,
            "scientific_e26a_started": False,
            "scientific_main_started": False,
            "repair_receipt_sha256": "1" * 64,
        },
    )
    source_path = tmp_path / "repair_source_receipt.json"
    write_json_strict(source_path, {"source": "fixture"})
    memmap_path = tmp_path / "general_memmaps_receipt.json"
    schedule_path = tmp_path / "paired_schedule_manifest.json"
    transaction_path = tmp_path / "transaction_manifest.json"
    corpus_path = tmp_path / "general_train.corpus_manifest.json"
    for path in (memmap_path, schedule_path, transaction_path, corpus_path):
        write_json_strict(path, {"fixture": path.name})
    readiness_path = tmp_path / "scientific_data_readiness_v3.json"
    readiness: dict[str, Any] = {
        "scientific_main_input_eligible": True,
        "repair_disposition": "ZERO_PROTECTED_TRAIN_FLAGS",
        "near_duplicate_flagged_pair_count": 0,
        "human_labels_used": False,
        "main_test_opened": False,
        "gpu_preflight_started": False,
        "scientific_e26a_started": False,
        "protocol_lock": {
            "path": str(protocol_path),
            "sha256": sha256_file(protocol_path),
        },
        "repair_receipt": {
            "path": str(repair_path),
            "sha256": sha256_file(repair_path),
        },
        "readiness_sha256": "2" * 64,
        "general_memmap_receipt": {
            "path": str(memmap_path),
            "sha256": sha256_file(memmap_path),
        },
        "general_corpora": {
            "general_train": {
                "corpus_manifest_path": str(corpus_path),
                "corpus_manifest_sha256": sha256_file(corpus_path),
            }
        },
        "schedule_manifest": {
            "path": str(schedule_path),
            "sha256": sha256_file(schedule_path),
        },
        "transaction_manifest": {
            "path": str(transaction_path),
            "sha256": sha256_file(transaction_path),
        },
    }
    write_json_strict(readiness_path, readiness)
    monkeypatch.setattr(
        "catena.lm.stage3c_data_lock.validate_zero_tolerance_data_bundle",
        lambda **kwargs: _Readiness(readiness),
    )
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "catena-test@example.invalid")
    _git(tmp_path, "config", "user.name", "CATENA Test")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "fixture")
    paths = {
        "parent": parent_path,
        "protocol": protocol_path,
        "repair": repair_path,
        "source": source_path,
        "readiness": readiness_path,
    }
    payload = build_stage3c_data_lock(
        parent_data_lock_path=parent_path,
        repair_protocol_path=protocol_path,
        repair_receipt_path=repair_path,
        repair_source_receipt_path=source_path,
        readiness_path=readiness_path,
        expected_readiness_sha256=sha256_file(readiness_path),
        stage3c_worktree=tmp_path,
    )
    return payload, paths


def _set_nested(payload: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


def test_stage3c_lock_validates_by_exact_reconstruction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload, paths = _fixture(tmp_path, monkeypatch)
    validate_stage3c_data_lock(payload, parent_data_lock_path=paths["parent"])
    assert payload["repository"]["e26_worktree"] == "/frozen/e26"
    assert payload["stage3c_execution"]["worktree"] == str(tmp_path)
    assert payload["claim_ceiling"] == "PROTOCOL_IDENTIFIABILITY_ONLY"


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("content_partition", "near_duplicate", "estimated_jaccard_flag_threshold"), 0.5),
        (("memmaps", "general_train_tokens_min"), 1),
        (("tooling", "environment"), "MUTATED"),
        (("resource_policy", "max_main_wall_clock_hours"), 169),
        (("stop_policy", "execute_scientific_e26a"), True),
        (("claim_ceiling",), "UNBOUNDED"),
        (("final_repaired_data", "schedule_manifest", "sha256"), "3" * 64),
        (("final_repaired_data", "human_labels_used"), True),
        (("final_repaired_data", "final_protected_train_flag_count"), 99),
    ],
)
def test_stage3c_lock_rejects_rehashed_semantic_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    path: tuple[str, ...],
    value: Any,
) -> None:
    payload, _ = _fixture(tmp_path, monkeypatch)
    mutated = deepcopy(payload)
    _set_nested(mutated, path, value)
    mutated.pop("lock_sha256")
    mutated["lock_sha256"] = sha256_canonical_json(mutated)
    with pytest.raises(Stage3CDataLockError):
        validate_stage3c_data_lock(mutated)


def test_stage3c_build_rejects_parent_not_bound_by_repair_protocol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, paths = _fixture(tmp_path, monkeypatch)
    alternate = tmp_path / "alternate_v1.yaml"
    changed = _parent()
    changed["resource_policy"]["max_main_wall_clock_hours"] = 999
    _write_yaml(alternate, changed)
    with pytest.raises(Stage3CDataLockError, match="exact V1 data lock"):
        build_stage3c_data_lock(
            parent_data_lock_path=alternate,
            repair_protocol_path=paths["protocol"],
            repair_receipt_path=paths["repair"],
            repair_source_receipt_path=paths["source"],
            readiness_path=paths["readiness"],
            expected_readiness_sha256=sha256_file(paths["readiness"]),
            stage3c_worktree=tmp_path,
        )


def test_stage3c_lock_conforms_to_json_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload, _ = _fixture(tmp_path, monkeypatch)
    root = Path(__file__).resolve().parents[1]
    schema = json.loads(
        (root / "schemas/v8_1/e26_stage3c_data_lock.schema.json").read_text(encoding="utf-8")
    )
    jsonschema.Draft202012Validator(schema).validate(payload)
