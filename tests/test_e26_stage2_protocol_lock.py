from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from catena.core.provenance_v61 import sha256_file, write_json_strict
from catena.lm.backend_lock import backend_candidate_lock_payload
from catena.lm.stage2_protocol_lock import (
    Stage2ProtocolInputs,
    build_stage2_protocol_lock,
)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def _fixture(tmp_path: Path) -> tuple[Path, Stage2ProtocolInputs]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "fixture@example.invalid")
    _git(repo, "config", "user.name", "fixture")
    config = {
        "schema_version": "catena-v8.1",
        "experiment": "e26a_operator_data_gate",
        "stage": "operator_data_numerical_throughput_gate",
        "registered_dispositions": ["GO_E26B", "NO_GO_DATA"],
        "throughput": {"deadline_fraction_max": 0.70},
        "model_candidates": [{"id": "candidate", "d_model": 32}],
        "data": {
            "transaction_generator_version": "v8.1",
            "operations": [
                "PRESERVE",
                "ADD",
                "INVALIDATE",
                "SUPERSEDE",
                "ADD_EXCEPTION",
            ],
        },
        "gate_population": {
            "generation_seed": 260001,
            "namespace": "fixture_validation_population",
            "splits": ["validation", "main_test", "heldout_domain"],
            "domains": [
                "access_control",
                "api_configuration",
                "workflow",
                "versioned_preference",
            ],
            "items_per_operation_per_split": 4,
            "distractor_units": 1,
            "population_hash_required": True,
        },
    }
    paths: dict[str, Path] = {}
    for name in Stage2ProtocolInputs.__dataclass_fields__:
        if name == "backend_candidate_lock":
            continue
        suffix = ".yaml" if name in {"config", "calibration_config", "data_lock"} else ".json"
        path = repo / f"{name}{suffix}"
        if name == "config":
            path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
        elif suffix == ".yaml":
            path.write_text("schema_version: catena-v8.1\n", encoding="utf-8")
        else:
            path.write_text('{"schema_version":"catena-v8.1"}\n', encoding="utf-8")
        paths[name] = path
    (repo / "runner.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "fixture")
    candidate_lock_path = tmp_path / "backend_candidate_lock.json"
    write_json_strict(
        candidate_lock_path,
        backend_candidate_lock_payload(
            repo_root=repo,
            config_path=paths["config"],
            candidates=config["model_candidates"],
        ),
    )
    paths["backend_candidate_lock"] = candidate_lock_path
    return repo, Stage2ProtocolInputs(**paths)


def test_stage2_protocol_lock_is_deterministic_and_acyclic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "catena.lm.stage2_protocol_lock.validate_frozen_invariance_receipt",
        lambda payload, *, data_lock: dict(payload),
    )
    repo, inputs = _fixture(tmp_path)
    first = build_stage2_protocol_lock(
        repo_root=repo,
        output_dir=tmp_path / "lock_a",
        lock_utc="2026-07-31T12:00:00Z",
        inputs=inputs,
    )
    second = build_stage2_protocol_lock(
        repo_root=repo,
        output_dir=tmp_path / "lock_b",
        lock_utc="2026-07-31T12:00:00+00:00",
        inputs=inputs,
    )
    first_protocol = Path(first["protocol_lock"]["path"])
    second_protocol = Path(second["protocol_lock"]["path"])
    assert first_protocol.read_bytes() == second_protocol.read_bytes()
    protocol = json.loads(first_protocol.read_text(encoding="utf-8"))
    execution_inputs = protocol["execution_inputs"]
    assert "source_tree_sha256" in execution_inputs
    assert "backend_candidate_lock_sha256" in execution_inputs
    assert "validation_population_lock_sha256" in execution_inputs
    assert "backend_manifest_sha256" not in execution_inputs
    assert "protocol_lock_sha256" not in execution_inputs
    assert "numerical_audit_sha256" not in execution_inputs
    assert "restart_audit_sha256" not in execution_inputs
    assert protocol["main_test_access_count"] == 0
    population_path = Path(first["validation_population_lock"]["path"])
    population = json.loads(population_path.read_text(encoding="utf-8"))
    assert population["manifest_type"] == "E26A_VALIDATION_POPULATION_LOCK"
    assert population["episode_count"] == 20
    assert population["main_test_opened"] is False
    assert population["main_test_access_count"] == 0
    assert population["heldout_domain_opened"] is False
    assert first["protocol_lock"]["sha256"] == sha256_file(first_protocol)
    assert Path(first["receipt_path"]).is_file()


def test_stage2_protocol_lock_cli_has_no_scientific_launcher() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "tools" / "prepare_e26a_stage2_protocol_lock.py").read_text(encoding="utf-8")
    assert "run_scientific_e26a" not in source
    assert "e26b" not in source.lower()
    assert "e26c" not in source.lower()


def test_stage2_protocol_lock_validates_stage3c_data_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "catena.lm.stage2_protocol_lock.validate_frozen_invariance_receipt",
        lambda payload, *, data_lock: dict(payload),
    )
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        "catena.lm.stage2_protocol_lock.validate_stage3c_data_lock",
        lambda payload: calls.append(dict(payload)),
    )
    repo, inputs = _fixture(tmp_path)
    final_lock = tmp_path / "stage3c_data_lock.json"
    write_json_strict(
        final_lock,
        {
            "schema_version": "catena-e26-data-lock-v3-final-preflight",
            "stage3c_execution": {"worktree": str(repo)},
        },
    )
    build_stage2_protocol_lock(
        repo_root=repo,
        output_dir=tmp_path / "stage3c_protocol_lock",
        lock_utc="2026-08-02T12:00:00Z",
        inputs=replace(inputs, data_lock=final_lock),
    )
    assert calls == [
        {
            "schema_version": "catena-e26-data-lock-v3-final-preflight",
            "stage3c_execution": {"worktree": str(repo)},
        }
    ]
