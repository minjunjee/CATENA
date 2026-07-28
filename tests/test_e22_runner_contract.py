from __future__ import annotations

import json
from pathlib import Path

import pytest

from catena.core.config import load_config
from catena.core.io import file_sha256, write_json
from catena.post_e21.contracts import (
    PostE21ContractError,
    validate_protocol_lock,
)
from catena.post_e21.locality_protocol import (
    load_parent_threshold_contract,
    validate_parent_binding,
    validate_selection_run_dependency,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
E22A_LOCK = REPO_ROOT / "docs/E22A_LOCALITY_METHOD_SELECTION_PROTOCOL_LOCK.json"
E22B_LOCK = REPO_ROOT / "docs/E22B_ACTIVE_PATH_LOCALITY_PROTOCOL_LOCK.json"


def test_static_phase_locks_bind_configs_parent_and_each_other() -> None:
    parent = load_parent_threshold_contract(repo_root=REPO_ROOT)
    e22a = validate_protocol_lock(
        lock_path=E22A_LOCK,
        config_path=REPO_ROOT / "configs/e22a_locality_method_selection.yaml",
        experiment_id="e22a_locality_method_selection",
        repo_root=REPO_ROOT,
    )
    validate_parent_binding(snapshot=e22a, parent=parent)
    e22b = validate_protocol_lock(
        lock_path=E22B_LOCK,
        config_path=REPO_ROOT / "configs/e22b_active_path_locality.yaml",
        experiment_id="e22b_active_path_locality",
        repo_root=REPO_ROOT,
    )
    validate_parent_binding(snapshot=e22b, parent=parent)
    assert e22b.payload["parent_e22a_static_lock_sha256"] == e22a.sha256
    assert e22a.payload["main_execution_started"] is False
    assert e22b.payload["main_execution_started"] is False


def test_registered_seeds_and_grid_are_fresh_and_exact() -> None:
    e22a = load_config(REPO_ROOT / "configs/e22a_locality_method_selection.yaml")
    e22b = load_config(REPO_ROOT / "configs/e22b_active_path_locality.yaml")
    dev = [int(value) for value in e22a["development_seeds"]]
    confirm = [int(value) for value in e22b["confirmatory_seeds"]]
    assert len(dev) == len(set(dev)) == 3
    assert len(confirm) == len(set(confirm)) == 8
    assert not set(dev) & set(confirm)
    assert e22b["evaluation"]["updates"] == [1, 4, 8]
    assert e22b["evaluation"]["gap_events"] == [0, 128, 512, 2048]
    assert e22b["selection_contract"]["baseline_method_id"] == "mean_retention"


def _fake_selection_run(
    tmp_path: Path,
    *,
    run_mode: str,
    selection_status: str,
    unlock: bool,
) -> Path:
    run = tmp_path / "20260728T000000.000000Z"
    run.mkdir()
    (run / "raw_metrics.jsonl").write_text("{}\n", encoding="utf-8")
    (run / "seed_metrics.jsonl").write_text("{}\n", encoding="utf-8")
    (run / "selection_scores.jsonl").write_text("{}\n", encoding="utf-8")
    (run / "active_cell_metrics.jsonl").write_text("{}\n", encoding="utf-8")
    parent = load_parent_threshold_contract(repo_root=REPO_ROOT)
    e22a_lock_sha = file_sha256(E22A_LOCK)
    selection = {
        "schema_version": 1,
        "experiment_id": "e22a_locality_method_selection",
        "run_id": run.name,
        "run_mode": run_mode,
        "selection_status": selection_status,
        "selected_method": {
            "method_id": "cvar_010",
            "objective": "cvar",
            "selection_eligible": True,
            "baseline": False,
            "tail_fraction": 0.1,
            "normalized_temperature": None,
            "active_fraction": None,
        },
        "baseline_method": {
            "method_id": "mean_retention",
            "objective": "mean",
            "selection_eligible": False,
            "baseline": True,
            "tail_fraction": None,
            "normalized_temperature": None,
            "active_fraction": None,
        },
        "parent_e21_lock_sha256": parent.sha256,
        "e22a_protocol_lock_sha256": e22a_lock_sha,
        "raw_metrics_sha256": file_sha256(run / "raw_metrics.jsonl"),
        "seed_metrics_sha256": file_sha256(run / "seed_metrics.jsonl"),
        "selection_scores_sha256": file_sha256(run / "selection_scores.jsonl"),
        "active_cell_metrics_sha256": file_sha256(run / "active_cell_metrics.jsonl"),
        "main_confirmatory_unlock": unlock,
    }
    write_json(run / "selection_lock.json", selection)
    report = {
        "status": "PASS",
        "run_mode": run_mode,
        "artifacts": {"selection_lock": {"sha256": file_sha256(run / "selection_lock.json")}},
    }
    write_json(run / "report.json", report)
    manifest = {
        "experiment_id": "e22a_locality_method_selection",
        "run_id": run.name,
        "run_mode": run_mode,
        "report_sha256": file_sha256(run / "report.json"),
    }
    write_json(run / "run_manifest.json", manifest)
    return run


def test_dry_selection_cannot_unlock_confirmatory_main(tmp_path: Path) -> None:
    parent = load_parent_threshold_contract(repo_root=REPO_ROOT)
    run = _fake_selection_run(
        tmp_path,
        run_mode="DRY_RUN",
        selection_status="DRY_RUN_SELECTED_NON_EVIDENCE",
        unlock=False,
    )
    dry = validate_selection_run_dependency(
        selection_run=run,
        parent=parent,
        expected_protocol_lock_sha256=file_sha256(E22A_LOCK),
        dry_run=True,
    )
    assert dry["run_mode"] == "DRY_RUN"
    with pytest.raises(PostE21ContractError, match="identity/status"):
        validate_selection_run_dependency(
            selection_run=run,
            parent=parent,
            expected_protocol_lock_sha256=file_sha256(E22A_LOCK),
            dry_run=False,
        )


def test_generic_mock_remains_reference_only() -> None:
    mock = REPO_ROOT / "mocks/post_e21_packet/experiments/e22_active_path_locality.py"
    assert mock.is_file()
    assert "Contract mock only" in mock.read_text(encoding="utf-8")
    assert not (REPO_ROOT / "experiments/e22_active_path_locality.py").exists()


def test_protocol_lock_json_is_plain_data() -> None:
    for path in (E22A_LOCK, E22B_LOCK):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["schema_version"] == 1
        assert payload["protocol_frozen_before_main"] is True
        assert payload["main_execution_started"] is False
        assert payload["files"]
