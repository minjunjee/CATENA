from __future__ import annotations

import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from catena.core.config import load_config
from catena.core.provenance_v61 import sha256_canonical_json
from catena.post_e21.contracts import PostE21ContractError
from catena.post_e21.e23b_sharded_execution import (
    _validate_device_bindings,
    aggregate_sharded_execution,
    balanced_seed_partitions,
    normalized_equivalence_payload,
    prepare_sharded_execution,
    run_shard_worker,
    validate_equivalence_report,
    validate_source_lock_tag,
)
from catena.post_e21.product_poset_eval import expected_grid_size
from catena.post_e21.product_poset_runner import product_poset_runtime

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/e23b_product_poset_confirmatory.yaml"
EQUIVALENCE_CHECK_KEYS = (
    "registered_four_seed_subset_exact",
    "raw_row_count_exact",
    "canonical_scientific_raw_rows_exact",
    "canonical_scientific_training_rows_exact",
    "checkpoint_state_hashes_exact",
    "seed_statistics_exact",
    "cell_statistics_exact",
    "assessment_exact",
    "runtime_contract_exact",
)
COMPARISON_EXCLUSIONS = [
    "examples_per_second",
    "peak_memory_bytes",
    "checkpoint absolute path",
    "checkpoint container file SHA-256",
]


def _valid_equivalence_fixture(
    *,
    source: dict[str, object],
    source_lock: dict[str, object],
    parent_sha: str,
    amendment_sha: str,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    config = load_config(CONFIG)
    locality_method = {
        "method_id": "mean_retention",
        "objective": "mean",
        "selection_eligible": True,
    }
    dependency: dict[str, object] = {
        "overall_execution_status": "PASS",
        "e22": {
            "report_sha256": "4" * 64,
            "boundary_mode": "capacity_only",
            "locality_method": locality_method,
            "locality_risk_scale": 0.0005,
        },
    }
    seeds = [int(value) for value in config["seeds"][:4]]
    serial_config = deepcopy(config)
    serial_config["seeds"] = list(seeds)
    serial_config["dry_run"]["seed_count"] = len(seeds)
    runtime = product_poset_runtime(serial_config, dry_run=True)
    expected_rows = expected_grid_size(
        seeds=seeds,
        intensities=runtime["intensities"],
        updates=runtime["updates"],
        gap_events=runtime["gap_events"],
    )
    report: dict[str, object] = {
        "schema_version": 1,
        "experiment_id": "e23b_product_poset_confirmatory",
        "execution_experiment_id": "e23b_product_poset_confirmatory_sharded_execution",
        "status": "PASS",
        "run_mode": "CPU_SERIAL_VS_SHARD_EQUIVALENCE",
        "scientific_evidence": False,
        "claim_eligible": False,
        "source_fingerprint": source,
        "source_lock": source_lock,
        "amendment_lock_sha256": amendment_sha,
        "base_protocol_lock_sha256": parent_sha,
        "config_sha256": "5" * 64,
        "dependency_sha256": sha256_canonical_json(dependency),
        "dependency": dependency,
        "boundary_mode": "capacity_only",
        "locality_method": locality_method,
        "locality_risk_scale": 0.0005,
        "seeds": seeds,
        "runtime_config_sha256": sha256_canonical_json(runtime),
        "serial_rows": expected_rows,
        "sharded_rows": expected_rows,
        "checks": {key: True for key in EQUIVALENCE_CHECK_KEYS},
        "comparison_exclusions": list(COMPARISON_EXCLUSIONS),
        "checkpoint_state_hash_comparison": "exact",
        "scientific_metric_comparison": "exact",
    }
    return config, dependency, report


def test_balanced_seed_partitions_preserve_registered_order() -> None:
    seeds = (2401, 2411, 2423, 2437, 2441, 2459, 2473, 2477)
    assert balanced_seed_partitions(seeds, 4) == (
        (2401, 2411),
        (2423, 2437),
        (2441, 2459),
        (2473, 2477),
    )
    assert balanced_seed_partitions(seeds, 3) == (
        (2401, 2411, 2423),
        (2437, 2441, 2459),
        (2473, 2477),
    )
    assert balanced_seed_partitions(seeds, 20) == tuple((seed,) for seed in seeds)
    with pytest.raises(ValueError):
        balanced_seed_partitions((), 4)
    with pytest.raises(ValueError):
        balanced_seed_partitions((1, 1), 2)
    with pytest.raises(ValueError):
        balanced_seed_partitions((1,), 0)


def test_gpu_bindings_require_unique_homogeneous_physical_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bindings = [
        {
            "device_type": "cuda",
            "physical_index": index,
            "uuid": f"GPU-{index}",
            "name": "MATCHED_GPU",
            "compute_capability": "12.0",
        }
        for index in range(4)
    ]
    monkeypatch.setattr(
        "catena.post_e21.e23b_sharded_execution._query_physical_gpu",
        lambda index: dict(bindings[index]),
    )
    assert _validate_device_bindings(bindings, dry_run=False) == bindings
    duplicate = [dict(binding) for binding in bindings]
    duplicate[1]["uuid"] = duplicate[0]["uuid"]
    with pytest.raises(PostE21ContractError, match="not unique"):
        _validate_device_bindings(duplicate, dry_run=False)
    heterogeneous = [dict(binding) for binding in bindings]
    heterogeneous[3]["compute_capability"] = "11.0"
    with pytest.raises(PostE21ContractError, match="not homogeneous"):
        _validate_device_bindings(heterogeneous, dry_run=False)


def test_annotated_source_lock_and_equivalence_binding_fail_closed(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    (repo / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "source.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "lock source"], cwd=repo, check=True)
    parent_sha = "1" * 64
    amendment_sha = "2" * 64
    tag = "e23b-test-lock"
    message = (
        "E23b test source lock\n"
        f"E23B_BASE_PROTOCOL_LOCK_SHA256={parent_sha}\n"
        f"E23B_SHARDED_EXECUTION_AMENDMENT_LOCK_SHA256={amendment_sha}"
    )
    subprocess.run(["git", "tag", "-a", tag, "-m", message], cwd=repo, check=True)
    source_lock = validate_source_lock_tag(
        repo_root=repo,
        tag=tag,
        parent_protocol_sha256=parent_sha,
        amendment_sha256=amendment_sha,
    )
    assert source_lock["dirty_status"] == "clean"

    source = {"sha256": "3" * 64, "files": 7}
    config, dependency, report = _valid_equivalence_fixture(
        source=source,
        source_lock=source_lock,
        parent_sha=parent_sha,
        amendment_sha=amendment_sha,
    )
    report_path = tmp_path / "equivalence.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    validated = validate_equivalence_report(
        path=report_path,
        source=source,
        source_lock=source_lock,
        amendment_sha256=amendment_sha,
        parent_protocol_sha256=parent_sha,
        config_sha256="5" * 64,
        config=config,
        dependency=dependency,
    )
    assert validated["status"] == "PASS"

    (repo / "source.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(PostE21ContractError, match="clean"):
        validate_source_lock_tag(
            repo_root=repo,
            tag=tag,
            parent_protocol_sha256=parent_sha,
            amendment_sha256=amendment_sha,
        )
    report["dependency"] = {"overall_execution_status": "FAIL"}
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(PostE21ContractError, match="binding mismatch"):
        validate_equivalence_report(
            path=report_path,
            source=source,
            source_lock=source_lock,
            amendment_sha256=amendment_sha,
            parent_protocol_sha256=parent_sha,
            config_sha256="5" * 64,
            config=config,
            dependency=dependency,
        )


def test_equivalence_requires_exact_nine_true_checks(tmp_path: Path) -> None:
    source: dict[str, object] = {"sha256": "3" * 64, "files": 7}
    source_lock: dict[str, object] = {
        "tag": "locked",
        "git_commit": "4" * 40,
        "dirty_status": "clean",
    }
    parent_sha = "1" * 64
    amendment_sha = "2" * 64
    for mutation, message in (
        ("missing", "check-key set is not exact"),
        ("extra", "check-key set is not exact"),
        ("false", "did not fully pass"),
    ):
        config, dependency, report = _valid_equivalence_fixture(
            source=source,
            source_lock=source_lock,
            parent_sha=parent_sha,
            amendment_sha=amendment_sha,
        )
        checks = report["checks"]
        assert isinstance(checks, dict)
        if mutation == "missing":
            checks.pop(EQUIVALENCE_CHECK_KEYS[0])
        elif mutation == "extra":
            checks["unregistered_extra_check"] = True
        elif mutation == "false":
            checks[EQUIVALENCE_CHECK_KEYS[0]] = False
        else:
            raise AssertionError(f"unhandled test mutation: {mutation}")

        report_path = tmp_path / f"equivalence_{mutation}.json"
        report_path.write_text(json.dumps(report), encoding="utf-8")
        with pytest.raises(PostE21ContractError, match=message):
            validate_equivalence_report(
                path=report_path,
                source=source,
                source_lock=source_lock,
                amendment_sha256=amendment_sha,
                parent_protocol_sha256=parent_sha,
                config_sha256="5" * 64,
                config=config,
                dependency=dependency,
            )


def test_equivalence_binds_fixed_fields_to_config_and_dependency(tmp_path: Path) -> None:
    wrong_fields: tuple[tuple[str, object], ...] = (
        ("seeds", [2401, 2411, 2423, 9999]),
        ("comparison_exclusions", list(reversed(COMPARISON_EXCLUSIONS))),
        ("checkpoint_state_hash_comparison", "approximate"),
        ("scientific_metric_comparison", "tolerant"),
        ("boundary_mode", "safe_minimality"),
        (
            "locality_method",
            {
                "method_id": "cvar_q010",
                "objective": "cvar",
                "selection_eligible": True,
            },
        ),
        ("locality_risk_scale", 0.0004),
        ("runtime_config_sha256", "0" * 64),
        ("serial_rows", 703),
        ("sharded_rows", 703),
    )
    source: dict[str, object] = {"sha256": "3" * 64, "files": 7}
    source_lock: dict[str, object] = {
        "tag": "locked",
        "git_commit": "4" * 40,
        "dirty_status": "clean",
    }
    parent_sha = "1" * 64
    amendment_sha = "2" * 64
    for field, wrong_value in wrong_fields:
        config, dependency, report = _valid_equivalence_fixture(
            source=source,
            source_lock=source_lock,
            parent_sha=parent_sha,
            amendment_sha=amendment_sha,
        )
        report[field] = wrong_value
        report_path = tmp_path / f"equivalence_wrong_{field}.json"
        report_path.write_text(json.dumps(report), encoding="utf-8")
        with pytest.raises(PostE21ContractError, match=f"binding mismatch: {field}"):
            validate_equivalence_report(
                path=report_path,
                source=source,
                source_lock=source_lock,
                amendment_sha256=amendment_sha,
                parent_protocol_sha256=parent_sha,
                config_sha256="5" * 64,
                config=config,
                dependency=dependency,
            )


def test_main_prepare_fails_closed_without_dependencies(tmp_path: Path) -> None:
    with pytest.raises(PostE21ContractError, match="BLOCKED_DEPENDENCY"):
        prepare_sharded_execution(
            repo_root=ROOT,
            config_path=CONFIG,
            artifact_root=tmp_path / "artifacts",
            e18_freeze=None,
            e23a_screen=None,
            e22b_run=None,
            shard_count=4,
            dry_run=False,
        )
    assert not (tmp_path / "artifacts" / "_e23b_product_poset_confirmatory_shards").exists()


def test_aggregate_refuses_incomplete_shard_set(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    workspace = prepare_sharded_execution(
        repo_root=ROOT,
        config_path=CONFIG,
        artifact_root=artifact_root,
        e18_freeze=None,
        e23a_screen=None,
        e22b_run=None,
        shard_count=4,
        dry_run=True,
    )
    with pytest.raises(FileNotFoundError):
        aggregate_sharded_execution(
            workspace=workspace,
            artifact_root=artifact_root,
        )
    assert not (artifact_root / "e23b_product_poset_confirmatory" / "latest.json").exists()


def _run(command: list[str], *, env: dict[str, str] | None = None) -> str:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return completed.stdout.strip().splitlines()[-1]


def test_e23b_sharded_cpu_dry_run_matches_serial(tmp_path: Path) -> None:
    serial_root = tmp_path / "serial"
    sharded_root = tmp_path / "sharded"
    env = dict(os.environ)
    inherited_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{ROOT / 'src'}:{ROOT}{':' + inherited_pythonpath if inherited_pythonpath else ''}"
    )

    _run(
        [
            sys.executable,
            "-m",
            "experiments.e23b_product_poset_confirmatory",
            "--config",
            str(CONFIG),
            "--device",
            "cpu",
            "--artifact-root",
            str(serial_root),
            "--dry-run",
        ],
        env=env,
    )
    serial_pointer = json.loads(
        (serial_root / "e23b_product_poset_confirmatory" / "latest.json").read_text(
            encoding="utf-8"
        )
    )
    serial_run = Path(serial_pointer["run_dir"])

    workspace = Path(
        _run(
            [
                sys.executable,
                "-m",
                "experiments.e23b_product_poset_confirmatory_sharded",
                "prepare",
                "--config",
                str(CONFIG),
                "--artifact-root",
                str(sharded_root),
                "--shard-count",
                "1",
                "--dry-run",
            ],
            env=env,
        )
    )
    run_shard_worker(
        workspace=workspace,
        shard_id="shard_00",
        device=torch.device("cpu"),
    )
    sharded_run = aggregate_sharded_execution(
        workspace=workspace,
        artifact_root=sharded_root,
    )
    assert normalized_equivalence_payload(sharded_run) == normalized_equivalence_payload(serial_run)

    report = json.loads((sharded_run / "report.json").read_text(encoding="utf-8"))
    topology = report["execution_topology"]
    assert topology["mode"] == "REGISTERED_SEED_SHARDED_V1"
    assert topology["scientific_protocol_unchanged"] is True
    assert topology["validation"] == {
        "registered_seed_cover_exact": True,
        "raw_cartesian_grid_exact": True,
        "training_grid_exact": True,
        "checkpoint_hashes_verified": True,
        "dependency_provenance_verified_per_row": True,
        "paired_data_digest_verified": True,
        "physical_gpu_bindings_verified": True,
        "physical_gpu_bindings_unique_and_homogeneous": True,
        "nonfinite_rows": 0,
        "duplicate_rows": 0,
    }
    assert report["run_mode"] == "DRY_RUN"
    assert report["claim_eligible"] is False
    assert report["claim_gate"]["status"] == "DRY_RUN_ONLY"
    assert (workspace / "aggregate_receipt.json").is_file()
    with pytest.raises(FileExistsError):
        aggregate_sharded_execution(
            workspace=workspace,
            artifact_root=sharded_root,
        )


def test_worker_refuses_to_overwrite_completed_shard(tmp_path: Path) -> None:
    workspace = prepare_sharded_execution(
        repo_root=ROOT,
        config_path=CONFIG,
        artifact_root=tmp_path / "artifacts",
        e18_freeze=None,
        e23a_screen=None,
        e22b_run=None,
        shard_count=1,
        dry_run=True,
    )
    run_shard_worker(
        workspace=workspace,
        shard_id="shard_00",
        device=torch.device("cpu"),
    )
    with pytest.raises(FileExistsError):
        run_shard_worker(
            workspace=workspace,
            shard_id="shard_00",
            device=torch.device("cpu"),
        )


def test_four_shard_dry_aggregation_is_complete_and_self_contained(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "artifacts"
    workspace = prepare_sharded_execution(
        repo_root=ROOT,
        config_path=CONFIG,
        artifact_root=artifact_root,
        e18_freeze=None,
        e23a_screen=None,
        e22b_run=None,
        shard_count=4,
        dry_run=True,
    )
    prepared = json.loads((workspace / "prepared_execution.json").read_text(encoding="utf-8"))[
        "payload"
    ]
    assert [record["seeds"] for record in prepared["shard_plan"]] == [
        [2401],
        [2411],
        [2423],
        [2437],
    ]
    assert len({record["device_binding"]["uuid"] for record in prepared["shard_plan"]}) == 4
    for index in range(4):
        run_shard_worker(
            workspace=workspace,
            shard_id=f"shard_{index:02d}",
            device=torch.device("cpu"),
        )
    run_dir = aggregate_sharded_execution(
        workspace=workspace,
        artifact_root=artifact_root,
    )
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    assert report["artifacts"]["rows"]["raw"]["rows"] == 4 * 176
    assert report["artifacts"]["rows"]["seed"]["rows"] == 4
    assert report["artifacts"]["training_runs"]["rows"] == 4 * 16
    assert len(report["checkpoint_hashes"]) == 4 * 16
    assert report["artifacts"]["prepared_execution"]["payload_sha256"]
    assert len(report["artifacts"]["shard_manifests"]) == 4
    assert all(
        Path(descriptor["path"]).is_relative_to(run_dir)
        for descriptor in report["artifacts"]["shard_manifests"]
    )
    assert all(
        descriptor["device_binding"]["name"] == "CPU_DRY_RUN"
        for descriptor in report["execution_topology"]["shards"]
    )


def test_tampered_shard_is_rejected_before_canonical_artifact(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    workspace = prepare_sharded_execution(
        repo_root=ROOT,
        config_path=CONFIG,
        artifact_root=artifact_root,
        e18_freeze=None,
        e23a_screen=None,
        e22b_run=None,
        shard_count=1,
        dry_run=True,
    )
    shard = run_shard_worker(
        workspace=workspace,
        shard_id="shard_00",
        device=torch.device("cpu"),
    )
    with (shard / "product_poset_raw_metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(PostE21ContractError, match="artifact hash mismatch"):
        aggregate_sharded_execution(
            workspace=workspace,
            artifact_root=artifact_root,
        )
    assert not (artifact_root / "e23b_product_poset_confirmatory" / "latest.json").exists()


def test_concurrent_aggregate_creates_exactly_one_canonical_run(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    workspace = prepare_sharded_execution(
        repo_root=ROOT,
        config_path=CONFIG,
        artifact_root=artifact_root,
        e18_freeze=None,
        e23a_screen=None,
        e22b_run=None,
        shard_count=1,
        dry_run=True,
    )
    run_shard_worker(
        workspace=workspace,
        shard_id="shard_00",
        device=torch.device("cpu"),
    )
    command = [
        sys.executable,
        "-m",
        "experiments.e23b_product_poset_confirmatory_sharded",
        "aggregate",
        "--workspace",
        str(workspace),
        "--artifact-root",
        str(artifact_root),
        "--device",
        "cpu",
    ]
    env = dict(os.environ)
    inherited_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{ROOT / 'src'}:{ROOT}{':' + inherited_pythonpath if inherited_pythonpath else ''}"
    )
    processes = [
        subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(2)
    ]
    completed = [process.communicate(timeout=60) for process in processes]
    return_codes = sorted(process.returncode for process in processes)
    assert return_codes == [0, 1], completed
    experiment_root = artifact_root / "e23b_product_poset_confirmatory"
    run_dirs = [
        path for path in experiment_root.iterdir() if path.is_dir() and path.name != "latest.json"
    ]
    assert len(run_dirs) == 1
    assert (workspace / "aggregate_receipt.json").is_file()
