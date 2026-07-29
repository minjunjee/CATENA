from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import torch

from catena.core.config import load_config
from catena.core.io import file_sha256
from catena.core.provenance_v61 import read_json_object_strict, write_json_strict
from catena.post_e21.contracts import PostE21ContractError
from catena.post_e21.e22b_sharding import (
    AMENDMENT_ID,
    AMENDMENT_LOCK_RELATIVE,
    BASE_LOCK_RELATIVE,
    EQUIVALENCE_CHECK_KEYS,
    EQUIVALENCE_FIXED_FIELDS,
    MAIN_ACK_ENV,
    MAIN_ACK_VALUE,
    SCIENTIFIC_NO_CHANGE_FLAGS,
    SHARD_COUNT,
    _materialize_canonical_checkpoints,
    _read_plan,
    _state_hashes_for_run,
    _validate_devices,
    acquire_exclusive_launcher_lock,
    build_equivalence_validation_contract,
    canonicalize_rows,
    compare_serial_and_sharded_outputs,
    copy_equivalence_report_into_run,
    registered_equivalence_validation_contract,
    registered_seed_shards,
    validate_amendment_payload,
    validate_equivalence_report,
    validate_execution_amendment,
    validate_gpu_inventory,
    validate_locked_equivalence_record,
    validate_main_authorization_and_runtime,
    validate_registered_shard_plan,
)
from catena.post_e21.locality_data import method_by_id, parse_locality_methods
from catena.post_e21.locality_protocol import load_parent_threshold_contract
from catena.post_e21.locality_runner import (
    run_locality_method_grid,
    runtime_locality_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_amendment_binds_frozen_protocol_and_exact_seed_partition() -> None:
    amendment = validate_execution_amendment(repo_root=REPO_ROOT)
    config = load_config(REPO_ROOT / "configs/e22b_active_path_locality.yaml")
    seeds = [int(seed) for seed in config["confirmatory_seeds"]]
    shards = validate_registered_shard_plan(amendment=amendment, seeds=seeds)
    assert amendment.payload["amendment_id"] == AMENDMENT_ID
    assert amendment.payload["base_protocol_lock_sha256"] == file_sha256(
        REPO_ROOT / BASE_LOCK_RELATIVE
    )
    assert SHARD_COUNT == 4
    assert shards == (
        (1301, 1381),
        (1319, 1409),
        (1327, 1423),
        (1361, 1451),
    )
    assert registered_seed_shards(seeds) == shards


def test_partition_and_device_contracts_fail_closed() -> None:
    with pytest.raises(Exception, match="exactly 8"):
        registered_seed_shards([1, 2, 3, 4])
    with pytest.raises(Exception, match="exactly four"):
        _validate_devices(["cpu"], dry_run=True)
    with pytest.raises(Exception, match="four cpu"):
        _validate_devices(["cpu", "cpu", "cpu", "cuda:0"], dry_run=True)
    assert _validate_devices(["cpu"] * 4, dry_run=True) == ("cpu",) * 4


def test_amendment_validator_asserts_every_no_change_flag_and_authorization() -> None:
    payload = read_json_object_strict(REPO_ROOT / AMENDMENT_LOCK_RELATIVE)
    validate_amendment_payload(payload)
    for flag in SCIENTIFIC_NO_CHANGE_FLAGS:
        mutated = copy.deepcopy(payload)
        mutated["scientific_invariants"][flag] = True
        with pytest.raises(PostE21ContractError, match="no-change flags"):
            validate_amendment_payload(mutated)
        missing = copy.deepcopy(payload)
        del missing["scientific_invariants"][flag]
        with pytest.raises(PostE21ContractError, match="no-change flags"):
            validate_amendment_payload(missing)
    for key, value in {
        "main_authorized": False,
        "source_lock_required_before_main": False,
    }.items():
        mutated = copy.deepcopy(payload)
        mutated[key] = value
        with pytest.raises(PostE21ContractError, match="identity"):
            validate_amendment_payload(mutated)
    mutated = copy.deepcopy(payload)
    mutated["authorization"]["user_authorized"] = False
    with pytest.raises(PostE21ContractError, match="authorization"):
        validate_amendment_payload(mutated)
    expected_equivalence = registered_equivalence_validation_contract()
    assert payload["registered_equivalence_validation"] == expected_equivalence
    missing_root = copy.deepcopy(payload)
    del missing_root["registered_equivalence_validation"]
    with pytest.raises(PostE21ContractError, match="equivalence-validation"):
        validate_amendment_payload(missing_root)
    for key in expected_equivalence:
        missing = copy.deepcopy(payload)
        del missing["registered_equivalence_validation"][key]
        with pytest.raises(PostE21ContractError, match="equivalence-validation"):
            validate_amendment_payload(missing)
        wrong = copy.deepcopy(payload)
        wrong["registered_equivalence_validation"][key] = "TAMPERED"
        with pytest.raises(PostE21ContractError, match="equivalence-validation"):
            validate_amendment_payload(wrong)
    fixed = expected_equivalence["fixed_non_evidence_fields"]
    assert isinstance(fixed, dict)
    for key in fixed:
        missing = copy.deepcopy(payload)
        del missing["registered_equivalence_validation"]["fixed_non_evidence_fields"][key]
        with pytest.raises(PostE21ContractError, match="equivalence-validation"):
            validate_amendment_payload(missing)
        wrong = copy.deepcopy(payload)
        wrong["registered_equivalence_validation"]["fixed_non_evidence_fields"][key] = "TAMPERED"
        with pytest.raises(PostE21ContractError, match="equivalence-validation"):
            validate_amendment_payload(wrong)


def test_main_ack_and_exact_interpreter_fail_closed(tmp_path: Path) -> None:
    prefix = tmp_path / "catena-v6"
    python = prefix / "bin/python"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"fake interpreter")
    good_environment = {MAIN_ACK_ENV: MAIN_ACK_VALUE}
    validated = validate_main_authorization_and_runtime(
        environ=good_environment,
        executable=python,
        prefix=prefix,
        expected_prefix=prefix,
    )
    assert validated["python_prefix"] == str(prefix.resolve())
    assert validated["python_executable"] == str(python.resolve())
    with pytest.raises(PostE21ContractError, match="requires CATENA_POST_E21_MAIN_ACK"):
        validate_main_authorization_and_runtime(
            environ={},
            executable=python,
            prefix=prefix,
            expected_prefix=prefix,
        )
    with pytest.raises(PostE21ContractError, match="requires CATENA_POST_E21_MAIN_ACK"):
        validate_main_authorization_and_runtime(
            environ={MAIN_ACK_ENV: "WRONG"},
            executable=python,
            prefix=prefix,
            expected_prefix=prefix,
        )
    wrong_python = prefix / "bin/python-wrong"
    wrong_python.write_bytes(b"wrong")
    with pytest.raises(PostE21ContractError, match="exact catena-v6"):
        validate_main_authorization_and_runtime(
            environ=good_environment,
            executable=wrong_python,
            prefix=prefix,
            expected_prefix=prefix,
        )


def _idle_gpu_rows() -> list[dict[str, Any]]:
    return [
        {
            "index": index,
            "uuid": f"GPU-{index}",
            "name": "NVIDIA Test GPU",
            "memory_total_mib": 49140,
            "memory_used_mib": 0,
            "compute_capability": "8.0",
        }
        for index in range(4)
    ]


def test_gpu_inventory_requires_four_idle_homogeneous_gpus() -> None:
    devices = [f"cuda:{index}" for index in range(4)]
    rows = _idle_gpu_rows()
    validated = validate_gpu_inventory(
        devices=devices,
        gpu_rows=rows,
        compute_apps=[],
    )
    assert validated["status"] == "PASS"
    assert validated["selected_gpus"] == rows

    busy_apps = [
        {
            "gpu_uuid": "GPU-2",
            "pid": 42,
            "process_name": "python",
            "used_memory_mib": 10,
        }
    ]
    with pytest.raises(PostE21ContractError, match="active compute"):
        validate_gpu_inventory(
            devices=devices,
            gpu_rows=rows,
            compute_apps=busy_apps,
        )
    high_memory = copy.deepcopy(rows)
    high_memory[0]["memory_used_mib"] = 513
    with pytest.raises(PostE21ContractError, match="non-idle memory"):
        validate_gpu_inventory(
            devices=devices,
            gpu_rows=high_memory,
            compute_apps=[],
        )
    for field, value in [
        ("name", "Different GPU"),
        ("memory_total_mib", 12345),
        ("compute_capability", "9.0"),
    ]:
        heterogeneous = copy.deepcopy(rows)
        heterogeneous[-1][field] = value
        with pytest.raises(PostE21ContractError, match="homogeneous"):
            validate_gpu_inventory(
                devices=devices,
                gpu_rows=heterogeneous,
                compute_apps=[],
            )


def test_exclusive_launcher_lock_blocks_second_coordinator(tmp_path: Path) -> None:
    first = acquire_exclusive_launcher_lock(artifact_root=tmp_path)
    try:
        with pytest.raises(PostE21ContractError, match="exclusive lock"):
            acquire_exclusive_launcher_lock(artifact_root=tmp_path)
    finally:
        first.release()
    second = acquire_exclusive_launcher_lock(artifact_root=tmp_path)
    second.release()


def test_canonical_checkpoints_are_independent_atomic_copies(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    shard_dir = tmp_path / "shard"
    shard_dir.mkdir()
    source = shard_dir / "seed1301.pt"
    original = b"checkpoint payload"
    source.write_bytes(original)
    digest = file_sha256(source)
    materialization = _materialize_canonical_checkpoints(
        run_dir=run_dir,
        checkpoint_paths={"seed1301": source},
        checkpoint_hashes={"seed1301": digest},
    )
    canonical = run_dir / "checkpoints/seed1301.pt"
    assert materialization == {"seed1301": "atomic_byte_copy"}
    assert canonical.read_bytes() == original
    assert source.stat().st_ino != canonical.stat().st_ino
    source.write_bytes(b"mutated shard checkpoint")
    assert canonical.read_bytes() == original
    assert file_sha256(canonical) == digest


def _equivalence_fixture(
    tmp_path: Path,
) -> tuple[
    Path,
    dict[str, Any],
    dict[str, str],
    dict[str, Any],
    dict[str, Any],
]:
    source_record = {"sha256": "a" * 64, "files": 7}
    bindings = {
        "amendment_sha256": "b" * 64,
        "base_protocol_sha256": "c" * 64,
        "config_sha256": "d" * 64,
        "selection_lock_sha256": "e" * 64,
    }
    config = load_config(REPO_ROOT / "configs/e22b_active_path_locality.yaml")
    registered = parse_locality_methods(
        load_config(REPO_ROOT / "configs/e22a_locality_method_selection.yaml")["methods"]
    )
    equivalence_contract = build_equivalence_validation_contract(
        confirmatory_seeds=[int(seed) for seed in config["confirmatory_seeds"]],
        methods=(
            method_by_id(registered, "mean_retention"),
            method_by_id(registered, "cvar_020"),
        ),
        dry_runtime=runtime_locality_config(config, dry_run=True),
    )
    payload = {
        "schema_version": 1,
        "experiment_id": "e22b_active_path_locality",
        "amendment_id": AMENDMENT_ID,
        "status": "PASS",
        "run_mode": "CPU_SERIAL_VS_SHARD_EQUIVALENCE",
        "source_fingerprint": source_record,
        "amendment_lock_sha256": bindings["amendment_sha256"],
        "base_protocol_lock_sha256": bindings["base_protocol_sha256"],
        "config_sha256": bindings["config_sha256"],
        "selection_lock_sha256": bindings["selection_lock_sha256"],
        **equivalence_contract,
        "checks": {key: True for key in EQUIVALENCE_CHECK_KEYS},
    }
    external = tmp_path / "external_equivalence.json"
    write_json_strict(external, payload)
    return external, source_record, bindings, equivalence_contract, payload


def _validate_equivalence_fixture(
    *,
    path: Path,
    source: dict[str, Any],
    bindings: dict[str, str],
    contract: dict[str, Any],
) -> dict[str, Any]:
    return validate_equivalence_report(
        path=path,
        source=source,
        amendment_sha256=bindings["amendment_sha256"],
        base_protocol_sha256=bindings["base_protocol_sha256"],
        config_sha256=bindings["config_sha256"],
        selection_lock_sha256=bindings["selection_lock_sha256"],
        equivalence_contract=contract,
    )


def test_equivalence_validator_rejects_incomplete_extra_or_false_checks(
    tmp_path: Path,
) -> None:
    external, source_record, bindings, contract, payload = _equivalence_fixture(tmp_path)

    def validate_current() -> dict[str, Any]:
        return _validate_equivalence_fixture(
            path=external,
            source=source_record,
            bindings=bindings,
            contract=contract,
        )

    validate_current()
    for key in EQUIVALENCE_CHECK_KEYS:
        missing = copy.deepcopy(payload)
        del missing["checks"][key]
        write_json_strict(external, missing)
        with pytest.raises(PostE21ContractError, match="exact six"):
            validate_current()
        false = copy.deepcopy(payload)
        false["checks"][key] = False
        write_json_strict(external, false)
        with pytest.raises(PostE21ContractError, match="exact six"):
            validate_current()
    extra = copy.deepcopy(payload)
    extra["checks"]["unregistered_check"] = True
    write_json_strict(external, extra)
    with pytest.raises(PostE21ContractError, match="exact six"):
        validate_current()


def test_equivalence_validator_rejects_every_fixed_field_tamper(
    tmp_path: Path,
) -> None:
    external, source_record, bindings, contract, payload = _equivalence_fixture(tmp_path)

    def validate_current() -> dict[str, Any]:
        return _validate_equivalence_fixture(
            path=external,
            source=source_record,
            bindings=bindings,
            contract=contract,
        )

    for key in EQUIVALENCE_FIXED_FIELDS:
        missing = copy.deepcopy(payload)
        del missing[key]
        write_json_strict(external, missing)
        with pytest.raises(PostE21ContractError, match=f"fixed-field mismatch: {key}"):
            validate_current()
        wrong = copy.deepcopy(payload)
        wrong[key] = "TAMPERED"
        write_json_strict(external, wrong)
        with pytest.raises(PostE21ContractError, match=f"fixed-field mismatch: {key}"):
            validate_current()
    malformed_contract = dict(contract)
    del malformed_contract[EQUIVALENCE_FIXED_FIELDS[0]]
    write_json_strict(external, payload)
    with pytest.raises(PostE21ContractError, match="field contract is malformed"):
        validate_equivalence_report(
            path=external,
            source=source_record,
            **bindings,
            equivalence_contract=malformed_contract,
        )


def test_equivalence_copy_is_canonical_and_tamper_evident(tmp_path: Path) -> None:
    external, source_record, bindings, contract, _ = _equivalence_fixture(tmp_path)
    validated_external = validate_equivalence_report(
        path=external,
        source=source_record,
        **bindings,
        equivalence_contract=contract,
    )
    run_dir = tmp_path / "canonical_run"
    run_dir.mkdir()
    locked = copy_equivalence_report_into_run(
        validated_source=validated_external,
        run_dir=run_dir,
        source=source_record,
        **bindings,
        equivalence_contract=contract,
    )
    canonical = run_dir / "cpu_serial_shard_equivalence.json"
    assert locked["path"] == str(canonical.resolve())
    assert locked["sha256"] == file_sha256(external)
    external.write_text("{}\n", encoding="utf-8")
    assert file_sha256(canonical) == locked["sha256"]
    validate_locked_equivalence_record(
        record=locked,
        run_dir=run_dir,
        source=source_record,
        **bindings,
        equivalence_contract=contract,
    )
    payload = read_json_object_strict(canonical)
    write_json_strict(canonical, {**payload, "tampered": True})
    with pytest.raises(PostE21ContractError, match="equivalence record changed"):
        validate_locked_equivalence_record(
            record=locked,
            run_dir=run_dir,
            source=source_record,
            **bindings,
            equivalence_contract=contract,
        )


def test_execution_plan_detached_hash_detects_mutation(tmp_path: Path) -> None:
    plan_path = tmp_path / "execution_plan.json"
    plan = {
        "schema_version": 1,
        "experiment_id": "e22b_active_path_locality",
        "amendment_id": AMENDMENT_ID,
        "status": "LOCKED_BEFORE_SHARD_EXECUTION",
    }
    write_json_strict(plan_path, plan)
    write_json_strict(
        plan_path.with_suffix(".sha256.json"),
        {
            "schema_version": 1,
            "experiment_id": "e22b_active_path_locality",
            "run_id": None,
            "sha256": file_sha256(plan_path),
        },
    )
    assert _read_plan(plan_path) == plan
    write_json_strict(plan_path, {**plan, "mutated": True})
    with pytest.raises(Exception, match="detached execution-plan hash"):
        _read_plan(plan_path)


def _merge_metadata(parts: list[dict[str, Any]]) -> dict[str, Any]:
    identifier = {str(part["identifier_codebook_sha256"]) for part in parts}
    parameter_counts = [dict(part["parameter_counts"]) for part in parts]
    assert len(identifier) == 1
    assert all(counts == parameter_counts[0] for counts in parameter_counts)
    initialization: dict[str, str] = {}
    for part in parts:
        for seed, digest in part["initialization_hashes"].items():
            assert seed not in initialization
            initialization[str(seed)] = str(digest)
    return {
        "identifier_codebook_sha256": next(iter(identifier)),
        "initialization_hashes": initialization,
        "parameter_counts": parameter_counts[0],
    }


def test_cpu_dry_serial_vs_seed_shards_are_scientifically_exact(tmp_path: Path) -> None:
    config = load_config(REPO_ROOT / "configs/e22b_active_path_locality.yaml")
    runtime = runtime_locality_config(config, dry_run=True)
    registered = parse_locality_methods(
        load_config(REPO_ROOT / "configs/e22a_locality_method_selection.yaml")["methods"]
    )
    methods = (
        method_by_id(registered, "mean_retention"),
        method_by_id(registered, "sparse_0250"),
    )
    seeds = (71, 83)
    parent = load_parent_threshold_contract(repo_root=REPO_ROOT)
    common = {
        "runtime": runtime,
        "methods": methods,
        "device": torch.device("cpu"),
        "parent_lock_sha256": parent.sha256,
        "protocol_lock_sha256": file_sha256(REPO_ROOT / BASE_LOCK_RELATIVE),
        "risk_scale": float(parent.thresholds["maximum_nontarget_degradation"]),
    }

    serial_dir = tmp_path / "serial"
    serial_dir.mkdir()
    serial_rows, serial_hashes, serial_metadata = run_locality_method_grid(
        **common,
        seeds=seeds,
        run_dir=serial_dir,
    )
    serial_rows = canonicalize_rows(
        serial_rows,
        runtime=runtime,
        methods=methods,
        seeds=seeds,
    )
    serial_states = _state_hashes_for_run(
        run_dir=serial_dir,
        checkpoint_hashes=serial_hashes,
    )

    sharded_rows: list[dict[str, Any]] = []
    sharded_states: dict[str, str] = {}
    metadata_parts: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds):
        shard_dir = tmp_path / f"shard-{index:03d}"
        shard_dir.mkdir()
        rows, hashes, metadata = run_locality_method_grid(
            **common,
            seeds=[seed],
            run_dir=shard_dir,
        )
        sharded_rows.extend(rows)
        sharded_states.update(
            _state_hashes_for_run(
                run_dir=shard_dir,
                checkpoint_hashes=hashes,
            )
        )
        metadata_parts.append(metadata)
    sharded_rows = canonicalize_rows(
        sharded_rows,
        runtime=runtime,
        methods=methods,
        seeds=seeds,
    )
    checks = compare_serial_and_sharded_outputs(
        serial_rows=serial_rows,
        sharded_rows=sharded_rows,
        serial_checkpoint_states=serial_states,
        sharded_checkpoint_states=sharded_states,
        serial_metadata=serial_metadata,
        sharded_metadata=_merge_metadata(metadata_parts),
    )
    assert checks
    assert all(checks.values())
