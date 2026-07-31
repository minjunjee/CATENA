#!/usr/bin/env python3
"""Measure the locked E26 candidates without opening canonical E26a."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import yaml

from catena.core.provenance_v61 import (
    read_json_object_strict,
    sha256_canonical_json,
    sha256_file,
    write_json_strict,
)
from catena.lm.artifacts import ArtifactRun
from catena.lm.audit_contract import (
    e26_execution_source_inventory,
    validate_e26_audit_locked_hashes,
)
from catena.lm.backend_lock import (
    cuda_hardware_inventory,
    observed_single_visible_cuda_device,
    validate_backend_candidate_lock,
    validate_backend_preflight_manifest,
)
from catena.lm.checkpointing import validate_restart_audit_coverage
from catena.lm.e26a_executor import RealE26AExecutionBackend
from catena.lm.e26a_gate import (
    ResourcePolicy,
    candidate_measurement_from_mapping,
    candidate_numerical_coverage,
    project_candidate_resources,
    select_candidate,
)
from catena.lm.general_corpus import TokenMemmap
from catena.lm.numerical_audit import NumericalTolerances
from catena.lm.preflight_audit import audit_target_gradient_accumulation
from catena.lm.tokenizer import ExternalScientificTokenizer

_WORKER_INPUT_NAMES = (
    "config",
    "calibration_config",
    "protocol_lock",
    "backend_candidate_lock",
    "backend_manifest",
    "tokenizer_manifest",
    "corpus_manifest",
    "data_readiness",
    "data_lock",
    "transaction_manifest",
    "validation_population_lock",
    "schedule_manifest",
    "numerical_audit",
    "restart_audit",
    "frozen_tree_receipt",
)


def _yaml_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return payload


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments],
        cwd=repo,
        text=True,
        stderr=subprocess.PIPE,
    ).strip()


def _regular_file(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{label} must be a regular file: {resolved}")
    return resolved


def _require_canonical_hash(
    payload: dict[str, Any],
    *,
    field: str,
    label: str,
) -> str:
    observed = payload.get(field)
    if not isinstance(observed, str):
        raise ValueError(f"{label} lacks {field}")
    unhashed = dict(payload)
    unhashed.pop(field)
    if observed != sha256_canonical_json(unhashed):
        raise ValueError(f"{label} canonical SHA-256 mismatch")
    return observed


def _validate_worker_spec(
    payload: dict[str, Any],
) -> tuple[Path, dict[str, Path], dict[str, Any]]:
    required = {
        "schema_version",
        "manifest_type",
        "scientific_evidence",
        "main_test_opened",
        "repo_root",
        "output_root",
        "source_commit",
        "source_inventory",
        "locked_hashes",
        "candidate_id",
        "candidate_config_sha256",
        "physical_device_index",
        "gpu_uuid",
        "input_paths",
        "input_hashes",
        "spec_sha256",
    }
    if set(payload) != required:
        raise ValueError("Resource worker spec fields differ from the locked contract")
    if (
        payload.get("schema_version") != "catena-v8.1"
        or payload.get("manifest_type") != "E26_RESOURCE_WORKER_SPEC"
        or payload.get("scientific_evidence") is not False
        or payload.get("main_test_opened") is not False
    ):
        raise ValueError("Resource worker spec evidence boundary is invalid")
    _require_canonical_hash(payload, field="spec_sha256", label="worker spec")
    repo = Path(str(payload["repo_root"])).expanduser().resolve(strict=True)
    if _git(repo, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("Resource worker requires a clean committed worktree")
    if payload.get("source_commit") != _git(repo, "rev-parse", "HEAD"):
        raise ValueError("Resource worker source commit differs from current HEAD")
    source_inventory = e26_execution_source_inventory(repo)
    if payload.get("source_inventory") != source_inventory:
        raise ValueError("Resource worker source inventory differs from current source")
    locked = payload.get("locked_hashes")
    if not isinstance(locked, dict):
        raise ValueError("Resource worker spec lacks locked hashes")
    normalized_locked = validate_e26_audit_locked_hashes(locked)
    raw_paths = payload.get("input_paths")
    raw_hashes = payload.get("input_hashes")
    if not isinstance(raw_paths, dict) or set(raw_paths) != set(_WORKER_INPUT_NAMES):
        raise ValueError("Resource worker input paths differ from the locked contract")
    expected_hash_keys = {f"{name}_sha256" for name in _WORKER_INPUT_NAMES}
    if not isinstance(raw_hashes, dict) or set(raw_hashes) != expected_hash_keys:
        raise ValueError("Resource worker input hashes differ from the locked contract")
    paths = {
        name: _regular_file(raw_paths[name], f"worker input {name}") for name in _WORKER_INPUT_NAMES
    }
    observed_hashes = {f"{name}_sha256": sha256_file(path) for name, path in paths.items()}
    if raw_hashes != observed_hashes:
        raise ValueError("Resource worker input file hashes changed")
    observed_locked = {
        "source_tree_sha256": str(source_inventory["source_tree_sha256"]),
        **{key: value for key, value in observed_hashes.items() if key in normalized_locked},
    }
    if dict(sorted(observed_locked.items())) != normalized_locked:
        raise ValueError("Resource worker inputs differ from the numerical lock")
    return repo, paths, source_inventory


def _fresh_output_root(path: Path) -> Path:
    unresolved = path.expanduser()
    if unresolved.exists() or unresolved.is_symlink():
        raise FileExistsError(f"Resource-preflight root must be fresh: {unresolved}")
    resolved_parent = unresolved.parent.resolve(strict=True)
    resolved = resolved_parent / unresolved.name
    if resolved.parent != Path("/tmp"):
        raise ValueError("Resource-preflight root must be a direct child of /tmp")
    if not resolved.name.startswith("catena_e26_dry_resource_"):
        raise ValueError("Resource-preflight root must start with catena_e26_dry_resource_")
    resolved.mkdir(mode=0o700)
    return resolved


def _validate_worker_result(
    *,
    payload: dict[str, Any],
    spec: dict[str, Any],
    output_root: Path,
    expected_hardware: dict[str, Any],
) -> dict[str, Any]:
    required = {
        "schema_version",
        "manifest_type",
        "candidate_id",
        "candidate_config_sha256",
        "measurement",
        "target_gradient_accumulation",
        "run_dir",
        "report_sha256",
        "worker_spec_sha256",
        "source_commit",
        "source_tree_sha256",
        "input_hashes",
        "execution_device",
        "scientific_evidence",
        "main_test_opened",
        "scientific_e26a_started",
        "receipt_sha256",
    }
    if set(payload) != required:
        raise ValueError("Resource worker receipt fields differ from the contract")
    if (
        payload.get("schema_version") != "catena-v8.1"
        or payload.get("manifest_type") != "E26_RESOURCE_WORKER_RECEIPT"
        or payload.get("scientific_evidence") is not False
        or payload.get("main_test_opened") is not False
        or payload.get("scientific_e26a_started") is not False
    ):
        raise ValueError("Resource worker receipt evidence boundary is invalid")
    _require_canonical_hash(payload, field="receipt_sha256", label="worker receipt")
    expected_bindings = {
        "candidate_id": spec["candidate_id"],
        "candidate_config_sha256": spec["candidate_config_sha256"],
        "worker_spec_sha256": spec["spec_sha256"],
        "source_commit": spec["source_commit"],
        "source_tree_sha256": spec["source_inventory"]["source_tree_sha256"],
        "input_hashes": spec["input_hashes"],
    }
    mismatched = [
        key for key, expected in expected_bindings.items() if payload.get(key) != expected
    ]
    if mismatched:
        raise ValueError(f"Resource worker receipt binding mismatch: {mismatched}")
    measurement = payload.get("measurement")
    if not isinstance(measurement, dict) or measurement.get("candidate_id") != spec["candidate_id"]:
        raise ValueError("Resource worker measurement candidate mismatch")
    if measurement.get("model_config_sha256") != spec["candidate_config_sha256"]:
        raise ValueError("Resource worker measurement config hash mismatch")
    target_accumulation = payload.get("target_gradient_accumulation")
    if (
        not isinstance(target_accumulation, dict)
        or target_accumulation.get("candidate_id") != spec["candidate_id"]
        or target_accumulation.get("model_config_sha256") != spec["candidate_config_sha256"]
        or target_accumulation.get("passed") is not True
    ):
        raise ValueError("Resource worker target gradient-accumulation audit is invalid")
    target_accumulation_without_hash = dict(target_accumulation)
    audit_sha256 = target_accumulation_without_hash.pop("audit_sha256", None)
    if not isinstance(audit_sha256, str) or audit_sha256 != sha256_canonical_json(
        target_accumulation_without_hash
    ):
        raise ValueError("Resource worker target gradient-accumulation hash is invalid")
    device = payload.get("execution_device")
    if not isinstance(device, dict):
        raise ValueError("Resource worker lacks an observed execution device")
    expected_device = {
        "physical_device_index": spec["physical_device_index"],
        "gpu_uuid": spec["gpu_uuid"],
        "worker_visible_cuda_index": 0,
        "cuda_visible_devices": str(spec["physical_device_index"]),
        "name": expected_hardware["name"],
        "total_memory_bytes": expected_hardware["total_memory_bytes"],
        "compute_capability": expected_hardware["compute_capability"],
        "observation": "PYTORCH_VISIBLE_DEVICE_UUID_VERIFIED",
    }
    if device != expected_device:
        raise ValueError("Resource worker observed device differs from parent inventory")

    raw_run_dir = Path(str(payload["run_dir"])).expanduser()
    if raw_run_dir.is_symlink():
        raise ValueError("Resource worker run directory cannot be a symlink")
    run_dir = raw_run_dir.resolve(strict=True)
    if raw_run_dir != run_dir or not run_dir.is_relative_to(output_root):
        raise ValueError("Resource worker run directory escapes the preflight root")
    report_path = _regular_file(run_dir / "report.json", "worker report")
    if payload.get("report_sha256") != sha256_file(report_path):
        raise ValueError("Resource worker report SHA-256 mismatch")
    report = read_json_object_strict(report_path)
    report_bindings = {
        "candidate_id": payload["candidate_id"],
        "candidate_config_sha256": payload["candidate_config_sha256"],
        "measurement": measurement,
        "target_gradient_accumulation": target_accumulation,
        "worker_spec_sha256": payload["worker_spec_sha256"],
        "source_commit": payload["source_commit"],
        "source_tree_sha256": payload["source_tree_sha256"],
        "input_hashes": payload["input_hashes"],
        "execution_device": device,
        "scientific_evidence": False,
        "main_test_opened": False,
        "scientific_e26a_started": False,
        "scientific_main_started": False,
        "passed": True,
    }
    report_mismatches = [
        key for key, expected in report_bindings.items() if report.get(key) != expected
    ]
    if report_mismatches:
        raise ValueError(f"Resource worker report binding mismatch: {report_mismatches}")
    manifest = read_json_object_strict(
        _regular_file(run_dir / "run_manifest.json", "worker run manifest")
    )
    if (
        manifest.get("report_sha256") != payload["report_sha256"]
        or manifest.get("visible_devices") != str(spec["physical_device_index"])
        or manifest.get("source_fingerprint_verified_at_completion") is not True
        or not isinstance(manifest.get("git"), dict)
        or manifest["git"].get("head") != spec["source_commit"]
        or manifest["git"].get("status_porcelain") != ""
    ):
        raise ValueError("Resource worker run manifest integrity check failed")
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="E26 target-context throughput/resource preflight (non-evidence)"
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("/home/minjun_dev/CATENA_E26"),
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--calibration-config", type=Path)
    parser.add_argument("--protocol-lock", type=Path)
    parser.add_argument("--backend-candidate-lock", type=Path)
    parser.add_argument("--backend-manifest", type=Path)
    parser.add_argument("--tokenizer-manifest", type=Path)
    parser.add_argument("--corpus-manifest", type=Path)
    parser.add_argument("--data-readiness", type=Path)
    parser.add_argument("--data-lock", type=Path)
    parser.add_argument("--transaction-manifest", type=Path)
    parser.add_argument("--validation-population-lock", type=Path)
    parser.add_argument("--schedule-manifest", type=Path)
    parser.add_argument("--numerical-audit", type=Path)
    parser.add_argument("--restart-audit", type=Path)
    parser.add_argument("--frozen-tree-receipt", type=Path)
    parser.add_argument("--devices", default="0,1,2")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-spec", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if args.worker_spec is None or args.worker_output is None:
            parser.error("worker requires --worker-spec and --worker-output")
        return args
    required = (
        "output_root",
        "config",
        "calibration_config",
        "protocol_lock",
        "backend_candidate_lock",
        "backend_manifest",
        "tokenizer_manifest",
        "corpus_manifest",
        "data_readiness",
        "data_lock",
        "transaction_manifest",
        "validation_population_lock",
        "schedule_manifest",
        "numerical_audit",
        "restart_audit",
        "frozen_tree_receipt",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error(f"missing required arguments: {', '.join(missing)}")
    return args


def _worker(args: argparse.Namespace) -> int:
    spec_path = _regular_file(args.worker_spec, "worker spec")
    spec = read_json_object_strict(spec_path)
    repo, paths, source_inventory = _validate_worker_spec(spec)
    output = args.worker_output.expanduser()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite worker output: {output}")
    execution_device = observed_single_visible_cuda_device(
        expected_physical_index=int(spec["physical_device_index"]),
        expected_gpu_uuid=str(spec["gpu_uuid"]),
    )
    config = _yaml_mapping(paths["config"])
    numerical = read_json_object_strict(paths["numerical_audit"])
    restart = read_json_object_strict(paths["restart_audit"])
    _require_canonical_hash(
        numerical,
        field="receipt_sha256",
        label="worker numerical audit",
    )
    _require_canonical_hash(
        restart,
        field="receipt_sha256",
        label="worker restart audit",
    )
    candidate_id = str(spec["candidate_id"])
    candidate_rows = {str(candidate["id"]): candidate for candidate in config["model_candidates"]}
    if candidate_id not in candidate_rows:
        raise ValueError(f"Unknown locked candidate: {candidate_id}")
    if spec["candidate_config_sha256"] != sha256_canonical_json(candidate_rows[candidate_id]):
        raise ValueError("Worker candidate config hash mismatch")
    if not candidate_numerical_coverage(config, numerical).get(candidate_id, False):
        raise ValueError("Worker candidate lacks numerical coverage")
    if not validate_restart_audit_coverage(
        restart,
        expected_candidate_ids=list(candidate_rows),
    ).get(candidate_id, False):
        raise ValueError("Worker candidate lacks restart coverage")
    candidate_lock = validate_backend_candidate_lock(
        read_json_object_strict(paths["backend_candidate_lock"]),
        repo_root=repo,
        config_path=paths["config"],
        candidates=config["model_candidates"],
    )
    backend = read_json_object_strict(paths["backend_manifest"])
    backend_hardware = backend.get("hardware_inventory")
    if not isinstance(backend_hardware, list) or not backend_hardware:
        raise ValueError("Worker backend manifest lacks hardware inventory")
    validate_backend_preflight_manifest(
        backend,
        repo_root=repo,
        candidate_lock_path=paths["backend_candidate_lock"],
        candidate_lock=candidate_lock,
        numerical_receipt_path=paths["numerical_audit"],
        numerical_receipt=numerical,
        restart_receipt_path=paths["restart_audit"],
        restart_receipt=restart,
        expected_hardware_inventory=backend_hardware,
    )
    root = Path(spec["output_root"]).resolve(strict=True)
    run = ArtifactRun(
        experiment=f"e26_resource_{candidate_id}",
        artifact_root=root,
        run_mode="DRY_RUN",
        dry_run=True,
        source_root=repo,
        scientific_evidence=False,
        evidence_tier="NON_EVIDENCE_VALIDATION",
        claim_ceiling="RESOURCE_FEASIBILITY_ONLY",
    )
    tokenizer = ExternalScientificTokenizer.from_manifest(paths["tokenizer_manifest"])
    corpus = TokenMemmap.from_scientific_manifest(
        paths["corpus_manifest"],
        tokenizer_manifest=tokenizer.manifest,
    )
    measurements = RealE26AExecutionBackend()._measure_candidates(
        run,
        config=config,
        numerical_audit=numerical,
        corpus=corpus,
        tokenizer=tokenizer,
        device=torch.device("cuda:0"),
        candidate_ids={candidate_id},
    )
    if len(measurements) != 1:
        raise RuntimeError("Resource worker did not produce exactly one candidate")
    selected_measurement = measurements[0]
    measurement = asdict(selected_measurement)
    throughput = config["throughput"]
    backend_gates = config["backend_gates"]
    target_gradient_accumulation = audit_target_gradient_accumulation(
        candidate_rows[candidate_id],
        device=torch.device("cuda:0"),
        target_global_batch_tokens=int(throughput["target_global_batch_tokens"]),
        selected_microbatch_sequences=int(selected_measurement.selected_microbatch_sequences),
        microbatch_size_candidates=tuple(
            int(value) for value in throughput["microbatch_size_candidates"]
        ),
        initialization_seed=int(throughput["target_grad_accum_initialization_seed"]),
        data_seed=int(throughput["target_grad_accum_data_seed"]),
        bf16_tolerances=NumericalTolerances(
            relative_l2_max=float(backend_gates["restart_and_grad_accum_bf16_relative_l2_max"]),
            max_abs_max=None,
        ),
    )
    if target_gradient_accumulation["passed"] is not True:
        raise RuntimeError("Target-layout gradient-accumulation audit failed")
    report = {
        "schema_version": "catena-v8.1",
        "experiment": run.experiment,
        "run_id": run.run_id,
        "run_mode": "DRY_RUN",
        "scientific_evidence": False,
        "evidence_tier": "NON_EVIDENCE_VALIDATION",
        "claim_ceiling": "RESOURCE_FEASIBILITY_ONLY",
        "candidate_id": candidate_id,
        "candidate_config_sha256": str(spec["candidate_config_sha256"]),
        "measurement": measurement,
        "target_gradient_accumulation": target_gradient_accumulation,
        "worker_spec_sha256": str(spec["spec_sha256"]),
        "source_commit": str(spec["source_commit"]),
        "source_tree_sha256": str(source_inventory["source_tree_sha256"]),
        "input_hashes": dict(spec["input_hashes"]),
        "execution_device": execution_device,
        "main_test_opened": False,
        "scientific_e26a_started": False,
        "scientific_main_started": False,
        "passed": True,
    }
    run.finalize(
        report,
        (
            "# E26 resource preflight\n\n"
            f"- Candidate: `{candidate_id}`\n"
            "- Evidence: `NON_EVIDENCE_VALIDATION`\n"
            "- Scientific E26a started: `false`\n"
        ),
    )
    payload = {
        "schema_version": "catena-v8.1",
        "manifest_type": "E26_RESOURCE_WORKER_RECEIPT",
        "candidate_id": candidate_id,
        "candidate_config_sha256": str(spec["candidate_config_sha256"]),
        "measurement": measurement,
        "target_gradient_accumulation": target_gradient_accumulation,
        "run_dir": str(run.run_dir),
        "report_sha256": sha256_file(run.run_dir / "report.json"),
        "worker_spec_sha256": str(spec["spec_sha256"]),
        "source_commit": str(spec["source_commit"]),
        "source_tree_sha256": str(source_inventory["source_tree_sha256"]),
        "input_hashes": dict(spec["input_hashes"]),
        "execution_device": execution_device,
        "scientific_evidence": False,
        "main_test_opened": False,
        "scientific_e26a_started": False,
    }
    payload["receipt_sha256"] = sha256_canonical_json(payload)
    write_json_strict(output, payload)
    return 0


def _parent(args: argparse.Namespace) -> int:
    repo = args.repo_root.expanduser().resolve(strict=True)
    if _git(repo, "status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("Resource preflight requires a clean committed worktree")
    output_root = _fresh_output_root(args.output_root)
    paths = {
        name: _regular_file(getattr(args, name), f"resource input {name}")
        for name in _WORKER_INPUT_NAMES
    }
    numerical_path = paths["numerical_audit"]
    restart_path = paths["restart_audit"]
    backend_path = paths["backend_manifest"]
    config = _yaml_mapping(paths["config"])
    candidates = config.get("model_candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("E26a config lacks model candidates")
    candidate_ids = [str(candidate["id"]) for candidate in candidates]
    devices = [value.strip() for value in args.devices.split(",") if value.strip()]
    if len(devices) != len(candidate_ids) or len(set(devices)) != len(devices):
        raise ValueError("Supply one distinct CUDA device per locked candidate")

    source_inventory = e26_execution_source_inventory(repo)
    numerical = read_json_object_strict(numerical_path)
    restart = read_json_object_strict(restart_path)
    _require_canonical_hash(
        numerical,
        field="receipt_sha256",
        label="parent numerical audit",
    )
    _require_canonical_hash(
        restart,
        field="receipt_sha256",
        label="parent restart audit",
    )
    locked = numerical.get("locked_hashes")
    if not isinstance(locked, dict):
        raise ValueError("Numerical audit lacks locked hashes")
    normalized_locked = validate_e26_audit_locked_hashes(locked)
    all_input_hashes = {f"{name}_sha256": sha256_file(path) for name, path in paths.items()}
    observed_locked = {
        "source_tree_sha256": str(source_inventory["source_tree_sha256"]),
        **{key: value for key, value in all_input_hashes.items() if key in normalized_locked},
    }
    if dict(sorted(observed_locked.items())) != normalized_locked:
        raise ValueError("Resource preflight inputs differ from numerical lock")
    if restart.get("locked_hashes") != normalized_locked:
        raise ValueError("Restart and numerical input locks differ")
    numerical_coverage = candidate_numerical_coverage(config, numerical)
    restart_coverage = validate_restart_audit_coverage(
        restart,
        expected_candidate_ids=candidate_ids,
    )
    if not all(numerical_coverage.values()) or not all(restart_coverage.values()):
        raise ValueError("Candidate numerical/restart coverage is incomplete")
    candidate_lock = validate_backend_candidate_lock(
        read_json_object_strict(paths["backend_candidate_lock"]),
        repo_root=repo,
        config_path=paths["config"],
        candidates=candidates,
    )
    backend = read_json_object_strict(backend_path)
    recorded_hardware = backend.get("hardware_inventory")
    if not isinstance(recorded_hardware, list) or not recorded_hardware:
        raise ValueError("Backend manifest lacks hardware inventory")
    physical_indices = [
        str(row["physical_device_index"]) for row in recorded_hardware if isinstance(row, dict)
    ]
    current_backend_hardware = cuda_hardware_inventory(physical_indices)
    validate_backend_preflight_manifest(
        backend,
        repo_root=repo,
        candidate_lock_path=paths["backend_candidate_lock"],
        candidate_lock=candidate_lock,
        numerical_receipt_path=numerical_path,
        numerical_receipt=numerical,
        restart_receipt_path=restart_path,
        restart_receipt=restart,
        expected_hardware_inventory=current_backend_hardware,
    )
    worker_hardware = cuda_hardware_inventory(devices)
    worker_hardware_by_index = {int(row["physical_device_index"]): row for row in worker_hardware}
    tool = Path(__file__).resolve()
    processes: list[tuple[str, subprocess.Popen[str], Any]] = []
    worker_specs: dict[str, dict[str, Any]] = {}
    source_commit = _git(repo, "rev-parse", "HEAD")
    candidates_by_id = {str(candidate["id"]): candidate for candidate in candidates}
    for candidate_id, device in zip(candidate_ids, devices, strict=True):
        physical_index = int(device)
        hardware = worker_hardware_by_index[physical_index]
        spec = {
            "schema_version": "catena-v8.1",
            "manifest_type": "E26_RESOURCE_WORKER_SPEC",
            "scientific_evidence": False,
            "main_test_opened": False,
            "repo_root": str(repo),
            "output_root": str(output_root),
            "source_commit": source_commit,
            "source_inventory": source_inventory,
            "locked_hashes": normalized_locked,
            "candidate_id": candidate_id,
            "candidate_config_sha256": sha256_canonical_json(candidates_by_id[candidate_id]),
            "physical_device_index": physical_index,
            "gpu_uuid": hardware["gpu_uuid"],
            "input_paths": {name: str(path) for name, path in paths.items()},
            "input_hashes": all_input_hashes,
        }
        spec["spec_sha256"] = sha256_canonical_json(spec)
        worker_specs[candidate_id] = spec
        spec_path = output_root / f"{candidate_id}_worker_spec.json"
        result_path = output_root / f"{candidate_id}_worker_result.json"
        log_handle = (output_root / f"{candidate_id}.log").open(
            "x",
            encoding="utf-8",
        )
        write_json_strict(spec_path, spec)
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = device
        process = subprocess.Popen(
            [
                sys.executable,
                str(tool),
                "--worker",
                "--worker-spec",
                str(spec_path),
                "--worker-output",
                str(result_path),
            ],
            cwd=repo,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((candidate_id, process, log_handle))
    failures: dict[str, int] = {}
    for candidate_id, process, log_handle in processes:
        return_code = process.wait()
        log_handle.close()
        if return_code:
            failures[candidate_id] = return_code
    if failures:
        write_json_strict(
            output_root / "failure_status.json",
            {
                "schema_version": "catena-v8.1",
                "manifest_type": "E26_RESOURCE_PREFLIGHT_FAILURE",
                "scientific_evidence": False,
                "main_test_opened": False,
                "failures": failures,
            },
        )
        return 1

    worker_results = {
        candidate_id: _validate_worker_result(
            payload=read_json_object_strict(
                _regular_file(
                    output_root / f"{candidate_id}_worker_result.json",
                    f"{candidate_id} worker result",
                )
            ),
            spec=worker_specs[candidate_id],
            output_root=output_root,
            expected_hardware=worker_hardware_by_index[
                int(worker_specs[candidate_id]["physical_device_index"])
            ],
        )
        for candidate_id in candidate_ids
    }
    measurements = tuple(
        candidate_measurement_from_mapping(worker_results[candidate_id]["measurement"])
        for candidate_id in candidate_ids
    )
    data_lock = _yaml_mapping(paths["data_lock"])
    resource = data_lock["resource_policy"]
    throughput = config["throughput"]
    policy = ResourcePolicy(
        deadline_reference_hours=float(resource["deadline_reference_hours"]),
        deadline_fraction_max=float(throughput["deadline_fraction_max"]),
        max_main_wall_clock_hours=float(resource["max_main_wall_clock_hours"]),
        safety_time_multiplier=float(resource["safety_time_multiplier"]),
        max_main_checkpoint_storage_gib=float(resource["max_main_checkpoint_storage_gib"]),
        token_budgets=tuple(int(value) for value in resource["main_token_budget_candidates"]),
    )
    selection = select_candidate(
        config=config,
        measurements=measurements,
        policy=policy,
    )
    payload = {
        "schema_version": "catena-v8.1",
        "manifest_type": "E26_RESOURCE_PREFLIGHT",
        "scientific_evidence": False,
        "evidence_tier": "NON_EVIDENCE_VALIDATION",
        "claim_ceiling": "RESOURCE_FEASIBILITY_ONLY",
        "source_commit": source_commit,
        "source_inventory": source_inventory,
        "locked_hashes": normalized_locked,
        "upstream_receipts": {
            "backend_manifest": {
                "sha256": sha256_file(backend_path),
                "manifest_sha256": backend["manifest_sha256"],
            },
            "numerical_audit": {
                "sha256": sha256_file(numerical_path),
                "receipt_sha256": numerical["receipt_sha256"],
            },
            "restart_audit": {
                "sha256": sha256_file(restart_path),
                "receipt_sha256": restart["receipt_sha256"],
            },
        },
        "hardware_inventory": worker_hardware,
        "candidates": [
            {
                **asdict(measurement),
                "candidate_config_sha256": worker_results[measurement.candidate_id][
                    "candidate_config_sha256"
                ],
                "resource_projections": list(project_candidate_resources(measurement, policy)),
                "worker_run_dir": worker_results[measurement.candidate_id]["run_dir"],
                "worker_report_sha256": worker_results[measurement.candidate_id]["report_sha256"],
                "worker_spec_sha256": worker_results[measurement.candidate_id][
                    "worker_spec_sha256"
                ],
                "worker_receipt_sha256": worker_results[measurement.candidate_id]["receipt_sha256"],
                "execution_device": worker_results[measurement.candidate_id]["execution_device"],
                "target_gradient_accumulation": worker_results[measurement.candidate_id][
                    "target_gradient_accumulation"
                ],
            }
            for measurement in measurements
        ],
        "selection": selection.as_dict(),
        "main_test_opened": False,
        "scientific_e26a_started": False,
        "scientific_e26b_started": False,
        "scientific_main_started": False,
        "canonical_e26_artifact_created": False,
        "passed": True,
    }
    payload["receipt_sha256"] = sha256_canonical_json(payload)
    write_json_strict(output_root / "resource_preflight.json", payload)
    return 0


def main() -> int:
    args = _parse_args()
    return _worker(args) if args.worker else _parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
