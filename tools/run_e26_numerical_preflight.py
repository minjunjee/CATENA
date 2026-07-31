#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from catena.core.provenance_v61 import (
    read_json_object_strict,
    sha256_canonical_json,
    sha256_file,
    write_json_strict,
)
from catena.lm.audit_contract import (
    e26_execution_source_inventory,
    validate_e26_audit_locked_hashes,
)
from catena.lm.backend_lock import (
    backend_preflight_manifest,
    cuda_hardware_inventory,
    observed_single_visible_cuda_device,
    validate_backend_candidate_lock,
    validate_backend_preflight_manifest,
)
from catena.lm.checkpointing import (
    TrainingProgress,
    load_training_checkpoint,
    restart_audit_receipt,
    save_training_checkpoint,
)
from catena.lm.hashing import tensor_tree_digest
from catena.lm.model import CatenaLM
from catena.lm.numerical_audit import (
    NumericalTolerances,
    candidate_matrix_numerical_audit_receipt,
)
from catena.lm.paired_stream import TokenBalancedPairedTrainingCursor
from catena.lm.preflight_audit import (
    audit_locked_candidate,
    audit_scientific_cursor_replay,
    build_scientific_training_cursor,
    final_restart_record,
    model_config_for_candidate,
    run_cursor_training_steps,
)
from catena.lm.recurrent_mixer import (
    optimized_backend_diagnostics,
    optimized_backend_metadata,
    reset_optimized_backend_diagnostics,
)
from catena.lm.trainer import make_optimizer


def _yaml_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return payload


def _fresh_tmp_root(path: Path) -> Path:
    unresolved = path.expanduser()
    if unresolved.exists() or unresolved.is_symlink():
        raise FileExistsError(f"Preflight root must be fresh: {unresolved}")
    resolved_parent = unresolved.parent.resolve(strict=True)
    resolved = resolved_parent / unresolved.name
    if resolved_parent != Path("/tmp") and Path("/tmp") not in resolved_parent.parents:
        raise ValueError("E26 numerical preflight may write only below /tmp")
    if not unresolved.name.startswith("catena_e26_preflight_"):
        raise ValueError("Preflight root must start with catena_e26_preflight_")
    resolved.mkdir(mode=0o700)
    return resolved


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments],
        cwd=repo,
        text=True,
        stderr=subprocess.PIPE,
    ).strip()


def _locked_hashes(
    *,
    source_inventory: dict[str, Any],
    paths: dict[str, Path],
) -> dict[str, str]:
    values = {
        "source_tree_sha256": str(source_inventory["source_tree_sha256"]),
        **{f"{name}_sha256": sha256_file(path) for name, path in paths.items()},
    }
    return validate_e26_audit_locked_hashes(values)


def _candidate_worker(args: argparse.Namespace) -> int:
    spec_path = Path(args.worker_spec).expanduser().resolve(strict=True)
    spec = read_json_object_strict(spec_path)
    output = Path(args.worker_output).expanduser()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite candidate receipt: {output}")
    execution_device = observed_single_visible_cuda_device(
        expected_physical_index=int(spec["physical_device_index"]),
        expected_gpu_uuid=str(spec["gpu_uuid"]),
    )
    device = torch.device("cuda:0")
    fp32 = NumericalTolerances(
        relative_l2_max=float(spec["fp32_relative_l2_max"]),
        max_abs_max=float(spec["fp32_max_abs_max"]),
    )
    bf16 = NumericalTolerances(
        relative_l2_max=float(spec["bf16_relative_l2_max"]),
        max_abs_max=None,
    )
    row = audit_locked_candidate(
        spec["candidate"],
        device=device,
        partition_length=int(spec["partition_length"]),
        prefix_length=int(spec["prefix_length"]),
        gradient_sequence_length=int(spec["gradient_sequence_length"]),
        random_partition_seeds=tuple(int(value) for value in spec["random_partition_seeds"]),
        initialization_seed=int(spec["initialization_seed"]),
        data_seed=int(spec["data_seed"]),
        fp32_tolerances=fp32,
        bf16_tolerances=bf16,
    )
    row["execution_device"] = execution_device
    write_json_strict(output, row)
    return 0 if row["passed"] is True else 1


def _restart_backend_manifest(
    *,
    spec: dict[str, Any],
    config: Any,
    device: torch.device,
    execution_device: dict[str, Any],
    training_graph_identity: dict[str, Any],
) -> dict[str, Any]:
    payload = optimized_backend_metadata(
        device=device,
        compiler="inductor",
        chunk_size=int(config.optimized_chunk_size),
        parity_verified=False,
    )
    payload.update(
        {
            "candidate_id": str(spec["candidate"]["id"]),
            "model_config_sha256": sha256_canonical_json(spec["candidate"]),
            "variant": str(spec["variant"]),
            "execution_device": dict(execution_device),
            "training_graph_identity": dict(training_graph_identity),
            "source_tree_sha256": str(spec["locked_hashes"]["source_tree_sha256"]),
        }
    )
    return payload


def _restart_graph_identity() -> dict[str, Any]:
    diagnostics = optimized_backend_diagnostics()
    code_hash = diagnostics.get("last_graph_code_sha256")
    node_count = diagnostics.get("last_graph_node_count")

    def positive_integer(value: object) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value > 0

    positive_execution = all(
        positive_integer(diagnostics.get(field))
        for field in ("graph_invocations", "optimized_calls", "chunks_executed")
    )
    valid_graph = (
        isinstance(code_hash, str)
        and len(code_hash) == 64
        and isinstance(node_count, int)
        and not isinstance(node_count, bool)
        and node_count > 0
    )
    return {
        "last_graph_code_sha256": code_hash,
        "last_graph_node_count": node_count,
        "positive_compiled_execution": positive_execution,
        "fallback_count": diagnostics.get("fallback_count"),
        "graph_break_count": diagnostics.get("graph_break_count"),
        "passed": (
            positive_execution
            and valid_graph
            and diagnostics.get("fallback_count") == 0
            and diagnostics.get("graph_break_count") == 0
        ),
    }


def _restart_worker(args: argparse.Namespace) -> int:
    spec_path = Path(args.worker_spec).expanduser().resolve(strict=True)
    spec = read_json_object_strict(spec_path)
    output = Path(args.worker_output).expanduser()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite restart output: {output}")
    execution_device = observed_single_visible_cuda_device(
        expected_physical_index=int(spec["physical_device_index"]),
        expected_gpu_uuid=str(spec["gpu_uuid"]),
    )
    device = torch.device("cuda:0")
    seed = int(spec["initialization_seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    config = model_config_for_candidate(
        spec["candidate"],
        variant=str(spec["variant"]),
    )
    reset_optimized_backend_diagnostics()
    model = CatenaLM(config).to(device)
    optimizer = make_optimizer(model)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: max(0.0, 1.0 - 0.01 * step),
    )
    corpus, tokenizer, cursor = build_scientific_training_cursor(
        tokenizer_manifest_path=str(spec["tokenizer_manifest"]),
        corpus_manifest_path=str(spec["corpus_manifest"]),
        sequence_length=int(spec["sequence_length"]),
        general_seed=int(spec["general_seed"]),
        transaction_seed=int(spec["transaction_seed"]),
    )
    mode = str(args.worker_restart)
    autocast_dtype = torch.bfloat16
    n_steps = int(spec["boundary_steps"])
    m_steps = int(spec["followup_steps"])
    payload: dict[str, Any]
    if mode == "baseline":
        loss, records = run_cursor_training_steps(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            cursor=cursor,
            steps=n_steps + m_steps,
            device=device,
            autocast_dtype=autocast_dtype,
        )
        training_graph_identity = _restart_graph_identity()
        backend_runtime_manifest = _restart_backend_manifest(
            spec=spec,
            config=config,
            device=device,
            execution_device=execution_device,
            training_graph_identity=training_graph_identity,
        )
        payload = {
            "mode": mode,
            "execution_device": execution_device,
            "backend_runtime_manifest": backend_runtime_manifest,
            "final": final_restart_record(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                cursor=cursor,
                final_loss=loss,
                device=device,
                autocast_dtype=autocast_dtype,
            ),
            "records": records,
            "compiled_graph_identity": training_graph_identity,
        }
    elif mode == "save":
        loss, records = run_cursor_training_steps(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            cursor=cursor,
            steps=n_steps,
            device=device,
            autocast_dtype=autocast_dtype,
        )
        # Capture the graph used by the training update before the state-clone
        # probe below can execute another compiled forward and overwrite the
        # backend's last-graph diagnostics.
        training_graph_identity = _restart_graph_identity()
        backend_runtime_manifest = _restart_backend_manifest(
            spec=spec,
            config=config,
            device=device,
            execution_device=execution_device,
            training_graph_identity=training_graph_identity,
        )
        probe = (
            torch.arange(
                min(32, model.config.context_length),
                device=device,
            )[None, :]
            % model.config.vocab_size
        )
        model.eval()
        with (
            torch.no_grad(),
            torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
            ),
        ):
            runtime_state = model(probe).runtime_state.clone(detach=True)
        snapshot = cursor.snapshot()
        progress = TrainingProgress(
            optimizer_step=n_steps,
            tokens_seen=int(snapshot["loss_bearing_tokens_emitted"]),
            general_sequences_seen=cursor.general.sequence_index,
            transaction_sequences_seen=cursor.transaction.sequence_index,
            document_index=cursor.general.sequence_index,
            episode_index=cursor.transaction.episode_index,
            cursor_snapshot=snapshot,
            last_source_type=str(records[-1]["source_type"]),
        )
        checkpoint = save_training_checkpoint(
            str(spec["checkpoint_path"]),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            progress=progress,
            locked_hashes=spec["locked_hashes"],
            runtime_state=runtime_state,
            amp_policy={"dtype": "bfloat16", "grad_scaler": None},
            backend_manifest=backend_runtime_manifest,
        )
        payload = {
            "mode": mode,
            "execution_device": execution_device,
            "checkpoint": checkpoint.as_dict(),
            "backend_runtime_manifest": backend_runtime_manifest,
            "last_source": records[-1]["source_type"],
            "last_loss": loss,
            "saved_runtime_state_sha256": tensor_tree_digest(runtime_state),
            "boundary_snapshot_sha256": snapshot["snapshot_sha256"],
            "boundary_actual_source_nonpadding_tokens": {
                "general": snapshot["general_unpadded_tokens"],
                "transaction": snapshot["transaction_unpadded_tokens"],
            },
            "compiled_graph_identity": training_graph_identity,
        }
    elif mode == "resume":
        save_payload = read_json_object_strict(spec["save_output"])
        checkpoint = save_payload["checkpoint"]
        saved_backend_manifest = save_payload["backend_runtime_manifest"]
        if not isinstance(saved_backend_manifest, dict):
            raise RuntimeError("Saved restart backend manifest is not a mapping")
        saved_training_graph_identity = saved_backend_manifest.get("training_graph_identity")
        if not isinstance(saved_training_graph_identity, dict):
            raise RuntimeError("Saved restart backend manifest lacks the training graph identity")
        expected_base_manifest = _restart_backend_manifest(
            spec=spec,
            config=config,
            device=device,
            execution_device=execution_device,
            training_graph_identity=saved_training_graph_identity,
        )
        if saved_backend_manifest != expected_base_manifest:
            raise RuntimeError(
                "Saved restart backend manifest differs from the resumed worker runtime"
            )
        loaded = load_training_checkpoint(
            checkpoint["path"],
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_locked_hashes=spec["locked_hashes"],
            expected_file_sha256=checkpoint["sha256"],
            map_location=device,
            expected_amp_policy={"dtype": "bfloat16", "grad_scaler": None},
            expected_backend_manifest=saved_backend_manifest,
        )
        cursor = TokenBalancedPairedTrainingCursor.from_snapshot(
            corpus,
            tokenizer,
            tokenizer_hash=tokenizer.manifest.manifest_hash,
            snapshot=loaded.progress.cursor_snapshot,
        )
        if loaded.runtime_state is None:
            raise RuntimeError("Restart checkpoint omitted the hybrid runtime state")
        restored_runtime_sha256 = tensor_tree_digest(loaded.runtime_state)
        loss, records = run_cursor_training_steps(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            cursor=cursor,
            steps=m_steps,
            device=device,
            autocast_dtype=autocast_dtype,
        )
        training_graph_identity = _restart_graph_identity()
        backend_runtime_manifest = _restart_backend_manifest(
            spec=spec,
            config=config,
            device=device,
            execution_device=execution_device,
            training_graph_identity=training_graph_identity,
        )
        graph_identity_matches_checkpoint = training_graph_identity == saved_training_graph_identity
        payload = {
            "mode": mode,
            "execution_device": execution_device,
            "backend_runtime_manifest": backend_runtime_manifest,
            "first_source": records[0]["source_type"],
            "restored_runtime_state_sha256": restored_runtime_sha256,
            "final": final_restart_record(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                cursor=cursor,
                final_loss=loss,
                device=device,
                autocast_dtype=autocast_dtype,
            ),
            "records": records,
            "compiled_graph_identity": training_graph_identity,
            "training_graph_identity_matches_checkpoint": (graph_identity_matches_checkpoint),
        }
    else:
        raise ValueError(f"Unknown restart worker mode: {mode}")
    write_json_strict(output, payload)
    return 0


def _parse_main() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="E26 actual-candidate numerical/restart preflight (non-evidence only)"
    )
    parser.add_argument("--repo-root", type=Path, default=Path("/home/minjun_dev/CATENA_E26"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--calibration-config", type=Path)
    parser.add_argument("--protocol-lock", type=Path)
    parser.add_argument("--backend-candidate-lock", type=Path)
    parser.add_argument("--tokenizer-manifest", type=Path)
    parser.add_argument("--corpus-manifest", type=Path)
    parser.add_argument("--data-readiness", type=Path)
    parser.add_argument("--data-lock", type=Path)
    parser.add_argument("--transaction-manifest", type=Path)
    parser.add_argument("--validation-population-lock", type=Path)
    parser.add_argument("--schedule-manifest", type=Path)
    parser.add_argument("--frozen-tree-receipt", type=Path)
    parser.add_argument("--devices", default="0,1,2")
    parser.add_argument("--restart-devices", default="0,1,2,3")
    parser.add_argument("--partition-length", type=int, default=416)
    parser.add_argument("--prefix-length", type=int, default=17)
    parser.add_argument("--gradient-sequence-length", type=int, default=64)
    parser.add_argument("--initialization-seed", type=int, default=260_301)
    parser.add_argument("--data-seed", type=int, default=260_302)
    parser.add_argument("--cursor-replay-tokens", type=int, default=1_000_000)
    parser.add_argument("--restart-sequence-length", type=int, default=1024)
    parser.add_argument("--restart-followup-steps", type=int, default=2)
    parser.add_argument("--worker-candidate", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-restart",
        choices=("baseline", "save", "resume"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--worker-spec", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_candidate or args.worker_restart:
        if not args.worker_spec or not args.worker_output:
            parser.error("worker requires --worker-spec and --worker-output")
        return args
    required = (
        "output_root",
        "config",
        "calibration_config",
        "protocol_lock",
        "backend_candidate_lock",
        "tokenizer_manifest",
        "corpus_manifest",
        "data_readiness",
        "data_lock",
        "transaction_manifest",
        "validation_population_lock",
        "schedule_manifest",
        "frozen_tree_receipt",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error(f"missing required arguments: {', '.join(missing)}")
    return args


def _main_parent(args: argparse.Namespace) -> int:
    repo = args.repo_root.expanduser().resolve(strict=True)
    if _git(repo, "status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("Preflight requires a clean committed worktree")
    output_root = _fresh_tmp_root(args.output_root)
    paths = {
        "config": args.config.expanduser().resolve(strict=True),
        "calibration_config": args.calibration_config.expanduser().resolve(strict=True),
        "protocol_lock": args.protocol_lock.expanduser().resolve(strict=True),
        "backend_candidate_lock": args.backend_candidate_lock.expanduser().resolve(strict=True),
        "tokenizer_manifest": args.tokenizer_manifest.expanduser().resolve(strict=True),
        "corpus_manifest": args.corpus_manifest.expanduser().resolve(strict=True),
        "data_readiness": args.data_readiness.expanduser().resolve(strict=True),
        "data_lock": args.data_lock.expanduser().resolve(strict=True),
        "transaction_manifest": args.transaction_manifest.expanduser().resolve(strict=True),
        "validation_population_lock": (
            args.validation_population_lock.expanduser().resolve(strict=True)
        ),
        "schedule_manifest": args.schedule_manifest.expanduser().resolve(strict=True),
        "frozen_tree_receipt": args.frozen_tree_receipt.expanduser().resolve(strict=True),
    }
    source_inventory = e26_execution_source_inventory(repo)
    locked_hashes = _locked_hashes(source_inventory=source_inventory, paths=paths)
    config = _yaml_mapping(paths["config"])
    backend_gates = config.get("backend_gates")
    candidates = config.get("model_candidates")
    if not isinstance(backend_gates, dict) or not isinstance(candidates, list):
        raise ValueError("E26a config lacks backend_gates or model_candidates")
    if not candidates or any(not isinstance(value, dict) for value in candidates):
        raise ValueError("E26a config model_candidates are malformed")
    candidate_ids = [str(value.get("id", "")) for value in candidates]
    if any(not value for value in candidate_ids) or len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("E26a candidate ids must be non-empty and unique")
    candidate_lock = validate_backend_candidate_lock(
        read_json_object_strict(paths["backend_candidate_lock"]),
        repo_root=repo,
        config_path=paths["config"],
        candidates=candidates,
    )
    devices = [value.strip() for value in str(args.devices).split(",") if value.strip()]
    if len(devices) < len(candidates) or len(set(devices)) != len(devices):
        raise ValueError("Supply one distinct CUDA device per locked model candidate")
    candidate_hardware = cuda_hardware_inventory(devices)
    candidate_hardware_by_index = {
        int(row["physical_device_index"]): row for row in candidate_hardware
    }

    random_seeds = backend_gates.get("arbitrary_partition_random_seeds")
    if not isinstance(random_seeds, list) or len(random_seeds) < 8:
        raise ValueError("Config must lock at least eight arbitrary partition seeds")
    common_spec = {
        "partition_length": args.partition_length,
        "prefix_length": args.prefix_length,
        "gradient_sequence_length": args.gradient_sequence_length,
        "random_partition_seeds": random_seeds,
        "initialization_seed": args.initialization_seed,
        "data_seed": args.data_seed,
        "fp32_relative_l2_max": backend_gates["restart_and_grad_accum_fp32_relative_l2_max"],
        "fp32_max_abs_max": backend_gates["restart_and_grad_accum_fp32_max_abs_max"],
        "bf16_relative_l2_max": backend_gates["restart_and_grad_accum_bf16_relative_l2_max"],
    }
    source_lock = {
        "schema_version": "catena-v8.1",
        "manifest_type": "E26_EXECUTION_SOURCE_INVENTORY",
        "scientific_evidence": False,
        "evidence_tier": "NON_EVIDENCE_VALIDATION",
        "main_test_opened": False,
        "git_head": _git(repo, "rev-parse", "HEAD"),
        "git_status": "",
        "inventory": source_inventory,
        "locked_hashes": locked_hashes,
    }
    source_lock["receipt_sha256"] = sha256_canonical_json(source_lock)
    write_json_strict(output_root / "source_lock.json", source_lock)

    processes: list[tuple[str, subprocess.Popen[str], Any]] = []
    tool = Path(__file__).resolve()
    for candidate, device in zip(candidates, devices, strict=False):
        candidate_id = str(candidate["id"])
        physical_index = int(device)
        hardware = candidate_hardware_by_index[physical_index]
        spec = {
            **common_spec,
            "candidate": candidate,
            "physical_device_index": physical_index,
            "gpu_uuid": hardware["gpu_uuid"],
        }
        spec_path = output_root / f"{candidate_id}_worker_spec.json"
        output_path = output_root / f"{candidate_id}_numerical.json"
        log_handle = (output_root / f"{candidate_id}.log").open("x", encoding="utf-8")
        write_json_strict(spec_path, spec)
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = device
        process = subprocess.Popen(
            [
                sys.executable,
                str(tool),
                "--worker-candidate",
                "--worker-spec",
                str(spec_path),
                "--worker-output",
                str(output_path),
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
        if return_code != 0:
            failures[candidate_id] = return_code
    if failures:
        failure = {
            "schema_version": "catena-v8.1",
            "manifest_type": "E26_NUMERICAL_PREFLIGHT_FAILURE",
            "scientific_evidence": False,
            "evidence_tier": "NON_EVIDENCE_VALIDATION",
            "main_test_opened": False,
            "failures": failures,
        }
        write_json_strict(output_root / "failure_status.json", failure)
        return 1

    candidate_rows = {
        candidate_id: read_json_object_strict(output_root / f"{candidate_id}_numerical.json")
        for candidate_id in candidate_ids
    }
    receipt = candidate_matrix_numerical_audit_receipt(
        candidate_audits=candidate_rows,
        expected_candidate_ids=candidate_ids,
        locked_hashes=locked_hashes,
        source_inventory=source_inventory,
    )
    write_json_strict(output_root / "numerical_audit.json", receipt)

    restart_devices = [
        value.strip() for value in str(args.restart_devices).split(",") if value.strip()
    ]
    if len(restart_devices) < 4 or len(set(restart_devices)) != len(restart_devices):
        raise ValueError("Supply at least four distinct restart CUDA devices")
    physical_devices = list(dict.fromkeys([*devices, *restart_devices]))
    hardware_inventory = cuda_hardware_inventory(physical_devices)
    hardware_by_index = {int(row["physical_device_index"]): row for row in hardware_inventory}
    required_transitions = {"general_to_transaction", "transaction_to_general"}
    cursor_replays: dict[str, dict[str, Any]] = {}
    resume_results: dict[str, dict[str, Any]] = {}
    for candidate_index, candidate in enumerate(candidates):
        candidate_id = str(candidate["id"])
        context_length = int(candidate["context_length"])
        cursor_replay = audit_scientific_cursor_replay(
            tokenizer_manifest_path=str(paths["tokenizer_manifest"]),
            corpus_manifest_path=str(paths["corpus_manifest"]),
            sequence_length=context_length,
            general_seed=260_401,
            transaction_seed=260_402,
            minimum_tokens=int(args.cursor_replay_tokens),
        )
        cursor_replays[candidate_id] = cursor_replay
        write_json_strict(
            output_root / f"{candidate_id}_cursor_replay.json",
            cursor_replay,
        )
        restart_sequence_length = min(
            int(args.restart_sequence_length),
            context_length,
        )
        if restart_sequence_length < 2:
            raise ValueError(f"{candidate_id}: restart sequence length is below two")
        _, _, preview_cursor = build_scientific_training_cursor(
            tokenizer_manifest_path=str(paths["tokenizer_manifest"]),
            corpus_manifest_path=str(paths["corpus_manifest"]),
            sequence_length=restart_sequence_length,
            general_seed=260_411,
            transaction_seed=260_412,
        )
        preview_rows, _ = preview_cursor.take(64)
        preview_sources = [row.source_type for row in preview_rows]
        transition_boundaries: dict[str, int] = {}
        for index in range(1, len(preview_sources)):
            transition = f"{preview_sources[index - 1]}_to_{preview_sources[index]}"
            if transition in required_transitions:
                transition_boundaries.setdefault(transition, index)
        if set(transition_boundaries) != required_transitions:
            raise RuntimeError(f"{candidate_id}: could not locate both packed-cursor transitions")

        restart_cases: list[dict[str, Any]] = []
        for variant in ("dual_delta_lm", "projected_tied_delta_lm"):
            for transition in sorted(required_transitions):
                left_source, right_source = transition.split("_to_", maxsplit=1)
                restart_cases.append(
                    {
                        "case_id": (f"{candidate_id}__{variant}__{transition}"),
                        "candidate_id": candidate_id,
                        "variant": variant,
                        "transition": transition,
                        "boundary_steps": transition_boundaries[transition],
                        "left_source": left_source,
                        "right_source": right_source,
                    }
                )
        for case, restart_device in zip(
            restart_cases,
            restart_devices,
            strict=False,
        ):
            case_root = output_root / "restart" / str(case["case_id"])
            case_root.mkdir(parents=True)
            physical_index = int(restart_device)
            hardware = hardware_by_index[physical_index]
            spec = {
                "candidate": candidate,
                "variant": case["variant"],
                "boundary_steps": case["boundary_steps"],
                "followup_steps": int(args.restart_followup_steps),
                "sequence_length": restart_sequence_length,
                "initialization_seed": 260_420 + candidate_index,
                "general_seed": 260_411,
                "transaction_seed": 260_412,
                "tokenizer_manifest": str(paths["tokenizer_manifest"]),
                "corpus_manifest": str(paths["corpus_manifest"]),
                "locked_hashes": locked_hashes,
                "physical_device_index": physical_index,
                "gpu_uuid": hardware["gpu_uuid"],
                "checkpoint_path": str(case_root / "checkpoint.pt"),
                "save_output": str(case_root / "save.json"),
            }
            spec_path = case_root / "worker_spec.json"
            write_json_strict(spec_path, spec)
            case["root"] = case_root
            case["spec_path"] = spec_path
            case["device"] = restart_device
            case["gpu_uuid"] = hardware["gpu_uuid"]

        for mode in ("baseline", "save", "resume"):
            stage_processes: list[tuple[dict[str, Any], subprocess.Popen[str], Any]] = []
            for case in restart_cases:
                case_root = Path(case["root"])
                output_path = case_root / f"{mode}.json"
                log_handle = (case_root / f"{mode}.log").open(
                    "x",
                    encoding="utf-8",
                )
                environment = dict(os.environ)
                environment["CUDA_VISIBLE_DEVICES"] = str(case["device"])
                process = subprocess.Popen(
                    [
                        sys.executable,
                        str(tool),
                        "--worker-restart",
                        mode,
                        "--worker-spec",
                        str(case["spec_path"]),
                        "--worker-output",
                        str(output_path),
                    ],
                    cwd=repo,
                    env=environment,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                stage_processes.append((case, process, log_handle))
            stage_failures: dict[str, int] = {}
            for case, process, log_handle in stage_processes:
                return_code = process.wait()
                log_handle.close()
                if return_code != 0:
                    stage_failures[str(case["case_id"])] = return_code
            if stage_failures:
                write_json_strict(
                    output_root / f"restart_{candidate_id}_{mode}_failure.json",
                    {
                        "schema_version": "catena-v8.1",
                        "manifest_type": "E26_RESTART_PREFLIGHT_FAILURE",
                        "scientific_evidence": False,
                        "evidence_tier": "NON_EVIDENCE_VALIDATION",
                        "main_test_opened": False,
                        "candidate_id": candidate_id,
                        "stage": mode,
                        "failures": stage_failures,
                    },
                )
                return 1

        for case in restart_cases:
            case_root = Path(case["root"])
            baseline = read_json_object_strict(case_root / "baseline.json")
            saved = read_json_object_strict(case_root / "save.json")
            resumed = read_json_object_strict(case_root / "resume.json")
            final_equal = baseline["final"] == resumed["final"]
            boundary_equal = (
                saved["last_source"] == case["left_source"]
                and resumed["first_source"] == case["right_source"]
            )
            runtime_equal = (
                saved["saved_runtime_state_sha256"] == resumed["restored_runtime_state_sha256"]
            )
            execution_device_equal = (
                baseline["execution_device"]
                == saved["execution_device"]
                == resumed["execution_device"]
            )
            backend_manifest_equal = (
                baseline["backend_runtime_manifest"]
                == saved["backend_runtime_manifest"]
                == resumed["backend_runtime_manifest"]
            )
            graph_identity_equal = (
                baseline["compiled_graph_identity"]
                == saved["compiled_graph_identity"]
                == resumed["compiled_graph_identity"]
            )
            graph_identity_passed = baseline["compiled_graph_identity"].get("passed") is True
            resume_graph_matches_checkpoint = (
                resumed.get("training_graph_identity_matches_checkpoint") is True
            )
            case_passed = (
                final_equal
                and boundary_equal
                and runtime_equal
                and execution_device_equal
                and backend_manifest_equal
                and graph_identity_equal
                and graph_identity_passed
                and resume_graph_matches_checkpoint
            )
            resume_results[str(case["case_id"])] = {
                "candidate_id": candidate_id,
                "model_config_sha256": sha256_canonical_json(candidate),
                "variant": case["variant"],
                "transition": case["transition"],
                "boundary_steps": case["boundary_steps"],
                "physical_device_index": int(case["device"]),
                "gpu_uuid": case["gpu_uuid"],
                "execution_device": baseline["execution_device"],
                "continuous_vs_new_process_final_equal": final_equal,
                "boundary_sources_equal": boundary_equal,
                "runtime_state_checkpoint_equal": runtime_equal,
                "execution_device_equal": execution_device_equal,
                "backend_manifest_checkpoint_equal": backend_manifest_equal,
                "compiled_graph_identity_equal": graph_identity_equal,
                "resume_training_graph_identity_matches_checkpoint": (
                    resume_graph_matches_checkpoint
                ),
                "compiled_graph_identity": baseline["compiled_graph_identity"],
                "checkpoint": saved["checkpoint"],
                "actual_source_nonpadding_tokens_at_boundary": saved[
                    "boundary_actual_source_nonpadding_tokens"
                ],
                "passed": case_passed,
            }
    restart_receipt = restart_audit_receipt(
        resume_cases=resume_results,
        cursor_replays=cursor_replays,
        expected_candidate_ids=candidate_ids,
        locked_hashes=locked_hashes,
        source_inventory=source_inventory,
    )
    restart_receipt_path = output_root / "restart_audit.json"
    write_json_strict(restart_receipt_path, restart_receipt)

    all_passed = receipt["passed"] is True and restart_receipt["passed"] is True
    promotion = backend_preflight_manifest(
        candidate_lock_path=paths["backend_candidate_lock"],
        candidate_lock=candidate_lock,
        numerical_receipt_path=output_root / "numerical_audit.json",
        numerical_receipt=receipt,
        restart_receipt_path=restart_receipt_path,
        restart_receipt=restart_receipt,
        hardware_inventory=hardware_inventory,
        source_inventory=source_inventory,
        source_commit=_git(repo, "rev-parse", "HEAD"),
    )
    if promotion["e26a_candidate_capable"] is not all_passed:
        raise AssertionError("Backend promotion capability disagrees with audit receipts")
    validate_backend_preflight_manifest(
        promotion,
        repo_root=repo,
        candidate_lock_path=paths["backend_candidate_lock"],
        candidate_lock=candidate_lock,
        numerical_receipt_path=output_root / "numerical_audit.json",
        numerical_receipt=receipt,
        restart_receipt_path=restart_receipt_path,
        restart_receipt=restart_receipt,
        expected_hardware_inventory=cuda_hardware_inventory(physical_devices),
    )
    write_json_strict(output_root / "backend_preflight_manifest.json", promotion)
    status = {
        "schema_version": "catena-v8.1",
        "manifest_type": "E26_NUMERICAL_PREFLIGHT_STATUS",
        "scientific_evidence": False,
        "evidence_tier": "NON_EVIDENCE_VALIDATION",
        "main_test_opened": False,
        "numerical_passed": receipt["passed"],
        "cursor_restart_passed": restart_receipt["passed"],
        "packed_cursor_algorithms": {
            candidate_id: row["cursor_algorithm"]
            for candidate_id, row in sorted(cursor_replays.items())
        },
        "passed": all_passed,
    }
    status["receipt_sha256"] = sha256_canonical_json(status)
    write_json_strict(output_root / "preflight_status.json", status)
    return 0 if all_passed else 1


def main() -> int:
    args = _parse_main()
    if args.worker_candidate:
        return _candidate_worker(args)
    if args.worker_restart:
        return _restart_worker(args)
    return _main_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
