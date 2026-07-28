from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from catena.core.config import load_config
from catena.core.io import file_sha256, write_jsonl
from catena.data.structured_sequence_localization import (
    StructuredTransferCondition,
    StructuredTransferDemand,
    make_structured_identifier_codebook,
    tensor_sha256,
)
from catena.eval.structured_sequence_localization import (
    assess_structured_sequence_transfer,
    compute_structured_sequence_seed_contrasts,
    structured_sequence_aggregate_summary_ko,
    structured_sequence_source_summary_ko,
)
from catena.models.structured_sequence_localization import (
    MatchedStructuredSequenceController,
    StructuredSequenceFreedom,
    structured_sequence_parameter_count,
)
from catena.training.structured_sequence_localization import (
    evaluate_structured_sequence_controller,
    structured_state_dict_sha256,
    train_structured_sequence_controller,
)
from experiments.common import finalize_run, initialize_run

SOURCE_EXPERIMENT_ID = "e21a_structured_sequence_localization_transfer"
AGGREGATE_EXPERIMENT_ID = "e21b_structured_sequence_localization_aggregate"
DEFAULT_CONFIG = "configs/e21_structured_sequence_localization_transfer.yaml"
REPO_ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = REPO_ROOT / "docs/E21_STRUCTURED_SEQUENCE_LOCALIZATION_TRANSFER_LOCK.json"


def build_local_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="E21 structured sequence bridge")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--artifact-root",
        default=os.getenv("CATENA_ARTIFACT_ROOT", "artifacts"),
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument(
        "--source-run",
        action="append",
        default=[],
        help="Explicit e21a run directory; repeat once per source seed.",
    )
    return parser


def _read_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise TypeError(f"Expected object at {path}:{line_number}")
            rows.append(payload)
    return rows


def validate_e21_protocol_lock(config_path: str | Path) -> dict[str, str]:
    if not LOCK_PATH.is_file() or LOCK_PATH.is_symlink():
        raise RuntimeError("E21 prospective protocol lock is missing or unsafe")
    lock = _read_json_object(LOCK_PATH)
    if (
        lock.get("schema_version") != 1
        or lock.get("experiment_family") != "E21"
        or lock.get("main_execution_started") is not False
        or lock.get("protocol_frozen_before_any_e21_evaluation") is not True
    ):
        raise RuntimeError("E21 lock does not certify a prospective protocol")
    parent_locks = lock.get("parent_lock_provenance")
    expected_parents = {
        "docs/E18_SEQUENCE_CONTROL_LATTICE_LOCK.json",
        "docs/E19_LOCALIZATION_CANDIDATE_LOCK.json",
    }
    if not isinstance(parent_locks, dict) or set(parent_locks) != expected_parents:
        raise RuntimeError("E21 lock lacks exact E18/E19 parent provenance")
    for relative, expected_hash in parent_locks.items():
        candidate = (REPO_ROOT / relative).resolve()
        if not candidate.is_file() or file_sha256(candidate) != expected_hash:
            raise RuntimeError(f"E21 parent lock changed: {relative}")
    files = lock.get("files")
    if not isinstance(files, dict) or not files:
        raise RuntimeError("E21 lock lacks a non-empty file hash map")
    for relative, expected_hash in files.items():
        candidate = (REPO_ROOT / str(relative)).resolve()
        try:
            candidate.relative_to(REPO_ROOT)
        except ValueError as error:
            raise RuntimeError("E21 locked path escapes repository") from error
        if (
            not candidate.is_file()
            or candidate.is_symlink()
            or file_sha256(candidate) != expected_hash
        ):
            raise RuntimeError(f"E21 locked file changed: {relative}")
    config = Path(config_path).resolve()
    expected_config = files.get(
        "configs/e21_structured_sequence_localization_transfer.yaml"
    )
    if expected_config is None or file_sha256(config) != expected_config:
        raise RuntimeError("E21 config no longer matches its protocol lock")
    return {
        "path": str(LOCK_PATH.resolve()),
        "sha256": file_sha256(LOCK_PATH),
        "config_sha256": file_sha256(config),
    }


def _runtime_config(config: dict[str, Any], *, dry_run: bool) -> dict[str, Any]:
    runtime = deepcopy(config)
    if not dry_run:
        return runtime
    runtime["data"]["slots"] = 8
    runtime["data"]["value_dim"] = 8
    runtime["data"]["identifier_code_dim"] = 8
    runtime["model"]["hidden_dim"] = 32
    runtime["training"]["steps"] = 2
    runtime["training"]["batch_size"] = 2
    runtime["training"]["updates"] = 1
    runtime["training"]["gap_events"] = 2
    runtime["evaluation"]["updates"] = [1]
    runtime["evaluation"]["gap_events"] = [0, 2]
    runtime["evaluation"]["batches"] = 1
    runtime["evaluation"]["batch_size"] = 2
    runtime["evaluation"]["stress"] = {"updates": 1, "gap_events": 2}
    return runtime


def _summary_descriptor(path: Path) -> dict[str, str | int]:
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "line_count": len(path.read_text(encoding="utf-8").splitlines()),
    }


def _run_source(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    lock: dict[str, str],
) -> Path:
    registered_seeds = [int(value) for value in config["seeds"]]
    development_seed = int(config["development"]["seed"])
    if args.dry_run:
        if args.seed is not None and int(args.seed) != development_seed:
            raise ValueError("E21 dry-run accepts only the excluded development seed")
        seed = development_seed
    else:
        if args.seed is None:
            raise ValueError("E21a MAIN requires one explicit registered --seed")
        seed = int(args.seed)
        if seed not in registered_seeds:
            raise ValueError(f"Unregistered E21 main seed: {seed}")
    runtime = _runtime_config(config, dry_run=args.dry_run)
    initialized, run_dir, device = initialize_run(
        experiment_id=SOURCE_EXPERIMENT_ID,
        config_path=args.config,
        artifact_root=args.artifact_root,
        device_request=args.device,
        run_mode="DRY_RUN" if args.dry_run else "MAIN",
    )
    if initialized != config:
        raise RuntimeError("E21 config changed between validation and run start")
    conditions = [
        StructuredTransferCondition(value) for value in runtime["conditions"]
    ]
    families = [
        StructuredTransferDemand(value)
        for value in runtime["demand_families"]
    ]
    variants = [
        StructuredSequenceFreedom(value)
        for value in runtime["model"]["variants"]
    ]
    slots = int(runtime["data"]["slots"])
    value_dim = int(runtime["data"]["value_dim"])
    identifier_dim = int(runtime["data"]["identifier_code_dim"])
    codebook = make_structured_identifier_codebook(
        slots=slots,
        code_dim=identifier_dim,
        seed=int(runtime["namespaces"]["identifier_schema_seed"]),
    )

    rows: list[dict[str, Any]] = []
    initialization_hashes: dict[str, str] = {}
    parameter_counts: dict[str, int] = {}
    checkpoint_hashes: dict[str, str] = {}
    for variant in variants:
        torch.manual_seed(10_000 + seed)
        model = MatchedStructuredSequenceController(
            freedom=variant,
            slots=slots,
            identifier_dim=identifier_dim,
            value_dim=value_dim,
            hidden_dim=int(runtime["model"]["hidden_dim"]),
            address_temperature=float(
                runtime["model"]["address_temperature"]
            ),
        )
        initialization_hashes[variant.value] = structured_state_dict_sha256(
            model.state_dict()
        )
        parameter_counts[variant.value] = structured_sequence_parameter_count(
            model
        )
        trace = train_structured_sequence_controller(
            model=model,
            conditions=conditions,
            families=families,
            steps=int(runtime["training"]["steps"]),
            batch_size=int(runtime["training"]["batch_size"]),
            slots=slots,
            value_dim=value_dim,
            updates=int(runtime["training"]["updates"]),
            gap_events=int(runtime["training"]["gap_events"]),
            state_scale=float(runtime["data"]["state_scale"]),
            identifier_codebook=codebook,
            learning_rate=float(runtime["training"]["learning_rate"]),
            address_loss_weight=float(
                runtime["training"]["address_loss_weight"]
            ),
            candidate_loss_weight=float(
                runtime["training"]["candidate_loss_weight"]
            ),
            activity_loss_weight=float(
                runtime["training"]["activity_loss_weight"]
            ),
            retention_weight=float(runtime["training"]["retention_weight"]),
            train_namespace=str(runtime["namespaces"]["train"]),
            distractor_namespace=str(runtime["namespaces"]["distractor"]),
            device=device,
            seed=20_000 + seed,
        )
        checkpoint_path = (
            run_dir / "checkpoints" / f"seed{seed}_{variant.value}.pt"
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "model_class": "MatchedStructuredSequenceController",
                "variant": variant.value,
                "seed": seed,
                "runtime_config": runtime,
                "protocol_lock_sha256": lock["sha256"],
            },
            checkpoint_path,
        )
        checkpoint_hash = file_sha256(checkpoint_path)
        checkpoint_hashes[variant.value] = checkpoint_hash
        for condition in conditions:
            for family_index, family in enumerate(families):
                for updates in [
                    int(value) for value in runtime["evaluation"]["updates"]
                ]:
                    evaluation_seed = (
                        300_000
                        + 10_000 * seed
                        + 100 * family_index
                        + updates
                    )
                    for gap_events in [
                        int(value)
                        for value in runtime["evaluation"]["gap_events"]
                    ]:
                        metrics = evaluate_structured_sequence_controller(
                            model=model,
                            condition=condition,
                            family=family,
                            batches=int(runtime["evaluation"]["batches"]),
                            batch_size=int(
                                runtime["evaluation"]["batch_size"]
                            ),
                            slots=slots,
                            value_dim=value_dim,
                            updates=updates,
                            gap_events=gap_events,
                            state_scale=float(runtime["data"]["state_scale"]),
                            identifier_codebook=codebook,
                            evaluation_namespace=str(
                                runtime["namespaces"]["evaluation"]
                            ),
                            distractor_namespace=str(
                                runtime["namespaces"]["distractor"]
                            ),
                            device=device,
                            seed=evaluation_seed,
                        )
                        rows.append(
                            {
                                "seed": seed,
                                "variant": variant.value,
                                "condition": condition.value,
                                "demand_family": family.value,
                                "updates": updates,
                                "gap_events": gap_events,
                                "evaluation_seed": evaluation_seed,
                                "parameter_count": parameter_counts[
                                    variant.value
                                ],
                                "initialization_sha256": initialization_hashes[
                                    variant.value
                                ],
                                "checkpoint": str(checkpoint_path.resolve()),
                                "checkpoint_sha256": checkpoint_hash,
                                "train_final_loss": trace.final_loss,
                                "train_best_loss": trace.best_loss,
                                "examples_per_second": (
                                    trace.examples_per_second
                                ),
                                "optimizer": trace.optimizer,
                                **metrics,
                            }
                        )

    full_errors = {
        (
            str(row["condition"]),
            str(row["demand_family"]),
            int(row["updates"]),
            int(row["gap_events"]),
        ): float(row["affected_mse"])
        for row in rows
        if str(row["variant"]) == "full"
    }
    for row in rows:
        key = (
            str(row["condition"]),
            str(row["demand_family"]),
            int(row["updates"]),
            int(row["gap_events"]),
        )
        row["architecture_extra_error"] = (
            float(row["affected_mse"]) - full_errors[key]
        )

    paired_initialization = len(set(initialization_hashes.values())) == 1
    matched_parameters = len(set(parameter_counts.values())) == 1
    digest_groups: dict[tuple[str, int, int], set[str]] = defaultdict(set)
    for row in rows:
        digest_groups[
            (
                str(row["demand_family"]),
                int(row["updates"]),
                int(row["gap_events"]),
            )
        ].add(str(row["base_transaction_digest"]))
    paired_transactions = all(
        len(digests) == 1 for digests in digest_groups.values()
    )
    paired = paired_initialization and matched_parameters and paired_transactions
    if not paired:
        raise RuntimeError("E21 paired maximal-surface/data contract failed")

    expected_rows = (
        len(variants)
        * len(conditions)
        * len(families)
        * len(runtime["evaluation"]["updates"])
        * len(runtime["evaluation"]["gap_events"])
    )
    if len(rows) != expected_rows:
        raise RuntimeError("E21 source metric grid is incomplete")
    metrics_path = run_dir / "structured_sequence_transfer_metrics.jsonl"
    write_jsonl(metrics_path, rows)
    summary_path = run_dir / "RESULTS_SUMMARY_KO.md"
    summary_path.write_text(
        structured_sequence_source_summary_ko(
            dry_run=args.dry_run,
            seed=seed,
            rows=rows,
            report_status="DRY_RUN" if args.dry_run else "PASS",
            paired=paired,
        ),
        encoding="utf-8",
    )
    if len(summary_path.read_text(encoding="utf-8").splitlines()) > 55:
        raise RuntimeError("E21a Korean result summary exceeds one-page contract")
    report = {
        "status": "DRY_RUN" if args.dry_run else "PASS",
        "run_mode": "DRY_RUN" if args.dry_run else "MAIN",
        "run_scope": "STRUCTURED_SEQUENCE_LOCALIZATION_STATE_READ_TRANSFER",
        "evidence_tier": "CONTROLLED_REFERENCE",
        "scientific_evidence": False,
        "seed": seed,
        "development_seed_claim_eligible": False if args.dry_run else None,
        "rows": len(rows),
        "expected_rows": expected_rows,
        "paired_contract": {
            "common_maximal_surface": True,
            "paired_initialization": paired_initialization,
            "initialization_hashes": initialization_hashes,
            "matched_parameter_count": matched_parameters,
            "parameter_counts": parameter_counts,
            "paired_training_stream": True,
            "paired_evaluation_stream": True,
            "paired_base_transaction_digest": paired_transactions,
            "identifier_codebook_sha256": tensor_sha256(codebook),
            "namespaces": runtime["namespaces"],
        },
        "protocol": {
            "lock_path": lock["path"],
            "lock_sha256": lock["sha256"],
            "source_config_sha256": lock["config_sha256"],
            "h5_reopened": False,
        },
        "artifacts": {
            "metrics_path": str(metrics_path.resolve()),
            "metrics_sha256": file_sha256(metrics_path),
            "checkpoint_hashes": checkpoint_hashes,
            "results_summary_ko": _summary_descriptor(summary_path),
        },
        "claim_gate": {
            "status": (
                "NOT_EVALUATED_DRY_RUN"
                if args.dry_run
                else "PENDING_AGGREGATE"
            ),
            "allowed_claim": (
                "Per-seed controlled structured-sequence evidence only after "
                "the prospectively locked five-seed E21b aggregate."
            ),
            "forbidden_claim": (
                "H5 semantic factorization, natural-language, novel-identifier, "
                "pretrained-model, agent, official-backend, or runtime transfer."
            ),
        },
    }
    finalize_run(
        experiment_id=SOURCE_EXPERIMENT_ID,
        artifact_root=args.artifact_root,
        run_dir=run_dir,
        report=report,
    )
    print(f"[{SOURCE_EXPERIMENT_ID}] {report['status']}: {run_dir}")
    return run_dir


def _finite(row: dict[str, Any], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"Missing or invalid E21 source metric {key!r}") from error
    if not math.isfinite(value):
        raise RuntimeError(f"Non-finite E21 source metric {key!r}")
    return value


def _validate_source_run(
    run_dir: Path,
    *,
    expected_seed: int,
    expected_mode: str,
    config: dict[str, Any],
    lock: dict[str, str],
    dry_run: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    report_path = run_dir / "report.json"
    manifest_path = run_dir / "run_manifest.json"
    metrics_path = run_dir / "structured_sequence_transfer_metrics.jsonl"
    summary_path = run_dir / "RESULTS_SUMMARY_KO.md"
    for path in (report_path, manifest_path, metrics_path, summary_path):
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Incomplete or unsafe E21 source artifact: {path}")
    report = _read_json_object(report_path)
    manifest = _read_json_object(manifest_path)
    expected_status = "DRY_RUN" if dry_run else "PASS"
    if (
        manifest.get("schema_version") != 2
        or manifest.get("experiment_id") != SOURCE_EXPERIMENT_ID
        or manifest.get("run_id") != run_dir.name
        or manifest.get("run_mode") != expected_mode
        or manifest.get("report_sha256") != file_sha256(report_path)
    ):
        raise RuntimeError(f"E21 source manifest contract failed: {run_dir}")
    if (
        report.get("status") != expected_status
        or report.get("run_mode") != expected_mode
        or int(report.get("seed", -1)) != expected_seed
        or report.get("protocol", {}).get("lock_sha256") != lock["sha256"]
        or report.get("protocol", {}).get("source_config_sha256")
        != lock["config_sha256"]
        or report.get("artifacts", {}).get("metrics_sha256")
        != file_sha256(metrics_path)
        or report.get("artifacts", {})
        .get("results_summary_ko", {})
        .get("sha256")
        != file_sha256(summary_path)
    ):
        raise RuntimeError(f"E21 source report/hash contract failed: {run_dir}")
    rows = _read_jsonl(metrics_path)
    runtime = _runtime_config(config, dry_run=dry_run)
    variants = [str(value) for value in runtime["model"]["variants"]]
    conditions = [str(value) for value in runtime["conditions"]]
    families = [str(value) for value in runtime["demand_families"]]
    updates = [int(value) for value in runtime["evaluation"]["updates"]]
    gaps = [int(value) for value in runtime["evaluation"]["gap_events"]]
    expected_grid = {
        (variant, condition, family, update, gap)
        for variant in variants
        for condition in conditions
        for family in families
        for update in updates
        for gap in gaps
    }
    observed_grid: set[tuple[str, str, str, int, int]] = set()
    initialization_hashes: set[str] = set()
    parameter_counts: set[int] = set()
    digest_groups: dict[tuple[str, int, int], set[str]] = defaultdict(set)
    checkpoint_hashes = report.get("artifacts", {}).get("checkpoint_hashes")
    if not isinstance(checkpoint_hashes, dict):
        raise RuntimeError("E21 source checkpoint index missing")
    for row in rows:
        if int(row.get("seed", -1)) != expected_seed:
            raise RuntimeError("E21 source metric seed mismatch")
        key = (
            str(row.get("variant")),
            str(row.get("condition")),
            str(row.get("demand_family")),
            int(row.get("updates", -1)),
            int(row.get("gap_events", -1)),
        )
        if key not in expected_grid or key in observed_grid:
            raise RuntimeError(f"E21 source metric grid mismatch: {key}")
        observed_grid.add(key)
        initialization_hashes.add(str(row.get("initialization_sha256")))
        parameter_counts.add(int(row.get("parameter_count", -1)))
        digest_groups[(key[2], key[3], key[4])].add(
            str(row.get("base_transaction_digest"))
        )
        for metric in (
            "state_mse",
            "affected_mse",
            "retention_mse",
            "address_accuracy",
            "candidate_recovery_mse",
            "verified_activity_mean",
            "distractor_activity_mean",
            "architecture_extra_error",
        ):
            _finite(row, metric)
        checkpoint = Path(str(row.get("checkpoint", ""))).resolve()
        try:
            checkpoint.relative_to((run_dir / "checkpoints").resolve())
        except ValueError as error:
            raise RuntimeError("E21 source checkpoint escapes run") from error
        expected_hash = checkpoint_hashes.get(key[0])
        if (
            not checkpoint.is_file()
            or checkpoint.is_symlink()
            or row.get("checkpoint_sha256") != expected_hash
            or file_sha256(checkpoint) != expected_hash
        ):
            raise RuntimeError("E21 source checkpoint hash mismatch")
    if (
        observed_grid != expected_grid
        or len(initialization_hashes) != 1
        or len(parameter_counts) != 1
        or not all(len(values) == 1 for values in digest_groups.values())
    ):
        raise RuntimeError("E21 source paired grid/provenance contract failed")
    provenance = {
        "seed": expected_seed,
        "run_dir": str(run_dir.resolve()),
        "report_path": str(report_path.resolve()),
        "report_sha256": file_sha256(report_path),
        "run_manifest_path": str(manifest_path.resolve()),
        "run_manifest_sha256": file_sha256(manifest_path),
        "metrics_path": str(metrics_path.resolve()),
        "metrics_sha256": file_sha256(metrics_path),
        "results_summary_path": str(summary_path.resolve()),
        "results_summary_sha256": file_sha256(summary_path),
        "checkpoint_hashes": checkpoint_hashes,
    }
    return rows, provenance


def _run_aggregate(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    lock: dict[str, str],
) -> Path:
    required_seeds = (
        [int(config["development"]["seed"])]
        if args.dry_run
        else [int(value) for value in config["seeds"]]
    )
    if len(args.source_run) != len(required_seeds):
        raise ValueError(
            "E21b requires one explicit --source-run per required paired seed"
        )
    initialized, run_dir, _device = initialize_run(
        experiment_id=AGGREGATE_EXPERIMENT_ID,
        config_path=args.config,
        artifact_root=args.artifact_root,
        device_request=args.device,
        run_mode="DRY_RUN" if args.dry_run else "MAIN",
    )
    if initialized != config:
        raise RuntimeError("E21 config changed between validation and aggregate")
    rows: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    observed_seeds: set[int] = set()
    for source_value in args.source_run:
        source = Path(source_value).resolve()
        report = _read_json_object(source / "report.json")
        seed = int(report.get("seed", -1))
        if seed not in required_seeds or seed in observed_seeds:
            raise RuntimeError("E21b source seed is duplicate or unregistered")
        source_rows, source_provenance = _validate_source_run(
            source,
            expected_seed=seed,
            expected_mode="DRY_RUN" if args.dry_run else "MAIN",
            config=config,
            lock=lock,
            dry_run=args.dry_run,
        )
        rows.extend(source_rows)
        provenance.append(source_provenance)
        observed_seeds.add(seed)
    if observed_seeds != set(required_seeds):
        raise RuntimeError("E21b exact paired seed grid is incomplete")
    stress = _runtime_config(config, dry_run=args.dry_run)["evaluation"][
        "stress"
    ]
    seed_rows = compute_structured_sequence_seed_contrasts(
        rows,
        seeds=required_seeds,
        stress_updates=int(stress["updates"]),
        stress_gap_events=int(stress["gap_events"]),
    )
    assessment = assess_structured_sequence_transfer(
        seed_rows,
        thresholds=config["claim_gate"],
        alpha=float(config["statistics"]["alpha"]),
        dry_run=args.dry_run,
    )
    metrics_path = run_dir / "structured_sequence_paired_metrics.jsonl"
    contrasts_path = run_dir / "structured_sequence_seed_contrasts.jsonl"
    provenance_path = run_dir / "source_run_provenance.jsonl"
    write_jsonl(metrics_path, rows)
    write_jsonl(contrasts_path, seed_rows)
    write_jsonl(provenance_path, provenance)
    summary_path = run_dir / "RESULTS_SUMMARY_KO.md"
    summary_path.write_text(
        structured_sequence_aggregate_summary_ko(
            dry_run=args.dry_run,
            assessment=assessment,
            seeds=required_seeds,
        ),
        encoding="utf-8",
    )
    if len(summary_path.read_text(encoding="utf-8").splitlines()) > 55:
        raise RuntimeError("E21b Korean result summary exceeds one-page contract")
    status = (
        "NOT_EVALUATED_DRY_RUN"
        if args.dry_run
        else ("SUPPORTED" if assessment["supported"] else "NOT_SUPPORTED")
    )
    report = {
        "status": "DRY_RUN" if args.dry_run else "PASS",
        "run_mode": "DRY_RUN" if args.dry_run else "MAIN",
        "run_scope": "STRUCTURED_SEQUENCE_LOCALIZATION_TRANSFER_AGGREGATE",
        "evidence_tier": "CONTROLLED_REFERENCE",
        "scientific_evidence": False,
        "source_contract": {
            "experiment_id": SOURCE_EXPERIMENT_ID,
            "required_seeds": required_seeds,
            "explicit_source_runs_only": True,
            "source_runs": provenance,
            "protocol_lock_path": lock["path"],
            "protocol_lock_sha256": lock["sha256"],
            "source_config_sha256": lock["config_sha256"],
        },
        "summary": assessment,
        "artifacts": {
            "paired_metrics_sha256": file_sha256(metrics_path),
            "seed_contrasts_sha256": file_sha256(contrasts_path),
            "source_provenance_sha256": file_sha256(provenance_path),
            "results_summary_ko": _summary_descriptor(summary_path),
        },
        "claim_gate": {
            "status": status,
            "supported": bool(assessment["supported"]),
            "allowed_claim": (
                "In controlled repeated structured-event sequences with a "
                "fixed identifier schema and explicit demand fields, learned "
                "separate localization and current-state erase-candidate reads "
                "selectively recover the registered demands across sequence "
                "lengths and distractor gaps."
            ),
            "forbidden_claim": (
                "H5 semantic factorization, natural-language, novel-identifier, "
                "pretrained/recurrent-LM, agent/planning, official-backend, or "
                "runtime-superiority transfer."
            ),
        },
    }
    finalize_run(
        experiment_id=AGGREGATE_EXPERIMENT_ID,
        artifact_root=args.artifact_root,
        run_dir=run_dir,
        report=report,
    )
    print(f"[{AGGREGATE_EXPERIMENT_ID}] {report['status']}: {run_dir}")
    return run_dir


def main() -> None:
    args = build_local_parser().parse_args()
    config = load_config(args.config)
    if (
        config.get("experiment_family") != "E21"
        or config.get("source_experiment_id") != SOURCE_EXPERIMENT_ID
        or config.get("aggregate_experiment_id") != AGGREGATE_EXPERIMENT_ID
    ):
        raise ValueError("E21 config identity mismatch")
    lock = validate_e21_protocol_lock(args.config)
    if args.aggregate:
        if args.seed is not None:
            raise ValueError("--seed is not valid in E21 aggregate mode")
        _run_aggregate(args=args, config=config, lock=lock)
    else:
        if args.source_run:
            raise ValueError("--source-run is valid only with --aggregate")
        _run_source(args=args, config=config, lock=lock)


if __name__ == "__main__":
    main()
