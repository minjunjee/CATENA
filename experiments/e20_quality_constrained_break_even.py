from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from catena.core.config import load_config
from catena.core.io import file_sha256, write_jsonl
from catena.eval.quality_break_even import (
    BASELINE_POLICIES,
    CACHED_POLICY,
    INTERNAL_POLICY,
    REGISTERED_CORRECTION_EPSILON,
    REGISTERED_POLICIES,
    REGISTERED_QUERY_COUNTS,
    REGISTERED_RETENTION_EPSILON,
    BreakEvenPolicy,
    assess_quality_constrained_break_even,
    benchmark_policy,
    generate_structured_break_even_workload,
)
from catena.systems.device import resolve_device
from experiments.common import finalize_run, initialize_run

EXPERIMENT_ID = "e20_quality_constrained_break_even"
DEFAULT_CONFIG = "configs/e20_quality_constrained_break_even.yaml"
REPO_ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = REPO_ROOT / "docs/E20_QUALITY_CONSTRAINED_BREAK_EVEN_LOCK.json"
EVIDENCE_TIER = "CONTROLLED_SYSTEMS_PROXY"


def _read_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _contained_repo_path(reference: str, *, label: str) -> Path:
    if not reference:
        raise RuntimeError(f"{label} is empty")
    candidate = (REPO_ROOT / reference).resolve()
    try:
        candidate.relative_to(REPO_ROOT)
    except ValueError as error:
        raise RuntimeError(
            f"{label} escapes repository root: {reference}"
        ) from error
    return candidate


def _validate_hash_map(
    records: object,
    *,
    label: str,
) -> None:
    if not isinstance(records, Mapping) or not records:
        raise RuntimeError(f"E20 lock has no {label} records")
    for relative, expected in records.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise RuntimeError(f"Invalid E20 {label} hash record")
        path = _contained_repo_path(relative, label=f"{label}.{relative}")
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"E20 locked file missing or unsafe: {path}")
        if file_sha256(path) != expected:
            raise RuntimeError(f"E20 locked file changed: {path}")


def validate_e20_config(config: dict[str, Any]) -> None:
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("E20 config experiment_id mismatch")
    if config.get("protocol", {}).get("version") != "E20_PREMAIN_V1":
        raise ValueError("E20 protocol version changed")
    if config.get("protocol", {}).get("main_evaluation_started") is not False:
        raise ValueError("E20 config no longer certifies pre-main status")

    evidence = config.get("evidence", {})
    if (
        evidence.get("tier") != EVIDENCE_TIER
        or evidence.get("scientific_evidence") is not False
        or evidence.get("official_backend_claim_eligible") is not False
        or evidence.get("production_systems_claim_eligible") is not False
        or evidence.get("language_model_claim_eligible") is not False
    ):
        raise ValueError("E20 evidence boundary changed")

    policies = config.get("policies", {})
    registered_order = tuple(
        BreakEvenPolicy(str(value))
        for value in policies.get("registered_order", [])
    )
    if registered_order != REGISTERED_POLICIES:
        raise ValueError("E20 registered policy order changed")
    if BreakEvenPolicy(str(policies.get("internal"))) is not INTERNAL_POLICY:
        raise ValueError("E20 internal policy changed")
    baselines = tuple(
        BreakEvenPolicy(str(value))
        for value in policies.get("baselines", [])
    )
    if baselines != BASELINE_POLICIES:
        raise ValueError("E20 baseline policies changed")
    if CACHED_POLICY not in baselines:
        raise ValueError("E20 cached-snapshot baseline is missing")

    query_counts = tuple(int(value) for value in config["query_counts"])
    if query_counts != REGISTERED_QUERY_COUNTS:
        raise ValueError("E20 query-count grid changed")
    quality = config.get("quality_guardrails", {})
    if (
        float(quality.get("affected_correction_mse_epsilon"))
        != REGISTERED_CORRECTION_EPSILON
        or float(quality.get("retention_mse_epsilon"))
        != REGISTERED_RETENTION_EPSILON
    ):
        raise ValueError("E20 quality thresholds changed")

    workload = config.get("workload", {})
    numeric_positive = (
        "batch_size",
        "slots",
        "value_dim",
        "state_scale",
    )
    for field in numeric_positive:
        value = float(workload.get(field, 0))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"workload.{field} must be positive")
    if int(workload["slots"]) <= max(query_counts):
        raise ValueError("workload.slots must exceed the largest m")
    if int(workload.get("logical_reads_per_query", 0)) != 2:
        raise ValueError("E20 logical query-pair contract changed")

    timing = config.get("timing", {})
    if int(timing.get("warmup_repeats", -1)) < 0:
        raise ValueError("timing.warmup_repeats is invalid")
    if int(timing.get("measured_repeats", 0)) <= 0:
        raise ValueError("timing.measured_repeats is invalid")
    if timing.get("primary_statistic") != "median_total_seconds":
        raise ValueError("E20 primary latency statistic changed")
    if (
        timing.get("device_sync_before_and_after_each_measurement")
        is not True
    ):
        raise ValueError("E20 device synchronization cannot be disabled")

    estimand = config.get("primary_estimand", {})
    if (
        estimand.get("name")
        != "minimum_quality_constrained_break_even_m"
        or estimand.get("internal_policy") != INTERNAL_POLICY.value
        or estimand.get("latency_relation")
        != "internal_median_total_seconds_lte_baseline"
        or estimand.get("quality_required_for_both_policies") is not True
        or estimand.get("cached_dominance_decision")
        != "NOT_SUPPORTED_BOUNDARY"
    ):
        raise ValueError("E20 primary estimand changed")

    execution = config.get("execution", {})
    if (
        execution.get("dry_run_required_device_type") != "cpu"
        or execution.get("main_required_device_type") != "cuda"
        or execution.get("dry_run_claim_eligible") is not False
    ):
        raise ValueError("E20 CPU-dry/CUDA-main split changed")


def validate_e20_protocol_lock(config_path: str | Path) -> str:
    if not LOCK_PATH.is_file() or LOCK_PATH.is_symlink():
        raise RuntimeError("E20 prospective protocol lock is missing")
    lock = _read_json_object(LOCK_PATH)
    if (
        lock.get("experiment_id") != EXPERIMENT_ID
        or lock.get("protocol_version") != "E20_PREMAIN_V1"
        or lock.get("main_evaluation_started") is not False
        or lock.get("gpu_main_executed") is not False
        or lock.get("main_artifact_existed_at_lock") is not False
    ):
        raise RuntimeError("E20 lock does not certify a pre-main protocol")
    _validate_hash_map(lock.get("files"), label="files")
    _validate_hash_map(
        lock.get("guardrail_source_files"),
        label="guardrail_source_files",
    )
    config_relative = "configs/e20_quality_constrained_break_even.yaml"
    expected_config = lock["files"].get(config_relative)
    resolved_config = Path(config_path).resolve()
    if resolved_config != (REPO_ROOT / config_relative).resolve():
        raise RuntimeError("E20 main/dry config must be the locked config path")
    if file_sha256(resolved_config) != expected_config:
        raise RuntimeError("E20 config does not match the prospective lock")
    return file_sha256(LOCK_PATH)


def validate_execution_device(
    *,
    device: torch.device,
    dry_run: bool,
) -> None:
    required = "cpu" if dry_run else "cuda"
    if device.type != required:
        mode = "DRY_RUN" if dry_run else "MAIN"
        raise RuntimeError(
            f"E20 {mode} requires device type {required!r}; got {device}"
        )


def build_local_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=EXPERIMENT_ID)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--artifact-root",
        default="artifacts",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_local_parser().parse_args()
    pre_config = load_config(args.config)
    validate_e20_config(pre_config)
    protocol_lock_sha256 = validate_e20_protocol_lock(args.config)
    source_config_sha256 = file_sha256(args.config)
    preflight_device = resolve_device(args.device)
    validate_execution_device(
        device=preflight_device,
        dry_run=args.dry_run,
    )

    config, run_dir, device = initialize_run(
        experiment_id=EXPERIMENT_ID,
        config_path=args.config,
        artifact_root=args.artifact_root,
        device_request=args.device,
        run_mode="DRY_RUN" if args.dry_run else "MAIN",
    )
    if device != preflight_device:
        raise RuntimeError("E20 device changed between preflight and run start")

    batch_size = int(config["workload"]["batch_size"])
    warmup_repeats = int(config["timing"]["warmup_repeats"])
    measured_repeats = int(config["timing"]["measured_repeats"])
    if args.dry_run:
        overrides = config["dry_run_overrides"]
        batch_size = int(overrides["batch_size"])
        warmup_repeats = int(overrides["warmup_repeats"])
        measured_repeats = int(overrides["measured_repeats"])

    rows: list[dict[str, Any]] = []
    base_workload_sha256: str | None = None
    for query_count in REGISTERED_QUERY_COUNTS:
        workload = generate_structured_break_even_workload(
            batch_size=batch_size,
            slots=int(config["workload"]["slots"]),
            value_dim=int(config["workload"]["value_dim"]),
            state_scale=float(config["workload"]["state_scale"]),
            seed=int(config["workload"]["seed"]),
            query_count=query_count,
        )
        if base_workload_sha256 is None:
            base_workload_sha256 = workload.base_workload_sha256
        elif workload.base_workload_sha256 != base_workload_sha256:
            raise RuntimeError("E20 state/update workload changed across m")
        for policy in REGISTERED_POLICIES:
            row = benchmark_policy(
                workload=workload,
                policy=policy,
                device=device,
                warmup_repeats=warmup_repeats,
                measured_repeats=measured_repeats,
            )
            row.update(
                {
                    "run_mode": "DRY_RUN" if args.dry_run else "MAIN",
                    "device": str(device),
                    "evidence_tier": EVIDENCE_TIER,
                    "scientific_evidence": False,
                    "correction_guardrail_pass": bool(
                        float(row["affected_correction_mse"])
                        <= REGISTERED_CORRECTION_EPSILON
                    ),
                    "retention_guardrail_pass": bool(
                        float(row["retention_mse"])
                        <= REGISTERED_RETENTION_EPSILON
                    ),
                }
            )
            rows.append(row)

    gate = assess_quality_constrained_break_even(
        rows,
        query_counts=REGISTERED_QUERY_COUNTS,
        correction_epsilon=REGISTERED_CORRECTION_EPSILON,
        retention_epsilon=REGISTERED_RETENTION_EPSILON,
        dry_run=args.dry_run,
    )
    metrics_path = run_dir / "quality_break_even_metrics.jsonl"
    write_jsonl(metrics_path, rows)

    if file_sha256(args.config) != source_config_sha256:
        raise RuntimeError("E20 config changed during execution")
    if validate_e20_protocol_lock(args.config) != protocol_lock_sha256:
        raise RuntimeError("E20 protocol lock changed during execution")

    report = {
        "execution_status": "DRY_RUN" if args.dry_run else "PASS",
        "run_mode": "DRY_RUN" if args.dry_run else "MAIN",
        "experiment_id": EXPERIMENT_ID,
        "evidence_tier": EVIDENCE_TIER,
        "scientific_evidence": False,
        "official_backend_claim_eligible": False,
        "production_systems_claim_eligible": False,
        "structured_systems_proxy_only": True,
        "device": str(device),
        "registered_query_counts": list(REGISTERED_QUERY_COUNTS),
        "registered_policies": [
            policy.value for policy in REGISTERED_POLICIES
        ],
        "workload_contract": {
            "schema": config["workload"]["schema"],
            "batch_size": batch_size,
            "slots": int(config["workload"]["slots"]),
            "value_dim": int(config["workload"]["value_dim"]),
            "seed": int(config["workload"]["seed"]),
            "base_workload_sha256": base_workload_sha256,
            "paired_workload_digest_per_m": True,
            "logical_reads_per_query": 2,
        },
        "timing_contract": {
            "warmup_repeats": warmup_repeats,
            "measured_repeats": measured_repeats,
            "primary_statistic": "median_total_seconds",
            "device_sync_before_and_after_each_measurement": True,
            "preparation_outside_timed_region": (
                "query descriptors and persistent internal state"
            ),
        },
        "quality_guardrails": {
            "affected_correction_mse_epsilon": (
                REGISTERED_CORRECTION_EPSILON
            ),
            "retention_mse_epsilon": REGISTERED_RETENTION_EPSILON,
        },
        "primary_estimand": {
            "name": "minimum_quality_constrained_break_even_m",
            "by_baseline": gate["minimum_m_by_baseline"],
        },
        "claim_gate": gate,
        "claim_boundary": {
            "allowed": (
                "Quality-constrained latency boundary in this controlled "
                "structured systems proxy only."
            ),
            "forbidden": (
                "Production, official-backend, pretrained-language-model, "
                "or general agent runtime superiority."
            ),
        },
        "protocol": {
            "lock_path": str(LOCK_PATH),
            "lock_sha256": protocol_lock_sha256,
            "source_config_sha256": source_config_sha256,
            "pre_main_lock_validated_before_artifact_creation": True,
        },
        "artifacts": {
            "metrics_path": str(metrics_path.resolve()),
            "metrics_sha256": file_sha256(metrics_path),
        },
    }
    finalize_run(
        experiment_id=EXPERIMENT_ID,
        artifact_root=args.artifact_root,
        run_dir=run_dir,
        report=report,
    )
    print(
        f"[{EXPERIMENT_ID}] {report['execution_status']} "
        f"{gate['status']}: {run_dir}"
    )


if __name__ == "__main__":
    main()
