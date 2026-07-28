from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from catena.core.config import load_config
from catena.eval.quality_break_even import (
    BASELINE_POLICIES,
    CACHED_POLICY,
    INTERNAL_POLICY,
    REGISTERED_CORRECTION_EPSILON,
    REGISTERED_POLICIES,
    REGISTERED_QUERY_COUNTS,
    REGISTERED_RETENTION_EPSILON,
    BreakEvenPolicy,
    StructuredBreakEvenWorkload,
    assess_quality_constrained_break_even,
    benchmark_policy,
    execute_prepared_policy,
    generate_structured_break_even_workload,
    policy_quality_metrics,
    prepare_policy,
)
from experiments.e20_quality_constrained_break_even import (
    EVIDENCE_TIER,
    LOCK_PATH,
    validate_e20_config,
    validate_e20_protocol_lock,
    validate_execution_device,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/e20_quality_constrained_break_even.yaml"


def _workload(query_count: int = 4) -> StructuredBreakEvenWorkload:
    return generate_structured_break_even_workload(
        batch_size=3,
        slots=10,
        value_dim=6,
        state_scale=0.5,
        seed=200,
        query_count=query_count,
    )


def test_structured_workload_is_nested_and_digest_sealed() -> None:
    small = _workload(query_count=2)
    large = _workload(query_count=4)
    assert small.base_workload_sha256 == large.base_workload_sha256
    assert small.paired_workload_sha256 != large.paired_workload_sha256
    assert torch.equal(
        small.query_addresses_cpu,
        large.query_addresses_cpu[:, :2],
    )
    assert torch.all(
        large.query_addresses_cpu[:, :, 0]
        != large.query_addresses_cpu[:, :, 1]
    )
    assert torch.equal(
        large.canonical_state_cpu[
            torch.arange(large.batch_size),
            large.affected_address_cpu,
        ],
        large.replacement_value_cpu,
    )


def test_all_registered_policies_answer_the_same_workload_exactly() -> None:
    workload = _workload()
    outputs: dict[BreakEvenPolicy, torch.Tensor] = {}
    for policy in REGISTERED_POLICIES:
        prepared = prepare_policy(
            workload,
            policy=policy,
            device=torch.device("cpu"),
        )
        output = execute_prepared_policy(prepared)
        outputs[policy] = output
        quality = policy_quality_metrics(
            output=output,
            workload=workload,
        )
        assert quality["affected_correction_mse"] == 0.0
        assert quality["retention_mse"] == 0.0
    reference = outputs[INTERNAL_POLICY]
    assert all(
        torch.equal(reference, output)
        for output in outputs.values()
    )


def test_cpu_benchmark_records_warmup_repeats_and_quality() -> None:
    result = benchmark_policy(
        workload=_workload(query_count=2),
        policy=CACHED_POLICY,
        device=torch.device("cpu"),
        warmup_repeats=1,
        measured_repeats=3,
    )
    assert len(result["latency_total_seconds_samples"]) == 3
    assert result["warmup_repeats"] == 1
    assert result["measured_repeats"] == 3
    assert result["device_synchronized_before_after_measurement"] is False
    assert result["affected_correction_mse"] == 0.0
    assert result["retention_mse"] == 0.0


def _synthetic_rows(
    *,
    cached_dominates: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for query_count in REGISTERED_QUERY_COUNTS:
        latencies = {
            INTERNAL_POLICY: 1.0,
            BreakEvenPolicy.EXTERNAL_CANONICAL_STATE_PER_QUERY: (
                query_count / 2
            ),
            CACHED_POLICY: 0.5 if cached_dominates else 1.2,
            BreakEvenPolicy.FULL_REFRESH: (
                0.8 if query_count == 1 else 1.2
            ),
        }
        digest = f"digest-{query_count}"
        for policy in REGISTERED_POLICIES:
            rows.append(
                {
                    "policy": policy.value,
                    "query_count": query_count,
                    "paired_workload_sha256": digest,
                    "affected_correction_mse": 0.0,
                    "retention_mse": 0.0,
                    "latency_total_seconds_median": latencies[policy],
                }
            )
    return rows


def test_primary_estimand_reports_minimum_m_per_baseline() -> None:
    result = assess_quality_constrained_break_even(
        _synthetic_rows(cached_dominates=False),
        query_counts=REGISTERED_QUERY_COUNTS,
        correction_epsilon=REGISTERED_CORRECTION_EPSILON,
        retention_epsilon=REGISTERED_RETENTION_EPSILON,
        dry_run=False,
    )
    assert result["status"] == "SUPPORTED_CONTROLLED_SYSTEMS_PROXY"
    assert result["minimum_m_by_baseline"] == {
        BreakEvenPolicy.EXTERNAL_CANONICAL_STATE_PER_QUERY.value: 2,
        CACHED_POLICY.value: 1,
        BreakEvenPolicy.FULL_REFRESH.value: 2,
    }


def test_cached_snapshot_dominance_is_a_not_supported_boundary() -> None:
    result = assess_quality_constrained_break_even(
        _synthetic_rows(cached_dominates=True),
        query_counts=REGISTERED_QUERY_COUNTS,
        correction_epsilon=REGISTERED_CORRECTION_EPSILON,
        retention_epsilon=REGISTERED_RETENTION_EPSILON,
        dry_run=False,
    )
    assert result["status"] == "NOT_SUPPORTED_BOUNDARY"
    assert result["claim_open"] is False
    assert result["cached_snapshot_dominates_registered_grid"] is True
    assert (
        result["reason"]
        == "CACHED_COMPACT_SNAPSHOT_DOMINATES_REGISTERED_GRID"
    )


def test_dry_run_never_opens_the_claim() -> None:
    result = assess_quality_constrained_break_even(
        _synthetic_rows(cached_dominates=False),
        query_counts=REGISTERED_QUERY_COUNTS,
        correction_epsilon=REGISTERED_CORRECTION_EPSILON,
        retention_epsilon=REGISTERED_RETENTION_EPSILON,
        dry_run=True,
    )
    assert result["status"] == "NOT_EVALUATED_DRY_RUN"
    assert result["claim_open"] is False


def test_quality_failure_excludes_an_apparent_latency_crossing() -> None:
    rows = _synthetic_rows(cached_dominates=False)
    for row in rows:
        if (
            row["policy"]
            == BreakEvenPolicy.EXTERNAL_CANONICAL_STATE_PER_QUERY.value
            and row["query_count"] == 2
        ):
            row["affected_correction_mse"] = 0.01
    result = assess_quality_constrained_break_even(
        rows,
        query_counts=REGISTERED_QUERY_COUNTS,
        correction_epsilon=REGISTERED_CORRECTION_EPSILON,
        retention_epsilon=REGISTERED_RETENTION_EPSILON,
        dry_run=False,
    )
    assert result["minimum_m_by_baseline"][
        BreakEvenPolicy.EXTERNAL_CANONICAL_STATE_PER_QUERY.value
    ] == 4


def test_config_and_lock_preserve_registered_boundaries() -> None:
    config = load_config(CONFIG_PATH)
    validate_e20_config(config)
    assert EVIDENCE_TIER == "CONTROLLED_SYSTEMS_PROXY"
    assert config["evidence"]["scientific_evidence"] is False
    assert config["quality_guardrails"] == {
        "affected_correction_mse_epsilon": 0.001,
        "retention_mse_epsilon": 0.0005,
    }
    assert tuple(config["query_counts"]) == REGISTERED_QUERY_COUNTS
    assert validate_e20_protocol_lock(CONFIG_PATH)
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    assert lock["main_evaluation_started"] is False
    assert lock["gpu_main_executed"] is False


def test_cpu_dry_and_cuda_main_device_contract_is_explicit() -> None:
    validate_execution_device(
        device=torch.device("cpu"),
        dry_run=True,
    )
    validate_execution_device(
        device=torch.device("cuda:0"),
        dry_run=False,
    )
    with pytest.raises(RuntimeError, match="MAIN requires"):
        validate_execution_device(
            device=torch.device("cpu"),
            dry_run=False,
        )
    with pytest.raises(RuntimeError, match="DRY_RUN requires"):
        validate_execution_device(
            device=torch.device("cuda:0"),
            dry_run=True,
        )


def test_registered_baselines_are_complete_and_cached_is_preserved() -> None:
    assert tuple(REGISTERED_POLICIES[1:]) == BASELINE_POLICIES
    assert CACHED_POLICY in BASELINE_POLICIES
