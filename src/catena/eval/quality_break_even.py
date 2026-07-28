from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import StrEnum
from statistics import median
from time import perf_counter
from typing import Any

import torch

REGISTERED_QUERY_COUNTS = (1, 2, 4, 8, 16, 32, 64)
REGISTERED_CORRECTION_EPSILON = 0.001
REGISTERED_RETENTION_EPSILON = 0.0005


class BreakEvenPolicy(StrEnum):
    ONE_TIME_INTERNAL_ASSIMILATION = "one_time_internal_assimilation"
    EXTERNAL_CANONICAL_STATE_PER_QUERY = (
        "external_canonical_state_per_query"
    )
    RETRIEVE_ONCE_CACHED_COMPACT_SNAPSHOT = (
        "retrieve_once_cached_compact_snapshot"
    )
    FULL_REFRESH = "full_refresh"


REGISTERED_POLICIES = tuple(BreakEvenPolicy)
INTERNAL_POLICY = BreakEvenPolicy.ONE_TIME_INTERNAL_ASSIMILATION
CACHED_POLICY = BreakEvenPolicy.RETRIEVE_ONCE_CACHED_COMPACT_SNAPSHOT
BASELINE_POLICIES = (
    BreakEvenPolicy.EXTERNAL_CANONICAL_STATE_PER_QUERY,
    CACHED_POLICY,
    BreakEvenPolicy.FULL_REFRESH,
)


@dataclass(frozen=True, slots=True)
class StructuredBreakEvenWorkload:
    initial_state_cpu: torch.Tensor
    canonical_state_cpu: torch.Tensor
    affected_address_cpu: torch.Tensor
    replacement_value_cpu: torch.Tensor
    query_addresses_cpu: torch.Tensor
    compact_addresses_cpu: torch.Tensor
    compact_inverse_cpu: torch.Tensor
    seed: int
    query_count: int
    base_workload_sha256: str
    paired_workload_sha256: str

    @property
    def batch_size(self) -> int:
        return int(self.initial_state_cpu.shape[0])

    @property
    def slots(self) -> int:
        return int(self.initial_state_cpu.shape[1])

    @property
    def value_dim(self) -> int:
        return int(self.initial_state_cpu.shape[2])


@dataclass(slots=True)
class PreparedPolicy:
    policy: BreakEvenPolicy
    workload: StructuredBreakEvenWorkload
    device: torch.device
    query_addresses_device: torch.Tensor
    batch_indices_device: torch.Tensor
    mutable_internal_state: torch.Tensor | None = None
    affected_address_device: torch.Tensor | None = None
    replacement_value_device: torch.Tensor | None = None
    compact_inverse_device: torch.Tensor | None = None


def _update_digest_with_tensor(
    digest: hashlib._Hash,
    *,
    name: str,
    tensor: torch.Tensor,
) -> None:
    value = tensor.detach().cpu().contiguous()
    digest.update(name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(repr(tuple(value.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(value.numpy().tobytes())
    digest.update(b"\0")


def _workload_digest(
    *,
    tensors: dict[str, torch.Tensor],
    metadata: dict[str, int | float | str],
) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            metadata,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(b"\0")
    for name in sorted(tensors):
        _update_digest_with_tensor(
            digest,
            name=name,
            tensor=tensors[name],
        )
    return digest.hexdigest()


def generate_structured_break_even_workload(
    *,
    batch_size: int,
    slots: int,
    value_dim: int,
    state_scale: float,
    seed: int,
    query_count: int,
) -> StructuredBreakEvenWorkload:
    if batch_size <= 0 or value_dim <= 0:
        raise ValueError("batch_size and value_dim must be positive")
    if query_count <= 0:
        raise ValueError("query_count must be positive")
    if slots <= query_count:
        raise ValueError(
            "slots must exceed query_count so every retention address is "
            "unaffected and unique"
        )
    if not math.isfinite(state_scale) or state_scale <= 0:
        raise ValueError("state_scale must be finite and positive")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    initial_state = (
        torch.randn(
            batch_size,
            slots,
            value_dim,
            generator=generator,
            dtype=torch.float32,
        )
        * state_scale
    )
    affected_address = torch.randint(
        0,
        slots,
        (batch_size,),
        generator=generator,
        dtype=torch.int64,
    )
    replacement_value = (
        torch.randn(
            batch_size,
            value_dim,
            generator=generator,
            dtype=torch.float32,
        )
        * state_scale
    )
    canonical_state = initial_state.clone()
    batch_indices = torch.arange(batch_size)
    canonical_state[batch_indices, affected_address] = replacement_value

    offsets = torch.arange(1, query_count + 1, dtype=torch.int64)
    retention_addresses = (
        affected_address[:, None] + offsets[None, :]
    ) % slots
    affected_queries = affected_address[:, None].expand(-1, query_count)
    query_addresses = torch.stack(
        (affected_queries, retention_addresses),
        dim=2,
    )
    compact_addresses = torch.cat(
        (affected_address[:, None], retention_addresses),
        dim=1,
    )
    compact_inverse = torch.empty(
        batch_size,
        query_count,
        2,
        dtype=torch.int64,
    )
    compact_inverse[:, :, 0] = 0
    compact_inverse[:, :, 1] = torch.arange(
        1,
        query_count + 1,
        dtype=torch.int64,
    )[None, :]

    base_tensors = {
        "affected_address": affected_address,
        "canonical_state": canonical_state,
        "initial_state": initial_state,
        "replacement_value": replacement_value,
    }
    base_metadata: dict[str, int | float | str] = {
        "schema": "CATENA_E20_STRUCTURED_WORKLOAD_V1",
        "batch_size": batch_size,
        "slots": slots,
        "value_dim": value_dim,
        "state_scale": state_scale,
        "seed": seed,
    }
    base_digest = _workload_digest(
        tensors=base_tensors,
        metadata=base_metadata,
    )
    paired_digest = _workload_digest(
        tensors={
            **base_tensors,
            "compact_addresses": compact_addresses,
            "compact_inverse": compact_inverse,
            "query_addresses": query_addresses,
        },
        metadata={
            **base_metadata,
            "query_count": query_count,
            "base_workload_sha256": base_digest,
        },
    )
    return StructuredBreakEvenWorkload(
        initial_state_cpu=initial_state,
        canonical_state_cpu=canonical_state,
        affected_address_cpu=affected_address,
        replacement_value_cpu=replacement_value,
        query_addresses_cpu=query_addresses,
        compact_addresses_cpu=compact_addresses,
        compact_inverse_cpu=compact_inverse,
        seed=seed,
        query_count=query_count,
        base_workload_sha256=base_digest,
        paired_workload_sha256=paired_digest,
    )


def _gather_state(
    state: torch.Tensor,
    addresses: torch.Tensor,
) -> torch.Tensor:
    if state.ndim != 3 or addresses.ndim != 3:
        raise ValueError("state and addresses must have ranks 3 and 3")
    if state.shape[0] != addresses.shape[0]:
        raise ValueError("state/address batch sizes differ")
    batch_indices = torch.arange(
        state.shape[0],
        device=state.device,
    )[:, None, None]
    return state[batch_indices, addresses]


def prepare_policy(
    workload: StructuredBreakEvenWorkload,
    *,
    policy: BreakEvenPolicy,
    device: torch.device,
) -> PreparedPolicy:
    query_addresses = workload.query_addresses_cpu.to(device=device)
    batch_indices = torch.arange(
        workload.batch_size,
        device=device,
    )
    prepared = PreparedPolicy(
        policy=policy,
        workload=workload,
        device=device,
        query_addresses_device=query_addresses,
        batch_indices_device=batch_indices,
    )
    if policy is INTERNAL_POLICY:
        prepared.mutable_internal_state = workload.initial_state_cpu.to(
            device=device
        )
        prepared.affected_address_device = (
            workload.affected_address_cpu.to(device=device)
        )
        prepared.replacement_value_device = (
            workload.replacement_value_cpu.to(device=device)
        )
    elif policy is CACHED_POLICY:
        prepared.compact_inverse_device = (
            workload.compact_inverse_cpu.to(device=device)
        )
    return prepared


def execute_prepared_policy(prepared: PreparedPolicy) -> torch.Tensor:
    workload = prepared.workload
    policy = prepared.policy
    if policy is INTERNAL_POLICY:
        state = prepared.mutable_internal_state
        affected_address = prepared.affected_address_device
        replacement_value = prepared.replacement_value_device
        if (
            state is None
            or affected_address is None
            or replacement_value is None
        ):
            raise RuntimeError("Internal-assimilation context is incomplete")
        state[
            prepared.batch_indices_device,
            affected_address,
        ] = replacement_value
        return _gather_state(
            state,
            prepared.query_addresses_device,
        )

    if policy is BreakEvenPolicy.EXTERNAL_CANONICAL_STATE_PER_QUERY:
        cpu_batch = torch.arange(workload.batch_size)[:, None]
        answers: list[torch.Tensor] = []
        for query_index in range(workload.query_count):
            pair_cpu = workload.canonical_state_cpu[
                cpu_batch,
                workload.query_addresses_cpu[:, query_index, :],
            ]
            answers.append(pair_cpu.to(device=prepared.device))
        return torch.stack(answers, dim=1)

    if policy is CACHED_POLICY:
        inverse = prepared.compact_inverse_device
        if inverse is None:
            raise RuntimeError("Cached-snapshot context is incomplete")
        cpu_batch = torch.arange(workload.batch_size)[:, None]
        compact_cpu = workload.canonical_state_cpu[
            cpu_batch,
            workload.compact_addresses_cpu,
        ]
        compact_device = compact_cpu.to(device=prepared.device)
        return _gather_state(compact_device, inverse)

    if policy is BreakEvenPolicy.FULL_REFRESH:
        refreshed = workload.canonical_state_cpu.to(
            device=prepared.device,
            copy=True,
        )
        return _gather_state(
            refreshed,
            prepared.query_addresses_device,
        )

    raise ValueError(f"Unsupported E20 policy: {policy}")


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def policy_quality_metrics(
    *,
    output: torch.Tensor,
    workload: StructuredBreakEvenWorkload,
) -> dict[str, float]:
    if output.shape != (
        workload.batch_size,
        workload.query_count,
        2,
        workload.value_dim,
    ):
        raise ValueError(
            f"Unexpected policy output shape: {tuple(output.shape)}"
        )
    expected_affected = workload.replacement_value_cpu.to(
        device=output.device
    )[:, None, :].expand(-1, workload.query_count, -1)
    cpu_batch = torch.arange(workload.batch_size)[:, None]
    expected_retention = workload.initial_state_cpu[
        cpu_batch,
        workload.query_addresses_cpu[:, :, 1],
    ].to(device=output.device)
    correction_delta = output[:, :, 0] - expected_affected
    retention_delta = output[:, :, 1] - expected_retention
    return {
        "affected_correction_mse": float(
            correction_delta.square().mean().item()
        ),
        "retention_mse": float(
            retention_delta.square().mean().item()
        ),
        "maximum_absolute_error": float(
            torch.maximum(
                correction_delta.abs().max(),
                retention_delta.abs().max(),
            ).item()
        ),
    }


def benchmark_policy(
    *,
    workload: StructuredBreakEvenWorkload,
    policy: BreakEvenPolicy,
    device: torch.device,
    warmup_repeats: int,
    measured_repeats: int,
) -> dict[str, Any]:
    if warmup_repeats < 0 or measured_repeats <= 0:
        raise ValueError("Invalid timing repeat counts")

    quality_context = prepare_policy(
        workload,
        policy=policy,
        device=device,
    )
    quality_output = execute_prepared_policy(quality_context)
    synchronize_device(device)
    quality = policy_quality_metrics(
        output=quality_output,
        workload=workload,
    )

    for _ in range(warmup_repeats):
        context = prepare_policy(
            workload,
            policy=policy,
            device=device,
        )
        synchronize_device(device)
        execute_prepared_policy(context)
        synchronize_device(device)

    samples: list[float] = []
    for _ in range(measured_repeats):
        context = prepare_policy(
            workload,
            policy=policy,
            device=device,
        )
        synchronize_device(device)
        started = perf_counter()
        execute_prepared_policy(context)
        synchronize_device(device)
        samples.append(perf_counter() - started)

    ordered = sorted(samples)
    return {
        "policy": policy.value,
        "query_count": workload.query_count,
        "logical_reads_per_query": 2,
        "batch_size": workload.batch_size,
        "base_workload_sha256": workload.base_workload_sha256,
        "paired_workload_sha256": workload.paired_workload_sha256,
        "warmup_repeats": warmup_repeats,
        "measured_repeats": measured_repeats,
        "device_synchronized_before_after_measurement": (
            device.type == "cuda"
        ),
        "latency_total_seconds_samples": samples,
        "latency_total_seconds_median": float(median(samples)),
        "latency_total_seconds_minimum": float(ordered[0]),
        "latency_total_seconds_maximum": float(ordered[-1]),
        "latency_seconds_per_logical_query_median": float(
            median(samples) / (workload.batch_size * workload.query_count)
        ),
        **quality,
    }


def _require_complete_rows(
    rows: list[dict[str, Any]],
    *,
    query_counts: tuple[int, ...],
) -> dict[tuple[BreakEvenPolicy, int], dict[str, Any]]:
    expected = {
        (policy, query_count)
        for policy in REGISTERED_POLICIES
        for query_count in query_counts
    }
    indexed: dict[tuple[BreakEvenPolicy, int], dict[str, Any]] = {}
    for row in rows:
        key = (
            BreakEvenPolicy(str(row["policy"])),
            int(row["query_count"]),
        )
        if key in indexed:
            raise ValueError(f"Duplicate E20 metric cell: {key}")
        indexed[key] = row
    if set(indexed) != expected:
        missing = sorted(expected - set(indexed))
        extra = sorted(set(indexed) - expected)
        raise ValueError(
            f"Incomplete E20 metric grid: missing={missing}, extra={extra}"
        )
    for query_count in query_counts:
        digests = {
            str(indexed[(policy, query_count)]["paired_workload_sha256"])
            for policy in REGISTERED_POLICIES
        }
        if len(digests) != 1:
            raise ValueError(
                f"Policies are not workload-paired at m={query_count}"
            )
    return indexed


def assess_quality_constrained_break_even(
    rows: list[dict[str, Any]],
    *,
    query_counts: tuple[int, ...],
    correction_epsilon: float,
    retention_epsilon: float,
    dry_run: bool,
) -> dict[str, Any]:
    if tuple(query_counts) != REGISTERED_QUERY_COUNTS:
        raise ValueError("E20 query grid differs from the registered grid")
    if correction_epsilon != REGISTERED_CORRECTION_EPSILON:
        raise ValueError("E20 correction epsilon changed")
    if retention_epsilon != REGISTERED_RETENTION_EPSILON:
        raise ValueError("E20 retention epsilon changed")
    indexed = _require_complete_rows(rows, query_counts=query_counts)

    def eligible(row: dict[str, Any]) -> bool:
        values = (
            float(row["affected_correction_mse"]),
            float(row["retention_mse"]),
            float(row["latency_total_seconds_median"]),
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("E20 metric row contains an invalid value")
        return bool(
            values[0] <= correction_epsilon
            and values[1] <= retention_epsilon
        )

    comparisons: dict[str, Any] = {}
    for baseline in BASELINE_POLICIES:
        qualifying: list[int] = []
        cell_records: list[dict[str, Any]] = []
        for query_count in query_counts:
            internal = indexed[(INTERNAL_POLICY, query_count)]
            baseline_row = indexed[(baseline, query_count)]
            quality_pass = eligible(internal) and eligible(baseline_row)
            internal_latency = float(
                internal["latency_total_seconds_median"]
            )
            baseline_latency = float(
                baseline_row["latency_total_seconds_median"]
            )
            latency_break_even = internal_latency <= baseline_latency
            if quality_pass and latency_break_even:
                qualifying.append(query_count)
            cell_records.append(
                {
                    "query_count": query_count,
                    "quality_guardrails_pass": quality_pass,
                    "internal_latency_seconds": internal_latency,
                    "baseline_latency_seconds": baseline_latency,
                    "internal_minus_baseline_seconds": (
                        internal_latency - baseline_latency
                    ),
                    "internal_breaks_even": (
                        quality_pass and latency_break_even
                    ),
                }
            )
        comparisons[baseline.value] = {
            "minimum_quality_constrained_break_even_m": (
                min(qualifying) if qualifying else None
            ),
            "break_even_observed": bool(qualifying),
            "cells": cell_records,
        }

    cached_cells = comparisons[CACHED_POLICY.value]["cells"]
    cached_dominates = all(
        bool(cell["quality_guardrails_pass"])
        and float(cell["baseline_latency_seconds"])
        <= float(cell["internal_latency_seconds"])
        for cell in cached_cells
    )
    all_break_evens_observed = all(
        bool(record["break_even_observed"])
        for record in comparisons.values()
    )
    if dry_run:
        status = "NOT_EVALUATED_DRY_RUN"
        claim_open = False
        reason = "CPU_DRY_RUN_IS_NON_EVIDENTIARY"
    elif cached_dominates:
        status = "NOT_SUPPORTED_BOUNDARY"
        claim_open = False
        reason = "CACHED_COMPACT_SNAPSHOT_DOMINATES_REGISTERED_GRID"
    elif all_break_evens_observed:
        status = "SUPPORTED_CONTROLLED_SYSTEMS_PROXY"
        claim_open = True
        reason = "ALL_REGISTERED_BASELINE_BREAK_EVENS_OBSERVED"
    else:
        status = "NOT_SUPPORTED_BOUNDARY"
        claim_open = False
        reason = "REGISTERED_GRID_CONTAINS_NO_BREAK_EVEN_FOR_A_BASELINE"

    return {
        "status": status,
        "claim_open": claim_open,
        "reason": reason,
        "correction_epsilon": correction_epsilon,
        "retention_epsilon": retention_epsilon,
        "latency_relation": (
            "one_time_internal_assimilation median total latency "
            "<= baseline median total latency"
        ),
        "minimum_m_by_baseline": {
            baseline: record[
                "minimum_quality_constrained_break_even_m"
            ]
            for baseline, record in comparisons.items()
        },
        "cached_snapshot_dominates_registered_grid": cached_dominates,
        "comparisons": comparisons,
    }
