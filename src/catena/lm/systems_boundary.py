from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class PolicyMeasurement:
    policy: str
    update_cost_us: float
    fixed_cost_us: float
    per_query_cost_us: float
    resident_bytes: int
    host_device_bytes_per_update: int = 0
    quality_pass: bool = True

    def total_cost_us(self, queries_per_update: int) -> float:
        if queries_per_update < 0:
            raise ValueError("queries_per_update must be non-negative")
        return (
            self.update_cost_us + self.fixed_cost_us + queries_per_update * self.per_query_cost_us
        )


@dataclass(frozen=True)
class BreakEven:
    recurrent_policy: str
    baseline_policy: str
    query_count: float | None
    reason: str


def break_even_queries(recurrent: PolicyMeasurement, baseline: PolicyMeasurement) -> BreakEven:
    if not recurrent.quality_pass or not baseline.quality_pass:
        return BreakEven(recurrent.policy, baseline.policy, None, "quality_gate_failed")
    denominator = baseline.per_query_cost_us - recurrent.per_query_cost_us
    numerator = (
        recurrent.update_cost_us
        + recurrent.fixed_cost_us
        - baseline.update_cost_us
        - baseline.fixed_cost_us
    )
    if denominator <= 0:
        return BreakEven(recurrent.policy, baseline.policy, None, "no_per_query_advantage")
    threshold = max(0.0, numerator / denominator)
    return BreakEven(recurrent.policy, baseline.policy, threshold, "finite_break_even")


def quality_constrained_pareto(
    measurements: Iterable[PolicyMeasurement],
    *,
    queries_per_update: int,
) -> list[PolicyMeasurement]:
    candidates = [item for item in measurements if item.quality_pass]
    frontier: list[PolicyMeasurement] = []
    for item in candidates:
        item_cost = item.total_cost_us(queries_per_update)
        dominated = False
        for other in candidates:
            if other is item:
                continue
            if (
                other.total_cost_us(queries_per_update) <= item_cost
                and other.resident_bytes <= item.resident_bytes
                and (
                    other.total_cost_us(queries_per_update) < item_cost
                    or other.resident_bytes < item.resident_bytes
                )
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(item)
    return sorted(frontier, key=lambda value: value.total_cost_us(queries_per_update))
