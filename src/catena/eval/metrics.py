from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    episode_id: str
    query_id: str
    query_kind: str
    policy: str
    prediction_index: int
    gold_index: int
    exact_prediction_index: int
    teacher_correct: bool
    logit_kl: float | None = None
    latency_ms: float | None = None
    state_bytes: int | None = None
    domain: str | None = None
    operation: str | None = None
    history_tokens: int | None = None
    dependency_depth: int | None = None
    query_gap_tokens: int | None = None
    chain_length: int | None = None


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def harmonic_mean(a: float, b: float) -> float:
    if not math.isfinite(a) or not math.isfinite(b) or a + b <= 0:
        return float("nan")
    return 2.0 * a * b / (a + b)


def aggregate(records: Iterable[PredictionRecord]) -> dict[str, float]:
    rows = list(records)
    if not rows:
        return {}
    gold = [float(r.prediction_index == r.gold_index) for r in rows]
    oracle = [float(r.prediction_index == r.exact_prediction_index) for r in rows]
    teacher_correct_rows = [r for r in rows if r.teacher_correct]
    affected = [
        r
        for r in rows
        if r.query_kind in {"affected_direct", "affected_derived", "old_rule_probe", "tool_call"}
    ]
    retained = [r for r in rows if r.query_kind == "unaffected"]
    c_update = _safe_mean(
        [float(r.prediction_index == r.exact_prediction_index) for r in affected]
    )
    c_retain = _safe_mean(
        [float(r.prediction_index == r.exact_prediction_index) for r in retained]
    )
    stale_rows = [r for r in rows if r.query_kind == "old_rule_probe"]
    stale_action_rate = _safe_mean(
        [float(r.prediction_index != r.gold_index) for r in stale_rows]
    )
    return {
        "gold_accuracy": _safe_mean(gold),
        "own_oracle_agreement": _safe_mean(oracle),
        "teacher_correct_agreement": _safe_mean(
            [float(r.prediction_index == r.exact_prediction_index) for r in teacher_correct_rows]
        ),
        "teacher_gold_accuracy": _safe_mean([float(r.teacher_correct) for r in rows]),
        "c_update": c_update,
        "c_retain": c_retain,
        "c_joint": harmonic_mean(c_update, c_retain),
        "stale_action_rate": stale_action_rate,
        "mean_logit_kl": _safe_mean(
            [r.logit_kl for r in rows if r.logit_kl is not None]
        ),
        "mean_latency_ms": _safe_mean(
            [r.latency_ms for r in rows if r.latency_ms is not None]
        ),
        "mean_state_bytes": _safe_mean(
            [float(r.state_bytes) for r in rows if r.state_bytes is not None]
        ),
        "n": float(len(rows)),
    }


def aggregate_by(
    records: Iterable[PredictionRecord], key: str
) -> dict[str, dict[str, float]]:
    groups: dict[str, list[PredictionRecord]] = defaultdict(list)
    for record in records:
        value = getattr(record, key)
        groups[str(value)].append(record)
    return {group: aggregate(group_rows) for group, group_rows in groups.items()}


def stratified_summary(records: Iterable[PredictionRecord]) -> dict[str, object]:
    rows = list(records)
    return {
        "overall": aggregate(rows),
        "by_query_kind": aggregate_by(rows, "query_kind"),
        "by_domain": aggregate_by(rows, "domain"),
        "by_operation": aggregate_by(rows, "operation"),
        "by_history_tokens": aggregate_by(rows, "history_tokens"),
        "by_dependency_depth": aggregate_by(rows, "dependency_depth"),
        "by_query_gap_tokens": aggregate_by(rows, "query_gap_tokens"),
    }


def drift_slope(chain_lengths: list[int], divergences: list[float]) -> float:
    if len(chain_lengths) < 2:
        return float("nan")
    x = np.asarray(chain_lengths, dtype=float)
    y = np.asarray(divergences, dtype=float)
    return float(np.polyfit(x, y, deg=1)[0])
