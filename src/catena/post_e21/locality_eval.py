from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from catena.eval.postcore_metrics import exact_sign_flip
from catena.eval.structured_sequence_localization_r1 import (
    compute_e21b_r1_seed_contrasts,
)
from catena.post_e21.locality_data import LocalityMethod
from catena.post_e21.locality_protocol import threshold_float

CONDITION_B = "B_learned_address_oracle_candidate"
CONDITION_C = "C_oracle_address_state_read_candidate"
CONDITION_D = "D_learned_address_state_read_candidate"
ADDRESS_FAMILY = "address_decoupling"
STATE_FAMILY = "state_conditioning"
SEPARATE_VARIANTS = frozenset({"separate_address", "full"})
STATE_READ_VARIANTS = frozenset({"state_aware", "full"})
RECOVERY_KEYS = (
    "b_separate_address_gain",
    "c_state_read_gain",
    "d_full_only_gain",
)
STRESS_KEYS = ("b_stress_gain", "c_stress_gain", "d_stress_gain")


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        raise ValueError("Cannot average an empty E22 collection")
    return sum(materialized) / len(materialized)


def _finite(row: Mapping[str, Any], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Missing/invalid E22 metric {key!r}") from error
    if not math.isfinite(value):
        raise ValueError(f"Non-finite E22 metric {key!r}")
    return value


def _row_key(
    row: Mapping[str, Any],
) -> tuple[int, str, str, str, str, int, int]:
    return (
        int(row["seed"]),
        str(row["method_id"]),
        str(row["variant"]),
        str(row["condition"]),
        str(row["demand_family"]),
        int(row["updates"]),
        int(row["gap_events"]),
    )


def validate_paired_metric_grid(
    rows: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int],
    methods: Sequence[str],
    variants: Sequence[str],
    conditions: Sequence[str],
    demand_families: Sequence[str],
    updates_grid: Sequence[int],
    gaps_grid: Sequence[int],
) -> None:
    expected = {
        (seed, method, variant, condition, family, updates, gap)
        for seed in seeds
        for method in methods
        for variant in variants
        for condition in conditions
        for family in demand_families
        for updates in updates_grid
        for gap in gaps_grid
    }
    observed = [_row_key(row) for row in rows]
    if len(observed) != len(set(observed)):
        raise ValueError("E22 paired metric grid contains duplicate cells")
    if set(observed) != expected:
        missing = len(expected - set(observed))
        extra = len(set(observed) - expected)
        raise ValueError(f"E22 paired metric grid mismatch: {missing=}, {extra=}")
    by_base: dict[tuple[int, str, str, int, int], set[str]] = {}
    for row in rows:
        base = (
            int(row["seed"]),
            str(row["condition"]),
            str(row["demand_family"]),
            int(row["updates"]),
            int(row["gap_events"]),
        )
        by_base.setdefault(base, set()).add(str(row["base_transaction_digest"]))
    if any(len(digests) != 1 for digests in by_base.values()):
        raise ValueError("E22 methods/variants did not share paired evaluation transactions")


def _active_axes(row: Mapping[str, Any]) -> list[str]:
    condition = str(row["condition"])
    variant = str(row["variant"])
    axes: list[str] = []
    if condition in {CONDITION_B, CONDITION_D} and variant in SEPARATE_VARIANTS:
        axes.append("separate_address")
    if condition in {CONDITION_C, CONDITION_D} and variant in STATE_READ_VARIANTS:
        axes.append("state_read")
    return axes


def _is_identifying_target(row: Mapping[str, Any], axes: Sequence[str]) -> bool:
    condition = str(row["condition"])
    family = str(row["demand_family"])
    return bool(
        (condition == CONDITION_B and family == ADDRESS_FAMILY and "separate_address" in axes)
        or (condition == CONDITION_C and family == STATE_FAMILY and "state_read" in axes)
        or (condition == CONDITION_D and family == ADDRESS_FAMILY and str(row["variant"]) == "full")
    )


def build_active_cell_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Materialize every cell whose architectural route is actually active."""

    result: list[dict[str, Any]] = []
    for row in rows:
        axes = _active_axes(row)
        if not axes:
            continue
        identifying = _is_identifying_target(row, axes)
        result.append(
            {
                "seed": int(row["seed"]),
                "method_id": str(row["method_id"]),
                "variant": str(row["variant"]),
                "condition": str(row["condition"]),
                "demand_family": str(row["demand_family"]),
                "updates": int(row["updates"]),
                "gap_events": int(row["gap_events"]),
                "active_axes": axes,
                "cell_role": "identifying_target" if identifying else "active_nontarget",
                "affected_mse": _finite(row, "affected_mse"),
                "retention_mse": _finite(row, "retention_mse"),
                "raw_route_mask_sha256": str(row["raw_route_mask_sha256"]),
                "active_route_mask_sha256": str(row["active_route_mask_sha256"]),
                "raw_route_support_size": _finite(row, "raw_route_support_size"),
                "raw_route_support_fraction": _finite(row, "raw_route_support_fraction"),
                "active_route_support_size": _finite(row, "active_route_support_size"),
                "active_route_support_fraction": _finite(row, "active_route_support_fraction"),
                "active_event_fraction": _finite(row, "active_event_fraction"),
                "post_mask_update_rms": _finite(row, "post_mask_update_rms"),
                "predicted_update_rms": _finite(row, "predicted_update_rms"),
                "target_update_rms": _finite(row, "target_update_rms"),
                "update_compute_units": _finite(row, "update_compute_units"),
                "base_transaction_digest": str(row["base_transaction_digest"]),
                "checkpoint_sha256": str(row["checkpoint_sha256"]),
            }
        )
    if not result:
        raise ValueError("E22 active-cell artifact would be empty")
    return result


def compute_locality_seed_summaries(
    rows: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int],
    method_ids: Sequence[str],
    updates_grid: Sequence[int],
    gaps_grid: Sequence[int],
    demand_families: Sequence[str],
    stress_updates: int,
    stress_gap_events: int,
) -> list[dict[str, Any]]:
    """Recompute the frozen E21 B/C/D and R1 guards independently per method."""

    result: list[dict[str, Any]] = []
    for method_id in method_ids:
        method_rows = [dict(row) for row in rows if str(row["method_id"]) == method_id]
        contrasts = compute_e21b_r1_seed_contrasts(
            method_rows,
            seeds=[int(seed) for seed in seeds],
            updates_grid=[int(value) for value in updates_grid],
            gaps_grid=[int(value) for value in gaps_grid],
            demand_families=[str(value) for value in demand_families],
            stress_updates=int(stress_updates),
            stress_gap_events=int(stress_gap_events),
        )
        for contrast in contrasts:
            seed = int(contrast["seed"])
            active = [row for row in method_rows if int(row["seed"]) == seed and _active_axes(row)]
            if not active:
                raise ValueError("E22 method lacks active route cells")
            result.append(
                {
                    **contrast,
                    "method_id": method_id,
                    "mean_raw_route_support_size": _mean(
                        _finite(row, "raw_route_support_size") for row in active
                    ),
                    "mean_active_path_support_size": _mean(
                        _finite(row, "raw_route_support_size") for row in active
                    ),
                    "mean_active_route_support_size": _mean(
                        _finite(row, "active_route_support_size") for row in active
                    ),
                    "mean_active_route_support_fraction": _mean(
                        _finite(row, "active_route_support_fraction") for row in active
                    ),
                    "mean_update_compute_units": _mean(
                        _finite(row, "update_compute_units") for row in active
                    ),
                    "mean_post_mask_update_rms": _mean(
                        _finite(row, "post_mask_update_rms") for row in active
                    ),
                }
            )
    return result


def _method_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    method_id: str,
) -> dict[str, Any]:
    selected = [row for row in rows if str(row["method_id"]) == method_id]
    if not selected:
        raise ValueError(f"No E22 seed rows for method {method_id}")
    summary: dict[str, Any] = {
        "method_id": method_id,
        "seed_count": len(selected),
        "maximum_nontarget_degradation": max(
            _finite(row, "maximum_nontarget_degradation") for row in selected
        ),
        "maximum_retention_degradation": max(
            _finite(row, "maximum_retention_degradation") for row in selected
        ),
        "maximum_capable_affected_mse": max(
            _finite(row, "maximum_capable_affected_mse") for row in selected
        ),
        "minimum_capable_address_accuracy": min(
            _finite(row, "minimum_capable_address_accuracy") for row in selected
        ),
        "maximum_capable_candidate_mse": max(
            _finite(row, "maximum_capable_candidate_mse") for row in selected
        ),
        "maximum_oracle_floor_mse": max(
            _finite(row, "maximum_oracle_floor_mse") for row in selected
        ),
        "minimum_verified_activity": min(
            _finite(row, "minimum_verified_activity") for row in selected
        ),
        "maximum_distractor_activity": max(
            _finite(row, "maximum_distractor_activity") for row in selected
        ),
        "mean_active_route_support_size": _mean(
            _finite(row, "mean_active_route_support_size") for row in selected
        ),
        "mean_active_path_support_size": _mean(
            _finite(row, "mean_active_path_support_size") for row in selected
        ),
        "mean_active_route_support_fraction": _mean(
            _finite(row, "mean_active_route_support_fraction") for row in selected
        ),
        "mean_update_compute_units": _mean(
            _finite(row, "mean_update_compute_units") for row in selected
        ),
        "mean_post_mask_update_rms": _mean(
            _finite(row, "mean_post_mask_update_rms") for row in selected
        ),
    }
    for key in RECOVERY_KEYS:
        summary[f"mean_{key}"] = _mean(_finite(row, key) for row in selected)
        summary[f"positive_fraction_{key}"] = _mean(
            1.0 if _finite(row, key) > 0.0 else 0.0 for row in selected
        )
        summary[f"sign_flip_p_{key}"] = exact_sign_flip(
            [_finite(row, key) for row in selected],
            alternative="greater",
        )
    for key in STRESS_KEYS:
        summary[f"minimum_{key}"] = min(_finite(row, key) for row in selected)
    return summary


def _recovery_pattern_passes(
    summary: Mapping[str, Any],
    *,
    thresholds: Mapping[str, float | bool],
    require_significance: bool,
) -> bool:
    for key in RECOVERY_KEYS:
        if _finite(summary, f"mean_{key}") < threshold_float(
            thresholds, "selective_gain"
        ) or _finite(summary, f"positive_fraction_{key}") < threshold_float(
            thresholds, "minimum_seed_direction_fraction"
        ):
            return False
        if require_significance and (
            _finite(summary, f"sign_flip_p_{key}")
            > threshold_float(thresholds, "exact_sign_flip_alpha")
        ):
            return False
    return all(_finite(summary, f"minimum_{key}") > 0.0 for key in STRESS_KEYS)


def _development_recovery_sesoi_passes(
    summary: Mapping[str, Any],
    *,
    thresholds: Mapping[str, float | bool],
) -> bool:
    """Apply only the preregistered E22a mean B/C/D recovery SESOIs."""

    return all(
        _finite(summary, f"mean_{key}") >= threshold_float(thresholds, "selective_gain")
        for key in RECOVERY_KEYS
    )


def _capacity_passes(
    summary: Mapping[str, Any],
    *,
    thresholds: Mapping[str, float | bool],
) -> bool:
    return bool(
        _finite(summary, "maximum_capable_affected_mse")
        <= threshold_float(thresholds, "maximum_capable_affected_mse")
        and _finite(summary, "minimum_capable_address_accuracy")
        >= threshold_float(thresholds, "minimum_address_accuracy")
        and _finite(summary, "maximum_capable_candidate_mse")
        <= threshold_float(thresholds, "maximum_candidate_recovery_mse")
        and _finite(summary, "maximum_oracle_floor_mse")
        <= threshold_float(thresholds, "maximum_oracle_floor_mse")
        and _finite(summary, "minimum_verified_activity")
        >= threshold_float(thresholds, "minimum_verified_activity")
    )


def _retention_passes(
    summary: Mapping[str, Any],
    *,
    thresholds: Mapping[str, float | bool],
) -> bool:
    return bool(
        _finite(summary, "maximum_retention_degradation")
        <= threshold_float(thresholds, "retention_noninferiority")
    )


def _absolute_locality_passes(
    summary: Mapping[str, Any],
    *,
    thresholds: Mapping[str, float | bool],
) -> bool:
    return bool(
        _finite(summary, "maximum_nontarget_degradation")
        <= threshold_float(thresholds, "maximum_nontarget_degradation")
        and _finite(summary, "maximum_distractor_activity")
        <= threshold_float(thresholds, "maximum_distractor_activity")
    )


def select_locality_method(
    seed_rows: Sequence[Mapping[str, Any]],
    *,
    methods: Sequence[LocalityMethod],
    thresholds: Mapping[str, float | bool],
    dry_run: bool,
) -> dict[str, Any]:
    method_summaries: list[dict[str, Any]] = []
    eligible_ids = {method.method_id for method in methods if method.selection_eligible}
    for method in methods:
        summary = _method_summary(seed_rows, method_id=method.method_id)
        summary["selection_eligible"] = method.method_id in eligible_ids
        summary["recovery_gate_passed"] = _development_recovery_sesoi_passes(
            summary,
            thresholds=thresholds,
        )
        summary["seed_and_stress_direction_diagnostic_passed"] = _recovery_pattern_passes(
            summary,
            thresholds=thresholds,
            require_significance=False,
        )
        summary["capacity_gate_passed"] = _capacity_passes(summary, thresholds=thresholds)
        summary["retention_gate_passed"] = _retention_passes(summary, thresholds=thresholds)
        summary["hard_gate_passed"] = bool(
            summary["selection_eligible"]
            and summary["recovery_gate_passed"]
            and summary["retention_gate_passed"]
        )
        method_summaries.append(summary)
    passed = [row for row in method_summaries if bool(row["hard_gate_passed"])]
    pool = passed
    status = "SELECTED"
    if not pool and dry_run:
        pool = [row for row in method_summaries if bool(row["selection_eligible"])]
        status = "DRY_RUN_SELECTED_NON_EVIDENCE"
    if not pool:
        return {
            "status": "NO_SELECTION",
            "selected_method_id": None,
            "method_summaries": method_summaries,
            "selection_rule": (
                "hard B/C/D recovery + primary-context retention, then minimum "
                "validation maximum active non-target degradation, smaller "
                "active route support, lower update compute, lexicographic method_id"
            ),
        }
    selected = min(
        pool,
        key=lambda row: (
            _finite(row, "maximum_nontarget_degradation"),
            _finite(row, "mean_active_path_support_size"),
            _finite(row, "mean_update_compute_units"),
            str(row["method_id"]),
        ),
    )
    return {
        "status": status,
        "selected_method_id": str(selected["method_id"]),
        "method_summaries": method_summaries,
        "selection_rule": (
            "hard B/C/D recovery + primary-context retention, then minimum "
            "validation maximum active non-target degradation, smaller active "
            "route support, lower update compute, lexicographic method_id"
        ),
    }


def assess_locality_confirmatory(
    seed_rows: Sequence[Mapping[str, Any]],
    *,
    selected_method_id: str,
    baseline_method_id: str,
    required_seeds: Sequence[int],
    thresholds: Mapping[str, float | bool],
    dry_run: bool,
) -> dict[str, Any]:
    selected_rows = [row for row in seed_rows if str(row["method_id"]) == selected_method_id]
    baseline_rows = [row for row in seed_rows if str(row["method_id"]) == baseline_method_id]
    required = set(required_seeds)
    if {int(row["seed"]) for row in selected_rows} != required or {
        int(row["seed"]) for row in baseline_rows
    } != required:
        raise ValueError("E22b selected/baseline lack the exact paired seed set")
    selected = _method_summary(selected_rows, method_id=selected_method_id)
    baseline_by_seed = {int(row["seed"]): row for row in baseline_rows}
    selected_by_seed = {int(row["seed"]): row for row in selected_rows}
    locality_gains = [
        _finite(baseline_by_seed[seed], "maximum_nontarget_degradation")
        - _finite(selected_by_seed[seed], "maximum_nontarget_degradation")
        for seed in required_seeds
    ]
    locality_comparison = {
        "mean_gain": _mean(locality_gains),
        "positive_seed_fraction": _mean(1.0 if value > 0.0 else 0.0 for value in locality_gains),
        "sign_flip_p": exact_sign_flip(locality_gains, alternative="greater"),
    }
    locality_comparison["passed"] = bool(
        locality_comparison["positive_seed_fraction"]
        >= threshold_float(thresholds, "minimum_seed_direction_fraction")
        and locality_comparison["sign_flip_p"]
        <= threshold_float(thresholds, "exact_sign_flip_alpha")
    )
    recovery_pattern = _recovery_pattern_passes(
        selected,
        thresholds=thresholds,
        require_significance=True,
    )
    capacity = _capacity_passes(selected, thresholds=thresholds)
    recovery_capacity = bool(recovery_pattern and capacity)
    absolute_locality = _absolute_locality_passes(selected, thresholds=thresholds)
    retention = _retention_passes(selected, thresholds=thresholds)
    locality_retention = bool(absolute_locality and retention and locality_comparison["passed"])
    if recovery_capacity and locality_retention:
        status = "SUPPORTED_SAFE_LOCALIZED_ASSIMILATION_CONTROLLED"
    elif recovery_capacity:
        status = "CAPACITY_SUPPORTED_LOCALITY_NOT_SUPPORTED"
    elif locality_retention:
        status = "OVERREGULARIZED_LOCALITY_TRADEOFF"
    else:
        status = "NOT_SUPPORTED"
    supported = bool(not dry_run and status == "SUPPORTED_SAFE_LOCALIZED_ASSIMILATION_CONTROLLED")
    return {
        "status": status,
        "supported": supported,
        "dry_run_non_evidence": bool(dry_run),
        "selected_method_id": selected_method_id,
        "baseline_method_id": baseline_method_id,
        "seed_count": len(selected_rows),
        "recovery_pattern_gate_passed": recovery_pattern,
        "capacity_gate_passed": capacity,
        "recovery_capacity_gate_passed": recovery_capacity,
        "absolute_locality_gate_passed": absolute_locality,
        "retention_gate_passed": retention,
        "selected_vs_mean_locality": locality_comparison,
        "locality_retention_gate_passed": locality_retention,
        "observed": selected,
    }


def selection_summary_ko(
    *,
    selection: Mapping[str, Any],
    seeds: Sequence[int],
    dry_run: bool,
) -> str:
    lines = [
        "# E22a Active-Path Locality 방법 선택",
        "",
        f"- 상태: **{selection['status']}**",
        f"- 개발 seed: `{len(seeds)}` (claim 비대상)",
        f"- selected: `{selection.get('selected_method_id')}`",
        f"- dry-run: `{str(dry_run).lower()}`",
        "- threshold: E21 lock에서 런타임 상속",
        "",
        "## 방법별 진단",
        "",
        "| Method | Min B/C/D gain | Max non-target | Support | Compute | Hard |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in selection["method_summaries"]:
        minimum_gain = min(_finite(row, f"mean_{key}") for key in RECOVERY_KEYS)
        lines.append(
            f"| `{row['method_id']}` | {minimum_gain:.6g} | "
            f"{_finite(row, 'maximum_nontarget_degradation'):.6g} | "
            f"{_finite(row, 'mean_active_path_support_size'):.4g} | "
            f"{_finite(row, 'mean_update_compute_units'):.4g} | "
            f"{'PASS' if row['hard_gate_passed'] else 'NO'} |"
        )
    lines.extend(
        (
            "",
            "## 경계",
            "",
            "- Protected projection은 diagnostic-only이고 선택 대상이 아니다.",
            "- E22a는 development-only이고 과학적 claim을 열지 않는다.",
            "- H5/자연어/LM/agent/official-backend claim은 닫혀 있다.",
            "",
        )
    )
    return "\n".join(lines)


def confirmatory_summary_ko(
    *,
    assessment: Mapping[str, Any],
    dry_run: bool,
) -> str:
    observed = assessment["observed"]
    comparison = assessment["selected_vs_mean_locality"]
    lines = [
        "# E22b Active-Path Locality Confirmatory 결과",
        "",
        f"- 판정: **{assessment['status']}**",
        f"- selected: `{assessment['selected_method_id']}` vs `{assessment['baseline_method_id']}`",
        f"- paired seeds: `{assessment['seed_count']}`",
        f"- dry-run/non-evidence: `{str(dry_run).lower()}`",
        "- evidence tier: `CONTROLLED_REFERENCE`",
        "",
        "## 분리 판정",
        "",
        "| Gate | Observed |",
        "|---|---|",
        f"| B/C/D recovery pattern | "
        f"{'PASS' if assessment['recovery_pattern_gate_passed'] else 'FAIL'} |",
        f"| Absolute capacity | {'PASS' if assessment['capacity_gate_passed'] else 'FAIL'} |",
        f"| Absolute locality | "
        f"{'PASS' if assessment['absolute_locality_gate_passed'] else 'FAIL'} |",
        f"| Retention | {'PASS' if assessment['retention_gate_passed'] else 'FAIL'} |",
        f"| Selected-vs-mean locality | {float(comparison['mean_gain']):.6g}; "
        f"p={float(comparison['sign_flip_p']):.6g} |",
        f"| Max non-target degradation | {float(observed['maximum_nontarget_degradation']):.6g} |",
        f"| Max retention degradation | {float(observed['maximum_retention_degradation']):.6g} |",
        "",
        "## 경계",
        "",
        "- E21b-R1을 재판정하거나 소급 수정하지 않는다.",
        "- Fixed identifier와 explicit demand field의 controlled evidence다.",
        "- H5/자연어/LM/agent/official-backend claim은 닫혀 있다.",
        "",
    ]
    return "\n".join(lines)
