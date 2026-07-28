from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

from catena.eval.postcore_metrics import exact_sign_flip


def _finite(row: Mapping[str, Any], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"missing or invalid E21 metric {key!r}") from error
    if not math.isfinite(value):
        raise ValueError(f"non-finite E21 metric {key!r}")
    return value


def _mean(values: Iterable[float]) -> float:
    rows = list(values)
    if not rows:
        raise ValueError("cannot average an empty E21 collection")
    return sum(rows) / len(rows)


def _selected_mean(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    variants: set[str],
    condition: str,
    families: set[str],
    metric: str,
    stress_only: bool = False,
    stress_updates: int = 8,
    stress_gap_events: int = 2048,
) -> float:
    selected = [
        _finite(row, metric)
        for row in rows
        if int(row["seed"]) == seed
        and str(row["variant"]) in variants
        and str(row["condition"]) == condition
        and str(row["demand_family"]) in families
        and (
            not stress_only
            or (
                int(row["updates"]) == stress_updates
                and int(row["gap_events"]) == stress_gap_events
            )
        )
    ]
    return _mean(selected)


def compute_structured_sequence_seed_contrasts(
    rows: list[dict[str, Any]],
    *,
    seeds: list[int],
    stress_updates: int,
    stress_gap_events: int,
) -> list[dict[str, float | int]]:
    condition_a = "A_oracle_address_oracle_candidate"
    condition_b = "B_learned_address_oracle_candidate"
    condition_c = "C_oracle_address_state_read_candidate"
    condition_d = "D_learned_address_state_read_candidate"
    address_family = {"address_decoupling"}
    state_family = {"state_conditioning"}
    non_address_families = {
        "magnitude_factorization",
        "value_granularity",
        "state_conditioning",
    }
    all_families = {
        "magnitude_factorization",
        "value_granularity",
        "address_decoupling",
        "state_conditioning",
    }
    no_separate = {"base", "state_aware"}
    separate = {"separate_address", "full"}
    no_state_read = {"base", "separate_address"}
    state_read = {"state_aware", "full"}
    incomplete = ({"base"}, {"separate_address"}, {"state_aware"})

    result: list[dict[str, float | int]] = []
    for seed in seeds:
        b_comparison = _selected_mean(
            rows,
            seed=seed,
            variants=no_separate,
            condition=condition_b,
            families=address_family,
            metric="affected_mse",
        )
        b_treatment = _selected_mean(
            rows,
            seed=seed,
            variants=separate,
            condition=condition_b,
            families=address_family,
            metric="affected_mse",
        )
        c_comparison = _selected_mean(
            rows,
            seed=seed,
            variants=no_state_read,
            condition=condition_c,
            families=state_family,
            metric="affected_mse",
        )
        c_treatment = _selected_mean(
            rows,
            seed=seed,
            variants=state_read,
            condition=condition_c,
            families=state_family,
            metric="affected_mse",
        )
        d_full = _selected_mean(
            rows,
            seed=seed,
            variants={"full"},
            condition=condition_d,
            families=address_family,
            metric="affected_mse",
        )
        d_incomplete = [
            _selected_mean(
                rows,
                seed=seed,
                variants=variant,
                condition=condition_d,
                families=address_family,
                metric="affected_mse",
            )
            for variant in incomplete
        ]

        def stress_gain(
            comparison: set[str],
            treatment: set[str],
            condition: str,
            families: set[str],
            *,
            seed_value: int = seed,
        ) -> float:
            return _selected_mean(
                rows,
                seed=seed_value,
                variants=comparison,
                condition=condition,
                families=families,
                metric="affected_mse",
                stress_only=True,
                stress_updates=stress_updates,
                stress_gap_events=stress_gap_events,
            ) - _selected_mean(
                rows,
                seed=seed_value,
                variants=treatment,
                condition=condition,
                families=families,
                metric="affected_mse",
                stress_only=True,
                stress_updates=stress_updates,
                stress_gap_events=stress_gap_events,
            )

        d_stress_full = _selected_mean(
            rows,
            seed=seed,
            variants={"full"},
            condition=condition_d,
            families=address_family,
            metric="affected_mse",
            stress_only=True,
            stress_updates=stress_updates,
            stress_gap_events=stress_gap_events,
        )
        d_stress_incomplete = min(
            _selected_mean(
                rows,
                seed=seed,
                variants=variant,
                condition=condition_d,
                families=address_family,
                metric="affected_mse",
                stress_only=True,
                stress_updates=stress_updates,
                stress_gap_events=stress_gap_events,
            )
            for variant in incomplete
        )

        b_retention_comparison = _selected_mean(
            rows,
            seed=seed,
            variants=no_separate,
            condition=condition_b,
            families=address_family,
            metric="retention_mse",
        )
        b_retention_treatment = _selected_mean(
            rows,
            seed=seed,
            variants=separate,
            condition=condition_b,
            families=address_family,
            metric="retention_mse",
        )
        c_retention_comparison = _selected_mean(
            rows,
            seed=seed,
            variants=no_state_read,
            condition=condition_c,
            families=state_family,
            metric="retention_mse",
        )
        c_retention_treatment = _selected_mean(
            rows,
            seed=seed,
            variants=state_read,
            condition=condition_c,
            families=state_family,
            metric="retention_mse",
        )
        d_retention_full = _selected_mean(
            rows,
            seed=seed,
            variants={"full"},
            condition=condition_d,
            families=address_family,
            metric="retention_mse",
        )
        d_retention_incomplete = min(
            _selected_mean(
                rows,
                seed=seed,
                variants=variant,
                condition=condition_d,
                families=address_family,
                metric="retention_mse",
            )
            for variant in incomplete
        )

        b_nontarget = _selected_mean(
            rows,
            seed=seed,
            variants=separate,
            condition=condition_b,
            families=non_address_families,
            metric="affected_mse",
        ) - _selected_mean(
            rows,
            seed=seed,
            variants=no_separate,
            condition=condition_b,
            families=non_address_families,
            metric="affected_mse",
        )
        state_oracle_candidate_degradations = []
        for condition in (condition_a, condition_b):
            state_oracle_candidate_degradations.append(
                _selected_mean(
                    rows,
                    seed=seed,
                    variants=state_read,
                    condition=condition,
                    families=all_families,
                    metric="affected_mse",
                )
                - _selected_mean(
                    rows,
                    seed=seed,
                    variants=no_state_read,
                    condition=condition,
                    families=all_families,
                    metric="affected_mse",
                )
            )

        matching_rows = [row for row in rows if int(row["seed"]) == seed]
        capable_address_rows = [
            row
            for row in matching_rows
            if (
                str(row["condition"]) == condition_b
                and str(row["variant"]) in separate
                and str(row["demand_family"]) == "address_decoupling"
            )
            or (
                str(row["condition"]) == condition_d
                and str(row["variant"]) == "full"
                and str(row["demand_family"]) == "address_decoupling"
            )
        ]
        capable_candidate_rows = [
            row
            for row in matching_rows
            if (
                str(row["condition"]) == condition_c
                and str(row["variant"]) in state_read
                and str(row["demand_family"]) == "state_conditioning"
            )
            or (
                str(row["condition"]) == condition_d
                and str(row["variant"]) == "full"
                and str(row["demand_family"]) == "address_decoupling"
            )
        ]
        capable_affected_rows = capable_address_rows + capable_candidate_rows
        oracle_rows = [
            row
            for row in matching_rows
            if str(row["condition"]) == condition_a
        ]
        result.append(
            {
                "seed": seed,
                "b_separate_address_gain": b_comparison - b_treatment,
                "c_state_read_gain": c_comparison - c_treatment,
                "d_full_only_gain": min(d_incomplete) - d_full,
                "b_stress_gain": stress_gain(
                    no_separate,
                    separate,
                    condition_b,
                    address_family,
                ),
                "c_stress_gain": stress_gain(
                    no_state_read,
                    state_read,
                    condition_c,
                    state_family,
                ),
                "d_stress_gain": d_stress_incomplete - d_stress_full,
                "maximum_nontarget_degradation": max(
                    b_nontarget,
                    *state_oracle_candidate_degradations,
                ),
                "maximum_retention_degradation": max(
                    b_retention_treatment - b_retention_comparison,
                    c_retention_treatment - c_retention_comparison,
                    d_retention_full - d_retention_incomplete,
                ),
                "minimum_capable_address_accuracy": min(
                    _finite(row, "address_accuracy")
                    for row in capable_address_rows
                ),
                "maximum_capable_candidate_mse": max(
                    _finite(row, "candidate_recovery_mse")
                    for row in capable_candidate_rows
                ),
                "maximum_capable_affected_mse": max(
                    _finite(row, "affected_mse")
                    for row in capable_affected_rows
                ),
                "maximum_oracle_floor_mse": max(
                    _finite(row, "affected_mse") for row in oracle_rows
                ),
                "minimum_verified_activity": min(
                    _finite(row, "verified_activity_mean")
                    for row in matching_rows
                ),
                "maximum_distractor_activity": max(
                    _finite(row, "distractor_activity_mean")
                    for row in matching_rows
                    if int(row["gap_events"]) > 0
                ),
            }
        )
    return result


def _gain_gate(
    values: list[float],
    *,
    sesoi: float,
    alpha: float,
    direction_fraction: float,
) -> dict[str, float | bool]:
    mean_gain = _mean(values)
    positive_fraction = sum(value > 0.0 for value in values) / len(values)
    p_value = exact_sign_flip(values, alternative="greater")
    return {
        "mean_gain": mean_gain,
        "positive_seed_fraction": positive_fraction,
        "sign_flip_p": p_value,
        "passed": bool(
            mean_gain >= sesoi
            and positive_fraction >= direction_fraction
            and p_value <= alpha
        ),
    }


def assess_structured_sequence_transfer(
    seed_rows: list[dict[str, float | int]],
    *,
    thresholds: Mapping[str, Any],
    alpha: float,
    dry_run: bool,
) -> dict[str, Any]:
    if not seed_rows:
        raise ValueError("E21 aggregate requires seed-level contrasts")
    sesoi = float(thresholds["selective_gain"])
    direction = float(thresholds["minimum_seed_direction_fraction"])
    pattern = {
        "b_separate_address_recovery": _gain_gate(
            [float(row["b_separate_address_gain"]) for row in seed_rows],
            sesoi=sesoi,
            alpha=alpha,
            direction_fraction=direction,
        ),
        "c_state_read_recovery": _gain_gate(
            [float(row["c_state_read_gain"]) for row in seed_rows],
            sesoi=sesoi,
            alpha=alpha,
            direction_fraction=direction,
        ),
        "d_full_only_recovery": _gain_gate(
            [float(row["d_full_only_gain"]) for row in seed_rows],
            sesoi=sesoi,
            alpha=alpha,
            direction_fraction=direction,
        ),
    }
    stress_direction = min(
        sum(float(row[key]) > 0.0 for row in seed_rows) / len(seed_rows)
        for key in ("b_stress_gain", "c_stress_gain", "d_stress_gain")
    )
    observed = {
        "maximum_nontarget_degradation": max(
            float(row["maximum_nontarget_degradation"]) for row in seed_rows
        ),
        "maximum_retention_degradation": max(
            float(row["maximum_retention_degradation"]) for row in seed_rows
        ),
        "minimum_capable_address_accuracy": min(
            float(row["minimum_capable_address_accuracy"]) for row in seed_rows
        ),
        "maximum_capable_candidate_mse": max(
            float(row["maximum_capable_candidate_mse"]) for row in seed_rows
        ),
        "maximum_capable_affected_mse": max(
            float(row["maximum_capable_affected_mse"]) for row in seed_rows
        ),
        "maximum_oracle_floor_mse": max(
            float(row["maximum_oracle_floor_mse"]) for row in seed_rows
        ),
        "minimum_verified_activity": min(
            float(row["minimum_verified_activity"]) for row in seed_rows
        ),
        "maximum_distractor_activity": max(
            float(row["maximum_distractor_activity"]) for row in seed_rows
        ),
        "minimum_stress_positive_seed_fraction": stress_direction,
    }
    gates = {
        "registered_selective_pattern": all(
            bool(value["passed"]) for value in pattern.values()
        ),
        "stress_direction": stress_direction >= direction,
        "nontarget_noninferiority": observed[
            "maximum_nontarget_degradation"
        ]
        <= float(thresholds["maximum_nontarget_degradation"]),
        "retention_noninferiority": observed[
            "maximum_retention_degradation"
        ]
        <= float(thresholds["retention_noninferiority"]),
        "learned_address_accuracy": observed[
            "minimum_capable_address_accuracy"
        ]
        >= float(thresholds["minimum_address_accuracy"]),
        "candidate_recovery": observed["maximum_capable_candidate_mse"]
        <= float(thresholds["maximum_candidate_recovery_mse"]),
        "capable_affected_floor": observed["maximum_capable_affected_mse"]
        <= float(thresholds["maximum_capable_affected_mse"]),
        "oracle_information_floor": observed["maximum_oracle_floor_mse"]
        <= float(thresholds["maximum_oracle_floor_mse"]),
        "verified_activity": observed["minimum_verified_activity"]
        >= float(thresholds["minimum_verified_activity"]),
        "distractor_activity": observed["maximum_distractor_activity"]
        <= float(thresholds["maximum_distractor_activity"]),
    }
    supported = bool(all(gates.values()) and not dry_run)
    return {
        "pattern": pattern,
        "observed": observed,
        "gates": gates,
        "supported": supported,
    }


def structured_sequence_source_summary_ko(
    *,
    dry_run: bool,
    seed: int,
    rows: list[dict[str, Any]],
    report_status: str,
    paired: bool,
) -> str:
    stress_rows = [
        row
        for row in rows
        if int(row["updates"]) == max(int(value["updates"]) for value in rows)
        and int(row["gap_events"])
        == max(int(value["gap_events"]) for value in rows)
    ]
    mean_affected = sum(float(row["affected_mse"]) for row in stress_rows) / max(
        len(stress_rows),
        1,
    )
    label = "CPU DRY-RUN — 과학적 증거 아님" if dry_run else "MAIN — aggregate 대기"
    return "\n".join(
        (
            "# E21a Structured Sequence Transfer 결과 요약",
            "",
            f"- 실행 구분: **{label}**",
            f"- seed: `{seed}`",
            f"- status: `{report_status}`",
            f"- metric rows: `{len(rows)}`",
            f"- paired maximal-surface contract: `{'PASS' if paired else 'FAIL'}`",
            f"- stress-grid 전체 row 평균 affected MSE: `{mean_affected:.8g}`",
            "",
            "## 이 run이 확인한 것",
            "",
            "- Structured identifier/event encoder, A–D 정보조건, repeated sequence와",
            "  distractor 경로가 동일 runner에서 끝까지 실행됐다.",
            "- old value와 integer slot/update mask는 event input에 포함되지 않았다.",
            "- MAIN source라도 단독으로 claim을 열지 않으며 5-seed E21b가 필요하다.",
            "",
            "## 해석 경계",
            "",
            "- H5 semantic factorization을 다시 열지 않는다.",
            "- 자연어, novel identifier, pretrained LM, agent, official backend 또는",
            "  runtime superiority evidence가 아니다.",
            "",
        )
    )


def structured_sequence_aggregate_summary_ko(
    *,
    dry_run: bool,
    assessment: dict[str, Any],
    seeds: list[int],
) -> str:
    status = (
        "NOT_EVALUATED_DRY_RUN"
        if dry_run
        else ("SUPPORTED" if assessment["supported"] else "NOT_SUPPORTED")
    )
    pattern = assessment["pattern"]
    observed = assessment["observed"]
    lines = [
        "# E21b Structured Sequence Transfer 종합 결과",
        "",
        f"- 판정: **{status}**",
        f"- paired seeds: `{len(seeds)}`",
        "- evidence tier: `CONTROLLED_REFERENCE`",
        "",
        "## 등록 contrast",
        "",
        "| Contrast | Mean gain | +seed | p | Gate |",
        "|---|---:|---:|---:|---|",
    ]
    for name, result in pattern.items():
        lines.append(
            f"| `{name}` | {float(result['mean_gain']):.6g} | "
            f"{float(result['positive_seed_fraction']):.3f} | "
            f"{float(result['sign_flip_p']):.6g} | "
            f"{'PASS' if result['passed'] else 'FAIL'} |"
        )
    lines.extend(
        (
            "",
            "## Guardrail",
            "",
            f"- min learned-address accuracy: "
            f"`{float(observed['minimum_capable_address_accuracy']):.6g}`",
            f"- max candidate MSE: "
            f"`{float(observed['maximum_capable_candidate_mse']):.6g}`",
            f"- max affected MSE: "
            f"`{float(observed['maximum_capable_affected_mse']):.6g}`",
            f"- max retention degradation: "
            f"`{float(observed['maximum_retention_degradation']):.6g}`",
            "",
            "## 해석 경계",
            "",
            "- Fixed structured identifier schema와 explicit demand field에서의",
            "  controlled repeated-sequence evidence로만 해석한다.",
            "- H5, 자연어/novel-ID, pretrained LM, agent, official backend claim은 닫혀 있다.",
            "",
        )
    )
    return "\n".join(lines)
