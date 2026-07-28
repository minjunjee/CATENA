from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from catena.eval.structured_sequence_localization import (
    assess_structured_sequence_transfer,
    compute_structured_sequence_seed_contrasts,
)

CONDITION_B = "B_learned_address_oracle_candidate"
CONDITION_C = "C_oracle_address_state_read_candidate"
CONDITION_D = "D_learned_address_state_read_candidate"
ADDRESS_FAMILY = "address_decoupling"
STATE_FAMILY = "state_conditioning"

NO_SEPARATE = {"base", "state_aware"}
SEPARATE = {"separate_address", "full"}
NO_STATE_READ = {"base", "separate_address"}
STATE_READ = {"state_aware", "full"}
INCOMPLETE = ("base", "separate_address", "state_aware")


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        raise ValueError("empty E21b-R1 metric selection")
    return sum(materialized) / len(materialized)


def _cell_mean(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    variants: set[str],
    condition: str,
    family: str,
    updates: int,
    gap_events: int,
    metric: str,
) -> float:
    selected = [
        float(row[metric])
        for row in rows
        if int(row["seed"]) == seed
        and str(row["variant"]) in variants
        and str(row["condition"]) == condition
        and str(row["demand_family"]) == family
        and int(row["updates"]) == updates
        and int(row["gap_events"]) == gap_events
    ]
    expected = len(variants)
    if len(selected) != expected:
        raise ValueError(
            "E21b-R1 cell is incomplete: "
            f"{seed=}, {condition=}, {family=}, {updates=}, {gap_events=}, "
            f"{metric=}, observed={len(selected)}, expected={expected}"
        )
    return _mean(selected)


def _best_incomplete_variant(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    updates_grid: list[int],
    gaps_grid: list[int],
) -> str:
    means: dict[str, float] = {}
    for variant in INCOMPLETE:
        means[variant] = _mean(
            _cell_mean(
                rows,
                seed=seed,
                variants={variant},
                condition=CONDITION_D,
                family=ADDRESS_FAMILY,
                updates=updates,
                gap_events=gap,
                metric="affected_mse",
            )
            for updates in updates_grid
            for gap in gaps_grid
        )
    return min(INCOMPLETE, key=lambda variant: (means[variant], variant))


def _maximum_with_descriptor(
    rows: list[tuple[float, str]],
) -> tuple[float, str]:
    if not rows:
        raise ValueError("E21b-R1 repaired guardrail has no cells")
    return max(rows, key=lambda item: item[0])


def compute_e21b_r1_seed_contrasts(
    rows: list[dict[str, Any]],
    *,
    seeds: list[int],
    updates_grid: list[int],
    gaps_grid: list[int],
    demand_families: list[str],
    stress_updates: int,
    stress_gap_events: int,
) -> list[dict[str, Any]]:
    """Preserve primary E21 contrasts and replace only the two broken guards."""

    original = compute_structured_sequence_seed_contrasts(
        rows,
        seeds=seeds,
        stress_updates=stress_updates,
        stress_gap_events=stress_gap_events,
    )
    by_seed: dict[int, dict[str, Any]] = {int(row["seed"]): dict(row) for row in original}
    if set(by_seed) != set(seeds):
        raise ValueError("E21b-R1 original primary contrast seed set mismatch")

    result: list[dict[str, Any]] = []
    for seed in seeds:
        best_incomplete = _best_incomplete_variant(
            rows,
            seed=seed,
            updates_grid=updates_grid,
            gaps_grid=gaps_grid,
        )
        nontarget: list[tuple[float, str]] = []
        retention: list[tuple[float, str]] = []

        # Separate-address freedom is active only with learned addresses B/D.
        for condition in (CONDITION_B, CONDITION_D):
            for family in demand_families:
                if family == ADDRESS_FAMILY:
                    continue
                for updates in updates_grid:
                    for gap in gaps_grid:
                        degradation = _cell_mean(
                            rows,
                            seed=seed,
                            variants=SEPARATE,
                            condition=condition,
                            family=family,
                            updates=updates,
                            gap_events=gap,
                            metric="affected_mse",
                        ) - _cell_mean(
                            rows,
                            seed=seed,
                            variants=NO_SEPARATE,
                            condition=condition,
                            family=family,
                            updates=updates,
                            gap_events=gap,
                            metric="affected_mse",
                        )
                        nontarget.append(
                            (
                                degradation,
                                f"separate|{condition}|{family}|u{updates}|g{gap}",
                            )
                        )

        # State read is active only in C/D. Exclude the two registered
        # identifying target cells rather than using bypassed A/B routes.
        for condition in (CONDITION_C, CONDITION_D):
            for family in demand_families:
                if (condition == CONDITION_C and family == STATE_FAMILY) or (
                    condition == CONDITION_D and family == ADDRESS_FAMILY
                ):
                    continue
                for updates in updates_grid:
                    for gap in gaps_grid:
                        degradation = _cell_mean(
                            rows,
                            seed=seed,
                            variants=STATE_READ,
                            condition=condition,
                            family=family,
                            updates=updates,
                            gap_events=gap,
                            metric="affected_mse",
                        ) - _cell_mean(
                            rows,
                            seed=seed,
                            variants=NO_STATE_READ,
                            condition=condition,
                            family=family,
                            updates=updates,
                            gap_events=gap,
                            metric="affected_mse",
                        )
                        nontarget.append(
                            (
                                degradation,
                                f"state_read|{condition}|{family}|u{updates}|g{gap}",
                            )
                        )

        # The D conjunction must also avoid harming every non-identifying D cell.
        for family in demand_families:
            if family == ADDRESS_FAMILY:
                continue
            for updates in updates_grid:
                for gap in gaps_grid:
                    degradation = _cell_mean(
                        rows,
                        seed=seed,
                        variants={"full"},
                        condition=CONDITION_D,
                        family=family,
                        updates=updates,
                        gap_events=gap,
                        metric="affected_mse",
                    ) - _cell_mean(
                        rows,
                        seed=seed,
                        variants={best_incomplete},
                        condition=CONDITION_D,
                        family=family,
                        updates=updates,
                        gap_events=gap,
                        metric="affected_mse",
                    )
                    nontarget.append(
                        (
                            degradation,
                            f"full|{CONDITION_D}|{family}|u{updates}|g{gap}",
                        )
                    )

        retention_contexts = (
            (CONDITION_B, ADDRESS_FAMILY, SEPARATE, NO_SEPARATE, "separate"),
            (CONDITION_C, STATE_FAMILY, STATE_READ, NO_STATE_READ, "state_read"),
            (
                CONDITION_D,
                ADDRESS_FAMILY,
                {"full"},
                {best_incomplete},
                "full",
            ),
        )
        for condition, family, treatment, comparison, label in retention_contexts:
            for updates in updates_grid:
                for gap in gaps_grid:
                    degradation = _cell_mean(
                        rows,
                        seed=seed,
                        variants=treatment,
                        condition=condition,
                        family=family,
                        updates=updates,
                        gap_events=gap,
                        metric="retention_mse",
                    ) - _cell_mean(
                        rows,
                        seed=seed,
                        variants=comparison,
                        condition=condition,
                        family=family,
                        updates=updates,
                        gap_events=gap,
                        metric="retention_mse",
                    )
                    retention.append(
                        (
                            degradation,
                            f"{label}|{condition}|{family}|u{updates}|g{gap}",
                        )
                    )

        max_nontarget, nontarget_descriptor = _maximum_with_descriptor(nontarget)
        max_retention, retention_descriptor = _maximum_with_descriptor(retention)
        repaired = by_seed[seed]
        repaired["maximum_nontarget_degradation"] = max_nontarget
        repaired["maximum_retention_degradation"] = max_retention
        repaired["maximum_nontarget_cell"] = nontarget_descriptor
        repaired["maximum_retention_cell"] = retention_descriptor
        repaired["nontarget_cell_count"] = len(nontarget)
        repaired["retention_cell_count"] = len(retention)
        repaired["d_best_incomplete_variant"] = best_incomplete
        result.append(repaired)
    return result


def assess_e21b_r1(
    seed_rows: list[dict[str, Any]],
    *,
    thresholds: Mapping[str, Any],
    alpha: float,
) -> dict[str, Any]:
    assessment = assess_structured_sequence_transfer(
        seed_rows,
        thresholds=thresholds,
        alpha=alpha,
        dry_run=False,
    )
    assessment["repair"] = {
        "primary_estimands_unchanged": True,
        "active_state_read_conditions": [CONDITION_C, CONDITION_D],
        "nontarget_aggregation": "cellwise_maximum",
        "retention_aggregation": "cellwise_maximum",
        "identifying_targets_excluded": [
            f"{CONDITION_B}|{ADDRESS_FAMILY}|separate_address",
            f"{CONDITION_C}|{STATE_FAMILY}|state_read",
            f"{CONDITION_D}|{ADDRESS_FAMILY}|full_only",
        ],
        "original_e21b_disposition": "INCONCLUSIVE_GATE_IMPLEMENTATION",
    }
    return assessment
