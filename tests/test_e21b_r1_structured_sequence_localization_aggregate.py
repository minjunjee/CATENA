from __future__ import annotations

from copy import deepcopy
from itertools import product
from pathlib import Path

import yaml

from catena.eval.structured_sequence_localization_r1 import (
    assess_e21b_r1,
    compute_e21b_r1_seed_contrasts,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SEEDS = [113, 223, 331, 449, 557]
VARIANTS = ["base", "separate_address", "state_aware", "full"]
CONDITIONS = [
    "A_oracle_address_oracle_candidate",
    "B_learned_address_oracle_candidate",
    "C_oracle_address_state_read_candidate",
    "D_learned_address_state_read_candidate",
]
FAMILIES = [
    "magnitude_factorization",
    "value_granularity",
    "address_decoupling",
    "state_conditioning",
]
UPDATES = [1, 4, 8]
GAPS = [0, 128, 512, 2048]


def _rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed, variant, condition, family, updates, gap in product(
        SEEDS,
        VARIANTS,
        CONDITIONS,
        FAMILIES,
        UPDATES,
        GAPS,
    ):
        affected = 0.0
        if (
            condition == "B_learned_address_oracle_candidate"
            and family == "address_decoupling"
            and variant in {"base", "state_aware"}
        ):
            affected = 0.02
        if (
            condition == "C_oracle_address_state_read_candidate"
            and family == "state_conditioning"
            and variant in {"base", "separate_address"}
        ):
            affected = 0.02
        if (
            condition == "D_learned_address_state_read_candidate"
            and family == "address_decoupling"
            and variant != "full"
        ):
            affected = 0.02
        rows.append(
            {
                "seed": seed,
                "variant": variant,
                "condition": condition,
                "demand_family": family,
                "updates": updates,
                "gap_events": gap,
                "affected_mse": affected,
                "retention_mse": 0.0,
                "address_accuracy": 1.0,
                "candidate_recovery_mse": 0.0,
                "verified_activity_mean": 1.0,
                "distractor_activity_mean": 0.0,
            }
        )
    return rows


def _contrasts(rows: list[dict[str, object]]):
    return compute_e21b_r1_seed_contrasts(
        rows,
        seeds=SEEDS,
        updates_grid=UPDATES,
        gaps_grid=GAPS,
        demand_families=FAMILIES,
        stress_updates=8,
        stress_gap_events=2048,
    )


def _assessment(rows: list[dict[str, object]]):
    config = yaml.safe_load(
        (REPO_ROOT / "configs/e21b_r1_structured_sequence_localization_aggregate.yaml").read_text(
            encoding="utf-8"
        )
    )
    return assess_e21b_r1(
        _contrasts(rows),
        thresholds=config["claim_gate"],
        alpha=float(config["statistics"]["alpha"]),
    )


def test_registered_supported_pattern_passes_repaired_gate():
    contrasts = _contrasts(_rows())
    assert all(row["nontarget_cell_count"] == 180 for row in contrasts)
    assert all(row["retention_cell_count"] == 36 for row in contrasts)
    assessment = _assessment(_rows())
    assert assessment["supported"] is True
    assert assessment["gates"]["nontarget_noninferiority"] is True
    assert assessment["gates"]["retention_noninferiority"] is True


def test_repair_preserves_original_seeds_grid_thresholds_and_primary_inputs():
    original = yaml.safe_load(
        (REPO_ROOT / "configs/e21_structured_sequence_localization_transfer.yaml").read_text(
            encoding="utf-8"
        )
    )
    repair = yaml.safe_load(
        (REPO_ROOT / "configs/e21b_r1_structured_sequence_localization_aggregate.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert repair["seeds"] == original["seeds"]
    assert repair["conditions"] == original["conditions"]
    assert repair["demand_families"] == original["demand_families"]
    assert repair["variants"] == original["model"]["variants"]
    assert repair["evaluation"] == original["evaluation"]
    assert repair["statistics"] == original["statistics"]
    assert repair["claim_gate"] == original["claim_gate"]


def test_active_state_read_single_cell_harm_cannot_average_away():
    rows = _rows()
    for row in rows:
        if (
            row["seed"] == 113
            and row["condition"] == "D_learned_address_state_read_candidate"
            and row["demand_family"] == "value_granularity"
            and row["updates"] == 8
            and row["gap_events"] == 2048
            and row["variant"] in {"state_aware", "full"}
        ):
            row["affected_mse"] = 0.001
    contrasts = _contrasts(rows)
    seed = next(row for row in contrasts if row["seed"] == 113)
    assert seed["maximum_nontarget_degradation"] == 0.001
    assert str(seed["maximum_nontarget_cell"]).startswith("state_read|D_")
    assert _assessment(rows)["gates"]["nontarget_noninferiority"] is False


def test_single_retention_cell_harm_cannot_average_away():
    rows = _rows()
    for row in rows:
        if (
            row["seed"] == 223
            and row["condition"] == "C_oracle_address_state_read_candidate"
            and row["demand_family"] == "state_conditioning"
            and row["updates"] == 4
            and row["gap_events"] == 512
            and row["variant"] in {"state_aware", "full"}
        ):
            row["retention_mse"] = 0.00075
    contrasts = _contrasts(rows)
    seed = next(row for row in contrasts if row["seed"] == 223)
    assert seed["maximum_retention_degradation"] == 0.00075
    assert _assessment(rows)["gates"]["retention_noninferiority"] is False


def test_original_primary_estimands_and_exact_sign_flip_are_preserved():
    assessment = _assessment(deepcopy(_rows()))
    for result in assessment["pattern"].values():
        assert abs(float(result["mean_gain"]) - 0.02) < 1e-12
        assert result["positive_seed_fraction"] == 1.0
        assert result["sign_flip_p"] == 0.03125
        assert result["passed"] is True
    assert assessment["repair"]["original_e21b_disposition"] == ("INCONCLUSIVE_GATE_IMPLEMENTATION")
