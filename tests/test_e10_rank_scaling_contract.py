from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from catena.data.learned_rank import best_rank_errors
from catena.eval.postcore_metrics import exact_sign_flip
from catena.eval.rank_scaling import (
    aggregate_intrinsic_rank_effects_by_seed,
    evaluate_minimum_rank_tracking,
    minimum_sufficient_rank_from_exact_target_recovery,
    oracle_normalized_rank_recovery,
    rank_cell_seed_provenance,
)
from catena.models.operator_controllers import LowRankOperatorController


def test_best_rank_error_cache_uses_one_svd_for_the_registered_rank_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(11)
    target = torch.randn(5, 8, 8, generator=generator)
    singular = torch.linalg.svdvals(target)
    call_count = 0
    original = torch.linalg.svdvals

    def counted_svdvals(value: torch.Tensor) -> torch.Tensor:
        nonlocal call_count
        call_count += 1
        return original(value)

    monkeypatch.setattr(torch.linalg, "svdvals", counted_svdvals)
    cached = best_rank_errors(target, [1, 2, 4, 8])

    assert call_count == 1
    for rank, observed in cached.items():
        expected = singular[..., rank:].square().sum(dim=-1) / 64.0
        assert torch.allclose(observed, expected)


def test_low_vs_high_rank_inference_aggregates_cells_to_independent_seeds() -> None:
    effects = {
        (101, 1): 1.0,
        (101, 2): 2.0,
        (101, 4): 3.0,
        (211, 1): 2.0,
        (211, 2): 3.0,
        (211, 4): 4.0,
    }
    rows = aggregate_intrinsic_rank_effects_by_seed(
        effects,
        seeds=[101, 211],
        intrinsic_ranks=[1, 2, 4],
    )

    assert [row["mean_low_vs_high_rank_gain"] for row in rows] == [2.0, 3.0]
    assert all(row["intrinsic_rank_cell_count"] == 3 for row in rows)
    assert exact_sign_flip(
        [float(row["mean_low_vs_high_rank_gain"]) for row in rows],
        alternative="greater",
    ) == pytest.approx(0.25)


def test_rank_cell_pairing_reuses_backbone_initialization_and_sampling_seed() -> None:
    provenance = rank_cell_seed_provenance(seed=101, intrinsic_rank=4)
    torch.manual_seed(provenance["model_seed"])
    rank_one = LowRankOperatorController(
        descriptor_dim=5,
        dimension=8,
        rank=1,
        hidden_dim=16,
    )
    torch.manual_seed(provenance["model_seed"])
    rank_four = LowRankOperatorController(
        descriptor_dim=5,
        dimension=8,
        rank=4,
        hidden_dim=16,
    )

    for left, right in zip(
        rank_one.backbone.parameters(),
        rank_four.backbone.parameters(),
        strict=True,
    ):
        assert torch.equal(left, right)
    assert provenance["optimizer_sampling_seed"] == 50_000 * 101 + 4
    assert rank_one.left_head.weight.shape != rank_four.left_head.weight.shape


def test_rank_tracking_requires_lower_bound_and_seedwise_nondecreasing_minima() -> None:
    valid = {
        (101, 1): 1,
        (101, 2): 4,
        (101, 4): 8,
        (211, 1): 2,
        (211, 2): 2,
        (211, 4): 4,
    }
    cells, seeds = evaluate_minimum_rank_tracking(
        valid,
        seeds=[101, 211],
        intrinsic_ranks=[1, 2, 4],
        max_rank_factor=2.0,
        max_available_rank=8,
    )
    assert all(bool(row["rank_tracking_matched"]) for row in cells)
    assert all(bool(row["minimum_qualifying_rank_nondecreasing"]) for row in seeds)

    insufficient = dict(valid)
    insufficient[(101, 4)] = 1
    cells, seeds = evaluate_minimum_rank_tracking(
        insufficient,
        seeds=[101, 211],
        intrinsic_ranks=[1, 2, 4],
        max_rank_factor=2.0,
        max_available_rank=8,
    )
    failed = next(row for row in cells if row["seed"] == 101 and row["intrinsic_rank"] == 4)
    assert failed["rank_tracking_matched"] is False
    assert seeds[0]["minimum_qualifying_rank_nondecreasing"] is False


def test_oracle_normalized_recovery_validates_metric_geometry() -> None:
    assert oracle_normalized_rank_recovery(
        baseline_error=0.10,
        model_error=0.04,
        oracle_error=0.02,
    ) == pytest.approx(0.75)
    with pytest.raises(ValueError, match="cannot beat"):
        oracle_normalized_rank_recovery(
            baseline_error=0.10,
            model_error=0.01,
            oracle_error=0.02,
        )
    with pytest.raises(ValueError, match="positive oracle headroom"):
        oracle_normalized_rank_recovery(
            baseline_error=0.02,
            model_error=0.02,
            oracle_error=0.02,
        )


def test_rank_one_can_reach_its_floor_without_recovering_the_exact_target() -> None:
    reachable_floor_recovery = oracle_normalized_rank_recovery(
        baseline_error=0.10,
        model_error=0.05,
        oracle_error=0.05,
    )
    exact_target_recovery = oracle_normalized_rank_recovery(
        baseline_error=0.10,
        model_error=0.05,
        oracle_error=0.0,
    )

    assert reachable_floor_recovery == pytest.approx(1.0)
    assert exact_target_recovery == pytest.approx(0.5)
    assert exact_target_recovery < 0.95
    assert (
        minimum_sufficient_rank_from_exact_target_recovery(
            {1: exact_target_recovery, 2: 0.96},
            threshold=0.95,
        )
        == 2
    )


def test_e10_prospective_repair_preserves_registered_grid_and_thresholds() -> None:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (root / "configs/e10_learned_rank_scaling.yaml").read_text(encoding="utf-8")
    )

    assert config["data"]["intrinsic_ranks"] == [1, 2, 4, 8, 16]
    assert config["model"]["learned_ranks"] == [1, 2, 4, 8, 16, 32]
    assert config["claim_gate"] == {
        "oracle_normalized_recovery": 0.95,
        "max_rank_factor": 2.0,
        "minimum_rank_match_fraction": 0.8,
        "monotonic_fraction": 0.9,
    }
    repair = config["protocol_repairs"]
    assert repair == [
        {
            "id": "prospective_pre_evaluation_gate_identifiability_repair",
            "timing": "before_any_evaluable_e10_main_report",
            "thresholds_or_rank_grids_changed": False,
            "require_minimum_qualifying_rank_at_least_intrinsic_rank": True,
            "require_seedwise_nondecreasing_minimum_qualifying_rank": True,
            "reachable_floor_recovery_role": "optimization_diagnostic_only",
            "minimum_rank_qualification_metric": "exact_target_recovery",
        }
    ]
