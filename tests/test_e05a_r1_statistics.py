from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from catena.eval.semantic_anchor_v61 import (
    CONTROL_NAMES,
    SemanticAnchorSeedMetrics,
)
from catena.eval.semantic_design_repair_r1 import (
    E05A_R1_BOOTSTRAP_SEEDS,
    E05A_R1_SEEDS,
    E05aR1Thresholds,
    evaluate_e05a_r1_design,
)


def _thresholds() -> E05aR1Thresholds:
    return E05aR1Thresholds(bootstrap_samples=256)


def _seed_metrics(
    seed: int,
    *,
    gain: float = 0.002,
    factorized_ai: float = 0.0002,
    shared_headroom: float | None = None,
    preserve_excess: float = 0.0001,
    retention_factorized: float = 0.0001,
    retention_shared: float = 0.0001,
    oracle: float = 0.0,
    control_degradation: float = 0.003,
) -> SemanticAnchorSeedMetrics:
    cells = list(
        product(
            ("api", "access"),
            ("structured", "indirect"),
            ("preserve", "add", "invalidate"),
        )
    )
    domains = np.asarray([cell[0] for cell in cells])
    templates = np.asarray([cell[1] for cell in cells])
    operations = np.asarray([cell[2] for cell in cells])
    asymmetric = np.isin(operations, ("add", "invalidate"))
    preserve = operations == "preserve"

    factorized = np.empty(len(cells), dtype=np.float64)
    factorized[asymmetric] = oracle + factorized_ai
    factorized[preserve] = oracle + preserve_excess
    shared = factorized.copy()
    shared[asymmetric] = (
        oracle
        + (
            factorized_ai + gain
            if shared_headroom is None
            else shared_headroom
        )
    )
    shared[preserve] = oracle + preserve_excess
    affected = {
        "factorized": factorized,
        "shared": shared,
        "oracle_demand": np.full(len(cells), oracle),
    }
    for control in CONTROL_NAMES:
        controlled = factorized.copy()
        controlled[asymmetric] += control_degradation
        affected[control] = controlled
    return SemanticAnchorSeedMetrics(
        episode_ids=np.asarray(
            [f"r1-{seed}-{index}" for index in range(len(cells))]
        ),
        domains=domains,
        templates=templates,
        operations=operations,
        affected=affected,
        retention={
            "factorized": np.full(len(cells), retention_factorized),
            "shared": np.full(len(cells), retention_shared),
            "oracle_demand": np.full(len(cells), oracle),
        },
    )


def _passing_data(
    *,
    gains: dict[int, float] | None = None,
    **metric_overrides: float,
) -> dict[int, SemanticAnchorSeedMetrics]:
    gains = {} if gains is None else gains
    return {
        seed: _seed_metrics(
            seed,
            gain=gains.get(seed, 0.002),
            **metric_overrides,
        )
        for seed in E05A_R1_SEEDS
    }


def test_r1_go_uses_eight_new_seed_clusters_and_all_registered_gates() -> None:
    report = evaluate_e05a_r1_design(
        _passing_data(), thresholds=_thresholds()
    )

    assert report["status"] == "GO"
    assert report["go"] is True
    assert report["diagnostic_reasons"] == []
    assert report["fixed_seeds"] == list(E05A_R1_SEEDS)
    assert report["original_e05a_outcomes_used_in_inference"] is False
    assert report["h5_claim_open"] is False
    assert report["e05b_execution_allowed"] is False
    assert report["bootstrap"]["seed_clusters_resampled"] is True
    assert report["bootstrap"]["episodes_resampled"] is False
    assert report["bootstrap"]["draws_per_replicate"] == 8
    assert (
        report["bootstrap"]["unit"]
        == "training_seed_and_fresh_namespace_cluster"
    )
    assert report["bootstrap"]["seeds"] == E05A_R1_BOOTSTRAP_SEEDS
    assert set(report["controls"]) == set(CONTROL_NAMES)
    assert all(gate["supported"] for gate in report["controls"].values())
    primary = report["primary_gain_shared_minus_factorized"]
    assert primary["estimate"] == pytest.approx(0.002)
    assert primary["mean_meets_sesoi"] is True
    assert primary["ci_lower_above_zero"] is True
    assert primary["exact_sign_flip"]["null_shift"] == 0.0
    assert primary["exact_sign_flip"]["p"] == pytest.approx(1 / 256)
    assert primary["all_seed_raw_direction_positive"] is True


def test_primary_keeps_mean_sesoi_ci_sign_flip_and_direction_separate() -> None:
    gains = {seed: 0.0009 for seed in E05A_R1_SEEDS}
    report = evaluate_e05a_r1_design(
        _passing_data(gains=gains), thresholds=_thresholds()
    )
    primary = report["primary_gain_shared_minus_factorized"]

    assert primary["mean_meets_sesoi"] is False
    assert primary["ci_lower_above_zero"] is True
    assert primary["exact_sign_flip"]["passed"] is True
    assert primary["all_seed_raw_direction_positive"] is True
    assert primary["supported"] is False
    assert report["status"] == "NO_GO"
    assert "PRIMARY_MEAN_BELOW_SESOI" in report["diagnostic_reasons"]
    assert report["h5_disposition"] == "TERMINATED_NOT_REFUTED"


def test_primary_seed_cluster_bootstrap_reflects_training_seed_heterogeneity() -> None:
    gains = {
        seed: gain
        for seed, gain in zip(
            E05A_R1_SEEDS,
            (0.0002, 0.0004, 0.0006, 0.0010, 0.0018, 0.0022, 0.0026, 0.0030),
            strict=True,
        )
    }
    report = evaluate_e05a_r1_design(
        _passing_data(gains=gains), thresholds=_thresholds()
    )
    low, high = report["primary_gain_shared_minus_factorized"]["ci95"]

    assert low < high
    assert report["bootstrap"]["seed_clusters_resampled"] is True
    assert report["bootstrap"]["paired_conditions_within_cluster"] is True


@pytest.mark.parametrize(
    ("override", "reason"),
    [
        ({"oracle": 2e-8}, "ORACLE_AFFECTED_ABOVE_CEILING"),
        (
            {"factorized_ai": 0.0006},
            "FACTORIZED_AI_ORACLE_EXCESS_ABOVE_MARGIN",
        ),
        (
            {"shared_headroom": 0.001},
            "INSUFFICIENT_ORACLE_HEADROOM",
        ),
        (
            {"preserve_excess": 0.0006},
            "FACTORIZED_PRESERVE_EXCESS_ABOVE_MARGIN",
        ),
        (
            {"retention_factorized": 0.0006},
            "FACTORIZED_ORACLE_EXCESS_FAILED",
        ),
        (
            {
                "retention_factorized": 0.0006,
                "retention_shared": 0.0,
            },
            "FACTORIZED_SHARED_NONINFERIORITY_FAILED",
        ),
        (
            {"control_degradation": 0.001},
            "CONTROL_SHUFFLED_FIELDS_FAILED",
        ),
    ],
)
def test_r1_boundary_failures_are_no_go_not_h5_refutations(
    override: dict[str, float], reason: str
) -> None:
    report = evaluate_e05a_r1_design(
        _passing_data(**override), thresholds=_thresholds()
    )

    assert report["status"] == "NO_GO"
    assert report["go"] is False
    assert reason in report["diagnostic_reasons"]
    assert report["h5_disposition"] == "TERMINATED_NOT_REFUTED"
    assert report["h5_claim_open"] is False


def test_all_five_controls_use_shifted_tests_and_positive_raw_directions() -> None:
    report = evaluate_e05a_r1_design(
        _passing_data(), thresholds=_thresholds()
    )
    for gate in report["controls"].values():
        assert gate["ci_lower_above_threshold"] is True
        assert gate["shifted_exact_sign_flip"]["null_shift"] == pytest.approx(
            0.001
        )
        assert gate["shifted_exact_sign_flip"]["p"] == pytest.approx(1 / 256)
        assert gate["all_seed_raw_direction_positive"] is True


def test_fixed_seed_and_bootstrap_contracts_are_exact() -> None:
    data = _passing_data()
    with pytest.raises(ValueError, match="exactly eight"):
        evaluate_e05a_r1_design(
            data,
            fixed_seeds=E05A_R1_SEEDS[:-1],
            thresholds=_thresholds(),
        )
    with pytest.raises(TypeError, match="exact tuple"):
        evaluate_e05a_r1_design(
            data,
            fixed_seeds=list(E05A_R1_SEEDS),  # type: ignore[arg-type]
            thresholds=_thresholds(),
        )
    reused = (101, *E05A_R1_SEEDS[1:])
    reused_data = {
        seed: _seed_metrics(seed)
        for seed in reused
    }
    with pytest.raises(ValueError, match="must not reuse original"):
        evaluate_e05a_r1_design(
            reused_data,
            fixed_seeds=reused,
            thresholds=_thresholds(),
        )
    invalid_bootstrap = dict(E05A_R1_BOOTSTRAP_SEEDS)
    invalid_bootstrap.pop("primary_gain")
    with pytest.raises(ValueError, match="keys differ"):
        evaluate_e05a_r1_design(
            data,
            thresholds=_thresholds(),
            bootstrap_seeds=invalid_bootstrap,
        )
    wrong_data = dict(data)
    wrong_data.pop(E05A_R1_SEEDS[-1])
    with pytest.raises(ValueError, match="expected fixed seeds"):
        evaluate_e05a_r1_design(
            wrong_data,
            thresholds=_thresholds(),
        )


def test_default_r1_seeds_do_not_reuse_original_e05a_seeds() -> None:
    assert set(E05A_R1_SEEDS).isdisjoint({101, 202, 303, 404})
