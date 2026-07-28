from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from catena.eval.semantic_anchor_v61 import (
    CONTROL_NAMES,
    E05A_BOOTSTRAP_SEEDS,
    E05A_SEEDS,
    E05B_BOOTSTRAP_SEEDS,
    E05B_SEEDS,
    SemanticAnchorSeedMetrics,
    SemanticAnchorThresholds,
    evaluate_e05a_go,
    evaluate_e05b_main,
    evaluate_e05b_secondary,
    evaluate_e05b_validation,
    exact_seed_tost,
)


def _labels(
    operations: tuple[str, ...], *, repeats: int = 2
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cells = list(product(("api", "access"), ("structured", "indirect"), operations))
    domains: list[str] = []
    templates: list[str] = []
    operation_values: list[str] = []
    for domain, template, operation in cells:
        for _ in range(repeats):
            domains.append(domain)
            templates.append(template)
            operation_values.append(operation)
    return (
        np.asarray(domains),
        np.asarray(templates),
        np.asarray(operation_values),
    )


def _metrics(
    *,
    seed: int,
    operations: tuple[str, ...],
    affected: dict[str, np.ndarray | float],
    retention: dict[str, np.ndarray | float],
    repeats: int = 2,
) -> SemanticAnchorSeedMetrics:
    domains, templates, operation_values = _labels(operations, repeats=repeats)
    length = len(domains)

    def vector(value: np.ndarray | float) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64)
        if array.ndim == 0:
            return np.full(length, float(array), dtype=np.float64)
        if len(array) != length:
            raise AssertionError("test vector length mismatch")
        return array

    return SemanticAnchorSeedMetrics(
        episode_ids=np.asarray([f"{seed}-{index}" for index in range(length)]),
        domains=domains,
        templates=templates,
        operations=operation_values,
        affected={name: vector(value) for name, value in affected.items()},
        retention={name: vector(value) for name, value in retention.items()},
    )


def _thresholds(*, samples: int = 128) -> SemanticAnchorThresholds:
    return SemanticAnchorThresholds(bootstrap_samples=samples)


def _e05b_bootstrap_seeds(*, headroom_closure: int | None = None) -> dict[str, int]:
    result = dict(E05B_BOOTSTRAP_SEEDS)
    if headroom_closure is not None:
        result["headroom_closure"] = headroom_closure
    return result


def _e05a_data() -> dict[int, SemanticAnchorSeedMetrics]:
    result = {}
    for seed in E05A_SEEDS:
        affected: dict[str, np.ndarray | float] = {
            "factorized": 1e-4,
            "shared": 1e-4,
            "oracle_demand": 0.0,
        }
        affected.update({control: 0.01 for control in CONTROL_NAMES})
        result[seed] = _metrics(
            seed=seed,
            operations=("preserve", "add", "invalidate"),
            affected=affected,
            retention={
                "factorized": 1e-4,
                "shared": 1e-4,
                "oracle_demand": 0.0,
            },
        )
    return result


def _validation_data() -> dict[int, SemanticAnchorSeedMetrics]:
    return {
        seed: _metrics(
            seed=seed,
            operations=("preserve", "add", "invalidate"),
            affected={
                "factorized": 1e-4,
                "shared": 1e-4,
                "oracle_demand": 0.0,
            },
            retention={
                "factorized": 1e-4,
                "shared": 1e-4,
                "oracle_demand": 0.0,
            },
        )
        for seed in E05B_SEEDS
    }


def _primary_data(
    *,
    shared: np.ndarray | float = 0.02,
    factorized: np.ndarray | float = 0.004,
) -> dict[int, SemanticAnchorSeedMetrics]:
    result = {}
    for seed in E05B_SEEDS:
        affected: dict[str, np.ndarray | float] = {
            "factorized": factorized,
            "shared": shared,
            "oracle_demand": 0.0,
        }
        if np.asarray(factorized).ndim == 0:
            control_value: np.ndarray | float = float(factorized) + 0.01
        else:
            control_value = np.asarray(factorized, dtype=np.float64) + 0.01
        affected.update({control: control_value for control in CONTROL_NAMES})
        result[seed] = _metrics(
            seed=seed,
            operations=("supersede",),
            affected=affected,
            retention={
                "factorized": 1e-4,
                "shared": 1e-4,
                "oracle_demand": 0.0,
            },
        )
    return result


def test_exact_seed_tost_is_exact_and_requires_eight_seeds_at_alpha_05() -> None:
    eight = exact_seed_tost(np.zeros(8), margin=0.0005)
    four = exact_seed_tost(np.zeros(4), margin=0.0005)
    assert eight["lower_p"] == pytest.approx(1 / 256)
    assert eight["upper_p"] == pytest.approx(1 / 256)
    assert eight["passed"] is True
    assert four["lower_p"] == pytest.approx(1 / 16)
    assert four["upper_p"] == pytest.approx(1 / 16)
    assert four["passed"] is False


def test_e05a_go_requires_oracle_excess_parity_and_all_five_controls() -> None:
    report = evaluate_e05a_go(_e05a_data(), thresholds=_thresholds())
    assert report["status"] == "GO"
    assert report["claim_evidence"] is False
    assert report["go"] is True
    assert set(report["controls"]) == set(CONTROL_NAMES)
    assert all(gate["supported"] for gate in report["controls"].values())
    assert report["bootstrap"]["paired_indices_across_models_and_controls"] is True
    assert (
        report["bootstrap"]["unit"]
        == "episode_within_seed_domain_template_operation_stratum"
    )
    assert report["bootstrap"]["seeds"] == E05A_BOOTSTRAP_SEEDS
    assert report["control_estimand"]["operations"] == ["add", "invalidate"]
    assert report["retention_parity_is_descriptive_only"] is True

    failed = _e05a_data()
    first = E05A_SEEDS[0]
    seed_data = failed[first]
    affected = dict(seed_data.affected)
    affected["transaction_only"] = np.asarray(affected["factorized"]).copy()
    failed[first] = SemanticAnchorSeedMetrics(
        episode_ids=seed_data.episode_ids,
        domains=seed_data.domains,
        templates=seed_data.templates,
        operations=seed_data.operations,
        affected=affected,
        retention=seed_data.retention,
    )
    failed_report = evaluate_e05a_go(failed, thresholds=_thresholds())
    assert failed_report["status"] == "NO_GO"
    assert failed_report["controls"]["transaction_only"]["supported"] is False


def test_e05a_control_conjunction_excludes_preserve() -> None:
    data = _e05a_data()
    for seed, seed_data in data.items():
        affected = dict(seed_data.affected)
        transaction_only = np.asarray(affected["transaction_only"]).copy()
        preserve = np.asarray(seed_data.operations) == "preserve"
        transaction_only[preserve] = np.asarray(affected["factorized"])[preserve]
        affected["transaction_only"] = transaction_only
        data[seed] = SemanticAnchorSeedMetrics(
            episode_ids=seed_data.episode_ids,
            domains=seed_data.domains,
            templates=seed_data.templates,
            operations=seed_data.operations,
            affected=affected,
            retention=seed_data.retention,
        )
    report = evaluate_e05a_go(data, thresholds=_thresholds())
    assert report["status"] == "GO"
    assert report["controls"]["transaction_only"]["supported"] is True


def test_e05b_supports_registered_d_h_q_r_c_conjunction() -> None:
    validation = _validation_data()
    validation_report = evaluate_e05b_validation(
        validation,
        thresholds=_thresholds(),
    )
    report = evaluate_e05b_main(
        validation=validation,
        primary=_primary_data(),
        thresholds=_thresholds(),
    )
    assert (
        report["validation_factorized_shared_equivalence"]
        == validation_report["validation_factorized_shared_equivalence"]
    )
    assert report["status"] == "SUPPORTED"
    assert report["supported"] is True
    assert report["headroom_identifiable"] is True
    assert report["validation_factorized_shared_equivalence"]["supported"] is True
    assert report["D_shared_minus_factorized"]["supported"] is True
    assert report["H_shared_minus_oracle"]["supported"] is True
    assert report["Q_headroom_fraction_closed"]["supported"] is True
    assert report["R_factorized_minus_shared_retention"]["supported"] is True
    assert all(gate["supported"] for gate in report["controls"].values())
    assert report["bootstrap"]["seeds"] == E05B_BOOTSTRAP_SEEDS
    assert (
        report["factorized_absolute_retention"]["shifted_exact_sign_flip_p"]
        is None
    )
    assert (
        report["D_shared_minus_factorized"]["shifted_exact_sign_flip_p"]
        == pytest.approx(1 / 256)
    )


def test_e05b_validation_seal_passes_without_primary_registry() -> None:
    report = evaluate_e05b_validation(
        _validation_data(),
        thresholds=_thresholds(),
    )
    assert report["status"] == "PASS"
    assert report["passed"] is True
    assert "supported" not in report
    assert report["primary_data_received"] is False
    assert (
        report["bootstrap"]["unit"]
        == "episode_within_seed_domain_template_stratum"
    )
    assert report["validation_factorized_shared_equivalence"]["supported"] is True


def test_e05b_validation_seal_fails_without_primary_registry() -> None:
    validation = _validation_data()
    for seed, seed_data in validation.items():
        affected = dict(seed_data.affected)
        affected["shared"] = np.full_like(
            np.asarray(affected["shared"]), 0.0021, dtype=np.float64
        )
        validation[seed] = SemanticAnchorSeedMetrics(
            episode_ids=seed_data.episode_ids,
            domains=seed_data.domains,
            templates=seed_data.templates,
            operations=seed_data.operations,
            affected=affected,
            retention=seed_data.retention,
        )
    report = evaluate_e05b_validation(
        validation,
        thresholds=_thresholds(),
    )
    gate = report["validation_factorized_shared_equivalence"]
    assert report["status"] == "NO_GO_MAIN_SEALED"
    assert report["passed"] is False
    assert report["primary_data_received"] is False
    assert gate["ci_within_margin"] is False
    assert gate["supported"] is False


def test_e05b_secondary_reports_frozen_descriptive_estimands_without_gates() -> None:
    report = evaluate_e05b_secondary(
        _primary_data(),
        thresholds=_thresholds(samples=256),
    )
    assert report["status"] == "DESCRIPTIVE_ONLY"
    assert report["descriptive_only"] is True
    assert report["contributes_to_primary_gate"] is False
    assert report["D_shared_minus_factorized"]["estimate"] == pytest.approx(0.016)
    assert report["H_shared_minus_oracle"]["estimate"] == pytest.approx(0.02)
    assert report["Q_headroom_fraction_closed"]["estimate"] == pytest.approx(0.8)
    assert report["R_factorized_minus_shared_retention"]["estimate"] == pytest.approx(
        0.0
    )
    assert report["bootstrap"]["seeds"] == {
        "D_shared_minus_factorized": 5205,
        "H_shared_minus_oracle": 5204,
        "Q_headroom_fraction_closed": 5206,
        "R_factorized_minus_shared_retention": 5207,
        "oracle_affected": 5202,
        "oracle_retention": 5203,
        "factorized_retention_minus_oracle": 5208,
    }
    assert (
        report["controls"]
        == "NOT_EVALUATED_NO_FROZEN_SECONDARY_CONTROL_PAIRINGS"
    )
    assert "supported" not in report
    for name in (
        "D_shared_minus_factorized",
        "H_shared_minus_oracle",
        "Q_headroom_fraction_closed",
        "R_factorized_minus_shared_retention",
    ):
        assert "supported" not in report[name]


def test_e05b_secondary_q_is_non_evaluable_at_frozen_headroom_boundary() -> None:
    report = evaluate_e05b_secondary(
        _primary_data(shared=0.001, factorized=0.0),
        thresholds=_thresholds(),
    )
    q_payload = report["Q_headroom_fraction_closed"]
    assert q_payload["evaluable"] is False
    assert (
        q_payload["reason"]
        == "POINT_SEED_HEADROOM_AT_OR_BELOW_MINIMUM"
    )


def test_paired_bootstrap_uses_identical_episode_indices_for_models() -> None:
    _, _, operations = _labels(("supersede",), repeats=2)
    common_noise = np.linspace(0.0, 0.003, len(operations))
    primary = _primary_data(
        factorized=0.004 + common_noise,
        shared=0.019 + common_noise,
    )
    report = evaluate_e05b_main(
        validation=_validation_data(),
        primary=primary,
        thresholds=_thresholds(samples=256),
    )
    d_gate = report["D_shared_minus_factorized"]
    assert d_gate["estimate"] == pytest.approx(0.015)
    assert d_gate["ci95"][0] == pytest.approx(0.015)
    assert d_gate["ci95"][1] == pytest.approx(0.015)


def test_e05b_point_headroom_at_threshold_is_inconclusive() -> None:
    report = evaluate_e05b_main(
        validation=_validation_data(),
        primary=_primary_data(shared=0.001, factorized=0.0),
        thresholds=_thresholds(),
    )
    assert report["status"] == "INCONCLUSIVE_ORACLE_HEADROOM"
    assert report["headroom_identifiable"] is False
    assert report["Q_headroom_fraction_closed"]["evaluable"] is False
    assert (
        report["Q_headroom_fraction_closed"]["reason"]
        == "POINT_SEED_HEADROOM_AT_OR_BELOW_MINIMUM"
    )


def test_e05b_any_bootstrap_seed_headroom_failure_is_inconclusive() -> None:
    _, _, operations = _labels(("supersede",), repeats=2)
    alternating = np.tile(np.asarray([0.0, 0.0022]), len(operations) // 2)
    assert alternating.mean() > 0.001
    report = evaluate_e05b_main(
        validation=_validation_data(),
        primary=_primary_data(shared=alternating, factorized=0.0),
        thresholds=_thresholds(samples=256),
        bootstrap_seeds=_e05b_bootstrap_seeds(headroom_closure=2),
    )
    assert report["status"] == "INCONCLUSIVE_ORACLE_HEADROOM"
    assert report["headroom_identifiable"] is False
    assert (
        report["Q_headroom_fraction_closed"]["reason"]
        == "BOOTSTRAP_SEED_HEADROOM_AT_OR_BELOW_MINIMUM"
    )


def test_e05b_validation_equivalence_does_not_require_every_seed_in_margin() -> None:
    validation = _validation_data()
    seed = E05B_SEEDS[-1]
    seed_data = validation[seed]
    affected = dict(seed_data.affected)
    affected["shared"] = np.full_like(
        np.asarray(affected["shared"]), 7e-4, dtype=np.float64
    )
    validation[seed] = SemanticAnchorSeedMetrics(
        episode_ids=seed_data.episode_ids,
        domains=seed_data.domains,
        templates=seed_data.templates,
        operations=seed_data.operations,
        affected=affected,
        retention=seed_data.retention,
    )
    report = evaluate_e05b_main(
        validation=validation,
        primary=_primary_data(),
        thresholds=_thresholds(samples=256),
    )
    parity = report["validation_factorized_shared_equivalence"]
    assert parity["all_seed_means_within_margin"] is False
    assert parity["ci_within_margin"] is True
    assert parity["exact_seed_tost"]["passed"] is True
    assert parity["supported"] is True


def test_e05b_oracle_and_absolute_retention_use_ci_not_all_seed_conjunction() -> None:
    primary = _primary_data()
    seed = E05B_SEEDS[-1]
    seed_data = primary[seed]
    affected = dict(seed_data.affected)
    retention = dict(seed_data.retention)
    affected["oracle_demand"] = np.full_like(
        np.asarray(affected["oracle_demand"]), 4e-8, dtype=np.float64
    )
    retention["factorized"] = np.full_like(
        np.asarray(retention["factorized"]), 9e-4, dtype=np.float64
    )
    primary[seed] = SemanticAnchorSeedMetrics(
        episode_ids=seed_data.episode_ids,
        domains=seed_data.domains,
        templates=seed_data.templates,
        operations=seed_data.operations,
        affected=affected,
        retention=retention,
    )
    report = evaluate_e05b_main(
        validation=_validation_data(),
        primary=primary,
        thresholds=_thresholds(samples=256),
    )
    oracle = report["oracle_low"]["affected"]
    absolute_retention = report["factorized_absolute_retention"]
    assert oracle["all_seed_means_within_upper"] is False
    assert oracle["supported"] is True
    assert absolute_retention["all_seed_means_within_upper"] is False
    assert absolute_retention["shifted_exact_sign_flip_p"] is None
    assert absolute_retention["supported"] is True


def test_bootstrap_seed_mapping_must_match_frozen_gate_keys() -> None:
    bad = dict(E05A_BOOTSTRAP_SEEDS)
    bad.pop("oracle_affected")
    with pytest.raises(ValueError, match="frozen protocol"):
        evaluate_e05a_go(
            _e05a_data(),
            thresholds=_thresholds(),
            bootstrap_seeds=bad,
        )


def test_semantic_anchor_rejects_nonfinite_rows_without_exclusion() -> None:
    data = _e05a_data()
    seed = E05A_SEEDS[0]
    seed_data = data[seed]
    affected = dict(seed_data.affected)
    bad = np.asarray(affected["factorized"]).copy()
    bad[0] = np.nan
    affected["factorized"] = bad
    data[seed] = SemanticAnchorSeedMetrics(
        episode_ids=seed_data.episode_ids,
        domains=seed_data.domains,
        templates=seed_data.templates,
        operations=seed_data.operations,
        affected=affected,
        retention=seed_data.retention,
    )
    with pytest.raises(ValueError, match="finite"):
        evaluate_e05a_go(data, thresholds=_thresholds())


def test_semantic_anchor_rejects_negative_mse() -> None:
    data = _e05a_data()
    seed = E05A_SEEDS[0]
    seed_data = data[seed]
    affected = dict(seed_data.affected)
    bad = np.asarray(affected["factorized"]).copy()
    bad[0] = -1e-9
    affected["factorized"] = bad
    data[seed] = SemanticAnchorSeedMetrics(
        episode_ids=seed_data.episode_ids,
        domains=seed_data.domains,
        templates=seed_data.templates,
        operations=seed_data.operations,
        affected=affected,
        retention=seed_data.retention,
    )
    with pytest.raises(ValueError, match="negative MSE"):
        evaluate_e05a_go(data, thresholds=_thresholds())


def test_semantic_anchor_requires_the_frozen_seed_sets() -> None:
    data = _e05a_data()
    data.pop(E05A_SEEDS[-1])
    with pytest.raises(ValueError, match="expected fixed seeds"):
        evaluate_e05a_go(data, thresholds=_thresholds())
