import numpy as np
import pytest

from catena.eval.seed_inference import exact_sign_flip_test
from catena.eval.statistics_v61 import (
    evaluate_fixed_effect_oos,
    fixed_seed_operation_stratified_bootstrap,
)


def _two_operation_design() -> tuple[np.ndarray, np.ndarray]:
    within_operation_x = np.array([-1.0, -0.5, 0.5, 1.0])
    operations = np.array(["preserve"] * 4 + ["add"] * 4)
    return np.tile(within_operation_x, 2), operations


def test_operation_only_signal_does_not_count_as_conditional_oos_fit() -> None:
    x, operations = _two_operation_design()
    operation_intercept = np.where(operations == "add", 10.0, 0.0)
    within_operation_noise = np.tile(np.array([0.1, -0.1, -0.1, 0.1]), 2)
    y = operation_intercept + within_operation_noise

    report = evaluate_fixed_effect_oos(
        x,
        y,
        operations,
        x,
        y,
        operations,
        operation_order=["preserve", "add"],
    )

    assert report.full_r2 > 0.99
    assert report.operation_only_r2 > 0.99
    assert report.conditional_r2 == pytest.approx(0.0, abs=1e-12)
    assert report.train_slope == pytest.approx(0.0, abs=1e-12)
    assert report.test_slope == pytest.approx(0.0, abs=1e-12)


def test_unseen_geometry_sign_reversal_is_exposed_by_test_slope() -> None:
    x, operations = _two_operation_design()
    operation_intercept = np.where(operations == "add", 3.0, -2.0)
    train_y = operation_intercept + 2.0 * x
    test_y = operation_intercept - 2.0 * x

    report = evaluate_fixed_effect_oos(
        x,
        train_y,
        operations,
        x,
        test_y,
        operations,
        operation_order=["preserve", "add"],
    )

    assert report.train_slope == pytest.approx(2.0)
    assert report.test_slope == pytest.approx(-2.0)
    assert report.conditional_r2 < 0.0


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_sign_flip_rejects_nonfinite_values(bad_value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        exact_sign_flip_test(np.array([1.0, bad_value]), "greater")


def test_sign_flip_rejects_unknown_alternative() -> None:
    with pytest.raises(ValueError, match="alternative"):
        exact_sign_flip_test(np.ones(8), "upward")


def test_fixed_seed_stratified_bootstrap_is_constant_and_reproducible() -> None:
    operations = {
        11: np.array(["add", "add", "preserve", "preserve"]),
        22: np.array(["add", "add", "preserve", "preserve"]),
    }
    values = {
        11: np.full(4, 0.25),
        22: np.full(4, 0.75),
    }
    observed_seed_sets: list[tuple[int, ...]] = []

    def statistic(indices_by_seed: dict[int, np.ndarray]) -> float:
        observed_seed_sets.append(tuple(sorted(indices_by_seed)))
        return float(
            np.mean(
                [
                    values[seed][indices].mean()
                    for seed, indices in indices_by_seed.items()
                ]
            )
        )

    first = fixed_seed_operation_stratified_bootstrap(
        operations, statistic, samples=100, seed=7
    )
    second = fixed_seed_operation_stratified_bootstrap(
        operations, statistic, samples=100, seed=7
    )

    assert first == second
    assert first.estimate == pytest.approx(0.5)
    assert first.low == pytest.approx(0.5)
    assert first.high == pytest.approx(0.5)
    assert set(observed_seed_sets) == {(11, 22)}
