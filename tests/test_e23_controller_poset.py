from __future__ import annotations

from itertools import product

import pytest

from catena.data.controller_poset import (
    CANONICAL_CONTROLLERS,
    CONTROLLER_BY_ID,
    DEMAND_FAMILIES,
    controller_leq,
    epsilon_minimal_controllers,
    immediate_lower_covers,
    immediate_upper_covers,
    minimal_elements,
    required_controller,
    same_rank_incomparable_controllers,
    theory_boundary_controller_ids,
    theory_minimal_controller_ids,
)
from catena.post_e21.product_poset_model import (
    ProductPosetProbeConfig,
    theoretical_affected_error,
)


def test_canonical_controller_family_has_exact_16_bit_patterns() -> None:
    assert len(CANONICAL_CONTROLLERS) == 16
    assert [controller.controller_id for controller in CANONICAL_CONTROLLERS] == [
        f"c{mask:04b}" for mask in range(16)
    ]
    assert len(CONTROLLER_BY_ID) == 16


def test_controller_relation_is_a_partial_order() -> None:
    for controller in CANONICAL_CONTROLLERS:
        assert controller_leq(controller, controller)
    for left, right in product(CANONICAL_CONTROLLERS, repeat=2):
        if controller_leq(left, right) and controller_leq(right, left):
            assert left == right
    for left, middle, right in product(CANONICAL_CONTROLLERS, repeat=3):
        if controller_leq(left, middle) and controller_leq(middle, right):
            assert controller_leq(left, right)


def test_minimal_elements_handles_incomparable_members() -> None:
    observed = minimal_elements(
        (
            CONTROLLER_BY_ID["c1000"],
            CONTROLLER_BY_ID["c0100"],
            CONTROLLER_BY_ID["c1100"],
        )
    )
    assert tuple(value.controller_id for value in observed) == (
        "c0100",
        "c1000",
    )


@pytest.mark.parametrize("demand", DEMAND_FAMILIES)
@pytest.mark.parametrize("intensity", (0.25, 0.5, 1.0))
@pytest.mark.parametrize("updates,gap", ((1, 0), (4, 512), (8, 2048)))
def test_theory_epsilon_minimal_set_matches_required_bits(
    demand: str,
    intensity: float,
    updates: int,
    gap: int,
) -> None:
    probe = ProductPosetProbeConfig(
        missing_axis_floor=0.004,
        numerical_floor=0.000001,
    )
    errors = {
        controller.controller_id: theoretical_affected_error(
            controller=controller,
            demand_family=demand,
            intensity=intensity,
            updates=updates,
            gap_events=gap,
            config=probe,
        )
        for controller in CANONICAL_CONTROLLERS
    }
    observed = tuple(
        value.controller_id for value in epsilon_minimal_controllers(errors, epsilon=0.0001)
    )
    assert observed == theory_minimal_controller_ids(demand)
    assert observed == (required_controller(demand).controller_id,)


def test_theory_boundary_is_derived_without_outcome_input() -> None:
    first = {demand: theory_boundary_controller_ids(demand) for demand in DEMAND_FAMILIES}
    second = {
        demand: theory_boundary_controller_ids(demand) for demand in reversed(DEMAND_FAMILIES)
    }
    assert first == second
    assert "c0000" in first["preserve"]
    assert "c1100" in first["magnitude_value"]
    assert set().union(*(set(values) for values in first.values())) == set(CONTROLLER_BY_ID)
    for demand in DEMAND_FAMILIES:
        target = required_controller(demand)
        boundary = set(first[demand])
        assert target.controller_id in boundary
        assert "c1111" in boundary
        assert {
            controller.controller_id for controller in immediate_lower_covers(target)
        } <= boundary
        assert {
            controller.controller_id for controller in immediate_upper_covers(target)
        } <= boundary
        assert {
            controller.controller_id for controller in same_rank_incomparable_controllers(target)
        } <= boundary
