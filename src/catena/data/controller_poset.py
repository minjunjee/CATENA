"""Canonical four-axis controller poset used by E23.

The controller family is the Boolean lattice over magnitude, value,
address, and state-conditioning freedom.  This module is intentionally free
of learned outcomes: every order relation, demand requirement, and theory
boundary is determined from the four-bit descriptors alone.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations

CONTROLLER_AXES: tuple[str, ...] = (
    "magnitude",
    "value",
    "address",
    "conditioning",
)

SINGLE_AXIS_DEMANDS: tuple[str, ...] = CONTROLLER_AXES
PAIRWISE_DEMANDS: tuple[str, ...] = tuple(
    f"{left}_{right}" for left, right in combinations(CONTROLLER_AXES, 2)
)
DEMAND_FAMILIES: tuple[str, ...] = (
    *SINGLE_AXIS_DEMANDS,
    *PAIRWISE_DEMANDS,
    "preserve",
)


@dataclass(frozen=True, order=True, slots=True)
class ControllerSpec:
    """One canonical controller in the four-dimensional Boolean lattice."""

    bits: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        if len(self.bits) != len(CONTROLLER_AXES):
            raise ValueError("Controller bits must have length four")
        if any(bit not in (0, 1) for bit in self.bits):
            raise ValueError("Controller bits must be binary")

    @property
    def controller_id(self) -> str:
        return "c" + "".join(str(bit) for bit in self.bits)

    @property
    def rank(self) -> int:
        return sum(self.bits)

    @property
    def enabled_axes(self) -> tuple[str, ...]:
        return tuple(axis for axis, bit in zip(CONTROLLER_AXES, self.bits, strict=True) if bit)

    def as_dict(self) -> dict[str, object]:
        return {
            "controller_id": self.controller_id,
            "bits": list(self.bits),
            "rank": self.rank,
            "enabled_axes": list(self.enabled_axes),
        }


def canonical_controllers() -> tuple[ControllerSpec, ...]:
    """Return all 16 controllers in lexicographic bit-string order."""

    return tuple(
        ControllerSpec(
            (
                (mask >> 3) & 1,
                (mask >> 2) & 1,
                (mask >> 1) & 1,
                mask & 1,
            )
        )
        for mask in range(16)
    )


CANONICAL_CONTROLLERS: tuple[ControllerSpec, ...] = canonical_controllers()
CONTROLLER_BY_ID: dict[str, ControllerSpec] = {
    controller.controller_id: controller for controller in CANONICAL_CONTROLLERS
}


def controller_leq(left: ControllerSpec, right: ControllerSpec) -> bool:
    """Return whether ``left`` is no richer than ``right`` on every axis."""

    return all(
        left_bit <= right_bit for left_bit, right_bit in zip(left.bits, right.bits, strict=True)
    )


def controller_lt(left: ControllerSpec, right: ControllerSpec) -> bool:
    return left != right and controller_leq(left, right)


def minimal_elements(
    controllers: Iterable[ControllerSpec],
) -> tuple[ControllerSpec, ...]:
    """Return the order-minimal members of an arbitrary controller subset."""

    unique = tuple(sorted(set(controllers)))
    return tuple(
        candidate
        for candidate in unique
        if not any(controller_lt(other, candidate) for other in unique if other != candidate)
    )


def epsilon_adequate_controllers(
    errors: Mapping[str, float],
    *,
    epsilon: float,
) -> tuple[ControllerSpec, ...]:
    """Return controllers within ``epsilon`` of the best finite error."""

    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    expected = set(CONTROLLER_BY_ID)
    if set(errors) != expected:
        missing = sorted(expected - set(errors))
        extra = sorted(set(errors) - expected)
        raise ValueError(
            f"Error map must cover the 16 canonical controllers; missing={missing}, extra={extra}"
        )
    numeric = {name: float(value) for name, value in errors.items()}
    if any(value != value or value in (float("inf"), float("-inf")) for value in numeric.values()):
        raise ValueError("Controller errors must be finite")
    floor = min(numeric.values())
    tolerance = floor + float(epsilon)
    return tuple(CONTROLLER_BY_ID[name] for name in sorted(numeric) if numeric[name] <= tolerance)


def epsilon_minimal_controllers(
    errors: Mapping[str, float],
    *,
    epsilon: float,
) -> tuple[ControllerSpec, ...]:
    """Return poset-minimal controllers within epsilon of the best error."""

    return minimal_elements(epsilon_adequate_controllers(errors, epsilon=epsilon))


def demand_required_axes(demand_family: str) -> tuple[str, ...]:
    if demand_family == "preserve":
        return ()
    if demand_family in SINGLE_AXIS_DEMANDS:
        return (demand_family,)
    if demand_family in PAIRWISE_DEMANDS:
        components = tuple(demand_family.split("_"))
        if len(components) != 2:
            raise AssertionError("Pairwise demand identifier is malformed")
        return components
    raise ValueError(f"Unknown demand family: {demand_family!r}")


def required_controller(demand_family: str) -> ControllerSpec:
    required = set(demand_required_axes(demand_family))
    return ControllerSpec(
        tuple(int(axis in required) for axis in CONTROLLER_AXES)  # type: ignore[arg-type]
    )


def missing_required_axes(
    controller: ControllerSpec,
    demand_family: str,
) -> tuple[str, ...]:
    required = set(demand_required_axes(demand_family))
    enabled = set(controller.enabled_axes)
    return tuple(axis for axis in CONTROLLER_AXES if axis in required - enabled)


def theory_minimal_controller_ids(demand_family: str) -> tuple[str, ...]:
    """Theory prediction locked before E23 outcome evaluation."""

    return (required_controller(demand_family).controller_id,)


def immediate_lower_covers(controller: ControllerSpec) -> tuple[ControllerSpec, ...]:
    covers = []
    for index, bit in enumerate(controller.bits):
        if bit:
            bits = list(controller.bits)
            bits[index] = 0
            covers.append(ControllerSpec(tuple(bits)))  # type: ignore[arg-type]
    return tuple(sorted(covers))


def immediate_upper_covers(controller: ControllerSpec) -> tuple[ControllerSpec, ...]:
    covers = []
    for index, bit in enumerate(controller.bits):
        if not bit:
            bits = list(controller.bits)
            bits[index] = 1
            covers.append(ControllerSpec(tuple(bits)))  # type: ignore[arg-type]
    return tuple(sorted(covers))


def same_rank_incomparable_controllers(
    controller: ControllerSpec,
) -> tuple[ControllerSpec, ...]:
    """Return registered same-cardinality alternatives to ``controller``."""

    return tuple(
        candidate
        for candidate in CANONICAL_CONTROLLERS
        if candidate != controller
        and candidate.rank == controller.rank
        and not controller_leq(candidate, controller)
        and not controller_leq(controller, candidate)
    )


def theory_adequate_controller_ids(
    demand_family: str,
) -> tuple[str, ...]:
    """Controllers whose enabled axes contain every required demand axis."""

    return tuple(
        controller.controller_id
        for controller in CANONICAL_CONTROLLERS
        if not missing_required_axes(controller, demand_family)
    )


def theory_boundary_controller_ids(demand_family: str) -> tuple[str, ...]:
    """Result-independent E23 boundary, frozen before any learned outcome."""

    target = required_controller(demand_family)
    boundary = {
        *immediate_lower_covers(target),
        target,
        *immediate_upper_covers(target),
        *same_rank_incomparable_controllers(target),
        CONTROLLER_BY_ID["c1111"],
    }
    return tuple(controller.controller_id for controller in sorted(boundary))


def theory_prediction_payload(
    *,
    affected_mse_tolerance: float,
    retention_mse_tolerance: float,
    locality_mse_tolerance: float,
    intensities: Sequence[float],
    updates: Sequence[int],
    gap_events: Sequence[int],
) -> dict[str, object]:
    """Build the complete outcome-independent theory declaration."""

    return {
        "rule": "absolute_adequacy_then_poset_minimal_v2",
        "absolute_adequacy": {
            "affected_mse_tolerance": float(affected_mse_tolerance),
            "retention_mse_tolerance": float(retention_mse_tolerance),
            "locality_mse_tolerance": float(locality_mse_tolerance),
            "capacity_only_uses_locality": False,
            "safe_minimality_uses_locality": True,
        },
        "controller_axes": list(CONTROLLER_AXES),
        "controller_ids": [controller.controller_id for controller in CANONICAL_CONTROLLERS],
        "demand_families": list(DEMAND_FAMILIES),
        "intensities": [float(value) for value in intensities],
        "sequence_grid": {
            "updates": [int(value) for value in updates],
            "gap_events": [int(value) for value in gap_events],
        },
        "poset_minimal_sets": {
            demand: list(theory_minimal_controller_ids(demand)) for demand in DEMAND_FAMILIES
        },
        "theory_adequate_sets": {
            demand: list(theory_adequate_controller_ids(demand)) for demand in DEMAND_FAMILIES
        },
        "confirmatory_boundary_sets": {
            demand: list(theory_boundary_controller_ids(demand)) for demand in DEMAND_FAMILIES
        },
        "adequacy_aggregates_worst_case_over_registered_sequence_grid": True,
        "boundary_union_covers_all_16_controllers": (
            set().union(
                *(set(theory_boundary_controller_ids(demand)) for demand in DEMAND_FAMILIES)
            )
            == set(CONTROLLER_BY_ID)
        ),
        "outcome_fields_used": [],
    }
