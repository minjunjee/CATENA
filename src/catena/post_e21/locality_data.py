from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class LocalityObjective(StrEnum):
    MEAN = "mean"
    CVAR = "cvar"
    SMOOTH_MAX = "smoothmax"
    SPARSE_ROUTE = "sparse"
    PROTECTED_DIAGNOSTIC = "protected_projection"


@dataclass(frozen=True, slots=True)
class LocalityMethod:
    method_id: str
    objective: LocalityObjective
    selection_eligible: bool
    baseline: bool
    tail_fraction: float | None = None
    normalized_temperature: float | None = None
    active_fraction: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "objective": self.objective.value,
            "selection_eligible": self.selection_eligible,
            "baseline": self.baseline,
            "tail_fraction": self.tail_fraction,
            "normalized_temperature": self.normalized_temperature,
            "active_fraction": self.active_fraction,
        }


def _optional_fraction(
    row: Mapping[str, Any],
    key: str,
) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    result = float(value)
    if not 0.0 < result <= 1.0:
        raise ValueError(f"{key} must lie in (0, 1]")
    return result


def parse_locality_methods(rows: Iterable[Mapping[str, Any]]) -> list[LocalityMethod]:
    methods: list[LocalityMethod] = []
    observed: set[str] = set()
    for raw in rows:
        method_id = str(raw["method_id"])
        if not method_id or method_id in observed:
            raise ValueError(f"Duplicate/empty E22 method id: {method_id!r}")
        observed.add(method_id)
        objective = LocalityObjective(str(raw["objective"]))
        method = LocalityMethod(
            method_id=method_id,
            objective=objective,
            selection_eligible=bool(raw.get("selection_eligible", False)),
            baseline=bool(raw.get("baseline", False)),
            tail_fraction=_optional_fraction(raw, "tail_fraction"),
            normalized_temperature=_optional_fraction(
                raw,
                "normalized_temperature",
            ),
            active_fraction=_optional_fraction(raw, "active_fraction"),
        )
        if objective is LocalityObjective.CVAR and method.tail_fraction is None:
            raise ValueError(f"CVaR method lacks tail_fraction: {method_id}")
        if objective is LocalityObjective.SMOOTH_MAX and method.normalized_temperature is None:
            raise ValueError(f"smoothmax method lacks temperature: {method_id}")
        if objective is LocalityObjective.SPARSE_ROUTE and (
            method.tail_fraction is None or method.active_fraction is None
        ):
            raise ValueError(f"sparse method lacks risk parameters: {method_id}")
        if objective is LocalityObjective.PROTECTED_DIAGNOSTIC and (
            method.selection_eligible or method.baseline
        ):
            raise ValueError("Protected diagnostic cannot be baseline or selectable")
        methods.append(method)
    baselines = [method for method in methods if method.baseline]
    if len(baselines) != 1 or baselines[0].objective is not LocalityObjective.MEAN:
        raise ValueError("E22 requires exactly one mean-retention baseline")
    if not any(method.objective is LocalityObjective.PROTECTED_DIAGNOSTIC for method in methods):
        raise ValueError("E22 method grid requires a protected diagnostic")
    required = {
        LocalityObjective.MEAN,
        LocalityObjective.CVAR,
        LocalityObjective.SMOOTH_MAX,
        LocalityObjective.SPARSE_ROUTE,
        LocalityObjective.PROTECTED_DIAGNOSTIC,
    }
    if {method.objective for method in methods} != required:
        raise ValueError("E22 method grid does not cover every registered objective")
    return methods


def method_by_id(
    methods: Iterable[LocalityMethod],
    method_id: str,
) -> LocalityMethod:
    matches = [method for method in methods if method.method_id == method_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one E22 method {method_id!r}")
    return matches[0]
