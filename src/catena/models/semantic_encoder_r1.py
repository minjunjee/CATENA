from __future__ import annotations

from typing import Protocol

import torch

R1_CURRENT_RELATION_PREFIX = "relation_at::"
R1_DAY_SCALE = 32.0
R1_VERSION_SCALE = 4.0
R1_FEATURE_NAMES = (
    "prior_valid_to_minus_observation_over_32",
    "evidence_valid_from_minus_observation_over_32",
    "evidence_valid_to_minus_observation_over_32",
    "evidence_version_minus_prior_version_over_4",
    "evidence_scope_equals_current_scope",
    "evidence_scope_differs_from_current_scope",
)
R1_FEATURE_DIM = len(R1_FEATURE_NAMES)


class RelationalSemanticRecord(Protocol):
    """The complete record surface read by the R1 gate encoder."""

    current_relation: str
    prior_version: int
    evidence_version: int
    observation_day: int
    prior_valid_to_day: int
    evidence_valid_from_day: int
    evidence_valid_to_day: int
    scope: str


def _integer_field(record: RelationalSemanticRecord, name: str) -> int:
    value = getattr(record, name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    return value


def _scope_pair(record: RelationalSemanticRecord) -> tuple[float, float]:
    current_relation = record.current_relation
    if not isinstance(current_relation, str) or not current_relation.startswith(
        R1_CURRENT_RELATION_PREFIX
    ):
        raise ValueError(
            "current_relation must start with the frozen relation_at:: prefix."
        )
    current_scope = current_relation.removeprefix(R1_CURRENT_RELATION_PREFIX)
    if not current_scope:
        raise ValueError("current_relation contains an empty current scope.")
    if not isinstance(record.scope, str) or not record.scope:
        raise ValueError("scope must be a nonempty string.")
    same = float(record.scope == current_scope)
    return same, 1.0 - same


class RelationalSemanticEncoderR1:
    """Fixed raw-relation encoder for the prospective E05a-R1 design.

    The encoder has no fitted state and accepts neither a memory read nor an
    address. Numeric relations remain continuous; no validity, version, erase,
    write, or operation decision is made here.
    """

    FEATURE_NAMES = R1_FEATURE_NAMES

    @property
    def semantic_dim(self) -> int:
        return R1_FEATURE_DIM

    @property
    def input_dim(self) -> int:
        return R1_FEATURE_DIM

    def encode(
        self,
        record: RelationalSemanticRecord,
        *,
        mask_semantics: bool = False,
    ) -> torch.Tensor:
        if not isinstance(mask_semantics, bool):
            raise TypeError("mask_semantics must be boolean.")
        if mask_semantics:
            return torch.zeros(R1_FEATURE_DIM, dtype=torch.float32)

        observation = _integer_field(record, "observation_day")
        prior_to = _integer_field(record, "prior_valid_to_day")
        evidence_from = _integer_field(record, "evidence_valid_from_day")
        evidence_to = _integer_field(record, "evidence_valid_to_day")
        prior_version = _integer_field(record, "prior_version")
        evidence_version = _integer_field(record, "evidence_version")
        scope_same, scope_different = _scope_pair(record)

        result = torch.tensor(
            (
                (prior_to - observation) / R1_DAY_SCALE,
                (evidence_from - observation) / R1_DAY_SCALE,
                (evidence_to - observation) / R1_DAY_SCALE,
                (evidence_version - prior_version) / R1_VERSION_SCALE,
                scope_same,
                scope_different,
            ),
            dtype=torch.float32,
        )
        if result.shape != (R1_FEATURE_DIM,):
            raise AssertionError("The R1 relational feature layout changed.")
        if not bool(torch.isfinite(result).all().item()):
            raise FloatingPointError("R1 relational encoding is non-finite.")
        return result
