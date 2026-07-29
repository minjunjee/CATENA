from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from catena.core.schema import Operation
from catena.data.semantic_transactions_v61 import (
    MINIMUM_NUMERIC_NAMESPACE,
    SafeSemanticRecord,
    SemanticExample,
    SemanticMemorySpec,
    SemanticNamespaceRegistry,
    build_semantic_example,
    derive_operation,
)


class R1EvidenceTime(StrEnum):
    ACTIVE = "active"
    EXPIRED = "expired"
    FUTURE = "future"


class R1VersionRelation(StrEnum):
    POSITIVE = "positive"
    NONPOSITIVE = "nonpositive"


class R1ScopeRelation(StrEnum):
    SAME = "same"
    DIFFERENT = "different"


@dataclass(frozen=True, slots=True)
class R1WriteStratum:
    evidence_time: R1EvidenceTime
    version_relation: R1VersionRelation
    scope_relation: R1ScopeRelation

    @property
    def key(self) -> str:
        return (
            f"{self.evidence_time.value}__{self.version_relation.value}"
            f"__{self.scope_relation.value}"
        )


R1_WRITE_TRUE_STRATUM = R1WriteStratum(
    evidence_time=R1EvidenceTime.ACTIVE,
    version_relation=R1VersionRelation.POSITIVE,
    scope_relation=R1ScopeRelation.SAME,
)
R1_WRITE_FALSE_STRATA = tuple(
    R1WriteStratum(
        evidence_time=evidence_time,
        version_relation=version_relation,
        scope_relation=scope_relation,
    )
    for evidence_time in R1EvidenceTime
    for version_relation in R1VersionRelation
    for scope_relation in R1ScopeRelation
    if (
        evidence_time,
        version_relation,
        scope_relation,
    )
    != (
        R1EvidenceTime.ACTIVE,
        R1VersionRelation.POSITIVE,
        R1ScopeRelation.SAME,
    )
)
R1_WRITE_FALSE_STRATUM_COUNT = 11

if len(R1_WRITE_FALSE_STRATA) != R1_WRITE_FALSE_STRATUM_COUNT:
    raise AssertionError("The R1 write-false factorial must contain eleven strata.")

_R1_ALLOWED_OPERATIONS = frozenset(
    {Operation.PRESERVE, Operation.ADD, Operation.INVALIDATE}
)
_ACTIVE_START_LAGS = (0, 1, 2, 4, 8)
_ACTIVE_END_LEADS = (0, 1, 2, 4, 8)
_INACTIVE_GAPS = (1, 2, 4, 8, 16)
_INACTIVE_WIDTHS = (2, 4, 8, 16)
_POSITIVE_VERSION_MARGINS = (1, 2, 3, 4)
_NONPOSITIVE_VERSION_MARGINS = (0, -1, -2, -3)
_ERASE_GAPS = (1, 2, 4, 8, 16)
_LIVE_PRIOR_LEADS = (0, 1, 2, 4, 8, 16)
_R1_OBSERVATION_BASE_DAY = 4096


def _opaque_token(kind: str, *parts: object) -> str:
    payload = "\0".join([kind, *(str(part) for part in parts)])
    digest = hashlib.sha256(payload.encode()).hexdigest()[:20]
    return f"{kind}_{digest}"


def _design_value(
    values: tuple[int, ...],
    *,
    replicate_index: int,
    domain_index: int,
    template_index: int,
    channel_offset: int,
) -> int:
    """Choose a frozen margin without consulting the numeric namespace."""

    design_index = (
        replicate_index * 7
        + domain_index * 5
        + template_index * 3
        + channel_offset
    )
    return values[design_index % len(values)]


def classify_r1_write_stratum(record: SafeSemanticRecord) -> R1WriteStratum:
    """Audit a generated raw record; this function is never a model input."""

    observation = record.observation_day
    if record.evidence_valid_from_day <= observation <= record.evidence_valid_to_day:
        evidence_time = R1EvidenceTime.ACTIVE
    elif record.evidence_valid_to_day < observation:
        evidence_time = R1EvidenceTime.EXPIRED
    elif record.evidence_valid_from_day > observation:
        evidence_time = R1EvidenceTime.FUTURE
    else:
        raise ValueError("Evidence interval has no valid R1 temporal stratum.")

    version_relation = (
        R1VersionRelation.POSITIVE
        if record.evidence_version > record.prior_version
        else R1VersionRelation.NONPOSITIVE
    )
    prefix = "relation_at::"
    if not record.current_relation.startswith(prefix):
        raise ValueError("R1 record has a malformed current relation.")
    current_scope = record.current_relation.removeprefix(prefix)
    scope_relation = (
        R1ScopeRelation.SAME
        if record.scope == current_scope
        else R1ScopeRelation.DIFFERENT
    )
    return R1WriteStratum(
        evidence_time=evidence_time,
        version_relation=version_relation,
        scope_relation=scope_relation,
    )


def build_r1_safe_record(
    *,
    operation: Operation,
    numeric_seed: int,
    checkpoint_seed: int,
    domain: str,
    template_surface: str,
    cell_index: int,
    domain_index: int,
    template_index: int,
) -> SafeSemanticRecord:
    """Build one R1 record from explicit factorial strata and margin cycles."""

    if operation not in _R1_ALLOWED_OPERATIONS:
        raise ValueError("E05a-R1 permits only PRESERVE, ADD, and INVALIDATE.")
    if numeric_seed < MINIMUM_NUMERIC_NAMESPACE:
        raise ValueError("numeric_seed lies below the frozen semantic namespace.")
    if cell_index < 0 or domain_index < 0 or template_index < 0:
        raise ValueError("R1 design indices must be nonnegative.")
    if not domain or not template_surface:
        raise ValueError("R1 domain and template must be nonempty.")

    if operation is Operation.ADD:
        stratum = R1_WRITE_TRUE_STRATUM
        replicate_index = cell_index
    else:
        stratum = R1_WRITE_FALSE_STRATA[
            cell_index % R1_WRITE_FALSE_STRATUM_COUNT
        ]
        replicate_index = cell_index // R1_WRITE_FALSE_STRATUM_COUNT

    observation_day = _R1_OBSERVATION_BASE_DAY + cell_index
    prior_version = 8 + (cell_index % 5)
    prior_valid_from_day = observation_day - 64
    if operation is Operation.INVALIDATE:
        prior_valid_to_day = observation_day - _design_value(
            _ERASE_GAPS,
            replicate_index=replicate_index,
            domain_index=domain_index,
            template_index=template_index,
            channel_offset=0,
        )
    else:
        prior_valid_to_day = observation_day + _design_value(
            _LIVE_PRIOR_LEADS,
            replicate_index=replicate_index,
            domain_index=domain_index,
            template_index=template_index,
            channel_offset=0,
        )

    if stratum.evidence_time is R1EvidenceTime.ACTIVE:
        evidence_valid_from_day = observation_day - _design_value(
            _ACTIVE_START_LAGS,
            replicate_index=replicate_index,
            domain_index=domain_index,
            template_index=template_index,
            channel_offset=1,
        )
        evidence_valid_to_day = observation_day + _design_value(
            _ACTIVE_END_LEADS,
            replicate_index=replicate_index,
            domain_index=domain_index,
            template_index=template_index,
            channel_offset=2,
        )
    elif stratum.evidence_time is R1EvidenceTime.EXPIRED:
        gap = _design_value(
            _INACTIVE_GAPS,
            replicate_index=replicate_index,
            domain_index=domain_index,
            template_index=template_index,
            channel_offset=3,
        )
        width = _design_value(
            _INACTIVE_WIDTHS,
            replicate_index=replicate_index,
            domain_index=domain_index,
            template_index=template_index,
            channel_offset=4,
        )
        evidence_valid_to_day = observation_day - gap
        evidence_valid_from_day = evidence_valid_to_day - width
    else:
        gap = _design_value(
            _INACTIVE_GAPS,
            replicate_index=replicate_index,
            domain_index=domain_index,
            template_index=template_index,
            channel_offset=5,
        )
        width = _design_value(
            _INACTIVE_WIDTHS,
            replicate_index=replicate_index,
            domain_index=domain_index,
            template_index=template_index,
            channel_offset=6,
        )
        evidence_valid_from_day = observation_day + gap
        evidence_valid_to_day = evidence_valid_from_day + width

    version_margins = (
        _POSITIVE_VERSION_MARGINS
        if stratum.version_relation is R1VersionRelation.POSITIVE
        else _NONPOSITIVE_VERSION_MARGINS
    )
    evidence_version = prior_version + _design_value(
        version_margins,
        replicate_index=replicate_index,
        domain_index=domain_index,
        template_index=template_index,
        channel_offset=7,
    )
    if evidence_version < 0:
        raise AssertionError("R1 evidence version became negative.")

    current_scope = _opaque_token(
        "context",
        numeric_seed,
        checkpoint_seed,
        domain,
        template_surface,
    )
    scope = (
        current_scope
        if stratum.scope_relation is R1ScopeRelation.SAME
        else _opaque_token(
            "context_other",
            numeric_seed,
            checkpoint_seed,
            domain,
            template_surface,
        )
    )
    record = SafeSemanticRecord(
        entity_description=_opaque_token(
            "entity",
            numeric_seed,
            checkpoint_seed,
        ),
        domain=domain,
        current_relation=f"relation_at::{current_scope}",
        incoming_evidence=_opaque_token(
            "statement",
            numeric_seed,
            checkpoint_seed,
        ),
        prior_version=prior_version,
        evidence_version=evidence_version,
        observation_day=observation_day,
        evidence_timestamp_day=observation_day
        - _design_value(
            (1, 2, 4),
            replicate_index=replicate_index,
            domain_index=domain_index,
            template_index=template_index,
            channel_offset=8,
        ),
        prior_valid_from_day=prior_valid_from_day,
        prior_valid_to_day=prior_valid_to_day,
        evidence_valid_from_day=evidence_valid_from_day,
        evidence_valid_to_day=evidence_valid_to_day,
        scope=scope,
        source=_opaque_token("source", numeric_seed, checkpoint_seed),
        provenance=_opaque_token("trace", numeric_seed, checkpoint_seed),
        incoming_value_token=_opaque_token(
            "value",
            numeric_seed,
            checkpoint_seed,
        ),
        template_surface=template_surface,
    )
    if classify_r1_write_stratum(record) != stratum:
        raise AssertionError("Generated R1 record differs from its assigned stratum.")
    if derive_operation(record) is not operation:
        raise AssertionError("Generated R1 raw predicates induce the wrong operation.")
    return record


def build_balanced_semantic_examples_r1(
    *,
    namespace_registry: SemanticNamespaceRegistry,
    namespace_name: str,
    checkpoint_seed: int,
    seed_slot: int,
    operations: Sequence[Operation],
    domains: Sequence[str],
    templates: Sequence[str],
    count_per_cell: int,
    memory_spec: SemanticMemorySpec,
) -> list[SemanticExample]:
    """Build balanced E05a-R1 cells without namespace-derived margins."""

    if count_per_cell <= 0:
        raise ValueError("count_per_cell must be positive.")
    if not operations or not domains or not templates:
        raise ValueError("R1 dataset axes must be nonempty.")
    if len(set(operations)) != len(operations):
        raise ValueError("R1 operations must be unique.")
    if len(set(domains)) != len(domains) or len(set(templates)) != len(templates):
        raise ValueError("R1 domains and templates must be unique.")
    unexpected = set(operations) - _R1_ALLOWED_OPERATIONS
    if unexpected:
        raise ValueError(f"E05a-R1 received unsupported operations: {unexpected}.")
    if (
        any(
            operation in {Operation.PRESERVE, Operation.INVALIDATE}
            for operation in operations
        )
        and count_per_cell % R1_WRITE_FALSE_STRATUM_COUNT
    ):
        raise ValueError(
            "PRESERVE/INVALIDATE count_per_cell must be divisible by 11."
        )

    examples: list[SemanticExample] = []
    numeric_index = 0
    for operation in operations:
        for domain_index, domain in enumerate(domains):
            for template_index, template_surface in enumerate(templates):
                for cell_index in range(count_per_cell):
                    numeric_seed = namespace_registry.numeric_seed(
                        namespace_name,
                        seed_slot=seed_slot,
                        index=numeric_index,
                    )
                    record = build_r1_safe_record(
                        operation=operation,
                        numeric_seed=numeric_seed,
                        checkpoint_seed=checkpoint_seed,
                        domain=domain,
                        template_surface=template_surface,
                        cell_index=cell_index,
                        domain_index=domain_index,
                        template_index=template_index,
                    )
                    example = build_semantic_example(
                        safe_record=record,
                        namespace_name=namespace_name,
                        numeric_seed=numeric_seed,
                        checkpoint_seed=checkpoint_seed,
                        memory_spec=memory_spec,
                    )
                    if example.operation is not operation:
                        raise AssertionError("R1 example operation changed during assembly.")
                    examples.append(example)
                    numeric_index += 1

    example_ids = [example.example_id for example in examples]
    episode_ids = [example.private_episode.episode_id for example in examples]
    if len(example_ids) != len(set(example_ids)):
        raise AssertionError("R1 dataset contains duplicate example identifiers.")
    if len(episode_ids) != len(set(episode_ids)):
        raise AssertionError("R1 dataset contains duplicate episode identifiers.")
    return examples
