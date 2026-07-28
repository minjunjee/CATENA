from __future__ import annotations

import inspect
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from catena.core.schema import Operation
from catena.data.semantic_transactions_r1 import (
    R1_WRITE_FALSE_STRATA,
    R1_WRITE_TRUE_STRATUM,
    build_balanced_semantic_examples_r1,
    build_r1_safe_record,
    classify_r1_write_stratum,
)
from catena.data.semantic_transactions_v61 import (
    MINIMUM_NUMERIC_NAMESPACE,
    SemanticMemorySpec,
    SemanticNamespaceRegistry,
    assert_disjoint_semantic_vocabularies,
    build_semantic_example,
    derive_operation,
)
from catena.models.semantic_encoder_r1 import (
    R1_FEATURE_DIM,
    R1_FEATURE_NAMES,
    RelationalSemanticEncoderR1,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _memory_spec() -> SemanticMemorySpec:
    return SemanticMemorySpec(
        num_associations=4,
        key_dim=8,
        value_dim=8,
        dtype=torch.float32,
    )


def _registry() -> SemanticNamespaceRegistry:
    return SemanticNamespaceRegistry(
        integer_root=6_000_000_000_000,
        split_stride=1_000_000,
        seed_stride=10_000,
        split_offsets=(("r1_train", 1), ("r1_validation", 2)),
        dry_run=False,
        prior_numeric_seed_max=100_000_000,
    )


def _record(
    operation: Operation = Operation.ADD,
    *,
    numeric_seed: int = MINIMUM_NUMERIC_NAMESPACE + 123,
    cell_index: int = 3,
):
    return build_r1_safe_record(
        operation=operation,
        numeric_seed=numeric_seed,
        checkpoint_seed=501,
        domain="api",
        template_surface="record",
        cell_index=cell_index,
        domain_index=0,
        template_index=0,
    )


def test_r1_encoder_has_exact_six_raw_relational_features_and_no_state_argument() -> None:
    encoder = RelationalSemanticEncoderR1()
    record = _record()
    encoded = encoder.encode(record)
    current_scope = record.current_relation.removeprefix("relation_at::")

    assert R1_FEATURE_DIM == 6
    assert encoder.semantic_dim == encoder.input_dim == 6
    assert encoder.FEATURE_NAMES == R1_FEATURE_NAMES
    assert encoded.dtype == torch.float32
    assert encoded.shape == (6,)
    assert torch.equal(
        encoded,
        torch.tensor(
            [
                (record.prior_valid_to_day - record.observation_day) / 32.0,
                (
                    record.evidence_valid_from_day - record.observation_day
                )
                / 32.0,
                (record.evidence_valid_to_day - record.observation_day) / 32.0,
                (record.evidence_version - record.prior_version) / 4.0,
                float(record.scope == current_scope),
                float(record.scope != current_scope),
            ],
            dtype=torch.float32,
        ),
    )
    assert torch.equal(
        encoder.encode(record, mask_semantics=True),
        torch.zeros(6, dtype=torch.float32),
    )
    assert tuple(inspect.signature(encoder.encode).parameters) == (
        "record",
        "mask_semantics",
    )


def test_r1_encoder_is_day_version_and_opaque_token_translation_invariant() -> None:
    encoder = RelationalSemanticEncoderR1()
    record = _record()
    day_shift = 777
    version_shift = 19
    shifted = replace(
        record,
        prior_version=record.prior_version + version_shift,
        evidence_version=record.evidence_version + version_shift,
        observation_day=record.observation_day + day_shift,
        evidence_timestamp_day=record.evidence_timestamp_day + day_shift,
        prior_valid_from_day=record.prior_valid_from_day + day_shift,
        prior_valid_to_day=record.prior_valid_to_day + day_shift,
        evidence_valid_from_day=record.evidence_valid_from_day + day_shift,
        evidence_valid_to_day=record.evidence_valid_to_day + day_shift,
        current_relation="relation_at::renamed_current_scope",
        scope="renamed_current_scope",
    )
    nuisance_changed = replace(
        shifted,
        entity_description="entity_fresh",
        domain="access",
        incoming_evidence="filing_fresh",
        source="source_fresh",
        provenance="trace_fresh",
        incoming_value_token="value_fresh",
        template_surface="ledger",
    )

    assert torch.equal(encoder.encode(record), encoder.encode(shifted))
    assert torch.equal(encoder.encode(record), encoder.encode(nuisance_changed))


def test_r1_encoder_state_address_and_new_value_cannot_change_gate_features() -> None:
    encoder = RelationalSemanticEncoderR1()
    record = _record()
    first = build_semantic_example(
        safe_record=record,
        namespace_name="r1_train",
        numeric_seed=MINIMUM_NUMERIC_NAMESPACE + 700,
        checkpoint_seed=501,
        memory_spec=_memory_spec(),
    )
    second = build_semantic_example(
        safe_record=record,
        namespace_name="r1_validation",
        numeric_seed=MINIMUM_NUMERIC_NAMESPACE + 701,
        checkpoint_seed=501,
        memory_spec=_memory_spec(),
    )

    assert not torch.equal(first.state, second.state)
    assert not torch.equal(
        first.keys[first.affected_index],
        second.keys[second.affected_index],
    )
    assert not torch.equal(first.new_value, second.new_value)
    assert torch.equal(
        encoder.encode(first.safe_record),
        encoder.encode(second.safe_record),
    )


@pytest.mark.parametrize(
    ("field_name", "coordinate", "expected_step"),
    [
        ("prior_valid_to_day", 0, 1.0 / 32.0),
        ("evidence_valid_from_day", 1, 1.0 / 32.0),
        ("evidence_valid_to_day", 2, 1.0 / 32.0),
        ("evidence_version", 3, 1.0 / 4.0),
    ],
)
def test_r1_numeric_features_are_linear_raw_differences(
    field_name: str,
    coordinate: int,
    expected_step: float,
) -> None:
    encoder = RelationalSemanticEncoderR1()
    record = _record()
    incremented = replace(record, **{field_name: getattr(record, field_name) + 1})
    delta = encoder.encode(incremented) - encoder.encode(record)
    expected = torch.zeros(6, dtype=torch.float32)
    expected[coordinate] = expected_step
    assert torch.equal(delta, expected)


def test_r1_scope_change_touches_only_the_categorical_slice() -> None:
    encoder = RelationalSemanticEncoderR1()
    same = _record()
    different = replace(same, scope="opaque_distinct_scope")
    same_features = encoder.encode(same)
    different_features = encoder.encode(different)

    assert torch.equal(same_features[:4], different_features[:4])
    assert torch.equal(same_features[4:], torch.tensor([1.0, 0.0]))
    assert torch.equal(different_features[4:], torch.tensor([0.0, 1.0]))


def test_r1_encoder_rejects_malformed_or_empty_current_relation() -> None:
    encoder = RelationalSemanticEncoderR1()
    values = {
        "prior_version": 3,
        "evidence_version": 4,
        "observation_day": 100,
        "prior_valid_to_day": 99,
        "evidence_valid_from_day": 90,
        "evidence_valid_to_day": 110,
        "scope": "tenant",
    }
    with pytest.raises(ValueError, match="relation_at::"):
        encoder.encode(SimpleNamespace(current_relation="tenant", **values))
    with pytest.raises(ValueError, match="empty current scope"):
        encoder.encode(SimpleNamespace(current_relation="relation_at::", **values))


def test_r1_encoder_source_has_no_hash_or_oracle_demand_dependency() -> None:
    source = (
        REPO_ROOT / "src/catena/models/semantic_encoder_r1.py"
    ).read_text(encoding="utf-8")
    for forbidden in (
        "hashlib",
        "derive_raw_demand",
        "derive_operation",
        "RawDemand",
        "Operation",
        "state_read",
        "visible_address",
        "target_state",
    ):
        assert forbidden not in source


def test_r1_generator_balances_all_eleven_write_false_strata_per_cell() -> None:
    examples = build_balanced_semantic_examples_r1(
        namespace_registry=_registry(),
        namespace_name="r1_train",
        checkpoint_seed=501,
        seed_slot=0,
        operations=[Operation.PRESERVE, Operation.ADD, Operation.INVALIDATE],
        domains=["api", "access"],
        templates=["record", "ledger"],
        count_per_cell=22,
        memory_spec=_memory_spec(),
    )
    assert len(examples) == 3 * 2 * 2 * 22

    for operation in (Operation.PRESERVE, Operation.INVALIDATE):
        for domain in ("api", "access"):
            for template in ("record", "ledger"):
                cell = [
                    example
                    for example in examples
                    if example.operation is operation
                    and example.domain == domain
                    and example.template == template
                ]
                counts = Counter(
                    classify_r1_write_stratum(example.safe_record) for example in cell
                )
                assert set(counts) == set(R1_WRITE_FALSE_STRATA)
                assert set(counts.values()) == {2}

    add_strata = {
        classify_r1_write_stratum(example.safe_record)
        for example in examples
        if example.operation is Operation.ADD
    }
    assert add_strata == {R1_WRITE_TRUE_STRATUM}
    assert all(derive_operation(example.safe_record) is example.operation for example in examples)


def test_r1_generator_requires_divisibility_and_rejects_supersede() -> None:
    common = {
        "namespace_registry": _registry(),
        "namespace_name": "r1_train",
        "checkpoint_seed": 501,
        "seed_slot": 0,
        "domains": ["api"],
        "templates": ["record"],
        "memory_spec": _memory_spec(),
    }
    with pytest.raises(ValueError, match="divisible by 11"):
        build_balanced_semantic_examples_r1(
            operations=[Operation.PRESERVE],
            count_per_cell=12,
            **common,
        )
    with pytest.raises(ValueError, match="unsupported operations"):
        build_balanced_semantic_examples_r1(
            operations=[Operation.SUPERSEDE],
            count_per_cell=11,
            **common,
        )


def test_r1_fresh_namespaces_have_disjoint_vocab_but_identical_relation_features() -> None:
    encoder = RelationalSemanticEncoderR1()
    common = {
        "namespace_registry": _registry(),
        "checkpoint_seed": 501,
        "seed_slot": 0,
        "operations": [Operation.PRESERVE, Operation.ADD, Operation.INVALIDATE],
        "domains": ["api"],
        "templates": ["record"],
        "count_per_cell": 11,
        "memory_spec": _memory_spec(),
    }
    train = build_balanced_semantic_examples_r1(
        namespace_name="r1_train",
        **common,
    )
    validation = build_balanced_semantic_examples_r1(
        namespace_name="r1_validation",
        **common,
    )
    assert_disjoint_semantic_vocabularies(
        {"r1_train": train, "r1_validation": validation}
    )
    assert not {
        example.example_id for example in train
    } & {example.example_id for example in validation}
    assert torch.equal(
        torch.stack([encoder.encode(example.safe_record) for example in train]),
        torch.stack(
            [encoder.encode(example.safe_record) for example in validation]
        ),
    )
