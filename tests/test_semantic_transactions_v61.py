from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path

import pytest
import torch

from catena.core.config import load_config
from catena.core.schema import Operation
from catena.data.semantic_controls_v61 import (
    SemanticControl,
    apply_visible_update,
    build_visible_update_context,
)
from catena.data.semantic_transactions_v61 import (
    ALLOWED_SAFE_FIELDS,
    BANNED_SURFACE_CUES,
    MINIMUM_NUMERIC_NAMESPACE,
    SafeSemanticRecord,
    SemanticExample,
    SemanticMemorySpec,
    SemanticNamespaceRegistry,
    assert_disjoint_semantic_vocabularies,
    build_balanced_semantic_examples,
    build_safe_record_for_operation,
    build_semantic_example,
    derive_operation,
    derive_raw_demand,
    find_banned_surface_cues,
    safe_record_field_names,
    semantic_vocabularies,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _protocol_config() -> dict[str, object]:
    return load_config(REPO_ROOT / "configs/e05a_semantic_protocol_lock.yaml")


def _memory_spec() -> SemanticMemorySpec:
    return SemanticMemorySpec(
        num_associations=16,
        key_dim=32,
        value_dim=32,
        dtype=torch.float32,
    )


def test_safe_record_fields_exactly_match_frozen_allow_list() -> None:
    config = _protocol_config()
    schema = config["semantic_schema"]
    assert isinstance(schema, dict)

    expected = tuple(schema["allowed_structured_fields"])
    assert expected == ALLOWED_SAFE_FIELDS
    assert safe_record_field_names() == expected
    assert tuple(field.name for field in fields(SafeSemanticRecord)) == expected
    assert not set(schema["forbidden_model_fields"]) & set(expected)


@pytest.mark.parametrize("operation", list(Operation))
@pytest.mark.parametrize("variant", range(6))
def test_raw_predicates_derive_every_operation_without_banned_cues(
    operation: Operation,
    variant: int,
) -> None:
    numeric_seed = MINIMUM_NUMERIC_NAMESPACE + 1000 + variant
    record = build_safe_record_for_operation(
        operation=operation,
        numeric_seed=numeric_seed,
        domain="api",
        template_surface="record",
    )

    demand = derive_raw_demand(record)
    assert demand.operation is operation
    assert derive_operation(record) is operation
    assert (float(demand.erase), float(demand.write)) == operation.demand
    assert record.incoming_evidence
    assert record.incoming_value_token
    assert find_banned_surface_cues(record) == ()
    all_text = " ".join(
        str(getattr(record, field_name)).lower() for field_name in ALLOWED_SAFE_FIELDS
    )
    assert all(f" {cue} " not in f" {all_text} " for cue in BANNED_SURFACE_CUES)


def test_safe_record_rejects_a_banned_surface_cue() -> None:
    record = build_safe_record_for_operation(
        operation=Operation.ADD,
        numeric_seed=MINIMUM_NUMERIC_NAMESPACE + 20,
        domain="api",
        template_surface="record",
    )
    with pytest.raises(ValueError, match="banned surface cues"):
        replace(record, incoming_evidence="replace")


def test_orthonormal_memory_is_deterministic_and_oracle_update_is_exact() -> None:
    numeric_seed = MINIMUM_NUMERIC_NAMESPACE + 12345
    record = build_safe_record_for_operation(
        operation=Operation.SUPERSEDE,
        numeric_seed=numeric_seed,
        domain="api",
        template_surface="ledger",
    )
    first = build_semantic_example(
        safe_record=record,
        namespace_name="e05b_primary",
        numeric_seed=numeric_seed,
        checkpoint_seed=11,
        memory_spec=_memory_spec(),
    )
    second = build_semantic_example(
        safe_record=record,
        namespace_name="e05b_primary",
        numeric_seed=numeric_seed,
        checkpoint_seed=11,
        memory_spec=_memory_spec(),
    )

    identity = first.keys @ first.keys.transpose(0, 1)
    assert torch.allclose(identity, torch.eye(16), atol=1e-5, rtol=0.0)
    assert first.example_id == second.example_id
    for name in ("keys", "state", "target_state", "new_value"):
        assert torch.equal(getattr(first, name), getattr(second, name))

    context = build_visible_update_context(first, SemanticControl.FULL)
    demand = derive_raw_demand(first.safe_record)
    output = apply_visible_update(
        context,
        erase=float(demand.erase),
        write=float(demand.write),
    )
    assert torch.equal(output, first.target_state)
    assert torch.allclose(
        context.address_resolved_state_read,
        first.private_episode.values[first.affected_index],
        atol=2e-6,
        rtol=0.0,
    )


def test_namespace_modes_are_disjoint_and_dry_never_opens_e05b() -> None:
    config = _protocol_config()
    namespace = config["namespace"]
    assert isinstance(namespace, dict)
    dry = SemanticNamespaceRegistry.from_config(namespace, dry_run=True)
    main = SemanticNamespaceRegistry.from_config(namespace, dry_run=False)

    assert set(dry.names) == {"pilot_train", "pilot_validation"}
    assert all(not name.startswith("e05b_") for name in dry.names)
    with pytest.raises(KeyError):
        dry.numeric_seed("e05b_train", seed_slot=0, index=0)

    dry_values = {
        dry.numeric_seed(name, seed_slot=0, index=index)
        for name in dry.names
        for index in range(3)
    }
    main_values = {
        main.numeric_seed(name, seed_slot=0, index=index)
        for name in main.names
        for index in range(3)
    }
    assert dry_values.isdisjoint(main_values)
    assert min(dry_values | main_values) >= MINIMUM_NUMERIC_NAMESPACE
    assert min(dry_values | main_values) > int(
        namespace["forbid_overlap_with_prior_numeric_seed_max"]
    )


def test_balanced_splits_have_disjoint_vocab_and_unique_identifiers() -> None:
    config = _protocol_config()
    namespace = config["namespace"]
    assert isinstance(namespace, dict)
    registry = SemanticNamespaceRegistry.from_config(namespace, dry_run=False)
    def build(namespace_name: str) -> list[SemanticExample]:
        return build_balanced_semantic_examples(
            namespace_registry=registry,
            namespace_name=namespace_name,
            checkpoint_seed=101,
            seed_slot=0,
            operations=[Operation.PRESERVE, Operation.ADD, Operation.INVALIDATE],
            domains=["api", "access"],
            templates=["record", "ledger"],
            count_per_cell=2,
            memory_spec=_memory_spec(),
        )

    train = build("pilot_train")
    validation = build("pilot_validation")

    assert len(train) == len(validation) == 24
    assert {example.operation for example in train} == {
        Operation.PRESERVE,
        Operation.ADD,
        Operation.INVALIDATE,
    }
    assert len({example.example_id for example in train + validation}) == 48
    assert len(
        {example.private_episode.episode_id for example in train + validation}
    ) == 48
    assert_disjoint_semantic_vocabularies(
        {"pilot_train": train, "pilot_validation": validation}
    )
    train_vocab = semantic_vocabularies(train)
    validation_vocab = semantic_vocabularies(validation)
    for vocabulary_name in ("entity", "old_value", "new_value"):
        assert train_vocab[vocabulary_name].isdisjoint(
            validation_vocab[vocabulary_name]
        )
