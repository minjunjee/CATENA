from __future__ import annotations

from catena.core.schema import Operation
from catena.data.semantic_controls_r1 import build_control_pairing_registry_r1
from catena.data.semantic_transactions_r1 import (
    build_balanced_semantic_examples_r1,
)
from catena.data.semantic_transactions_v61 import (
    SemanticMemorySpec,
    SemanticNamespaceRegistry,
    derive_operation,
)


def _examples():
    registry = SemanticNamespaceRegistry(
        integer_root=6_000_000_000_000,
        split_stride=100_000_000,
        seed_stride=100_000,
        split_offsets=(("r1_validation", 2),),
        dry_run=True,
        prior_numeric_seed_max=5_002_700_000_000,
    )
    return build_balanced_semantic_examples_r1(
        namespace_registry=registry,
        namespace_name="r1_validation",
        checkpoint_seed=9919,
        seed_slot=0,
        operations=(Operation.PRESERVE, Operation.ADD, Operation.INVALIDATE),
        domains=("api",),
        templates=("record",),
        count_per_cell=11,
        memory_spec=SemanticMemorySpec(
            num_associations=4,
            key_dim=8,
            value_dim=8,
        ),
    )


def test_r1_pairing_is_total_operation_changing_and_outcome_blind() -> None:
    examples = _examples()
    registry = build_control_pairing_registry_r1(examples)

    assert len(registry.pairings) == len(examples)
    assert registry.mappings_use_outcomes is False
    by_id = {example.example_id: example for example in examples}
    for pairing in registry.pairings:
        recipient = by_id[pairing.example_id]
        target = derive_operation(pairing.wrong_semantics_record)
        assert target is not recipient.operation
        assert derive_operation(pairing.shuffled_record) is target
        assert pairing.mappings_use_outcomes is False
        assert pairing.maximum_wrong_address_norm_mismatch <= 1e-6
        predicate_donors = {
            donor_id
            for field_name, donor_id in pairing.shuffled_field_donor_ids
            if field_name not in {"source", "provenance"}
        }
        assert len(predicate_donors) >= 2


def test_r1_pairing_is_deterministic() -> None:
    examples = _examples()
    first = build_control_pairing_registry_r1(examples)
    second = build_control_pairing_registry_r1(list(reversed(examples)))

    assert first.sha256 == second.sha256
    assert first.to_rows() == second.to_rows()
