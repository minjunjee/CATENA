from catena.core.schema import Operation
from catena.data.semantic_registry_v61 import (
    audit_id,
    render_naturalized_record,
    semantic_example_from_registry_row,
    semantic_example_registry_row,
    semantic_registry_row_from_design,
    validate_semantic_registry_rows,
)
from catena.data.semantic_transactions_v61 import (
    SemanticMemorySpec,
    SemanticNamespaceRegistry,
    build_balanced_semantic_examples,
)


def test_semantic_registry_round_trip_and_neutral_render():
    registry = SemanticNamespaceRegistry(
        integer_root=5_000_000_000_000,
        split_stride=100_000_000,
        seed_stride=100_000,
        split_offsets=(("primary", 1),),
        dry_run=False,
        prior_numeric_seed_max=100_000_000,
    )
    spec = SemanticMemorySpec(num_associations=4, key_dim=8, value_dim=8)
    example = build_balanced_semantic_examples(
        namespace_registry=registry,
        namespace_name="primary",
        checkpoint_seed=11,
        seed_slot=0,
        operations=(Operation.SUPERSEDE,),
        domains=("api",),
        templates=("record",),
        count_per_cell=1,
        memory_spec=spec,
    )[0]
    row = semantic_example_registry_row(example, split="primary", seed_slot=0)
    light_row = semantic_registry_row_from_design(
        namespace_name="primary",
        split="primary",
        numeric_seed=example.private_episode.numeric_seed,
        checkpoint_seed=11,
        seed_slot=0,
        operation=Operation.SUPERSEDE,
        domain="api",
        template_surface="record",
    )
    assert light_row == row
    rebuilt = semantic_example_from_registry_row(row, memory_spec=spec)
    assert rebuilt.example_id == example.example_id
    assert rebuilt.operation is Operation.SUPERSEDE
    text = render_naturalized_record(rebuilt.safe_record)
    assert "supersede" not in text.lower()
    assert audit_id("primary", rebuilt.example_id).startswith("audit_")
    validate_semantic_registry_rows(
        [row],
        expected_split="primary",
        expected_seeds=(11,),
        expected_rows_per_seed=1,
        expected_namespace_name="primary",
        expected_seed_slots={11: 0},
        namespace_registry=registry,
        memory_spec=spec,
        reconstruct=True,
    )


def test_semantic_registry_rejects_namespace_and_identifier_mutation():
    registry = SemanticNamespaceRegistry(
        integer_root=5_000_000_000_000,
        split_stride=100_000_000,
        seed_stride=100_000,
        split_offsets=(("primary", 1),),
        dry_run=False,
        prior_numeric_seed_max=100_000_000,
    )
    spec = SemanticMemorySpec(num_associations=4, key_dim=8, value_dim=8)
    example = build_balanced_semantic_examples(
        namespace_registry=registry,
        namespace_name="primary",
        checkpoint_seed=11,
        seed_slot=0,
        operations=(Operation.SUPERSEDE,),
        domains=("api",),
        templates=("record",),
        count_per_cell=1,
        memory_spec=spec,
    )[0]
    row = semantic_example_registry_row(example, split="primary", seed_slot=0)
    mutated = dict(row)
    mutated["transaction_id"] = "t_mutated"
    try:
        validate_semantic_registry_rows(
            [mutated],
            expected_split="primary",
            expected_seeds=(11,),
            expected_rows_per_seed=1,
            expected_namespace_name="primary",
            expected_seed_slots={11: 0},
            namespace_registry=registry,
            memory_spec=spec,
            reconstruct=False,
        )
    except ValueError as error:
        assert "private identifier" in str(error)
    else:
        raise AssertionError("Mutated registry identifier was accepted.")
