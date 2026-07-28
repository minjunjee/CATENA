from __future__ import annotations

import inspect
from dataclasses import fields, replace
from pathlib import Path

import torch

from catena.core.config import load_config
from catena.core.schema import Operation
from catena.data.semantic_controls_v61 import (
    SemanticControl,
    VisibleUpdateContext,
    apply_visible_update,
    build_control_pairing_registry,
    build_control_view,
    build_visible_candidates,
    build_visible_update_context,
    semantic_record_for_control,
)
from catena.data.semantic_transactions_v61 import (
    SafeSemanticRecord,
    SemanticExample,
    SemanticMemorySpec,
    SemanticNamespaceRegistry,
    build_balanced_semantic_examples,
    derive_operation,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _examples() -> list[SemanticExample]:
    config = load_config(REPO_ROOT / "configs/e05a_semantic_protocol_lock.yaml")
    namespace = config["namespace"]
    assert isinstance(namespace, dict)
    registry = SemanticNamespaceRegistry.from_config(namespace, dry_run=False)
    return build_balanced_semantic_examples(
        namespace_registry=registry,
        namespace_name="pilot_validation",
        checkpoint_seed=101,
        seed_slot=0,
        operations=[Operation.PRESERVE, Operation.ADD, Operation.INVALIDATE],
        domains=["api"],
        templates=["record"],
        count_per_cell=3,
        memory_spec=SemanticMemorySpec(),
    )


def test_control_pairing_is_deterministic_outcome_independent_and_norm_matched() -> None:
    examples = _examples()
    first = build_control_pairing_registry(examples, norm_tolerance=1e-6)
    second = build_control_pairing_registry(examples, norm_tolerance=1e-6)

    assert first.sha256 == second.sha256
    assert first.to_rows() == second.to_rows()
    assert first.mappings_use_outcomes is False
    assert len(first.pairings) == len(examples)

    for example in examples:
        pairing = first.for_example(example)
        assert pairing.mappings_use_outcomes is False
        assert pairing.wrong_address_index != example.affected_index
        assert pairing.maximum_wrong_address_norm_mismatch <= 1e-6
        assert pairing.wrong_semantics_donor_id != example.example_id
        assert derive_operation(pairing.wrong_semantics_record) is not example.operation
        assert derive_operation(pairing.shuffled_record) is not example.operation
        assert derive_operation(pairing.shuffled_record) is derive_operation(
            pairing.wrong_semantics_record
        )
        assert pairing.wrong_semantics_record.entity_description == (
            example.safe_record.entity_description
        )
        assert pairing.wrong_semantics_record.incoming_value_token == (
            example.safe_record.incoming_value_token
        )
        assert pairing.wrong_semantics_record.domain == example.domain
        assert pairing.wrong_semantics_record.template_surface == example.template
        assert all(
            donor_id != example.example_id
            for _, donor_id in pairing.shuffled_field_donor_ids
        )
        predicate_donor_ids = {
            donor_id
            for field_name, donor_id in pairing.shuffled_field_donor_ids
            if field_name not in {"source", "provenance"}
        }
        assert len(predicate_donor_ids) >= 2

    changed_targets = [
        replace(
            example,
            private_episode=replace(
                example.private_episode,
                target_state=example.target_state + 0.25,
            ),
        )
        for example in examples
    ]
    changed_registry = build_control_pairing_registry(
        changed_targets,
        norm_tolerance=1e-6,
    )
    assert changed_registry.to_rows() == first.to_rows()


def test_five_controls_have_the_frozen_information_access_semantics() -> None:
    examples = _examples()
    registry = build_control_pairing_registry(examples)
    example = next(item for item in examples if item.operation is Operation.ADD)
    pairing = registry.for_example(example)

    full = build_control_view(example, SemanticControl.FULL, pairing)
    transaction_only = build_control_view(
        example,
        SemanticControl.TRANSACTION_ONLY,
        pairing,
    )
    state_only = build_control_view(example, SemanticControl.STATE_ONLY, pairing)
    shuffled = build_control_view(
        example,
        SemanticControl.SHUFFLED_FIELDS,
        pairing,
    )
    wrong_address = build_control_view(
        example,
        SemanticControl.WRONG_ADDRESS,
        pairing,
    )
    wrong_semantics = build_control_view(
        example,
        SemanticControl.WRONG_SEMANTICS,
        pairing,
    )

    assert full.semantic_record == example.safe_record
    assert torch.equal(full.update_context.visible_state, example.state)
    assert torch.equal(full.update_context.incoming_value, example.new_value)

    assert transaction_only.semantic_record == example.safe_record
    assert torch.count_nonzero(transaction_only.update_context.visible_state) == 0
    transaction_erase, _ = build_visible_candidates(transaction_only.update_context)
    assert torch.count_nonzero(transaction_erase) == 0

    assert state_only.semantic_record is None
    assert state_only.semantic_fields_visible is False
    assert torch.equal(state_only.update_context.visible_state, example.state)
    _, state_write = build_visible_candidates(state_only.update_context)
    assert torch.count_nonzero(state_write) == 0

    assert shuffled.semantic_record == pairing.shuffled_record
    assert derive_operation(shuffled.semantic_record) is not example.operation
    assert wrong_semantics.semantic_record == pairing.wrong_semantics_record
    assert derive_operation(wrong_semantics.semantic_record) is not example.operation
    assert torch.equal(wrong_semantics.update_context.visible_state, example.state)
    assert torch.equal(wrong_semantics.update_context.incoming_value, example.new_value)
    assert torch.equal(
        wrong_semantics.update_context.visible_address,
        example.keys[example.affected_index],
    )

    assert wrong_address.semantic_record == example.safe_record
    assert torch.equal(wrong_address.update_context.visible_state, example.state)
    assert torch.equal(wrong_address.update_context.incoming_value, example.new_value)
    assert torch.equal(
        wrong_address.update_context.visible_address,
        example.keys[pairing.wrong_address_index],
    )
    assert not torch.equal(
        wrong_address.update_context.visible_address,
        full.update_context.visible_address,
    )
    full_candidates = build_visible_candidates(full.update_context)
    wrong_candidates = build_visible_candidates(wrong_address.update_context)
    for correct, wrong in zip(full_candidates, wrong_candidates, strict=True):
        assert abs(
            float(torch.linalg.vector_norm(correct).item())
            - float(torch.linalg.vector_norm(wrong).item())
        ) <= 1e-6


def test_public_update_context_and_apply_function_exclude_private_targets() -> None:
    assert tuple(field.name for field in fields(VisibleUpdateContext)) == (
        "visible_state",
        "visible_address",
        "incoming_value",
        "erase_candidate_scale",
        "write_candidate_scale",
    )
    signature = inspect.signature(apply_visible_update)
    assert tuple(signature.parameters) == ("context", "erase", "write")
    source = inspect.getsource(apply_visible_update)
    for forbidden in (
        "target_state",
        "old_value",
        "operation",
        "affected_index",
        "private_episode",
    ):
        assert forbidden not in source


def test_correct_visible_oracle_update_matches_private_target_for_all_operations() -> None:
    examples = _examples()
    for example in examples:
        erase, write = example.operation.demand
        context = build_visible_update_context(example, SemanticControl.FULL)
        output = apply_visible_update(context, erase, write)
        assert torch.equal(output, example.target_state)


def test_semantic_control_record_requires_pairing_only_when_registered() -> None:
    example = _examples()[0]
    assert (
        semantic_record_for_control(example, SemanticControl.FULL)
        == example.safe_record
    )
    assert semantic_record_for_control(example, SemanticControl.STATE_ONLY) is None
    for control in (
        SemanticControl.SHUFFLED_FIELDS,
        SemanticControl.WRONG_SEMANTICS,
    ):
        try:
            semantic_record_for_control(example, control)
        except ValueError as error:
            assert "requires a frozen pairing" in str(error)
        else:
            raise AssertionError(f"{control.value} unexpectedly accepted no pairing.")


def test_safe_semantic_record_does_not_contain_private_field_names() -> None:
    safe_names = {field.name for field in fields(SafeSemanticRecord)}
    assert safe_names.isdisjoint(
        {
            "operation",
            "operation_features",
            "erase",
            "write",
            "demand",
            "target",
            "target_state",
            "exact_mask",
            "affected_index",
            "erase_candidate",
            "write_candidate",
            "old_value",
            "old_value_token",
            "split",
            "namespace",
        }
    )
