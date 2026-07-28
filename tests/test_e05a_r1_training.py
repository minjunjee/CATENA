from dataclasses import replace

import torch
from catena.models.semantic_encoder_r1 import RelationalSemanticEncoderR1

from catena.core.schema import Operation
from catena.data.semantic_controls_v61 import (
    SemanticControl,
    build_control_pairing_registry,
)
from catena.data.semantic_transactions_v61 import (
    SemanticExample,
    SemanticMemorySpec,
    SemanticNamespaceRegistry,
    build_balanced_semantic_examples,
)
from catena.models.semantic_controllers_v61 import (
    MatchedSemanticControllerV61,
    SemanticRoute,
)
from catena.training.semantic_probe_r1 import (
    evaluate_semantic_model_r1,
    tensorize_semantic_examples_r1,
    train_matched_semantic_pair_r1,
)
from catena.training.semantic_probe_v61 import (
    SemanticTrainingConfigV61,
    apply_batched_visible_update,
)


def _examples() -> list[SemanticExample]:
    registry = SemanticNamespaceRegistry(
        integer_root=6_000_000_000_000,
        split_stride=100_000_000,
        seed_stride=100_000,
        split_offsets=(("r1_test", 1),),
        dry_run=True,
        prior_numeric_seed_max=5_002_699_999_999,
    )
    return build_balanced_semantic_examples(
        namespace_registry=registry,
        namespace_name="r1_test",
        checkpoint_seed=1_103,
        seed_slot=0,
        operations=(Operation.PRESERVE, Operation.ADD, Operation.INVALIDATE),
        domains=("api",),
        templates=("record",),
        count_per_cell=2,
        memory_spec=SemanticMemorySpec(
            num_associations=4,
            key_dim=8,
            value_dim=8,
        ),
    )


def _private_tensor_variant(example: SemanticExample) -> SemanticExample:
    private = example.private_episode
    return replace(
        example,
        private_episode=replace(
            private,
            keys=-private.keys,
            state=private.state + 0.25,
            new_value=-private.new_value,
        ),
    )


def test_r1_features_ignore_state_address_and_incoming_value() -> None:
    example = _examples()[1]
    variant = _private_tensor_variant(example)
    encoder = RelationalSemanticEncoderR1()

    original = tensorize_semantic_examples_r1([example], encoder=encoder)
    changed = tensorize_semantic_examples_r1([variant], encoder=encoder)

    assert torch.equal(original.visible.features, changed.visible.features)
    assert not torch.equal(
        original.visible.visible_state,
        changed.visible.visible_state,
    )
    assert not torch.equal(
        original.visible.visible_address,
        changed.visible.visible_address,
    )
    assert not torch.equal(
        original.visible.incoming_value,
        changed.visible.incoming_value,
    )


def test_r1_wrong_address_changes_context_but_not_features() -> None:
    examples = _examples()
    pairings = build_control_pairing_registry(examples)
    encoder = RelationalSemanticEncoderR1()

    full = tensorize_semantic_examples_r1(
        examples,
        encoder=encoder,
        control=SemanticControl.FULL,
    )
    wrong = tensorize_semantic_examples_r1(
        examples,
        encoder=encoder,
        control=SemanticControl.WRONG_ADDRESS,
        pairing_registry=pairings,
    )

    assert torch.equal(full.visible.features, wrong.visible.features)
    assert not torch.equal(
        full.visible.visible_address,
        wrong.visible.visible_address,
    )


def test_r1_state_only_masks_semantics_and_transaction_only_keeps_them() -> None:
    examples = _examples()
    pairings = build_control_pairing_registry(examples)
    encoder = RelationalSemanticEncoderR1()

    full = tensorize_semantic_examples_r1(
        examples,
        encoder=encoder,
        control=SemanticControl.FULL,
    )
    state_only = tensorize_semantic_examples_r1(
        examples,
        encoder=encoder,
        control=SemanticControl.STATE_ONLY,
        pairing_registry=pairings,
    )
    transaction_only = tensorize_semantic_examples_r1(
        examples,
        encoder=encoder,
        control=SemanticControl.TRANSACTION_ONLY,
        pairing_registry=pairings,
    )

    assert torch.count_nonzero(state_only.visible.features).item() == 0
    assert torch.equal(full.visible.features, transaction_only.visible.features)
    assert torch.count_nonzero(transaction_only.visible.visible_state).item() == 0


def test_r1_candidate_update_uses_address_resolved_state_read() -> None:
    example = _examples()[1]
    batch = tensorize_semantic_examples_r1(
        [example],
        encoder=RelationalSemanticEncoderR1(),
    )
    context = batch.visible
    output = apply_batched_visible_update(
        context,
        erase=torch.ones(1),
        write=torch.zeros(1),
    )

    state_read = torch.bmm(
        context.visible_address.unsqueeze(1),
        context.visible_state,
    ).squeeze(1)
    erase_candidate = (
        context.visible_address.unsqueeze(2) * state_read.unsqueeze(1)
    ) * context.erase_candidate_scale[:, None, None]
    expected = context.visible_state - erase_candidate

    assert torch.allclose(output, expected)
    assert torch.linalg.vector_norm(erase_candidate).item() > 0.0


def test_r1_training_and_evaluation_keep_legacy_result_shapes() -> None:
    examples = _examples()
    encoder = RelationalSemanticEncoderR1()
    config = SemanticTrainingConfigV61(
        steps=2,
        batch_size=4,
        learning_rate=0.002,
        weight_decay=0.0,
        gradient_clip_norm=1.0,
        affected_read_weight=1.0,
        unaffected_retention_weight=1.0,
    )
    result = train_matched_semantic_pair_r1(
        examples,
        encoder=encoder,
        hidden_dim=8,
        config=config,
        seed=1_103,
        device=torch.device("cpu"),
    )

    assert isinstance(result.factorized, MatchedSemanticControllerV61)
    assert result.factorized.route is SemanticRoute.FACTORIZED
    assert result.shared.route is SemanticRoute.SHARED
    assert set(result.final_loss) == {"factorized", "shared"}
    assert result.factorized.input_dim == encoder.input_dim

    rows, affected, retention = evaluate_semantic_model_r1(
        result.factorized,
        examples,
        encoder=encoder,
        control=SemanticControl.FULL,
        pairing_registry=None,
        oracle_demand=False,
        device=torch.device("cpu"),
        batch_size=2,
    )
    assert len(rows) == len(examples)
    assert affected.shape == (len(examples),)
    assert retention.shape == (len(examples),)

    oracle_rows, oracle_affected, oracle_retention = evaluate_semantic_model_r1(
        None,
        examples,
        encoder=encoder,
        control=SemanticControl.FULL,
        pairing_registry=None,
        oracle_demand=True,
        device=torch.device("cpu"),
        batch_size=2,
    )
    assert len(oracle_rows) == len(examples)
    assert oracle_affected.shape == (len(examples),)
    assert oracle_retention.shape == (len(examples),)
