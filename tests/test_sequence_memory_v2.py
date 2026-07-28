from __future__ import annotations

import inspect

import torch

from catena.data.transactional_sequence import (
    TransactionalSequenceBatch,
    generate_transactional_sequence_batch,
)
from catena.data.transactional_sequence_v2 import (
    TransactionalSequenceBatchV2,
    TransactionalSequenceInputV2,
    base_transaction_digest_v2,
    generate_transactional_sequence_batch_v2,
    sequence_model_input_v2,
)
from catena.models.sequence_memory import (
    SequenceControl,
    TransactionalSequenceMemory,
)
from catena.models.sequence_memory_v2 import (
    SequenceControlV2,
    TransactionalSequenceMemoryV2,
    sequence_parameter_count_v2,
)
from catena.training.sequence_training_v2 import (
    evaluate_sequence_memory_v2,
    train_sequence_memory_v2,
)


def _old_model() -> TransactionalSequenceMemory:
    return TransactionalSequenceMemory(
        control=SequenceControl.DUAL,
        num_entities=8,
        value_vocab=16,
        embedding_dim=8,
        hidden_dim=16,
    )


def _v2_model(control: SequenceControlV2) -> TransactionalSequenceMemoryV2:
    return TransactionalSequenceMemoryV2(
        control=control,
        num_entities=8,
        value_vocab=16,
        embedding_dim=8,
        hidden_dim=16,
    )


def _v2_batch(*, updates: int, gap_events: int, seed: int = 71) -> TransactionalSequenceBatchV2:
    return generate_transactional_sequence_batch_v2(
        batch_size=16,
        num_entities=8,
        value_vocab=16,
        updates=updates,
        gap_events=gap_events,
        seed=seed,
        device=torch.device("cpu"),
    )


def _verified_rows(batch: TransactionalSequenceBatchV2, value: torch.Tensor) -> torch.Tensor:
    batch_size = batch.update_mask.shape[0]
    updates = int(batch.update_mask[0].sum().item())
    return value[batch.update_mask].reshape(batch_size, updates, *value.shape[2:])


def test_old_appended_distractors_are_structurally_hard_masked() -> None:
    generator = torch.Generator(device="cpu").manual_seed(13)
    full = generate_transactional_sequence_batch(
        batch_size=8,
        num_entities=8,
        value_vocab=16,
        updates=2,
        gap_events=5,
        generator=generator,
        device=torch.device("cpu"),
    )
    update_only = TransactionalSequenceBatch(
        initial_state=full.initial_state,
        entity_ids=full.entity_ids[:, :2],
        old_value_ids=full.old_value_ids[:, :2],
        new_value_ids=full.new_value_ids[:, :2],
        semantic_features=full.semantic_features[:, :2],
        update_mask=full.update_mask[:, :2],
        target_state=full.target_state,
        affected_entities=full.affected_entities,
        operations=full.operations[:, :2],
    )
    torch.manual_seed(19)
    model = _old_model().eval()

    full_state = model(full)
    update_only_state = model(update_only)

    assert torch.equal(full_state, update_only_state)
    assert not full.update_mask[:, 2:].any()
    assert torch.equal(
        full.semantic_features[:, 2:, 5],
        torch.zeros_like(full.semantic_features[:, 2:, 5]),
    )


def test_v2_places_one_exact_distractor_block_after_first_update() -> None:
    batch = _v2_batch(updates=3, gap_events=4)
    expected = torch.tensor([True, False, False, False, False, True, True])

    assert torch.equal(batch.update_mask[0], expected)
    assert torch.equal(
        batch.update_mask,
        expected[None, :].expand_as(batch.update_mask),
    )
    assert torch.equal(
        batch.operations[~batch.update_mask],
        torch.full_like(batch.operations[~batch.update_mask], -1),
    )
    assert torch.equal(
        batch.inputs.semantic_features[:, 1:5, 5],
        torch.zeros_like(batch.inputs.semantic_features[:, 1:5, 5]),
    )

    one_update = _v2_batch(updates=1, gap_events=4)
    assert torch.equal(
        one_update.update_mask[0],
        torch.tensor([True, False, False, False, False]),
    )


def test_v2_base_transactions_are_gap_independent_and_distractors_are_nested() -> None:
    gap_zero = _v2_batch(updates=3, gap_events=0, seed=83)
    gap_two = _v2_batch(updates=3, gap_events=2, seed=83)
    gap_five = _v2_batch(updates=3, gap_events=5, seed=83)

    assert base_transaction_digest_v2(gap_zero) == base_transaction_digest_v2(gap_two)
    assert base_transaction_digest_v2(gap_two) == base_transaction_digest_v2(gap_five)
    for field in (
        "entity_ids",
        "old_value_ids",
        "new_value_ids",
        "semantic_features",
    ):
        zero_rows = _verified_rows(gap_zero, getattr(gap_zero.inputs, field))
        five_rows = _verified_rows(gap_five, getattr(gap_five.inputs, field))
        assert torch.equal(zero_rows, five_rows)
    assert torch.equal(
        gap_zero.inputs.initial_state,
        gap_five.inputs.initial_state,
    )
    assert torch.equal(gap_zero.target_state, gap_five.target_state)

    assert torch.equal(
        gap_two.inputs.entity_ids[:, 1:3],
        gap_five.inputs.entity_ids[:, 1:3],
    )
    assert torch.equal(
        gap_two.inputs.old_value_ids[:, 1:3],
        gap_five.inputs.old_value_ids[:, 1:3],
    )
    assert torch.equal(
        gap_two.inputs.new_value_ids[:, 1:3],
        gap_five.inputs.new_value_ids[:, 1:3],
    )
    assert torch.equal(
        gap_two.inputs.semantic_features[:, 1:3],
        gap_five.inputs.semantic_features[:, 1:3],
    )


def test_v2_random_model_has_a_causally_active_distractor_path() -> None:
    batch = _v2_batch(updates=2, gap_events=5, seed=97)
    update_only = TransactionalSequenceInputV2(
        initial_state=batch.inputs.initial_state,
        entity_ids=_verified_rows(batch, batch.inputs.entity_ids),
        old_value_ids=_verified_rows(batch, batch.inputs.old_value_ids),
        new_value_ids=_verified_rows(batch, batch.inputs.new_value_ids),
        semantic_features=_verified_rows(batch, batch.inputs.semantic_features),
    )
    torch.manual_seed(101)
    model = _v2_model(SequenceControlV2.DUAL).eval()

    full_state = model(sequence_model_input_v2(batch)).state
    update_only_state = model(update_only).state

    assert not torch.allclose(full_state, update_only_state, atol=1e-8, rtol=0.0)


def test_v2_model_cannot_access_oracle_update_mask() -> None:
    model_fields = set(TransactionalSequenceInputV2.__dataclass_fields__)
    forward_source = inspect.getsource(TransactionalSequenceMemoryV2.forward)
    batch = _v2_batch(updates=2, gap_events=3, seed=103)
    changed_metadata = TransactionalSequenceBatchV2(
        inputs=batch.inputs,
        update_mask=~batch.update_mask,
        target_state=batch.target_state,
        affected_entities=batch.affected_entities,
        operations=batch.operations,
    )
    torch.manual_seed(107)
    model = _v2_model(SequenceControlV2.DUAL).eval()

    original = model(batch.inputs).state
    metadata_changed = model(changed_metadata.inputs).state

    assert "update_mask" not in model_fields
    assert "update_mask" not in forward_source
    assert torch.equal(original, metadata_changed)


def test_v2_active_path_assay_changes_only_distractor_verification_feature() -> None:
    batch = _v2_batch(updates=2, gap_events=3, seed=109)
    normal = sequence_model_input_v2(batch)
    activated = sequence_model_input_v2(
        batch,
        activate_distractor_verified=True,
    )
    distractor_mask = ~batch.update_mask

    assert torch.equal(normal.initial_state, activated.initial_state)
    assert torch.equal(normal.entity_ids, activated.entity_ids)
    assert torch.equal(normal.old_value_ids, activated.old_value_ids)
    assert torch.equal(normal.new_value_ids, activated.new_value_ids)
    assert torch.equal(
        normal.semantic_features[:, :, :5],
        activated.semantic_features[:, :, :5],
    )
    assert torch.equal(
        normal.semantic_features[:, :, 5][batch.update_mask],
        activated.semantic_features[:, :, 5][batch.update_mask],
    )
    assert torch.equal(
        activated.semantic_features[:, :, 5][distractor_mask],
        torch.ones_like(activated.semantic_features[:, :, 5][distractor_mask]),
    )


def test_v2_tied_and_dual_have_the_same_registered_parameter_surface() -> None:
    torch.manual_seed(113)
    tied = _v2_model(SequenceControlV2.TIED)
    torch.manual_seed(113)
    dual = _v2_model(SequenceControlV2.DUAL)

    assert sequence_parameter_count_v2(tied) == sequence_parameter_count_v2(dual)
    assert tied.state_dict().keys() == dual.state_dict().keys()
    assert all(
        torch.equal(tied.state_dict()[name], dual.state_dict()[name])
        for name in tied.state_dict()
    )


def test_v2_evaluation_reports_paired_digest_and_distractor_gates() -> None:
    torch.manual_seed(127)
    model = _v2_model(SequenceControlV2.DUAL)
    gap_two = evaluate_sequence_memory_v2(
        model=model,
        batches=2,
        batch_size=4,
        num_entities=8,
        value_vocab=16,
        updates=2,
        gap_events=2,
        device=torch.device("cpu"),
        seed=131,
    )
    gap_five = evaluate_sequence_memory_v2(
        model=model,
        batches=2,
        batch_size=4,
        num_entities=8,
        value_vocab=16,
        updates=2,
        gap_events=5,
        device=torch.device("cpu"),
        seed=131,
    )
    activated = evaluate_sequence_memory_v2(
        model=model,
        batches=2,
        batch_size=4,
        num_entities=8,
        value_vocab=16,
        updates=2,
        gap_events=5,
        device=torch.device("cpu"),
        seed=131,
        activate_distractor_verified=True,
    )

    assert gap_two["base_transaction_digest"] == gap_five["base_transaction_digest"]
    assert gap_five["base_transaction_digest"] == activated["base_transaction_digest"]
    assert gap_two["distractor_event_count"] == 16
    assert gap_five["distractor_event_count"] == 40
    assert gap_five["verified_event_count"] == 16
    assert activated["activate_distractor_verified"] is True
    assert gap_five["activate_distractor_verified"] is False
    assert gap_five["distractor_erase_gate_mean"] != (
        activated["distractor_erase_gate_mean"]
    )


def test_v2_training_path_updates_the_unmasked_event_controller() -> None:
    torch.manual_seed(137)
    model = _v2_model(SequenceControlV2.DUAL)
    before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }

    result = train_sequence_memory_v2(
        model=model,
        steps=1,
        batch_size=4,
        num_entities=8,
        value_vocab=16,
        updates=2,
        gap_events=3,
        learning_rate=0.001,
        retention_weight=1.0,
        device=torch.device("cpu"),
        seed=139,
    )

    assert torch.isfinite(torch.tensor(result.final_loss))
    assert result.examples_per_second > 0.0
    assert any(
        not torch.equal(before[name], value)
        for name, value in model.state_dict().items()
    )
