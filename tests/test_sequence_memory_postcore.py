from __future__ import annotations

import torch

from catena.data.transactional_sequence import generate_transactional_sequence_batch
from catena.models.sequence_memory import SequenceControl, TransactionalSequenceMemory


def test_transactional_sequence_generation_and_forward() -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(13)
    batch = generate_transactional_sequence_batch(
        batch_size=4,
        num_entities=8,
        value_vocab=12,
        updates=3,
        gap_events=5,
        generator=generator,
        device=torch.device("cpu"),
    )
    model = TransactionalSequenceMemory(
        control=SequenceControl.DUAL,
        num_entities=8,
        value_vocab=12,
        embedding_dim=16,
        hidden_dim=32,
    )
    output = model(batch)
    assert output.shape == (4, 8, 12)
    assert torch.isfinite(output).all()
    loss = (output - batch.target_state).square().mean()
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())



def test_sequence_evaluation_uses_actual_affected_counts() -> None:
    import torch
    from torch import nn

    from catena.training.sequence_training import evaluate_sequence_memory

    class UnitErrorModel(nn.Module):
        def forward(self, batch):  # type: ignore[no-untyped-def]
            return batch.target_state + 1.0

    # Eight updates target the same single entity.  The metric denominator must
    # count the affected entity once per episode, not once per update.
    metrics = evaluate_sequence_memory(
        model=UnitErrorModel(),  # type: ignore[arg-type]
        batches=2,
        batch_size=8,
        num_entities=1,
        value_vocab=8,
        updates=8,
        gap_events=0,
        device=torch.device("cpu"),
        seed=123,
    )
    assert abs(metrics["affected_mse"] - 1.0) < 1e-6
