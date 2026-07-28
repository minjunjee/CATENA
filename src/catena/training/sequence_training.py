from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import torch
from torch import nn

from catena.data.transactional_sequence import generate_transactional_sequence_batch
from catena.models.sequence_memory import TransactionalSequenceMemory


@dataclass(slots=True)
class SequenceTrainResult:
    final_loss: float
    best_loss: float
    examples_per_second: float
    peak_memory_bytes: int


def _sequence_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    affected: torch.Tensor,
    retention_weight: float,
) -> torch.Tensor:
    error = (prediction - target).square().mean(dim=-1)
    affected_loss = error[affected].mean() if affected.any() else error.mean()
    unaffected = ~affected
    retention_loss = error[unaffected].mean() if unaffected.any() else error.mean() * 0.0
    return affected_loss + retention_weight * retention_loss


def train_sequence_memory(
    *,
    model: TransactionalSequenceMemory,
    steps: int,
    batch_size: int,
    num_entities: int,
    value_vocab: int,
    updates: int,
    gap_events: int,
    learning_rate: float,
    retention_weight: float,
    device: torch.device,
    seed: int,
) -> SequenceTrainResult:
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = perf_counter()
    best = float("inf")
    final = float("nan")
    for step in range(int(steps)):
        batch = generate_transactional_sequence_batch(
            batch_size=batch_size,
            num_entities=num_entities,
            value_vocab=value_vocab,
            updates=updates,
            gap_events=gap_events,
            generator=generator,
            device=device,
        )
        prediction = model(batch)
        loss = _sequence_loss(
            prediction,
            batch.target_state,
            batch.affected_entities,
            retention_weight,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite sequence loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        final = float(loss.detach().cpu())
        best = min(best, final)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = max(perf_counter() - started, 1e-8)
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    return SequenceTrainResult(
        final_loss=final,
        best_loss=best,
        examples_per_second=float(steps * batch_size / elapsed),
        peak_memory_bytes=int(peak),
    )


@torch.no_grad()
def evaluate_sequence_memory(
    *,
    model: TransactionalSequenceMemory,
    batches: int,
    batch_size: int,
    num_entities: int,
    value_vocab: int,
    updates: int,
    gap_events: int,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    model.to(device)
    model.eval()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    affected_error = 0.0
    retention_error = 0.0
    old_residual = 0.0
    exact = 0.0
    affected_exact = 0.0
    rows = 0
    affected_rows = 0
    unaffected_rows = 0
    for _ in range(int(batches)):
        batch = generate_transactional_sequence_batch(
            batch_size=batch_size,
            num_entities=num_entities,
            value_vocab=value_vocab,
            updates=updates,
            gap_events=gap_events,
            generator=generator,
            device=device,
        )
        prediction = model(batch)
        error = (prediction - batch.target_state).square().mean(dim=-1)
        affected_mask = batch.affected_entities
        unaffected_mask = ~affected_mask
        affected_error += float(error[affected_mask].sum())
        retention_error += float(error[unaffected_mask].sum())
        affected_rows += int(affected_mask.sum().item())
        unaffected_rows += int(unaffected_mask.sum().item())
        predicted_bits = prediction > 0.5
        entity_exact = (
            predicted_bits == (batch.target_state > 0.5)
        ).all(dim=-1).to(torch.float32)
        exact += float(entity_exact.sum())
        affected_exact += float(entity_exact[affected_mask].sum())
        # Old-rule residual: mass assigned where target is zero.
        old_residual += float((prediction * (1.0 - batch.target_state)).mean(dim=-1).sum())
        rows += batch_size * num_entities
    return {
        "affected_mse": affected_error / max(affected_rows, 1),
        "retention_mse": retention_error / max(unaffected_rows, 1),
        "old_rule_residual": old_residual / rows,
        "entity_exact_match": exact / rows,
        "affected_entity_exact_match": affected_exact / max(affected_rows, 1),
        "affected_entity_count": affected_rows,
        "unaffected_entity_count": unaffected_rows,
    }
