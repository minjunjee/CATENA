from __future__ import annotations

import hashlib
from dataclasses import dataclass
from time import perf_counter

import torch
from torch import nn

from catena.data.transactional_sequence_v2 import (
    base_transaction_digest_v2,
    generate_transactional_sequence_batch_v2,
    sequence_model_input_v2,
)
from catena.models.sequence_memory_v2 import TransactionalSequenceMemoryV2


@dataclass(slots=True)
class SequenceTrainResultV2:
    final_loss: float
    best_loss: float
    examples_per_second: float
    peak_memory_bytes: int


def _indexed_seed(seed: int, namespace: str, index: int) -> int:
    digest = hashlib.sha256(f"{seed}\0{namespace}\0{index}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def _sequence_loss_v2(
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


def train_sequence_memory_v2(
    *,
    model: TransactionalSequenceMemoryV2,
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
) -> SequenceTrainResultV2:
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = perf_counter()
    best = float("inf")
    final = float("nan")

    for step in range(int(steps)):
        batch = generate_transactional_sequence_batch_v2(
            batch_size=batch_size,
            num_entities=num_entities,
            value_vocab=value_vocab,
            updates=updates,
            gap_events=gap_events,
            seed=_indexed_seed(seed, "train-batch", step),
            device=device,
        )
        output = model(sequence_model_input_v2(batch))
        loss = _sequence_loss_v2(
            output.state,
            batch.target_state,
            batch.affected_entities,
            retention_weight,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite sequence loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        final = float(loss.detach().cpu())
        best = min(best, final)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = max(perf_counter() - started, 1e-8)
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    return SequenceTrainResultV2(
        final_loss=final,
        best_loss=best,
        examples_per_second=float(steps * batch_size / elapsed),
        peak_memory_bytes=int(peak),
    )


@torch.no_grad()
def evaluate_sequence_memory_v2(
    *,
    model: TransactionalSequenceMemoryV2,
    batches: int,
    batch_size: int,
    num_entities: int,
    value_vocab: int,
    updates: int,
    gap_events: int,
    device: torch.device,
    seed: int,
    activate_distractor_verified: bool = False,
) -> dict[str, float | int | str | bool]:
    """Evaluate a paired gap condition and expose learned distractor gates.

    When ``activate_distractor_verified`` is true, only semantic feature five
    changes from zero to one on metadata-marked distractors.  Targets and all
    base transactions remain fixed.
    """
    model.to(device)
    model.eval()
    affected_error = 0.0
    retention_error = 0.0
    old_residual = 0.0
    exact = 0.0
    affected_exact = 0.0
    rows = 0
    affected_rows = 0
    unaffected_rows = 0
    verified_erase_sum = 0.0
    verified_write_sum = 0.0
    distractor_erase_sum = 0.0
    distractor_write_sum = 0.0
    verified_events = 0
    distractor_events = 0
    digest = hashlib.sha256()

    for batch_index in range(int(batches)):
        batch = generate_transactional_sequence_batch_v2(
            batch_size=batch_size,
            num_entities=num_entities,
            value_vocab=value_vocab,
            updates=updates,
            gap_events=gap_events,
            seed=_indexed_seed(seed, "evaluation-batch", batch_index),
            device=device,
        )
        digest.update(bytes.fromhex(base_transaction_digest_v2(batch)))
        inputs = sequence_model_input_v2(
            batch,
            activate_distractor_verified=activate_distractor_verified,
        )
        output = model(inputs)
        prediction = output.state
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
        old_residual += float(
            (prediction * (1.0 - batch.target_state)).mean(dim=-1).sum()
        )
        rows += batch_size * num_entities

        update_mask = batch.update_mask
        distractor_mask = ~update_mask
        verified_erase_sum += float(output.erase_gates[update_mask].sum())
        verified_write_sum += float(output.write_gates[update_mask].sum())
        distractor_erase_sum += float(output.erase_gates[distractor_mask].sum())
        distractor_write_sum += float(output.write_gates[distractor_mask].sum())
        verified_events += int(update_mask.sum().item())
        distractor_events += int(distractor_mask.sum().item())

    return {
        "affected_mse": affected_error / max(affected_rows, 1),
        "retention_mse": retention_error / max(unaffected_rows, 1),
        "old_rule_residual": old_residual / max(rows, 1),
        "entity_exact_match": exact / max(rows, 1),
        "affected_entity_exact_match": affected_exact / max(affected_rows, 1),
        "affected_entity_count": affected_rows,
        "unaffected_entity_count": unaffected_rows,
        "verified_event_count": verified_events,
        "distractor_event_count": distractor_events,
        "verified_erase_gate_mean": verified_erase_sum / max(verified_events, 1),
        "verified_write_gate_mean": verified_write_sum / max(verified_events, 1),
        "distractor_erase_gate_mean": distractor_erase_sum
        / max(distractor_events, 1),
        "distractor_write_gate_mean": distractor_write_sum
        / max(distractor_events, 1),
        "distractor_joint_gate_mass_per_sequence": (
            distractor_erase_sum + distractor_write_sum
        )
        / max(int(batches) * batch_size, 1),
        "activate_distractor_verified": activate_distractor_verified,
        "base_transaction_digest": digest.hexdigest(),
    }
