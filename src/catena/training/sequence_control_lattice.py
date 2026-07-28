from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter

import torch
from torch import nn

from catena.data.sequence_control_lattice import (
    SequenceDemandFamily,
    base_sequence_control_digest,
    generate_sequence_control_lattice_batch,
    sequence_control_lattice_model_input,
)
from catena.models.sequence_control_lattice import MatchedSequenceControlLattice


@dataclass(slots=True)
class SequenceLatticeTrainResult:
    final_loss: float
    best_loss: float
    examples_per_second: float
    peak_memory_bytes: int
    optimizer: str


def indexed_sequence_lattice_seed(
    seed: int,
    namespace: str,
    index: int,
) -> int:
    digest = hashlib.sha256(f"{seed}\0{namespace}\0{index}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def state_dict_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state_dict.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _sequence_lattice_loss(
    *,
    prediction: torch.Tensor,
    target: torch.Tensor,
    affected: torch.Tensor,
    retention_weight: float,
) -> torch.Tensor:
    error = (prediction - target).square().mean(dim=-1)
    affected_loss = error[affected].mean() if affected.any() else error.mean()
    unaffected = ~affected
    retention_loss = (
        error[unaffected].mean()
        if unaffected.any()
        else error.mean() * 0.0
    )
    return affected_loss + float(retention_weight) * retention_loss


def train_sequence_control_lattice(
    *,
    model: MatchedSequenceControlLattice,
    families: list[SequenceDemandFamily],
    steps: int,
    batch_size: int,
    num_entities: int,
    value_dim: int,
    updates: int,
    gap_events: int,
    learning_rate: float,
    retention_weight: float,
    device: torch.device,
    seed: int,
) -> SequenceLatticeTrainResult:
    if not families:
        raise ValueError("families must not be empty")
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = perf_counter()
    best = float("inf")
    final = float("nan")

    for step in range(int(steps)):
        family = families[step % len(families)]
        batch = generate_sequence_control_lattice_batch(
            family=family,
            batch_size=batch_size,
            num_entities=num_entities,
            value_dim=value_dim,
            updates=updates,
            gap_events=gap_events,
            seed=indexed_sequence_lattice_seed(seed, "train-batch", step),
            device=device,
        )
        output = model(sequence_control_lattice_model_input(batch))
        loss = _sequence_lattice_loss(
            prediction=output.state,
            target=batch.target_state,
            affected=batch.affected_entities,
            retention_weight=retention_weight,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"non-finite sequence lattice loss at step {step}"
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        final = float(loss.detach().cpu())
        best = min(best, final)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = max(perf_counter() - started, 1e-8)
    peak = (
        torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else 0
    )
    return SequenceLatticeTrainResult(
        final_loss=final,
        best_loss=best,
        examples_per_second=float(steps * batch_size / elapsed),
        peak_memory_bytes=int(peak),
        optimizer="AdamW",
    )


@torch.no_grad()
def evaluate_sequence_control_lattice(
    *,
    model: MatchedSequenceControlLattice,
    family: SequenceDemandFamily,
    batches: int,
    batch_size: int,
    num_entities: int,
    value_dim: int,
    updates: int,
    gap_events: int,
    device: torch.device,
    seed: int,
    activate_distractor_verified: bool = False,
) -> dict[str, float | int | str | bool]:
    model.to(device)
    model.eval()
    affected_error = 0.0
    retention_error = 0.0
    state_error = 0.0
    exact = 0.0
    affected_exact = 0.0
    affected_rows = 0
    unaffected_rows = 0
    all_rows = 0
    verified_erase_sum = 0.0
    verified_write_sum = 0.0
    distractor_erase_sum = 0.0
    distractor_write_sum = 0.0
    verified_gate_values = 0
    distractor_gate_values = 0
    digest = hashlib.sha256()

    for batch_index in range(int(batches)):
        batch = generate_sequence_control_lattice_batch(
            family=family,
            batch_size=batch_size,
            num_entities=num_entities,
            value_dim=value_dim,
            updates=updates,
            gap_events=gap_events,
            seed=indexed_sequence_lattice_seed(
                seed,
                "evaluation-batch",
                batch_index,
            ),
            device=device,
        )
        digest.update(bytes.fromhex(base_sequence_control_digest(batch)))
        inputs = sequence_control_lattice_model_input(
            batch,
            activate_distractor_verified=activate_distractor_verified,
        )
        output = model(inputs)
        entity_error = (output.state - batch.target_state).square().mean(dim=-1)
        affected = batch.affected_entities
        unaffected = ~affected
        affected_error += float(entity_error[affected].sum())
        retention_error += float(entity_error[unaffected].sum())
        state_error += float(entity_error.sum())
        affected_rows += int(affected.sum())
        unaffected_rows += int(unaffected.sum())
        all_rows += int(entity_error.numel())

        entity_exact = (
            (output.state - batch.target_state).abs() <= 1e-3
        ).all(dim=-1).to(torch.float32)
        exact += float(entity_exact.sum())
        affected_exact += float(entity_exact[affected].sum())

        update_mask = batch.update_mask.unsqueeze(-1).expand_as(
            output.erase_gates
        )
        distractor_mask = ~update_mask
        verified_erase_sum += float(output.erase_gates[update_mask].sum())
        verified_write_sum += float(output.write_gates[update_mask].sum())
        distractor_erase_sum += float(output.erase_gates[distractor_mask].sum())
        distractor_write_sum += float(output.write_gates[distractor_mask].sum())
        verified_gate_values += int(update_mask.sum())
        distractor_gate_values += int(distractor_mask.sum())

    return {
        "state_mse": state_error / max(all_rows, 1),
        "affected_mse": affected_error / max(affected_rows, 1),
        "retention_mse": retention_error / max(unaffected_rows, 1),
        "entity_exact_match": exact / max(all_rows, 1),
        "affected_entity_exact_match": (
            affected_exact / max(affected_rows, 1)
        ),
        "affected_entity_count": affected_rows,
        "unaffected_entity_count": unaffected_rows,
        "verified_erase_gate_mean": (
            verified_erase_sum / max(verified_gate_values, 1)
        ),
        "verified_write_gate_mean": (
            verified_write_sum / max(verified_gate_values, 1)
        ),
        "distractor_erase_gate_mean": (
            distractor_erase_sum / max(distractor_gate_values, 1)
        ),
        "distractor_write_gate_mean": (
            distractor_write_sum / max(distractor_gate_values, 1)
        ),
        "distractor_joint_gate_mass_per_sequence": (
            distractor_erase_sum + distractor_write_sum
        )
        / max(int(batches) * batch_size, 1),
        "activate_distractor_verified": activate_distractor_verified,
        "base_transaction_digest": digest.hexdigest(),
    }
