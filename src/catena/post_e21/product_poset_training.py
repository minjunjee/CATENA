"""E18-compatible learning and evaluation for the E23 product poset."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter

import torch
from torch import nn

from catena.data.product_poset_sequence import (
    generate_product_poset_sequence_batch,
)
from catena.data.sequence_control_lattice import base_sequence_control_digest
from catena.post_e21.locality_data import LocalityMethod
from catena.post_e21.locality_training import locality_retention_risk
from catena.post_e21.product_poset_model import (
    MatchedProductPosetSequenceController,
)
from catena.training.sequence_control_lattice import (
    indexed_sequence_lattice_seed,
)


@dataclass(frozen=True, slots=True)
class ProductPosetTrainResult:
    final_loss: float
    best_loss: float
    examples_per_second: float
    peak_memory_bytes: int
    optimizer: str
    locality_method_id: str
    locality_objective: str


def train_product_poset_controller(
    *,
    model: MatchedProductPosetSequenceController,
    demand_families: Sequence[str],
    intensities: Sequence[float],
    steps: int,
    batch_size: int,
    num_entities: int,
    value_dim: int,
    updates: int,
    gap_events: int,
    learning_rate: float,
    retention_weight: float,
    locality_method: LocalityMethod,
    locality_risk_scale: float,
    device: torch.device,
    seed: int,
) -> ProductPosetTrainResult:
    if not demand_families or not intensities or steps <= 0:
        raise ValueError("E23 training schedule must be nonempty")
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
        demand_index = step % len(demand_families)
        intensity_index = (step // len(demand_families)) % len(intensities)
        batch = generate_product_poset_sequence_batch(
            demand_family=str(demand_families[demand_index]),
            intensity=float(intensities[intensity_index]),
            batch_size=int(batch_size),
            num_entities=int(num_entities),
            value_dim=int(value_dim),
            updates=int(updates),
            gap_events=int(gap_events),
            seed=indexed_sequence_lattice_seed(seed, "train-batch", step),
            device=device,
        )
        output = model(batch.inputs)
        entity_error = (output.state - batch.target_state).square().mean(dim=-1)
        affected_loss = entity_error[batch.affected_entities].mean()
        retained_values = entity_error[~batch.affected_entities]
        retention = locality_retention_risk(
            retained_values,
            method=locality_method,
            risk_scale=locality_risk_scale,
        )
        loss = affected_loss + float(retention_weight) * retention
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite E23 loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        final = float(loss.detach().cpu())
        best = min(best, final)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = max(perf_counter() - started, 1.0e-8)
    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    return ProductPosetTrainResult(
        final_loss=final,
        best_loss=best,
        examples_per_second=float(steps * batch_size / elapsed),
        peak_memory_bytes=peak,
        optimizer="AdamW",
        locality_method_id=locality_method.method_id,
        locality_objective=locality_method.objective.value,
    )


@torch.no_grad()
def evaluate_product_poset_controller(
    *,
    model: MatchedProductPosetSequenceController,
    demand_family: str,
    intensity: float,
    batches: int,
    batch_size: int,
    num_entities: int,
    value_dim: int,
    updates: int,
    gap_events: int,
    device: torch.device,
    seed: int,
) -> dict[str, float | int | str]:
    model.to(device)
    model.eval()
    affected_sum = 0.0
    retention_sum = 0.0
    state_sum = 0.0
    baseline_affected_sum = 0.0
    affected_rows = 0
    unaffected_rows = 0
    all_rows = 0
    worst_nontarget = 0.0
    digest = hashlib.sha256()
    for batch_index in range(int(batches)):
        batch = generate_product_poset_sequence_batch(
            demand_family=demand_family,
            intensity=float(intensity),
            batch_size=int(batch_size),
            num_entities=int(num_entities),
            value_dim=int(value_dim),
            updates=int(updates),
            gap_events=int(gap_events),
            seed=indexed_sequence_lattice_seed(
                seed,
                "evaluation-batch",
                batch_index,
            ),
            device=device,
        )
        digest.update(bytes.fromhex(base_sequence_control_digest(batch)))
        output = model(batch.inputs)
        entity_error = (output.state - batch.target_state).square().mean(dim=-1)
        baseline_error = (batch.inputs.initial_state - batch.target_state).square().mean(dim=-1)
        affected = batch.affected_entities
        unaffected = ~affected
        affected_sum += float(entity_error[affected].sum())
        retention_sum += float(entity_error[unaffected].sum())
        state_sum += float(entity_error.sum())
        baseline_affected_sum += float(baseline_error[affected].sum())
        affected_rows += int(affected.sum())
        unaffected_rows += int(unaffected.sum())
        all_rows += int(entity_error.numel())
        if bool(unaffected.any()):
            worst_nontarget = max(
                worst_nontarget,
                float(entity_error[unaffected].max()),
            )
    affected_mse = affected_sum / max(affected_rows, 1)
    baseline_affected = baseline_affected_sum / max(affected_rows, 1)
    return {
        "affected_mse": affected_mse,
        "retention_mse": retention_sum / max(unaffected_rows, 1),
        "state_mse": state_sum / max(all_rows, 1),
        "worst_nontarget_mse": worst_nontarget,
        "target_correction_gain": baseline_affected - affected_mse,
        "no_update_affected_mse": baseline_affected,
        "affected_entity_count": affected_rows,
        "unaffected_entity_count": unaffected_rows,
        "base_transaction_digest": digest.hexdigest(),
    }
