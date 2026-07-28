from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from catena.data.control_lattice import DemandAxis, generate_control_lattice_batch
from catena.models.lattice_controllers import MatchedControlLatticeController


@dataclass(slots=True)
class LatticeTrainResult:
    final_loss: float
    best_loss: float


def train_lattice_controller(
    *,
    model: MatchedControlLatticeController,
    families: list[DemandAxis],
    steps: int,
    batch_size: int,
    slots: int,
    value_dim: int,
    learning_rate: float,
    device: torch.device,
    seed: int,
) -> LatticeTrainResult:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    best = float("inf")
    final = float("nan")
    for step in range(int(steps)):
        family = families[step % len(families)]
        batch = generate_control_lattice_batch(
            family=family,
            batch_size=batch_size,
            slots=slots,
            value_dim=value_dim,
            generator=generator,
            device=device,
        )
        output = model(batch)
        loss = nn.functional.mse_loss(output.state, batch.target)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite lattice loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        final = float(loss.detach().cpu())
        best = min(best, final)
    return LatticeTrainResult(final_loss=final, best_loss=best)


@torch.no_grad()
def evaluate_lattice_controller(
    *,
    model: MatchedControlLatticeController,
    family: DemandAxis,
    episodes: int,
    batch_size: int,
    slots: int,
    value_dim: int,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    model.to(device)
    model.eval()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    total_mse = 0.0
    total_affected = 0.0
    total_retention = 0.0
    total_retention_slots = 0
    count = 0
    for _ in range((episodes + batch_size - 1) // batch_size):
        current = min(batch_size, episodes - count)
        if current <= 0:
            break
        batch = generate_control_lattice_batch(
            family=family,
            batch_size=current,
            slots=slots,
            value_dim=value_dim,
            generator=generator,
            device=device,
        )
        output = model(batch)
        error = (output.state - batch.target).square()
        total_mse += float(error.mean()) * current
        batch_index = torch.arange(current, device=device)
        affected = error[batch_index, batch.erase_address].mean(dim=-1)
        if family is DemandAxis.ADDRESS:
            affected = 0.5 * (
                affected + error[batch_index, batch.write_address].mean(dim=-1)
            )
        total_affected += float(affected.sum())
        mask = torch.ones(current, slots, dtype=torch.bool, device=device)
        mask[batch_index, batch.erase_address] = False
        mask[batch_index, batch.write_address] = False
        if mask.any():
            retention = error[mask].mean(dim=-1)
            total_retention += float(retention.sum())
            total_retention_slots += int(mask.sum())
        count += current
    return {
        "state_mse": total_mse / count,
        "affected_mse": total_affected / count,
        "retention_mse": total_retention / max(total_retention_slots, 1),
    }
