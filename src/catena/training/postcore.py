from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn


@dataclass(slots=True)
class TrainTrace:
    initial_loss: float
    final_loss: float
    best_loss: float
    steps: int


def train_matrix_controller(
    *,
    model: nn.Module,
    descriptors: torch.Tensor,
    targets: torch.Tensor,
    steps: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    seed: int,
    extra_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
) -> TrainTrace:
    if descriptors.shape[0] != targets.shape[0]:
        raise ValueError("descriptors and targets must contain the same number of rows")
    model.to(device)
    descriptors = descriptors.to(device)
    targets = targets.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    initial_loss = float("nan")
    best_loss = float("inf")
    final_loss = float("nan")
    count = descriptors.shape[0]

    model.train()
    for step in range(int(steps)):
        indices = torch.randint(
            count,
            (min(batch_size, count),),
            generator=generator,
            device="cpu",
        ).to(device)
        prediction = model(descriptors[indices])
        target = targets[indices]
        loss = torch.mean((prediction - target) ** 2)
        if extra_loss is not None:
            loss = loss + extra_loss(prediction, target)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}: {loss.item()}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        value = float(loss.detach().cpu())
        if step == 0:
            initial_loss = value
        best_loss = min(best_loss, value)
        final_loss = value

    return TrainTrace(
        initial_loss=initial_loss,
        final_loss=final_loss,
        best_loss=best_loss,
        steps=int(steps),
    )


@torch.no_grad()
def evaluate_matrix_controller(
    *,
    model: nn.Module,
    descriptors: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
) -> tuple[float, torch.Tensor]:
    model.to(device)
    model.eval()
    predictions: list[torch.Tensor] = []
    for start in range(0, descriptors.shape[0], batch_size):
        end = min(start + batch_size, descriptors.shape[0])
        predictions.append(model(descriptors[start:end].to(device)).cpu())
    prediction = torch.cat(predictions, dim=0)
    per_example = (prediction - targets.cpu()).square().mean(dim=(-2, -1))
    return float(per_example.mean()), per_example
