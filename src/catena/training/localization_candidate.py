from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from catena.data.localization_candidate import (
    LocalizationCandidateCondition,
    generate_localization_candidate_batch,
)
from catena.models.localization_candidate import (
    MatchedLocalizationCandidateController,
)


@dataclass(slots=True)
class LocalizationCandidateTrainResult:
    final_loss: float
    best_loss: float


def _address_negative_log_likelihood(
    weights: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    probability = weights.gather(1, target[:, None]).squeeze(1)
    return -probability.clamp_min(1e-8).log().mean()


def train_localization_candidate_controller(
    *,
    model: MatchedLocalizationCandidateController,
    conditions: list[LocalizationCandidateCondition],
    steps: int,
    batch_size: int,
    slots: int,
    value_dim: int,
    state_scale: float,
    address_codebook: torch.Tensor,
    learning_rate: float,
    address_loss_weight: float,
    candidate_loss_weight: float,
    retention_weight: float,
    device: torch.device,
    seed: int,
) -> LocalizationCandidateTrainResult:
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    best = float("inf")
    final = float("nan")
    for step in range(int(steps)):
        condition = conditions[step % len(conditions)]
        batch = generate_localization_candidate_batch(
            batch_size=batch_size,
            slots=slots,
            value_dim=value_dim,
            state_scale=state_scale,
            address_codebook=address_codebook,
            generator=generator,
            device=device,
        )
        output = model(batch, condition)
        error = (output.state - batch.target).square()
        row = torch.arange(batch_size, device=device)
        affected = 0.5 * (
            error[row, batch.erase_address].mean()
            + error[row, batch.write_address].mean()
        )
        unaffected = torch.ones(
            batch_size,
            slots,
            dtype=torch.bool,
            device=device,
        )
        unaffected[row, batch.erase_address] = False
        unaffected[row, batch.write_address] = False
        retention = (
            error[unaffected].mean()
            if unaffected.any()
            else torch.zeros((), device=device)
        )
        loss = affected + float(retention_weight) * retention
        if not condition.uses_oracle_address:
            loss = loss + float(address_loss_weight) * (
                _address_negative_log_likelihood(
                    output.erase_address_weights,
                    batch.erase_address,
                )
                + _address_negative_log_likelihood(
                    output.write_address_weights,
                    batch.write_address,
                )
            )
        if not condition.uses_oracle_candidate:
            loss = loss + float(candidate_loss_weight) * nn.functional.mse_loss(
                output.erase_candidate,
                batch.old_candidate,
            )

        # Oracle paths can make a capable controller's task loss independent
        # of its maximal head. Preserve a valid paired optimizer step.
        loss = (
            loss
            + 0.0 * output.raw_address_logits.sum()
            + 0.0 * output.raw_candidate.sum()
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"non-finite localization/candidate loss at step {step}"
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        final = float(loss.detach().cpu())
        best = min(best, final)
    return LocalizationCandidateTrainResult(final_loss=final, best_loss=best)


@torch.no_grad()
def evaluate_localization_candidate_controller(
    *,
    model: MatchedLocalizationCandidateController,
    condition: LocalizationCandidateCondition,
    episodes: int,
    batch_size: int,
    slots: int,
    value_dim: int,
    state_scale: float,
    address_codebook: torch.Tensor,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    model.to(device)
    model.eval()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    totals = {
        "state_mse": 0.0,
        "address_accuracy": 0.0,
        "candidate_recovery_mse": 0.0,
        "affected_mse": 0.0,
        "retention_mse": 0.0,
        "old_residual": 0.0,
    }
    retention_slots = 0
    count = 0
    for _ in range((episodes + batch_size - 1) // batch_size):
        current = min(batch_size, episodes - count)
        if current <= 0:
            break
        batch = generate_localization_candidate_batch(
            batch_size=current,
            slots=slots,
            value_dim=value_dim,
            state_scale=state_scale,
            address_codebook=address_codebook,
            generator=generator,
            device=device,
        )
        output = model(batch, condition)
        error = (output.state - batch.target).square()
        row = torch.arange(current, device=device)
        erase_error = error[row, batch.erase_address].mean(dim=-1)
        write_error = error[row, batch.write_address].mean(dim=-1)
        erase_accuracy = (
            output.erase_address_weights.argmax(dim=-1) == batch.erase_address
        ).to(torch.float32)
        write_accuracy = (
            output.write_address_weights.argmax(dim=-1) == batch.write_address
        ).to(torch.float32)
        candidate_error = (
            output.erase_candidate - batch.old_candidate
        ).square().mean(dim=-1)
        old_residual = output.state[
            row,
            batch.erase_address,
        ].square().mean(dim=-1)
        unaffected = torch.ones(
            current,
            slots,
            dtype=torch.bool,
            device=device,
        )
        unaffected[row, batch.erase_address] = False
        unaffected[row, batch.write_address] = False

        totals["state_mse"] += float(error.mean()) * current
        totals["address_accuracy"] += float(
            (0.5 * (erase_accuracy + write_accuracy)).sum()
        )
        totals["candidate_recovery_mse"] += float(candidate_error.sum())
        totals["affected_mse"] += float(
            (0.5 * (erase_error + write_error)).sum()
        )
        totals["old_residual"] += float(old_residual.sum())
        if unaffected.any():
            retention_error = error[unaffected].mean(dim=-1)
            totals["retention_mse"] += float(retention_error.sum())
            retention_slots += int(unaffected.sum())
        count += current

    if count != episodes:
        raise RuntimeError("localization/candidate evaluation row count mismatch")
    return {
        "state_mse": totals["state_mse"] / count,
        "address_accuracy": totals["address_accuracy"] / count,
        "candidate_recovery_mse": totals["candidate_recovery_mse"] / count,
        "affected_mse": totals["affected_mse"] / count,
        "retention_mse": totals["retention_mse"] / max(retention_slots, 1),
        "old_residual": totals["old_residual"] / count,
    }
