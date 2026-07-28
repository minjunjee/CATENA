from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter

import torch
from torch import nn

from catena.data.structured_sequence_localization import (
    StructuredTransferCondition,
    StructuredTransferDemand,
    generate_structured_sequence_transfer_batch,
    indexed_structured_sequence_seed,
    structured_base_transaction_digest,
)
from catena.models.structured_sequence_localization import (
    MatchedStructuredSequenceController,
)


@dataclass(slots=True)
class StructuredSequenceTrainResult:
    final_loss: float
    best_loss: float
    examples_per_second: float
    optimizer: str


def structured_state_dict_sha256(
    state_dict: Mapping[str, torch.Tensor],
) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state_dict.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(repr(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _address_nll(
    weights: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    probabilities = weights.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    selected = probabilities[mask]
    if selected.numel() == 0:
        return weights.sum() * 0.0
    return -selected.clamp_min(1e-8).log().mean()


def _balanced_activity_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    losses = nn.functional.binary_cross_entropy_with_logits(
        logits,
        target.to(logits.dtype),
        reduction="none",
    )
    positive = target
    negative = ~target
    parts: list[torch.Tensor] = []
    if positive.any():
        parts.append(losses[positive].mean())
    if negative.any():
        parts.append(losses[negative].mean())
    if not parts:
        return losses.mean()
    return torch.stack(parts).mean()


def train_structured_sequence_controller(
    *,
    model: MatchedStructuredSequenceController,
    conditions: list[StructuredTransferCondition],
    families: list[StructuredTransferDemand],
    steps: int,
    batch_size: int,
    slots: int,
    value_dim: int,
    updates: int,
    gap_events: int,
    state_scale: float,
    identifier_codebook: torch.Tensor,
    learning_rate: float,
    address_loss_weight: float,
    candidate_loss_weight: float,
    activity_loss_weight: float,
    retention_weight: float,
    train_namespace: str,
    distractor_namespace: str,
    device: torch.device,
    seed: int,
) -> StructuredSequenceTrainResult:
    if not conditions or not families:
        raise ValueError("conditions and families must not be empty")
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate))
    started = perf_counter()
    best = float("inf")
    final = float("nan")
    for step_index in range(int(steps)):
        family = families[step_index % len(families)]
        condition = conditions[
            (step_index // len(families)) % len(conditions)
        ]
        batch = generate_structured_sequence_transfer_batch(
            family=family,
            batch_size=batch_size,
            slots=slots,
            value_dim=value_dim,
            updates=updates,
            gap_events=gap_events,
            state_scale=state_scale,
            identifier_codebook=identifier_codebook,
            seed=indexed_structured_sequence_seed(
                seed,
                train_namespace,
                step_index,
            ),
            base_namespace=train_namespace,
            distractor_namespace=distractor_namespace,
            device=device,
        )
        output = model(batch, condition)
        entity_error = (output.state - batch.target_state).square().mean(dim=-1)
        affected = (
            entity_error[batch.affected_entities].mean()
            if batch.affected_entities.any()
            else entity_error.mean()
        )
        unaffected_mask = ~batch.affected_entities
        retention = (
            entity_error[unaffected_mask].mean()
            if unaffected_mask.any()
            else entity_error.mean() * 0.0
        )
        loss = affected + float(retention_weight) * retention
        if not condition.uses_oracle_address:
            loss = loss + float(address_loss_weight) * (
                _address_nll(
                    output.erase_address_weights,
                    batch.erase_addresses,
                    batch.update_mask,
                )
                + _address_nll(
                    output.write_address_weights,
                    batch.write_addresses,
                    batch.update_mask,
                )
            )
        if not condition.uses_oracle_candidate:
            candidate_error = (
                output.erase_candidates - batch.old_candidates
            ).square().mean(dim=-1)
            loss = loss + float(candidate_loss_weight) * candidate_error[
                batch.update_mask
            ].mean()
        loss = loss + float(activity_loss_weight) * _balanced_activity_loss(
            output.raw_activity_logits,
            batch.update_mask,
        )
        # Preserve a valid paired optimizer step when an oracle route bypasses
        # one of the registered maximal heads.
        loss = (
            loss
            + 0.0 * output.raw_address_logits.sum()
            + 0.0 * output.raw_candidates.sum()
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"non-finite E21 loss at training step {step_index}"
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        final = float(loss.detach().cpu())
        best = min(best, final)
    elapsed = max(perf_counter() - started, 1e-8)
    return StructuredSequenceTrainResult(
        final_loss=final,
        best_loss=best,
        examples_per_second=float(steps * batch_size / elapsed),
        optimizer="AdamW",
    )


@torch.no_grad()
def evaluate_structured_sequence_controller(
    *,
    model: MatchedStructuredSequenceController,
    condition: StructuredTransferCondition,
    family: StructuredTransferDemand,
    batches: int,
    batch_size: int,
    slots: int,
    value_dim: int,
    updates: int,
    gap_events: int,
    state_scale: float,
    identifier_codebook: torch.Tensor,
    evaluation_namespace: str,
    distractor_namespace: str,
    device: torch.device,
    seed: int,
) -> dict[str, float | int | str]:
    model.to(device)
    model.eval()
    totals = {
        "state_error": 0.0,
        "affected_error": 0.0,
        "retention_error": 0.0,
        "address_correct": 0.0,
        "candidate_error": 0.0,
        "verified_activity": 0.0,
        "distractor_activity": 0.0,
    }
    all_entities = 0
    affected_entities = 0
    retained_entities = 0
    verified_events = 0
    distractor_events = 0
    digest = hashlib.sha256()
    for batch_index in range(int(batches)):
        batch = generate_structured_sequence_transfer_batch(
            family=family,
            batch_size=batch_size,
            slots=slots,
            value_dim=value_dim,
            updates=updates,
            gap_events=gap_events,
            state_scale=state_scale,
            identifier_codebook=identifier_codebook,
            seed=indexed_structured_sequence_seed(
                seed,
                evaluation_namespace,
                batch_index,
            ),
            base_namespace=evaluation_namespace,
            distractor_namespace=distractor_namespace,
            device=device,
        )
        digest.update(bytes.fromhex(structured_base_transaction_digest(batch)))
        output = model(batch, condition)
        entity_error = (output.state - batch.target_state).square().mean(dim=-1)
        affected = batch.affected_entities
        retained = ~affected
        totals["state_error"] += float(entity_error.sum())
        totals["affected_error"] += float(entity_error[affected].sum())
        totals["retention_error"] += float(entity_error[retained].sum())
        all_entities += int(entity_error.numel())
        affected_entities += int(affected.sum())
        retained_entities += int(retained.sum())

        verified = batch.update_mask
        distractor = ~verified
        erase_correct = (
            output.erase_address_weights.argmax(dim=-1)
            == batch.erase_addresses
        ).to(torch.float32)
        write_correct = (
            output.write_address_weights.argmax(dim=-1)
            == batch.write_addresses
        ).to(torch.float32)
        totals["address_correct"] += float(
            (0.5 * (erase_correct + write_correct))[verified].sum()
        )
        candidate_error = (
            output.erase_candidates - batch.old_candidates
        ).square().mean(dim=-1)
        totals["candidate_error"] += float(candidate_error[verified].sum())
        totals["verified_activity"] += float(
            output.activity_gates[verified].sum()
        )
        totals["distractor_activity"] += float(
            output.activity_gates[distractor].sum()
        )
        verified_events += int(verified.sum())
        distractor_events += int(distractor.sum())

    return {
        "state_mse": totals["state_error"] / max(all_entities, 1),
        "affected_mse": totals["affected_error"]
        / max(affected_entities, 1),
        "retention_mse": totals["retention_error"]
        / max(retained_entities, 1),
        "address_accuracy": totals["address_correct"]
        / max(verified_events, 1),
        "candidate_recovery_mse": totals["candidate_error"]
        / max(verified_events, 1),
        "verified_activity_mean": totals["verified_activity"]
        / max(verified_events, 1),
        "distractor_activity_mean": totals["distractor_activity"]
        / max(distractor_events, 1),
        "verified_event_count": verified_events,
        "distractor_event_count": distractor_events,
        "affected_entity_count": affected_entities,
        "retained_entity_count": retained_entities,
        "base_transaction_digest": digest.hexdigest(),
    }
