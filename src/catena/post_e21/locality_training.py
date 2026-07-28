from __future__ import annotations

import hashlib
import math
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
    StructuredSequenceFreedom,
)
from catena.post_e21.locality_data import LocalityMethod, LocalityObjective
from catena.post_e21.locality_models import (
    LocalityStructuredSequenceController,
    ProtectedLocalityDiagnosticController,
)


@dataclass(frozen=True, slots=True)
class LocalityTrainResult:
    final_loss: float
    best_loss: float
    examples_per_second: float
    optimizer: str
    objective: str


def build_locality_controller(
    *,
    method: LocalityMethod,
    freedom: StructuredSequenceFreedom,
    slots: int,
    identifier_dim: int,
    value_dim: int,
    hidden_dim: int,
    address_temperature: float,
) -> MatchedStructuredSequenceController:
    controller_class = (
        ProtectedLocalityDiagnosticController
        if method.objective is LocalityObjective.PROTECTED_DIAGNOSTIC
        else LocalityStructuredSequenceController
    )
    return controller_class(
        freedom=freedom,
        slots=slots,
        identifier_dim=identifier_dim,
        value_dim=value_dim,
        hidden_dim=hidden_dim,
        address_temperature=address_temperature,
        active_fraction=(
            method.active_fraction if method.objective is LocalityObjective.SPARSE_ROUTE else None
        ),
    )


def upper_tail_mean(values: torch.Tensor, fraction: float) -> torch.Tensor:
    flattened = values.reshape(-1)
    if flattened.numel() == 0:
        return values.sum() * 0.0
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("tail fraction must lie in (0, 1]")
    count = max(1, math.ceil(float(fraction) * flattened.numel()))
    return flattened.topk(count, largest=True, sorted=False).values.mean()


def normalized_smooth_max(
    values: torch.Tensor,
    *,
    normalized_temperature: float,
    risk_scale: float,
) -> torch.Tensor:
    flattened = values.reshape(-1)
    if flattened.numel() == 0:
        return values.sum() * 0.0
    if normalized_temperature <= 0.0 or risk_scale <= 0.0:
        raise ValueError("smoothmax temperatures and risk scale must be positive")
    temperature = float(normalized_temperature) * float(risk_scale)
    log_count = math.log(flattened.numel())
    return temperature * (torch.logsumexp(flattened / temperature, dim=0) - log_count)


def locality_retention_risk(
    values: torch.Tensor,
    *,
    method: LocalityMethod,
    risk_scale: float,
) -> torch.Tensor:
    if values.numel() == 0:
        return values.sum() * 0.0
    if method.objective in {
        LocalityObjective.MEAN,
        LocalityObjective.PROTECTED_DIAGNOSTIC,
    }:
        return values.mean()
    if method.objective in {
        LocalityObjective.CVAR,
        LocalityObjective.SPARSE_ROUTE,
    }:
        if method.tail_fraction is None:
            raise ValueError("CVaR locality method lacks tail_fraction")
        return upper_tail_mean(values, method.tail_fraction)
    if method.objective is LocalityObjective.SMOOTH_MAX:
        if method.normalized_temperature is None:
            raise ValueError("smoothmax locality method lacks temperature")
        return normalized_smooth_max(
            values,
            normalized_temperature=method.normalized_temperature,
            risk_scale=risk_scale,
        )
    raise AssertionError(f"Unhandled E22 objective: {method.objective}")


def _address_nll(
    weights: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    selected = weights.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    return -torch.log(selected.clamp_min(1e-8))[mask].mean()


def _balanced_activity_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    losses = nn.functional.binary_cross_entropy_with_logits(
        logits,
        target.to(logits.dtype),
        reduction="none",
    )
    parts: list[torch.Tensor] = []
    if target.any():
        parts.append(losses[target].mean())
    if (~target).any():
        parts.append(losses[~target].mean())
    return torch.stack(parts).mean() if parts else losses.mean()


def _sparse_route_penalty(
    activity: torch.Tensor,
    update_mask: torch.Tensor,
    *,
    active_fraction: float,
) -> torch.Tensor:
    distractor_activity = activity[~update_mask]
    if distractor_activity.numel() == 0:
        return activity.sum() * 0.0
    return upper_tail_mean(distractor_activity, active_fraction)


def train_locality_controller(
    *,
    model: MatchedStructuredSequenceController,
    method: LocalityMethod,
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
    sparse_route_weight: float,
    risk_scale: float,
    train_namespace: str,
    distractor_namespace: str,
    device: torch.device,
    seed: int,
) -> LocalityTrainResult:
    if not conditions or not families:
        raise ValueError("E22 conditions and families must not be empty")
    if steps <= 0 or batch_size <= 0:
        raise ValueError("E22 steps and batch size must be positive")
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate))
    started = perf_counter()
    best = float("inf")
    final = float("nan")
    for step_index in range(int(steps)):
        family = families[step_index % len(families)]
        condition = conditions[(step_index // len(families)) % len(conditions)]
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
        retained_values = entity_error[~batch.affected_entities]
        retention = locality_retention_risk(
            retained_values,
            method=method,
            risk_scale=risk_scale,
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
            candidate_error = (output.erase_candidates - batch.old_candidates).square().mean(dim=-1)
            loss = loss + float(candidate_loss_weight) * candidate_error[batch.update_mask].mean()
        loss = loss + float(activity_loss_weight) * _balanced_activity_loss(
            output.raw_activity_logits,
            batch.update_mask,
        )
        if method.objective is LocalityObjective.SPARSE_ROUTE:
            if method.active_fraction is None:
                raise ValueError("Sparse locality method lacks active_fraction")
            loss = loss + float(sparse_route_weight) * _sparse_route_penalty(
                output.activity_gates,
                batch.update_mask,
                active_fraction=method.active_fraction,
            )
        loss = loss + 0.0 * output.raw_address_logits.sum() + 0.0 * output.raw_candidates.sum()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite E22 loss at step {step_index}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        final = float(loss.detach().cpu())
        best = min(best, final)
    elapsed = max(perf_counter() - started, 1e-8)
    return LocalityTrainResult(
        final_loss=final,
        best_loss=best,
        examples_per_second=float(steps * batch_size / elapsed),
        optimizer="AdamW",
        objective=method.objective.value,
    )


@torch.no_grad()
def evaluate_locality_controller(
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
    route_weight_threshold: float,
    activity_threshold: float,
    device: torch.device,
    seed: int,
) -> dict[str, float | int | str]:
    """Evaluate one registered cell with route support and update geometry."""

    if not 0.0 < route_weight_threshold <= 1.0:
        raise ValueError("route_weight_threshold must lie in (0, 1]")
    if not 0.0 < activity_threshold < 1.0:
        raise ValueError("activity_threshold must lie in (0, 1)")
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
        "route_support": 0.0,
        "route_capacity": 0.0,
        "active_events": 0.0,
        "event_count": 0.0,
        "predicted_update_square": 0.0,
        "target_update_square": 0.0,
        "update_element_count": 0.0,
        "update_compute": 0.0,
    }
    all_entities = 0
    affected_entities = 0
    retained_entities = 0
    verified_events = 0
    distractor_events = 0
    transaction_digest = hashlib.sha256()
    raw_route_digest = hashlib.sha256()
    active_route_digest = hashlib.sha256()
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
        transaction_digest.update(bytes.fromhex(structured_base_transaction_digest(batch)))
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
        erase_correct = (output.erase_address_weights.argmax(dim=-1) == batch.erase_addresses).to(
            torch.float32
        )
        write_correct = (output.write_address_weights.argmax(dim=-1) == batch.write_addresses).to(
            torch.float32
        )
        totals["address_correct"] += float((0.5 * (erase_correct + write_correct))[verified].sum())
        candidate_error = (output.erase_candidates - batch.old_candidates).square().mean(dim=-1)
        totals["candidate_error"] += float(candidate_error[verified].sum())
        totals["verified_activity"] += float(output.activity_gates[verified].sum())
        totals["distractor_activity"] += float(output.activity_gates[distractor].sum())
        verified_events += int(verified.sum())
        distractor_events += int(distractor.sum())

        active_event = output.activity_gates >= float(activity_threshold)
        if hasattr(output, "applied_route_mask"):
            raw_route_mask = output.applied_route_mask
        else:
            raw_route_mask = torch.stack(
                (
                    output.erase_address_weights >= float(route_weight_threshold),
                    output.write_address_weights >= float(route_weight_threshold),
                ),
                dim=2,
            )
        active_route_mask = raw_route_mask & active_event[:, :, None, None]
        raw_route_cpu = raw_route_mask.detach().cpu().contiguous()
        active_route_cpu = active_route_mask.detach().cpu().contiguous()
        raw_route_digest.update(str(tuple(raw_route_cpu.shape)).encode("utf-8"))
        raw_route_digest.update(raw_route_cpu.numpy().tobytes())
        active_route_digest.update(str(tuple(active_route_cpu.shape)).encode("utf-8"))
        active_route_digest.update(active_route_cpu.numpy().tobytes())
        raw_support = int(raw_route_mask.sum())
        support = int(active_route_mask.sum())
        event_count = int(active_event.numel())
        active_count = int(active_event.sum())
        totals.setdefault("raw_route_support", 0.0)
        totals["raw_route_support"] += raw_support
        totals["route_support"] += support
        totals["route_capacity"] += int(raw_route_mask.numel())
        totals["active_events"] += active_count
        totals["event_count"] += event_count
        totals["update_compute"] += float(raw_support * value_dim + event_count * 2 * value_dim)

        predicted_delta = output.state - batch.inputs.initial_state
        target_delta = batch.target_state - batch.inputs.initial_state
        applied_update_deltas = getattr(
            output,
            "applied_update_deltas",
            predicted_delta[:, None],
        )
        totals.setdefault("post_mask_update_square", 0.0)
        totals.setdefault("post_mask_update_count", 0.0)
        totals["post_mask_update_square"] += float(applied_update_deltas.square().sum())
        totals["post_mask_update_count"] += int(applied_update_deltas.numel())
        totals["predicted_update_square"] += float(predicted_delta.square().sum())
        totals["target_update_square"] += float(target_delta.square().sum())
        totals["update_element_count"] += int(predicted_delta.numel())

    event_denominator = max(totals["event_count"], 1.0)
    route_capacity = max(totals["route_capacity"], 1.0)
    update_elements = max(totals["update_element_count"], 1.0)
    sequence_examples = max(int(batches) * int(batch_size), 1)
    return {
        "state_mse": totals["state_error"] / max(all_entities, 1),
        "affected_mse": totals["affected_error"] / max(affected_entities, 1),
        "retention_mse": totals["retention_error"] / max(retained_entities, 1),
        "address_accuracy": totals["address_correct"] / max(verified_events, 1),
        "candidate_recovery_mse": totals["candidate_error"] / max(verified_events, 1),
        "verified_activity_mean": totals["verified_activity"] / max(verified_events, 1),
        "distractor_activity_mean": totals["distractor_activity"] / max(distractor_events, 1),
        "verified_event_count": verified_events,
        "distractor_event_count": distractor_events,
        "affected_entity_count": affected_entities,
        "retained_entity_count": retained_entities,
        "base_transaction_digest": transaction_digest.hexdigest(),
        "raw_route_mask_sha256": raw_route_digest.hexdigest(),
        "active_route_mask_sha256": active_route_digest.hexdigest(),
        "route_mask_sha256": active_route_digest.hexdigest(),
        "raw_route_support_size": totals["raw_route_support"] / event_denominator,
        "raw_route_support_fraction": totals["raw_route_support"] / route_capacity,
        "active_route_support_size": totals["route_support"] / event_denominator,
        "active_route_support_fraction": totals["route_support"] / route_capacity,
        "active_event_fraction": totals["active_events"] / event_denominator,
        "predicted_update_rms": math.sqrt(totals["predicted_update_square"] / update_elements),
        "post_mask_update_rms": math.sqrt(
            totals["post_mask_update_square"] / max(totals["post_mask_update_count"], 1.0)
        ),
        "target_update_rms": math.sqrt(totals["target_update_square"] / update_elements),
        "update_compute_units": totals["update_compute"] / sequence_examples,
        "route_weight_threshold": float(route_weight_threshold),
        "activity_support_threshold": float(activity_threshold),
    }
