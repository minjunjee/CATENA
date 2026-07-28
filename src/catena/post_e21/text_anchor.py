"""Shared frozen text encoder and matched controller family for E25b."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, cast

import numpy as np
import torch
from torch import nn

from catena.post_e21.text_transactions import (
    MagnitudeOperation,
    OldRuleStatus,
    TextDemand,
    TextTransaction,
    decode_visible_policy_candidate,
    tokenize,
)


class TextController(StrEnum):
    TIED = "tied"
    DUAL = "dual"
    DIAGONAL = "diagonal"
    SEPARATE_ADDRESS = "separate_address"
    STATE_AWARE = "state_aware"


@dataclass(slots=True)
class TextControllerOutput:
    state: torch.Tensor
    erase_address: torch.Tensor
    write_address: torch.Tensor
    erase_gate: torch.Tensor
    write_gate: torch.Tensor
    candidate: torch.Tensor
    incoming: torch.Tensor


class FrozenHashNgramEncoder(nn.Module):
    """Deterministic, non-pretrained text encoder shared by all controllers."""

    embedding_table: torch.Tensor

    def __init__(
        self,
        *,
        output_dim: int,
        buckets: int,
        ngram_min: int,
        ngram_max: int,
        seed: int,
    ) -> None:
        super().__init__()
        if output_dim < 8 or buckets < 64 or not 1 <= ngram_min <= ngram_max:
            raise ValueError("invalid frozen hash encoder dimensions")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        table = torch.randn(buckets, output_dim, generator=generator)
        table = table / table.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
        self.register_buffer("embedding_table", table, persistent=True)
        self.output_dim = output_dim
        self.buckets = buckets
        self.ngram_min = ngram_min
        self.ngram_max = ngram_max
        self.seed = seed
        self._feature_cache: dict[str, torch.Tensor] = {}

    def _features(self, text: str) -> torch.Tensor:
        cached = self._feature_cache.get(text)
        if cached is not None and cached.device == self.embedding_table.device:
            return cached
        tokens = tokenize(text)
        indices: list[int] = []
        signs: list[float] = []
        for width in range(self.ngram_min, self.ngram_max + 1):
            for start in range(max(0, len(tokens) - width + 1)):
                gram = "\x1f".join(tokens[start : start + width])
                digest = hashlib.sha256(f"{self.seed}:{width}:{gram}".encode()).digest()
                indices.append(int.from_bytes(digest[:8], "big") % self.buckets)
                signs.append(1.0 if digest[8] & 1 else -1.0)
        if not indices:
            return self.embedding_table.new_zeros(self.output_dim)
        selected = self.embedding_table[torch.tensor(indices, device=self.embedding_table.device)]
        sign = self.embedding_table.new_tensor(signs).unsqueeze(-1)
        pooled = (selected * sign).sum(dim=0) / max(1.0, float(len(indices)) ** 0.5)
        normalized = cast(torch.Tensor, pooled / pooled.norm().clamp_min(1.0e-8))
        self._feature_cache[text] = normalized
        return normalized

    def forward(self, texts: Sequence[str]) -> torch.Tensor:
        return torch.stack([self._features(text) for text in texts], dim=0)

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(
            (
                f"{self.seed}:{self.output_dim}:{self.buckets}:{self.ngram_min}:{self.ngram_max}"
            ).encode()
        )
        digest.update(self.embedding_table.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()


class MatchedTextTransactionController(nn.Module):
    """One maximal parameter surface with variant-specific reachable projections."""

    def __init__(
        self,
        *,
        variant: TextController,
        encoder: FrozenHashNgramEncoder,
        slots: int,
        value_dim: int,
        hidden_dim: int,
        semantic_value_seed: int,
    ) -> None:
        super().__init__()
        self.variant = variant
        self.encoder = encoder
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.slots = slots
        self.value_dim = value_dim
        self.semantic_value_seed = semantic_value_seed
        self.backbone = nn.Sequential(
            nn.Linear(encoder.output_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.erase_gate_head = nn.Linear(hidden_dim, value_dim)
        self.write_gate_head = nn.Linear(hidden_dim, value_dim)
        self.erase_address_head = nn.Linear(hidden_dim, encoder.output_dim)
        self.write_address_head = nn.Linear(hidden_dim, encoder.output_dim)
        self.candidate_head = nn.Linear(hidden_dim, value_dim)
        self.state_context_head = nn.Linear(value_dim, hidden_dim)

    def forward(
        self,
        *,
        texts: Sequence[str],
        state: torch.Tensor,
        memory_entities: Sequence[Sequence[str]],
    ) -> TextControllerOutput:
        if len(texts) != int(state.shape[0]) or len(memory_entities) != len(texts):
            raise ValueError("text, state, and memory-entity batches must align")
        if any(len(entities) != self.slots for entities in memory_entities):
            raise ValueError("each memory-entity row must match the registered slot count")
        features = self.encoder(texts).to(device=state.device, dtype=state.dtype)
        flattened_entities = [entity for entities in memory_entities for entity in entities]
        memory_keys = self.encoder(flattened_entities).to(
            device=state.device,
            dtype=state.dtype,
        )
        memory_keys = memory_keys.view(
            len(texts),
            self.slots,
            self.encoder.output_dim,
        )
        text_hidden = self.backbone(features)
        preliminary_erase_query = self.erase_address_head(text_hidden)
        preliminary_erase_logits = (
            torch.einsum(
                "bd,bsd->bs",
                preliminary_erase_query,
                memory_keys,
            )
            / float(self.encoder.output_dim) ** 0.5
        )
        preliminary_erase_address = torch.softmax(
            preliminary_erase_logits,
            dim=-1,
        )
        preliminary_candidate = torch.einsum(
            "bs,bsv->bv",
            preliminary_erase_address,
            state,
        )
        hidden = (
            text_hidden + torch.tanh(self.state_context_head(preliminary_candidate))
            if self.variant is TextController.STATE_AWARE
            else text_hidden
        )
        erase_vector = torch.sigmoid(self.erase_gate_head(hidden))
        write_vector = torch.sigmoid(self.write_gate_head(hidden))
        erase_query = self.erase_address_head(hidden)
        write_query = self.write_address_head(hidden)
        erase_logits = (
            torch.einsum(
                "bd,bsd->bs",
                erase_query,
                memory_keys,
            )
            / float(self.encoder.output_dim) ** 0.5
        )
        write_logits = (
            torch.einsum(
                "bd,bsd->bs",
                write_query,
                memory_keys,
            )
            / float(self.encoder.output_dim) ** 0.5
        )

        if self.variant is TextController.TIED:
            scalar = 0.5 * (
                erase_vector.mean(dim=-1, keepdim=True) + write_vector.mean(dim=-1, keepdim=True)
            )
            erase_gate = scalar.expand_as(erase_vector)
            write_gate = scalar.expand_as(write_vector)
        elif self.variant is TextController.DUAL:
            erase_gate = erase_vector.mean(dim=-1, keepdim=True).expand_as(erase_vector)
            write_gate = write_vector.mean(dim=-1, keepdim=True).expand_as(write_vector)
        else:
            erase_gate = erase_vector
            write_gate = write_vector

        if self.variant in {
            TextController.TIED,
            TextController.DUAL,
            TextController.DIAGONAL,
        }:
            shared_logits = 0.5 * (erase_logits + write_logits)
            erase_address = torch.softmax(shared_logits, dim=-1)
            write_address = erase_address
        else:
            erase_address = torch.softmax(erase_logits, dim=-1)
            write_address = torch.softmax(write_logits, dim=-1)

        predicted_candidate = self.candidate_head(hidden)
        state_read_candidate = torch.einsum("bs,bsv->bv", erase_address, state)
        # Candidate recovery is common to the controller family.  The
        # state-aware axis changes whether that read can condition the control
        # decision, not whether the old content can be removed at all.
        candidate = state_read_candidate + 0.0 * predicted_candidate
        # Every controller sees exactly the same B decoded by a registered,
        # deterministic map from the visible policy token.  No controller-
        # specific learned head can suppress or redefine this candidate.
        incoming = torch.stack(
            [
                decode_visible_policy_candidate(
                    text,
                    dimension=self.value_dim,
                    semantic_value_seed=self.semantic_value_seed,
                )
                for text in texts
            ]
        ).to(device=state.device, dtype=state.dtype)
        erase_update = torch.einsum(
            "bs,bv->bsv",
            erase_address,
            erase_gate * candidate,
        )
        write_update = torch.einsum(
            "bs,bv->bsv",
            write_address,
            write_gate * incoming,
        )
        return TextControllerOutput(
            state=state - erase_update + write_update,
            erase_address=erase_address,
            write_address=write_address,
            erase_gate=erase_gate,
            write_gate=write_gate,
            candidate=candidate,
            incoming=incoming,
        )


def matched_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _batch(
    examples: Sequence[TextTransaction],
    indices: Sequence[int],
    *,
    device: torch.device,
) -> tuple[list[str], list[tuple[str, ...]], torch.Tensor, torch.Tensor]:
    selected = [examples[index] for index in indices]
    texts = [example.text for example in selected]
    memory_entities = [example.memory_entities for example in selected]
    state = torch.stack([example.state for example in selected]).to(device)
    target = torch.stack([example.target_state for example in selected]).to(device)
    return texts, memory_entities, state, target


def train_text_controller(
    model: MatchedTextTransactionController,
    examples: Sequence[TextTransaction],
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    affected_weight: float,
    retention_weight: float,
    gradient_clip_norm: float,
    seed: int,
    device: torch.device,
) -> list[dict[str, float | int]]:
    if not examples:
        raise ValueError("training examples cannot be empty")
    model.to(device).train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    history: list[dict[str, float | int]] = []
    for step in range(steps):
        indices = torch.randint(
            len(examples),
            (batch_size,),
            generator=generator,
        ).tolist()
        texts, memory_entities, state, target = _batch(
            examples,
            indices,
            device=device,
        )
        output = model(
            texts=texts,
            state=state,
            memory_entities=memory_entities,
        )
        changed = target.ne(state).any(dim=-1)
        affected = (output.state - target).square().mean(dim=-1)
        affected_loss = affected[changed].mean() if bool(changed.any()) else affected.mean()
        retention_loss = (
            affected[~changed].mean() if bool((~changed).any()) else affected.new_zeros(())
        )
        loss = affected_weight * affected_loss + retention_weight * retention_loss
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("E25b training loss is non-finite")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        if step == 0 or step + 1 == steps:
            history.append(
                {
                    "step": step + 1,
                    "loss": float(loss.detach().cpu().item()),
                    "affected_loss": float(affected_loss.detach().cpu().item()),
                    "retention_loss": float(retention_loss.detach().cpu().item()),
                }
            )
    return history


def _evaluate_output(
    *,
    examples: Sequence[TextTransaction],
    output: TextControllerOutput,
    condition: str,
    seed: int,
    accuracy_mse_threshold: float,
) -> list[dict[str, Any]]:
    predicted = output.state.detach().cpu()
    erase_address = output.erase_address.detach().cpu()
    write_address = output.write_address.detach().cpu()
    rows: list[dict[str, Any]] = []
    for index, example in enumerate(examples):
        changed = example.target_state.ne(example.state).any(dim=-1)
        squared = (predicted[index] - example.target_state).square().mean(dim=-1)
        affected_mse = float(squared[changed].mean().item()) if bool(changed.any()) else 0.0
        unaffected_mse = float(squared[~changed].mean().item()) if bool((~changed).any()) else 0.0
        direct_index = (
            example.write_index if example.demand.value == "address" else example.erase_index
        )
        target_row = predicted[index, direct_index]
        direct_fact_mse = float(
            torch.square(target_row - example.target_state[direct_index]).mean().item()
        )
        old_row = predicted[index, example.erase_index]
        old_residual = float(
            torch.square(
                torch.dot(old_row, example.old_value)
                - torch.dot(example.target_state[example.erase_index], example.old_value)
            ).item()
        )
        old_axis = (
            example.old_value
            - torch.dot(
                example.old_value,
                example.new_value,
            )
            * example.new_value
        )
        old_component_coefficient = float(
            (torch.dot(old_row, old_axis) / torch.dot(old_axis, old_axis).clamp_min(1.0e-8)).item()
        )
        predicted_old_rule_status = _classify_old_rule_status(
            example=example,
            predicted_old_row=old_row,
        )
        rows.append(
            {
                "seed": seed,
                "example_id": example.example_id,
                "split": example.split.value,
                "demand_family": example.demand.value,
                "magnitude_operation": example.magnitude_operation,
                "condition": condition,
                "affected_correction_mse": affected_mse,
                "unaffected_retention_mse": unaffected_mse,
                "old_rule_residual": old_residual,
                "old_rule_component_coefficient": old_component_coefficient,
                "gold_old_rule_status": example.old_rule_status.value,
                "predicted_old_rule_status": predicted_old_rule_status.value,
                "direct_fact_mse": direct_fact_mse,
                "direct_fact_accuracy": float(direct_fact_mse <= float(accuracy_mse_threshold)),
                "derived_action_accuracy": float(
                    int(target_row.argmax().item() % 4) == example.derived_action
                ),
                "old_rule_accuracy": float(predicted_old_rule_status is example.old_rule_status),
                "erase_address_accuracy": float(
                    int(erase_address[index].argmax().item()) == example.erase_index
                ),
                "write_address_accuracy": float(
                    int(write_address[index].argmax().item()) == example.write_index
                ),
            }
        )
    return rows


def _old_rule_status_prototypes(
    example: TextTransaction,
) -> dict[OldRuleStatus, torch.Tensor]:
    """Construct private demand-aware status prototypes at the erase address."""

    old = example.old_value.detach().cpu()
    incoming = example.new_value.detach().cpu()
    if example.demand is TextDemand.MAGNITUDE:
        operation = MagnitudeOperation(example.magnitude_operation)
        candidate = (
            incoming
            if operation in {MagnitudeOperation.ADD, MagnitudeOperation.SUPERSEDE}
            else torch.zeros_like(old)
        )
        partial_old = 0.5 * old
    elif example.demand is TextDemand.VALUE:
        candidate = example.coordinate_mask.detach().cpu() * incoming
        partial_old = (1.0 - example.coordinate_mask.detach().cpu()) * old
    elif example.demand is TextDemand.ADDRESS:
        candidate = torch.zeros_like(old)
        partial_old = 0.5 * old
    else:
        candidate = incoming if example.active else torch.zeros_like(old)
        partial_old = 0.5 * old
    return {
        OldRuleStatus.FULL: candidate + old,
        OldRuleStatus.PARTIAL: candidate + partial_old,
        OldRuleStatus.NONE: candidate,
    }


def _classify_old_rule_status(
    *,
    example: TextTransaction,
    predicted_old_row: torch.Tensor,
) -> OldRuleStatus:
    """Classify old-rule status by nearest registered private prototype."""

    predicted = predicted_old_row.detach().cpu()
    prototypes = _old_rule_status_prototypes(example)
    ordered = (OldRuleStatus.FULL, OldRuleStatus.PARTIAL, OldRuleStatus.NONE)
    return min(
        ordered,
        key=lambda status: float(torch.square(predicted - prototypes[status]).mean().item()),
    )


def evaluate_text_controller(
    model: MatchedTextTransactionController,
    examples: Sequence[TextTransaction],
    *,
    seed: int,
    device: torch.device,
    batch_size: int,
    accuracy_mse_threshold: float,
    text_overrides: Mapping[str, str] | None = None,
    zero_state: bool = False,
    state_only: bool = False,
    condition: str | None = None,
) -> list[dict[str, Any]]:
    model.to(device).eval()
    rows: list[dict[str, Any]] = []
    label = model.variant.value if condition is None else condition
    for start in range(0, len(examples), batch_size):
        chunk = list(examples[start : start + batch_size])
        texts = [
            (
                "A current record is available."
                if state_only
                else (
                    text_overrides.get(example.example_id, example.text)
                    if text_overrides is not None
                    else example.text
                )
            )
            for example in chunk
        ]
        state = torch.stack([example.state for example in chunk]).to(device)
        memory_entities = [example.memory_entities for example in chunk]
        if zero_state:
            state = torch.zeros_like(state)
        with torch.no_grad():
            output = model(
                texts=texts,
                state=state,
                memory_entities=memory_entities,
            )
        rows.extend(
            _evaluate_output(
                examples=chunk,
                output=output,
                condition=label,
                seed=seed,
                accuracy_mse_threshold=accuracy_mse_threshold,
            )
        )
    return rows


def oracle_rows(
    examples: Sequence[TextTransaction],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    if not examples:
        return []
    states = torch.stack([example.target_state for example in examples])
    slots = int(states.shape[1])
    erase_address = torch.zeros(len(examples), slots, dtype=states.dtype)
    write_address = torch.zeros_like(erase_address)
    for index, example in enumerate(examples):
        erase_address[index, example.erase_index] = 1.0
        write_address[index, example.write_index] = 1.0
    output = TextControllerOutput(
        state=states,
        erase_address=erase_address,
        write_address=write_address,
        erase_gate=torch.zeros_like(states[:, 0]),
        write_gate=torch.zeros_like(states[:, 0]),
        candidate=torch.stack([example.old_value for example in examples]),
        incoming=torch.stack([example.new_value for example in examples]),
    )
    rows = _evaluate_output(
        examples=examples,
        output=output,
        condition="oracle_demand",
        seed=seed,
        accuracy_mse_threshold=1.0e-12,
    )
    for row, example in zip(rows, examples, strict=True):
        changed = example.target_state.ne(example.state).any(dim=-1)
        row["changed_slots"] = int(changed.sum().item())
    return rows


def summarize_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    minimum_identifiable_oracle_headroom: float,
) -> list[dict[str, Any]]:
    if minimum_identifiable_oracle_headroom < 0.0:
        raise ValueError("minimum identifiable oracle headroom cannot be negative")
    buckets: dict[tuple[int, str, str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            int(row["seed"]),
            str(row["split"]),
            str(row["demand_family"]),
            str(row["magnitude_operation"]),
            str(row["condition"]),
        )
        buckets.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    metric_names = (
        "affected_correction_mse",
        "unaffected_retention_mse",
        "old_rule_residual",
        "old_rule_component_coefficient",
        "direct_fact_mse",
        "direct_fact_accuracy",
        "derived_action_accuracy",
        "old_rule_accuracy",
        "erase_address_accuracy",
        "write_address_accuracy",
    )
    for aggregate_key, bucket in sorted(buckets.items()):
        summary: dict[str, Any] = {
            "seed": aggregate_key[0],
            "split": aggregate_key[1],
            "demand_family": aggregate_key[2],
            "magnitude_operation": aggregate_key[3],
            "condition": aggregate_key[4],
            "episodes": len(bucket),
        }
        for metric in metric_names:
            summary[metric] = float(np.mean([float(row[metric]) for row in bucket]))
        output.append(summary)
    index = {
        (
            int(row["seed"]),
            str(row["split"]),
            str(row["demand_family"]),
            str(row["magnitude_operation"]),
            str(row["condition"]),
        ): row
        for row in output
    }
    for row in output:
        measurement_key = (
            int(row["seed"]),
            str(row["split"]),
            str(row["demand_family"]),
            str(row["magnitude_operation"]),
        )
        tied = index.get((*measurement_key, TextController.TIED.value))
        oracle = index.get((*measurement_key, "oracle_demand"))
        if tied is None or oracle is None:
            row["oracle_headroom_identifiable"] = False
            row["oracle_headroom_normalized_recovery"] = None
            continue
        tied_error = float(tied["affected_correction_mse"])
        oracle_error = float(oracle["affected_correction_mse"])
        headroom = tied_error - oracle_error
        row["oracle_headroom_identifiable"] = headroom > minimum_identifiable_oracle_headroom
        row["oracle_headroom_normalized_recovery"] = (
            (tied_error - float(row["affected_correction_mse"])) / headroom
            if headroom > minimum_identifiable_oracle_headroom
            else None
        )
    return output
