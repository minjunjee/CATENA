from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from catena.data.semantic_controls_v61 import (
    ControlPairingRegistry,
    SemanticControl,
    build_control_view,
)
from catena.data.semantic_transactions_v61 import SemanticExample
from catena.eval.semantic_anchor_v61 import SemanticAnchorSeedMetrics
from catena.models.semantic_controllers_v61 import (
    MatchedSemanticControllerV61,
    SemanticRoute,
    assert_matched_semantic_pair,
)
from catena.models.semantic_encoder_v61 import FrozenSemanticFieldEncoderV61


@dataclass(frozen=True, slots=True)
class SemanticTrainingConfigV61:
    steps: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    affected_read_weight: float
    unaffected_retention_weight: float
    target_state_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.steps <= 0 or self.batch_size <= 0:
            raise ValueError("steps and batch_size must be positive.")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("optimizer settings are invalid.")
        if self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive.")
        if (
            self.affected_read_weight <= 0.0
            or self.unaffected_retention_weight <= 0.0
        ):
            raise ValueError("Both behavioral loss weights must be positive.")
        if self.target_state_weight != 0.0:
            raise ValueError("E05 behavioral-only training requires state weight zero.")


@dataclass(frozen=True, slots=True)
class BatchedVisibleUpdateContext:
    features: torch.Tensor
    visible_state: torch.Tensor
    visible_address: torch.Tensor
    incoming_value: torch.Tensor
    erase_candidate_scale: torch.Tensor
    write_candidate_scale: torch.Tensor

    @property
    def size(self) -> int:
        return int(self.features.shape[0])

    def to(self, device: torch.device | str) -> BatchedVisibleUpdateContext:
        return BatchedVisibleUpdateContext(
            features=self.features.to(device),
            visible_state=self.visible_state.to(device),
            visible_address=self.visible_address.to(device),
            incoming_value=self.incoming_value.to(device),
            erase_candidate_scale=self.erase_candidate_scale.to(device),
            write_candidate_scale=self.write_candidate_scale.to(device),
        )

    def select(self, indices: torch.Tensor) -> BatchedVisibleUpdateContext:
        return BatchedVisibleUpdateContext(
            features=self.features[indices],
            visible_state=self.visible_state[indices],
            visible_address=self.visible_address[indices],
            incoming_value=self.incoming_value[indices],
            erase_candidate_scale=self.erase_candidate_scale[indices],
            write_candidate_scale=self.write_candidate_scale[indices],
        )


@dataclass(frozen=True, slots=True)
class SemanticTensorBatch:
    visible: BatchedVisibleUpdateContext
    score_keys: torch.Tensor
    affected_index: torch.Tensor
    target_state: torch.Tensor
    original_state: torch.Tensor
    operation_demand: torch.Tensor
    example_ids: tuple[str, ...]
    domains: tuple[str, ...]
    templates: tuple[str, ...]
    operations: tuple[str, ...]

    @property
    def size(self) -> int:
        return len(self.example_ids)

    def to(self, device: torch.device | str) -> SemanticTensorBatch:
        return SemanticTensorBatch(
            visible=self.visible.to(device),
            score_keys=self.score_keys.to(device),
            affected_index=self.affected_index.to(device),
            target_state=self.target_state.to(device),
            original_state=self.original_state.to(device),
            operation_demand=self.operation_demand.to(device),
            example_ids=self.example_ids,
            domains=self.domains,
            templates=self.templates,
            operations=self.operations,
        )

    def select(self, indices: torch.Tensor) -> SemanticTensorBatch:
        local = indices.detach().cpu().tolist()
        return SemanticTensorBatch(
            visible=self.visible.select(indices),
            score_keys=self.score_keys[indices],
            affected_index=self.affected_index[indices],
            target_state=self.target_state[indices],
            original_state=self.original_state[indices],
            operation_demand=self.operation_demand[indices],
            example_ids=tuple(self.example_ids[index] for index in local),
            domains=tuple(self.domains[index] for index in local),
            templates=tuple(self.templates[index] for index in local),
            operations=tuple(self.operations[index] for index in local),
        )

    def training_tensors(self) -> SemanticTrainingTensorBatch:
        return SemanticTrainingTensorBatch(
            visible=self.visible,
            score_keys=self.score_keys,
            affected_index=self.affected_index,
            target_state=self.target_state,
            original_state=self.original_state,
        )


@dataclass(frozen=True, slots=True)
class SemanticTrainingTensorBatch:
    """Tensor-only training view that never synchronizes metadata to the CPU."""

    visible: BatchedVisibleUpdateContext
    score_keys: torch.Tensor
    affected_index: torch.Tensor
    target_state: torch.Tensor
    original_state: torch.Tensor

    @property
    def size(self) -> int:
        return int(self.affected_index.shape[0])

    def select(self, indices: torch.Tensor) -> SemanticTrainingTensorBatch:
        return SemanticTrainingTensorBatch(
            visible=self.visible.select(indices),
            score_keys=self.score_keys[indices],
            affected_index=self.affected_index[indices],
            target_state=self.target_state[indices],
            original_state=self.original_state[indices],
        )


@dataclass(frozen=True, slots=True)
class SemanticTrainingResult:
    factorized: MatchedSemanticControllerV61
    shared: MatchedSemanticControllerV61
    initial_state_sha256: str
    schedule_sha256: str
    final_loss: Mapping[str, float]
    parameter_count: int
    dense_multiply_adds_per_example: int


def _state_dict_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        local = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(local.dtype).encode("utf-8"))
        digest.update(b"\0")
        digest.update(repr(tuple(local.shape)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(local.numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _schedule_indices(
    count: int,
    *,
    steps: int,
    batch_size: int,
    seed: int,
) -> np.ndarray:
    if count <= 0:
        raise ValueError("Cannot schedule an empty semantic dataset.")
    rng = np.random.default_rng(seed)
    required = steps * batch_size
    chunks: list[np.ndarray] = []
    remaining = required
    while remaining:
        permutation = rng.permutation(count).astype(np.int64)
        take = min(remaining, count)
        chunks.append(permutation[:take])
        remaining -= take
    return np.concatenate(chunks).reshape(steps, batch_size)


def _schedule_sha256(indices: np.ndarray, example_ids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for index in indices.reshape(-1):
        digest.update(example_ids[int(index)].encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def tensorize_semantic_examples(
    examples: Sequence[SemanticExample],
    *,
    encoder: FrozenSemanticFieldEncoderV61,
    control: SemanticControl = SemanticControl.FULL,
    pairing_registry: ControlPairingRegistry | None = None,
) -> SemanticTensorBatch:
    if not examples:
        raise ValueError("Cannot tensorize an empty semantic dataset.")
    features: list[torch.Tensor] = []
    visible_states: list[torch.Tensor] = []
    visible_addresses: list[torch.Tensor] = []
    incoming_values: list[torch.Tensor] = []
    erase_scales: list[float] = []
    write_scales: list[float] = []
    keys: list[torch.Tensor] = []
    affected_indices: list[int] = []
    targets: list[torch.Tensor] = []
    original_states: list[torch.Tensor] = []
    demands: list[tuple[float, float]] = []

    for example in examples:
        pairing = (
            pairing_registry.for_example(example)
            if pairing_registry is not None
            else None
        )
        view = build_control_view(example, control, pairing)
        state_read = view.update_context.address_resolved_state_read
        record = view.semantic_record or example.safe_record
        features.append(
            encoder.encode(
                record,
                state_read,
                mask_semantics=view.semantic_record is None,
                mask_state_read=False,
            )
        )
        context = view.update_context
        visible_states.append(context.visible_state)
        visible_addresses.append(context.visible_address)
        incoming_values.append(context.incoming_value)
        erase_scales.append(context.erase_candidate_scale)
        write_scales.append(context.write_candidate_scale)
        keys.append(example.keys)
        affected_indices.append(example.affected_index)
        targets.append(example.target_state)
        original_states.append(example.state)
        demands.append(example.operation.demand)

    return SemanticTensorBatch(
        visible=BatchedVisibleUpdateContext(
            features=torch.stack(features),
            visible_state=torch.stack(visible_states),
            visible_address=torch.stack(visible_addresses),
            incoming_value=torch.stack(incoming_values),
            erase_candidate_scale=torch.tensor(erase_scales, dtype=torch.float32),
            write_candidate_scale=torch.tensor(write_scales, dtype=torch.float32),
        ),
        score_keys=torch.stack(keys),
        affected_index=torch.tensor(affected_indices, dtype=torch.long),
        target_state=torch.stack(targets),
        original_state=torch.stack(original_states),
        operation_demand=torch.tensor(demands, dtype=torch.float32),
        example_ids=tuple(example.example_id for example in examples),
        domains=tuple(example.domain for example in examples),
        templates=tuple(example.template for example in examples),
        operations=tuple(example.operation.value for example in examples),
    )


def apply_batched_visible_update(
    context: BatchedVisibleUpdateContext,
    erase: torch.Tensor,
    write: torch.Tensor,
) -> torch.Tensor:
    erase = erase.reshape(-1)
    write = write.reshape(-1)
    if erase.shape != (context.size,) or write.shape != (context.size,):
        raise ValueError("Batched semantic gates do not match the batch size.")
    if not (
        bool(torch.isfinite(erase).all().item())
        and bool(torch.isfinite(write).all().item())
    ):
        raise FloatingPointError("Batched semantic gates are non-finite.")
    if bool(((erase < 0.0) | (erase > 1.0)).any().item()) or bool(
        ((write < 0.0) | (write > 1.0)).any().item()
    ):
        raise ValueError("Batched semantic gates leave [0,1].")
    state_read = torch.bmm(
        context.visible_address.unsqueeze(1),
        context.visible_state,
    ).squeeze(1)
    erase_candidate = (
        context.visible_address.unsqueeze(2) * state_read.unsqueeze(1)
    ) * context.erase_candidate_scale[:, None, None]
    write_candidate = (
        context.visible_address.unsqueeze(2) * context.incoming_value.unsqueeze(1)
    ) * context.write_candidate_scale[:, None, None]
    return (
        context.visible_state
        - erase[:, None, None] * erase_candidate
        + write[:, None, None] * write_candidate
    )


def per_example_behavioral_metrics(
    output_state: torch.Tensor,
    batch: SemanticTensorBatch | SemanticTrainingTensorBatch,
) -> dict[str, torch.Tensor]:
    if output_state.shape != batch.target_state.shape:
        raise ValueError("Output and target state shapes differ.")
    predicted_reads = torch.bmm(batch.score_keys, output_state)
    target_reads = torch.bmm(batch.score_keys, batch.target_state)
    original_reads = torch.bmm(batch.score_keys, batch.original_state)
    batch_indices = torch.arange(batch.size, device=output_state.device)
    affected_prediction = predicted_reads[batch_indices, batch.affected_index]
    affected_target = target_reads[batch_indices, batch.affected_index]
    affected_original = original_reads[batch_indices, batch.affected_index]
    affected = torch.mean((affected_prediction - affected_target) ** 2, dim=1)

    squared = torch.mean((predicted_reads - target_reads) ** 2, dim=2)
    keep_mask = torch.ones_like(squared, dtype=torch.bool)
    keep_mask[batch_indices, batch.affected_index] = False
    retention = squared[keep_mask].reshape(batch.size, -1).mean(dim=1)
    state_mse = torch.mean((output_state - batch.target_state) ** 2, dim=(1, 2))
    old_denominator = torch.sum(affected_original**2, dim=1).clamp_min(1e-8)
    old_residual = torch.abs(
        torch.sum(affected_prediction * affected_original, dim=1)
        / old_denominator
    )
    return {
        "affected_read_mse": affected,
        "unaffected_retention_mse": retention,
        "target_state_mse": state_mse,
        "old_association_residual": old_residual,
        "new_write_mse": affected.clone(),
    }


def _train_one(
    model: MatchedSemanticControllerV61,
    batch: SemanticTensorBatch,
    schedule: np.ndarray,
    config: SemanticTrainingConfigV61,
) -> float:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    model.train()
    training_batch = batch.training_tensors()
    final_loss = float("nan")
    for indices in schedule:
        selection = torch.as_tensor(
            indices,
            dtype=torch.long,
            device=batch.visible.features.device,
        )
        local = training_batch.select(selection)
        gates = model(local.visible.features)
        output = apply_batched_visible_update(local.visible, gates.erase, gates.write)
        metrics = per_example_behavioral_metrics(output, local)
        loss = (
            config.affected_read_weight * metrics["affected_read_mse"].mean()
            + config.unaffected_retention_weight
            * metrics["unaffected_retention_mse"].mean()
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.gradient_clip_norm,
        )
        optimizer.step()
        final_loss = float(loss.detach().cpu().item())
    if not np.isfinite(final_loss):
        raise FloatingPointError("Semantic training produced a non-finite final loss.")
    return final_loss


def train_matched_semantic_pair(
    examples: Sequence[SemanticExample],
    *,
    encoder: FrozenSemanticFieldEncoderV61,
    hidden_dim: int,
    config: SemanticTrainingConfigV61,
    seed: int,
    device: torch.device,
) -> SemanticTrainingResult:
    tensor_batch = tensorize_semantic_examples(
        examples,
        encoder=encoder,
        control=SemanticControl.FULL,
    ).to(device)
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    factorized = MatchedSemanticControllerV61(
        encoder.config.input_dim,
        hidden_dim,
        SemanticRoute.FACTORIZED,
    ).to(device)
    shared = MatchedSemanticControllerV61(
        encoder.config.input_dim,
        hidden_dim,
        SemanticRoute.SHARED,
    ).to(device)
    shared.load_state_dict(factorized.state_dict())
    assert_matched_semantic_pair(factorized, shared)
    initial_hash = _state_dict_sha256(factorized)
    schedule = _schedule_indices(
        len(examples),
        steps=config.steps,
        batch_size=config.batch_size,
        seed=int(seed) + 70_005,
    )
    schedule_hash = _schedule_sha256(schedule, tensor_batch.example_ids)
    final_losses = {
        "factorized": _train_one(factorized, tensor_batch, schedule, config),
        "shared": _train_one(shared, tensor_batch, schedule, config),
    }
    return SemanticTrainingResult(
        factorized=factorized,
        shared=shared,
        initial_state_sha256=initial_hash,
        schedule_sha256=schedule_hash,
        final_loss=final_losses,
        parameter_count=sum(parameter.numel() for parameter in factorized.parameters()),
        dense_multiply_adds_per_example=(
            factorized.registered_dense_multiply_adds_per_example()
        ),
    )


def evaluate_semantic_model(
    model: MatchedSemanticControllerV61 | None,
    examples: Sequence[SemanticExample],
    *,
    encoder: FrozenSemanticFieldEncoderV61,
    control: SemanticControl,
    pairing_registry: ControlPairingRegistry | None,
    oracle_demand: bool,
    device: torch.device,
    batch_size: int = 512,
) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray]:
    if oracle_demand and model is not None:
        raise ValueError("Oracle-demand evaluation must not receive a learned model.")
    if not oracle_demand and model is None:
        raise ValueError("Learned evaluation requires a model.")
    tensor_batch = tensorize_semantic_examples(
        examples,
        encoder=encoder,
        control=control,
        pairing_registry=pairing_registry,
    )
    rows: list[dict[str, object]] = []
    affected_parts: list[np.ndarray] = []
    retention_parts: list[np.ndarray] = []
    if model is not None:
        model.eval()
    with torch.no_grad():
        for start in range(0, tensor_batch.size, batch_size):
            stop = min(start + batch_size, tensor_batch.size)
            selection = torch.arange(start, stop, dtype=torch.long)
            local = tensor_batch.select(selection).to(device)
            if oracle_demand:
                erase = local.operation_demand[:, 0]
                write = local.operation_demand[:, 1]
            else:
                if model is None:
                    raise AssertionError("Validated learned model is missing.")
                gates = model(local.visible.features)
                erase, write = gates.erase, gates.write
            output = apply_batched_visible_update(local.visible, erase, write)
            metrics = per_example_behavioral_metrics(output, local)
            affected_parts.append(
                metrics["affected_read_mse"].detach().cpu().numpy()
            )
            retention_parts.append(
                metrics["unaffected_retention_mse"].detach().cpu().numpy()
            )
            for offset, index in enumerate(range(start, stop)):
                rows.append(
                    {
                        "example_id": tensor_batch.example_ids[index],
                        "domain": tensor_batch.domains[index],
                        "template": tensor_batch.templates[index],
                        "operation": tensor_batch.operations[index],
                        "applied_erase": float(erase[offset].detach().cpu().item()),
                        "applied_write": float(write[offset].detach().cpu().item()),
                        **{
                            name: float(values[offset].detach().cpu().item())
                            for name, values in metrics.items()
                        },
                    }
                )
    return (
        rows,
        np.concatenate(affected_parts).astype(np.float64),
        np.concatenate(retention_parts).astype(np.float64),
    )


def seed_metrics_from_condition_arrays(
    examples: Sequence[SemanticExample],
    *,
    affected: Mapping[str, np.ndarray],
    retention: Mapping[str, np.ndarray],
) -> SemanticAnchorSeedMetrics:
    if not examples:
        raise ValueError("Cannot build seed metrics from an empty example set.")
    return SemanticAnchorSeedMetrics(
        episode_ids=np.asarray([example.example_id for example in examples]),
        domains=np.asarray([example.domain for example in examples]),
        templates=np.asarray([example.template for example in examples]),
        operations=np.asarray([example.operation.value for example in examples]),
        affected=affected,
        retention=retention,
    )
