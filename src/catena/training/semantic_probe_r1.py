from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
import torch

from catena.data.semantic_controls_v61 import (
    ControlPairingRegistry,
    SemanticControl,
    build_control_view,
)
from catena.data.semantic_transactions_v61 import SemanticExample
from catena.models.semantic_controllers_v61 import (
    MatchedSemanticControllerV61,
    SemanticRoute,
    assert_matched_semantic_pair,
)
from catena.models.semantic_encoder_r1 import (
    RelationalSemanticEncoderR1,
    RelationalSemanticRecord,
)
from catena.training.semantic_probe_v61 import (
    BatchedVisibleUpdateContext,
    SemanticTensorBatch,
    SemanticTrainingConfigV61,
    SemanticTrainingResult,
    _schedule_indices,
    _schedule_sha256,
    _state_dict_sha256,
    _train_one,
    apply_batched_visible_update,
    per_example_behavioral_metrics,
)


def tensorize_semantic_examples_r1(
    examples: Sequence[SemanticExample],
    *,
    encoder: RelationalSemanticEncoderR1,
    control: SemanticControl = SemanticControl.FULL,
    pairing_registry: ControlPairingRegistry | None = None,
) -> SemanticTensorBatch:
    """Tensorize R1 records without exposing memory tensors to the encoder.

    Semantic features depend only on the selected structured record. State,
    address, and incoming value stay in the public update context, where the
    existing visible-update primitive constructs erase and write candidates.
    """

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
        record = view.semantic_record or example.safe_record
        features.append(
            encoder.encode(
                cast(RelationalSemanticRecord, record),
                mask_semantics=view.semantic_record is None,
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


def train_matched_semantic_pair_r1(
    examples: Sequence[SemanticExample],
    *,
    encoder: RelationalSemanticEncoderR1,
    hidden_dim: int,
    config: SemanticTrainingConfigV61,
    seed: int,
    device: torch.device,
) -> SemanticTrainingResult:
    """Train the matched controller pair on state-independent R1 features."""

    tensor_batch = tensorize_semantic_examples_r1(
        examples,
        encoder=encoder,
        control=SemanticControl.FULL,
    ).to(device)
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))

    factorized = MatchedSemanticControllerV61(
        encoder.input_dim,
        hidden_dim,
        SemanticRoute.FACTORIZED,
    ).to(device)
    shared = MatchedSemanticControllerV61(
        encoder.input_dim,
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
        parameter_count=sum(
            parameter.numel() for parameter in factorized.parameters()
        ),
        dense_multiply_adds_per_example=(
            factorized.registered_dense_multiply_adds_per_example()
        ),
    )


def evaluate_semantic_model_r1(
    model: MatchedSemanticControllerV61 | None,
    examples: Sequence[SemanticExample],
    *,
    encoder: RelationalSemanticEncoderR1,
    control: SemanticControl,
    pairing_registry: ControlPairingRegistry | None,
    oracle_demand: bool,
    device: torch.device,
    batch_size: int = 512,
) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray]:
    """Evaluate an R1 controller with the legacy-compatible result schema."""

    if oracle_demand and model is not None:
        raise ValueError("Oracle-demand evaluation must not receive a learned model.")
    if not oracle_demand and model is None:
        raise ValueError("Learned evaluation requires a model.")

    tensor_batch = tensorize_semantic_examples_r1(
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
