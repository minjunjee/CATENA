from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from catena.core.io import file_sha256
from catena.data.structured_sequence_localization import (
    StructuredTransferCondition,
    StructuredTransferDemand,
    make_structured_identifier_codebook,
    tensor_sha256,
)
from catena.models.structured_sequence_localization import (
    StructuredSequenceFreedom,
    structured_sequence_parameter_count,
)
from catena.post_e21.locality_data import LocalityMethod
from catena.post_e21.locality_training import (
    build_locality_controller,
    evaluate_locality_controller,
    train_locality_controller,
)
from catena.training.structured_sequence_localization import (
    structured_state_dict_sha256,
)


def runtime_locality_config(
    config: dict[str, Any],
    *,
    dry_run: bool,
) -> dict[str, Any]:
    runtime = deepcopy(config)
    if not dry_run:
        return runtime
    runtime["data"]["slots"] = 8
    runtime["data"]["value_dim"] = 8
    runtime["data"]["identifier_code_dim"] = 8
    runtime["model"]["hidden_dim"] = 32
    runtime["training"]["steps"] = 2
    runtime["training"]["batch_size"] = 2
    runtime["training"]["updates"] = 1
    runtime["training"]["gap_events"] = 2
    runtime["evaluation"]["updates"] = [1]
    runtime["evaluation"]["gap_events"] = [0, 2]
    runtime["evaluation"]["batches"] = 1
    runtime["evaluation"]["batch_size"] = 2
    runtime["evaluation"]["stress"] = {"updates": 1, "gap_events": 2}
    return runtime


def run_locality_method_grid(
    *,
    runtime: dict[str, Any],
    methods: Sequence[LocalityMethod],
    seeds: Sequence[int],
    run_dir: Path,
    device: torch.device,
    parent_lock_sha256: str,
    protocol_lock_sha256: str,
    risk_scale: float,
) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, Any]]:
    conditions = [StructuredTransferCondition(value) for value in runtime["conditions"]]
    families = [StructuredTransferDemand(value) for value in runtime["demand_families"]]
    variants = [StructuredSequenceFreedom(value) for value in runtime["model"]["variants"]]
    if set(variants) != set(StructuredSequenceFreedom):
        raise ValueError("E22 requires the exact four registered E21 variants")
    slots = int(runtime["data"]["slots"])
    value_dim = int(runtime["data"]["value_dim"])
    identifier_dim = int(runtime["data"]["identifier_code_dim"])
    codebook = make_structured_identifier_codebook(
        slots=slots,
        code_dim=identifier_dim,
        seed=int(runtime["namespaces"]["identifier_schema_seed"]),
    )
    rows: list[dict[str, Any]] = []
    checkpoint_hashes: dict[str, str] = {}
    initial_hashes: dict[int, set[str]] = {}
    parameter_counts: dict[str, int] = {}
    for seed in seeds:
        for method in methods:
            for variant in variants:
                torch.manual_seed(int(runtime["initialization_seed_offset"]) + seed)
                model = build_locality_controller(
                    method=method,
                    freedom=variant,
                    slots=slots,
                    identifier_dim=identifier_dim,
                    value_dim=value_dim,
                    hidden_dim=int(runtime["model"]["hidden_dim"]),
                    address_temperature=float(runtime["model"]["address_temperature"]),
                )
                initial_hash = structured_state_dict_sha256(model.state_dict())
                initial_hashes.setdefault(seed, set()).add(initial_hash)
                parameter_key = f"{method.method_id}:{variant.value}"
                parameter_counts[parameter_key] = structured_sequence_parameter_count(model)
                trace = train_locality_controller(
                    model=model,
                    method=method,
                    conditions=conditions,
                    families=families,
                    steps=int(runtime["training"]["steps"]),
                    batch_size=int(runtime["training"]["batch_size"]),
                    slots=slots,
                    value_dim=value_dim,
                    updates=int(runtime["training"]["updates"]),
                    gap_events=int(runtime["training"]["gap_events"]),
                    state_scale=float(runtime["data"]["state_scale"]),
                    identifier_codebook=codebook,
                    learning_rate=float(runtime["training"]["learning_rate"]),
                    address_loss_weight=float(runtime["training"]["address_loss_weight"]),
                    candidate_loss_weight=float(runtime["training"]["candidate_loss_weight"]),
                    activity_loss_weight=float(runtime["training"]["activity_loss_weight"]),
                    retention_weight=float(runtime["training"]["retention_weight"]),
                    sparse_route_weight=float(runtime["training"]["sparse_route_weight"]),
                    risk_scale=risk_scale,
                    train_namespace=str(runtime["namespaces"]["train"]),
                    distractor_namespace=str(runtime["namespaces"]["distractor"]),
                    device=device,
                    seed=int(runtime["training_seed_offset"]) + seed,
                )
                checkpoint_path = (
                    run_dir / "checkpoints" / f"seed{seed}_{method.method_id}_{variant.value}.pt"
                )
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model": model.state_dict(),
                        "model_class": type(model).__name__,
                        "method": method.as_dict(),
                        "variant": variant.value,
                        "seed": seed,
                        "runtime_config": runtime,
                        "parent_e21_lock_sha256": parent_lock_sha256,
                        "protocol_lock_sha256": protocol_lock_sha256,
                    },
                    checkpoint_path,
                )
                checkpoint_key = f"seed{seed}_{method.method_id}_{variant.value}"
                checkpoint_hashes[checkpoint_key] = file_sha256(checkpoint_path)
                for condition in conditions:
                    for family in families:
                        for updates in runtime["evaluation"]["updates"]:
                            for gap in runtime["evaluation"]["gap_events"]:
                                metrics = evaluate_locality_controller(
                                    model=model,
                                    condition=condition,
                                    family=family,
                                    batches=int(runtime["evaluation"]["batches"]),
                                    batch_size=int(runtime["evaluation"]["batch_size"]),
                                    slots=slots,
                                    value_dim=value_dim,
                                    updates=int(updates),
                                    gap_events=int(gap),
                                    state_scale=float(runtime["data"]["state_scale"]),
                                    identifier_codebook=codebook,
                                    evaluation_namespace=str(runtime["namespaces"]["evaluation"]),
                                    distractor_namespace=str(runtime["namespaces"]["distractor"]),
                                    route_weight_threshold=float(
                                        runtime["evaluation"]["route_weight_threshold"]
                                    ),
                                    activity_threshold=float(
                                        runtime["evaluation"]["activity_support_threshold"]
                                    ),
                                    device=device,
                                    seed=int(runtime["evaluation_seed_offset"]) + seed,
                                )
                                rows.append(
                                    {
                                        "seed": seed,
                                        "method_id": method.method_id,
                                        "objective": method.objective.value,
                                        "selection_eligible": (method.selection_eligible),
                                        "variant": variant.value,
                                        "condition": condition.value,
                                        "demand_family": family.value,
                                        "updates": int(updates),
                                        "gap_events": int(gap),
                                        "initialization_sha256": initial_hash,
                                        "identifier_codebook_sha256": tensor_sha256(codebook),
                                        "checkpoint_sha256": checkpoint_hashes[checkpoint_key],
                                        "parameter_count": parameter_counts[parameter_key],
                                        "final_train_loss": trace.final_loss,
                                        "best_train_loss": trace.best_loss,
                                        "examples_per_second": (trace.examples_per_second),
                                        **metrics,
                                    }
                                )
    if any(len(hashes) != 1 for hashes in initial_hashes.values()):
        raise RuntimeError("E22 paired methods did not share initialization")
    metadata = {
        "identifier_codebook_sha256": tensor_sha256(codebook),
        "initialization_hashes": {
            str(seed): next(iter(hashes)) for seed, hashes in initial_hashes.items()
        },
        "parameter_counts": parameter_counts,
    }
    return rows, checkpoint_hashes, metadata
