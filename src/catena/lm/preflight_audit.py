from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from typing import Any, cast

import torch

from catena.core.provenance_v61 import sha256_canonical_json

from .checkpointing import RNGSnapshot, runtime_state_to_payload
from .config import ModelConfig
from .general_corpus import TokenMemmap
from .hashing import state_dict_digest, tensor_tree_digest
from .model import CatenaLM, RuntimeState
from .numerical_audit import (
    NumericalTolerances,
    audit_arbitrary_partitions,
    audit_gradient_accumulation,
    fixed_partition_suite,
)
from .paired_stream import (
    PackedTransactionCursor,
    TokenBalancedPairedTrainingCursor,
    replay_digest,
)
from .recurrent_mixer import (
    optimized_backend_diagnostics,
    reset_optimized_backend_diagnostics,
)
from .tokenizer import ExternalScientificTokenizer
from .trainer import optimizer_step_microbatches

E26_VARIANTS = ("dual_delta_lm", "projected_tied_delta_lm")


def model_config_for_candidate(
    candidate: Mapping[str, Any],
    *,
    variant: str,
) -> ModelConfig:
    payload = dict(candidate)
    payload.pop("id", None)
    payload.update(
        {
            "variant": variant,
            "backend_id": "compiled_scan",
            "backend_scientific_main_capable": False,
        }
    )
    return ModelConfig.from_mapping(payload)


def _fresh_model(
    config: ModelConfig,
    *,
    seed: int,
    device: torch.device,
) -> CatenaLM:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    return CatenaLM(config).to(device)


def _prefilled_state(
    model: CatenaLM,
    prefix: torch.Tensor,
    *,
    autocast_dtype: torch.dtype | None,
) -> RuntimeState:
    context = (
        torch.autocast(device_type=prefix.device.type, dtype=autocast_dtype)
        if autocast_dtype is not None
        else nullcontext()
    )
    model.eval()
    with torch.no_grad(), context:
        state = model(prefix).runtime_state
    return cast(RuntimeState, state.clone(detach=True))


def audit_locked_candidate(
    candidate: Mapping[str, Any],
    *,
    device: torch.device,
    partition_length: int,
    prefix_length: int,
    gradient_sequence_length: int,
    random_partition_seeds: Sequence[int],
    initialization_seed: int,
    data_seed: int,
    fp32_tolerances: NumericalTolerances,
    bf16_tolerances: NumericalTolerances,
) -> dict[str, Any]:
    """Run the complete per-candidate, per-variant numerical grid."""

    candidate_id = str(candidate.get("id", ""))
    if not candidate_id:
        raise ValueError("Candidate is missing its locked id")
    context_length = int(candidate["context_length"])
    if not 416 <= partition_length <= context_length:
        raise ValueError("partition_length must be in [416, candidate context_length]")
    if not 0 < prefix_length < context_length:
        raise ValueError("prefix_length must be positive and below context length")
    if not 2 <= gradient_sequence_length <= context_length:
        raise ValueError("gradient_sequence_length is outside candidate context")
    if device.type != "cuda":
        raise ValueError("Actual-candidate compiled numerical preflight requires CUDA")

    partitions = fixed_partition_suite(
        partition_length,
        random_seeds=random_partition_seeds,
    )
    cpu_generator = torch.Generator(device="cpu")
    cpu_generator.manual_seed(data_seed)
    input_cpu = torch.randint(
        0,
        int(candidate["vocab_size"]),
        (1, partition_length),
        generator=cpu_generator,
    )
    prefix_cpu = torch.randint(
        0,
        int(candidate["vocab_size"]),
        (1, prefix_length),
        generator=cpu_generator,
    )
    gradient_cpu = torch.randint(
        0,
        int(candidate["vocab_size"]),
        (4, gradient_sequence_length),
        generator=cpu_generator,
    )
    input_ids = input_cpu.to(device)
    prefix = prefix_cpu.to(device)
    gradient_batch = gradient_cpu.to(device)

    variants: dict[str, Any] = {}
    for variant in E26_VARIANTS:
        reset_optimized_backend_diagnostics()
        config = model_config_for_candidate(candidate, variant=variant)
        seed = initialization_seed
        model = _fresh_model(config, seed=seed, device=device)
        state_rows: dict[str, Any] = {}
        for state_name in ("zero_state", "prefilled_state"):
            precision_rows: dict[str, Any] = {}
            for precision, dtype, tolerances in (
                ("fp32", None, fp32_tolerances),
                ("bf16", torch.bfloat16, bf16_tolerances),
            ):
                initial_state = (
                    None
                    if state_name == "zero_state"
                    else _prefilled_state(model, prefix, autocast_dtype=dtype)
                )
                report = audit_arbitrary_partitions(
                    model,
                    input_ids,
                    partitions=partitions,
                    tolerances=tolerances,
                    autocast_dtype=dtype,
                    initial_state=initial_state,
                )
                precision_rows[precision] = report.as_dict()
            state_rows[state_name] = precision_rows

        fp32_accumulation = audit_gradient_accumulation(
            model,
            gradient_batch,
            accumulation_layouts=((4,), (2, 2), (1, 1, 1, 1)),
            tolerances=fp32_tolerances,
            autocast_dtype=None,
        )
        bf16_accumulation = audit_gradient_accumulation(
            model,
            gradient_batch,
            accumulation_layouts=((4,), (2, 2), (1, 1, 1, 1)),
            tolerances=bf16_tolerances,
            autocast_dtype=torch.bfloat16,
        )
        accumulation_rows = {
            "fp32": [row.as_dict() for row in fp32_accumulation],
            "bf16": [row.as_dict() for row in bf16_accumulation],
        }
        passed = all(
            precision_row["passed"] is True
            for state_row in state_rows.values()
            for precision_row in state_row.values()
        ) and all(
            row["passed"] is True
            for precision_rows in accumulation_rows.values()
            for row in precision_rows
        )
        diagnostics = optimized_backend_diagnostics()
        passed = (
            passed and diagnostics["fallback_count"] == 0 and diagnostics["graph_break_count"] == 0
        )
        variants[variant] = {
            "variant": variant,
            "initialization_seed": seed,
            "initial_parameter_digest": state_dict_digest(model),
            "arbitrary_partitions": state_rows,
            "gradient_accumulation": accumulation_rows,
            "compiled_backend_diagnostics": diagnostics,
            "passed": passed,
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    initialization_digests = {str(row["initial_parameter_digest"]) for row in variants.values()}
    initialization_matched = len(initialization_digests) == 1
    return {
        "candidate_id": candidate_id,
        "model_config_sha256": sha256_canonical_json(dict(candidate)),
        "partition_length": partition_length,
        "prefix_length": prefix_length,
        "gradient_sequence_length": gradient_sequence_length,
        "partitions": [list(value) for value in partitions],
        "random_partition_seeds": [int(value) for value in random_partition_seeds],
        "variants": variants,
        "initialization_matched_across_variants": initialization_matched,
        "passed": initialization_matched
        and all(row["passed"] is True for row in variants.values()),
    }


def audit_target_gradient_accumulation(
    candidate: Mapping[str, Any],
    *,
    device: torch.device,
    target_global_batch_tokens: int,
    selected_microbatch_sequences: int,
    microbatch_size_candidates: Sequence[int],
    initialization_seed: int,
    data_seed: int,
    bf16_tolerances: NumericalTolerances,
) -> dict[str, Any]:
    """Audit the exact target-context global-batch layouts used by training.

    The first layout is the mandatory all-sequences-at-once accumulation-1
    reference required by the Stage-2 protocol. The resource-selected layout
    and every smaller preregistered divisor are compared against that same
    fixed global token batch. If the mandatory reference is not executable,
    the preflight fails closed rather than weakening the contract.
    """

    candidate_id = str(candidate.get("id", ""))
    if not candidate_id:
        raise ValueError("Candidate is missing its locked id")
    context_length = int(candidate["context_length"])
    if target_global_batch_tokens <= 0 or target_global_batch_tokens % context_length:
        raise ValueError("Target global token batch must be divisible by context length")
    global_sequences = target_global_batch_tokens // context_length
    selected = int(selected_microbatch_sequences)
    selected_and_smaller = sorted(
        {
            int(value)
            for value in microbatch_size_candidates
            if 0 < int(value) <= selected and global_sequences % int(value) == 0
        },
        reverse=True,
    )
    if not selected_and_smaller or selected_and_smaller[0] != selected:
        raise ValueError("Target gradient audit requires the selected preregistered microbatch")
    audited_microbatches = [global_sequences]
    audited_microbatches.extend(
        value for value in selected_and_smaller if value != global_sequences
    )
    if len(audited_microbatches) < 2:
        raise ValueError("Target gradient audit requires accumulation-1 and accumulated layouts")
    layouts = tuple(
        tuple(microbatch for _ in range(global_sequences // microbatch))
        for microbatch in audited_microbatches
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(data_seed)
    global_batch = torch.randint(
        0,
        int(candidate["vocab_size"]),
        (global_sequences, context_length),
        generator=generator,
    ).to(device)

    def scheduler_factory(optimizer: torch.optim.Optimizer) -> Any:
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda _step: 1.0,
        )

    variants: dict[str, Any] = {}
    for variant in E26_VARIANTS:
        reset_optimized_backend_diagnostics()
        config = model_config_for_candidate(candidate, variant=variant)
        model = _fresh_model(config, seed=initialization_seed, device=device)
        rows = audit_gradient_accumulation(
            model,
            global_batch,
            accumulation_layouts=layouts,
            tolerances=bf16_tolerances,
            autocast_dtype=torch.bfloat16,
            scheduler_factory=scheduler_factory,
        )
        diagnostics = optimized_backend_diagnostics()
        passed = (
            all(row.passed for row in rows)
            and diagnostics["fallback_count"] == 0
            and diagnostics["graph_break_count"] == 0
        )
        variants[variant] = {
            "variant": variant,
            "precision": "bf16_actual_training",
            "rows": [row.as_dict() for row in rows],
            "compiled_backend_diagnostics": diagnostics,
            "passed": passed,
        }
        del model
        torch.cuda.empty_cache()
    payload = {
        "candidate_id": candidate_id,
        "model_config_sha256": sha256_canonical_json(dict(candidate)),
        "context_length": context_length,
        "target_global_batch_tokens": target_global_batch_tokens,
        "global_batch_sequences": global_sequences,
        "selected_microbatch_sequences": selected,
        "accumulation_steps": global_sequences // selected,
        "audited_microbatch_sequences": audited_microbatches,
        "accumulation_layouts": [list(layout) for layout in layouts],
        "variants": variants,
        "passed": all(row["passed"] is True for row in variants.values()),
    }
    payload["audit_sha256"] = sha256_canonical_json(payload)
    return payload


def copy_candidate_mapping(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe detached candidate mapping for worker specifications."""

    return copy.deepcopy(dict(candidate))


def build_scientific_training_cursor(
    *,
    tokenizer_manifest_path: str,
    corpus_manifest_path: str,
    sequence_length: int,
    general_seed: int,
    transaction_seed: int,
) -> tuple[
    TokenMemmap,
    ExternalScientificTokenizer,
    TokenBalancedPairedTrainingCursor,
]:
    tokenizer = ExternalScientificTokenizer.from_manifest(tokenizer_manifest_path)
    corpus = TokenMemmap.from_scientific_manifest(
        corpus_manifest_path,
        tokenizer_manifest=tokenizer.manifest,
    )
    tokenizer_hash = tokenizer.manifest.manifest_hash
    transaction = PackedTransactionCursor(
        tokenizer,
        tokenizer_hash=tokenizer_hash,
        seed=transaction_seed,
        sequence_length=sequence_length,
        pad_token_id=tokenizer.manifest.special_tokens["pad"],
    )
    cursor = TokenBalancedPairedTrainingCursor(
        corpus.paired_cursor(seed=general_seed, sequence_length=sequence_length),
        transaction,
    )
    return corpus, tokenizer, cursor


def audit_scientific_cursor_replay(
    *,
    tokenizer_manifest_path: str,
    corpus_manifest_path: str,
    sequence_length: int,
    general_seed: int,
    transaction_seed: int,
    minimum_tokens: int,
) -> dict[str, Any]:
    corpus, tokenizer, cursor = build_scientific_training_cursor(
        tokenizer_manifest_path=tokenizer_manifest_path,
        corpus_manifest_path=corpus_manifest_path,
        sequence_length=sequence_length,
        general_seed=general_seed,
        transaction_seed=transaction_seed,
    )
    tokenizer_hash = tokenizer.manifest.manifest_hash
    tied = cursor.fork(corpus, tokenizer, tokenizer_hash=tokenizer_hash)
    dual = cursor.fork(corpus, tokenizer, tokenizer_hash=tokenizer_hash)
    tied_first = replay_digest(tied, minimum_tokens=minimum_tokens)
    dual_first = replay_digest(dual, minimum_tokens=minimum_tokens)
    boundary_snapshot = tied.snapshot()
    tied_second = replay_digest(tied, minimum_tokens=minimum_tokens)
    restored = TokenBalancedPairedTrainingCursor.from_snapshot(
        corpus,
        tokenizer,
        tokenizer_hash=tokenizer_hash,
        snapshot=boundary_snapshot,
    )
    restored_second = replay_digest(restored, minimum_tokens=minimum_tokens)
    dual_second = replay_digest(dual, minimum_tokens=minimum_tokens)

    receipt_keys = (
        "start_sequence_index",
        "end_sequence_index",
        "sequences",
        "tokens",
        "general_sequences",
        "transaction_sequences",
        "loss_bearing_tokens",
        "general_unpadded_tokens",
        "transaction_unpadded_tokens",
        "padding_tokens",
        "realized_general_fraction",
        "realized_transaction_fraction",
        "metadata_sha256",
        "token_bytes_sha256",
        "data_order_sha256",
        "requested_minimum_tokens",
        "overrun_tokens",
    )

    def comparable(value: Mapping[str, Any]) -> dict[str, Any]:
        return {key: value[key] for key in receipt_keys}

    paired_first_equal = comparable(tied_first) == comparable(dual_first)
    paired_second_equal = comparable(tied_second) == comparable(dual_second)
    resume_equal = comparable(tied_second) == comparable(restored_second)
    return {
        "cursor_algorithm": "token_balanced_complete_example_80_20_v2",
        "sequence_length": sequence_length,
        "minimum_tokens_per_phase": minimum_tokens,
        "first_phase": tied_first,
        "post_boundary_phase": tied_second,
        "restored_post_boundary_phase": restored_second,
        "dual_first_phase": dual_first,
        "dual_post_boundary_phase": dual_second,
        "boundary_snapshot_sha256": boundary_snapshot["snapshot_sha256"],
        "paired_first_phase_equal": paired_first_equal,
        "paired_post_boundary_equal": paired_second_equal,
        "resume_post_boundary_equal": resume_equal,
        "actual_source_nonpadding_tokens": {
            "first_general": tied_first["general_unpadded_tokens"],
            "first_transaction": tied_first["transaction_unpadded_tokens"],
            "post_boundary_general": tied_second["general_unpadded_tokens"],
            "post_boundary_transaction": tied_second["transaction_unpadded_tokens"],
        },
        "passed": paired_first_equal and paired_second_equal and resume_equal,
    }


def run_cursor_training_steps(
    *,
    model: CatenaLM,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    cursor: TokenBalancedPairedTrainingCursor,
    steps: int,
    device: torch.device,
    autocast_dtype: torch.dtype | None = None,
) -> tuple[float, list[dict[str, Any]]]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    records: list[dict[str, Any]] = []
    last_loss = float("nan")
    for _ in range(steps):
        rows, receipt = cursor.take(1)
        row = rows[0]
        batch = torch.as_tensor(row.token_ids, dtype=torch.long, device=device)[None, :]
        loss_mask = torch.zeros_like(batch, dtype=torch.float32)
        loss_mask[:, : row.unpadded_tokens] = 1.0
        step = optimizer_step_microbatches(
            model,
            [batch],
            optimizer=optimizer,
            scheduler=scheduler,
            loss_masks=[loss_mask],
            autocast_dtype=autocast_dtype,
        )
        last_loss = step.loss
        records.append(
            {
                "source_type": row.source_type,
                "source_index": row.source_index,
                "unpadded_tokens": row.unpadded_tokens,
                "padding_tokens": row.padding_tokens,
                "valid_prediction_tokens": step.valid_prediction_tokens,
                "cursor_receipt": receipt.as_dict(),
            }
        )
    return last_loss, records


def final_restart_record(
    *,
    model: CatenaLM,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    cursor: TokenBalancedPairedTrainingCursor,
    final_loss: float,
    device: torch.device,
    autocast_dtype: torch.dtype | None = None,
) -> dict[str, Any]:
    probe_length = min(32, model.config.context_length)
    probe = torch.arange(probe_length, device=device)[None, :] % model.config.vocab_size
    context = (
        torch.autocast(device_type=device.type, dtype=autocast_dtype)
        if autocast_dtype is not None
        else nullcontext()
    )
    model.eval()
    with torch.no_grad(), context:
        output = model(probe)
    rng = RNGSnapshot.capture()
    snapshot = cursor.snapshot()
    return {
        "model_sha256": state_dict_digest(model),
        "optimizer_sha256": tensor_tree_digest(optimizer.state_dict()),
        "scheduler_sha256": tensor_tree_digest(scheduler.state_dict()),
        "cursor_snapshot_sha256": snapshot["snapshot_sha256"],
        "cursor_snapshot": snapshot,
        "final_loss": final_loss,
        "final_logits_sha256": tensor_tree_digest(output.logits),
        "final_runtime_state_sha256": tensor_tree_digest(
            runtime_state_to_payload(output.runtime_state)
        ),
        "rng_sha256": tensor_tree_digest(rng.as_payload()),
    }
