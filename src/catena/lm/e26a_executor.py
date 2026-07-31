"""Callable scientific E26a executor.

This module is intentionally limited to the protocol-identifiability gate.  It
can measure the locked candidates and run the paired <=20M-token pilot when the
future canonical command is explicitly authorized, but it contains no E26b or
E26c launcher.
"""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from torch.nn import functional as F

from catena.core.provenance_v61 import read_json_object_strict, sha256_canonical_json

from .artifacts import ArtifactRun
from .config import ModelConfig
from .e26a_gate import (
    PRIMARY_OPERATIONS,
    SATURATION_HIGH,
    SATURATION_LOW,
    CandidateMeasurement,
    E26AGateAdmission,
    E26AGateBlocked,
    candidate_numerical_coverage,
    create_scientific_gate_run,
    exclusive_scientific_gate_lock,
    finalize_scientific_gate_run,
    project_candidate_resources,
    require_locked_execution_device,
    require_locked_resource_selection,
    select_candidate,
)
from .evaluator import evaluate_episode_branched, transaction_score
from .general_corpus import TokenMemmap
from .hashing import optimizer_state_signature, parameter_signature_hash, state_dict_digest
from .model import CatenaLM, assert_matched_models, build_paired_models
from .paired_stream import (
    PackedTransactionCursor,
    TokenBalancedPairedTrainingCursor,
    TrainingSequence,
)
from .recurrent_mixer import (
    optimized_backend_diagnostics,
    reset_optimized_backend_diagnostics,
)
from .tokenizer import ExternalScientificTokenizer
from .trainer import (
    compare_optimizer_signatures,
    make_optimizer,
    measure_checkpoint_io,
    optimizer_step_microbatches,
)
from .transactional_stream import TransactionEpisode


@dataclass(frozen=True, slots=True)
class E26AExecutionResult:
    measurements: tuple[CandidateMeasurement, ...]
    gates: tuple[dict[str, Any], ...]
    model_manifest: dict[str, Any]
    backend_manifest: dict[str, Any]
    data_manifest: dict[str, Any]
    pilot_summary: dict[str, Any]


class E26AExecutionBackend(Protocol):
    def execute(
        self,
        admission: E26AGateAdmission,
        run: ArtifactRun,
        *,
        device: torch.device,
    ) -> E26AExecutionResult: ...


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot take a percentile of an empty sequence")
    position = min(len(ordered) - 1, max(0, math.ceil(probability * len(ordered)) - 1))
    return float(ordered[position])


def _gate(
    name: str,
    passed: bool,
    observed: Any,
    criterion: Any,
    note: str = "",
) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "observed": observed,
        "criterion": criterion,
        "note": note,
    }


def bounded_80_20_nonpadding_token_mix(
    accounting: Mapping[str, int],
    *,
    context_length: int,
) -> bool:
    total = int(accounting.get("nonpadding_input_tokens", 0))
    general = int(accounting.get("general_nonpadding_input_tokens", -1))
    transaction = int(accounting.get("transaction_nonpadding_input_tokens", -1))
    return (
        total > 0
        and general >= 0
        and transaction >= 0
        and general + transaction == total
        and abs(4 * transaction - general) <= 4 * context_length
    )


def _candidate_config(candidate: Mapping[str, Any], *, variant: str) -> ModelConfig:
    mapping = dict(candidate)
    mapping.pop("id", None)
    mapping.update(
        {
            "variant": variant,
            "backend_id": "compiled_scan",
            "backend_scientific_main_capable": True,
        }
    )
    return ModelConfig.from_mapping(mapping)


def _paired_cursor(
    corpus: TokenMemmap,
    tokenizer: ExternalScientificTokenizer,
    *,
    seed: int,
    sequence_length: int,
) -> TokenBalancedPairedTrainingCursor:
    general = corpus.paired_cursor(seed=seed, sequence_length=sequence_length)
    transaction = PackedTransactionCursor(
        tokenizer,
        tokenizer_hash=tokenizer.manifest.manifest_hash,
        seed=seed,
        sequence_length=sequence_length,
        pad_token_id=tokenizer.manifest.special_tokens["pad"],
        split="train",
    )
    return TokenBalancedPairedTrainingCursor(general, transaction)


def _microbatches(
    rows: Sequence[TrainingSequence],
    *,
    microbatch_size: int,
    device: torch.device,
) -> tuple[list[torch.Tensor], list[torch.Tensor], dict[str, int]]:
    if not rows or len(rows) % microbatch_size:
        raise ValueError("Rows must form complete microbatches")
    batches: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    accounting = {
        "nonpadding_input_tokens": 0,
        "valid_prediction_tokens": 0,
        "general_nonpadding_input_tokens": 0,
        "transaction_nonpadding_input_tokens": 0,
        "general_valid_prediction_tokens": 0,
        "transaction_valid_prediction_tokens": 0,
    }
    for start in range(0, len(rows), microbatch_size):
        selected = rows[start : start + microbatch_size]
        token_array = np.stack([row.token_ids for row in selected])
        batch = torch.as_tensor(token_array, dtype=torch.long, device=device)
        mask = torch.zeros(batch.shape, dtype=torch.float32, device=device)
        for index, row in enumerate(selected):
            mask[index, : row.unpadded_tokens] = 1.0
            source = str(row.source_type)
            if source not in {"general", "transaction"}:
                raise E26AGateBlocked(f"Unknown training source type: {source}")
            unpadded = int(row.unpadded_tokens)
            valid_predictions = max(0, unpadded - 1)
            accounting["nonpadding_input_tokens"] += unpadded
            accounting["valid_prediction_tokens"] += valid_predictions
            accounting[f"{source}_nonpadding_input_tokens"] += unpadded
            accounting[f"{source}_valid_prediction_tokens"] += valid_predictions
        batches.append(batch)
        masks.append(mask)
    return batches, masks, accounting


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _cpu_state_snapshot(model: CatenaLM) -> dict[str, torch.Tensor]:
    """Keep feasibility probes from consuming an extra model copy on the GPU."""

    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _training_step(
    model: CatenaLM,
    optimizer: torch.optim.Optimizer,
    cursor: TokenBalancedPairedTrainingCursor,
    *,
    global_sequences: int,
    microbatch_size: int,
    device: torch.device,
    scheduler: Any = None,
) -> tuple[Any, dict[str, Any], float, dict[str, int]]:
    rows, receipt = cursor.take(global_sequences)
    batches, masks, accounting = _microbatches(
        rows,
        microbatch_size=microbatch_size,
        device=device,
    )
    _synchronize(device)
    started = time.perf_counter()
    metric = optimizer_step_microbatches(
        model,
        batches,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_masks=masks,
        grad_clip_norm=1.0,
        autocast_dtype=torch.bfloat16,
    )
    _synchronize(device)
    if metric.valid_prediction_tokens != accounting["valid_prediction_tokens"]:
        raise E26AGateBlocked(
            "Optimizer loss-mask denominator differs from source token accounting"
        )
    return metric, receipt.as_dict(), time.perf_counter() - started, accounting


def _largest_feasible_microbatch(
    model: CatenaLM,
    *,
    corpus: TokenMemmap,
    tokenizer: ExternalScientificTokenizer,
    seed: int,
    global_sequences: int,
    candidates: Sequence[int],
    device: torch.device,
) -> int:
    initial = _cpu_state_snapshot(model)
    eligible = sorted(
        {
            int(value)
            for value in candidates
            if int(value) > 0 and global_sequences % int(value) == 0
        },
        reverse=True,
    )
    if not eligible:
        raise E26AGateBlocked("No locked microbatch candidate divides the global batch")
    for microbatch_size in eligible:
        optimizer = make_optimizer(model)
        cursor = _paired_cursor(
            corpus,
            tokenizer,
            seed=seed,
            sequence_length=model.config.context_length,
        )
        try:
            _training_step(
                model,
                optimizer,
                cursor,
                global_sequences=global_sequences,
                microbatch_size=microbatch_size,
                device=device,
            )
        except torch.cuda.OutOfMemoryError:
            model.load_state_dict(initial, strict=True)
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            continue
        model.load_state_dict(initial, strict=True)
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        return microbatch_size
    raise E26AGateBlocked("Every locked microbatch candidate exhausted device memory")


def _measure_variant(
    model: CatenaLM,
    *,
    candidate_id: str,
    corpus: TokenMemmap,
    tokenizer: ExternalScientificTokenizer,
    seed: int,
    global_sequences: int,
    microbatch_candidates: Sequence[int],
    warmup_steps: int,
    measured_steps: int,
    run: ArtifactRun,
    device: torch.device,
) -> dict[str, Any]:
    initial = _cpu_state_snapshot(model)
    microbatch_size = _largest_feasible_microbatch(
        model,
        corpus=corpus,
        tokenizer=tokenizer,
        seed=seed,
        global_sequences=global_sequences,
        candidates=microbatch_candidates,
        device=device,
    )
    model.load_state_dict(initial, strict=True)
    optimizer = make_optimizer(model)
    cursor = _paired_cursor(
        corpus,
        tokenizer,
        seed=seed,
        sequence_length=model.config.context_length,
    )
    reset_optimized_backend_diagnostics()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    durations: list[float] = []
    nonpadding_tokens = 0
    measured_source_accounting = {
        "nonpadding_input_tokens": 0,
        "valid_prediction_tokens": 0,
        "general_nonpadding_input_tokens": 0,
        "transaction_nonpadding_input_tokens": 0,
        "general_valid_prediction_tokens": 0,
        "transaction_valid_prediction_tokens": 0,
    }
    full_source_accounting = {
        "nonpadding_input_tokens": 0,
        "valid_prediction_tokens": 0,
        "general_nonpadding_input_tokens": 0,
        "transaction_nonpadding_input_tokens": 0,
        "general_valid_prediction_tokens": 0,
        "transaction_valid_prediction_tokens": 0,
    }
    losses: list[float] = []
    compile_seconds = 0.0
    for step in range(warmup_steps + measured_steps):
        metric, _, duration, accounting = _training_step(
            model,
            optimizer,
            cursor,
            global_sequences=global_sequences,
            microbatch_size=microbatch_size,
            device=device,
        )
        if step == 0:
            compile_seconds = duration
        for name, value in accounting.items():
            full_source_accounting[name] += int(value)
        if step >= warmup_steps:
            durations.append(duration)
            nonpadding_tokens += accounting["nonpadding_input_tokens"]
            for name, value in accounting.items():
                measured_source_accounting[name] += int(value)
            losses.append(float(metric.loss))
    if not durations or not all(math.isfinite(value) for value in [*durations, *losses]):
        raise E26AGateBlocked(f"{candidate_id}/{model.config.variant}: non-finite throughput run")
    checkpoint_probe = (
        run.checkpoint_dir() / f"{candidate_id}_{model.config.variant}_ephemeral_probe.pt"
    )
    checkpoint = measure_checkpoint_io(
        model,
        optimizer,
        checkpoint_probe,
    )
    checkpoint_probe.unlink()
    checkpoint.pop("path", None)
    checkpoint["path_policy"] = "EPHEMERAL_PROBE_REMOVED_AFTER_HASH_AND_FSYNC"
    diagnostics = optimized_backend_diagnostics()
    total_nonpadding_tokens = full_source_accounting["nonpadding_input_tokens"]
    total_prediction_tokens = full_source_accounting["valid_prediction_tokens"]
    source_accounting_report: dict[str, Any] = {
        **full_source_accounting,
        "realized_general_nonpadding_fraction": (
            full_source_accounting["general_nonpadding_input_tokens"] / total_nonpadding_tokens
        ),
        "realized_transaction_nonpadding_fraction": (
            full_source_accounting["transaction_nonpadding_input_tokens"] / total_nonpadding_tokens
        ),
        "realized_general_prediction_fraction": (
            full_source_accounting["general_valid_prediction_tokens"] / total_prediction_tokens
        ),
        "realized_transaction_prediction_fraction": (
            full_source_accounting["transaction_valid_prediction_tokens"] / total_prediction_tokens
        ),
        "abs_4t_minus_g": abs(
            4 * full_source_accounting["transaction_nonpadding_input_tokens"]
            - full_source_accounting["general_nonpadding_input_tokens"]
        ),
        "max_abs_4t_minus_g": 4 * model.config.context_length,
    }
    return {
        "variant": model.config.variant,
        "microbatch_size": microbatch_size,
        "accumulation_steps": global_sequences // microbatch_size,
        "tokens_per_second": nonpadding_tokens / sum(durations),
        "p50_step_seconds": float(statistics.median(durations)),
        "p95_step_seconds": _percentile(durations, 0.95),
        "compile_seconds": compile_seconds,
        "checkpoint": checkpoint,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "diagnostics": diagnostics,
        "measured_steps": measured_steps,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "source_token_accounting": source_accounting_report,
        "measured_source_token_accounting": measured_source_accounting,
        "token_mix_bounded_discrepancy_passed": bounded_80_20_nonpadding_token_mix(
            full_source_accounting,
            context_length=model.config.context_length,
        ),
    }


def _general_validation_manifest(admission: E26AGateAdmission) -> Path:
    records = admission.data_readiness.get("general_corpora")
    if not isinstance(records, Mapping):
        raise E26AGateBlocked("Data readiness lacks general_corpora")
    for name, record in records.items():
        if "validation" not in str(name) or not isinstance(record, Mapping):
            continue
        value = record.get("corpus_manifest_path")
        if isinstance(value, str):
            return Path(value).expanduser().resolve(strict=True)
    raise E26AGateBlocked("Data readiness lacks the general-validation corpus manifest")


def _model_gate_state_metrics(
    model: CatenaLM,
    input_ids: torch.Tensor,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad(), torch.autocast(device_type=input_ids.device.type, dtype=torch.bfloat16):
        output = model(input_ids, return_gate_trace=True)
    gate_values: list[torch.Tensor] = []
    for trace in output.gate_traces.values():
        gate_values.extend((trace.erase.detach().float(), trace.write.detach().float()))
    if not gate_values:
        raise E26AGateBlocked("Selected model emitted no recurrent gate traces")
    flattened = torch.cat([value.reshape(-1) for value in gate_values])
    saturation = ((flattened <= SATURATION_LOW) | (flattened >= SATURATION_HIGH)).float()
    state_norms = [
        float(state.matrix.detach().float().norm().item() / math.sqrt(state.matrix.numel()))
        for state in output.runtime_state.recurrent
    ]
    return {
        "gate_saturation_fraction": float(saturation.mean().item()),
        "state_norm_max_normalized_frobenius": max(state_norms),
    }


def _stale_episode(episode: TransactionEpisode) -> TransactionEpisode:
    prefix = "\n\n".join(
        part for part in (episode.materialization_text, episode.distractor_text) if part
    )
    return replace(episode, branch_prefix_text=prefix)


def _evaluate_pilot_model(
    model: CatenaLM,
    tokenizer: ExternalScientificTokenizer,
    episodes: Sequence[TransactionEpisode],
    *,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    forbidden = sorted({episode.split for episode in episodes if episode.split != "validation"})
    if forbidden:
        raise E26AGateBlocked(f"E26a evaluator received a non-validation split: {forbidden}")
    model.eval()
    scores: dict[str, dict[str, list[float]]] = {}
    query_scores: dict[str, dict[str, dict[str, list[float]]]] = {}
    raw_rows: list[dict[str, Any]] = []
    for episode in episodes:
        condition_rows = {
            "learned": evaluate_episode_branched(model, tokenizer, episode, device=device),
            "exact_refresh": evaluate_episode_branched(
                model, tokenizer, episode, device=device, exact_refresh=True
            ),
            "stale": evaluate_episode_branched(
                model, tokenizer, _stale_episode(episode), device=device
            ),
        }
        operation_scores = scores.setdefault(
            episode.operation,
            {"learned": [], "exact_refresh": [], "stale": []},
        )
        operation_query_scores = query_scores.setdefault(episode.operation, {})
        for condition, rows in condition_rows.items():
            score = transaction_score(rows)
            operation_scores[condition].append(score)
            condition_query_scores = operation_query_scores.setdefault(condition, {})
            for row in rows:
                condition_query_scores.setdefault(row.query_type, []).append(float(row.correct))
                raw_rows.append(
                    {
                        "variant": model.config.variant,
                        "episode_id": episode.episode_id,
                        "split": episode.split,
                        "operation": episode.operation,
                        "condition": condition,
                        **row.to_dict(),
                    }
                )
    summary: dict[str, Any] = {}
    for operation_name, conditions in scores.items():
        means: dict[str, Any] = {
            condition: float(statistics.fmean(values)) for condition, values in conditions.items()
        }
        means["exact_minus_stale"] = means["exact_refresh"] - means["stale"]
        means["query_accuracy"] = {
            condition: {
                query_type: float(statistics.fmean(values))
                for query_type, values in by_query.items()
            }
            for condition, by_query in query_scores[operation_name].items()
        }
        summary[operation_name] = means
    return summary, raw_rows


def _general_perplexity(
    model: CatenaLM,
    corpus: TokenMemmap,
    *,
    seed: int,
    sequence_length: int,
    minimum_tokens: int,
    microbatch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    cursor = corpus.paired_cursor(seed=seed, sequence_length=sequence_length)
    sequence_count = math.ceil(minimum_tokens / sequence_length)
    rows, receipt = cursor.take(sequence_count)
    total_nll = 0.0
    total_predictions = 0
    model.eval()
    for start in range(0, len(rows), microbatch_size):
        selected = rows[start : start + microbatch_size]
        batch = torch.as_tensor(np.stack(selected), dtype=torch.long, device=device)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = model(batch).logits
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                batch[:, 1:].reshape(-1),
                reduction="sum",
            )
        total_nll += float(loss.detach().float().item())
        total_predictions += int(batch.shape[0] * (batch.shape[1] - 1))
    value = math.exp(total_nll / total_predictions)
    return {
        "perplexity": value,
        "nll_sum": total_nll,
        "prediction_tokens": total_predictions,
        "cursor_receipt": receipt.as_dict(),
    }


class RealE26AExecutionBackend:
    """The default future canonical implementation; never called by imports/tests."""

    @staticmethod
    def _selection_prerequisite_gates(
        admission: E26AGateAdmission,
        measurements: Sequence[CandidateMeasurement],
    ) -> list[dict[str, Any]]:
        parameter_min = int(admission.config["candidate_selection"]["parameter_count_min"])
        parameter_max = int(admission.config["candidate_selection"]["parameter_count_max"])
        in_range = [
            row for row in measurements if parameter_min <= row.parameter_count <= parameter_max
        ]
        matched = [row for row in in_range if row.matching_passed]
        numerical = [row for row in matched if row.numerical_passed]
        mixed = [row for row in numerical if row.token_mix_bounded_discrepancy_passed]
        compiled = [row for row in mixed if row.graph_break_count == 0 and row.fallback_count == 0]
        resource_eligible = [
            row
            for row in compiled
            if any(
                projection["eligible"]
                for projection in project_candidate_resources(
                    row,
                    admission.resource_policy,
                )
            )
        ]
        stages = (
            (
                "candidate_parameter_range_available",
                in_range,
                f"at least one candidate in [{parameter_min},{parameter_max}]",
            ),
            (
                "candidate_parameter_initialization_optimizer_matching_available",
                matched,
                "at least one in-range matched candidate",
            ),
            (
                "candidate_numerical_parity_restart_contract_available",
                numerical,
                "at least one matched numerical-contract candidate",
            ),
            (
                "candidate_token_mix_data_contract_available",
                mixed,
                "at least one numerical candidate with bounded actual-token mix",
            ),
            (
                "candidate_compiled_backend_without_graph_break_or_fallback_available",
                compiled,
                "at least one compiled candidate with zero fallback/graph break",
            ),
            (
                "candidate_throughput_deadline_storage_budget_available",
                resource_eligible,
                "at least one resource-eligible candidate",
            ),
        )
        gates: list[dict[str, Any]] = []
        for name, rows, criterion in stages:
            gates.append(
                _gate(
                    name,
                    bool(rows),
                    [row.candidate_id for row in rows],
                    criterion,
                )
            )
            if not rows:
                break
        return gates

    @staticmethod
    def _backend_manifest_from_input(
        admission: E26AGateAdmission,
        measurements: Sequence[CandidateMeasurement],
    ) -> dict[str, Any]:
        backend_manifest = read_json_object_strict(admission.paths.backend_manifest)
        upstream_manifest_sha256 = backend_manifest.pop("manifest_sha256", None)
        backend_manifest.update(
            {
                "schema_version": "catena-v8.1",
                "upstream_manifest_sha256": upstream_manifest_sha256,
                "source_input_sha256": admission.input_hashes["backend_manifest_sha256"],
                "candidate_measurements": [row.candidate_id for row in measurements],
            }
        )
        return backend_manifest

    def _measure_candidates(
        self,
        run: ArtifactRun,
        *,
        config: Mapping[str, Any],
        numerical_audit: Mapping[str, Any],
        corpus: TokenMemmap,
        tokenizer: ExternalScientificTokenizer,
        device: torch.device,
        candidate_ids: set[str] | None = None,
    ) -> tuple[CandidateMeasurement, ...]:
        throughput = config["throughput"]
        global_tokens = int(throughput["target_global_batch_tokens"])
        measurements: list[CandidateMeasurement] = []
        numerical_coverage = candidate_numerical_coverage(
            config,
            numerical_audit,
        )
        for candidate_index, candidate in enumerate(config["model_candidates"]):
            candidate_id = str(candidate["id"])
            if candidate_ids is not None and candidate_id not in candidate_ids:
                continue
            context_length = int(candidate["context_length"])
            if global_tokens % context_length:
                raise E26AGateBlocked(
                    f"{candidate_id}: global token batch is not divisible by context length"
                )
            global_sequences = global_tokens // context_length
            base = _candidate_config(candidate, variant="dual_delta_lm")
            tied, dual = build_paired_models(
                base,
                seed=26_000,
                device="cpu",
            )
            try:
                assert_matched_models(tied, dual)
                tied_optimizer = make_optimizer(tied)
                dual_optimizer = make_optimizer(dual)
                optimizer_match = compare_optimizer_signatures(
                    tied_optimizer, dual_optimizer
                ).matched
                del tied_optimizer
                del dual_optimizer
                matching_passed = optimizer_match
                candidate_config_sha256 = sha256_canonical_json(candidate)
                paired_initialization_digest = tied.initialization_digest()
                if paired_initialization_digest != dual.initialization_digest():
                    raise E26AGateBlocked(f"{candidate_id}: paired initialization digest differs")
                signature_sha256 = parameter_signature_hash(tied)
                if signature_sha256 != parameter_signature_hash(dual):
                    raise E26AGateBlocked(f"{candidate_id}: paired parameter signature differs")
                variant_rows: list[dict[str, Any]] = []
                for model in (tied, dual):
                    model.to(device)
                    variant_rows.append(
                        _measure_variant(
                            model,
                            candidate_id=candidate_id,
                            corpus=corpus,
                            tokenizer=tokenizer,
                            seed=260_100 + candidate_index,
                            global_sequences=global_sequences,
                            microbatch_candidates=tuple(
                                int(value) for value in throughput["microbatch_size_candidates"]
                            ),
                            warmup_steps=int(throughput["warmup_steps"]),
                            measured_steps=int(throughput["measured_steps"]),
                            run=run,
                            device=device,
                        )
                    )
                    model.to("cpu")
                    torch.cuda.empty_cache()
                diagnostics = [row["diagnostics"] for row in variant_rows]
                measurements.append(
                    CandidateMeasurement(
                        candidate_id=candidate_id,
                        parameter_count=dual.parameter_count(),
                        matching_passed=matching_passed,
                        numerical_passed=numerical_coverage[candidate_id],
                        tokens_per_second_by_variant={
                            str(row["variant"]): float(row["tokens_per_second"])
                            for row in variant_rows
                        },
                        checkpoint_bytes=max(
                            int(row["checkpoint"]["bytes"]) for row in variant_rows
                        ),
                        peak_allocated_bytes=max(
                            int(row["peak_allocated_bytes"]) for row in variant_rows
                        ),
                        peak_reserved_bytes=max(
                            int(row["peak_reserved_bytes"]) for row in variant_rows
                        ),
                        p50_step_seconds=max(
                            float(row["p50_step_seconds"]) for row in variant_rows
                        ),
                        p95_step_seconds=max(
                            float(row["p95_step_seconds"]) for row in variant_rows
                        ),
                        compile_seconds=sum(float(row["compile_seconds"]) for row in variant_rows),
                        graph_break_count=max(
                            int(row.get("graph_break_count", 0)) for row in diagnostics
                        ),
                        fallback_count=max(
                            int(row.get("fallback_count", 0)) for row in diagnostics
                        ),
                        context_length=context_length,
                        selected_microbatch_sequences=min(
                            int(row["microbatch_size"]) for row in variant_rows
                        ),
                        accumulation_steps=max(
                            int(row["accumulation_steps"]) for row in variant_rows
                        ),
                        measured_optimizer_steps=int(throughput["measured_steps"]),
                        descriptive_stability_steps=0,
                        model_config_sha256=candidate_config_sha256,
                        parameter_signature_sha256=signature_sha256,
                        paired_initialization_digest=paired_initialization_digest,
                        token_mix_bounded_discrepancy_passed=all(
                            bool(row["token_mix_bounded_discrepancy_passed"])
                            for row in variant_rows
                        ),
                    )
                )
                run.append(
                    "throughput_metrics.jsonl",
                    [
                        {
                            "candidate_id": candidate_id,
                            **row,
                        }
                        for row in variant_rows
                    ],
                )
            finally:
                tied.to("cpu")
                dual.to("cpu")
                del tied
                del dual
                torch.cuda.empty_cache()
        return tuple(measurements)

    def _pilot(
        self,
        admission: E26AGateAdmission,
        run: ArtifactRun,
        *,
        selection: Any,
        measurements: Sequence[CandidateMeasurement],
        corpus: TokenMemmap,
        tokenizer: ExternalScientificTokenizer,
        device: torch.device,
    ) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
        candidate = next(
            row
            for row in admission.config["model_candidates"]
            if row["id"] == selection.candidate_id
        )
        measurement = next(
            row for row in measurements if row.candidate_id == selection.candidate_id
        )
        base = _candidate_config(candidate, variant="dual_delta_lm")
        tied, dual = build_paired_models(
            base,
            seed=int(admission.calibration_config["paired_seed"]),
            device="cpu",
        )
        assert_matched_models(tied, dual)
        initial_digest = tied.initialization_digest()
        optimizer_config = admission.calibration_config["optimizer_candidate"]
        global_tokens = int(optimizer_config["target_global_batch_tokens"])
        context_length = int(candidate["context_length"])
        global_sequences = global_tokens // context_length
        microbatch_size = int(measurement.selected_microbatch_sequences)
        pilot_limit = int(admission.config["floor_gate"]["pilot_tokens_max"])
        steps = pilot_limit // global_tokens
        nominal_exposed_tokens = steps * global_tokens
        if steps <= 0 or nominal_exposed_tokens > pilot_limit:
            raise E26AGateBlocked("Pilot token budget is malformed")
        episodes = tuple(
            episode
            for episode in admission.validation_episodes
            if episode.operation in PRIMARY_OPERATIONS
        )
        expected_primary_count = int(
            admission.config["gate_population"]["items_per_operation_per_split"]
        ) * len(PRIMARY_OPERATIONS)
        if len(episodes) != expected_primary_count:
            raise E26AGateBlocked(
                "Locked validation population lacks the registered primary episodes"
            )
        validation_manifest = _general_validation_manifest(admission)
        validation_corpus = TokenMemmap.from_scientific_manifest(
            validation_manifest,
            tokenizer_manifest=tokenizer.manifest,
        )
        summaries: dict[str, Any] = {}
        cursor_digests: dict[str, str] = {}
        source_mix_by_variant: dict[str, dict[str, Any]] = {}
        final_model_digests: dict[str, str] = {}
        evaluation_rows: list[dict[str, Any]] = []
        gate_state: dict[str, Any] = {}
        general_ppl: dict[str, Any] = {}
        for model in (tied, dual):
            model.to(device)
            beta_values = tuple(float(value) for value in optimizer_config["betas"])
            if len(beta_values) != 2:
                raise E26AGateBlocked("AdamW beta lock must contain exactly two values")
            optimizer = make_optimizer(
                model,
                learning_rate=float(optimizer_config["learning_rate"]),
                weight_decay=float(optimizer_config["weight_decay"]),
                betas=(beta_values[0], beta_values[1]),
            )
            warmup_steps = max(1, math.ceil(float(optimizer_config["warmup_fraction"]) * steps))

            def lr_lambda(
                step_index: int,
                *,
                locked_warmup_steps: int = warmup_steps,
                locked_steps: int = steps,
            ) -> float:
                if step_index < locked_warmup_steps:
                    return (step_index + 1) / locked_warmup_steps
                progress = (step_index - locked_warmup_steps) / max(
                    1, locked_steps - locked_warmup_steps
                )
                return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            cursor = _paired_cursor(
                corpus,
                tokenizer,
                seed=int(admission.calibration_config["paired_seed"]),
                sequence_length=context_length,
            )
            receipt_hashes: list[str] = []
            general_sequences = 0
            transaction_sequences = 0
            source_token_totals = {
                "nonpadding_input_tokens": 0,
                "valid_prediction_tokens": 0,
                "general_nonpadding_input_tokens": 0,
                "transaction_nonpadding_input_tokens": 0,
                "general_valid_prediction_tokens": 0,
                "transaction_valid_prediction_tokens": 0,
            }
            losses: list[float] = []
            for step in range(1, steps + 1):
                metric, receipt, duration, accounting = _training_step(
                    model,
                    optimizer,
                    cursor,
                    global_sequences=global_sequences,
                    microbatch_size=microbatch_size,
                    device=device,
                    scheduler=scheduler,
                )
                receipt_hashes.append(str(receipt["data_order_sha256"]))
                general_sequences += int(receipt["general_sequences"])
                transaction_sequences += int(receipt["transaction_sequences"])
                for name, value in accounting.items():
                    source_token_totals[name] += int(value)
                losses.append(float(metric.loss))
                run.append(
                    "pilot_training_metrics.jsonl",
                    [
                        {
                            "variant": model.config.variant,
                            "step": step,
                            "tokens_seen": source_token_totals["nonpadding_input_tokens"],
                            **accounting,
                            "loss": metric.loss,
                            "grad_norm": metric.gradient_norm_before_clip,
                            "seconds": duration,
                            "data_order_sha256": receipt["data_order_sha256"],
                        }
                    ],
                )
            cursor_digests[model.config.variant] = sha256_canonical_json(receipt_hashes)
            total_prediction_tokens = source_token_totals["valid_prediction_tokens"]
            general_prediction_tokens = source_token_totals["general_valid_prediction_tokens"]
            transaction_prediction_tokens = source_token_totals[
                "transaction_valid_prediction_tokens"
            ]
            source_mix_by_variant[model.config.variant] = {
                "general_sequences": general_sequences,
                "transaction_sequences": transaction_sequences,
                "total_sequences": general_sequences + transaction_sequences,
                **source_token_totals,
                "realized_general_nonpadding_fraction": (
                    source_token_totals["general_nonpadding_input_tokens"]
                    / source_token_totals["nonpadding_input_tokens"]
                ),
                "realized_transaction_nonpadding_fraction": (
                    source_token_totals["transaction_nonpadding_input_tokens"]
                    / source_token_totals["nonpadding_input_tokens"]
                ),
                "realized_general_prediction_fraction": (
                    general_prediction_tokens / total_prediction_tokens
                ),
                "realized_transaction_prediction_fraction": (
                    transaction_prediction_tokens / total_prediction_tokens
                ),
                "abs_4t_minus_g": abs(
                    4 * source_token_totals["transaction_nonpadding_input_tokens"]
                    - source_token_totals["general_nonpadding_input_tokens"]
                ),
                "max_abs_4t_minus_g": 4 * context_length,
            }
            final_model_digests[model.config.variant] = state_dict_digest(model)
            summaries[model.config.variant], raw = _evaluate_pilot_model(
                model,
                tokenizer,
                episodes,
                device=device,
            )
            evaluation_rows.extend(raw)
            probe_cursor = _paired_cursor(
                corpus,
                tokenizer,
                seed=260_777,
                sequence_length=context_length,
            )
            probe_rows, _ = probe_cursor.take(max(1, microbatch_size))
            probe_batches, _, _ = _microbatches(
                probe_rows,
                microbatch_size=max(1, microbatch_size),
                device=device,
            )
            gate_state[model.config.variant] = _model_gate_state_metrics(model, probe_batches[0])
            general_ppl[model.config.variant] = _general_perplexity(
                model,
                validation_corpus,
                seed=260_888,
                sequence_length=context_length,
                minimum_tokens=int(admission.config["floor_gate"]["general_validation_tokens"]),
                microbatch_size=microbatch_size,
                device=device,
            )
            del optimizer
            del scheduler
            model.to("cpu")
            torch.cuda.empty_cache()
        run.append("pilot_evaluation_metrics.jsonl", evaluation_rows)
        observed_evaluation_splits = sorted({str(row["split"]) for row in evaluation_rows})
        expected_episode_ids = sorted(episode.episode_id for episode in episodes)
        observed_episode_ids = sorted({str(row["episode_id"]) for row in evaluation_rows})
        if (
            observed_evaluation_splits != ["validation"]
            or observed_episode_ids != expected_episode_ids
        ):
            raise E26AGateBlocked(
                "E26a evaluation access trace differs from the locked validation population"
            )
        pilot = {
            "schema_version": "catena-v8.1",
            "completed": True,
            "scientific_evidence": False,
            "claim_ceiling": "PROTOCOL_IDENTIFIABILITY_ONLY",
            "paired_seed": int(admission.calibration_config["paired_seed"]),
            "candidate_id": selection.candidate_id,
            "pilot_token_limit_per_variant": pilot_limit,
            "nominal_exposed_tokens_per_variant": nominal_exposed_tokens,
            "realized_nonpadding_tokens_per_variant": {
                variant: value["nonpadding_input_tokens"]
                for variant, value in source_mix_by_variant.items()
            },
            "realized_valid_prediction_tokens_per_variant": {
                variant: value["valid_prediction_tokens"]
                for variant, value in source_mix_by_variant.items()
            },
            "optimizer_steps_per_variant": steps,
            "global_batch_tokens": global_tokens,
            "microbatch_sequences": microbatch_size,
            "initialization_digest": initial_digest,
            "cursor_digest_by_variant": cursor_digests,
            "source_mix_by_variant": source_mix_by_variant,
            "final_model_digest_by_variant": final_model_digests,
            "operation_scores": summaries,
            "general_validation": general_ppl,
            "gate_and_state": gate_state,
            "evaluation_access": {
                "instrumentation": "PER_EVALUATION_ROW_SPLIT_AND_EPISODE_TRACE",
                "observed_splits": observed_evaluation_splits,
                "unique_episode_count": len(observed_episode_ids),
                "episode_ids_sha256": sha256_canonical_json(observed_episode_ids),
                "expected_episode_ids_sha256": sha256_canonical_json(expected_episode_ids),
                "evaluation_row_count": len(evaluation_rows),
                "main_test_access_count": 0,
                "heldout_domain_access_count": 0,
            },
            "main_test_opened": False,
        }
        gates: list[dict[str, Any]] = []
        threshold = float(
            admission.config["floor_gate"]["exact_minus_stale_score_min_each_primary_operation"]
        )
        for variant, operations in summaries.items():
            for operation in PRIMARY_OPERATIONS:
                observed = float(operations[operation]["exact_minus_stale"])
                gates.append(
                    _gate(
                        f"floor_headroom_{variant}_{operation.lower()}",
                        observed >= threshold,
                        observed,
                        f">={threshold}",
                    )
                )
            learned_query_metrics = {
                query_type: float(
                    statistics.fmean(
                        float(operations[operation]["query_accuracy"]["learned"][query_type])
                        for operation in PRIMARY_OPERATIONS
                    )
                )
                for query_type in ("current_state", "derived_action", "stale_probe")
            }
            nondegenerate_interval = tuple(
                float(value)
                for value in admission.config["floor_gate"][
                    "primary_query_metrics_nondegenerate_open_interval"
                ]
            )
            if len(nondegenerate_interval) != 2:
                raise E26AGateBlocked("Nondegenerate query-metric interval must have two endpoints")
            gates.append(
                _gate(
                    f"nondegenerate_primary_query_metrics_{variant}",
                    all(
                        nondegenerate_interval[0] < value < nondegenerate_interval[1]
                        for value in learned_query_metrics.values()
                    ),
                    learned_query_metrics,
                    (
                        "each current_state/derived_action/stale_probe accuracy in "
                        f"({nondegenerate_interval[0]},{nondegenerate_interval[1]})"
                    ),
                )
            )
        gates.extend(
            [
                _gate(
                    "paired_cursor_exact_token_match",
                    len(set(cursor_digests.values())) == 1,
                    cursor_digests,
                    "identical",
                ),
                _gate(
                    "paired_token_and_optimizer_step_match",
                    len(
                        {
                            value["nonpadding_input_tokens"]
                            for value in source_mix_by_variant.values()
                        }
                    )
                    == 1
                    and all(
                        value["nonpadding_input_tokens"] <= pilot_limit
                        for value in source_mix_by_variant.values()
                    )
                    and steps > 0,
                    {
                        "tokens_per_variant": {
                            variant: value["nonpadding_input_tokens"]
                            for variant, value in source_mix_by_variant.items()
                        },
                        "steps_per_variant": steps,
                    },
                    {"tokens_max": pilot_limit, "steps_equal": True},
                ),
                _gate(
                    "locked_80_20_target_bounded_token_discrepancy",
                    all(
                        bounded_80_20_nonpadding_token_mix(
                            value,
                            context_length=context_length,
                        )
                        for value in source_mix_by_variant.values()
                    ),
                    source_mix_by_variant,
                    {
                        "nominal_general_fraction": 0.8,
                        "nominal_transaction_fraction": 0.2,
                        "discrepancy": (
                            "abs(4 * transaction_nonpadding_input_tokens - "
                            "general_nonpadding_input_tokens) <= 4 * context_length"
                        ),
                        "exact_fraction_claim": False,
                    },
                ),
                _gate(
                    "finite_general_ppl",
                    all(
                        math.isfinite(float(value["perplexity"])) for value in general_ppl.values()
                    ),
                    {key: value["perplexity"] for key, value in general_ppl.items()},
                    "finite",
                ),
                _gate(
                    "gate_saturation_below_registered_max",
                    all(
                        float(value["gate_saturation_fraction"])
                        < float(admission.config["floor_gate"]["max_gate_saturation_fraction"])
                        for value in gate_state.values()
                    ),
                    {key: value["gate_saturation_fraction"] for key, value in gate_state.items()},
                    (f"<{admission.config['floor_gate']['max_gate_saturation_fraction']}"),
                ),
                _gate(
                    "finite_max_normalized_recurrent_state_frobenius",
                    all(
                        math.isfinite(float(value["state_norm_max_normalized_frobenius"]))
                        for value in gate_state.values()
                    ),
                    {
                        key: value["state_norm_max_normalized_frobenius"]
                        for key, value in gate_state.items()
                    },
                    "finite",
                ),
            ]
        )
        model_manifest = {
            "schema_version": "catena-v8.1",
            "candidate_id": selection.candidate_id,
            "model_config": base.to_dict(),
            "parameter_count": tied.parameter_count(),
            "parameter_signature_hash": parameter_signature_hash(tied),
            "initialization_digest": initial_digest,
            "optimizer_initial_signature": optimizer_state_signature(make_optimizer(tied)),
            "paired_variants": list(admission.config["variants"]),
            "final_model_digests": final_model_digests,
        }
        tied.to("cpu")
        dual.to("cpu")
        del tied
        del dual
        torch.cuda.empty_cache()
        return pilot, model_manifest, gates

    def execute(
        self,
        admission: E26AGateAdmission,
        run: ArtifactRun,
        *,
        device: torch.device,
    ) -> E26AExecutionResult:
        device = require_locked_execution_device(admission, device)
        tokenizer = ExternalScientificTokenizer.from_manifest(admission.paths.tokenizer_manifest)
        corpus = TokenMemmap.from_scientific_manifest(
            admission.paths.corpus_manifest,
            tokenizer_manifest=tokenizer.manifest,
        )
        measurements = self._measure_candidates(
            run,
            config=admission.config,
            numerical_audit=admission.numerical_audit,
            corpus=corpus,
            tokenizer=tokenizer,
            device=device,
        )
        prerequisite_gates = self._selection_prerequisite_gates(
            admission,
            measurements,
        )
        try:
            selection = select_candidate(
                config=admission.config,
                measurements=measurements,
                policy=admission.resource_policy,
            )
            require_locked_resource_selection(admission, selection)
        except E26AGateBlocked as error:
            return E26AExecutionResult(
                measurements=measurements,
                gates=tuple(prerequisite_gates),
                model_manifest={
                    "schema_version": "catena-v8.1",
                    "selected": False,
                    "candidate_measurements": [row.candidate_id for row in measurements],
                    "reason": str(error),
                },
                backend_manifest=self._backend_manifest_from_input(
                    admission,
                    measurements,
                ),
                data_manifest={
                    "schema_version": "catena-v8.1",
                    "scientific_evidence": False,
                    "data_readiness_sha256": admission.data_readiness["readiness_sha256"],
                    "main_test_opened": False,
                },
                pilot_summary={
                    "schema_version": "catena-v8.1",
                    "completed": False,
                    "skip_reason": "NO_RESOURCE_AND_CONTRACT_ELIGIBLE_CANDIDATE",
                    "detail": str(error),
                    "main_test_opened": False,
                },
            )
        pilot, model_manifest, pilot_gates = self._pilot(
            admission,
            run,
            selection=selection,
            measurements=measurements,
            corpus=corpus,
            tokenizer=tokenizer,
            device=device,
        )
        parameter_min = int(admission.config["candidate_selection"]["parameter_count_min"])
        parameter_max = int(admission.config["candidate_selection"]["parameter_count_max"])
        selected_measurement = next(
            row for row in measurements if row.candidate_id == selection.candidate_id
        )
        gates = [
            *prerequisite_gates,
            _gate(
                "selected_candidate_parameter_range",
                parameter_min <= selected_measurement.parameter_count <= parameter_max,
                {
                    "candidate_id": selection.candidate_id,
                    "parameter_count": selected_measurement.parameter_count,
                },
                f"[{parameter_min},{parameter_max}]",
            ),
            _gate(
                "selected_candidate_parameter_initialization_optimizer_matching",
                selected_measurement.matching_passed,
                {
                    "candidate_id": selection.candidate_id,
                    "passed": selected_measurement.matching_passed,
                },
                True,
            ),
            _gate(
                "arbitrary_partition_gradient_accumulation_restart_contract",
                selected_measurement.numerical_passed,
                {
                    "numerical_audit": admission.numerical_audit.get("receipt_sha256"),
                    "restart_audit": admission.restart_audit.get("receipt_sha256"),
                },
                "both frozen audits pass",
            ),
            _gate(
                "selected_compiled_backend_no_graph_break_or_fallback",
                selected_measurement.graph_break_count == 0
                and selected_measurement.fallback_count == 0,
                {
                    "candidate_id": selection.candidate_id,
                    "graph_break_count": selected_measurement.graph_break_count,
                    "fallback_count": selected_measurement.fallback_count,
                },
                {"graph_break_count": 0, "fallback_count": 0},
            ),
            _gate(
                "selected_candidate_80_20_target_bounded_token_discrepancy",
                selected_measurement.token_mix_bounded_discrepancy_passed,
                {
                    "candidate_id": selection.candidate_id,
                    "token_mix_bounded_discrepancy_passed": (
                        selected_measurement.token_mix_bounded_discrepancy_passed
                    ),
                },
                True,
            ),
            _gate(
                "throughput_deadline_storage_candidate_selection",
                True,
                selection.as_dict(),
                "first passing config-order candidate and largest eligible budget",
            ),
            *pilot_gates,
        ]
        backend_manifest = self._backend_manifest_from_input(
            admission,
            measurements,
        )
        data_manifest = {
            "schema_version": "catena-v8.1",
            "scientific_evidence": False,
            "data_readiness_sha256": admission.data_readiness["readiness_sha256"],
            "tokenizer_manifest_sha256": admission.input_hashes["tokenizer_manifest_sha256"],
            "general_train_manifest_sha256": admission.input_hashes["corpus_manifest_sha256"],
            "transaction_manifest_sha256": admission.input_hashes["transaction_manifest_sha256"],
            "validation_population_lock_sha256": admission.input_hashes[
                "validation_population_lock_sha256"
            ],
            "validation_population_records_sha256": (
                admission.validation_population_lock["records_sha256"]
            ),
            "schedule_manifest_sha256": admission.input_hashes["schedule_manifest_sha256"],
            "main_test_opened": False,
        }
        return E26AExecutionResult(
            measurements=measurements,
            gates=tuple(gates),
            model_manifest=model_manifest,
            backend_manifest=backend_manifest,
            data_manifest=data_manifest,
            pilot_summary=pilot,
        )


def _failure_result(
    admission: E26AGateAdmission,
    error: BaseException,
) -> E26AExecutionResult:
    return E26AExecutionResult(
        measurements=(),
        gates=(
            _gate(
                "scientific_e26a_executor_completed",
                False,
                f"{type(error).__name__}: {error}",
                "complete without operational error",
            ),
        ),
        model_manifest={
            "schema_version": "catena-v8.1",
            "not_available": True,
            "reason": type(error).__name__,
        },
        backend_manifest={
            "schema_version": "catena-v8.1",
            "source_input_sha256": admission.input_hashes.get("backend_manifest_sha256"),
            "executor_failed": True,
        },
        data_manifest={
            "schema_version": "catena-v8.1",
            "data_readiness_sha256": admission.data_readiness.get("readiness_sha256"),
            "main_test_opened": False,
        },
        pilot_summary={
            "schema_version": "catena-v8.1",
            "completed": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "main_test_opened": False,
        },
    )


def run_scientific_e26a(
    admission: E26AGateAdmission,
    *,
    device: torch.device | str,
    backend: E26AExecutionBackend | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Run and finalize E26a only, preserving any post-admission failure artifact."""

    resolved_device = require_locked_execution_device(admission, device)
    selected_backend = backend or RealE26AExecutionBackend()
    with exclusive_scientific_gate_lock(admission.artifact_root):
        run = create_scientific_gate_run(admission)
        try:
            result = selected_backend.execute(
                admission,
                run,
                device=resolved_device,
            )
        except Exception as error:
            result = _failure_result(admission, error)
        report = finalize_scientific_gate_run(
            run=run,
            admission=admission,
            measurements=result.measurements,
            gates=result.gates,
            model_manifest=result.model_manifest,
            backend_manifest=result.backend_manifest,
            data_manifest=result.data_manifest,
            pilot_summary=result.pilot_summary,
        )
        return run.run_dir, report
