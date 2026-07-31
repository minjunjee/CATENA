from __future__ import annotations

import math
import os
import statistics
import time
from collections.abc import Iterable, Iterator, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from catena.core.provenance_v61 import sha256_file

from .hashing import optimizer_state_signature, tensor_tree_digest
from .model import CatenaLM, cross_entropy_loss


@dataclass(frozen=True)
class TrainStepMetric:
    step: int
    tokens_seen: int
    loss: float
    grad_norm: float
    seconds: float
    tokens_per_second: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "step": self.step,
            "tokens_seen": self.tokens_seen,
            "loss": self.loss,
            "grad_norm": self.grad_norm,
            "seconds": self.seconds,
            "tokens_per_second": self.tokens_per_second,
        }


@dataclass(frozen=True)
class PairedOptimizerSignatures:
    left: str
    right: str
    matched: bool


@dataclass(frozen=True)
class NonEvidenceSmokeSummary:
    """Measured runtime properties from a fixed-step, non-evidence GPU smoke."""

    warmup_steps: int
    measured_steps: int
    sequence_length: int
    batch_size: int
    tokens_seen: int
    tokens_per_second: float
    step_time_p50_seconds: float
    step_time_p95_seconds: float
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    loss_first: float
    loss_last: float
    max_grad_norm: float
    compile_or_first_step_seconds: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "warmup_steps": self.warmup_steps,
            "measured_steps": self.measured_steps,
            "sequence_length": self.sequence_length,
            "batch_size": self.batch_size,
            "tokens_seen": self.tokens_seen,
            "tokens_per_second": self.tokens_per_second,
            "step_time_p50_seconds": self.step_time_p50_seconds,
            "step_time_p95_seconds": self.step_time_p95_seconds,
            "peak_allocated_bytes": self.peak_allocated_bytes,
            "peak_reserved_bytes": self.peak_reserved_bytes,
            "loss_first": self.loss_first,
            "loss_last": self.loss_last,
            "max_grad_norm": self.max_grad_norm,
            "compile_or_first_step_seconds": self.compile_or_first_step_seconds,
        }


@dataclass
class GradAccumulationStep:
    """One global-token optimizer update and its execution-semantics receipt."""

    loss: float
    valid_prediction_tokens: int
    exposed_input_tokens: int
    microbatch_count: int
    gradient_norm_before_clip: float
    clip_coefficient: float
    gradient_digest_before_clip: str
    learning_rates_after_step: tuple[float, ...]
    gradients_before_clip: dict[str, torch.Tensor] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "loss": self.loss,
            "valid_prediction_tokens": self.valid_prediction_tokens,
            "exposed_input_tokens": self.exposed_input_tokens,
            "microbatch_count": self.microbatch_count,
            "gradient_norm_before_clip": self.gradient_norm_before_clip,
            "clip_coefficient": self.clip_coefficient,
            "gradient_digest_before_clip": self.gradient_digest_before_clip,
            "learning_rates_after_step": list(self.learning_rates_after_step),
        }


def make_optimizer(
    model: CatenaLM,
    *,
    learning_rate: float = 3.0e-4,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=betas,
    )


def grad_global_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    squares = []
    for parameter in parameters:
        if parameter.grad is not None:
            squares.append(parameter.grad.detach().float().pow(2).sum())
    if not squares:
        return 0.0
    return float(torch.sqrt(torch.stack(squares).sum()).item())


def autoregressive_loss_sum(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    *,
    loss_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return summed next-token CE and its (possibly weighted) denominator."""

    if logits.shape[:2] != input_ids.shape:
        raise ValueError("logits and input_ids sequence shapes differ")
    if input_ids.shape[1] <= 1:
        raise ValueError("autoregressive loss requires at least two tokens")
    per_token = F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        input_ids[:, 1:].reshape(-1),
        reduction="none",
    ).view(input_ids.shape[0], input_ids.shape[1] - 1)
    if loss_mask is None:
        weights = torch.ones_like(per_token)
    else:
        if loss_mask.shape == input_ids.shape:
            weights = loss_mask[:, 1:].to(device=per_token.device, dtype=per_token.dtype)
        elif loss_mask.shape == per_token.shape:
            weights = loss_mask.to(device=per_token.device, dtype=per_token.dtype)
        else:
            raise ValueError("loss_mask must match input_ids or shifted target shape")
        if not bool(torch.isfinite(weights).all().item()) or bool((weights < 0).any().item()):
            raise ValueError("loss_mask weights must be finite and non-negative")
    denominator = weights.sum()
    if float(denominator.detach().item()) <= 0:
        raise ValueError("loss_mask selects no prediction tokens")
    return (per_token * weights).sum(), denominator


def optimizer_step_microbatches(
    model: CatenaLM,
    microbatches: Sequence[torch.Tensor],
    *,
    optimizer: torch.optim.Optimizer,
    scheduler: Any = None,
    loss_masks: Sequence[torch.Tensor | None] | None = None,
    grad_clip_norm: float = 1.0,
    autocast_dtype: torch.dtype | None = None,
    capture_gradients: bool = False,
) -> GradAccumulationStep:
    """Apply one optimizer update for one fixed global token batch.

    Every microbatch loss is normalized by the total number of valid prediction
    tokens. Gradient clipping, AdamW, and scheduler advancement happen exactly
    once, independent of microbatch count.
    """

    if not microbatches:
        raise ValueError("At least one microbatch is required")
    if grad_clip_norm <= 0:
        raise ValueError("grad_clip_norm must be positive")
    masks = list(loss_masks) if loss_masks is not None else [None] * len(microbatches)
    if len(masks) != len(microbatches):
        raise ValueError("loss_masks and microbatches must have equal length")
    device = next(model.parameters()).device
    if any(batch.device != device for batch in microbatches):
        raise ValueError("All microbatches and model parameters must share one device")

    valid_counts: list[torch.Tensor] = []
    for batch, mask in zip(microbatches, masks, strict=True):
        if batch.ndim != 2 or batch.shape[1] <= 1:
            raise ValueError("Each microbatch must have shape [batch, sequence>1]")
        target_shape = (batch.shape[0], batch.shape[1] - 1)
        if mask is None:
            valid_counts.append(torch.tensor(target_shape[0] * target_shape[1], device=device))
        else:
            shifted = mask[:, 1:] if mask.shape == batch.shape else mask
            if shifted.shape != target_shape:
                raise ValueError("loss_mask must match input_ids or shifted target shape")
            shifted = shifted.to(device=device, dtype=torch.float32)
            if not bool(torch.isfinite(shifted).all().item()) or bool((shifted < 0).any().item()):
                raise ValueError("loss_mask weights must be finite and non-negative")
            valid_counts.append(shifted.sum())
    total_valid = torch.stack([value.float() for value in valid_counts]).sum()
    if float(total_valid.item()) <= 0:
        raise ValueError("Global batch contains no valid prediction tokens")

    optimizer.zero_grad(set_to_none=True)
    detached_loss_sum = 0.0
    model.train()
    for batch, mask in zip(microbatches, masks, strict=True):
        context = (
            torch.autocast(device_type=device.type, dtype=autocast_dtype)
            if autocast_dtype is not None
            else nullcontext()
        )
        with context:
            output = model(batch)
            loss_sum, _ = autoregressive_loss_sum(output.logits, batch, loss_mask=mask)
            normalized_loss = loss_sum / total_valid
        normalized_loss.backward()  # type: ignore[no-untyped-call]
        detached_loss_sum += float(loss_sum.detach().float().item())

    gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    if len(gradients) != sum(1 for _ in model.parameters()):
        raise RuntimeError("One or more registered parameters received no gradient")
    if not all(bool(torch.isfinite(value).all().item()) for value in gradients.values()):
        raise FloatingPointError("Non-finite gradient in accumulated optimizer step")
    norm = grad_global_norm(model.parameters())
    clip_coefficient = min(1.0, grad_clip_norm / (norm + 1.0e-6))
    gradient_digest = tensor_tree_digest(gradients)
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return GradAccumulationStep(
        loss=detached_loss_sum / float(total_valid.item()),
        valid_prediction_tokens=int(total_valid.item()),
        exposed_input_tokens=sum(int(batch.numel()) for batch in microbatches),
        microbatch_count=len(microbatches),
        gradient_norm_before_clip=norm,
        clip_coefficient=clip_coefficient,
        gradient_digest_before_clip=gradient_digest,
        learning_rates_after_step=tuple(float(group["lr"]) for group in optimizer.param_groups),
        gradients_before_clip=gradients if capture_gradients else None,
    )


def train_reference_steps(
    model: CatenaLM,
    batches: Iterator[torch.Tensor],
    *,
    steps: int,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip_norm: float = 1.0,
) -> tuple[list[TrainStepMetric], torch.optim.Optimizer]:
    """Tiny non-evidence training loop for packet validation only."""

    if model.config.backend_scientific_main_capable:
        raise ValueError("Reference smoke helper is not intended for a scientific backend")
    optimizer = optimizer or make_optimizer(model)
    metrics: list[TrainStepMetric] = []
    tokens_seen = 0
    model.train()
    for step in range(1, steps + 1):
        batch = next(batches)
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        output = model(batch)
        loss = cross_entropy_loss(output.logits, batch)
        loss.backward()  # type: ignore[no-untyped-call]
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        duration = max(time.perf_counter() - start, 1.0e-12)
        token_count = int(batch.numel())
        tokens_seen += token_count
        metrics.append(
            TrainStepMetric(
                step=step,
                tokens_seen=tokens_seen,
                loss=float(loss.detach().item()),
                grad_norm=float(norm.detach().item() if torch.is_tensor(norm) else norm),
                seconds=duration,
                tokens_per_second=token_count / duration,
            )
        )
    return metrics, optimizer


def compare_optimizer_signatures(
    left: torch.optim.Optimizer,
    right: torch.optim.Optimizer,
) -> PairedOptimizerSignatures:
    left_signature = optimizer_state_signature(left)
    right_signature = optimizer_state_signature(right)
    return PairedOptimizerSignatures(
        left=left_signature,
        right=right_signature,
        matched=left_signature == right_signature,
    )


def cycle_tensor_batches(
    sequences: Sequence[Sequence[int]],
    *,
    batch_size: int,
    device: torch.device | str,
) -> Iterator[torch.Tensor]:
    if not sequences:
        raise ValueError("At least one sequence is required")
    length = len(sequences[0])
    if any(len(sequence) != length for sequence in sequences):
        raise ValueError("All sequences must have equal length")
    index = 0
    while True:
        selected = []
        for _ in range(batch_size):
            selected.append(sequences[index % len(sequences)])
            index += 1
        yield torch.tensor(selected, dtype=torch.long, device=device)


def projected_training_time_seconds(total_tokens: int, measured_tokens_per_second: float) -> float:
    if total_tokens <= 0 or measured_tokens_per_second <= 0:
        raise ValueError("total_tokens and measured_tokens_per_second must be positive")
    return total_tokens / measured_tokens_per_second


def finite_training_metrics(metrics: Sequence[TrainStepMetric]) -> bool:
    return bool(metrics) and all(
        math.isfinite(item.loss)
        and math.isfinite(item.grad_norm)
        and math.isfinite(item.tokens_per_second)
        for item in metrics
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("Cannot compute a percentile of an empty sequence")
    ordered = sorted(values)
    position = min(len(ordered) - 1, max(0, math.ceil(probability * len(ordered)) - 1))
    return float(ordered[position])


def train_non_evidence_smoke(
    model: CatenaLM,
    batches: Iterator[torch.Tensor],
    *,
    warmup_steps: int,
    measured_steps: int,
    grad_clip_norm: float = 1.0,
    learning_rate: float = 3.0e-4,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[NonEvidenceSmokeSummary, torch.optim.Optimizer]:
    """Run a fixed-step GPU feasibility smoke with no scientific disposition.

    The caller must write artifacts under ``/tmp`` and mark them
    ``NON_EVIDENCE_VALIDATION``.  This routine deliberately does not select a
    checkpoint or alter any prospective scientific threshold.
    """

    if warmup_steps < 0 or measured_steps <= 0:
        raise ValueError("warmup_steps must be non-negative and measured_steps positive")
    device = next(model.parameters()).device
    if device.type != "cuda":
        raise ValueError("The 100-step feasibility smoke requires a CUDA device")
    optimizer = optimizer or make_optimizer(model, learning_rate=learning_rate)
    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    durations: list[float] = []
    losses: list[float] = []
    grad_norms: list[float] = []
    compile_or_first = 0.0
    total_steps = warmup_steps + measured_steps
    sequence_length = 0
    batch_size = 0
    for index in range(total_steps):
        batch = next(batches)
        batch_size, sequence_length = (int(batch.shape[0]), int(batch.shape[1]))
        _synchronize(device)
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(batch)
            loss = cross_entropy_loss(output.logits, batch)
        loss.backward()  # type: ignore[no-untyped-call]
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        _synchronize(device)
        duration = time.perf_counter() - start
        if index == 0:
            compile_or_first = duration
        if index >= warmup_steps:
            durations.append(duration)
            losses.append(float(loss.detach().item()))
            grad_norms.append(float(norm.detach().item()))
    if not all(math.isfinite(value) for value in [*durations, *losses, *grad_norms]):
        raise FloatingPointError("Non-finite value in GPU smoke metrics")
    tokens = measured_steps * batch_size * sequence_length
    elapsed = sum(durations)
    return (
        NonEvidenceSmokeSummary(
            warmup_steps=warmup_steps,
            measured_steps=measured_steps,
            sequence_length=sequence_length,
            batch_size=batch_size,
            tokens_seen=tokens,
            tokens_per_second=tokens / elapsed,
            step_time_p50_seconds=float(statistics.median(durations)),
            step_time_p95_seconds=_percentile(durations, 0.95),
            peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
            peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
            loss_first=losses[0],
            loss_last=losses[-1],
            max_grad_norm=max(grad_norms),
            compile_or_first_step_seconds=compile_or_first,
        ),
        optimizer,
    )


def measure_checkpoint_io(
    model: CatenaLM,
    optimizer: torch.optim.Optimizer,
    destination: str | Path,
) -> dict[str, float | int | str]:
    """Measure local checkpoint round-trip without changing the live model."""

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if next(model.parameters()).is_cuda:
        torch.cuda.synchronize(next(model.parameters()).device)
    save_start = time.perf_counter()
    torch.save(payload, path)
    with path.open("rb") as handle:
        os.fsync(handle.fileno())
    save_seconds = time.perf_counter() - save_start
    load_start = time.perf_counter()
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    load_seconds = time.perf_counter() - load_start
    if set(loaded) != {"model", "optimizer"}:
        raise RuntimeError("Checkpoint round-trip lost required top-level keys")
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "save_seconds": save_seconds,
        "load_seconds": load_seconds,
        "durability": "file_fsync_before_timing_stop",
    }
