from __future__ import annotations

import math
import os
import statistics
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from catena.core.provenance_v61 import sha256_file

from .hashing import optimizer_state_signature
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
