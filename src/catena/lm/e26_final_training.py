"""Fail-closed training-layout primitives for the E26 Final speed preflight.

This module contains no official GDN-2 import and no scientific launcher.  It
defines the physical single-GPU training recipe, deterministic non-evidence
token stream, matched-variant receipts, and outcome-independent autotune rule
used by :mod:`tools.run_e26_final_speed_preflight`.

The four GPUs run four *independent model runs*.  Consequently the registered
global batch of 32 sequences is per run/GPU; a physical microbatch of ``m``
uses ``32 / m`` accumulation steps.  This is intentionally not DDP.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Final, Literal, TypeAlias, cast

import torch

from catena.core.provenance_v61 import sha256_canonical_json

E26FinalVariant: TypeAlias = Literal["tied", "dual"]

SEQUENCE_LENGTH: Final = 4_096
GLOBAL_BATCH_SEQUENCES: Final = 32
MICROBATCH_CANDIDATES: Final = (1, 2, 4, 8, 16)
AUTOTUNE_WARMUP_STEPS: Final = 5
AUTOTUNE_MEASURED_STEPS: Final = 10
SELECTED_LAYOUT_WARMUP_STEPS: Final = 5
SELECTED_LAYOUT_MEASURED_STEPS: Final = 200
VOCAB_SIZE: Final = 32_000
EXPECTED_FULL_PARAMETER_COUNT: Final = 1_450_096_416
EXPECTED_TRANSFORMER_H_PARAMETER_COUNT: Final = 1_302_638_112
PEAK_VRAM_GIB_MAX: Final = 92.0
DATA_FORMULA_ID: Final = "E26_FINAL_COUNTER_BASED_TOKEN_STREAM_V1"
OFFICIAL_KERNEL_ID: Final = "official_gdn2_chunk_gdn2_95709fc"
REGISTERED_VARIANTS: Final = ("tied", "dual")


class E26FinalTrainingError(RuntimeError):
    """Raised when a preflight input or matched-training invariant drifts."""


@dataclass(frozen=True, slots=True)
class PhysicalTrainingLayout:
    """One immutable single-GPU physical training layout."""

    sequence_length: int
    global_batch_sequences: int
    physical_microbatch_sequences: int
    gradient_accumulation_steps: int
    global_batch_tokens: int
    precision: str
    parameter_dtype: str
    optimizer_state_dtype: str
    loss_accumulation_dtype: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def digest(self) -> str:
        return sha256_canonical_json(self.as_dict())


@dataclass(frozen=True, slots=True)
class CandidateObservation:
    """Systems-only output from one fresh-process autotune candidate."""

    microbatch_sequences: int
    passed: bool
    disposition: str
    tokens_per_second: float | None
    peak_vram_gib: float | None
    receipt_path: str
    receipt_sha256: str


def registered_layout(microbatch_sequences: int) -> PhysicalTrainingLayout:
    """Construct a registered layout or reject an unregistered batch shape."""

    if (
        isinstance(microbatch_sequences, bool)
        or not isinstance(microbatch_sequences, int)
        or microbatch_sequences not in MICROBATCH_CANDIDATES
    ):
        raise E26FinalTrainingError(
            f"physical microbatch must be one of {list(MICROBATCH_CANDIDATES)}"
        )
    if GLOBAL_BATCH_SEQUENCES % microbatch_sequences != 0:
        raise E26FinalTrainingError("microbatch must divide the per-run global batch")
    return PhysicalTrainingLayout(
        sequence_length=SEQUENCE_LENGTH,
        global_batch_sequences=GLOBAL_BATCH_SEQUENCES,
        physical_microbatch_sequences=microbatch_sequences,
        gradient_accumulation_steps=GLOBAL_BATCH_SEQUENCES // microbatch_sequences,
        global_batch_tokens=SEQUENCE_LENGTH * GLOBAL_BATCH_SEQUENCES,
        precision="OFFICIAL_BF16_MIXED_AUTOCAST",
        parameter_dtype="torch.float32",
        optimizer_state_dtype="torch.float32",
        loss_accumulation_dtype="torch.float32",
    )


def deterministic_token_batch(
    *,
    seed: int,
    optimizer_step: int,
    microbatch_index: int,
    layout: PhysicalTrainingLayout,
    device: torch.device | str,
) -> torch.Tensor:
    """Generate one counter-based non-evidence speed batch.

    No mutable generator state is used, so Tied and Dual receive byte-identical
    IDs even when executed in distinct fresh processes.  IDs ``1..31999`` are
    used; the tokenizer's unknown ID 0 is deliberately absent.
    """

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise E26FinalTrainingError("seed must be a non-negative integer")
    if (
        isinstance(optimizer_step, bool)
        or not isinstance(optimizer_step, int)
        or optimizer_step < 0
    ):
        raise E26FinalTrainingError("optimizer_step must be a non-negative integer")
    if (
        isinstance(microbatch_index, bool)
        or not isinstance(microbatch_index, int)
        or not 0 <= microbatch_index < layout.gradient_accumulation_steps
    ):
        raise E26FinalTrainingError("microbatch_index is outside this layout")

    microbatch = layout.physical_microbatch_sequences
    first_sequence = (
        optimizer_step * layout.global_batch_sequences
        + microbatch_index * microbatch
    )
    sequence_ids = torch.arange(
        first_sequence,
        first_sequence + microbatch,
        dtype=torch.int64,
        device=device,
    ).unsqueeze(1)
    positions = torch.arange(
        layout.sequence_length + 1,
        dtype=torch.int64,
        device=device,
    ).unsqueeze(0)
    # Prime multipliers make every sequence/position deterministic without a
    # Python token loop.  The calculation is vectorized on the selected GPU.
    return ((sequence_ids * 104_729 + positions * 8_191 + seed) % (VOCAB_SIZE - 1)) + 1


def token_plan_digest(
    *,
    seed: int,
    layout: PhysicalTrainingLayout,
    warmup_steps: int,
    measured_steps: int,
) -> str:
    """Hash the complete deterministic token exposure plan, not outcome data."""

    for name, value in (("warmup_steps", warmup_steps), ("measured_steps", measured_steps)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise E26FinalTrainingError(f"{name} must be a non-negative integer")
    return sha256_canonical_json(
        {
            "formula": DATA_FORMULA_ID,
            "seed": seed,
            "layout": layout.as_dict(),
            "warmup_steps": warmup_steps,
            "measured_steps": measured_steps,
            "total_optimizer_steps": warmup_steps + measured_steps,
        }
    )


def parameter_inventory(model: torch.nn.Module) -> dict[str, Any]:
    """Return a value-independent parameter-surface inventory and digest."""

    rows: list[dict[str, Any]] = []
    total = 0
    transformer_h_total = 0
    b_projection_shapes: dict[str, list[int]] = {}
    w_projection_shapes: dict[str, list[int]] = {}
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        total += count
        if name.startswith("transformer.h."):
            transformer_h_total += count
        row = {
            "name": name,
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
            "requires_grad": parameter.requires_grad,
            "numel": count,
        }
        rows.append(row)
        if name.endswith("attn.b_proj.weight"):
            b_projection_shapes[name.removesuffix("b_proj.weight")] = list(parameter.shape)
        if name.endswith("attn.w_proj.weight"):
            w_projection_shapes[name.removesuffix("w_proj.weight")] = list(parameter.shape)

    projections_match = bool(b_projection_shapes) and b_projection_shapes == w_projection_shapes
    payload: dict[str, Any] = {
        "parameter_count": total,
        "transformer_h_parameter_count": transformer_h_total,
        "parameter_tensor_count": len(rows),
        "parameter_surface_sha256": sha256_canonical_json(rows),
        "gate_projection_layer_count": len(b_projection_shapes),
        "gate_projection_shapes_match": projections_match,
        "all_parameters_fp32": all(row["dtype"] == "torch.float32" for row in rows),
    }
    payload["passed"] = (
        total == EXPECTED_FULL_PARAMETER_COUNT
        and transformer_h_total == EXPECTED_TRANSFORMER_H_PARAMETER_COUNT
        and len(b_projection_shapes) == 18
        and projections_match
        and payload["all_parameters_fp32"] is True
    )
    return payload


_PARITY_FIELDS: Final = (
    "parameter_surface_sha256",
    "parameter_count",
    "transformer_h_parameter_count",
    "optimizer_surface_sha256",
    "initialization_sha256",
    "layout_sha256",
    "token_plan_sha256",
    "checkpoint_sha256",
    "official_source_commit",
    "official_runtime_source_sha256",
    "precision",
)


def validate_matched_variant_receipts(
    tied: Mapping[str, Any],
    dual: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove that a Tied/Dual speed pair differs only in gate policy."""

    differences = [field for field in _PARITY_FIELDS if tied.get(field) != dual.get(field)]
    variant_exact = tied.get("variant") == "tied" and dual.get("variant") == "dual"
    checks = {
        "registered_variant_pair": variant_exact,
        "all_registered_matching_fields_equal": not differences,
        "tied_worker_passed": tied.get("passed") is True,
        "dual_worker_passed": dual.get("passed") is True,
    }
    return {
        "matching_fields": list(_PARITY_FIELDS),
        "differing_fields": differences,
        "hard_checks": checks,
        "passed": all(checks.values()),
    }


def select_autotune_candidate(
    observations: Sequence[CandidateObservation],
) -> CandidateObservation:
    """Select solely by measured throughput after OOM/VRAM admission.

    A throughput tie is resolved by the smaller active microbatch, which is
    conservative in memory and does not inspect any task or loss outcome.
    """

    expected = list(MICROBATCH_CANDIDATES)
    observed = [row.microbatch_sequences for row in observations]
    if sorted(observed) != expected or len(set(observed)) != len(expected):
        raise E26FinalTrainingError(
            "autotune must contain exactly one fresh-process result per registered candidate"
        )
    eligible: list[CandidateObservation] = []
    for row in observations:
        if not row.passed:
            continue
        throughput = row.tokens_per_second
        memory = row.peak_vram_gib
        if (
            throughput is None
            or memory is None
            or not math.isfinite(throughput)
            or not math.isfinite(memory)
            or throughput <= 0.0
            or memory < 0.0
        ):
            raise E26FinalTrainingError("passing candidate has invalid systems measurements")
        if memory <= PEAK_VRAM_GIB_MAX:
            eligible.append(row)
    if not eligible:
        raise E26FinalTrainingError("no microbatch candidate passed OOM/VRAM admission")
    return max(
        eligible,
        key=lambda row: (
            cast(float, row.tokens_per_second),
            -row.microbatch_sequences,
        ),
    )


def telemetry_summary(samples: Sequence[Mapping[str, float]]) -> dict[str, float | int]:
    """Summarize physical-GPU telemetry without manufacturing missing samples."""

    if not samples:
        raise E26FinalTrainingError("GPU telemetry has no valid samples")
    utilization = [float(row["utilization_percent"]) for row in samples]
    power = [float(row["power_watts"]) for row in samples]
    memory = [float(row["memory_used_mib"]) for row in samples]
    if any(
        not math.isfinite(value) or value < 0.0
        for value in (*utilization, *power, *memory)
    ):
        raise E26FinalTrainingError("GPU telemetry contains invalid values")
    return {
        "sample_count": len(samples),
        "median_utilization_percent": float(statistics.median(utilization)),
        "mean_power_watts": float(statistics.fmean(power)),
        "peak_nvidia_smi_memory_gib": max(memory) / 1024.0,
    }


__all__ = [
    "AUTOTUNE_MEASURED_STEPS",
    "AUTOTUNE_WARMUP_STEPS",
    "CandidateObservation",
    "DATA_FORMULA_ID",
    "E26FinalTrainingError",
    "E26FinalVariant",
    "GLOBAL_BATCH_SEQUENCES",
    "MICROBATCH_CANDIDATES",
    "OFFICIAL_KERNEL_ID",
    "PhysicalTrainingLayout",
    "REGISTERED_VARIANTS",
    "SELECTED_LAYOUT_MEASURED_STEPS",
    "SELECTED_LAYOUT_WARMUP_STEPS",
    "SEQUENCE_LENGTH",
    "deterministic_token_batch",
    "parameter_inventory",
    "registered_layout",
    "select_autotune_candidate",
    "telemetry_summary",
    "token_plan_digest",
    "validate_matched_variant_receipts",
]
