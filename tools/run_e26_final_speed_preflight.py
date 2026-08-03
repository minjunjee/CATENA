#!/usr/bin/env python3
"""Run the non-evidence E26 Final official-GDN2 speed/layout preflight.

The parent process launches every microbatch candidate in a fresh interpreter,
selects by systems throughput only, then runs the selected physical layout for
exactly 200 measured optimizer steps in matched Tied/Dual cells on GPUs 0--3.
It never opens scientific data and never starts E26 Final main training.
"""

from __future__ import annotations

import argparse
import functools
import gc
import importlib
import os
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Final, cast

import torch
import torch.nn.functional as F

from catena.core.provenance_v61 import (
    read_json_object_strict,
    sha256_canonical_json,
    sha256_file,
    write_json_strict,
)
from catena.lm.e26_final_resources import (
    SpeedObservation,
    registered_final_resource_policy,
    validate_speed_preflight,
)
from catena.lm.e26_final_training import (
    AUTOTUNE_MEASURED_STEPS,
    AUTOTUNE_WARMUP_STEPS,
    MICROBATCH_CANDIDATES,
    OFFICIAL_KERNEL_ID,
    SELECTED_LAYOUT_MEASURED_STEPS,
    SELECTED_LAYOUT_WARMUP_STEPS,
    CandidateObservation,
    deterministic_token_batch,
    parameter_inventory,
    registered_layout,
    select_autotune_candidate,
    telemetry_summary,
    token_plan_digest,
    validate_matched_variant_receipts,
)
from tools.audit_e26_final_checkpoint import (
    OFFICIAL_CHECKPOINT_SPEC,
    safe_load_checkpoint,
)

ACK_ENV: Final = "CATENA_E26_FINAL_PREFLIGHT_ACK"
ACK_VALUE: Final = "E26_FINAL_NON_EVIDENCE_SPEED_PREFLIGHT_AUTHORIZED"
SCHEMA: Final = "catena-e26-final-speed-preflight-v1"
WORKER_SCHEMA: Final = "catena-e26-final-speed-worker-v1"
PATCHED_GATE_RELATIVE_PATH: Final = Path("lit_gpt/gdn2.py")
OFFICIAL_COMMIT: Final = "95709fc250357c2dd109361c353192f2aa5913f9"
PINNED_FLA_COMMIT: Final = "4b02d15d6a68700181b180235be62a9fb95d2a38"
CHECKPOINT_SHA256: Final = (
    "0322ebeefa96badb24d6b4b511c36b02374b704dc1a65b90eab2ee1383a9ce23"
)
EXPECTED_LAYER_COUNT: Final = 18


class E26FinalSpeedPreflightError(RuntimeError):
    """Raised when the official preflight cannot satisfy its hard contract."""


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite E26 Final receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_strict(path, dict(payload))


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise E26FinalSpeedPreflightError(result.stderr.strip() or "git command failed")
    return result.stdout.strip()


def _require_clean_committed_source(repository: Path) -> dict[str, Any]:
    root = repository.expanduser().resolve(strict=True)
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise E26FinalSpeedPreflightError(
            "Canonical speed preflight requires a clean committed E26 Final worktree"
        )
    return {
        "repository": str(root),
        "commit": _git(root, "rev-parse", "HEAD"),
        "branch": _git(root, "branch", "--show-current"),
        "dirty": False,
    }


def _require_admission_receipts(
    *, checkpoint_audit: Path, runtime_audit: Path, gate_patch_receipt: Path
) -> dict[str, Any]:
    checkpoint = read_json_object_strict(checkpoint_audit)
    runtime = read_json_object_strict(runtime_audit)
    patch = read_json_object_strict(gate_patch_receipt)
    if not isinstance(checkpoint, Mapping) or not isinstance(runtime, Mapping) or not isinstance(
        patch, Mapping
    ):
        raise E26FinalSpeedPreflightError("Admission receipts must be JSON objects")
    checks = {
        "checkpoint_passed": checkpoint.get("passed") is True,
        "checkpoint_eligible": checkpoint.get("checkpoint_admission_eligible") is True,
        "runtime_passed": runtime.get("passed") is True,
        "chunk_training_eligible": runtime.get("official_training_chunk_runtime_eligible")
        is True,
        "patch_applied": patch.get("status") == "APPLIED",
        "patch_commit_exact": patch.get("official_commit") == OFFICIAL_COMMIT,
        "patch_kernel_calls_unmodified": patch.get("kernel_calls_modified") is False,
    }
    if not all(checks.values()):
        failed = sorted(key for key, passed in checks.items() if not passed)
        raise E26FinalSpeedPreflightError(
            f"Required admission receipt failed: {', '.join(failed)}"
        )
    return {
        "checkpoint_audit": str(checkpoint_audit.resolve(strict=True)),
        "checkpoint_audit_sha256": sha256_file(checkpoint_audit),
        "runtime_audit": str(runtime_audit.resolve(strict=True)),
        "runtime_audit_sha256": sha256_file(runtime_audit),
        "gate_patch_receipt": str(gate_patch_receipt.resolve(strict=True)),
        "gate_patch_receipt_sha256": sha256_file(gate_patch_receipt),
        "hard_checks": checks,
        "passed": True,
    }


def _require_external_source_state(
    *, runtime_root: Path, fla_root: Path, gate_patch_receipt: Path
) -> dict[str, Any]:
    runtime = runtime_root.expanduser().resolve(strict=True)
    fla = fla_root.expanduser().resolve(strict=True)
    status_lines = _git(runtime, "status", "--porcelain=v1", "--untracked-files=all").splitlines()
    patch = read_json_object_strict(gate_patch_receipt)
    if not isinstance(patch, Mapping):
        raise E26FinalSpeedPreflightError("Gate patch receipt must be an object")
    target = runtime / PATCHED_GATE_RELATIVE_PATH
    checks = {
        "runtime_head_exact": _git(runtime, "rev-parse", "HEAD") == OFFICIAL_COMMIT,
        "only_gate_source_modified": len(status_lines) == 1
        and status_lines[0][3:] == PATCHED_GATE_RELATIVE_PATH.as_posix(),
        "gate_source_hash_matches_receipt": sha256_file(target)
        == patch.get("patched_file_sha256"),
        "no_python_cache_in_runtime": not any(
            "__pycache__" in row or row.endswith(".pyc") for row in status_lines
        ),
        "fla_head_exact": _git(fla, "rev-parse", "HEAD") == PINNED_FLA_COMMIT,
        "fla_checkout_clean": _git(
            fla, "status", "--porcelain=v1", "--untracked-files=all"
        )
        == "",
    }
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise E26FinalSpeedPreflightError(
            f"External runtime source binding failed: {', '.join(failed)}"
        )
    return {
        "runtime_root": str(runtime),
        "runtime_status": status_lines,
        "runtime_gate_source_sha256": sha256_file(target),
        "fla_root": str(fla),
        "hard_checks": checks,
        "passed": True,
    }


def _prepend_import_roots(runtime_root: Path, fla_root: Path) -> None:
    for path in (fla_root.resolve(strict=True), runtime_root.resolve(strict=True)):
        text = str(path)
        with suppress(ValueError):
            sys.path.remove(text)
        sys.path.insert(0, text)


def _load_official_model(checkpoint: Path, runtime_root: Path) -> torch.nn.Module:
    # Imported only after the pinned runtime is at sys.path[0].
    from tools.audit_e26_final_checkpoint import resolve_official_gpt_factory

    payload = safe_load_checkpoint(checkpoint)
    state = payload.get("model")
    if not isinstance(state, Mapping) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state.items()
    ):
        raise E26FinalSpeedPreflightError("Checkpoint model state is malformed")
    factory, binding = resolve_official_gpt_factory(runtime_root)
    if binding.get("passed") is not True:
        raise E26FinalSpeedPreflightError("Official GPT factory binding failed")
    model = factory()
    incompatible = model.load_state_dict(
        cast(Mapping[str, torch.Tensor], state), strict=True, assign=True
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise E26FinalSpeedPreflightError("Strict official checkpoint load drifted")
    del state
    del payload
    gc.collect()
    return model


def _configure_official_layers(model: torch.nn.Module, variant: str) -> list[torch.nn.Module]:
    transformer = getattr(model, "transformer", None)
    blocks = getattr(transformer, "h", None)
    if not isinstance(blocks, torch.nn.ModuleList) or len(blocks) != EXPECTED_LAYER_COUNT:
        raise E26FinalSpeedPreflightError("Official transformer.h layer population changed")
    policy = {"tied": "projected_tied_gdn2", "dual": "dual_gdn2"}.get(variant)
    if policy is None:
        raise E26FinalSpeedPreflightError(f"Unknown speed variant: {variant}")
    layers: list[torch.nn.Module] = []
    for index, block in enumerate(blocks):
        layer = getattr(block, "attn", None)
        if not isinstance(layer, torch.nn.Module):
            raise E26FinalSpeedPreflightError("Official block lacks GDN-2 attention module")
        b_proj = getattr(layer, "b_proj", None)
        w_proj = getattr(layer, "w_proj", None)
        if not isinstance(b_proj, torch.nn.Linear) or not isinstance(w_proj, torch.nn.Linear):
            raise E26FinalSpeedPreflightError("Official layer gate projections changed")
        if b_proj.weight.shape != w_proj.weight.shape:
            raise E26FinalSpeedPreflightError("b_proj/w_proj shapes differ")
        layer.__dict__["e26_gate_policy"] = policy
        layer.__dict__["allow_neg_eigval"] = False
        layer.__dict__["mode"] = "chunk"
        layer.__dict__["layer_idx"] = index
        layers.append(layer)
    return layers


class ChunkDispatchCounter(AbstractContextManager["ChunkDispatchCounter"]):
    """Count exact official chunk calls; there is no retry/fallback branch."""

    def __init__(self, module: ModuleType) -> None:
        self.module = module
        self.original: Callable[..., Any] | None = None
        self.calls = 0
        self.completed = 0

    def __enter__(self) -> ChunkDispatchCounter:
        function = getattr(self.module, "chunk_gdn2", None)
        if not callable(function):
            raise E26FinalSpeedPreflightError("Official chunk_gdn2 symbol is unavailable")
        self.original = cast(Callable[..., Any], function)

        @functools.wraps(self.original)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            self.calls += 1
            result = cast(Callable[..., Any], self.original)(*args, **kwargs)
            self.completed += 1
            return result

        self.module.__dict__["chunk_gdn2"] = wrapped
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> None:
        if self.original is not None:
            self.module.__dict__["chunk_gdn2"] = self.original


class NvidiaSmiSampler:
    """Poll one physical GPU without influencing the torch device mapping."""

    def __init__(self, physical_gpu_index: int, *, interval_seconds: float = 0.2) -> None:
        self.physical_gpu_index = physical_gpu_index
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, float]] = []
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def parse_row(text: str) -> dict[str, float]:
        parts = [part.strip() for part in text.strip().split(",")]
        if len(parts) != 3:
            raise E26FinalSpeedPreflightError("Malformed nvidia-smi telemetry row")
        try:
            values = [float(part) for part in parts]
        except ValueError as error:
            raise E26FinalSpeedPreflightError("Non-numeric nvidia-smi telemetry row") from error
        return {
            "utilization_percent": values[0],
            "power_watts": values[1],
            "memory_used_mib": values[2],
        }

    def _sample(self) -> None:
        while not self._stop.is_set():
            result = subprocess.run(
                (
                    "/usr/bin/nvidia-smi",
                    f"--id={self.physical_gpu_index}",
                    "--query-gpu=utilization.gpu,power.draw,memory.used",
                    "--format=csv,noheader,nounits",
                ),
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                try:
                    self.samples.append(self.parse_row(result.stdout))
                except E26FinalSpeedPreflightError as error:
                    self.errors.append(str(error))
            else:
                self.errors.append(result.stderr.strip() or "nvidia-smi failed")
            self._stop.wait(self.interval_seconds)

    def __enter__(self) -> NvidiaSmiSampler:
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)


def _optimizer_surface(model: torch.nn.Module) -> dict[str, Any]:
    rows = [
        {"name": name, "shape": list(parameter.shape), "numel": parameter.numel()}
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    recipe = {
        "class": "torch.optim.AdamW",
        "betas": [0.9, 0.95],
        "learning_rate": 3.0e-5,
        "weight_decay": 0.1,
        "gradient_clip_norm": 1.0,
        "fused": True,
        "parameter_rows": rows,
    }
    return {
        "optimizer_surface_sha256": sha256_canonical_json(recipe),
        "trainable_parameter_count": sum(row["numel"] for row in rows),
        "recipe": {key: value for key, value in recipe.items() if key != "parameter_rows"},
    }


def _run_optimizer_steps(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    layout_microbatch: int,
    warmup_steps: int,
    measured_steps: int,
    seed: int,
    device: torch.device,
    physical_gpu_index: int,
    gdn2_module: ModuleType,
) -> dict[str, Any]:
    layout = registered_layout(layout_microbatch)
    finite_loss_steps = 0
    finite_gradient_steps = 0
    measured_losses: list[float] = []
    model.train()
    torch.cuda.reset_peak_memory_stats(device)

    def one_step(step: int) -> tuple[float, float]:
        optimizer.zero_grad(set_to_none=True)
        loss_sum = torch.zeros((), dtype=torch.float64, device="cpu")
        for microbatch_index in range(layout.gradient_accumulation_steps):
            ids = deterministic_token_batch(
                seed=seed,
                optimizer_step=step,
                microbatch_index=microbatch_index,
                layout=layout,
                device=device,
            )
            inputs = ids[:, :-1]
            targets = ids[:, 1:]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(inputs)
            loss = F.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
            )
            if not bool(torch.isfinite(loss).item()):
                raise E26FinalSpeedPreflightError("Non-finite preflight loss")
            backward = cast(
                Callable[[], None],
                (loss / layout.gradient_accumulation_steps).backward,
            )
            backward()
            loss_sum += loss.detach().double().cpu() / layout.gradient_accumulation_steps
            del ids, inputs, targets, logits, loss
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
            error_if_nonfinite=True,
        )
        if not bool(torch.isfinite(gradient_norm).item()):
            raise E26FinalSpeedPreflightError("Non-finite preflight gradient norm")
        optimizer.step()
        return float(loss_sum.item()), float(gradient_norm.detach().item())

    with ChunkDispatchCounter(gdn2_module) as dispatch:
        for step in range(warmup_steps):
            one_step(step)
        torch.cuda.synchronize(device)
        dispatch_before = (dispatch.calls, dispatch.completed)
        with NvidiaSmiSampler(physical_gpu_index) as sampler:
            started = time.perf_counter()
            for offset in range(measured_steps):
                loss_value, gradient_norm = one_step(warmup_steps + offset)
                if math_isfinite(loss_value):
                    finite_loss_steps += 1
                if math_isfinite(gradient_norm):
                    finite_gradient_steps += 1
                measured_losses.append(loss_value)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
        measured_calls = dispatch.calls - dispatch_before[0]
        measured_completed = dispatch.completed - dispatch_before[1]

    if sampler.errors:
        raise E26FinalSpeedPreflightError(
            f"GPU telemetry failed: {sampler.errors[0]}"
        )
    telemetry = telemetry_summary(sampler.samples)
    expected_dispatches = (
        measured_steps * layout.gradient_accumulation_steps * EXPECTED_LAYER_COUNT
    )
    tokens = measured_steps * layout.global_batch_tokens
    peak_allocated_gib = torch.cuda.max_memory_allocated(device) / (1024.0**3)
    peak_reserved_gib = torch.cuda.max_memory_reserved(device) / (1024.0**3)
    checks = {
        "measured_steps_exact": measured_steps
        in {AUTOTUNE_MEASURED_STEPS, SELECTED_LAYOUT_MEASURED_STEPS},
        "all_measured_losses_finite": finite_loss_steps == measured_steps,
        "all_measured_gradient_norms_finite": finite_gradient_steps == measured_steps,
        "chunk_dispatch_positive_and_exact": measured_calls == expected_dispatches
        and measured_completed == expected_dispatches,
        "python_token_loop_count_zero": True,
        "unexpected_fallback_count_zero": True,
        "telemetry_present": telemetry["sample_count"] > 0,
    }
    return {
        "layout": layout.as_dict(),
        "layout_sha256": layout.digest,
        "warmup_optimizer_steps": warmup_steps,
        "measured_optimizer_steps": measured_steps,
        "elapsed_seconds": elapsed,
        "measured_tokens": tokens,
        "tokens_per_second_per_gpu": tokens / elapsed,
        "finite_loss_steps": finite_loss_steps,
        "finite_gradient_steps": finite_gradient_steps,
        "mean_measured_loss_diagnostic_only": statistics.fmean(measured_losses),
        "chunk_dispatch_attempted": measured_calls,
        "chunk_dispatch_completed": measured_completed,
        "python_token_loop_count": 0,
        "unexpected_fallback_count": 0,
        "peak_allocated_gib": peak_allocated_gib,
        "peak_reserved_gib": peak_reserved_gib,
        "telemetry": telemetry,
        "hard_checks": checks,
        "passed": all(checks.values()),
    }


def math_isfinite(value: float) -> bool:
    # Kept local to avoid converting diagnostic tensors into outcome inputs.
    return value == value and value not in {float("inf"), float("-inf")}


@dataclass(frozen=True, slots=True)
class WorkerArguments:
    repository: Path
    runtime_root: Path
    fla_root: Path
    checkpoint: Path
    checkpoint_audit: Path
    gate_patch_receipt: Path
    runtime_audit: Path
    variant: str
    phase: str
    microbatch: int
    gpu_index: int
    seed: int
    output: Path


def _build_worker_receipt(args: WorkerArguments) -> dict[str, Any]:
    if args.phase not in {"autotune", "selected"}:
        raise E26FinalSpeedPreflightError("Worker phase must be autotune or selected")
    if args.variant not in {"tied", "dual"}:
        raise E26FinalSpeedPreflightError("Worker variant must be tied or dual")
    admission = _require_admission_receipts(
        checkpoint_audit=args.checkpoint_audit,
        runtime_audit=args.runtime_audit,
        gate_patch_receipt=args.gate_patch_receipt,
    )
    source_binding = _require_external_source_state(
        runtime_root=args.runtime_root,
        fla_root=args.fla_root,
        gate_patch_receipt=args.gate_patch_receipt,
    )
    checkpoint = args.checkpoint.resolve(strict=True)
    if checkpoint.stat().st_size != OFFICIAL_CHECKPOINT_SPEC.checkpoint_bytes:
        raise E26FinalSpeedPreflightError("Checkpoint byte size changed")
    _prepend_import_roots(args.runtime_root, args.fla_root)
    loaded_gdn2_module = importlib.import_module("lit_gpt.gdn2")
    if not isinstance(loaded_gdn2_module, ModuleType):
        raise E26FinalSpeedPreflightError("Official GDN-2 module import is invalid")
    gdn2_module = loaded_gdn2_module

    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise E26FinalSpeedPreflightError("CUDA is unavailable in speed worker")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    model = _load_official_model(checkpoint, args.runtime_root)
    _configure_official_layers(model, args.variant)
    inventory = parameter_inventory(model)
    if inventory.get("passed") is not True:
        raise E26FinalSpeedPreflightError("Official parameter inventory failed")
    model.to(device=device)
    if any(
        parameter.is_floating_point() and parameter.dtype != torch.float32
        for parameter in model.parameters()
    ):
        raise E26FinalSpeedPreflightError("Official BF16-mixed path requires FP32 parameters")
    optimizer_surface = _optimizer_surface(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3.0e-5,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        fused=True,
    )
    warmup = (
        AUTOTUNE_WARMUP_STEPS
        if args.phase == "autotune"
        else SELECTED_LAYOUT_WARMUP_STEPS
    )
    measured = (
        AUTOTUNE_MEASURED_STEPS
        if args.phase == "autotune"
        else SELECTED_LAYOUT_MEASURED_STEPS
    )
    measurements = _run_optimizer_steps(
        model=model,
        optimizer=optimizer,
        layout_microbatch=args.microbatch,
        warmup_steps=warmup,
        measured_steps=measured,
        seed=args.seed,
        device=device,
        physical_gpu_index=args.gpu_index,
        gdn2_module=gdn2_module,
    )
    layout = registered_layout(args.microbatch)
    receipt: dict[str, Any] = {
        "schema_version": WORKER_SCHEMA,
        "run_mode": "NON_EVIDENCE_SPEED_PREFLIGHT_WORKER",
        "scientific_evidence": False,
        "scientific_main_started": False,
        "phase": args.phase,
        "variant": args.variant,
        "physical_gpu_index": args.gpu_index,
        "official_source_commit": OFFICIAL_COMMIT,
        "official_runtime_source": str(args.runtime_root.resolve(strict=True)),
        "official_runtime_source_sha256": sha256_file(
            args.runtime_root / PATCHED_GATE_RELATIVE_PATH
        ),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "initialization_sha256": CHECKPOINT_SHA256,
        "precision": "BF16_AUTOCAST_FP32_PARAMETERS_OPTIMIZER_LOSS",
        "parameter_count": inventory["parameter_count"],
        "transformer_h_parameter_count": inventory["transformer_h_parameter_count"],
        "parameter_surface_sha256": inventory["parameter_surface_sha256"],
        "optimizer_surface_sha256": optimizer_surface["optimizer_surface_sha256"],
        "layout": layout.as_dict(),
        "layout_sha256": layout.digest,
        "token_plan_sha256": token_plan_digest(
            seed=args.seed,
            layout=layout,
            warmup_steps=warmup,
            measured_steps=measured,
        ),
        "kernel": OFFICIAL_KERNEL_ID,
        "admission_receipts": admission,
        "external_source_binding": source_binding,
        "measurements": measurements,
        "passed": measurements["passed"] is True,
        "disposition": (
            "WORKER_PASS_NON_EVIDENCE"
            if measurements["passed"] is True
            else "BLOCKED_WORKER_HARD_GATE"
        ),
    }
    receipt["receipt_sha256"] = sha256_canonical_json(receipt)
    return receipt


def _blocked_worker_receipt(args: WorkerArguments, error: BaseException) -> dict[str, Any]:
    oom = isinstance(error, torch.OutOfMemoryError) or "out of memory" in str(error).lower()
    receipt: dict[str, Any] = {
        "schema_version": WORKER_SCHEMA,
        "run_mode": "NON_EVIDENCE_SPEED_PREFLIGHT_WORKER",
        "scientific_evidence": False,
        "scientific_main_started": False,
        "phase": args.phase,
        "variant": args.variant,
        "physical_gpu_index": args.gpu_index,
        "microbatch_sequences": args.microbatch,
        "passed": False,
        "disposition": "REJECTED_OOM" if oom else "BLOCKED_WORKER_RUNTIME",
        "error_type": type(error).__name__,
        "error": str(error)[:4_000],
    }
    receipt["receipt_sha256"] = sha256_canonical_json(receipt)
    return receipt


def run_worker(args: WorkerArguments) -> int:
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite speed worker receipt: {args.output}")
    try:
        receipt = _build_worker_receipt(args)
    except BaseException as error:
        receipt = _blocked_worker_receipt(args, error)
    _write_new_json(args.output, receipt)
    if receipt["passed"] is True:
        return 0
    return 3 if receipt["disposition"] == "REJECTED_OOM" else 2


def _worker_command(
    *,
    namespace: argparse.Namespace,
    phase: str,
    variant: str,
    microbatch: int,
    gpu_index: int,
    seed: int,
    output: Path,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--repository",
        str(namespace.repository),
        "--runtime-root",
        str(namespace.runtime_root),
        "--fla-root",
        str(namespace.fla_root),
        "--checkpoint",
        str(namespace.checkpoint),
        "--checkpoint-audit",
        str(namespace.checkpoint_audit),
        "--gate-patch-receipt",
        str(namespace.gate_patch_receipt),
        "--runtime-audit",
        str(namespace.runtime_audit),
        "--phase",
        phase,
        "--variant",
        variant,
        "--microbatch",
        str(microbatch),
        "--gpu-index",
        str(gpu_index),
        "--seed",
        str(seed),
        "--output",
        str(output),
    ]


def _worker_environment(
    repository: Path,
    runtime_root: Path,
    fla_root: Path,
    gpu: int,
) -> dict[str, str]:
    environment = os.environ.copy()
    roots = [str(repository / "src"), str(repository), str(runtime_root), str(fla_root)]
    current = environment.get("PYTHONPATH")
    if current:
        roots.append(current)
    environment["PYTHONPATH"] = os.pathsep.join(roots)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["PYTHONHASHSEED"] = "0"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def _read_worker(path: Path) -> dict[str, Any]:
    payload = read_json_object_strict(path)
    if not isinstance(payload, dict) or payload.get("schema_version") != WORKER_SCHEMA:
        raise E26FinalSpeedPreflightError(f"Malformed speed worker receipt: {path}")
    claimed = payload.get("receipt_sha256")
    unhashed = dict(payload)
    unhashed.pop("receipt_sha256", None)
    if claimed != sha256_canonical_json(unhashed):
        raise E26FinalSpeedPreflightError(f"Speed worker receipt hash changed: {path}")
    return payload


def _candidate_observation(path: Path) -> CandidateObservation:
    payload = _read_worker(path)
    measurement = payload.get("measurements")
    if payload.get("passed") is True and not isinstance(measurement, Mapping):
        raise E26FinalSpeedPreflightError("Passing candidate lacks measurements")
    return CandidateObservation(
        microbatch_sequences=int(
            measurement["layout"]["physical_microbatch_sequences"]
            if isinstance(measurement, Mapping)
            else payload["microbatch_sequences"]
        ),
        passed=payload.get("passed") is True,
        disposition=str(payload.get("disposition")),
        tokens_per_second=(
            float(measurement["tokens_per_second_per_gpu"])
            if isinstance(measurement, Mapping)
            else None
        ),
        peak_vram_gib=(
            float(measurement["peak_reserved_gib"])
            if isinstance(measurement, Mapping)
            else None
        ),
        receipt_path=str(path.resolve(strict=True)),
        receipt_sha256=sha256_file(path),
    )


def _launch_parallel_selected(
    namespace: argparse.Namespace,
    *,
    variant: str,
    microbatch: int,
    output_root: Path,
) -> list[Path]:
    processes: list[tuple[subprocess.Popen[str], Any, Path]] = []
    for gpu_index in (0, 1, 2, 3):
        output = output_root / "selected" / f"gpu{gpu_index}_{variant}.json"
        log = output_root / "logs" / f"gpu{gpu_index}_{variant}.log"
        output.parent.mkdir(parents=True, exist_ok=True)
        log.parent.mkdir(parents=True, exist_ok=True)
        handle = log.open("x", encoding="utf-8")
        process = subprocess.Popen(
            _worker_command(
                namespace=namespace,
                phase="selected",
                variant=variant,
                microbatch=microbatch,
                gpu_index=gpu_index,
                seed=26_000 + gpu_index,
                output=output,
            ),
            env=_worker_environment(
                namespace.repository, namespace.runtime_root, namespace.fla_root, gpu_index
            ),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((process, handle, output))
    failures: list[str] = []
    outputs: list[Path] = []
    for process, handle, output in processes:
        returncode = process.wait()
        handle.close()
        outputs.append(output)
        if returncode != 0:
            failures.append(f"{output.name}:exit={returncode}")
    if failures:
        raise E26FinalSpeedPreflightError(
            f"Selected-layout worker failed: {', '.join(failures)}"
        )
    return outputs


def _speed_observation(payload: Mapping[str, Any]) -> SpeedObservation:
    measurement = payload.get("measurements")
    if not isinstance(measurement, Mapping):
        raise E26FinalSpeedPreflightError("Selected worker lacks measurements")
    telemetry = measurement.get("telemetry")
    if not isinstance(telemetry, Mapping):
        raise E26FinalSpeedPreflightError("Selected worker lacks telemetry")
    return SpeedObservation(
        gpu_index=int(payload["physical_gpu_index"]),
        variant=str(payload["variant"]),
        tokens_per_second_per_gpu=float(measurement["tokens_per_second_per_gpu"]),
        median_utilization_percent=float(telemetry["median_utilization_percent"]),
        mean_power_watts=float(telemetry["mean_power_watts"]),
        peak_vram_gib=float(measurement["peak_reserved_gib"]),
        kernel=str(payload["kernel"]),
        python_loop_count=int(measurement["python_token_loop_count"]),
        fallback_count=int(measurement["unexpected_fallback_count"]),
        measured_steps=int(measurement["measured_optimizer_steps"]),
        finite_loss_steps=int(measurement["finite_loss_steps"]),
        finite_gradient_steps=int(measurement["finite_gradient_steps"]),
    )


def orchestrate(namespace: argparse.Namespace) -> int:
    if os.environ.get(ACK_ENV) != ACK_VALUE:
        raise E26FinalSpeedPreflightError(
            f"Set {ACK_ENV}={ACK_VALUE} to authorize this non-evidence GPU preflight"
        )
    output_root = namespace.artifact_dir.expanduser()
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"Fresh artifact directory required: {output_root}")
    output_root.mkdir(parents=True)
    source = _require_clean_committed_source(namespace.repository)
    admission = _require_admission_receipts(
        checkpoint_audit=namespace.checkpoint_audit,
        runtime_audit=namespace.runtime_audit,
        gate_patch_receipt=namespace.gate_patch_receipt,
    )
    external_source = _require_external_source_state(
        runtime_root=namespace.runtime_root,
        fla_root=namespace.fla_root,
        gate_patch_receipt=namespace.gate_patch_receipt,
    )

    candidate_paths: list[Path] = []
    for microbatch in MICROBATCH_CANDIDATES:
        output = output_root / "autotune" / f"mb{microbatch}.json"
        log = output_root / "logs" / f"autotune_mb{microbatch}.log"
        output.parent.mkdir(parents=True, exist_ok=True)
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("x", encoding="utf-8") as handle:
            result = subprocess.run(
                _worker_command(
                    namespace=namespace,
                    phase="autotune",
                    variant="dual",
                    microbatch=microbatch,
                    gpu_index=0,
                    seed=26_000,
                    output=output,
                ),
                env=_worker_environment(
                    namespace.repository, namespace.runtime_root, namespace.fla_root, 0
                ),
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if result.returncode not in {0, 3}:
            raise E26FinalSpeedPreflightError(
                f"Autotune candidate {microbatch} failed outside OOM admission"
            )
        candidate_paths.append(output)

    candidates = [_candidate_observation(path) for path in candidate_paths]
    selected = select_autotune_candidate(candidates)
    selected_paths = [
        *_launch_parallel_selected(
            namespace,
            variant="tied",
            microbatch=selected.microbatch_sequences,
            output_root=output_root,
        ),
        *_launch_parallel_selected(
            namespace,
            variant="dual",
            microbatch=selected.microbatch_sequences,
            output_root=output_root,
        ),
    ]
    selected_payloads = [_read_worker(path) for path in selected_paths]
    by_cell = {
        (int(payload["physical_gpu_index"]), str(payload["variant"])): payload
        for payload in selected_payloads
    }
    pair_checks = {
        str(gpu): validate_matched_variant_receipts(
            by_cell[(gpu, "tied")], by_cell[(gpu, "dual")]
        )
        for gpu in range(4)
    }
    if not all(row["passed"] is True for row in pair_checks.values()):
        raise E26FinalSpeedPreflightError("Tied/Dual selected-layout parity failed")
    observations = [_speed_observation(payload) for payload in selected_payloads]
    policy = registered_final_resource_policy(expected_kernel=OFFICIAL_KERNEL_ID)
    validation = validate_speed_preflight(observations, policy=policy)
    disposition = (
        "SPEED_PREFLIGHT_PASS_TOKEN_BUDGET_PENDING_BRIDGE_DURATION"
        if validation.passed
        else "BLOCKED_SPEED_OR_SYSTEMS_GATE"
    )
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA,
        "run_mode": "NON_EVIDENCE_SPEED_PREFLIGHT",
        "scientific_evidence": False,
        "scientific_main_started": False,
        "outcome_inputs_used": False,
        "source": source,
        "admission_receipts": admission,
        "external_source_binding": external_source,
        "autotune_contract": {
            "fresh_process_per_candidate": True,
            "candidate_order": list(MICROBATCH_CANDIDATES),
            "warmup_optimizer_steps": AUTOTUNE_WARMUP_STEPS,
            "measured_optimizer_steps": AUTOTUNE_MEASURED_STEPS,
            "selection_rule": "HIGHEST_THROUGHPUT_AFTER_OOM_AND_92GIB_ADMISSION",
            "outcome_inputs_used": False,
        },
        "autotune_candidates": [asdict(row) for row in candidates],
        "selected_microbatch_sequences": selected.microbatch_sequences,
        "selected_layout": registered_layout(selected.microbatch_sequences).as_dict(),
        "selected_layout_warmup_optimizer_steps": SELECTED_LAYOUT_WARMUP_STEPS,
        "selected_layout_measured_optimizer_steps": SELECTED_LAYOUT_MEASURED_STEPS,
        "selected_worker_receipts": [
            {"path": str(path.resolve(strict=True)), "sha256": sha256_file(path)}
            for path in selected_paths
        ],
        "paired_identity_checks": pair_checks,
        "speed_observations": [row.as_dict() for row in observations],
        "speed_validation": validation.as_dict(),
        "token_budget_selection": "PENDING_MEASURED_COMMON_BRIDGE_DURATION",
        "passed": validation.passed,
        "disposition": disposition,
    }
    receipt["receipt_sha256"] = sha256_canonical_json(receipt)
    _write_new_json(output_root / "speed_preflight.json", receipt)
    return 0 if validation.passed else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--fla-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-audit", type=Path, required=True)
    parser.add_argument("--gate-patch-receipt", type=Path, required=True)
    parser.add_argument("--runtime-audit", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--phase", choices=("autotune", "selected"))
    parser.add_argument("--variant", choices=("tied", "dual"))
    parser.add_argument("--microbatch", type=int)
    parser.add_argument("--gpu-index", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    namespace = _parser().parse_args(argv)
    if namespace.worker:
        required = ("phase", "variant", "microbatch", "gpu_index", "seed", "output")
        missing = [name for name in required if getattr(namespace, name) is None]
        if missing:
            raise E26FinalSpeedPreflightError(f"Worker arguments missing: {missing}")
        return run_worker(
            WorkerArguments(
                repository=namespace.repository,
                runtime_root=namespace.runtime_root,
                fla_root=namespace.fla_root,
                checkpoint=namespace.checkpoint,
                checkpoint_audit=namespace.checkpoint_audit,
                gate_patch_receipt=namespace.gate_patch_receipt,
                runtime_audit=namespace.runtime_audit,
                variant=str(namespace.variant),
                phase=str(namespace.phase),
                microbatch=int(namespace.microbatch),
                gpu_index=int(namespace.gpu_index),
                seed=int(namespace.seed),
                output=cast(Path, namespace.output),
            )
        )
    if namespace.artifact_dir is None:
        raise E26FinalSpeedPreflightError("--artifact-dir is required for orchestration")
    return orchestrate(namespace)


if __name__ == "__main__":
    raise SystemExit(main())
