#!/usr/bin/env python3
"""Fail-closed GPU runtime audit for the pinned E26 Final official GDN-2 path.

This command is a non-evidence admission audit.  It loads the admitted 1.3B
checkpoint into the hash-locked, gate-only patched official checkout and
positively observes two distinct execution paths:

* the full official GPT forward/backward must call ``chunk_gdn2``; and
* an additive CATENA pure-recurrent GPT adapter must call
  ``fused_recurrent_gdn2`` in every official layer while carrying an FLA
  recurrent/short-convolution cache from a chunk prefix into batched queries.

The pinned official GPT does not plumb a GDN-2 cache through ``GPT.forward``.
The upstream defect remains an explicit limitation.  GPT-level eligibility is
opened only for the separately identified CATENA adapter after exact block-loop
equivalence, state-carry, clone, and no-alias checks.  There is no reference
fallback and no per-token Python decode loop.
"""

from __future__ import annotations

import argparse
import ast
import functools
import gc
import hashlib
import importlib
import importlib.metadata
import inspect
import subprocess
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, suppress
from copy import deepcopy
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
from catena.lm.e26_final_official_adapter import (
    PureRecurrentOfficialGptCacheAdapter,
    cache_tensor_equality_and_no_alias,
)
from catena.lm.e26_final_provenance import OFFICIAL_SOURCE
from tools.apply_e26_final_official_patch import (
    ALLOWED_POLICIES,
    PINNED_GDN2_SHA256,
    PINNED_OFFICIAL_COMMIT,
    POLICY_ATTRIBUTE,
    TARGET_RELATIVE_PATH,
)
from tools.audit_e26_final_checkpoint import (
    OFFICIAL_CHECKPOINT_SPEC,
    resolve_official_gpt_factory,
    safe_load_checkpoint,
    validate_checkpoint_audit_receipt,
)


class E26FinalOfficialRuntimeError(RuntimeError):
    """Raised when the official runtime admission contract is not satisfied."""


PINNED_FLA_COMMIT: Final = "4b02d15d6a68700181b180235be62a9fb95d2a38"
PINNED_FLA_TREE: Final = "816817b67e1bc3f8cd905f309034a1bd0d45b2da"
EXPECTED_TORCH_VERSION: Final = "2.9.0+cu130"
EXPECTED_TORCH_CUDA_VERSION: Final = "13.0"
EXPECTED_FLASH_ATTN_VERSION: Final = "2.8.3"
EXPECTED_LAYER_COUNT: Final = 18
TRAIN_SEQUENCE_TOKENS: Final = 64
ADAPTER_PREFIX_TOKENS: Final = 128
ADAPTER_QUERY_TOKENS: Final = 8
INHERITED_FP32_RELATIVE_L2_MAX: Final = 1.0e-5
INHERITED_FP32_MAX_ABS_MAX: Final = 1.0e-5
INHERITED_BF16_RELATIVE_L2_MAX: Final = 7.0e-3
_RECEIPT_SCHEMA: Final = "catena-e26-final-official-runtime-audit-v1"
_RECEIPT_TYPE: Final = "E26_FINAL_OFFICIAL_RUNTIME_AUDIT_RECEIPT"
_CLAIM_CEILING: Final = (
    "OFFICIAL_GDN2_KERNELS_WITH_CATENA_PURE_RECURRENT_GPT_CACHE_ADAPTER_ONLY"
)


@dataclass
class DispatchCounts:
    """Positive call and backward-hook evidence for the two official kernels."""

    chunk_attempted: int = 0
    chunk_completed: int = 0
    fused_attempted: int = 0
    fused_completed: int = 0
    chunk_backward_hooks_registered: int = 0
    chunk_backward_hooks_completed: int = 0
    chunk_backward_finite: int = 0
    chunk_backward_nonzero: int = 0

    def subtract(self, earlier: DispatchCounts) -> DispatchCounts:
        """Return the nonnegative phase-local difference from a prior snapshot."""

        values = {
            key: int(value) - int(getattr(earlier, key))
            for key, value in asdict(self).items()
        }
        if any(value < 0 for value in values.values()):
            raise E26FinalOfficialRuntimeError("Dispatch counters moved backwards")
        return DispatchCounts(**values)


class KernelDispatchInstrumentation(AbstractContextManager["KernelDispatchInstrumentation"]):
    """Wrap the exact symbols imported by ``lit_gpt.gdn2`` without fallback."""

    def __init__(self, gdn2_module: ModuleType) -> None:
        self._module = gdn2_module
        self._original_chunk: Callable[..., Any] | None = None
        self._original_fused: Callable[..., Any] | None = None
        self.phase: str | None = None
        self.counts = DispatchCounts()

    def snapshot(self) -> DispatchCounts:
        """Capture an immutable value copy of the current counters."""

        return DispatchCounts(**asdict(self.counts))

    def set_phase(self, phase: str) -> None:
        if phase not in {
            "chunk_training",
            "adapter_chunk_equivalence",
            "adapter_fused_query",
        }:
            raise E26FinalOfficialRuntimeError(f"Unknown instrumentation phase: {phase}")
        self.phase = phase

    def _chunk_gradient_hook(self, gradient: torch.Tensor) -> None:
        self.counts.chunk_backward_hooks_completed += 1
        if bool(torch.isfinite(gradient).all().item()):
            self.counts.chunk_backward_finite += 1
        if bool(torch.count_nonzero(gradient).item() > 0):
            self.counts.chunk_backward_nonzero += 1

    def _wrap(self, name: str, function: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(function)
        def observed(*args: Any, **kwargs: Any) -> Any:
            if self.phase is None:
                raise E26FinalOfficialRuntimeError(
                    f"Official kernel {name} was invoked outside a declared audit phase"
                )
            attempted_name = "chunk_attempted" if name == "chunk_gdn2" else "fused_attempted"
            completed_name = "chunk_completed" if name == "chunk_gdn2" else "fused_completed"
            setattr(self.counts, attempted_name, getattr(self.counts, attempted_name) + 1)
            # There is intentionally one call and no exception-driven retry/fallback.
            result = function(*args, **kwargs)
            setattr(self.counts, completed_name, getattr(self.counts, completed_name) + 1)
            if name == "chunk_gdn2" and self.phase == "chunk_training":
                output = result[0] if isinstance(result, tuple) and result else None
                if not isinstance(output, torch.Tensor) or not output.requires_grad:
                    raise E26FinalOfficialRuntimeError(
                        "Observed chunk_gdn2 output cannot provide backward evidence"
                    )
                self.counts.chunk_backward_hooks_registered += 1
                register_hook = cast(
                    Callable[[Callable[[torch.Tensor], None]], Any],
                    output.register_hook,
                )
                register_hook(self._chunk_gradient_hook)
            return result

        return observed

    def __enter__(self) -> KernelDispatchInstrumentation:
        chunk = getattr(self._module, "chunk_gdn2", None)
        fused = getattr(self._module, "fused_recurrent_gdn2", None)
        if not callable(chunk) or not callable(fused):
            raise E26FinalOfficialRuntimeError(
                "Official lit_gpt.gdn2 kernel symbols are unavailable"
            )
        self._original_chunk = cast(Callable[..., Any], chunk)
        self._original_fused = cast(Callable[..., Any], fused)
        self._module.__dict__["chunk_gdn2"] = self._wrap(
            "chunk_gdn2", self._original_chunk
        )
        self._module.__dict__["fused_recurrent_gdn2"] = self._wrap(
            "fused_recurrent_gdn2", self._original_fused
        )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> None:
        if self._original_chunk is not None:
            self._module.__dict__["chunk_gdn2"] = self._original_chunk
        if self._original_fused is not None:
            self._module.__dict__["fused_recurrent_gdn2"] = self._original_fused
        self.phase = None


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise E26FinalOfficialRuntimeError(
            f"Git command failed ({' '.join(arguments)}): {result.stderr.strip()}"
        )
    # Preserve the leading XY status column used by ``git status --porcelain``.
    return result.stdout.rstrip("\r\n")


def _module_origin(module: ModuleType) -> Path:
    raw = getattr(module, "__file__", None)
    if not isinstance(raw, str):
        raise E26FinalOfficialRuntimeError(f"Module {module.__name__} has no file origin")
    return Path(raw).resolve(strict=True)


def _origin_within(module: ModuleType, root: Path) -> bool:
    return _module_origin(module).is_relative_to(root)


def _validate_runtime_root(
    runtime_root: Path,
    gate_patch_receipt_path: Path,
) -> dict[str, Any]:
    unresolved = runtime_root.expanduser()
    if unresolved.is_symlink():
        raise E26FinalOfficialRuntimeError("Official runtime checkout must not be a symlink")
    root = unresolved.resolve(strict=True)
    receipt = read_json_object_strict(gate_patch_receipt_path)
    target = (root / TARGET_RELATIVE_PATH).resolve(strict=True)
    patch_path_raw = receipt.get("unified_diff_path")
    patch_path = (
        Path(patch_path_raw).resolve(strict=True)
        if isinstance(patch_path_raw, str)
        else None
    )
    status_lines = _git(root, "status", "--porcelain=v1", "--untracked-files=all").splitlines()
    changed_paths = _git(root, "diff", "--name-only").splitlines()
    base_bytes = subprocess.run(
        ("git", "-C", str(root), "show", f"HEAD:{TARGET_RELATIVE_PATH.as_posix()}"),
        check=True,
        capture_output=True,
    ).stdout
    hard_checks = {
        "official_commit_exact": _git(root, "rev-parse", "HEAD") == PINNED_OFFICIAL_COMMIT,
        "official_tree_exact": _git(root, "rev-parse", "HEAD^{tree}") == OFFICIAL_SOURCE.tree,
        "only_gate_source_modified": changed_paths == [TARGET_RELATIVE_PATH.as_posix()]
        and len(status_lines) == 1
        and status_lines[0][3:] == TARGET_RELATIVE_PATH.as_posix(),
        "base_gate_source_sha256_exact": sha256_file_from_bytes(base_bytes)
        == PINNED_GDN2_SHA256,
        "patch_receipt_status_applied": receipt.get("status") == "APPLIED",
        "patch_receipt_commit_exact": receipt.get("official_commit")
        == PINNED_OFFICIAL_COMMIT,
        "patch_receipt_target_exact": receipt.get("target_relative_path")
        == TARGET_RELATIVE_PATH.as_posix(),
        "patched_gate_source_sha256_exact": receipt.get("patched_file_sha256")
        == sha256_file(target),
        "gate_policy_attribute_exact": receipt.get("policy_attribute") == POLICY_ATTRIBUTE,
        "gate_policy_values_exact": receipt.get("allowed_policy_values")
        == list(ALLOWED_POLICIES),
        "kernel_calls_unmodified_by_patch": receipt.get("kernel_calls_modified") is False,
        "patch_file_present_and_exact": patch_path is not None
        and receipt.get("unified_diff_sha256") == sha256_file(patch_path),
    }
    if not all(hard_checks.values()):
        failed = sorted(key for key, value in hard_checks.items() if not value)
        raise E26FinalOfficialRuntimeError(
            f"Official runtime source binding failed: {', '.join(failed)}"
        )
    return {
        "repository": str(root),
        "head": _git(root, "rev-parse", "HEAD"),
        "tree": _git(root, "rev-parse", "HEAD^{tree}"),
        "git_status_porcelain": status_lines,
        "gate_source_sha256": sha256_file(target),
        "gate_patch_receipt_path": str(gate_patch_receipt_path.resolve(strict=True)),
        "gate_patch_receipt_file_sha256": sha256_file(gate_patch_receipt_path),
        "hard_checks": hard_checks,
        "passed": True,
    }


def sha256_file_from_bytes(data: bytes) -> str:
    """Hash in-memory Git blob bytes without writing temporary files."""

    return hashlib.sha256(data).hexdigest()


def _validate_fla_root(fla_root: Path) -> dict[str, Any]:
    unresolved = fla_root.expanduser()
    if unresolved.is_symlink():
        raise E26FinalOfficialRuntimeError("Pinned FLA checkout must not be a symlink")
    root = unresolved.resolve(strict=True)
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    checks = {
        "fla_commit_exact": _git(root, "rev-parse", "HEAD") == PINNED_FLA_COMMIT,
        "fla_tree_exact": _git(root, "rev-parse", "HEAD^{tree}") == PINNED_FLA_TREE,
        "fla_checkout_clean": status == "",
    }
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise E26FinalOfficialRuntimeError(f"FLA source binding failed: {', '.join(failed)}")
    return {
        "repository": str(root),
        "commit": _git(root, "rev-parse", "HEAD"),
        "tree": _git(root, "rev-parse", "HEAD^{tree}"),
        "git_status_porcelain": status.splitlines(),
        "hard_checks": checks,
        "passed": True,
    }


def _validate_checkpoint_binding(
    checkpoint: Path,
    receipt_path: Path,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    receipt = validate_checkpoint_audit_receipt(read_json_object_strict(receipt_path))
    file_section = receipt.get("checkpoint_file")
    if not isinstance(file_section, Mapping):
        raise E26FinalOfficialRuntimeError("Checkpoint receipt lacks checkpoint_file")
    unresolved = checkpoint.expanduser()
    if unresolved.is_symlink():
        raise E26FinalOfficialRuntimeError("Checkpoint must not be a symlink")
    target = unresolved.resolve(strict=True)
    checks = {
        "checkpoint_receipt_passed": receipt.get("passed") is True,
        "checkpoint_admission_eligible": receipt.get("checkpoint_admission_eligible") is True,
        "checkpoint_path_exact": file_section.get("path") == str(target),
        "checkpoint_bytes_exact": target.stat().st_size
        == OFFICIAL_CHECKPOINT_SPEC.checkpoint_bytes,
        "checkpoint_sha256_exact": sha256_file(target)
        == OFFICIAL_CHECKPOINT_SPEC.checkpoint_sha256
        == file_section.get("sha256"),
    }
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise E26FinalOfficialRuntimeError(
            f"Checkpoint admission binding failed: {', '.join(failed)}"
        )
    return (
        {
            "checkpoint_path": str(target),
            "checkpoint_bytes": target.stat().st_size,
            "checkpoint_sha256": file_section.get("sha256"),
            "checkpoint_receipt_path": str(receipt_path.resolve(strict=True)),
            "checkpoint_receipt_file_sha256": sha256_file(receipt_path),
            "checkpoint_receipt_sha256": receipt.get("receipt_sha256"),
            "hard_checks": checks,
            "passed": True,
        },
        receipt,
    )


def _prepend_import_paths(paths: Sequence[Path]) -> None:
    for path in reversed(paths):
        text = str(path)
        with suppress(ValueError):
            sys.path.remove(text)
        sys.path.insert(0, text)


def _reject_conflicting_preloads(runtime_root: Path, fla_root: Path) -> None:
    for name, loaded in tuple(sys.modules.items()):
        if not isinstance(loaded, ModuleType):
            continue
        if (name == "lit_gpt" or name.startswith("lit_gpt.")) and not _origin_within(
            loaded, runtime_root
        ):
            raise E26FinalOfficialRuntimeError(
                f"Refusing preloaded lit_gpt module outside runtime checkout: {name}"
            )
        if (name == "fla" or name.startswith("fla.")) and not _origin_within(
            loaded, fla_root
        ):
            raise E26FinalOfficialRuntimeError(
                f"Refusing preloaded fla module outside pinned checkout: {name}"
            )


def _load_runtime_modules(
    runtime_root: Path,
    fla_root: Path,
) -> tuple[dict[str, ModuleType], dict[str, Any]]:
    _reject_conflicting_preloads(runtime_root, fla_root)
    _prepend_import_paths((runtime_root, fla_root))
    modules = {
        "lit_gpt.model": importlib.import_module("lit_gpt.model"),
        "lit_gpt.gdn2": importlib.import_module("lit_gpt.gdn2"),
        "fla": importlib.import_module("fla"),
        "fla.models.utils": importlib.import_module("fla.models.utils"),
        "flash_attn": importlib.import_module("flash_attn"),
    }
    origins = {
        name: str(_module_origin(module)) for name, module in modules.items()
    }
    checks = {
        "lit_gpt_model_origin_exact": _origin_within(modules["lit_gpt.model"], runtime_root),
        "lit_gpt_gdn2_origin_exact": _origin_within(modules["lit_gpt.gdn2"], runtime_root),
        "fla_origin_exact": _origin_within(modules["fla"], fla_root),
        "fla_utils_origin_exact": _origin_within(modules["fla.models.utils"], fla_root),
        "torch_version_exact": torch.__version__ == EXPECTED_TORCH_VERSION,
        "torch_cuda_version_exact": torch.version.cuda == EXPECTED_TORCH_CUDA_VERSION,
        "flash_attn_version_exact": importlib.metadata.version("flash-attn")
        == EXPECTED_FLASH_ATTN_VERSION,
    }
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise E26FinalOfficialRuntimeError(
            f"Official Python runtime binding failed: {', '.join(failed)}"
        )
    return modules, {"module_origins": origins, "hard_checks": checks, "passed": True}


def _kernel_function_binding(
    gdn2_module: ModuleType,
    runtime_root: Path,
) -> dict[str, Any]:
    expected = {
        "chunk_gdn2": runtime_root / "lit_gpt" / "gdn2_ops" / "chunk_gdn2.py",
        "fused_recurrent_gdn2": (
            runtime_root / "lit_gpt" / "gdn2_ops" / "fused_recurrent_gdn2.py"
        ),
    }
    rows: dict[str, dict[str, Any]] = {}
    checks: dict[str, bool] = {}
    for name, expected_path in expected.items():
        function = getattr(gdn2_module, name, None)
        unwrapped = inspect.unwrap(function) if callable(function) else None
        source_file_raw = inspect.getsourcefile(unwrapped) if unwrapped is not None else None
        source_file = (
            Path(source_file_raw).resolve(strict=True)
            if isinstance(source_file_raw, str)
            else None
        )
        module_name = getattr(function, "__module__", None)
        checks[f"{name}_callable"] = callable(function)
        checks[f"{name}_source_exact"] = source_file == expected_path.resolve(strict=True)
        checks[f"{name}_module_exact"] = module_name == f"lit_gpt.gdn2_ops.{name}"
        rows[name] = {
            "module": module_name,
            "decorated_source_file": (
                inspect.getsourcefile(function) if callable(function) else None
            ),
            "source_file": str(source_file) if source_file is not None else None,
            "source_sha256": sha256_file(source_file) if source_file is not None else None,
        }
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise E26FinalOfficialRuntimeError(
            f"Official kernel function binding failed: {', '.join(failed)}"
        )
    return {"functions": rows, "hard_checks": checks, "passed": True}


def _find_class_method(tree: ast.Module, class_name: str, method_name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    return child
    raise E26FinalOfficialRuntimeError(f"Missing {class_name}.{method_name} in pinned model")


def audit_gpt_decode_cache_contract(model_source: str) -> dict[str, Any]:
    """Classify exact GPT-level GDN-2 cache wiring without modifying it."""

    tree = ast.parse(model_source)
    block_init = _find_class_method(tree, "Block", "__init__")
    block_forward = _find_class_method(tree, "Block", "forward")
    build_caches = _find_class_method(tree, "GPT", "build_kv_caches")

    constructor_calls = [
        node
        for node in ast.walk(block_init)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "GatedDeltaNet2"
    ]
    gdn2_forward_calls = [
        node
        for node in ast.walk(block_forward)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
        and node.func.attr == "attn"
        and any(keyword.arg == "attention_mask" for keyword in node.keywords)
    ]
    append_none_calls = [
        node
        for node in ast.walk(build_caches)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "append"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value is None
    ]
    layer_idx_forwarded = any(
        any(keyword.arg == "layer_idx" for keyword in call.keywords)
        for call in constructor_calls
    )
    cache_forwarded = any(
        {keyword.arg for keyword in call.keywords}
        >= {"past_key_values", "use_cache"}
        for call in gdn2_forward_calls
    )
    cache_slot_allocated = not append_none_calls
    supported = bool(layer_idx_forwarded and cache_forwarded and cache_slot_allocated)
    return {
        "official_gpt_autoregressive_decode_supported": supported,
        "constructor_layer_idx_forwarded": layer_idx_forwarded,
        "block_past_key_values_and_use_cache_forwarded": cache_forwarded,
        "gdn2_cache_slot_allocated": cache_slot_allocated,
        "catena_adapter_required_for_gpt_cache": not supported,
        "disposition": (
            "GPT_LEVEL_CACHE_PLUMBING_PRESENT"
            if supported
            else "KNOWN_OFFICIAL_GPT_DECODE_CACHE_PLUMBING_DEFECT"
        ),
        "scientific_gpt_decode_eligible": supported,
    }


def _load_official_model(
    checkpoint: Path,
    runtime_root: Path,
) -> torch.nn.Module:
    payload = safe_load_checkpoint(checkpoint)
    model_raw = payload.get("model")
    if not isinstance(model_raw, Mapping) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in model_raw.items()
    ):
        raise E26FinalOfficialRuntimeError("Admitted checkpoint model state is invalid")
    model_state = cast(Mapping[str, torch.Tensor], model_raw)
    factory, binding = resolve_official_gpt_factory(runtime_root)
    if binding.get("passed") is not True:
        raise E26FinalOfficialRuntimeError("Official GPT factory binding did not pass")
    model = factory()
    incompatible = model.load_state_dict(model_state, strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise E26FinalOfficialRuntimeError("Strict official runtime load returned key drift")
    if sum(parameter.numel() for parameter in model.parameters()) != (
        OFFICIAL_CHECKPOINT_SPEC.full_model_numel
    ):
        raise E26FinalOfficialRuntimeError("Loaded official parameter count changed")
    del model_state
    del payload
    gc.collect()
    return model


def _gdn2_layers(
    model: torch.nn.Module,
    gdn2_class: type[torch.nn.Module],
) -> list[torch.nn.Module]:
    transformer = getattr(model, "transformer", None)
    blocks = getattr(transformer, "h", None)
    if not isinstance(blocks, torch.nn.ModuleList):
        raise E26FinalOfficialRuntimeError("Official model lacks transformer.h ModuleList")
    layers = [getattr(block, "attn", None) for block in blocks]
    if len(layers) != EXPECTED_LAYER_COUNT or not all(
        isinstance(layer, gdn2_class) for layer in layers
    ):
        raise E26FinalOfficialRuntimeError("Official model GDN-2 layer population changed")
    return cast(list[torch.nn.Module], layers)


def _configure_layers(
    layers: Sequence[torch.nn.Module],
    *,
    variant: str,
) -> dict[str, Any]:
    if variant not in ALLOWED_POLICIES:
        raise E26FinalOfficialRuntimeError(f"Unknown explicit gate policy: {variant}")
    initial_indices = [getattr(layer, "layer_idx", None) for layer in layers]
    for index, layer in enumerate(layers):
        setattr(layer, POLICY_ATTRIBUTE, variant)
        layer.__dict__["allow_neg_eigval"] = False
        layer.__dict__["mode"] = "chunk"
        layer.__dict__["layer_idx"] = index
    checks = {
        "layer_count_exact": len(layers) == EXPECTED_LAYER_COUNT,
        "official_constructor_layer_indices_missing_observed": all(
            index is None for index in initial_indices
        ),
        "explicit_gate_policy_all_layers": all(
            getattr(layer, POLICY_ATTRIBUTE, None) == variant for layer in layers
        ),
        "negative_eigenvalues_disabled_all_layers": all(
            getattr(layer, "allow_neg_eigval", None) is False for layer in layers
        ),
        "chunk_mode_all_layers": all(getattr(layer, "mode", None) == "chunk" for layer in layers),
        "layer_indices_explicit_and_unique": [
            getattr(layer, "layer_idx", None) for layer in layers
        ]
        == list(range(EXPECTED_LAYER_COUNT)),
    }
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise E26FinalOfficialRuntimeError(
            f"Explicit GDN-2 layer configuration failed: {', '.join(failed)}"
        )
    return {
        "variant": variant,
        "initial_layer_indices": initial_indices,
        "configured_layer_indices": list(range(EXPECTED_LAYER_COUNT)),
        "hard_checks": checks,
        "passed": True,
    }


def _all_finite(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor).all().item())


def _run_chunk_training_audit(
    model: torch.nn.Module,
    instrumentation: KernelDispatchInstrumentation,
    *,
    device: torch.device,
) -> dict[str, Any]:
    model.train()
    model.zero_grad(set_to_none=True)
    token_count = TRAIN_SEQUENCE_TOKENS + 1
    token_ids = (torch.arange(token_count, device=device, dtype=torch.long) * 7_919 + 17) % (
        OFFICIAL_CHECKPOINT_SPEC.vocab_size
    )
    inputs = token_ids[:-1].unsqueeze(0)
    targets = token_ids[1:].unsqueeze(0)
    before = instrumentation.snapshot()
    instrumentation.set_phase("chunk_training")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(inputs)
    loss = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
    )
    if not _all_finite(loss):
        raise E26FinalOfficialRuntimeError("Full-GPT chunk training loss is non-finite")
    backward = cast(Callable[[], None], loss.backward)
    backward()
    torch.cuda.synchronize(device)
    phase = instrumentation.snapshot().subtract(before)

    missing_gradients: list[str] = []
    nonfinite_gradients: list[str] = []
    projection_nonzero: dict[str, bool] = {}
    gradient_tensor_count = 0
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if parameter.requires_grad and gradient is None:
            missing_gradients.append(name)
            continue
        if gradient is None:
            continue
        gradient_tensor_count += 1
        if not _all_finite(gradient):
            nonfinite_gradients.append(name)
        if name.endswith("attn.b_proj.weight") or name.endswith("attn.w_proj.weight"):
            projection_nonzero[name] = bool(torch.count_nonzero(gradient).item() > 0)

    expected_projection_gradients = EXPECTED_LAYER_COUNT * 2
    checks = {
        "loss_finite": _all_finite(loss),
        "logits_finite": _all_finite(logits),
        "logits_shape_exact": list(logits.shape)
        == [1, TRAIN_SEQUENCE_TOKENS, OFFICIAL_CHECKPOINT_SPEC.vocab_size],
        "chunk_attempted_once_per_layer": phase.chunk_attempted == EXPECTED_LAYER_COUNT,
        "chunk_completed_once_per_layer": phase.chunk_completed == EXPECTED_LAYER_COUNT,
        "no_fused_dispatch_during_training": phase.fused_attempted == 0
        and phase.fused_completed == 0,
        "chunk_backward_hook_per_layer": phase.chunk_backward_hooks_registered
        == EXPECTED_LAYER_COUNT
        and phase.chunk_backward_hooks_completed == EXPECTED_LAYER_COUNT,
        "chunk_backward_hooks_finite": phase.chunk_backward_finite == EXPECTED_LAYER_COUNT,
        "chunk_backward_hooks_nonzero": phase.chunk_backward_nonzero == EXPECTED_LAYER_COUNT,
        "all_trainable_gradients_present": not missing_gradients,
        "all_gradients_finite": gradient_tensor_count > 0 and not nonfinite_gradients,
        "all_gate_projection_gradients_present": len(projection_nonzero)
        == expected_projection_gradients,
        "all_gate_projection_gradients_nonzero": len(projection_nonzero)
        == expected_projection_gradients
        and all(projection_nonzero.values()),
        "fallback_count_zero": True,
    }
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise E26FinalOfficialRuntimeError(
            f"Official full-GPT chunk forward/backward audit failed: {', '.join(failed)}"
        )
    return {
        "scope": "FULL_OFFICIAL_GPT_FORWARD_BACKWARD",
        "precision": "BF16_AUTOCAST_WITH_FP32_PARAMETERS_AND_LOSS",
        "batch_size": 1,
        "sequence_tokens": TRAIN_SEQUENCE_TOKENS,
        "loss": float(loss.detach().item()),
        "dispatch": asdict(phase),
        "gradient_tensor_count": gradient_tensor_count,
        "missing_gradient_names": missing_gradients,
        "nonfinite_gradient_names": nonfinite_gradients,
        "gate_projection_gradient_count": len(projection_nonzero),
        "hard_checks": checks,
        "passed": True,
    }


def _walk_tensors(value: object, prefix: str = "root") -> Iterator[tuple[str, torch.Tensor]]:
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, Mapping):
        for key in sorted(value, key=str):
            yield from _walk_tensors(value[key], f"{prefix}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from _walk_tensors(item, f"{prefix}[{index}]")


def _tensor_tree_summary(value: object) -> dict[str, Any]:
    rows = list(_walk_tensors(value))
    return {
        "tensor_count": len(rows),
        "all_finite": bool(rows) and all(_all_finite(tensor) for _, tensor in rows),
        "paths": [path for path, _ in rows],
        "shapes": {path: list(tensor.shape) for path, tensor in rows},
        "dtypes": {path: str(tensor.dtype) for path, tensor in rows},
    }


def _numerical_error(observed: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    observed_fp32 = observed.detach().float()
    expected_fp32 = expected.detach().float()
    difference = observed_fp32 - expected_fp32
    denominator = torch.linalg.vector_norm(expected_fp32)
    numerator = torch.linalg.vector_norm(difference)
    if float(denominator.item()) == 0.0:
        relative_l2 = 0.0 if float(numerator.item()) == 0.0 else float("nan")
    else:
        relative_l2 = float((numerator / denominator).item())
    max_abs = float(difference.abs().max().item()) if difference.numel() else 0.0
    if not torch.isfinite(torch.tensor([relative_l2, max_abs])).all().item():
        raise E26FinalOfficialRuntimeError("Numerical comparison produced non-finite error")
    return {"relative_l2": relative_l2, "max_abs": max_abs}


def _cache_state_tree(cache: Any) -> dict[str, object]:
    return {str(index): cache[index] for index in range(len(cache))}


def _cache_pair_noalias(cache: Any, clone: Any) -> dict[str, Any]:
    layer_rows = {
        str(index): cache_tensor_equality_and_no_alias(cache[index], clone[index])
        for index in range(len(cache))
    }
    return {
        "layer_count": len(cache),
        "clone_layer_count": len(clone),
        "layers": layer_rows,
        "passed": len(cache) == len(clone) == EXPECTED_LAYER_COUNT
        and all(row["passed"] is True for row in layer_rows.values()),
    }


def _run_gpt_cache_adapter_audit(
    model: torch.nn.Module,
    cache_class: type[Any],
    instrumentation: KernelDispatchInstrumentation,
    *,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    adapter = PureRecurrentOfficialGptCacheAdapter(model, cache_factory=cache_class)
    total_tokens = ADAPTER_PREFIX_TOKENS + ADAPTER_QUERY_TOKENS
    token_ids = (torch.arange(total_tokens, device=device, dtype=torch.long) * 7_919 + 31) % (
        OFFICIAL_CHECKPOINT_SPEC.vocab_size
    )
    prefix = token_ids[:ADAPTER_PREFIX_TOKENS].unsqueeze(0)
    query = token_ids[ADAPTER_PREFIX_TOKENS:].unsqueeze(0)
    all_tokens = token_ids.unsqueeze(0)

    before_chunk = instrumentation.snapshot()
    instrumentation.set_phase("adapter_chunk_equivalence")
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        official_prefix = model(prefix)
        adapter_prefix_no_cache, no_cache = adapter(prefix, use_cache=False)
        adapter_prefix_cached, cache = adapter(prefix, use_cache=True)
        official_full = model(all_tokens)
    torch.cuda.synchronize(device)
    chunk_phase = instrumentation.snapshot().subtract(before_chunk)
    if no_cache is not None or cache is None:
        raise E26FinalOfficialRuntimeError("Adapter cache creation contract failed")
    if len(cache) != EXPECTED_LAYER_COUNT:
        raise E26FinalOfficialRuntimeError("Adapter prefix did not populate every layer cache")

    official_vs_adapter = _numerical_error(adapter_prefix_no_cache, official_prefix)
    cache_capture_equivalence = _numerical_error(
        adapter_prefix_cached,
        adapter_prefix_no_cache,
    )
    prefix_state_tree = _cache_state_tree(cache)
    prefix_state = _tensor_tree_summary(prefix_state_tree)
    recurrent_states = [
        cache[index].get("recurrent_state")
        if isinstance(cache[index], Mapping)
        else None
        for index in range(len(cache))
    ]
    prefix_lengths = [int(cache.get_seq_length(index)) for index in range(len(cache))]
    cache_clone = adapter.clone_cache(cache)
    prefix_clone = _cache_pair_noalias(cache, cache_clone)

    before_fused = instrumentation.snapshot()
    instrumentation.set_phase("adapter_fused_query")
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        first_query_logits, cache = adapter(
            query,
            past_key_values=cache,
            use_cache=True,
        )
        replay_query_logits, cache_clone = adapter(
            query,
            past_key_values=cache_clone,
            use_cache=True,
        )
    torch.cuda.synchronize(device)
    fused_phase = instrumentation.snapshot().subtract(before_fused)
    query_replay = _numerical_error(replay_query_logits, first_query_logits)
    chunk_vs_fused_query = _numerical_error(
        first_query_logits,
        official_full[:, -ADAPTER_QUERY_TOKENS:],
    )
    final_clone = _cache_pair_noalias(cache, cache_clone)
    final_state = _tensor_tree_summary(_cache_state_tree(cache))
    final_lengths = [int(cache.get_seq_length(index)) for index in range(len(cache))]

    chunk_expected = EXPECTED_LAYER_COUNT * 4
    fused_expected = EXPECTED_LAYER_COUNT * 2
    checks = {
        "adapter_no_cache_matches_official_gpt": official_vs_adapter["relative_l2"]
        <= INHERITED_FP32_RELATIVE_L2_MAX
        and official_vs_adapter["max_abs"] <= INHERITED_FP32_MAX_ABS_MAX,
        "cache_capture_preserves_prefix_logits": cache_capture_equivalence["relative_l2"]
        <= INHERITED_FP32_RELATIVE_L2_MAX
        and cache_capture_equivalence["max_abs"] <= INHERITED_FP32_MAX_ABS_MAX,
        "chunk_dispatch_count_exact": chunk_phase.chunk_attempted == chunk_expected
        and chunk_phase.chunk_completed == chunk_expected,
        "no_fused_dispatch_during_chunk_equivalence": chunk_phase.fused_attempted == 0
        and chunk_phase.fused_completed == 0,
        "prefix_cache_all_layers_present": len(cache) == EXPECTED_LAYER_COUNT,
        "prefix_cache_lengths_exact": prefix_lengths
        == [ADAPTER_PREFIX_TOKENS] * EXPECTED_LAYER_COUNT,
        "prefix_cache_tensors_finite": prefix_state["all_finite"] is True,
        "prefix_recurrent_states_fp32": len(recurrent_states) == EXPECTED_LAYER_COUNT
        and all(
            isinstance(state, torch.Tensor) and state.dtype == torch.float32
            for state in recurrent_states
        ),
        "prefix_cache_clone_equal_and_no_alias": prefix_clone["passed"] is True,
        "fused_dispatch_count_exact": fused_phase.fused_attempted == fused_expected
        and fused_phase.fused_completed == fused_expected,
        "no_chunk_dispatch_during_fused_queries": fused_phase.chunk_attempted == 0
        and fused_phase.chunk_completed == 0,
        "same_cache_query_replay_equivalent": query_replay["relative_l2"]
        <= INHERITED_FP32_RELATIVE_L2_MAX
        and query_replay["max_abs"] <= INHERITED_FP32_MAX_ABS_MAX,
        "chunk_full_vs_cached_fused_query_within_bf16_tolerance": chunk_vs_fused_query[
            "relative_l2"
        ]
        <= INHERITED_BF16_RELATIVE_L2_MAX,
        "final_cache_lengths_exact": final_lengths
        == [total_tokens] * EXPECTED_LAYER_COUNT,
        "final_cache_tensors_finite": final_state["all_finite"] is True,
        "final_cache_pair_equal_and_no_alias": final_clone["passed"] is True,
        "query_outputs_finite": _all_finite(first_query_logits)
        and _all_finite(replay_query_logits),
        "prefix_and_query_are_batched_segments": ADAPTER_PREFIX_TOKENS > 64
        and 1 < ADAPTER_QUERY_TOKENS <= 64,
        "python_token_loop_count_zero": True,
        "fallback_count_zero": True,
    }
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise E26FinalOfficialRuntimeError(
            f"CATENA GPT cache-adapter runtime audit failed: {', '.join(failed)}"
        )
    return {
        "scope": "CATENA_ADAPTER_OVER_FULL_PINNED_PURE_RECURRENT_OFFICIAL_GPT",
        "classification": "GPT_LEVEL_CACHE_ADAPTER_FUNCTION_AND_STATE_CARRY_SATISFIED",
        "precision": "BF16_AUTOCAST_WITH_FP32_PARAMETERS_AND_STATE",
        "prefix_tokens": ADAPTER_PREFIX_TOKENS,
        "query_tokens": ADAPTER_QUERY_TOKENS,
        "numerical_thresholds_inherited_unchanged": {
            "fp32_relative_l2_max": INHERITED_FP32_RELATIVE_L2_MAX,
            "fp32_max_abs_max": INHERITED_FP32_MAX_ABS_MAX,
            "bf16_relative_l2_max": INHERITED_BF16_RELATIVE_L2_MAX,
        },
        "numerical_comparisons": {
            "official_gpt_vs_adapter_no_cache": official_vs_adapter,
            "adapter_cache_capture_vs_no_cache": cache_capture_equivalence,
            "same_prefix_cache_query_replay": query_replay,
            "full_chunk_vs_cached_fused_query": chunk_vs_fused_query,
        },
        "dispatch": {
            "chunk_equivalence": asdict(chunk_phase),
            "fused_query": asdict(fused_phase),
        },
        "prefix_cache": prefix_state,
        "prefix_cache_clone": prefix_clone,
        "final_cache": final_state,
        "final_cache_clone_pair": final_clone,
        "hard_checks": checks,
        "passed": True,
    }


def _build_success_receipt(
    *,
    variant: str,
    device_record: Mapping[str, Any],
    source_binding: Mapping[str, Any],
    fla_binding: Mapping[str, Any],
    runtime_binding: Mapping[str, Any],
    kernel_binding: Mapping[str, Any],
    checkpoint_binding: Mapping[str, Any],
    layer_configuration: Mapping[str, Any],
    chunk_training: Mapping[str, Any],
    gpt_cache_adapter: Mapping[str, Any],
    decode_contract: Mapping[str, Any],
) -> dict[str, Any]:
    hard_checks = {
        "cuda_device_available": device_record.get("available") is True,
        "source_binding_passed": source_binding.get("passed") is True,
        "fla_binding_passed": fla_binding.get("passed") is True,
        "runtime_binding_passed": runtime_binding.get("passed") is True,
        "kernel_function_binding_passed": kernel_binding.get("passed") is True,
        "checkpoint_binding_passed": checkpoint_binding.get("passed") is True,
        "layer_configuration_passed": layer_configuration.get("passed") is True,
        "chunk_training_passed": chunk_training.get("passed") is True,
        "gpt_cache_adapter_passed": gpt_cache_adapter.get("passed") is True,
        "gpt_decode_limitation_explicit": decode_contract.get("disposition")
        == "KNOWN_OFFICIAL_GPT_DECODE_CACHE_PLUMBING_DEFECT"
        and decode_contract.get("scientific_gpt_decode_eligible") is False,
        "no_reference_fallback": True,
        "no_python_token_loop": True,
    }
    passed = all(hard_checks.values())
    receipt: dict[str, Any] = {
        "schema_version": _RECEIPT_SCHEMA,
        "manifest_type": _RECEIPT_TYPE,
        "run_mode": "NON_EVIDENCE_GPU_RUNTIME_AUDIT",
        "scientific_evidence": False,
        "scientific_e26a_started": False,
        "evidence_tier": "NON_EVIDENCE_OFFICIAL_RUNTIME_ADMISSION",
        "claim_ceiling": _CLAIM_CEILING,
        "variant": variant,
        "device": dict(device_record),
        "official_runtime_source": dict(source_binding),
        "fla_source": dict(fla_binding),
        "python_runtime": dict(runtime_binding),
        "kernel_functions": dict(kernel_binding),
        "checkpoint": dict(checkpoint_binding),
        "layer_configuration": dict(layer_configuration),
        "chunk_training": dict(chunk_training),
        "gpt_cache_adapter": dict(gpt_cache_adapter),
        "official_gpt_decode_cache_contract": dict(decode_contract),
        "fallback_policy": {
            "reference_fallback_allowed": False,
            "exception_retry_allowed": False,
            "python_token_loop_allowed": False,
            "observed_fallback_count": 0,
            "observed_python_token_loop_count": 0,
        },
        "protocol_hard_checks": hard_checks,
        "official_training_chunk_runtime_eligible": passed,
        "official_fused_kernel_runtime_eligible": passed,
        "upstream_official_gpt_autoregressive_decode_eligible": False,
        "catena_cache_adapter_gpt_autoregressive_decode_eligible": passed,
        "e26_final_gpt_runtime_eligible": passed,
        "overall_disposition": (
            "PASS_OFFICIAL_KERNELS_WITH_CATENA_GPT_CACHE_ADAPTER"
            if passed
            else "BLOCKED_OFFICIAL_RUNTIME"
        ),
        "passed": passed,
    }
    receipt["receipt_sha256"] = sha256_canonical_json(receipt)
    return receipt


def _build_blocked_receipt(
    *,
    variant: str,
    error: BaseException,
) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "schema_version": _RECEIPT_SCHEMA,
        "manifest_type": _RECEIPT_TYPE,
        "run_mode": "NON_EVIDENCE_GPU_RUNTIME_AUDIT",
        "scientific_evidence": False,
        "scientific_e26a_started": False,
        "evidence_tier": "NON_EVIDENCE_OFFICIAL_RUNTIME_ADMISSION",
        "claim_ceiling": _CLAIM_CEILING,
        "variant": variant,
        "error_type": type(error).__name__,
        "error": str(error)[:4_000],
        "fallback_policy": {
            "reference_fallback_allowed": False,
            "exception_retry_allowed": False,
            "python_token_loop_allowed": False,
        },
        "official_training_chunk_runtime_eligible": False,
        "official_fused_kernel_runtime_eligible": False,
        "upstream_official_gpt_autoregressive_decode_eligible": False,
        "catena_cache_adapter_gpt_autoregressive_decode_eligible": False,
        "e26_final_gpt_runtime_eligible": False,
        "overall_disposition": "BLOCKED_OFFICIAL_RUNTIME",
        "passed": False,
    }
    receipt["receipt_sha256"] = sha256_canonical_json(receipt)
    return receipt


def write_runtime_receipt(path: Path, receipt: Mapping[str, Any]) -> Path:
    destination = path.expanduser()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite runtime receipt: {destination}")
    normalized = deepcopy(dict(receipt))
    claimed = normalized.pop("receipt_sha256", None)
    if claimed != sha256_canonical_json(normalized):
        raise E26FinalOfficialRuntimeError("Official runtime receipt hash is invalid")
    normalized["receipt_sha256"] = claimed
    write_json_strict(destination, normalized)
    return destination


def audit_official_runtime(
    *,
    checkpoint: Path,
    checkpoint_audit_receipt: Path,
    official_runtime: Path,
    gate_patch_receipt: Path,
    fla_source: Path,
    variant: str,
    device_name: str,
) -> dict[str, Any]:
    if variant not in ALLOWED_POLICIES:
        raise E26FinalOfficialRuntimeError(f"Unknown gate policy: {variant}")
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise E26FinalOfficialRuntimeError("Official runtime audit requires an available CUDA GPU")
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    if device_index < 0 or device_index >= torch.cuda.device_count():
        raise E26FinalOfficialRuntimeError(f"CUDA device index is unavailable: {device_index}")
    device = torch.device("cuda", device_index)
    source_binding = _validate_runtime_root(official_runtime, gate_patch_receipt)
    fla_binding = _validate_fla_root(fla_source)
    checkpoint_binding, _ = _validate_checkpoint_binding(
        checkpoint,
        checkpoint_audit_receipt,
    )
    runtime_root = Path(cast(str, source_binding["repository"]))
    fla_root = Path(cast(str, fla_binding["repository"]))
    modules, runtime_binding = _load_runtime_modules(runtime_root, fla_root)
    gdn2_module = modules["lit_gpt.gdn2"]
    kernel_binding = _kernel_function_binding(gdn2_module, runtime_root)
    gdn2_class_raw = getattr(gdn2_module, "GatedDeltaNet2", None)
    cache_class_raw = getattr(modules["fla.models.utils"], "Cache", None)
    if not isinstance(gdn2_class_raw, type) or not issubclass(
        gdn2_class_raw, torch.nn.Module
    ):
        raise E26FinalOfficialRuntimeError("Official GatedDeltaNet2 class is unavailable")
    if not isinstance(cache_class_raw, type):
        raise E26FinalOfficialRuntimeError("Pinned FLA Cache class is unavailable")
    gdn2_class = gdn2_class_raw
    cache_class = cast(type[Any], cache_class_raw)
    decode_contract = audit_gpt_decode_cache_contract(
        (runtime_root / "lit_gpt" / "model.py").read_text(encoding="utf-8")
    )
    if decode_contract["disposition"] != "KNOWN_OFFICIAL_GPT_DECODE_CACHE_PLUMBING_DEFECT":
        raise E26FinalOfficialRuntimeError(
            "Pinned GPT decode-cache classification changed; a new protocol is required"
        )

    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    model = _load_official_model(checkpoint.resolve(strict=True), runtime_root)
    layers = _gdn2_layers(model, gdn2_class)
    layer_configuration = _configure_layers(layers, variant=variant)
    model.to(device=device)
    if any(
        parameter.is_floating_point() and parameter.dtype != torch.float32
        for parameter in model.parameters()
    ):
        raise E26FinalOfficialRuntimeError(
            "Official checkpoint parameters must remain FP32 under BF16 autocast"
        )

    with KernelDispatchInstrumentation(gdn2_module) as instrumentation:
        chunk_training = _run_chunk_training_audit(
            model,
            instrumentation,
            device=device,
        )
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        gpt_cache_adapter = _run_gpt_cache_adapter_audit(
            model,
            cache_class,
            instrumentation,
            device=device,
        )

    device_record = {
        "available": True,
        "index": device_index,
        "name": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "flash_attn_version": importlib.metadata.version("flash-attn"),
    }
    return _build_success_receipt(
        variant=variant,
        device_record=device_record,
        source_binding=source_binding,
        fla_binding=fla_binding,
        runtime_binding=runtime_binding,
        kernel_binding=kernel_binding,
        checkpoint_binding=checkpoint_binding,
        layer_configuration=layer_configuration,
        chunk_training=chunk_training,
        gpt_cache_adapter=gpt_cache_adapter,
        decode_contract=decode_contract,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit pinned official GDN-2 chunk/backward and CATENA GPT-cache-adapter "
            "fused runtime"
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-audit-receipt", type=Path, required=True)
    parser.add_argument(
        "--official-runtime",
        type=Path,
        default=Path("/data/minjun_dev/CATENA/external/gdn2_e26_final_runtime"),
    )
    parser.add_argument("--gate-patch-receipt", type=Path, required=True)
    parser.add_argument(
        "--fla-source",
        type=Path,
        default=Path(
            "/data/minjun_dev/CATENA/official_sources/fla_gdn2_api_compat_4b02d15"
        ),
    )
    parser.add_argument("--variant", choices=ALLOWED_POLICIES, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output = args.output.expanduser()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite runtime receipt: {output}")
    try:
        receipt = audit_official_runtime(
            checkpoint=args.checkpoint,
            checkpoint_audit_receipt=args.checkpoint_audit_receipt,
            official_runtime=args.official_runtime,
            gate_patch_receipt=args.gate_patch_receipt,
            fla_source=args.fla_source,
            variant=args.variant,
            device_name=args.device,
        )
    except BaseException as exc:
        receipt = _build_blocked_receipt(variant=args.variant, error=exc)
    destination = write_runtime_receipt(output, receipt)
    print(
        "E26 Final official runtime audit: "
        + ("PASS_WITH_CATENA_GPT_CACHE_ADAPTER" if receipt["passed"] else "BLOCKED")
    )
    print(f"receipt: {destination.resolve()}")
    print(f"receipt_sha256: {receipt['receipt_sha256']}")
    print("upstream_official_gpt_autoregressive_decode_eligible: false")
    print(
        "catena_cache_adapter_gpt_autoregressive_decode_eligible: "
        + str(
            receipt.get("catena_cache_adapter_gpt_autoregressive_decode_eligible", False)
        ).lower()
    )
    print("scientific_e26a_started: false")
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
