#!/usr/bin/env python3
"""Fail-closed structural and official-load audit for the E26 Final checkpoint.

The only deserialization path in this module is ``torch.load`` with
``weights_only=True``, CPU placement, and mmap enabled.  There is deliberately
no compatibility retry or unsafe fallback.  The command is an input-admission
audit, never scientific evidence and never a scientific launcher.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import re
import subprocess
import sys
from collections.abc import Callable, Mapping
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

import torch

from catena.core.provenance_v61 import (
    sha256_canonical_json,
    sha256_file,
    write_json_strict,
)
from catena.lm.e26_final_provenance import CHECKPOINT_FILE, OFFICIAL_SOURCE


class E26FinalCheckpointAuditError(RuntimeError):
    """Raised when a checkpoint audit receipt or official load is invalid."""


@dataclass(frozen=True)
class CheckpointAuditSpec:
    """Exact structural contract for one admitted checkpoint."""

    checkpoint_bytes: int
    checkpoint_sha256: str
    model_key_count: int
    full_model_numel: int
    transformer_h_numel: int
    vocab_size: int
    hidden_size: int
    layer_count: int
    projection_output_size: int
    max_tokens: int
    model_name: str = "gdn2_1.3B"

    @property
    def vocab_shape(self) -> tuple[int, int]:
        return (self.vocab_size, self.hidden_size)

    @property
    def projection_shape(self) -> tuple[int, int]:
        return (self.projection_output_size, self.hidden_size)


OFFICIAL_CHECKPOINT_SPEC: Final = CheckpointAuditSpec(
    checkpoint_bytes=CHECKPOINT_FILE.size,
    checkpoint_sha256=CHECKPOINT_FILE.sha256,
    model_key_count=399,
    full_model_numel=1_450_096_416,
    transformer_h_numel=1_302_638_112,
    vocab_size=32_000,
    hidden_size=2_304,
    layer_count=18,
    # Official GatedDeltaNet2 defaults: 16 heads x 128 dimensions.
    projection_output_size=2_048,
    max_tokens=100_000_000_000,
)

EXPECTED_TOP_LEVEL_KEYS: Final = frozenset(
    {"model", "optimizer", "hparams", "iter_num", "step_count"}
)
PROJECTION_KEY_PATTERN: Final = re.compile(
    r"^transformer\.h\.(?P<layer>\d+)\.attn\.(?P<kind>[bw]_proj)\.weight$"
)
_RECEIPT_SCHEMA: Final = "catena-e26-final-checkpoint-audit-v1"
_RECEIPT_TYPE: Final = "E26_FINAL_CHECKPOINT_AUDIT_RECEIPT"
_FINITE_CHUNK_ELEMENTS: Final = 1_048_576

ModelFactory = Callable[[], torch.nn.Module]


def safe_load_checkpoint(path: str | Path) -> Mapping[str, Any]:
    """Load one checkpoint with the sole admitted torch deserialization call."""

    target = Path(path)
    payload = torch.load(
        target,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if not isinstance(payload, Mapping):
        raise E26FinalCheckpointAuditError("Checkpoint root must be a mapping")
    return cast(Mapping[str, Any], payload)


def _is_nonnegative_int(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _tensor_is_finite(tensor: torch.Tensor) -> bool:
    if not (tensor.is_floating_point() or tensor.is_complex()):
        return True
    try:
        flat = tensor.detach().reshape(-1)
        for start in range(0, flat.numel(), _FINITE_CHUNK_ELEMENTS):
            part = flat.narrow(
                0,
                start,
                min(_FINITE_CHUNK_ELEMENTS, flat.numel() - start),
            )
            if not bool(torch.isfinite(part).all().item()):
                return False
    except (RuntimeError, TypeError):
        return False
    return True


def _projection_inventory(
    model: Mapping[str, object],
) -> tuple[dict[str, dict[int, list[int]]], list[str]]:
    inventory: dict[str, dict[int, list[int]]] = {"b_proj": {}, "w_proj": {}}
    malformed: list[str] = []
    for key, value in model.items():
        match = PROJECTION_KEY_PATTERN.fullmatch(key)
        if match is None:
            continue
        if not isinstance(value, torch.Tensor):
            malformed.append(key)
            continue
        inventory[match.group("kind")][int(match.group("layer"))] = list(value.shape)
    return inventory, sorted(malformed)


def audit_checkpoint_payload(
    payload: Mapping[Any, Any],
    *,
    spec: CheckpointAuditSpec = OFFICIAL_CHECKPOINT_SPEC,
) -> dict[str, Any]:
    """Audit exact checkpoint keys, tensor population, shapes, and hparams."""

    top_keys_are_strings = all(isinstance(key, str) for key in payload)
    top_keys = {str(key) for key in payload}
    model_raw = payload.get("model")
    model_is_mapping = isinstance(model_raw, Mapping)
    model: dict[str, object] = {}
    model_keys_are_strings = False
    if isinstance(model_raw, Mapping):
        typed_model_raw = cast(Mapping[object, object], model_raw)
        model_keys_are_strings = all(isinstance(key, str) for key in typed_model_raw)
        if model_keys_are_strings:
            model = {str(key): value for key, value in typed_model_raw.items()}

    non_tensor_keys = sorted(
        key for key, value in model.items() if not isinstance(value, torch.Tensor)
    )
    tensor_items = {
        key: value for key, value in model.items() if isinstance(value, torch.Tensor)
    }
    nonfinite_keys = sorted(
        key for key, tensor in tensor_items.items() if not _tensor_is_finite(tensor)
    )
    full_numel = sum(tensor.numel() for tensor in tensor_items.values())
    transformer_h_numel = sum(
        tensor.numel()
        for key, tensor in tensor_items.items()
        if key.startswith("transformer.h.")
    )
    projection_inventory, malformed_projection_keys = _projection_inventory(model)
    expected_layers = set(range(spec.layer_count))
    b_layers = set(projection_inventory["b_proj"])
    w_layers = set(projection_inventory["w_proj"])
    expected_projection_shape = list(spec.projection_shape)
    projection_shapes_exact = all(
        shape == expected_projection_shape
        for group in projection_inventory.values()
        for shape in group.values()
    )
    projection_pairs_match = all(
        projection_inventory["b_proj"].get(layer)
        == projection_inventory["w_proj"].get(layer)
        for layer in expected_layers
    )

    wte = tensor_items.get("transformer.wte.weight")
    lm_head = tensor_items.get("lm_head.weight")
    hparams_raw = payload.get("hparams")
    hparams_is_mapping = isinstance(hparams_raw, Mapping)
    observed_max_tokens = (
        cast(Mapping[object, object], hparams_raw).get("max_tokens")
        if isinstance(hparams_raw, Mapping)
        else None
    )

    checks = {
        "top_level_keys_exact": top_keys_are_strings
        and top_keys == EXPECTED_TOP_LEVEL_KEYS,
        "model_is_mapping": model_is_mapping,
        "model_keys_are_strings": model_keys_are_strings,
        "all_model_values_are_tensors": model_is_mapping
        and model_keys_are_strings
        and not non_tensor_keys,
        "all_model_tensors_finite": bool(tensor_items)
        and not non_tensor_keys
        and not nonfinite_keys,
        "model_key_count_exact": len(model) == spec.model_key_count,
        "full_model_numel_exact": full_numel == spec.full_model_numel,
        "transformer_h_numel_exact": transformer_h_numel
        == spec.transformer_h_numel,
        "transformer_wte_shape_exact": isinstance(wte, torch.Tensor)
        and tuple(wte.shape) == spec.vocab_shape,
        "lm_head_shape_exact": isinstance(lm_head, torch.Tensor)
        and tuple(lm_head.shape) == spec.vocab_shape,
        "b_projection_layers_exact": b_layers == expected_layers,
        "w_projection_layers_exact": w_layers == expected_layers,
        "projection_shapes_exact": projection_shapes_exact
        and not malformed_projection_keys,
        "paired_b_w_projection_shapes_match": projection_pairs_match,
        "hparams_is_mapping": hparams_is_mapping,
        "hparams_max_tokens_exact": _is_nonnegative_int(observed_max_tokens)
        and observed_max_tokens == spec.max_tokens,
        "iter_num_is_nonnegative_integer": _is_nonnegative_int(payload.get("iter_num")),
        "step_count_is_nonnegative_integer": _is_nonnegative_int(
            payload.get("step_count")
        ),
    }
    return {
        "expected": {
            "top_level_keys": sorted(EXPECTED_TOP_LEVEL_KEYS),
            "model_key_count": spec.model_key_count,
            "full_model_numel": spec.full_model_numel,
            "transformer_h_numel": spec.transformer_h_numel,
            "vocab_shape": list(spec.vocab_shape),
            "projection_layers_per_kind": spec.layer_count,
            "projection_shape": expected_projection_shape,
            "hparams_max_tokens": spec.max_tokens,
        },
        "observed": {
            "top_level_keys": sorted(top_keys),
            "model_key_count": len(model),
            "tensor_count": len(tensor_items),
            "full_model_numel": full_numel,
            "transformer_h_numel": transformer_h_numel,
            "transformer_wte_shape": list(wte.shape)
            if isinstance(wte, torch.Tensor)
            else None,
            "lm_head_shape": list(lm_head.shape)
            if isinstance(lm_head, torch.Tensor)
            else None,
            "projection_inventory": {
                kind: {str(layer): shape for layer, shape in sorted(rows.items())}
                for kind, rows in projection_inventory.items()
            },
            "hparams_max_tokens": observed_max_tokens,
            "iter_num": payload.get("iter_num"),
            "step_count": payload.get("step_count"),
            "optimizer_payload_type": (
                "None"
                if payload.get("optimizer") is None
                else type(payload.get("optimizer")).__name__
            ),
        },
        "non_tensor_model_keys": non_tensor_keys,
        "nonfinite_model_keys": nonfinite_keys,
        "malformed_projection_keys": malformed_projection_keys,
        "hard_checks": checks,
        "passed": all(checks.values()),
    }


def _module_numel(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def audit_strict_model_load(
    model_state: Mapping[str, torch.Tensor],
    *,
    model_factory: ModelFactory,
    spec: CheckpointAuditSpec = OFFICIAL_CHECKPOINT_SPEC,
) -> dict[str, Any]:
    """Strictly assign a state dict to a supplied official-model factory.

    The injectable factory makes this contract testable with tiny models.  The
    production CLI obtains its factory only from the pinned official checkout.
    """

    checks = {
        "factory_returned_torch_module": False,
        "official_full_parameter_numel_exact": False,
        "official_transformer_h_parameter_numel_exact": False,
        "official_state_key_set_exact": False,
        "strict_state_dict_load_completed": False,
        "strict_load_missing_keys_empty": False,
        "strict_load_unexpected_keys_empty": False,
        "loaded_state_shapes_exact": False,
    }
    observed: dict[str, Any] = {
        "model_class": None,
        "model_module": None,
        "full_parameter_numel": None,
        "transformer_h_parameter_numel": None,
        "missing_keys": None,
        "unexpected_keys": None,
        "load_error_type": None,
        "load_error": None,
    }
    try:
        model = model_factory()
        checks["factory_returned_torch_module"] = isinstance(model, torch.nn.Module)
        if not isinstance(model, torch.nn.Module):
            raise TypeError("Official model factory did not return torch.nn.Module")
        observed["model_class"] = type(model).__qualname__
        observed["model_module"] = type(model).__module__
        observed["full_parameter_numel"] = _module_numel(model)
        checks["official_full_parameter_numel_exact"] = (
            observed["full_parameter_numel"] == spec.full_model_numel
        )
        transformer = getattr(model, "transformer", None)
        transformer_h = getattr(transformer, "h", None)
        if not isinstance(transformer_h, torch.nn.Module):
            raise TypeError("Official model lacks transformer.h module")
        observed["transformer_h_parameter_numel"] = _module_numel(transformer_h)
        checks["official_transformer_h_parameter_numel_exact"] = (
            observed["transformer_h_parameter_numel"] == spec.transformer_h_numel
        )
        before = model.state_dict()
        checks["official_state_key_set_exact"] = set(before) == set(model_state)
        incompatible = model.load_state_dict(model_state, strict=True, assign=True)
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)
        observed["missing_keys"] = missing
        observed["unexpected_keys"] = unexpected
        checks["strict_state_dict_load_completed"] = True
        checks["strict_load_missing_keys_empty"] = not missing
        checks["strict_load_unexpected_keys_empty"] = not unexpected
        after = model.state_dict()
        checks["loaded_state_shapes_exact"] = set(after) == set(model_state) and all(
            tuple(after[key].shape) == tuple(value.shape)
            for key, value in model_state.items()
        )
    except Exception as exc:  # fail closed and preserve the exact failure class
        observed["load_error_type"] = type(exc).__name__
        observed["load_error"] = str(exc)[:2_000]
    return {
        "load_policy": {
            "strict": True,
            "assign": True,
            "missing_keys_allowed": False,
            "unexpected_keys_allowed": False,
        },
        "observed": observed,
        "hard_checks": checks,
        "passed": all(checks.values()),
    }


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise E26FinalCheckpointAuditError(
            f"Git command failed ({' '.join(arguments)}): {result.stderr.strip()}"
        )
    return result.stdout.strip()


def _module_origin_within(module: object, root: Path) -> bool:
    filename = getattr(module, "__file__", None)
    if not isinstance(filename, str):
        return False
    try:
        return Path(filename).resolve(strict=True).is_relative_to(root)
    except (FileNotFoundError, OSError):
        return False


def resolve_official_gpt_factory(
    official_source: str | Path,
    *,
    spec: CheckpointAuditSpec = OFFICIAL_CHECKPOINT_SPEC,
) -> tuple[ModelFactory, dict[str, Any]]:
    """Resolve GPT only from the pinned checkout; never use an installed fallback."""

    unresolved = Path(official_source).expanduser()
    if unresolved.is_symlink():
        raise E26FinalCheckpointAuditError("Official source must not be a symlink")
    root = unresolved.resolve(strict=True)
    if not root.is_dir():
        raise E26FinalCheckpointAuditError("Official source is not a directory")
    expected_model_file = (root / "lit_gpt" / "model.py").resolve(strict=True)
    head = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    if head != OFFICIAL_SOURCE.commit or tree != OFFICIAL_SOURCE.tree:
        raise E26FinalCheckpointAuditError("Official source commit/tree does not match pin")

    for name, loaded in tuple(sys.modules.items()):
        if (name == "lit_gpt" or name.startswith("lit_gpt.")) and not _module_origin_within(
            loaded, root
        ):
            raise E26FinalCheckpointAuditError(
                f"Refusing preloaded non-official Python module: {name}"
            )

    root_text = str(root)
    sys.path.insert(0, root_text)
    try:
        module = importlib.import_module("lit_gpt.model")
    finally:
        if sys.path and sys.path[0] == root_text:
            sys.path.pop(0)
        else:
            with suppress(ValueError):
                sys.path.remove(root_text)
    gpt_class = getattr(module, "GPT", None)
    if not isinstance(gpt_class, type) or not issubclass(gpt_class, torch.nn.Module):
        raise E26FinalCheckpointAuditError("Official lit_gpt.model.GPT is unavailable")
    class_file = Path(inspect.getfile(gpt_class)).resolve(strict=True)
    if class_file != expected_model_file:
        raise E26FinalCheckpointAuditError("GPT class was imported outside official checkout")
    typed_gpt_class = cast(Any, gpt_class)

    def factory() -> torch.nn.Module:
        with torch.device("meta"):
            model: object = typed_gpt_class.from_name(spec.model_name)
        if not isinstance(model, torch.nn.Module):
            raise E26FinalCheckpointAuditError("GPT.from_name returned a non-module")
        return model

    binding: dict[str, Any] = {
        "repository": str(root),
        "commit": head,
        "tree": tree,
        "model_file": str(class_file),
        "model_name": spec.model_name,
        "construction_device": "meta",
        "strict_assign_load": True,
        "hard_checks": {
            "official_commit_exact": head == OFFICIAL_SOURCE.commit,
            "official_tree_exact": tree == OFFICIAL_SOURCE.tree,
            "gpt_class_file_exact": class_file == expected_model_file,
        },
    }
    binding["passed"] = all(binding["hard_checks"].values())
    return factory, binding


def audit_official_gpt_load(
    model_state: Mapping[str, torch.Tensor],
    *,
    official_source: str | Path,
    spec: CheckpointAuditSpec = OFFICIAL_CHECKPOINT_SPEC,
) -> dict[str, Any]:
    """Construct and strictly load the exact official GPT implementation."""

    try:
        factory, source = resolve_official_gpt_factory(official_source, spec=spec)
    except Exception as exc:
        checks = {
            "official_source_resolved": False,
            "official_gpt_strict_load": False,
        }
        return {
            "source_binding": None,
            "strict_load": None,
            "error_type": type(exc).__name__,
            "error": str(exc)[:2_000],
            "hard_checks": checks,
            "passed": False,
        }
    strict_load = audit_strict_model_load(
        model_state,
        model_factory=factory,
        spec=spec,
    )
    source_checks = source.get("hard_checks")
    if not isinstance(source_checks, Mapping):
        raise E26FinalCheckpointAuditError("Official source binding lacks hard checks")
    checks = {
        **{f"source.{key}": bool(value) for key, value in source_checks.items()},
        **{
            f"load.{key}": bool(value)
            for key, value in strict_load["hard_checks"].items()
        },
    }
    return {
        "source_binding": source,
        "strict_load": strict_load,
        "hard_checks": checks,
        "passed": all(checks.values()),
    }


def _blocked_section(reason: str) -> dict[str, Any]:
    return {
        "reason": reason,
        "hard_checks": {"evaluated_and_passed": False},
        "passed": False,
    }


def build_checkpoint_audit_receipt(
    *,
    checkpoint_file: Mapping[str, Any],
    structure: Mapping[str, Any],
    official_load: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a deterministic non-evidence receipt with warning separation."""

    sections = {
        "checkpoint_file": deepcopy(dict(checkpoint_file)),
        "checkpoint_structure": deepcopy(dict(structure)),
        "official_gpt_load": deepcopy(dict(official_load)),
    }
    checks: dict[str, bool] = {}
    for section_name, section in sections.items():
        section_checks = section.get("hard_checks")
        if not isinstance(section_checks, Mapping) or not section_checks or not all(
            isinstance(value, bool) for value in section_checks.values()
        ):
            raise E26FinalCheckpointAuditError(
                f"{section_name} lacks non-empty boolean hard checks"
            )
        section_passed = all(section_checks.values())
        if section.get("passed") is not section_passed:
            raise E26FinalCheckpointAuditError(
                f"{section_name} disposition differs from hard checks"
            )
        checks.update(
            {f"{section_name}.{key}": value for key, value in section_checks.items()}
        )
    warnings = [
        {
            "code": "CHECKPOINT_95B_AND_100B_BYTE_IDENTICAL",
            "severity": "HIGH",
            "protocol_hard_gate": False,
            "detail": (
                "Upstream model-95b.pth and model-100b.pth metadata bind byte-identical "
                "content; the 95b alias is a provenance warning only and never changes "
                "the structural or strict-load admission checks."
            ),
        }
    ]
    passed = all(checks.values())
    receipt: dict[str, Any] = {
        "schema_version": _RECEIPT_SCHEMA,
        "manifest_type": _RECEIPT_TYPE,
        "scientific_evidence": False,
        "evidence_tier": "NON_EVIDENCE_CHECKPOINT_ADMISSION",
        "claim_ceiling": "CHECKPOINT_STRUCTURE_AND_OFFICIAL_STRICT_LOAD_ONLY",
        "deserialization_policy": {
            "map_location": "cpu",
            "weights_only": True,
            "mmap": True,
            "unsafe_fallback_allowed": False,
        },
        **sections,
        "protocol_hard_checks": checks,
        "warnings": warnings,
        "warning_count": len(warnings),
        "checkpoint_admission_eligible": passed,
        "scientific_e26a_started": False,
        "passed": passed,
    }
    receipt["receipt_sha256"] = sha256_canonical_json(receipt)
    return validate_checkpoint_audit_receipt(receipt)


def validate_checkpoint_audit_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate receipt hash, warning separation, and section dispositions."""

    normalized = deepcopy(dict(payload))
    claimed = normalized.pop("receipt_sha256", None)
    if claimed != sha256_canonical_json(normalized):
        raise E26FinalCheckpointAuditError("Checkpoint receipt SHA-256 changed")
    normalized["receipt_sha256"] = claimed
    if (
        normalized.get("schema_version") != _RECEIPT_SCHEMA
        or normalized.get("manifest_type") != _RECEIPT_TYPE
        or normalized.get("scientific_evidence") is not False
        or normalized.get("scientific_e26a_started") is not False
    ):
        raise E26FinalCheckpointAuditError("Checkpoint receipt contract changed")
    checks = normalized.get("protocol_hard_checks")
    if not isinstance(checks, Mapping) or not checks or not all(
        isinstance(value, bool) for value in checks.values()
    ):
        raise E26FinalCheckpointAuditError("Checkpoint receipt hard-check map is invalid")
    for section_name in (
        "checkpoint_file",
        "checkpoint_structure",
        "official_gpt_load",
    ):
        section = normalized.get(section_name)
        if not isinstance(section, Mapping):
            raise E26FinalCheckpointAuditError(f"Receipt lacks {section_name}")
        section_checks = section.get("hard_checks")
        if not isinstance(section_checks, Mapping) or section.get("passed") is not all(
            section_checks.values()
        ):
            raise E26FinalCheckpointAuditError(
                f"{section_name} disposition differs from hard checks"
            )
    warnings = normalized.get("warnings")
    if not isinstance(warnings, list) or normalized.get("warning_count") != len(warnings):
        raise E26FinalCheckpointAuditError("Checkpoint warning population is invalid")
    if any(
        not isinstance(row, Mapping)
        or row.get("protocol_hard_gate") is not False
        or not isinstance(row.get("code"), str)
        or not isinstance(row.get("detail"), str)
        for row in warnings
    ):
        raise E26FinalCheckpointAuditError("Warnings were conflated with hard gates")
    expected = all(checks.values())
    if normalized.get("passed") is not expected or normalized.get(
        "checkpoint_admission_eligible"
    ) is not expected:
        raise E26FinalCheckpointAuditError("Checkpoint receipt disposition is inconsistent")
    return normalized


def write_checkpoint_audit_receipt(
    path: str | Path,
    payload: Mapping[str, Any],
) -> Path:
    """Write one immutable strict-JSON checkpoint receipt."""

    destination = Path(path).expanduser()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite checkpoint receipt: {destination}")
    validated = validate_checkpoint_audit_receipt(payload)
    write_json_strict(destination, validated)
    return destination


def audit_checkpoint_file(
    checkpoint: str | Path,
    *,
    official_source: str | Path,
    spec: CheckpointAuditSpec = OFFICIAL_CHECKPOINT_SPEC,
) -> dict[str, Any]:
    """Hash, safely mmap-load, structurally audit, and strict-load a checkpoint."""

    unresolved = Path(checkpoint).expanduser()
    if unresolved.is_symlink():
        raise E26FinalCheckpointAuditError("Checkpoint must not be a symlink")
    target = unresolved.resolve(strict=True)
    if not target.is_file():
        raise E26FinalCheckpointAuditError("Checkpoint path is not a regular file")
    observed_bytes = target.stat().st_size
    observed_sha256 = sha256_file(target)
    file_checks = {
        "regular_file_not_symlink": True,
        "checkpoint_bytes_exact": observed_bytes == spec.checkpoint_bytes,
        "checkpoint_sha256_exact": observed_sha256 == spec.checkpoint_sha256,
    }
    file_section = {
        "path": str(target),
        "bytes": observed_bytes,
        "sha256": observed_sha256,
        "expected_bytes": spec.checkpoint_bytes,
        "expected_sha256": spec.checkpoint_sha256,
        "hard_checks": file_checks,
        "passed": all(file_checks.values()),
    }
    if not file_section["passed"]:
        return build_checkpoint_audit_receipt(
            checkpoint_file=file_section,
            structure=_blocked_section("Checkpoint byte binding failed; deserialization refused"),
            official_load=_blocked_section("Checkpoint byte binding failed"),
        )
    try:
        payload = safe_load_checkpoint(target)
    except Exception as exc:
        return build_checkpoint_audit_receipt(
            checkpoint_file=file_section,
            structure=_blocked_section(
                f"Safe weights-only mmap load failed: {type(exc).__name__}: {str(exc)[:1000]}"
            ),
            official_load=_blocked_section("Safe checkpoint load failed"),
        )
    structure = audit_checkpoint_payload(payload, spec=spec)
    model_raw = payload.get("model")
    if not isinstance(model_raw, Mapping) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in model_raw.items()
    ):
        official_load = _blocked_section("Checkpoint model state is not a tensor mapping")
    else:
        model_state = cast(Mapping[str, torch.Tensor], model_raw)
        official_load = audit_official_gpt_load(
            model_state,
            official_source=official_source,
            spec=spec,
        )
    return build_checkpoint_audit_receipt(
        checkpoint_file=file_section,
        structure=structure,
        official_load=official_load,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Safely mmap-load and structurally audit the exact E26 Final checkpoint, "
            "then strictly load it into the pinned official GPT implementation"
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--official-source",
        type=Path,
        default=Path("/data/minjun_dev/CATENA/external/gdn2_official"),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    receipt = audit_checkpoint_file(
        args.checkpoint,
        official_source=args.official_source,
    )
    output = write_checkpoint_audit_receipt(args.output, receipt)
    print(f"E26 Final checkpoint admission: {'PASS' if receipt['passed'] else 'BLOCKED'}")
    print(f"receipt: {output.resolve()}")
    print(f"receipt_sha256: {receipt['receipt_sha256']}")
    print("scientific_e26a_started: false")
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
