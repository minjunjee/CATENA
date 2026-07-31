from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def hash_mapping(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def named_parameter_signature(module: torch.nn.Module) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
            "requires_grad": bool(parameter.requires_grad),
        }
        for name, parameter in module.named_parameters()
    ]


def parameter_signature_hash(module: torch.nn.Module) -> str:
    return hash_mapping(named_parameter_signature(module))


def state_dict_digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        value = tensor.detach().cpu().contiguous()
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def optimizer_state_signature(optimizer: torch.optim.Optimizer) -> str:
    records: list[dict[str, Any]] = []
    for group_index, group in enumerate(optimizer.param_groups):
        for parameter_index, parameter in enumerate(group["params"]):
            state = optimizer.state.get(parameter, {})
            fields: dict[str, Any] = {}
            for key, value in sorted(state.items(), key=lambda item: str(item[0])):
                if torch.is_tensor(value):
                    fields[str(key)] = {
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                    }
                else:
                    fields[str(key)] = type(value).__name__
            records.append(
                {
                    "group": group_index,
                    "parameter": parameter_index,
                    "shape": list(parameter.shape),
                    "state": fields,
                }
            )
    return hash_mapping(records)


def tree_digest(paths: Iterable[str | Path], root: str | Path | None = None) -> str:
    root_path = Path(root).resolve() if root is not None else None
    records: list[dict[str, str]] = []
    for value in sorted(Path(path).resolve() for path in paths):
        relative = str(value.relative_to(root_path)) if root_path else str(value)
        records.append({"path": relative, "sha256": sha256_file(value)})
    return hash_mapping(records)
