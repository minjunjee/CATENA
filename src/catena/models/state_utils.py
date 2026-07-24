from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any


def clone_tree(value: Any) -> Any:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().clone()
    except ImportError:
        pass
    if isinstance(value, Mapping):
        return type(value)((k, clone_tree(v)) for k, v in value.items())
    if isinstance(value, tuple):
        return tuple(clone_tree(v) for v in value)
    if isinstance(value, list):
        return [clone_tree(v) for v in value]
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


def detach_tree(value: Any) -> Any:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach()
    except ImportError:
        pass
    if isinstance(value, Mapping):
        return type(value)((k, detach_tree(v)) for k, v in value.items())
    if isinstance(value, tuple):
        return tuple(detach_tree(v) for v in value)
    if isinstance(value, list):
        return [detach_tree(v) for v in value]
    return value


def tensor_tree_bytes(value: Any) -> int:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.numel() * value.element_size()
    except ImportError:
        pass
    if isinstance(value, Mapping):
        return sum(tensor_tree_bytes(v) for v in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return sum(tensor_tree_bytes(v) for v in value)
    # Hugging Face cache objects usually expose layers containing tensors.
    if hasattr(value, "layers"):
        return tensor_tree_bytes(getattr(value, "layers"))
    if hasattr(value, "key_cache") or hasattr(value, "value_cache"):
        return tensor_tree_bytes(getattr(value, "key_cache", [])) + tensor_tree_bytes(
            getattr(value, "value_cache", [])
        )
    if hasattr(value, "__dict__"):
        return tensor_tree_bytes(vars(value))
    return 0
