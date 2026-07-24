from __future__ import annotations

from typing import Any

from catena.config import load_yaml

from .hf_stateful import HFStatefulAdapter
from .mock import MockStatefulModel
from .rwkv_pip import RWKVPipAdapter


def load_model(config_or_path: dict[str, Any] | str, device: str = "cuda"):
    config = load_yaml(config_or_path) if isinstance(config_or_path, str) else config_or_path
    backend = str(config.get("backend", "mock"))
    if backend == "mock":
        return MockStatefulModel()
    if backend == "rwkv_pip":
        return RWKVPipAdapter.from_config(config)
    if backend == "hf_stateful":
        return HFStatefulAdapter.from_config(config, device=device)
    raise ValueError(f"Unknown model backend: {backend}")
