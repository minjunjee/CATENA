from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when a configuration file is missing or malformed."""


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    p = Path(path)
    if not p.exists():
        raise ConfigError(f"Configuration file does not exist: {p}")
    with p.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ConfigError(f"Top-level YAML value must be a mapping: {p}")
    return payload


def deep_get(mapping: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    value: Any = mapping
    for key in dotted_key.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def deep_set(mapping: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    node = mapping
    for key in keys[:-1]:
        child = node.get(key)
        if child is None:
            child = {}
            node[key] = child
        if not isinstance(child, dict):
            raise ConfigError(f"Cannot set {dotted_key}: {key} is not a mapping")
        node = child
    node[keys[-1]] = value


@dataclass(frozen=True)
class RunPaths:
    root: Path
    output_dir: Path

    @classmethod
    def from_config(cls, config: dict[str, Any], root: str | Path = ".") -> "RunPaths":
        root_path = Path(root).resolve()
        output = config.get("output_dir")
        if not output:
            raise ConfigError("Configuration must define output_dir")
        output_path = (root_path / str(output)).resolve()
        try:
            output_path.relative_to(root_path)
        except ValueError as exc:
            raise ConfigError(
                f"Configuration output_dir must stay inside repository: {output_path}"
            ) from exc
        output_path.mkdir(parents=True, exist_ok=True)
        return cls(root=root_path, output_dir=output_path)
