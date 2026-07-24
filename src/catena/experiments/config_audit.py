from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from catena.config import load_yaml

_CONFIG_RE = re.compile(r"configs/(?:data|models|experiments)/[A-Za-z0-9_.-]+\.ya?ml")


def _path_value(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v) for v in value if isinstance(v, str)]
    return []


def audit_configs(root: str | Path = ".") -> dict[str, Any]:
    root_path = Path(root).resolve()
    config_paths = sorted((root_path / "configs").rglob("*.yaml"))
    errors: list[str] = []
    warnings: list[str] = []
    loaded: dict[str, dict[str, Any]] = {}

    for path in config_paths:
        rel = path.relative_to(root_path).as_posix()
        try:
            cfg = load_yaml(path)
            loaded[rel] = cfg
        except Exception as exc:
            errors.append(f"{rel}: YAML load failed: {exc!r}")
            continue
        if path.parent.name == "experiments":
            if "experiment" not in cfg:
                errors.append(f"{rel}: missing experiment")
            if "output_dir" not in cfg:
                errors.append(f"{rel}: missing output_dir")
        if path.parent.name == "models" and "backend" not in cfg:
            errors.append(f"{rel}: missing backend")

        for key in ("model", "models"):
            for candidate in _path_value(cfg.get(key)):
                if candidate.startswith("configs/") and not (root_path / candidate).exists():
                    errors.append(f"{rel}: {key} references missing {candidate}")
        for key in ("data_config",):
            for candidate in _path_value(cfg.get(key)):
                if candidate.startswith("configs/") and not (root_path / candidate).exists():
                    errors.append(f"{rel}: {key} references missing {candidate}")

    script_refs: set[str] = set()
    for script in sorted((root_path / "scripts").glob("*.sh")):
        text = script.read_text(encoding="utf-8")
        for match in _CONFIG_RE.findall(text):
            script_refs.add(match)
            if not (root_path / match).exists():
                errors.append(f"{script.relative_to(root_path)} references missing {match}")

    unreferenced_experiments = sorted(
        rel
        for rel in loaded
        if rel.startswith("configs/experiments/") and rel not in script_refs
    )
    for rel in unreferenced_experiments:
        warnings.append(f"experiment config is not referenced by a shell entry point: {rel}")

    payload = {
        "root": str(root_path),
        "config_count": len(config_paths),
        "script_config_reference_count": len(script_refs),
        "errors": errors,
        "warnings": warnings,
        "passed": not errors,
    }
    out = root_path / "artifacts" / "logs" / "config_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload
