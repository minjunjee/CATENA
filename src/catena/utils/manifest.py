from __future__ import annotations

import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _git_value(args: list[str], cwd: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=cwd,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def runtime_manifest(
    config: dict[str, Any] | None = None,
    root: str | Path = ".",
) -> dict[str, Any]:
    root_path = Path(root).resolve()
    git_status = _git_value(["status", "--porcelain"], root_path)
    payload: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "conda_environment": os.getenv("CONDA_DEFAULT_ENV"),
        "git_commit": _git_value(["rev-parse", "HEAD"], root_path),
        "git_dirty": None if git_status is None else bool(git_status),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "config": config or {},
    }
    e00_path = root_path / "artifacts" / "profiles" / "e00_audit" / "latest_passed.json"
    if e00_path.is_file():
        try:
            payload["e00_gate"] = json.loads(e00_path.read_text(encoding="utf-8"))
        except Exception as exc:
            payload["e00_gate_error"] = repr(exc)
    else:
        payload["e00_gate"] = None
    try:
        import torch

        payload["torch"] = torch.__version__
        payload["torch_cuda_runtime"] = torch.version.cuda
        payload["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            payload["devices"] = [
                {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "capability": torch.cuda.get_device_capability(i),
                    "total_memory": torch.cuda.get_device_properties(i).total_memory,
                }
                for i in range(torch.cuda.device_count())
            ]
    except Exception as exc:
        payload["torch_error"] = repr(exc)
    return payload


def write_manifest(output_dir: str | Path, config: dict[str, Any] | None = None) -> Path:
    root = Path.cwd().resolve()
    output = Path(output_dir)
    output_path = output.resolve() if output.is_absolute() else (root / output).resolve()
    try:
        output_path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"manifest output must stay inside repository: {output_path}") from exc
    path = output_path / "run_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(runtime_manifest(config, root), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path
