from __future__ import annotations

import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _git_value(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def runtime_manifest(config: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "git_commit": _git_value(["rev-parse", "HEAD"]),
        "git_dirty": bool(_git_value(["status", "--porcelain"])),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "config": config or {},
    }
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
    path = Path(output_dir) / "run_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(runtime_manifest(config), indent=2, ensure_ascii=False), encoding="utf-8")
    return path
