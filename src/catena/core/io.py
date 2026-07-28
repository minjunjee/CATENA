from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def ensure_artifact_dir(root: str | Path, experiment_id: str) -> Path:
    run_dir = Path(root) / experiment_id / utc_run_id()
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_snapshot() -> dict[str, Any]:
    import platform

    import torch

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "cwd": os.getcwd(),
    }


def write_latest_pointer(root: str | Path, experiment_id: str, run_dir: str | Path) -> None:
    target = Path(root) / experiment_id / "latest.json"
    write_json(target, {"run_dir": str(Path(run_dir).resolve())})


def read_latest_pointer(root: str | Path, experiment_id: str) -> Path:
    path = Path(root) / experiment_id / "latest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No latest pointer for {experiment_id}. Run the prerequisite experiment first."
        )
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return Path(payload["run_dir"])
