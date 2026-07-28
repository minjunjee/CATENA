from __future__ import annotations

import hashlib
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from catena.core.config import load_config
from catena.core.io import environment_snapshot, write_json
from catena.core.schema import CandidateMode, ControllerKind, Operation
from catena.data.tamp import TAMPConfig, build_episode, validate_episode
from catena.models.reference_recurrence import (
    gdn2_reference_update,
    kda_tied_reference_update,
)
from catena.theory.control_geometry import local_control_geometry
from experiments.common import build_parser, finalize_run, initialize_run

EXPERIMENT_ID = "e00_protocol_lock"
DEFAULT_CONFIG = "configs/e00_protocol_lock.yaml"
REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_command(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "output_tail": completed.stdout[-8000:],
    }


def _source_fingerprint() -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    allowed = {".py", ".yaml", ".yml", ".md", ".toml", ".tex", ".bib"}
    excluded_parts = {"artifacts", ".git", ".pytest_cache", "__pycache__"}
    for path in sorted(REPO_ROOT.rglob("*")):
        if not path.is_file() or path.suffix not in allowed:
            continue
        if any(part in excluded_parts for part in path.parts):
            continue
        relative = path.relative_to(REPO_ROOT).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
        count += 1
    return digest.hexdigest(), count


def _git_snapshot() -> dict[str, Any]:
    if not (REPO_ROOT / ".git").exists():
        return {"available": False, "reason": "release archive has no .git directory"}
    sha = _run_command(["git", "rev-parse", "HEAD"])
    status = _run_command(["git", "status", "--short"])
    return {
        "available": True,
        "head": sha["output_tail"].strip(),
        "dirty": bool(status["output_tail"].strip()),
        "status": status["output_tail"],
    }


def _storage_round_trip(run_dir: Path, size_mib: int) -> dict[str, Any]:
    path = run_dir / "storage_probe.bin"
    block = hashlib.sha256(b"CATENA-E00-storage-probe").digest() * 4096
    target_bytes = size_mib * 1024 * 1024
    written = 0
    source_hash = hashlib.sha256()
    with path.open("wb") as handle:
        while written < target_bytes:
            chunk = block[: min(len(block), target_bytes - written)]
            handle.write(chunk)
            source_hash.update(chunk)
            written += len(chunk)
        handle.flush()
        os.fsync(handle.fileno())
    read_hash = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            read_hash.update(chunk)
    path.unlink(missing_ok=True)
    return {
        "bytes": written,
        "source_sha256": source_hash.hexdigest(),
        "read_sha256": read_hash.hexdigest(),
        "match": source_hash.hexdigest() == read_hash.hexdigest(),
    }


def _gpu_bf16_checks(indices: list[int]) -> tuple[list[dict[str, Any]], float | None]:
    generator = torch.Generator().manual_seed(20260726)
    left = torch.randn(96, 96, generator=generator, dtype=torch.float32)
    right = torch.randn(96, 96, generator=generator, dtype=torch.float32)
    reference = left @ right
    outputs: list[torch.Tensor] = []
    checks: list[dict[str, Any]] = []
    for index in indices:
        device = torch.device(f"cuda:{index}")
        output = (left.to(device=device, dtype=torch.bfloat16) @ right.to(
            device=device, dtype=torch.bfloat16
        )).float().cpu()
        relative_l2 = float((output - reference).norm() / reference.norm().clamp_min(1e-12))
        outputs.append(output)
        checks.append(
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "finite": bool(torch.isfinite(output).all().item()),
                "relative_l2_vs_fp32": relative_l2,
                "capability": list(torch.cuda.get_device_capability(index)),
                "total_memory_bytes": int(torch.cuda.get_device_properties(index).total_memory),
            }
        )
    cross_gpu_max_abs: float | None = None
    if outputs:
        anchor = outputs[0]
        cross_gpu_max_abs = max(float((output - anchor).abs().max().item()) for output in outputs)
    return checks, cross_gpu_max_abs


def _audit_configs() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted((REPO_ROOT / "configs").glob("*.yaml")):
        try:
            data = load_config(path)
            rows.append(
                {
                    "path": str(path.relative_to(REPO_ROOT)),
                    "ok": isinstance(data, dict) and bool(data.get("experiment_id")),
                    "experiment_id": data.get("experiment_id"),
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append({"path": str(path.relative_to(REPO_ROOT)), "ok": False, "error": str(exc)})
    return {"count": len(rows), "rows": rows, "all_ok": all(row["ok"] for row in rows)}


def main() -> None:
    parser = build_parser(EXPERIMENT_ID, DEFAULT_CONFIG)
    args = parser.parse_args()
    config, run_dir, _ = initialize_run(
        experiment_id=EXPERIMENT_ID,
        config_path=args.config,
        artifact_root=args.artifact_root,
        device_request=args.device,
    )
    failures: list[str] = []
    warnings: list[str] = []
    passes: list[str] = []

    expected = config["environment"]
    python_major_minor = ".".join(platform.python_version().split(".")[:2])
    if python_major_minor != str(expected["required_python_major_minor"]):
        message = f"Python {python_major_minor} != required {expected['required_python_major_minor']}"
        (warnings if args.dry_run else failures).append(message)
    else:
        passes.append("python_version")
    if not str(torch.__version__).startswith(str(expected["required_torch_prefix"])):
        message = f"PyTorch {torch.__version__} does not match {expected['required_torch_prefix']}"
        (warnings if args.dry_run else failures).append(message)
    else:
        passes.append("torch_version")
    if str(torch.version.cuda) != str(expected["required_torch_cuda"]):
        message = f"torch CUDA {torch.version.cuda} != required {expected['required_torch_cuda']}"
        (warnings if args.dry_run else failures).append(message)
    else:
        passes.append("torch_cuda_version")

    tamp_cfg = TAMPConfig(**config["tamp"])
    episode_checks: list[dict[str, object]] = []
    for mode in CandidateMode:
        for operation in Operation:
            episode = build_episode(
                seed=1000 + len(episode_checks),
                operation=operation,
                candidate_mode=mode,
                config=tamp_cfg,
            )
            try:
                validate_episode(episode, atol=float(config["tolerances"]["target_atol"]))
                geometry = local_control_geometry(episode, ControllerKind.TIED_SCALAR)
                episode_checks.append(
                    {
                        "mode": mode.value,
                        "operation": operation.value,
                        "valid": True,
                        "tied_projection_regret": geometry.projection_regret,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                failures.append(f"Episode invariant failed: {mode.value}/{operation.value}: {exc}")
    if len(episode_checks) == len(CandidateMode) * len(Operation):
        passes.append("tamp_invariants")

    dtype = torch.float64
    generator = torch.Generator().manual_seed(7)
    dim = int(config["reference_parity"]["dim"])
    state = torch.randn(dim, dim, generator=generator, dtype=dtype)
    key = torch.randn(dim, generator=generator, dtype=dtype)
    key = key / key.norm()
    value = torch.randn(dim, generator=generator, dtype=dtype)
    decay = torch.sigmoid(torch.randn(dim, generator=generator, dtype=dtype))
    beta = torch.sigmoid(torch.randn((), generator=generator, dtype=dtype))
    gdn2 = gdn2_reference_update(
        state, key, value, decay, beta.expand_as(key), beta.expand_as(value)
    )
    kda = kda_tied_reference_update(state, key, value, decay, beta)
    parity_max_abs = float((gdn2 - kda).abs().max().item())
    if parity_max_abs > float(config["tolerances"]["fp64_parity_max_abs"]):
        failures.append(f"Reference tied/KDA parity exceeded tolerance: {parity_max_abs}")
    else:
        passes.append("reference_tied_kda_parity")

    selected_gpus = [int(value) for value in expected["selected_gpu_indices"]]
    gpu_checks: list[dict[str, Any]] = []
    cross_gpu_max_abs: float | None = None
    if torch.cuda.is_available() and torch.cuda.device_count() > max(selected_gpus):
        gpu_checks, cross_gpu_max_abs = _gpu_bf16_checks(selected_gpus)
        bf16_limit = float(config["tolerances"]["bf16_relative_l2"])
        if any((not row["finite"]) or float(row["relative_l2_vs_fp32"]) > bf16_limit for row in gpu_checks):
            failures.append("At least one selected GPU failed the BF16 numerical tolerance.")
        else:
            passes.append("selected_gpu_bf16")
        if cross_gpu_max_abs is not None and cross_gpu_max_abs > float(
            config["tolerances"]["cross_gpu_max_abs"]
        ):
            failures.append(f"Cross-GPU BF16 mismatch is {cross_gpu_max_abs}")
        else:
            passes.append("cross_gpu_parity")
    else:
        message = (
            f"Visible GPU count {torch.cuda.device_count()} cannot cover selected indices "
            f"{selected_gpus}."
        )
        (warnings if args.dry_run else failures).append(message)

    disk = shutil.disk_usage(run_dir)
    free_gib = disk.free / (1024**3)
    if free_gib < float(config["storage"]["warn_free_gib"]):
        warnings.append(f"Free disk space is {free_gib:.1f} GiB, below the recommended threshold.")
    storage_probe = _storage_round_trip(run_dir, int(config["storage"]["probe_mib"]))
    if not storage_probe["match"]:
        failures.append("Storage SHA-256 round-trip failed.")
    else:
        passes.append("storage_round_trip")

    config_audit = _audit_configs() if config["repository_checks"]["audit_all_configs"] else {}
    if config_audit and not config_audit["all_ok"]:
        failures.append("At least one YAML config failed to load or lacked experiment_id.")
    elif config_audit:
        passes.append("config_audit")

    repository_checks: dict[str, Any] = {}
    if config["repository_checks"]["run_compileall"]:
        repository_checks["compileall"] = _run_command(
            [sys.executable, "-m", "compileall", "-q", "src", "experiments", "tests"]
        )
        if repository_checks["compileall"]["returncode"] != 0:
            failures.append("Python compileall failed.")
        else:
            passes.append("compileall")
    if config["repository_checks"]["run_pytest"]:
        repository_checks["pytest"] = _run_command([sys.executable, "-m", "pytest", "-q"])
        if repository_checks["pytest"]["returncode"] != 0:
            failures.append("Pytest failed.")
        else:
            passes.append("pytest")

    source_sha, source_count = _source_fingerprint()
    git = _git_snapshot()
    if bool(git.get("dirty")):
        warnings.append("Git worktree is dirty; source fingerprint is recorded for provenance.")
    if not os.getenv("CATENA_GDN2_REPO"):
        warnings.append("CATENA_GDN2_REPO is unset; official recurrent backend is not yet pinned.")
    if not os.getenv("CATENA_KVERASER_REPO"):
        warnings.append("CATENA_KVERASER_REPO is unset; official KVEraser boundary is not ready.")

    status = "PASS" if not failures else "FAIL"
    report = {
        "status": status,
        "counts": {"pass": len(passes), "warn": len(warnings), "fail": len(failures)},
        "passes": passes,
        "warnings": warnings,
        "failures": failures,
        "environment": environment_snapshot(),
        "selected_gpu_indices": selected_gpus,
        "gpu_bf16_checks": gpu_checks,
        "cross_gpu_max_abs": cross_gpu_max_abs,
        "episode_invariants": episode_checks,
        "reference_tied_kda_max_abs": parity_max_abs,
        "free_disk_gib": free_gib,
        "storage_probe": storage_probe,
        "config_audit": config_audit,
        "repository_checks": repository_checks,
        "source_fingerprint": {"sha256": source_sha, "files": source_count},
        "git": git,
        "scientific_evidence": False,
        "meaning": "E00 certifies infrastructure and protocol readiness, not a scientific hypothesis.",
    }
    write_json(run_dir / "protocol_lock.json", config.get("protocol_lock", {}))
    finalize_run(
        experiment_id=EXPERIMENT_ID,
        artifact_root=args.artifact_root,
        run_dir=run_dir,
        report=report,
    )
    print(f"[{EXPERIMENT_ID}] {status}: {run_dir}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
