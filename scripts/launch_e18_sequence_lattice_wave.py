#!/usr/bin/env python3
"""Safely schedule the 25 registered E18a cells on GPUs 0, 1, and 2.

The launcher is intentionally operational only. It never selects from
``latest.json``, never launches E18b, and never retries a completed MAIN cell.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO

from check_e18_status import (
    SOURCE_EXPERIMENT_ID,
    Cell,
    canonical_cells,
    scan_e18_status,
)

GPU_IDS = (0, 1, 2)
LAUNCH_LOCK_PATH = Path("/tmp/catena_e18_sequence_lattice.launch.lock")
SOURCE_CONFIG = "configs/e18a_sequence_control_lattice.yaml"


def deterministic_schedule() -> tuple[dict[str, int | str], ...]:
    """Assign the full registered grid without changing assignments on resume."""

    return tuple(
        {
            "index": index,
            "gpu": GPU_IDS[index % len(GPU_IDS)],
            "seed": cell.seed,
            "variant": cell.variant,
        }
        for index, cell in enumerate(canonical_cells())
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _write_new_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = (
        json.dumps(
            payload,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(encoded)
            handle.flush()
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_python(python_bin: Path, expected_prefix: Path) -> None:
    if not python_bin.is_file() or not os.access(python_bin, os.X_OK):
        raise RuntimeError(f"Python is not executable: {python_bin}")
    actual = subprocess.check_output(
        [
            str(python_bin),
            "-c",
            "import pathlib,sys; print(pathlib.Path(sys.prefix).resolve())",
        ],
        text=True,
    ).strip()
    if Path(actual).resolve() != expected_prefix.resolve():
        raise RuntimeError(
            "refusing non-catena-v6 Python: "
            f"expected={expected_prefix.resolve()} actual={actual}"
        )


def _validate_three_gpus(python_bin: Path) -> None:
    count_text = subprocess.check_output(
        [
            str(python_bin),
            "-c",
            "import torch; print(torch.cuda.device_count())",
        ],
        text=True,
    ).strip()
    try:
        count = int(count_text)
    except ValueError as error:
        raise RuntimeError(f"invalid CUDA device count: {count_text!r}") from error
    if count <= max(GPU_IDS):
        raise RuntimeError(
            f"E18 requires visible GPUs 0-2, but torch reports {count} device(s)"
        )


def _acquire_launch_lock() -> BinaryIO:
    handle = LAUNCH_LOCK_PATH.open("a+b")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError(
            "another E18 launcher/worker set holds the launch lock"
        ) from None
    os.set_inheritable(handle.fileno(), True)
    return handle


def _print_schedule(
    *,
    status: Any,
    artifact_root: Path,
) -> list[dict[str, int | str]]:
    runnable: list[dict[str, int | str]] = []
    print(
        "[E18 SCHEDULE] canonical=25 gpus=0,1,2 "
        "order=seed-major/controller-lattice"
    )
    print(f"[E18 SCHEDULE] artifact_root={artifact_root}")
    print("[E18 SCHEDULE] latest_pointer_used=false aggregate_autorun=false")
    for item in deterministic_schedule():
        cell = Cell(seed=int(item["seed"]), variant=str(item["variant"]))
        state = status.cell_state(cell)
        action = "RUN" if state == "MISSING" else f"SKIP_{state}"
        print(
            f"{int(item['index']):02d} "
            f"gpu={item['gpu']} "
            f"seed={item['seed']} "
            f"variant={str(item['variant']):<18} "
            f"action={action}"
        )
        if state == "MISSING":
            runnable.append(item)
    return runnable


def _target_is_still_missing(
    *,
    repo_root: Path,
    artifact_root: Path,
    cell: Cell,
) -> None:
    status = scan_e18_status(
        repo_root=repo_root,
        artifact_root=artifact_root,
        include_live=False,
    )
    state = status.cell_state(cell)
    if state != "MISSING":
        raise RuntimeError(
            f"worker refuses target {cell.label}: current state={state}"
        )
    target_incomplete = [
        item
        for item in status.incomplete_runs
        if item.seed == cell.seed and item.variant == cell.variant
    ]
    if target_incomplete:
        raise RuntimeError(
            f"worker refuses incomplete target {cell.label}: "
            f"{target_incomplete[0].run_dir}"
        )


def _run_worker(queue_path: Path, held_lock_fd: int) -> int:
    os.fstat(held_lock_fd)
    queue = _read_json_object(queue_path.resolve())
    if queue.get("schema_version") != 1:
        raise RuntimeError("invalid E18 worker queue schema")
    gpu = int(queue["gpu"])
    if gpu not in GPU_IDS:
        raise RuntimeError(f"unregistered E18 GPU: {gpu}")
    repo_root = Path(str(queue["repo_root"])).resolve()
    artifact_root = Path(str(queue["artifact_root"])).resolve()
    python_bin = Path(str(queue["python_bin"])).resolve()
    log_root = queue_path.resolve().parent
    if (
        log_root.parent
        != (artifact_root / "_launcher_logs").resolve()
        or not log_root.name.startswith("e18_sequence_lattice_")
    ):
        raise RuntimeError("E18 worker queue is outside its launcher log namespace")
    jobs = queue.get("jobs")
    if not isinstance(jobs, list):
        raise RuntimeError("E18 worker queue has no job list")

    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "CATENA_ARTIFACT_ROOT": str(artifact_root),
        "PYTHONPATH": os.pathsep.join(
            value
            for value in (
                str(repo_root / "src"),
                str(repo_root),
                os.environ.get("PYTHONPATH", ""),
            )
            if value
        ),
    }
    for raw_job in jobs:
        if not isinstance(raw_job, dict):
            raise RuntimeError("invalid E18 worker job")
        index = int(raw_job["index"])
        cell = Cell(
            seed=int(raw_job["seed"]),
            variant=str(raw_job["variant"]),
        )
        registered = deterministic_schedule()[index]
        if (
            int(registered["gpu"]) != gpu
            or int(registered["seed"]) != cell.seed
            or str(registered["variant"]) != cell.variant
        ):
            raise RuntimeError(f"worker queue changed registered cell {index}")
        _target_is_still_missing(
            repo_root=repo_root,
            artifact_root=artifact_root,
            cell=cell,
        )
        name = f"{index:02d}_{cell.variant}_seed{cell.seed}"
        log_path = log_root / f"{name}.log"
        result_path = log_root / f"{name}.result.json"
        command = [
            str(python_bin),
            "-m",
            f"experiments.{SOURCE_EXPERIMENT_ID}",
            "--config",
            SOURCE_CONFIG,
            "--variant",
            cell.variant,
            "--seed",
            str(cell.seed),
            "--device",
            "cuda:0",
            "--artifact-root",
            str(artifact_root),
        ]
        started_at = datetime.now(UTC).isoformat()
        with log_path.open("xb") as log_handle:
            log_handle.write(
                (
                    f"[E18 WORKER] gpu={gpu} cell={cell.label}\n"
                    f"[E18 WORKER] command={' '.join(command)}\n"
                ).encode()
            )
            log_handle.flush()
            result = subprocess.run(
                command,
                cwd=repo_root,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        completed_at = datetime.now(UTC).isoformat()
        _write_new_json(
            result_path,
            {
                "schema_version": 1,
                "index": index,
                "gpu": gpu,
                "seed": cell.seed,
                "variant": cell.variant,
                "started_at_utc": started_at,
                "completed_at_utc": completed_at,
                "returncode": result.returncode,
                "log_path": str(log_path.resolve()),
            },
        )
        if result.returncode != 0:
            return result.returncode
        post_status = scan_e18_status(
            repo_root=repo_root,
            artifact_root=artifact_root,
            include_live=False,
        )
        if post_status.cell_state(cell) != "COMPLETED":
            raise RuntimeError(
                f"E18 child exited zero without one valid MAIN run: {cell.label}"
            )
    return 0


def _launch_workers(
    *,
    repo_root: Path,
    artifact_root: Path,
    python_bin: Path,
    runnable: list[dict[str, int | str]],
    lock_handle: BinaryIO,
) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    log_root = (
        artifact_root
        / "_launcher_logs"
        / f"e18_sequence_lattice_{timestamp}"
    )
    log_root.mkdir(parents=True, exist_ok=False)
    queues = {
        gpu: [job for job in runnable if int(job["gpu"]) == gpu]
        for gpu in GPU_IDS
    }
    plan = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "experiment_id": SOURCE_EXPERIMENT_ID,
        "repo_root": str(repo_root),
        "artifact_root": str(artifact_root),
        "python_bin": str(python_bin),
        "gpu_ids": list(GPU_IDS),
        "latest_pointer_used": False,
        "aggregate_autorun": False,
        "canonical_schedule": list(deterministic_schedule()),
        "runnable_jobs": runnable,
    }
    _write_new_json(log_root / "launch_plan.json", plan)

    script_path = Path(__file__).resolve()
    for gpu, jobs in queues.items():
        if not jobs:
            continue
        queue_path = log_root / f"gpu{gpu}_queue.json"
        _write_new_json(
            queue_path,
            {
                "schema_version": 1,
                "gpu": gpu,
                "repo_root": str(repo_root),
                "artifact_root": str(artifact_root),
                "python_bin": str(python_bin),
                "jobs": jobs,
            },
        )
        worker_log_path = log_root / f"gpu{gpu}_worker.log"
        worker_log = worker_log_path.open("xb")
        try:
            process = subprocess.Popen(
                [
                    str(python_bin),
                    str(script_path),
                    "--worker-queue",
                    str(queue_path),
                    "--held-lock-fd",
                    str(lock_handle.fileno()),
                ],
                cwd=repo_root,
                env={
                    **os.environ,
                    "PYTHONPATH": os.pathsep.join(
                        value
                        for value in (
                            str(repo_root / "src"),
                            str(repo_root),
                            os.environ.get("PYTHONPATH", ""),
                        )
                        if value
                    ),
                },
                stdout=worker_log,
                stderr=subprocess.STDOUT,
                pass_fds=(lock_handle.fileno(),),
                start_new_session=True,
            )
        finally:
            worker_log.close()
        (log_root / f"gpu{gpu}_worker.pid").write_text(
            f"{process.pid}\n",
            encoding="utf-8",
        )
        print(
            f"[LAUNCH] gpu={gpu} worker_pid={process.pid} "
            f"jobs={len(jobs)}"
        )
    return log_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default="/home/minjun_dev/CATENA")
    parser.add_argument(
        "--artifact-root",
        default=os.getenv(
            "CATENA_ARTIFACT_ROOT",
            "/data/minjun_dev/CATENA/artifacts",
        ),
    )
    parser.add_argument(
        "--python-bin",
        default=os.getenv("CATENA_PYTHON", sys.executable),
    )
    parser.add_argument(
        "--expected-python-prefix",
        default=os.getenv(
            "CATENA_V6_PREFIX",
            "/home/minjun_dev/miniconda3/envs/catena-v6",
        ),
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--execute", action="store_true")
    parser.add_argument("--worker-queue", help=argparse.SUPPRESS)
    parser.add_argument("--held-lock-fd", type=int, help=argparse.SUPPRESS)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.worker_queue is not None:
        if args.held_lock_fd is None:
            raise RuntimeError("E18 worker requires the inherited launch lock")
        return _run_worker(Path(args.worker_queue), args.held_lock_fd)
    if not args.dry_run and not args.execute:
        print(
            "[BLOCKED] Choose --dry-run to inspect or --execute to launch.",
            file=sys.stderr,
        )
        return 2

    repo_root = Path(args.repo_root).resolve()
    artifact_root = Path(args.artifact_root).resolve()
    python_bin = Path(args.python_bin).resolve()
    expected_prefix = Path(args.expected_python_prefix).resolve()
    _validate_python(python_bin, expected_prefix)
    lock_handle = _acquire_launch_lock()
    try:
        status = scan_e18_status(
            repo_root=repo_root,
            artifact_root=artifact_root,
        )
        runnable = _print_schedule(
            status=status,
            artifact_root=artifact_root,
        )
        if status.blockers:
            print(
                "[BLOCKED] E18 launch preflight failed: "
                + ", ".join(status.blockers),
                file=sys.stderr,
            )
            return 3
        if args.dry_run:
            print(
                "[DRY_RUN] No artifact, launcher-log, subprocess, or GPU job "
                "was created."
            )
            return 0
        if not runnable:
            print(
                "[NOOP] All 25 registered MAIN cells already exist. "
                "Run check_e18_status.py --require-aggregate-ready, then "
                "launch E18b separately on CPU."
            )
            return 0
        _validate_three_gpus(python_bin)
        log_root = _launch_workers(
            repo_root=repo_root,
            artifact_root=artifact_root,
            python_bin=python_bin,
            runnable=runnable,
            lock_handle=lock_handle,
        )
        print(f"[DONE] E18 workers launched. Logs: {log_root}")
        print("[SAFETY] E18b aggregate was not launched.")
        return 0
    finally:
        lock_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
