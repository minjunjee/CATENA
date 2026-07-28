#!/usr/bin/env python3
"""Safely inspect and launch registered E21a source seeds in detached tmux.

This is operational tooling only.  It does not alter the locked E21 config,
protocol, metrics, claim gate, or scientific artifacts.  A worker always
starts the existing E21a entry point from the beginning, which creates a new
UTC run directory through ``initialize_run``.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Any, BinaryIO

import yaml

SOURCE_EXPERIMENT_ID = "e21a_structured_sequence_localization_transfer"
SOURCE_MODULE = "experiments.e21_structured_sequence_localization_transfer"
SOURCE_CONFIG_RELATIVE = Path(
    "configs/e21_structured_sequence_localization_transfer.yaml"
)
PROTOCOL_LOCK_RELATIVE = Path(
    "docs/E21_STRUCTURED_SEQUENCE_LOCALIZATION_TRANSFER_LOCK.json"
)
EXPECTED_PROTOCOL_LOCK_SHA256 = (
    "e07139064b6f2cf1ca990f4f595d38c64f295cd7b25ef2fd3a935cbefe498579"
)
REGISTERED_SEEDS = (113, 223, 331, 449, 557)
GPU_IDS = (0, 1, 2, 3)
REGISTERED_VARIANTS = ("base", "separate_address", "state_aware", "full")
REGISTERED_CONDITIONS = (
    "A_oracle_address_oracle_candidate",
    "B_learned_address_oracle_candidate",
    "C_oracle_address_state_read_candidate",
    "D_learned_address_state_read_candidate",
)
REGISTERED_FAMILIES = (
    "magnitude_factorization",
    "value_granularity",
    "address_decoupling",
    "state_conditioning",
)
REGISTERED_UPDATES = (1, 4, 8)
REGISTERED_GAPS = (0, 128, 512, 2048)
EXPECTED_SOURCE_ROWS = 768
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = Path(
    "/home/minjun_dev/miniconda3/envs/catena-v6/bin/python"
)
DEFAULT_PYTHON_PREFIX = Path(
    "/home/minjun_dev/miniconda3/envs/catena-v6"
)
LAUNCH_LOCK_PATH = Path("/tmp/catena_e21_source_tmux.launch.lock")
RUN_ID_PATTERN = re.compile(r"^\d{8}T\d{6}\.\d{6}Z$")
LAUNCH_DIR_PATTERN = re.compile(
    r"^e21_source_tmux_(\d{8}T\d{6}\.\d{6}Z)_seed(\d+)$"
)
CHECKPOINT_SEED_PATTERN = re.compile(r"^seed(\d+)_.*\.pt$")


@dataclass(frozen=True)
class CompletedRun:
    seed: int
    run_dir: str


@dataclass(frozen=True)
class IncompleteRun:
    seed: int | None
    run_dir: str
    reason: str


@dataclass(frozen=True)
class ActiveProcess:
    pid: int
    seed: int
    gpu: int | None
    command: tuple[str, ...]


@dataclass(frozen=True)
class ActiveReservation:
    seed: int
    gpu: int
    session_name: str
    launch_dir: str


@dataclass(frozen=True)
class StatusSnapshot:
    artifact_root: str
    completed_runs: tuple[CompletedRun, ...]
    incomplete_runs: tuple[IncompleteRun, ...]
    active_processes: tuple[ActiveProcess, ...]
    active_reservations: tuple[ActiveReservation, ...]
    invalid_launcher_records: tuple[str, ...]

    def completed_for_seed(self, seed: int) -> tuple[CompletedRun, ...]:
        return tuple(item for item in self.completed_runs if item.seed == seed)

    def incomplete_for_seed(self, seed: int) -> tuple[IncompleteRun, ...]:
        return tuple(item for item in self.incomplete_runs if item.seed == seed)

    def processes_for_seed(self, seed: int) -> tuple[ActiveProcess, ...]:
        return tuple(item for item in self.active_processes if item.seed == seed)

    def reservations_for_seed(
        self, seed: int
    ) -> tuple[ActiveReservation, ...]:
        return tuple(
            item for item in self.active_reservations if item.seed == seed
        )

    def seed_state(self, seed: int) -> str:
        completed = self.completed_for_seed(seed)
        active = self.processes_for_seed(seed) or self.reservations_for_seed(seed)
        if completed and active:
            return "CONFLICT_COMPLETED_ACTIVE"
        if len(completed) > 1:
            return "DUPLICATE_COMPLETED"
        if completed:
            return "COMPLETED"
        if active:
            return "ACTIVE"
        if self.incomplete_for_seed(seed):
            return "INCOMPLETE"
        return "MISSING"

    def active_gpus(self) -> set[int]:
        return {
            gpu
            for gpu in (
                *(item.gpu for item in self.active_processes),
                *(item.gpu for item in self.active_reservations),
            )
            if gpu is not None
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "experiment_id": SOURCE_EXPERIMENT_ID,
            "artifact_root": self.artifact_root,
            "registered_seeds": list(REGISTERED_SEEDS),
            "gpu_ids": list(GPU_IDS),
            "seed_status": [
                {
                    "seed": seed,
                    "state": self.seed_state(seed),
                    "completed_runs": [
                        item.run_dir for item in self.completed_for_seed(seed)
                    ],
                    "incomplete_runs": [
                        item.run_dir for item in self.incomplete_for_seed(seed)
                    ],
                    "active_pids": [
                        item.pid for item in self.processes_for_seed(seed)
                    ],
                    "active_sessions": [
                        item.session_name
                        for item in self.reservations_for_seed(seed)
                    ],
                }
                for seed in REGISTERED_SEEDS
            ],
            "unknown_seed_incomplete_runs": [
                item.run_dir
                for item in self.incomplete_runs
                if item.seed is None
            ],
            "active_processes": [
                {
                    **asdict(item),
                    "command": list(item.command),
                }
                for item in self.active_processes
            ],
            "active_reservations": [
                asdict(item) for item in self.active_reservations
            ],
            "invalid_launcher_records": list(
                self.invalid_launcher_records
            ),
        }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _write_new_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = (
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(encoded)
            handle.flush()
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def validate_registered_contract(
    *, repo_root: Path, config_path: Path
) -> dict[str, str]:
    repo_root = repo_root.resolve()
    expected_config = (repo_root / SOURCE_CONFIG_RELATIVE).resolve()
    config_path = config_path.resolve()
    if config_path != expected_config:
        raise RuntimeError(
            f"refusing noncanonical E21 config: {config_path}"
        )
    lock_path = (repo_root / PROTOCOL_LOCK_RELATIVE).resolve()
    for path in (config_path, lock_path):
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"missing or unsafe E21 contract file: {path}")
    if _sha256(lock_path) != EXPECTED_PROTOCOL_LOCK_SHA256:
        raise RuntimeError("E21 protocol lock SHA-256 changed")
    lock = _read_json_object(lock_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise RuntimeError("E21 config is not a mapping")
    registered = lock.get("registered_design", {}).get("main_seeds")
    if (
        lock.get("schema_version") != 1
        or lock.get("experiment_family") != "E21"
        or lock.get("source_experiment_id") != SOURCE_EXPERIMENT_ID
        or tuple(int(value) for value in registered or []) != REGISTERED_SEEDS
        or tuple(int(value) for value in config.get("seeds", []))
        != REGISTERED_SEEDS
        or config.get("source_experiment_id") != SOURCE_EXPERIMENT_ID
    ):
        raise RuntimeError("E21 registered seed/source identity changed")
    expected_config_hash = lock.get("files", {}).get(
        str(SOURCE_CONFIG_RELATIVE)
    )
    if expected_config_hash != _sha256(config_path):
        raise RuntimeError("E21 source config no longer matches its lock")
    return {
        "config_path": str(config_path),
        "config_sha256": expected_config_hash,
        "lock_path": str(lock_path),
        "lock_sha256": EXPECTED_PROTOCOL_LOCK_SHA256,
    }


def _infer_incomplete_seed(run_dir: Path) -> int | None:
    report_path = run_dir / "report.json"
    if report_path.is_file():
        try:
            raw_seed = _read_json_object(report_path).get("seed")
            if not isinstance(raw_seed, (int, str)):
                raise TypeError("seed is not integer-like")
            value = int(raw_seed)
        except (TypeError, ValueError, json.JSONDecodeError):
            value = -1
        if value in REGISTERED_SEEDS:
            return value
    observed: set[int] = set()
    checkpoint_root = run_dir / "checkpoints"
    if checkpoint_root.is_dir():
        for path in checkpoint_root.iterdir():
            match = CHECKPOINT_SEED_PATTERN.fullmatch(path.name)
            if match is not None:
                observed.add(int(match.group(1)))
    return next(iter(observed)) if len(observed) == 1 else None


def _completed_seed(run_dir: Path) -> int | None:
    report_path = run_dir / "report.json"
    manifest_path = run_dir / "run_manifest.json"
    metrics_path = run_dir / "structured_sequence_transfer_metrics.jsonl"
    summary_path = run_dir / "RESULTS_SUMMARY_KO.md"
    required = (report_path, manifest_path, metrics_path, summary_path)
    if any(not path.is_file() or path.is_symlink() for path in required):
        return None
    try:
        report = _read_json_object(report_path)
        manifest = _read_json_object(manifest_path)
        raw_seed = report.get("seed")
        if not isinstance(raw_seed, (int, str)):
            raise TypeError("seed is not integer-like")
        seed = int(raw_seed)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if seed not in REGISTERED_SEEDS:
        return None
    artifacts = report.get("artifacts")
    protocol = report.get("protocol")
    if not isinstance(artifacts, dict) or not isinstance(protocol, dict):
        return None
    summary = artifacts.get("results_summary_ko")
    checkpoint_hashes = artifacts.get("checkpoint_hashes")
    if not isinstance(summary, dict) or not isinstance(
        checkpoint_hashes, dict
    ):
        return None
    summary_lines = summary_path.read_text(encoding="utf-8").splitlines()
    try:
        summary_line_count = int(summary.get("line_count", -1))
    except (TypeError, ValueError):
        return None
    if (
        report.get("status") != "PASS"
        or report.get("run_mode") != "MAIN"
        or manifest.get("schema_version") != 2
        or manifest.get("experiment_id") != SOURCE_EXPERIMENT_ID
        or manifest.get("run_id") != run_dir.name
        or manifest.get("run_mode") != "MAIN"
        or not isinstance(manifest.get("completed_at_utc"), str)
        or manifest.get("report_sha256") != _sha256(report_path)
        or manifest.get("config_file_sha256")
        != protocol.get("source_config_sha256")
        or protocol.get("lock_sha256") != EXPECTED_PROTOCOL_LOCK_SHA256
        or artifacts.get("metrics_sha256") != _sha256(metrics_path)
        or summary.get("sha256") != _sha256(summary_path)
        or summary_line_count != len(summary_lines)
        or len(summary_lines) > 55
        or set(checkpoint_hashes) != set(REGISTERED_VARIANTS)
    ):
        return None

    checkpoint_paths: dict[str, Path] = {}
    for variant in REGISTERED_VARIANTS:
        checkpoint = (
            run_dir / "checkpoints" / f"seed{seed}_{variant}.pt"
        ).resolve()
        try:
            checkpoint.relative_to((run_dir / "checkpoints").resolve())
        except ValueError:
            return None
        expected_hash = checkpoint_hashes.get(variant)
        if (
            not isinstance(expected_hash, str)
            or not checkpoint.is_file()
            or checkpoint.is_symlink()
            or _sha256(checkpoint) != expected_hash
        ):
            return None
        checkpoint_paths[variant] = checkpoint

    expected_grid = set(
        product(
            REGISTERED_VARIANTS,
            REGISTERED_CONDITIONS,
            REGISTERED_FAMILIES,
            REGISTERED_UPDATES,
            REGISTERED_GAPS,
        )
    )
    observed_grid: set[tuple[str, str, str, int, int]] = set()
    initialization_hashes: set[str] = set()
    parameter_counts: set[int] = set()
    digest_groups: dict[tuple[str, int, int], set[str]] = {}
    try:
        with metrics_path.open("r", encoding="utf-8") as handle:
            rows = [
                json.loads(
                    line,
                    parse_constant=_reject_json_constant,
                )
                for line in handle
                if line.strip()
            ]
        if len(rows) != EXPECTED_SOURCE_ROWS:
            return None
        for row in rows:
            if not isinstance(row, dict) or int(row.get("seed", -1)) != seed:
                return None
            key = (
                str(row.get("variant")),
                str(row.get("condition")),
                str(row.get("demand_family")),
                int(row.get("updates", -1)),
                int(row.get("gap_events", -1)),
            )
            if key not in expected_grid or key in observed_grid:
                return None
            observed_grid.add(key)
            variant = key[0]
            if (
                Path(str(row.get("checkpoint", ""))).resolve()
                != checkpoint_paths[variant]
                or row.get("checkpoint_sha256")
                != checkpoint_hashes[variant]
            ):
                return None
            initialization_hashes.add(
                str(row.get("initialization_sha256", ""))
            )
            parameter_counts.add(int(row.get("parameter_count", -1)))
            digest_key = (key[2], key[3], key[4])
            digest_groups.setdefault(digest_key, set()).add(
                str(row.get("base_transaction_digest", ""))
            )
    except (
        json.JSONDecodeError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ):
        return None
    if (
        observed_grid != expected_grid
        or len(initialization_hashes) != 1
        or "" in initialization_hashes
        or len(parameter_counts) != 1
        or min(parameter_counts) <= 0
        or not digest_groups
        or any(
            len(values) != 1 or "" in values
            for values in digest_groups.values()
        )
    ):
        return None
    return seed


def scan_artifact_runs(
    artifact_root: Path,
) -> tuple[tuple[CompletedRun, ...], tuple[IncompleteRun, ...]]:
    source_root = artifact_root.resolve() / SOURCE_EXPERIMENT_ID
    completed: list[CompletedRun] = []
    incomplete: list[IncompleteRun] = []
    if not source_root.is_dir():
        return (), ()
    for run_dir in sorted(source_root.iterdir()):
        if not run_dir.is_dir() or not RUN_ID_PATTERN.fullmatch(run_dir.name):
            continue
        seed = _completed_seed(run_dir)
        if seed is not None:
            completed.append(
                CompletedRun(seed=seed, run_dir=str(run_dir.resolve()))
            )
            continue
        inferred = _infer_incomplete_seed(run_dir)
        reason = (
            "missing_or_invalid_completed_contract"
            if (run_dir / "report.json").exists()
            else "report_missing"
        )
        incomplete.append(
            IncompleteRun(
                seed=inferred,
                run_dir=str(run_dir.resolve()),
                reason=reason,
            )
        )
    return tuple(completed), tuple(incomplete)


def _option(tokens: Sequence[str], name: str) -> str | None:
    try:
        index = tokens.index(name)
    except ValueError:
        return None
    return tokens[index + 1] if index + 1 < len(tokens) else None


def parse_source_process(
    *,
    pid: int,
    command: Sequence[str],
    environ: dict[str, str],
    cwd: Path,
    artifact_root: Path,
) -> ActiveProcess | None:
    tokens = tuple(command)
    module = _option(tokens, "-m")
    entrypoint_match = module == SOURCE_MODULE or any(
        token.endswith(
            "/experiments/e21_structured_sequence_localization_transfer.py"
        )
        for token in tokens
    )
    if (
        not entrypoint_match
        or "--aggregate" in tokens
        or "--dry-run" in tokens
    ):
        return None
    seed_text = _option(tokens, "--seed")
    root_text = _option(tokens, "--artifact-root") or environ.get(
        "CATENA_ARTIFACT_ROOT"
    )
    if seed_text is None or root_text is None:
        return None
    try:
        seed = int(seed_text)
    except ValueError:
        return None
    if seed not in REGISTERED_SEEDS:
        return None
    observed_root = Path(root_text)
    if not observed_root.is_absolute():
        observed_root = cwd / observed_root
    if observed_root.resolve() != artifact_root.resolve():
        return None
    device = _option(tokens, "--device")
    gpu: int | None = None
    if device and device.startswith("cuda:"):
        try:
            logical_gpu = int(device.split(":", 1)[1])
        except ValueError:
            logical_gpu = -1
        visible = [
            value.strip()
            for value in environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            if value.strip()
        ]
        try:
            gpu = (
                int(visible[logical_gpu])
                if visible and 0 <= logical_gpu < len(visible)
                else logical_gpu
            )
        except ValueError:
            gpu = None
    return ActiveProcess(pid=pid, seed=seed, gpu=gpu, command=tokens)


def scan_active_processes(
    artifact_root: Path, *, proc_root: Path = Path("/proc")
) -> tuple[ActiveProcess, ...]:
    records: list[ActiveProcess] = []
    try:
        entries: Iterable[Path] = proc_root.iterdir()
    except OSError:
        return ()
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            raw_command = (entry / "cmdline").read_bytes()
            command = tuple(
                value.decode(errors="surrogateescape")
                for value in raw_command.split(b"\0")
                if value
            )
            raw_environment = (entry / "environ").read_bytes()
            environ = {}
            for value in raw_environment.split(b"\0"):
                if b"=" not in value:
                    continue
                key, content = value.split(b"=", 1)
                environ[key.decode(errors="ignore")] = content.decode(
                    errors="surrogateescape"
                )
            cwd = (entry / "cwd").resolve()
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
            continue
        record = parse_source_process(
            pid=int(entry.name),
            command=command,
            environ=environ,
            cwd=cwd,
            artifact_root=artifact_root,
        )
        if record is not None:
            records.append(record)
    return tuple(sorted(records, key=lambda item: item.pid))


def _tmux_sessions(tmux_bin: str) -> set[str]:
    result = subprocess.run(
        [tmux_bin, "list-sessions", "-F", "#{session_name}"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return set()
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def scan_active_reservations(
    artifact_root: Path, *, tmux_bin: str = "tmux"
) -> tuple[tuple[ActiveReservation, ...], tuple[str, ...]]:
    launcher_root = artifact_root.resolve() / "_launcher_logs"
    if not launcher_root.is_dir():
        return (), ()
    sessions = _tmux_sessions(tmux_bin)
    reservations: list[ActiveReservation] = []
    invalid: list[str] = []
    for launch_dir in sorted(launcher_root.glob("e21_source_tmux_*")):
        if (
            not launch_dir.is_dir()
            or LAUNCH_DIR_PATTERN.fullmatch(launch_dir.name) is None
        ):
            continue
        record_path = launch_dir / "launch_record.json"
        result_path = launch_dir / "result.json"
        if result_path.exists():
            continue
        try:
            record = _read_json_object(record_path)
            seed = int(record["seed"])
            gpu = int(record["gpu"])
            session = str(record["session_name"])
            if (
                record.get("schema_version") != 1
                or record.get("experiment_id") != SOURCE_EXPERIMENT_ID
                or seed not in REGISTERED_SEEDS
                or gpu not in GPU_IDS
                or Path(str(record["artifact_root"])).resolve()
                != artifact_root.resolve()
            ):
                raise ValueError("record contract mismatch")
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            invalid.append(str(record_path.resolve()))
            continue
        if session in sessions:
            reservations.append(
                ActiveReservation(
                    seed=seed,
                    gpu=gpu,
                    session_name=session,
                    launch_dir=str(launch_dir.resolve()),
                )
            )
    return tuple(reservations), tuple(invalid)


def scan_status(
    *, artifact_root: Path, tmux_bin: str = "tmux"
) -> StatusSnapshot:
    completed, incomplete = scan_artifact_runs(artifact_root)
    processes = scan_active_processes(artifact_root)
    reservations, invalid = scan_active_reservations(
        artifact_root, tmux_bin=tmux_bin
    )
    return StatusSnapshot(
        artifact_root=str(artifact_root.resolve()),
        completed_runs=completed,
        incomplete_runs=incomplete,
        active_processes=processes,
        active_reservations=reservations,
        invalid_launcher_records=invalid,
    )


def _print_status(snapshot: StatusSnapshot) -> None:
    print(
        "[E21 SOURCE STATUS] "
        f"artifact_root={snapshot.artifact_root} "
        "registered_seeds=113,223,331,449,557 gpus=0,1,2,3"
    )
    for seed in REGISTERED_SEEDS:
        completed = snapshot.completed_for_seed(seed)
        incomplete = snapshot.incomplete_for_seed(seed)
        processes = snapshot.processes_for_seed(seed)
        reservations = snapshot.reservations_for_seed(seed)
        print(
            f"seed={seed} state={snapshot.seed_state(seed)} "
            f"completed={len(completed)} incomplete={len(incomplete)} "
            f"pids={','.join(str(item.pid) for item in processes) or '-'} "
            "sessions="
            f"{','.join(item.session_name for item in reservations) or '-'}"
        )
    unknown = [
        item for item in snapshot.incomplete_runs if item.seed is None
    ]
    print(
        "[E21 SOURCE STATUS] "
        f"unknown_seed_incomplete={len(unknown)} "
        f"invalid_launcher_records={len(snapshot.invalid_launcher_records)}"
    )


def _acquire_launch_lock(*, blocking: bool = False) -> BinaryIO:
    handle = LAUNCH_LOCK_PATH.open("a+b")
    try:
        operation = fcntl.LOCK_EX
        if not blocking:
            operation |= fcntl.LOCK_NB
        fcntl.flock(handle.fileno(), operation)
    except BlockingIOError:
        handle.close()
        raise RuntimeError("another E21 source launcher holds the lock") from None
    return handle


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


def _assert_launchable(
    snapshot: StatusSnapshot,
    *,
    seed: int,
    gpu: int,
    allow_incomplete_retry: bool,
    ignore_session: str | None = None,
) -> None:
    completed = snapshot.completed_for_seed(seed)
    processes = snapshot.processes_for_seed(seed)
    reservations = tuple(
        item
        for item in snapshot.reservations_for_seed(seed)
        if item.session_name != ignore_session
    )
    if completed:
        raise RuntimeError(
            f"seed {seed} already has {len(completed)} completed MAIN run(s)"
        )
    if processes or reservations:
        raise RuntimeError(
            f"seed {seed} already has an active source run/reservation"
        )
    busy_processes = [
        item
        for item in snapshot.active_processes
        if item.gpu == gpu and item.seed != seed
    ]
    busy_reservations = [
        item
        for item in snapshot.active_reservations
        if (
            item.gpu == gpu
            and item.seed != seed
            and item.session_name != ignore_session
        )
    ]
    if busy_processes or busy_reservations:
        raise RuntimeError(f"physical GPU {gpu} already has an active E21 source")
    incomplete = snapshot.incomplete_for_seed(seed)
    if incomplete and not allow_incomplete_retry:
        raise RuntimeError(
            f"seed {seed} has {len(incomplete)} preserved incomplete run(s); "
            "pass --allow-incomplete-retry to start a fresh UTC run"
        )


def build_source_command(
    *,
    python_bin: Path,
    config_path: Path,
    seed: int,
    artifact_root: Path,
) -> tuple[str, ...]:
    return (
        str(python_bin.resolve()),
        "-m",
        SOURCE_MODULE,
        "--config",
        str(config_path.resolve()),
        "--seed",
        str(seed),
        "--device",
        "cuda:0",
        "--artifact-root",
        str(artifact_root.resolve()),
    )


def _launch_dir(
    *, artifact_root: Path, seed: int, timestamp: str
) -> Path:
    return (
        artifact_root.resolve()
        / "_launcher_logs"
        / f"e21_source_tmux_{timestamp}_seed{seed}"
    )


def _create_launch_record(
    *,
    repo_root: Path,
    artifact_root: Path,
    config_path: Path,
    python_bin: Path,
    seed: int,
    gpu: int,
    allow_incomplete_retry: bool,
    contract: dict[str, str],
) -> tuple[Path, dict[str, Any]]:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    launch_dir = _launch_dir(
        artifact_root=artifact_root, seed=seed, timestamp=timestamp
    )
    launch_dir.mkdir(parents=True, exist_ok=False)
    session_name = f"catena_e21_seed{seed}_{timestamp.replace('.', '_')}"
    command = build_source_command(
        python_bin=python_bin,
        config_path=config_path,
        seed=seed,
        artifact_root=artifact_root,
    )
    record: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "experiment_id": SOURCE_EXPERIMENT_ID,
        "operational_only": True,
        "repo_root": str(repo_root.resolve()),
        "artifact_root": str(artifact_root.resolve()),
        "config_path": str(config_path.resolve()),
        "config_sha256": contract["config_sha256"],
        "protocol_lock_path": contract["lock_path"],
        "protocol_lock_sha256": contract["lock_sha256"],
        "python_bin": str(python_bin.resolve()),
        "seed": seed,
        "gpu": gpu,
        "session_name": session_name,
        "allow_incomplete_retry": allow_incomplete_retry,
        "source_command": list(command),
        "resume_from_partial_checkpoint": False,
        "aggregate_autorun": False,
        "scientific_artifact_mutation": False,
    }
    record_path = launch_dir / "launch_record.json"
    _write_new_json(record_path, record)
    return record_path, record


def _worker(record_path: Path, *, tmux_bin: str) -> int:
    record_path = record_path.resolve()
    record = _read_json_object(record_path)
    launch_dir = record_path.parent
    match = LAUNCH_DIR_PATTERN.fullmatch(launch_dir.name)
    if (
        match is None
        or record.get("schema_version") != 1
        or record.get("experiment_id") != SOURCE_EXPERIMENT_ID
    ):
        raise RuntimeError("invalid E21 tmux launch record")
    seed = int(record["seed"])
    gpu = int(record["gpu"])
    if seed not in REGISTERED_SEEDS or gpu not in GPU_IDS:
        raise RuntimeError("launch record contains unregistered seed/GPU")
    repo_root = Path(str(record["repo_root"])).resolve()
    artifact_root = Path(str(record["artifact_root"])).resolve()
    config_path = Path(str(record["config_path"])).resolve()
    python_bin = Path(str(record["python_bin"])).resolve()
    session_name = str(record["session_name"])
    expected_command = build_source_command(
        python_bin=python_bin,
        config_path=config_path,
        seed=seed,
        artifact_root=artifact_root,
    )
    if tuple(str(value) for value in record["source_command"]) != expected_command:
        raise RuntimeError("E21 source command changed after launch planning")
    contract = validate_registered_contract(
        repo_root=repo_root, config_path=config_path
    )
    if (
        record.get("config_sha256") != contract["config_sha256"]
        or record.get("protocol_lock_sha256") != contract["lock_sha256"]
    ):
        raise RuntimeError("E21 launch record provenance changed")
    _validate_python(python_bin, DEFAULT_PYTHON_PREFIX)

    # The launcher parent holds this lock until ``tmux new-session`` returns.
    # Blocking here is intentional: it closes the parent/worker hand-off race.
    lock_handle = _acquire_launch_lock(blocking=True)
    try:
        snapshot = scan_status(
            artifact_root=artifact_root, tmux_bin=tmux_bin
        )
        _assert_launchable(
            snapshot,
            seed=seed,
            gpu=gpu,
            allow_incomplete_retry=bool(record["allow_incomplete_retry"]),
            ignore_session=session_name,
        )
        source_root = artifact_root / SOURCE_EXPERIMENT_ID
        before = {
            path.resolve()
            for path in source_root.iterdir()
            if source_root.is_dir() and path.is_dir()
        } if source_root.is_dir() else set()
    finally:
        lock_handle.close()

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
    log_path = launch_dir / "worker.log"
    started_at = datetime.now(UTC).isoformat()
    with log_path.open("xb") as log_handle:
        log_handle.write(
            (
                f"[E21 TMUX WORKER] session={session_name} "
                f"physical_gpu={gpu} seed={seed}\n"
                f"[E21 TMUX WORKER] command={' '.join(expected_command)}\n"
            ).encode()
        )
        log_handle.flush()
        result = subprocess.run(
            expected_command,
            cwd=repo_root,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    after = {
        path.resolve()
        for path in source_root.iterdir()
        if source_root.is_dir() and path.is_dir()
    } if source_root.is_dir() else set()
    _write_new_json(
        launch_dir / "result.json",
        {
            "schema_version": 1,
            "experiment_id": SOURCE_EXPERIMENT_ID,
            "seed": seed,
            "gpu": gpu,
            "session_name": session_name,
            "started_at_utc": started_at,
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "returncode": result.returncode,
            "new_source_run_dirs": [
                str(path) for path in sorted(after - before)
            ],
            "worker_log_path": str(log_path.resolve()),
            "worker_log_sha256": _sha256(log_path),
        },
    )
    return int(result.returncode)


def _status_command(args: argparse.Namespace) -> int:
    snapshot = scan_status(
        artifact_root=Path(args.artifact_root), tmux_bin=args.tmux_bin
    )
    if args.json:
        print(
            json.dumps(
                snapshot.to_json(),
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
        )
    else:
        _print_status(snapshot)
    return 0


def _launch_command(args: argparse.Namespace) -> int:
    seed = int(args.seed)
    gpu = int(args.gpu)
    if seed not in REGISTERED_SEEDS:
        raise RuntimeError(
            f"unregistered E21 seed {seed}; expected {REGISTERED_SEEDS}"
        )
    if gpu not in GPU_IDS:
        raise RuntimeError(f"unregistered physical GPU {gpu}; expected {GPU_IDS}")
    repo_root = Path(args.repo_root).resolve()
    artifact_root = Path(args.artifact_root).resolve()
    config_path = Path(args.config).resolve()
    python_bin = Path(args.python_bin).resolve()
    contract = validate_registered_contract(
        repo_root=repo_root, config_path=config_path
    )
    snapshot = scan_status(
        artifact_root=artifact_root, tmux_bin=args.tmux_bin
    )
    _assert_launchable(
        snapshot,
        seed=seed,
        gpu=gpu,
        allow_incomplete_retry=args.allow_incomplete_retry,
    )
    command = build_source_command(
        python_bin=python_bin,
        config_path=config_path,
        seed=seed,
        artifact_root=artifact_root,
    )
    print(
        f"[E21 LAUNCH PLAN] seed={seed} physical_gpu={gpu} "
        f"detached_tmux=true execute={args.execute}"
    )
    print(f"[E21 LAUNCH PLAN] command={' '.join(command)}")
    if not args.execute:
        print("[E21 LAUNCH PLAN] no files or processes created")
        return 0

    _validate_python(python_bin, Path(args.expected_python_prefix))
    if not Path(args.tmux_bin).is_file() and not shutil.which(args.tmux_bin):
        raise RuntimeError(f"tmux executable not found: {args.tmux_bin}")
    lock_handle = _acquire_launch_lock()
    try:
        refreshed = scan_status(
            artifact_root=artifact_root, tmux_bin=args.tmux_bin
        )
        _assert_launchable(
            refreshed,
            seed=seed,
            gpu=gpu,
            allow_incomplete_retry=args.allow_incomplete_retry,
        )
        record_path, record = _create_launch_record(
            repo_root=repo_root,
            artifact_root=artifact_root,
            config_path=config_path,
            python_bin=python_bin,
            seed=seed,
            gpu=gpu,
            allow_incomplete_retry=args.allow_incomplete_retry,
            contract=contract,
        )
        subprocess.run(
            [
                args.tmux_bin,
                "new-session",
                "-d",
                "-s",
                str(record["session_name"]),
                "--",
                str(python_bin),
                str(Path(__file__).resolve()),
                "--repo-root",
                str(repo_root),
                "--artifact-root",
                str(artifact_root),
                "--config",
                str(config_path),
                "--python-bin",
                str(python_bin),
                "--tmux-bin",
                args.tmux_bin,
                "_worker",
                "--launch-record",
                str(record_path),
            ],
            check=True,
            cwd=repo_root,
        )
    finally:
        lock_handle.close()
    print(
        f"[E21 LAUNCHED] session={record['session_name']} "
        f"launch_dir={record_path.parent}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detached tmux operations for registered E21a source seeds"
    )
    parser.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT))
    parser.add_argument(
        "--artifact-root",
        default=os.getenv(
            "CATENA_ARTIFACT_ROOT",
            "/data/minjun_dev/CATENA/artifacts",
        ),
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_REPO_ROOT / SOURCE_CONFIG_RELATIVE),
    )
    parser.add_argument("--python-bin", default=str(DEFAULT_PYTHON))
    parser.add_argument(
        "--expected-python-prefix", default=str(DEFAULT_PYTHON_PREFIX)
    )
    parser.add_argument("--tmux-bin", default="tmux")
    subparsers = parser.add_subparsers(dest="command", required=True)

    status = subparsers.add_parser("status")
    status.add_argument("--json", action="store_true")

    launch = subparsers.add_parser("launch")
    launch.add_argument("--seed", type=int, required=True)
    launch.add_argument("--gpu", type=int, required=True)
    launch.add_argument("--allow-incomplete-retry", action="store_true")
    launch.add_argument(
        "--execute",
        action="store_true",
        help="Actually create a detached tmux session; default is plan-only.",
    )

    worker = subparsers.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("--launch-record", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "status":
        code = _status_command(args)
    elif args.command == "launch":
        code = _launch_command(args)
    else:
        code = _worker(
            Path(args.launch_record), tmux_bin=str(args.tmux_bin)
        )
    raise SystemExit(code)


if __name__ == "__main__":
    main()
