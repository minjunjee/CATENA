#!/usr/bin/env python3
"""Read-only operational status and provenance checks for E18.

This script deliberately scans every run directory in the E18a namespace.
It never reads ``latest.json`` when selecting scientific inputs.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any

import yaml

SOURCE_EXPERIMENT_ID = "e18a_sequence_control_lattice"
AGGREGATE_EXPERIMENT_ID = "e18b_sequence_control_lattice_aggregate"
LOCK_RELATIVE_PATH = Path("docs/E18_SEQUENCE_CONTROL_LATTICE_LOCK.json")
SOURCE_CONFIG_RELATIVE_PATH = Path("configs/e18a_sequence_control_lattice.yaml")
AGGREGATE_CONFIG_RELATIVE_PATH = Path(
    "configs/e18b_sequence_control_lattice_aggregate.yaml"
)
CANONICAL_VARIANTS = (
    "tied_scalar",
    "dual_scalar",
    "diagonal_value",
    "separate_address",
    "state_aware",
)
CANONICAL_SEEDS = (101, 211, 307, 401, 503)


@dataclass(frozen=True, order=True)
class Cell:
    seed: int
    variant: str

    @property
    def label(self) -> str:
        return f"{self.variant}/seed{self.seed}"


@dataclass(frozen=True)
class IncompleteRun:
    run_dir: str
    reason: str
    seed: int | None
    variant: str | None


@dataclass(frozen=True)
class LiveRun:
    pid: int
    command: str
    seed: int | None
    variant: str | None


@dataclass
class E18Status:
    repo_root: str
    artifact_root: str
    protocol_lock_sha256: str
    completed_runs: dict[Cell, list[str]]
    incomplete_runs: list[IncompleteRun]
    ignored_dry_runs: list[str]
    live_runs: list[LiveRun]
    aggregate_provenance_error: str | None = None
    aggregate_source_rows: int = 0
    aggregate_source_runs: int = 0

    @property
    def canonical_cells(self) -> tuple[Cell, ...]:
        return canonical_cells()

    @property
    def duplicates(self) -> dict[Cell, list[str]]:
        return {
            cell: paths
            for cell, paths in self.completed_runs.items()
            if len(paths) > 1
        }

    @property
    def completed_cells(self) -> set[Cell]:
        return {
            cell
            for cell, paths in self.completed_runs.items()
            if len(paths) == 1
        }

    @property
    def missing_cells(self) -> set[Cell]:
        return set(self.canonical_cells) - self.completed_cells

    @property
    def blockers(self) -> list[str]:
        blockers: list[str] = []
        if self.incomplete_runs:
            blockers.append(f"incomplete_main_runs={len(self.incomplete_runs)}")
        if self.duplicates:
            blockers.append(f"duplicate_completed_cells={len(self.duplicates)}")
        if self.live_runs:
            blockers.append(f"live_main_processes={len(self.live_runs)}")
        return blockers

    @property
    def launch_safe(self) -> bool:
        return not self.blockers and bool(self.missing_cells)

    @property
    def aggregate_ready(self) -> bool:
        return bool(
            not self.blockers
            and not self.missing_cells
            and self.aggregate_source_runs == len(self.canonical_cells)
            and self.aggregate_source_rows == 1200
            and self.aggregate_provenance_error is None
        )

    def cell_state(self, cell: Cell) -> str:
        if cell in self.duplicates:
            return "DUPLICATE"
        if cell in self.completed_cells:
            return "COMPLETED"
        for run in self.incomplete_runs:
            if run.seed == cell.seed and run.variant == cell.variant:
                return "INCOMPLETE"
        return "MISSING"

    def as_json(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "experiment_id": SOURCE_EXPERIMENT_ID,
            "repo_root": self.repo_root,
            "artifact_root": self.artifact_root,
            "protocol_lock_sha256": self.protocol_lock_sha256,
            "latest_pointer_used": False,
            "counts": {
                "registered_cells": len(self.canonical_cells),
                "completed_cells": len(self.completed_cells),
                "missing_cells": len(self.missing_cells),
                "duplicate_completed_cells": len(self.duplicates),
                "incomplete_main_runs": len(self.incomplete_runs),
                "ignored_dry_runs": len(self.ignored_dry_runs),
                "live_main_processes": len(self.live_runs),
            },
            "cells": [
                {
                    "index": index,
                    "seed": cell.seed,
                    "variant": cell.variant,
                    "status": self.cell_state(cell),
                    "run_dirs": self.completed_runs.get(cell, []),
                }
                for index, cell in enumerate(self.canonical_cells)
            ],
            "incomplete_runs": [asdict(run) for run in self.incomplete_runs],
            "duplicate_runs": [
                {
                    "seed": cell.seed,
                    "variant": cell.variant,
                    "run_dirs": paths,
                }
                for cell, paths in sorted(self.duplicates.items())
            ],
            "ignored_dry_runs": self.ignored_dry_runs,
            "live_runs": [asdict(run) for run in self.live_runs],
            "blockers": self.blockers,
            "launch_safe": self.launch_safe,
            "aggregate": {
                "ready": self.aggregate_ready,
                "source_runs": self.aggregate_source_runs,
                "source_rows": self.aggregate_source_rows,
                "provenance_error": self.aggregate_provenance_error,
                "must_run_separately_on_cpu": True,
            },
        }


def canonical_cells() -> tuple[Cell, ...]:
    """Return the registered seed-major, lattice-order E18 schedule."""

    return tuple(
        Cell(seed=seed, variant=variant)
        for seed, variant in product(CANONICAL_SEEDS, CANONICAL_VARIANTS)
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    payload = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_nonfinite,
    )
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _read_yaml_object(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected YAML object: {path}")
    return payload


def _prepare_import_path(repo_root: Path) -> None:
    root = str(repo_root)
    source = str(repo_root / "src")
    if root not in sys.path:
        sys.path.insert(0, root)
    if source not in sys.path:
        sys.path.insert(0, source)


def load_locked_contract(
    repo_root: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Validate the prospective lock without mutating it."""

    root = Path(repo_root).resolve()
    lock_path = root / LOCK_RELATIVE_PATH
    source_config_path = root / SOURCE_CONFIG_RELATIVE_PATH
    aggregate_config_path = root / AGGREGATE_CONFIG_RELATIVE_PATH
    lock = _read_json_object(lock_path)
    if (
        lock.get("schema_version") != 1
        or lock.get("experiment_id") != SOURCE_EXPERIMENT_ID
        or lock.get("aggregate_experiment_id") != AGGREGATE_EXPERIMENT_ID
        or lock.get("protocol_frozen_before_evaluation") is not True
    ):
        raise RuntimeError("invalid E18 prospective protocol lock")
    files = lock.get("files")
    if not isinstance(files, dict):
        raise RuntimeError("E18 lock does not contain a file hash map")
    for relative, expected in files.items():
        candidate = (root / str(relative)).resolve()
        if (
            candidate.parent != root
            and root not in candidate.parents
        ):
            raise RuntimeError(f"E18 locked path escapes repository: {relative}")
        if not candidate.is_file() or _sha256(candidate) != str(expected):
            raise RuntimeError(f"E18 locked file changed: {relative}")
    if files.get(SOURCE_CONFIG_RELATIVE_PATH.as_posix()) != _sha256(
        source_config_path
    ):
        raise RuntimeError("E18a config is not pinned by the lock")
    if files.get(AGGREGATE_CONFIG_RELATIVE_PATH.as_posix()) != _sha256(
        aggregate_config_path
    ):
        raise RuntimeError("E18b config is not pinned by the lock")

    source_config = _read_yaml_object(source_config_path)
    aggregate_config = _read_yaml_object(aggregate_config_path)
    seeds = tuple(int(value) for value in source_config["seeds"])
    variants = tuple(str(value) for value in source_config["model"]["variants"])
    if seeds != CANONICAL_SEEDS or variants != CANONICAL_VARIANTS:
        raise RuntimeError("E18 registered seed/variant order changed")
    return source_config, aggregate_config, _sha256(lock_path)


def _infer_identity(run_dir: Path) -> Cell | None:
    candidates: set[Cell] = set()
    report_path = run_dir / "report.json"
    if report_path.is_file():
        try:
            report = _read_json_object(report_path)
            candidates.add(
                Cell(
                    seed=int(report["seed"]),
                    variant=str(report["variant"]),
                )
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass

    metrics_path = run_dir / "sequence_control_lattice_metrics.jsonl"
    if metrics_path.is_file():
        try:
            for line in metrics_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                candidates.add(
                    Cell(
                        seed=int(row["seed"]),
                        variant=str(row["variant"]),
                    )
                )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass

    checkpoint_root = run_dir / "checkpoints"
    if checkpoint_root.is_dir():
        for checkpoint in checkpoint_root.glob("*_seed*.pt"):
            stem = checkpoint.stem
            variant, separator, seed_text = stem.rpartition("_seed")
            if separator:
                with contextlib.suppress(ValueError):
                    candidates.add(Cell(seed=int(seed_text), variant=variant))
    return next(iter(candidates)) if len(candidates) == 1 else None


def _command_option(args: list[str], name: str) -> str | None:
    for index, item in enumerate(args):
        if item == name and index + 1 < len(args):
            return args[index + 1]
        if item.startswith(f"{name}="):
            return item.split("=", 1)[1]
    return None


def _live_e18_processes() -> list[LiveRun]:
    live: list[LiveRun] = []
    own_pid = os.getpid()
    for proc in sorted(Path("/proc").glob("[0-9]*")):
        try:
            pid = int(proc.name)
            if pid == own_pid:
                continue
            raw = (proc / "cmdline").read_bytes()
        except (OSError, ValueError):
            continue
        args = [
            item.decode("utf-8", errors="replace")
            for item in raw.split(b"\0")
            if item
        ]
        if not args:
            continue
        target = (
            SOURCE_EXPERIMENT_ID in args
            or f"experiments.{SOURCE_EXPERIMENT_ID}" in args
            or any(item.endswith(f"{SOURCE_EXPERIMENT_ID}.py") for item in args)
        )
        if not target or "--dry-run" in args:
            continue

        seed_text = _command_option(args, "--seed")
        variant = _command_option(args, "--variant")
        try:
            seed = int(seed_text) if seed_text is not None else None
        except ValueError:
            seed = None
        live.append(
            LiveRun(
                pid=pid,
                command=" ".join(args),
                seed=seed,
                variant=variant,
            )
        )
    return live


def scan_e18_status(
    *,
    repo_root: str | Path,
    artifact_root: str | Path,
    include_live: bool = True,
) -> E18Status:
    """Scan all E18a run directories and validate every completed MAIN run."""

    root = Path(repo_root).resolve()
    artifacts = Path(artifact_root).resolve()
    source_config, aggregate_config, lock_sha256 = load_locked_contract(root)
    _prepare_import_path(root)
    validator_module = importlib.import_module(
        "experiments.e18b_sequence_control_lattice_aggregate"
    )
    validate_run = validator_module._validate_source_run
    source_contract = validator_module._source_contract
    seeds, variants, demands, updates, gaps = source_contract(aggregate_config)
    completed: dict[Cell, list[str]] = {}
    incomplete: list[IncompleteRun] = []
    ignored_dry: list[str] = []
    namespace = artifacts / SOURCE_EXPERIMENT_ID
    if namespace.is_dir():
        for run_dir in sorted(path for path in namespace.iterdir() if path.is_dir()):
            manifest_path = run_dir / "run_manifest.json"
            identity = _infer_identity(run_dir)
            try:
                manifest = _read_json_object(manifest_path)
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
                incomplete.append(
                    IncompleteRun(
                        run_dir=str(run_dir.resolve()),
                        reason=f"missing_or_invalid_manifest: {error}",
                        seed=identity.seed if identity else None,
                        variant=identity.variant if identity else None,
                    )
                )
                continue
            mode = manifest.get("run_mode")
            if mode == "DRY_RUN":
                ignored_dry.append(str(run_dir.resolve()))
                continue
            if mode != "MAIN":
                incomplete.append(
                    IncompleteRun(
                        run_dir=str(run_dir.resolve()),
                        reason=f"unexpected_run_mode={mode!r}",
                        seed=identity.seed if identity else None,
                        variant=identity.variant if identity else None,
                    )
                )
                continue
            try:
                result = validate_run(
                    run_dir=run_dir,
                    expected_config=source_config,
                    source_config_path=root / SOURCE_CONFIG_RELATIVE_PATH,
                    variants=variants,
                    seeds=seeds,
                    demands=demands,
                    updates=updates,
                    gaps=gaps,
                    protocol_lock_sha256=lock_sha256,
                )
                if result is None:
                    raise RuntimeError("MAIN run was classified as non-MAIN")
                key, _, _ = result
                cell = Cell(seed=int(key[0]), variant=str(key[1]))
                completed.setdefault(cell, []).append(str(run_dir.resolve()))
            except Exception as error:  # validator supplies the exact contract error
                incomplete.append(
                    IncompleteRun(
                        run_dir=str(run_dir.resolve()),
                        reason=f"{type(error).__name__}: {error}",
                        seed=identity.seed if identity else None,
                        variant=identity.variant if identity else None,
                    )
                )

    status = E18Status(
        repo_root=str(root),
        artifact_root=str(artifacts),
        protocol_lock_sha256=lock_sha256,
        completed_runs=completed,
        incomplete_runs=incomplete,
        ignored_dry_runs=ignored_dry,
        live_runs=_live_e18_processes() if include_live else [],
    )
    if (
        len(status.completed_cells) == len(status.canonical_cells)
        and not status.incomplete_runs
        and not status.duplicates
    ):
        try:
            aggregate_runtime_config = json.loads(
                json.dumps(aggregate_config)
            )
            aggregate_runtime_config["source"]["config_path"] = str(
                (root / SOURCE_CONFIG_RELATIVE_PATH).resolve()
            )
            collect_sources = validator_module.collect_main_sources
            rows, provenance = collect_sources(
                artifact_root=artifacts,
                config=aggregate_runtime_config,
                protocol_lock_sha256=lock_sha256,
            )
            status.aggregate_source_rows = len(rows)
            status.aggregate_source_runs = len(provenance)
            if len(rows) != 1200 or len(provenance) != 25:
                status.aggregate_provenance_error = (
                    "validated source cardinality is not 25 runs / 1,200 rows"
                )
        except Exception as error:
            status.aggregate_provenance_error = (
                f"{type(error).__name__}: {error}"
            )
    return status


def _print_human(status: E18Status) -> None:
    payload = status.as_json()
    counts = payload["counts"]
    print(
        "[E18 STATUS] "
        f"completed={counts['completed_cells']}/25 "
        f"missing={counts['missing_cells']} "
        f"incomplete={counts['incomplete_main_runs']} "
        f"duplicates={counts['duplicate_completed_cells']} "
        f"live={counts['live_main_processes']}"
    )
    print("[E18 STATUS] latest_pointer_used=false")
    for item in payload["cells"]:
        print(
            f"{item['index']:02d} "
            f"seed={item['seed']} "
            f"variant={item['variant']:<18} "
            f"status={item['status']}"
        )
    for item in payload["incomplete_runs"]:
        print(
            "[BLOCKER] INCOMPLETE "
            f"{item['run_dir']} "
            f"identity={item['variant']}/seed{item['seed']} "
            f"reason={item['reason']}"
        )
    for item in payload["duplicate_runs"]:
        print(
            "[BLOCKER] DUPLICATE "
            f"{item['variant']}/seed{item['seed']} "
            f"runs={item['run_dirs']}"
        )
    for item in payload["live_runs"]:
        print(
            "[BLOCKER] LIVE "
            f"pid={item['pid']} "
            f"identity={item['variant']}/seed{item['seed']}"
        )
    aggregate = payload["aggregate"]
    print(
        "[E18 AGGREGATE] "
        f"ready={str(aggregate['ready']).lower()} "
        f"source_runs={aggregate['source_runs']} "
        f"source_rows={aggregate['source_rows']} "
        "mode=SEPARATE_CPU_COMMAND_ONLY"
    )
    if aggregate["provenance_error"] is not None:
        print(f"[E18 AGGREGATE] provenance_error={aggregate['provenance_error']}")


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
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--require-launch-safe", action="store_true")
    parser.add_argument("--require-aggregate-ready", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    status = scan_e18_status(
        repo_root=args.repo_root,
        artifact_root=args.artifact_root,
    )
    if args.json:
        print(json.dumps(status.as_json(), indent=2, sort_keys=True))
    else:
        _print_human(status)
    if args.require_launch_safe and not status.launch_safe:
        print(
            "[BLOCKED] E18 launch is not safe: "
            + (", ".join(status.blockers) or "no missing cells"),
            file=sys.stderr,
        )
        return 2
    if args.require_aggregate_ready and not status.aggregate_ready:
        print(
            "[BLOCKED] E18 aggregate provenance is not 25/25 ready.",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
