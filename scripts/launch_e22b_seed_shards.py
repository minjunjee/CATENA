#!/usr/bin/env python3
"""Canonical launcher for the performance-only E22b seed-shard amendment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    repository = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repository))
    sys.path.insert(0, str(repository / "src"))

from catena.post_e21.e22b_sharding import (
    DEFAULT_CONFIG,
    coordinate_sharded_run,
    run_cpu_serial_shard_equivalence,
    run_shard_worker,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "run",
        help="Run one canonical four-worker E22b artifact.",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--selection-run", required=True)
    parser.add_argument(
        "--devices",
        default="cuda:0,cuda:1,cuda:2,cuda:3",
        help="Exactly four comma-separated explicit devices.",
    )
    parser.add_argument("--source-lock-tag")
    parser.add_argument("--equivalence-report")
    parser.add_argument("--dry-run", action="store_true")


def _equivalence_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "verify-equivalence",
        help="Run non-evidence CPU serial-vs-shard equivalence.",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--selection-run", required=True)
    parser.add_argument(
        "--output-root",
        required=True,
        help="Fresh path below /tmp.",
    )
    parser.add_argument(
        "--selection-is-dry-run",
        action="store_true",
        help="Only for a non-evidence E22a dry-run dependency.",
    )


def _worker_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser("_worker", help=argparse.SUPPRESS)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--shard-index", required=True, type=int)
    parser.add_argument("--device", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Performance-only paired-seed sharding for the frozen E22b scientific protocol"
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _run_parser(subparsers)
    _equivalence_parser(subparsers)
    _worker_parser(subparsers)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "_worker":
        manifest = run_shard_worker(
            repo_root=REPO_ROOT,
            plan_path=Path(args.plan),
            shard_index=int(args.shard_index),
            device_request=str(args.device),
        )
        print(f"[e22b-shard-worker] PASS: {manifest}")
        return
    if args.command == "verify-equivalence":
        report = run_cpu_serial_shard_equivalence(
            repo_root=REPO_ROOT,
            config_path=args.config,
            selection_run=args.selection_run,
            output_root=Path(args.output_root),
            selection_is_dry_run=bool(args.selection_is_dry_run),
        )
        print(f"[e22b-shard-equivalence] PASS: {report}")
        return
    devices = tuple(part.strip() for part in str(args.devices).split(","))
    run_dir = coordinate_sharded_run(
        repo_root=REPO_ROOT,
        config_path=args.config,
        artifact_root=args.artifact_root,
        selection_run=args.selection_run,
        devices=devices,
        dry_run=bool(args.dry_run),
        source_lock_tag=args.source_lock_tag,
        equivalence_report=args.equivalence_report,
    )
    print(f"[e22b-sharded] PASS: {run_dir}")


if __name__ == "__main__":
    main()
