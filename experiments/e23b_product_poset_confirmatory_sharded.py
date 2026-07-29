"""Operational seed-sharding entry point for the frozen E23b experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catena.post_e21.e23b_sharded_execution import (
    DEFAULT_CONFIG,
    aggregate_sharded_execution,
    prepare_sharded_execution,
    require_main_acknowledgement,
    run_cpu_serial_shard_equivalence,
    run_shard_worker,
)
from catena.systems.device import resolve_device

REPO_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Conservative seed-sharded execution for frozen e23b_product_poset_confirmatory"
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--config", default=DEFAULT_CONFIG)
    prepare.add_argument("--artifact-root", required=True)
    prepare.add_argument("--e18-freeze")
    prepare.add_argument("--e23a-screen")
    prepare.add_argument("--e22b-run")
    prepare.add_argument("--shard-count", type=int, default=4)
    prepare.add_argument("--source-lock-tag")
    prepare.add_argument("--equivalence-report")
    prepare.add_argument(
        "--gpu-indices",
        help="Four comma-separated physical GPU indices for MAIN.",
    )
    prepare.add_argument("--dry-run", action="store_true")

    equivalence = subparsers.add_parser("verify-equivalence")
    equivalence.add_argument("--config", default=DEFAULT_CONFIG)
    equivalence.add_argument("--output-root", required=True)
    equivalence.add_argument("--e18-freeze", required=True)
    equivalence.add_argument("--e23a-screen", required=True)
    equivalence.add_argument("--e22b-run", required=True)
    equivalence.add_argument("--source-lock-tag", required=True)

    worker = subparsers.add_parser("worker")
    worker.add_argument("--workspace", required=True)
    worker.add_argument("--shard-id", required=True)
    worker.add_argument("--device", default="auto")

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--workspace", required=True)
    aggregate.add_argument("--artifact-root", required=True)
    aggregate.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        if not args.dry_run:
            require_main_acknowledgement()
        gpu_indices = (
            None
            if args.gpu_indices is None
            else tuple(int(value.strip()) for value in args.gpu_indices.split(","))
        )
        workspace = prepare_sharded_execution(
            repo_root=REPO_ROOT,
            config_path=args.config,
            artifact_root=args.artifact_root,
            e18_freeze=args.e18_freeze,
            e23a_screen=args.e23a_screen,
            e22b_run=args.e22b_run,
            shard_count=args.shard_count,
            dry_run=args.dry_run,
            source_lock_tag=args.source_lock_tag,
            equivalence_report=args.equivalence_report,
            gpu_indices=gpu_indices,
        )
        print(workspace)
        return
    if args.command == "verify-equivalence":
        report = run_cpu_serial_shard_equivalence(
            repo_root=REPO_ROOT,
            config_path=args.config,
            output_root=Path(args.output_root),
            e18_freeze=args.e18_freeze,
            e23a_screen=args.e23a_screen,
            e22b_run=args.e22b_run,
            source_lock_tag=args.source_lock_tag,
        )
        print(report)
        return
    if args.command == "worker":
        device = resolve_device(args.device)
        shard_dir = run_shard_worker(
            workspace=args.workspace,
            shard_id=args.shard_id,
            device=device,
        )
        print(shard_dir)
        return
    if args.command == "aggregate":
        run_dir = aggregate_sharded_execution(
            workspace=args.workspace,
            artifact_root=args.artifact_root,
            device_request=args.device,
        )
        print(run_dir)
        return
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
