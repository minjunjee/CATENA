from __future__ import annotations

import argparse

import torch

from catena.core.io import file_sha256, read_latest_pointer, write_jsonl
from catena.models.sequence_memory import (
    SequenceControl,
    TransactionalSequenceMemory,
    sequence_parameter_count,
)
from catena.training.sequence_training import (
    evaluate_sequence_memory,
    train_sequence_memory,
)
from experiments.common import finalize_run, initialize_run

EXPERIMENT_ID = "e13b_transactional_sequence_memory"
DEFAULT_CONFIG = "configs/e13b_transactional_sequence_memory.yaml"
CALIBRATION_EXPERIMENT_ID = "e13a_r1_sequence_floor_throughput"


def build_local_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=EXPERIMENT_ID)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--variant", choices=[value.value for value in SequenceControl])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ignore-calibration", action="store_true")
    return parser


def main() -> None:
    args = build_local_parser().parse_args()
    config, run_dir, device = initialize_run(
        experiment_id=EXPERIMENT_ID,
        config_path=args.config,
        artifact_root=args.artifact_root,
        device_request=args.device,
        run_mode="DRY_RUN" if args.dry_run else "MAIN",
    )
    calibration_dependency: dict[str, str] | None = None
    if not args.ignore_calibration and not args.dry_run:
        calibration_dir = read_latest_pointer(
            args.artifact_root, CALIBRATION_EXPERIMENT_ID
        )
        import json

        calibration_report = calibration_dir / "report.json"
        with calibration_report.open("r", encoding="utf-8") as handle:
            calibration = json.load(handle)
        if not calibration.get("claim_gate", {}).get("go_for_e13b", False):
            raise RuntimeError(
                "Prospective E13a-R1 did not open the E13b calibration gate"
            )
        calibration_dependency = {
            "experiment_id": CALIBRATION_EXPERIMENT_ID,
            "run_dir": str(calibration_dir.resolve()),
            "report_path": str(calibration_report.resolve()),
            "report_sha256": file_sha256(calibration_report),
        }

    variants = (
        [args.variant]
        if args.variant
        else [str(value) for value in config["model"]["variants"]]
    )
    seeds = (
        [args.seed]
        if args.seed is not None
        else [int(value) for value in config["seeds"]]
    )
    steps = int(config["training"]["steps"])
    eval_updates = [int(value) for value in config["evaluation"]["updates"]]
    eval_gaps = [int(value) for value in config["evaluation"]["gap_events"]]
    eval_batches = int(config["evaluation"]["batches"])
    eval_batch_size = int(config["evaluation"]["batch_size"])
    if args.dry_run:
        variants = variants[:1]
        seeds = seeds[:1]
        steps = min(steps, 2)
        eval_updates = [1]
        eval_gaps = [0]
        eval_batches = 1
        eval_batch_size = 2
        # Dry-run validates plumbing, not scientific scale.  Keep it small so
        # the full post-core smoke suite finishes on CPU.
        config = {
            **config,
            "data": {**config["data"], "num_entities": 8, "value_vocab": 16},
            "model": {**config["model"], "embedding_dim": 16, "hidden_dim": 32},
            "training": {
                **config["training"],
                "batch_size": 2,
                "updates": 1,
                "gap_events": 0,
            },
        }

    rows: list[dict[str, float | int | str]] = []
    for variant_name in variants:
        variant = SequenceControl(variant_name)
        for seed in seeds:
            torch.manual_seed(seed)
            model = TransactionalSequenceMemory(
                control=variant,
                num_entities=int(config["data"]["num_entities"]),
                value_vocab=int(config["data"]["value_vocab"]),
                embedding_dim=int(config["model"]["embedding_dim"]),
                hidden_dim=int(config["model"]["hidden_dim"]),
            )
            trace = train_sequence_memory(
                model=model,
                steps=steps,
                batch_size=int(config["training"]["batch_size"]),
                num_entities=int(config["data"]["num_entities"]),
                value_vocab=int(config["data"]["value_vocab"]),
                updates=int(config["training"]["updates"]),
                gap_events=int(config["training"]["gap_events"]),
                learning_rate=float(config["training"]["learning_rate"]),
                retention_weight=float(config["training"]["retention_weight"]),
                device=device,
                seed=seed,
            )
            checkpoint = run_dir / "checkpoints" / f"{variant.value}_seed{seed}.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "variant": variant.value,
                    "seed": seed,
                    "config": config,
                },
                checkpoint,
            )
            for updates in eval_updates:
                for gap in eval_gaps:
                    metrics = evaluate_sequence_memory(
                        model=model,
                        batches=eval_batches,
                        batch_size=eval_batch_size,
                        num_entities=int(config["data"]["num_entities"]),
                        value_vocab=int(config["data"]["value_vocab"]),
                        updates=updates,
                        gap_events=gap,
                        device=device,
                        seed=100_000 + 1000 * seed + 10 * updates + gap,
                    )
                    rows.append(
                        {
                            "variant": variant.value,
                            "seed": seed,
                            "updates": updates,
                            "gap_events": gap,
                            "checkpoint": str(checkpoint.resolve()),
                            "parameter_count": sequence_parameter_count(model),
                            "train_final_loss": trace.final_loss,
                            "train_best_loss": trace.best_loss,
                            "examples_per_second": trace.examples_per_second,
                            "peak_memory_bytes": trace.peak_memory_bytes,
                            **metrics,
                        }
                    )

    report = {
        "status": "DRY_RUN" if args.dry_run else "PASS",
        "run_scope": "TRANSACTIONAL_EVENT_SEQUENCE",
        "run_mode": "DRY_RUN" if args.dry_run else "MAIN",
        "variant": variants[0] if len(variants) == 1 else "MULTIPLE",
        "seed": seeds[0] if len(seeds) == 1 else "MULTIPLE",
        "calibration_dependency": calibration_dependency,
        "rows": len(rows),
        "claim_gate": {
            "status": "DRY_RUN" if args.dry_run else "PENDING_AGGREGATE",
            "allowed_claim": "Per-run sequence-memory evidence only after E13c aggregation.",
            "forbidden_claim": "Natural-language or official-backbone transfer.",
        },
    }
    write_jsonl(run_dir / "sequence_main_metrics.jsonl", rows)
    finalize_run(
        experiment_id=EXPERIMENT_ID,
        artifact_root=args.artifact_root,
        run_dir=run_dir,
        report=report,
    )
    print(f"[{EXPERIMENT_ID}] {report['status']}: {run_dir}")


if __name__ == "__main__":
    main()
