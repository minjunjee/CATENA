from __future__ import annotations

from pathlib import Path

import torch

from catena.core.io import write_jsonl
from catena.models.sequence_memory import (
    SequenceControl,
    TransactionalSequenceMemory,
    sequence_parameter_count,
)
from catena.training.sequence_training import (
    evaluate_sequence_memory,
    train_sequence_memory,
)
from experiments.common import build_parser, finalize_run, initialize_run

EXPERIMENT_ID = "e13a_sequence_floor_throughput"
DEFAULT_CONFIG = "configs/e13a_sequence_floor_throughput.yaml"


def main() -> None:
    parser = build_parser(EXPERIMENT_ID, DEFAULT_CONFIG)
    args = parser.parse_args()
    config, run_dir, device = initialize_run(
        experiment_id=EXPERIMENT_ID,
        config_path=args.config,
        artifact_root=args.artifact_root,
        device_request=args.device,
    )
    variants = [SequenceControl(value) for value in config["model"]["variants"]]
    steps = int(config["training"]["steps"])
    if args.dry_run:
        steps = min(steps, 2)
        config = {
            **config,
            "data": {
                **config["data"],
                "num_entities": 8,
                "value_vocab": 16,
                "updates": 1,
                "gap_events": 0,
            },
            "model": {**config["model"], "embedding_dim": 16, "hidden_dim": 32},
            "training": {**config["training"], "batch_size": 2},
            "evaluation": {**config["evaluation"], "batches": 1, "batch_size": 2},
            "claim_gate": {
                **config["claim_gate"],
                "minimum_dual_exact_match": 0.0,
                "minimum_affected_gain": -1.0,
                "max_retention_mse": 1.0,
                "minimum_examples_per_second": 0.0,
            },
        }

    rows: list[dict[str, float | int | str]] = []
    models: dict[str, TransactionalSequenceMemory] = {}
    for index, variant in enumerate(variants):
        seed = int(config["seed"]) + index
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
            updates=int(config["data"]["updates"]),
            gap_events=int(config["data"]["gap_events"]),
            learning_rate=float(config["training"]["learning_rate"]),
            retention_weight=float(config["training"]["retention_weight"]),
            device=device,
            seed=seed,
        )
        evaluation = evaluate_sequence_memory(
            model=model,
            batches=int(config["evaluation"]["batches"]),
            batch_size=int(config["evaluation"]["batch_size"]),
            num_entities=int(config["data"]["num_entities"]),
            value_vocab=int(config["data"]["value_vocab"]),
            updates=int(config["data"]["updates"]),
            gap_events=int(config["data"]["gap_events"]),
            device=device,
            seed=10_000 + seed,
        )
        checkpoint = run_dir / "checkpoints" / f"{variant.value}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "variant": variant.value, "config": config}, checkpoint)
        models[variant.value] = model
        rows.append(
            {
                "variant": variant.value,
                "seed": seed,
                "parameter_count": sequence_parameter_count(model),
                "checkpoint": str(checkpoint.resolve()),
                "final_loss": trace.final_loss,
                "best_loss": trace.best_loss,
                "examples_per_second": trace.examples_per_second,
                "peak_memory_bytes": trace.peak_memory_bytes,
                **evaluation,
            }
        )

    row_by_variant = {str(row["variant"]): row for row in rows}
    tied = row_by_variant[SequenceControl.TIED.value]
    dual = row_by_variant[SequenceControl.DUAL.value]
    affected_gain = float(tied["affected_mse"]) - float(dual["affected_mse"])
    retention_ok = float(dual["retention_mse"]) <= float(config["claim_gate"]["max_retention_mse"])
    floor_ok = float(dual["entity_exact_match"]) >= float(config["claim_gate"]["minimum_dual_exact_match"])
    gain_ok = affected_gain >= float(config["claim_gate"]["minimum_affected_gain"])
    throughput_ok = min(float(row["examples_per_second"]) for row in rows) >= float(
        config["claim_gate"]["minimum_examples_per_second"]
    )
    go = floor_ok and gain_ok and retention_ok and throughput_ok
    report = {
        "status": "PASS",
        "run_scope": "SEQUENCE_BRIDGE_CALIBRATION",
        "summary": {
            "affected_gain_tied_minus_dual": affected_gain,
            "floor_ok": floor_ok,
            "gain_ok": gain_ok,
            "retention_ok": retention_ok,
            "throughput_ok": throughput_ok,
        },
        "claim_gate": {
            "go_for_e13b": go,
            "allowed_claim": "Calibration only; no confirmatory sequence-memory claim.",
            "forbidden_claim": "Pretrained recurrent-language-model transfer.",
        },
    }
    write_jsonl(run_dir / "sequence_calibration_metrics.jsonl", rows)
    finalize_run(
        experiment_id=EXPERIMENT_ID,
        artifact_root=args.artifact_root,
        run_dir=run_dir,
        report=report,
    )
    print(f"[{EXPERIMENT_ID}] PASS: {run_dir}")


if __name__ == "__main__":
    main()
