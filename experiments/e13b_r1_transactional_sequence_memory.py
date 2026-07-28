from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from catena.core.io import file_sha256, read_latest_pointer, write_jsonl
from catena.models.sequence_memory_v2 import (
    SequenceControlV2,
    TransactionalSequenceMemoryV2,
    sequence_parameter_count_v2,
)
from catena.training.sequence_training_v2 import (
    evaluate_sequence_memory_v2,
    train_sequence_memory_v2,
)
from experiments.common import finalize_run, initialize_run

EXPERIMENT_ID = "e13b_r1_transactional_sequence_memory"
DEFAULT_CONFIG = "configs/e13b_r1_transactional_sequence_memory.yaml"
CALIBRATION_EXPERIMENT_ID = "e13a_r2_sequence_floor_throughput"


def build_local_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=EXPERIMENT_ID)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument(
        "--variant",
        choices=[value.value for value in SequenceControlV2],
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ignore-calibration", action="store_true")
    return parser


def _read_json_object(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _load_calibration_dependency(
    *,
    artifact_root: str,
    source_config_path: str,
) -> dict[str, str]:
    calibration_dir = read_latest_pointer(
        artifact_root,
        CALIBRATION_EXPERIMENT_ID,
    ).resolve()
    expected_root = (
        Path(artifact_root) / CALIBRATION_EXPERIMENT_ID
    ).resolve()
    if calibration_dir.parent != expected_root:
        raise RuntimeError(
            "E13a-R2 latest pointer resolves outside its experiment namespace"
        )
    report_path = calibration_dir / "report.json"
    manifest_path = calibration_dir / "run_manifest.json"
    report = _read_json_object(report_path)
    manifest = _read_json_object(manifest_path)
    report_sha256 = file_sha256(report_path)
    if report.get("status") != "PASS":
        raise RuntimeError("E13a-R2 dependency is not a completed MAIN PASS")
    if (
        manifest.get("schema_version") != 2
        or manifest.get("experiment_id") != CALIBRATION_EXPERIMENT_ID
        or manifest.get("run_mode") != "MAIN"
        or manifest.get("run_id") != calibration_dir.name
        or manifest.get("report_sha256") != report_sha256
    ):
        raise RuntimeError(
            "E13a-R2 manifest does not pin a completed schema-v2 MAIN report"
        )
    if not report.get("claim_gate", {}).get("go_for_e13b_r1", False):
        raise RuntimeError("E13a-R2 did not open the repaired E13b-R1 gate")
    if not report.get("distractor_path_contract", {}).get("passed", False):
        raise RuntimeError("E13a-R2 distractor-path contract did not pass")
    if not report.get("e13b_scale_feasibility", {}).get("passed", False):
        raise RuntimeError("E13a-R2 scale-feasibility gate did not pass")
    expected_source_hash = (
        report.get("e13b_scale_feasibility", {})
        .get("source_config_file_sha256")
    )
    actual_source_hash = file_sha256(source_config_path)
    if expected_source_hash != actual_source_hash:
        raise RuntimeError(
            "E13b-R1 config no longer matches the configuration calibrated "
            "by E13a-R2"
        )
    return {
        "experiment_id": CALIBRATION_EXPERIMENT_ID,
        "run_dir": str(calibration_dir.resolve()),
        "report_path": str(report_path.resolve()),
        "report_sha256": report_sha256,
        "run_manifest_path": str(manifest_path.resolve()),
        "run_manifest_sha256": file_sha256(manifest_path),
        "source_config_sha256": actual_source_hash,
    }


def _prefixed_metrics(
    prefix: str,
    metrics: dict[str, float | int | str | bool],
) -> dict[str, float | int | str | bool]:
    return {
        f"{prefix}_{key}": value
        for key, value in metrics.items()
        if key != "base_transaction_digest"
    }


def main() -> None:
    args = build_local_parser().parse_args()
    calibration_dependency: dict[str, str] | None = None
    if not args.ignore_calibration and not args.dry_run:
        calibration_dependency = _load_calibration_dependency(
            artifact_root=args.artifact_root,
            source_config_path=args.config,
        )

    config, run_dir, device = initialize_run(
        experiment_id=EXPERIMENT_ID,
        config_path=args.config,
        artifact_root=args.artifact_root,
        device_request=args.device,
        run_mode="DRY_RUN" if args.dry_run else "MAIN",
    )
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
    assay_updates = int(
        config["evaluation"]["active_path_assay"]["updates"]
    )
    assay_gap = int(
        config["evaluation"]["active_path_assay"]["gap_events"]
    )
    if args.dry_run:
        variants = variants[:1]
        seeds = seeds[:1]
        steps = min(steps, 2)
        eval_updates = [1]
        eval_gaps = [0]
        eval_batches = 1
        eval_batch_size = 2
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
        variant = SequenceControlV2(variant_name)
        for seed in seeds:
            torch.manual_seed(seed)
            model = TransactionalSequenceMemoryV2(
                control=variant,
                num_entities=int(config["data"]["num_entities"]),
                value_vocab=int(config["data"]["value_vocab"]),
                embedding_dim=int(config["model"]["embedding_dim"]),
                hidden_dim=int(config["model"]["hidden_dim"]),
            )
            trace = train_sequence_memory_v2(
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
            checkpoint = (
                run_dir
                / "checkpoints"
                / f"{variant.value}_seed{seed}.pt"
            )
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "variant": variant.value,
                    "seed": seed,
                    "config": config,
                    "model_class": "TransactionalSequenceMemoryV2",
                },
                checkpoint,
            )
            checkpoint_sha256 = file_sha256(checkpoint)
            for updates in eval_updates:
                # The same seed is deliberately reused across every gap.
                evaluation_seed = 100_000 + 1000 * seed + 10 * updates
                for gap in eval_gaps:
                    metrics = evaluate_sequence_memory_v2(
                        model=model,
                        batches=eval_batches,
                        batch_size=eval_batch_size,
                        num_entities=int(config["data"]["num_entities"]),
                        value_vocab=int(config["data"]["value_vocab"]),
                        updates=updates,
                        gap_events=gap,
                        device=device,
                        seed=evaluation_seed,
                    )
                    row: dict[str, float | int | str] = {
                        "variant": variant.value,
                        "seed": seed,
                        "updates": updates,
                        "gap_events": gap,
                        "evaluation_seed": evaluation_seed,
                        "checkpoint": str(checkpoint.resolve()),
                        "checkpoint_sha256": checkpoint_sha256,
                        "parameter_count": sequence_parameter_count_v2(model),
                        "train_final_loss": trace.final_loss,
                        "train_best_loss": trace.best_loss,
                        "examples_per_second": trace.examples_per_second,
                        "peak_memory_bytes": trace.peak_memory_bytes,
                        **metrics,
                    }
                    if (
                        not args.dry_run
                        and updates == assay_updates
                        and gap == assay_gap
                    ):
                        activated = evaluate_sequence_memory_v2(
                            model=model,
                            batches=eval_batches,
                            batch_size=eval_batch_size,
                            num_entities=int(config["data"]["num_entities"]),
                            value_vocab=int(config["data"]["value_vocab"]),
                            updates=updates,
                            gap_events=gap,
                            device=device,
                            seed=evaluation_seed,
                            activate_distractor_verified=True,
                        )
                        if (
                            activated["base_transaction_digest"]
                            != metrics["base_transaction_digest"]
                        ):
                            raise RuntimeError(
                                "Active-path assay changed the base transaction"
                            )
                        row.update(
                            _prefixed_metrics(
                                "distractor_activation",
                                activated,
                            )
                        )
                    rows.append(row)

    digests: dict[tuple[str, int, int], set[str]] = defaultdict(set)
    for row in rows:
        digests[
            (str(row["variant"]), int(row["seed"]), int(row["updates"]))
        ].add(str(row["base_transaction_digest"]))
    base_pairing_passed = all(len(values) == 1 for values in digests.values())
    if not base_pairing_passed:
        raise RuntimeError("Base transaction digest changed across gap conditions")

    report = {
        "status": "DRY_RUN" if args.dry_run else "PASS",
        "run_scope": "LEARNED_DISTRACTOR_TRANSACTIONAL_SEQUENCE",
        "run_mode": "DRY_RUN" if args.dry_run else "MAIN",
        "variant": variants[0] if len(variants) == 1 else "MULTIPLE",
        "seed": seeds[0] if len(seeds) == 1 else "MULTIPLE",
        "calibration_dependency": calibration_dependency,
        "distractor_contract": {
            "layout": "one_total_block_after_first_verified_update",
            "verified_role": "semantic_input_only",
            "update_mask_role": "audit_metadata_only",
            "base_transaction_digest_paired_across_gaps": base_pairing_passed,
            "active_path_assay_cell": {
                "updates": assay_updates,
                "gap_events": assay_gap,
            },
        },
        "rows": len(rows),
        "claim_gate": {
            "status": "DRY_RUN" if args.dry_run else "PENDING_AGGREGATE",
            "allowed_claim": (
                "Per-run learned-distractor sequence evidence only after "
                "E13c-R1 aggregation."
            ),
            "forbidden_claim": (
                "Natural-language, recurrent-LM, agent, planning, or "
                "official-backend transfer."
            ),
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
