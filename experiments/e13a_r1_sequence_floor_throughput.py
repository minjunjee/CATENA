from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from statistics import median
from time import perf_counter

import torch

from catena.core.config import load_config
from catena.core.io import file_sha256, write_jsonl
from catena.data.transactional_sequence import (
    TransactionalSequenceBatch,
    generate_transactional_sequence_batch,
)
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

EXPERIMENT_ID = "e13a_r1_sequence_floor_throughput"
DEFAULT_CONFIG = "configs/e13a_r1_sequence_floor_throughput.yaml"


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _state_dict_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def measure_paired_forward_throughput(
    *,
    models: Mapping[str, TransactionalSequenceMemory],
    batch: TransactionalSequenceBatch,
    device: torch.device,
    warmup_repeats: int,
    measured_repeats: int,
) -> dict[str, dict[str, float | int | str]]:
    if warmup_repeats < 0 or measured_repeats <= 0:
        raise ValueError("timing repeats must be nonnegative/positive")
    names = tuple(models)
    if not names:
        raise ValueError("models must not be empty")
    for model in models.values():
        model.to(device).eval()
    with torch.no_grad():
        for _ in range(warmup_repeats):
            for name in names:
                models[name](batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        durations: dict[str, list[float]] = {name: [] for name in names}
        for repeat in range(measured_repeats):
            order = names if repeat % 2 == 0 else tuple(reversed(names))
            for name in order:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = perf_counter()
                models[name](batch)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                durations[name].append(perf_counter() - started)

    batch_size = int(batch.initial_state.shape[0])
    result: dict[str, dict[str, float | int | str]] = {}
    for name in names:
        median_seconds = median(durations[name])
        result[name] = {
            "timing_method": "paired_alternating_forward_only",
            "warmup_repeats": warmup_repeats,
            "measured_repeats": measured_repeats,
            "median_forward_seconds": median_seconds,
            "examples_per_second": batch_size / max(median_seconds, 1e-12),
        }
    return result


def project_e13b_schedule(
    *,
    examples_per_second: Mapping[str, float],
    steps: int,
    batch_size: int,
    planned_wave_jobs: tuple[int, ...],
) -> dict[str, object]:
    if not examples_per_second or any(
        not math.isfinite(float(value)) or float(value) <= 0.0
        for value in examples_per_second.values()
    ):
        raise ValueError("examples_per_second must contain positive finite values")
    if steps <= 0 or batch_size <= 0:
        raise ValueError("steps and batch_size must be positive")
    if not planned_wave_jobs or any(value <= 0 for value in planned_wave_jobs):
        raise ValueError("planned_wave_jobs must contain positive counts")
    seconds_per_run = {
        variant: steps * batch_size / throughput
        for variant, throughput in examples_per_second.items()
    }
    slowest_run_seconds = max(seconds_per_run.values())
    wave_seconds = [slowest_run_seconds for _ in planned_wave_jobs]
    return {
        "projected_seconds_per_run": seconds_per_run,
        "projected_slowest_run_seconds": slowest_run_seconds,
        "planned_wave_jobs": list(planned_wave_jobs),
        "projected_seconds_per_wave": wave_seconds,
        "projected_total_sequential_wave_seconds": sum(wave_seconds),
        "assumption": (
            "one isolated job per GPU; jobs within a wave run concurrently; "
            "waves run sequentially"
        ),
    }


def measure_e13b_scale_training_feasibility(
    *,
    source_config: dict,
    device: torch.device,
    seed: int,
    warmup_steps: int,
    measured_steps: int,
    planned_wave_jobs: tuple[int, ...],
) -> tuple[dict[str, dict[str, float | int]], dict[str, object]]:
    if warmup_steps < 0 or measured_steps <= 0:
        raise ValueError("feasibility steps must be nonnegative/positive")
    variants = tuple(SequenceControl(value) for value in source_config["model"]["variants"])
    measurements: dict[str, dict[str, float | int]] = {}
    for variant in variants:
        torch.manual_seed(seed)
        model = TransactionalSequenceMemory(
            control=variant,
            num_entities=int(source_config["data"]["num_entities"]),
            value_vocab=int(source_config["data"]["value_vocab"]),
            embedding_dim=int(source_config["model"]["embedding_dim"]),
            hidden_dim=int(source_config["model"]["hidden_dim"]),
        )
        if warmup_steps:
            train_sequence_memory(
                model=model,
                steps=warmup_steps,
                batch_size=int(source_config["training"]["batch_size"]),
                num_entities=int(source_config["data"]["num_entities"]),
                value_vocab=int(source_config["data"]["value_vocab"]),
                updates=int(source_config["training"]["updates"]),
                gap_events=int(source_config["training"]["gap_events"]),
                learning_rate=float(source_config["training"]["learning_rate"]),
                retention_weight=float(source_config["training"]["retention_weight"]),
                device=device,
                seed=seed,
            )
        measured = train_sequence_memory(
            model=model,
            steps=measured_steps,
            batch_size=int(source_config["training"]["batch_size"]),
            num_entities=int(source_config["data"]["num_entities"]),
            value_vocab=int(source_config["data"]["value_vocab"]),
            updates=int(source_config["training"]["updates"]),
            gap_events=int(source_config["training"]["gap_events"]),
            learning_rate=float(source_config["training"]["learning_rate"]),
            retention_weight=float(source_config["training"]["retention_weight"]),
            device=device,
            seed=seed + 1,
        )
        measurements[variant.value] = {
            "warmup_training_steps": warmup_steps,
            "measured_training_steps": measured_steps,
            "training_examples_per_second": measured.examples_per_second,
            "peak_memory_bytes": measured.peak_memory_bytes,
            "parameter_count": sequence_parameter_count(model),
            "num_entities": int(source_config["data"]["num_entities"]),
            "value_vocab": int(source_config["data"]["value_vocab"]),
            "embedding_dim": int(source_config["model"]["embedding_dim"]),
            "hidden_dim": int(source_config["model"]["hidden_dim"]),
            "updates": int(source_config["training"]["updates"]),
            "gap_events": int(source_config["training"]["gap_events"]),
            "batch_size": int(source_config["training"]["batch_size"]),
            "target_training_steps": int(source_config["training"]["steps"]),
        }
    schedule = project_e13b_schedule(
        examples_per_second={
            variant: float(row["training_examples_per_second"])
            for variant, row in measurements.items()
        },
        steps=int(source_config["training"]["steps"]),
        batch_size=int(source_config["training"]["batch_size"]),
        planned_wave_jobs=planned_wave_jobs,
    )
    return measurements, schedule


def _build_model(
    *,
    variant: SequenceControl,
    config: dict,
) -> TransactionalSequenceMemory:
    return TransactionalSequenceMemory(
        control=variant,
        num_entities=int(config["data"]["num_entities"]),
        value_vocab=int(config["data"]["value_vocab"]),
        embedding_dim=int(config["model"]["embedding_dim"]),
        hidden_dim=int(config["model"]["hidden_dim"]),
    )


def main() -> None:
    parser = build_parser(EXPERIMENT_ID, DEFAULT_CONFIG)
    args = parser.parse_args()
    config, run_dir, device = initialize_run(
        experiment_id=EXPERIMENT_ID,
        config_path=args.config,
        artifact_root=args.artifact_root,
        device_request=args.device,
        run_mode="DRY_RUN" if args.dry_run else "MAIN",
    )
    variants = [SequenceControl(value) for value in config["model"]["variants"]]
    if set(variants) != {SequenceControl.TIED, SequenceControl.DUAL}:
        raise ValueError("E13a-R1 requires exactly tied and dual variants")
    steps = int(config["training"]["steps"])
    evaluation_batches = int(config["evaluation"]["batches"])
    evaluation_batch_size = int(config["evaluation"]["batch_size"])
    timing_warmup = int(config["timing"]["warmup_repeats"])
    timing_repeats = int(config["timing"]["measured_repeats"])
    timing_batch_size = int(config["timing"]["batch_size"])
    feasibility_warmup = int(config["feasibility"]["warmup_training_steps"])
    feasibility_steps = int(config["feasibility"]["measured_training_steps"])
    planned_wave_jobs = tuple(
        int(value) for value in config["feasibility"]["planned_wave_jobs"]
    )
    if planned_wave_jobs != (4, 4, 2):
        raise ValueError("feasibility.planned_wave_jobs must equal [4, 4, 2]")
    if args.dry_run:
        steps = min(steps, 2)
        evaluation_batches = 1
        evaluation_batch_size = 2
        timing_warmup = 1
        timing_repeats = 2
        timing_batch_size = 2
        feasibility_warmup = 0
        feasibility_steps = 1
        config = {
            **config,
            "data": {
                **config["data"],
                "num_entities": 8,
                "value_vocab": 16,
                "updates": 1,
                "gap_events": 0,
            },
            "model": {
                **config["model"],
                "embedding_dim": 16,
                "hidden_dim": 32,
            },
            "training": {**config["training"], "batch_size": 2},
        }

    paired_seed = int(config["seed"])
    rows: list[dict[str, float | int | str | bool]] = []
    models: dict[str, TransactionalSequenceMemory] = {}
    initialization_hashes: dict[str, str] = {}
    parameter_counts: dict[str, int] = {}
    for variant in variants:
        torch.manual_seed(paired_seed)
        model = _build_model(variant=variant, config=config)
        initialization_hashes[variant.value] = _state_dict_sha256(
            model.state_dict()
        )
        parameter_counts[variant.value] = sequence_parameter_count(model)
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
            seed=paired_seed,
        )
        evaluation = evaluate_sequence_memory(
            model=model,
            batches=evaluation_batches,
            batch_size=evaluation_batch_size,
            num_entities=int(config["data"]["num_entities"]),
            value_vocab=int(config["data"]["value_vocab"]),
            updates=int(config["data"]["updates"]),
            gap_events=int(config["data"]["gap_events"]),
            device=device,
            seed=int(config["evaluation"]["seed"]),
        )
        checkpoint = run_dir / "checkpoints" / f"{variant.value}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "variant": variant.value,
                "seed": paired_seed,
                "config": config,
            },
            checkpoint,
        )
        models[variant.value] = model
        rows.append(
            {
                "variant": variant.value,
                "training_seed": paired_seed,
                "evaluation_seed": int(config["evaluation"]["seed"]),
                "parameter_count": parameter_counts[variant.value],
                "initialization_sha256": initialization_hashes[variant.value],
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": file_sha256(checkpoint),
                "final_loss": trace.final_loss,
                "best_loss": trace.best_loss,
                **evaluation,
            }
        )

    timing_generator = torch.Generator(device="cpu")
    timing_generator.manual_seed(int(config["timing"]["seed"]))
    timing_batch = generate_transactional_sequence_batch(
        batch_size=timing_batch_size,
        num_entities=int(config["data"]["num_entities"]),
        value_vocab=int(config["data"]["value_vocab"]),
        updates=int(config["data"]["updates"]),
        gap_events=int(config["data"]["gap_events"]),
        generator=timing_generator,
        device=device,
    )
    timing = measure_paired_forward_throughput(
        models=models,
        batch=timing_batch,
        device=device,
        warmup_repeats=timing_warmup,
        measured_repeats=timing_repeats,
    )
    for row in rows:
        row.update(timing[str(row["variant"])])

    e13b_source_config = load_config(
        str(config["feasibility"]["source_config_path"])
    )
    if args.dry_run:
        e13b_source_config = {
            **e13b_source_config,
            "data": {
                **e13b_source_config["data"],
                "num_entities": 8,
                "value_vocab": 16,
            },
            "model": {
                **e13b_source_config["model"],
                "embedding_dim": 16,
                "hidden_dim": 32,
            },
            "training": {
                **e13b_source_config["training"],
                "steps": 2,
                "batch_size": 2,
                "updates": 1,
                "gap_events": 0,
            },
        }
    feasibility_rows, projected_schedule = (
        measure_e13b_scale_training_feasibility(
            source_config=e13b_source_config,
            device=device,
            seed=int(config["feasibility"]["seed"]),
            warmup_steps=feasibility_warmup,
            measured_steps=feasibility_steps,
            planned_wave_jobs=planned_wave_jobs,
        )
    )
    if not args.dry_run:
        projected_schedule = project_e13b_schedule(
            examples_per_second={
                variant: float(row["training_examples_per_second"])
                for variant, row in feasibility_rows.items()
            },
            steps=int(e13b_source_config["training"]["steps"]),
            batch_size=int(e13b_source_config["training"]["batch_size"]),
            planned_wave_jobs=planned_wave_jobs,
        )

    row_by_variant = {str(row["variant"]): row for row in rows}
    tied = row_by_variant[SequenceControl.TIED.value]
    dual = row_by_variant[SequenceControl.DUAL.value]
    paired_initialization = len(set(initialization_hashes.values())) == 1
    matched_parameters = len(set(parameter_counts.values())) == 1
    affected_gain = float(tied["affected_mse"]) - float(dual["affected_mse"])
    retention_sanity = all(
        int(row["unaffected_entity_count"]) > 0
        and float(row["retention_mse"]) >= 0.0
        for row in rows
    )
    retention_ok = (
        retention_sanity
        and max(float(row["retention_mse"]) for row in rows)
        <= float(config["claim_gate"]["max_retention_mse"])
    )
    exact_floor_ok = float(dual["affected_entity_exact_match"]) >= float(
        config["claim_gate"]["minimum_dual_affected_exact_match"]
    )
    mse_floor_ok = float(dual["affected_mse"]) <= float(
        config["claim_gate"]["maximum_dual_affected_mse"]
    )
    gain_ok = affected_gain >= float(
        config["claim_gate"]["minimum_affected_gain"]
    )
    throughput_ok = min(
        float(row["examples_per_second"]) for row in rows
    ) >= float(config["claim_gate"]["minimum_examples_per_second"])
    paired_contract_ok = paired_initialization and matched_parameters
    feasibility_ok = bool(
        float(projected_schedule["projected_slowest_run_seconds"])
        <= float(config["feasibility"]["maximum_projected_seconds_per_run"])
        and max(
            float(value)
            for value in projected_schedule["projected_seconds_per_wave"]
        )
        <= float(config["feasibility"]["maximum_projected_seconds_per_wave"])
        and float(projected_schedule["projected_total_sequential_wave_seconds"])
        <= float(
            config["feasibility"]["maximum_projected_seconds_all_waves"]
        )
    )
    go = bool(
        paired_contract_ok
        and exact_floor_ok
        and mse_floor_ok
        and gain_ok
        and retention_ok
        and throughput_ok
        and feasibility_ok
        and not args.dry_run
    )
    report = {
        "status": "DRY_RUN" if args.dry_run else "PASS",
        "run_scope": "SEQUENCE_BRIDGE_CALIBRATION_REPAIR",
        "protocol_disposition": {
            "original_e13a_immutable": True,
            "original_e13a_used_as_confirmatory_dependency": False,
            "repair_is_prospective": True,
        },
        "provenance": {
            "config_path": str(args.config),
            "config_file_sha256": file_sha256(args.config),
            "resolved_config_canonical_sha256": _canonical_sha256(config),
            "checkpoint_hashes": {
                str(row["variant"]): str(row["checkpoint_sha256"])
                for row in rows
            },
        },
        "paired_contract": {
            "training_data_seed": paired_seed,
            "initialization_hashes": initialization_hashes,
            "identical_initialization": paired_initialization,
            "parameter_counts": parameter_counts,
            "matched_parameter_count": matched_parameters,
            "evaluation_seed": int(config["evaluation"]["seed"]),
            "timing_seed": int(config["timing"]["seed"]),
            "timing_excludes_training_and_data_generation": True,
            "timing_uses_paired_alternating_order": True,
            "e13b_scale_probe_seed": int(config["feasibility"]["seed"]),
        },
        "e13b_scale_feasibility": {
            "source_config_path": str(
                config["feasibility"]["source_config_path"]
            ),
            "source_config_file_sha256": file_sha256(
                str(config["feasibility"]["source_config_path"])
            ),
            "source_config_canonical_sha256": _canonical_sha256(
                e13b_source_config
            ),
            "measurements": feasibility_rows,
            "projected_schedule": projected_schedule,
            "maximum_projected_seconds_per_run": float(
                config["feasibility"]["maximum_projected_seconds_per_run"]
            ),
            "maximum_projected_seconds_per_wave": float(
                config["feasibility"]["maximum_projected_seconds_per_wave"]
            ),
            "maximum_projected_seconds_all_waves": float(
                config["feasibility"][
                    "maximum_projected_seconds_all_waves"
                ]
            ),
            "passed": feasibility_ok,
        },
        "summary": {
            "affected_gain_tied_minus_dual": affected_gain,
            "dual_affected_entity_exact_match": float(
                dual["affected_entity_exact_match"]
            ),
            "dual_affected_mse": float(dual["affected_mse"]),
            "paired_contract_ok": paired_contract_ok,
            "exact_floor_ok": exact_floor_ok,
            "mse_floor_ok": mse_floor_ok,
            "gain_ok": gain_ok,
            "retention_sanity_ok": retention_sanity,
            "retention_ok": retention_ok,
            "throughput_ok": throughput_ok,
            "e13b_scale_feasibility_ok": feasibility_ok,
        },
        "claim_gate": {
            "go_for_e13b": go,
            "allowed_claim": (
                "Prospective paired calibration only; no confirmatory "
                "sequence-memory claim."
            ),
            "forbidden_claim": (
                "Pretrained recurrent-language-model transfer or use of the "
                "original E13a pilot as the E13b dependency."
            ),
        },
    }
    write_jsonl(run_dir / "sequence_calibration_repair_metrics.jsonl", rows)
    write_jsonl(
        run_dir / "e13b_scale_feasibility_metrics.jsonl",
        [
            {"variant": variant, **measurement}
            for variant, measurement in feasibility_rows.items()
        ],
    )
    finalize_run(
        experiment_id=EXPERIMENT_ID,
        artifact_root=args.artifact_root,
        run_dir=run_dir,
        report=report,
    )
    print(f"[{EXPERIMENT_ID}] {report['status']}: {run_dir}")


if __name__ == "__main__":
    main()
