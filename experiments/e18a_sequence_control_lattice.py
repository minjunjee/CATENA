from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from catena.core.io import file_sha256, write_jsonl
from catena.data.sequence_control_lattice import (
    SequenceDemandFamily,
    base_sequence_control_digest,
    generate_sequence_control_lattice_batch,
)
from catena.models.sequence_control_lattice import (
    MatchedSequenceControlLattice,
    SequenceControlFreedom,
    sequence_lattice_parameter_count,
)
from catena.training.sequence_control_lattice import (
    evaluate_sequence_control_lattice,
    state_dict_sha256,
    train_sequence_control_lattice,
)
from experiments.common import finalize_run, initialize_run

EXPERIMENT_ID = "e18a_sequence_control_lattice"
DEFAULT_CONFIG = "configs/e18a_sequence_control_lattice.yaml"
PROTOCOL_LOCK = Path("docs/E18_SEQUENCE_CONTROL_LATTICE_LOCK.json")


def build_local_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=EXPERIMENT_ID)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument(
        "--variant",
        choices=[value.value for value in SequenceControlFreedom],
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _read_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def validate_protocol_lock(config_path: str | Path) -> dict[str, str]:
    repo_root = Path(__file__).resolve().parents[1]
    lock_path = PROTOCOL_LOCK.resolve()
    lock = _read_json_object(lock_path)
    if (
        lock.get("schema_version") != 1
        or lock.get("experiment_id") != EXPERIMENT_ID
        or lock.get("evaluation_started") is not False
        or lock.get("protocol_frozen_before_evaluation") is not True
    ):
        raise RuntimeError("E18 protocol lock is invalid or not prospective")
    files = lock.get("files")
    if not isinstance(files, dict):
        raise RuntimeError("E18 protocol lock lacks its file hash map")
    for relative_path, expected_hash in files.items():
        candidate = (repo_root / str(relative_path)).resolve()
        if not candidate.is_file() or file_sha256(candidate) != expected_hash:
            raise RuntimeError(f"E18 locked file changed: {relative_path}")
    relative_config = Path(config_path).resolve().relative_to(
        repo_root
    ).as_posix()
    if files.get(relative_config) != file_sha256(config_path):
        raise RuntimeError("E18a config no longer matches its protocol lock")
    return {
        "path": str(lock_path),
        "sha256": file_sha256(lock_path),
        "config_sha256": file_sha256(config_path),
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


def _structural_distractor_contract(
    *,
    config: dict,
    seed: int,
) -> dict[str, bool | float | str]:
    device = torch.device("cpu")
    updates = 2
    gap_events = 4
    common = {
        "family": SequenceDemandFamily.MAGNITUDE,
        "batch_size": 3,
        "num_entities": int(config["data"]["num_entities"]),
        "value_dim": int(config["data"]["value_dim"]),
        "updates": updates,
        "seed": 70_000 + int(seed),
        "device": device,
    }
    no_gap = generate_sequence_control_lattice_batch(
        **common,
        gap_events=0,
    )
    with_gap = generate_sequence_control_lattice_batch(
        **common,
        gap_events=gap_events,
    )
    digest_paired = (
        base_sequence_control_digest(no_gap)
        == base_sequence_control_digest(with_gap)
    )
    positions = (
        with_gap.update_mask[0]
        .nonzero(as_tuple=False)
        .flatten()
        .tolist()
    )
    expected_positions = [0, gap_events + 1]
    model_input_excludes_update_mask = not hasattr(
        with_gap.inputs,
        "update_mask",
    )
    distractor_verified_zero = bool(
        torch.all(
            with_gap.inputs.demand_features[:, :, -1][~with_gap.update_mask]
            == 0.0
        )
    )

    torch.manual_seed(80_000 + int(seed))
    probe = MatchedSequenceControlLattice(
        freedom=SequenceControlFreedom.STATE_AWARE,
        num_entities=int(config["data"]["num_entities"]),
        value_dim=int(config["data"]["value_dim"]),
        embedding_dim=int(config["model"]["embedding_dim"]),
        hidden_dim=int(config["model"]["hidden_dim"]),
    ).eval()
    with torch.no_grad():
        no_gap_state = probe(no_gap.inputs).state
        with_gap_state = probe(with_gap.inputs).state
    path_delta = float((with_gap_state - no_gap_state).abs().max())
    minimum_delta = float(
        config["claim_gate"]["minimum_unmasked_path_delta"]
    )
    passed = bool(
        digest_paired
        and positions == expected_positions
        and model_input_excludes_update_mask
        and distractor_verified_zero
        and path_delta > minimum_delta
    )
    return {
        "passed": passed,
        "layout": "one_total_block_after_first_verified_update",
        "base_transaction_digest_matched_across_gap": digest_paired,
        "verified_event_positions_matched_contract": (
            positions == expected_positions
        ),
        "model_input_excludes_update_mask": model_input_excludes_update_mask,
        "distractor_verified_is_model_visible_zero": (
            distractor_verified_zero
        ),
        "random_initialization_full_vs_no_gap_max_abs_delta": path_delta,
        "minimum_unmasked_path_delta": minimum_delta,
    }


def _runtime_config(config: dict, *, dry_run: bool) -> dict:
    runtime = deepcopy(config)
    if not dry_run:
        return runtime
    runtime["data"]["num_entities"] = 8
    runtime["data"]["value_dim"] = 8
    runtime["model"]["embedding_dim"] = 16
    runtime["model"]["hidden_dim"] = 32
    runtime["training"]["steps"] = 4
    runtime["training"]["batch_size"] = 2
    runtime["training"]["updates"] = 1
    runtime["training"]["gap_events"] = 4
    runtime["evaluation"]["updates"] = [1]
    runtime["evaluation"]["gap_events"] = [0, 4]
    runtime["evaluation"]["batches"] = 1
    runtime["evaluation"]["batch_size"] = 2
    runtime["evaluation"]["active_path_assay"] = {
        "updates": 1,
        "gap_events": 4,
        "intervention": "activate_distractor_verified_bit",
    }
    return runtime


def main() -> None:
    args = build_local_parser().parse_args()
    lock = validate_protocol_lock(args.config)
    config, run_dir, device = initialize_run(
        experiment_id=EXPERIMENT_ID,
        config_path=args.config,
        artifact_root=args.artifact_root,
        device_request=args.device,
        run_mode="DRY_RUN" if args.dry_run else "MAIN",
    )
    configured_variants = [
        SequenceControlFreedom(value)
        for value in config["model"]["variants"]
    ]
    configured_seeds = [int(value) for value in config["seeds"]]
    if args.dry_run:
        variant = (
            SequenceControlFreedom(args.variant)
            if args.variant is not None
            else configured_variants[0]
        )
        seed = args.seed if args.seed is not None else configured_seeds[0]
    else:
        if args.variant is None or args.seed is None:
            raise ValueError("MAIN E18a requires one explicit --variant and --seed")
        variant = SequenceControlFreedom(args.variant)
        seed = int(args.seed)
    if variant not in configured_variants or seed not in configured_seeds:
        raise ValueError("variant/seed is outside the registered E18a grid")

    runtime = _runtime_config(config, dry_run=args.dry_run)
    families = [
        SequenceDemandFamily(value)
        for value in runtime["data"]["families"]
    ]
    structural_contract = _structural_distractor_contract(
        config=runtime,
        seed=seed,
    )
    if not structural_contract["passed"]:
        raise RuntimeError("E18 model-visible distractor contract failed")

    torch.manual_seed(10_000 + seed)
    model = MatchedSequenceControlLattice(
        freedom=variant,
        num_entities=int(runtime["data"]["num_entities"]),
        value_dim=int(runtime["data"]["value_dim"]),
        embedding_dim=int(runtime["model"]["embedding_dim"]),
        hidden_dim=int(runtime["model"]["hidden_dim"]),
    )
    initialization_sha256 = state_dict_sha256(model.state_dict())
    trace = train_sequence_control_lattice(
        model=model,
        families=families,
        steps=int(runtime["training"]["steps"]),
        batch_size=int(runtime["training"]["batch_size"]),
        num_entities=int(runtime["data"]["num_entities"]),
        value_dim=int(runtime["data"]["value_dim"]),
        updates=int(runtime["training"]["updates"]),
        gap_events=int(runtime["training"]["gap_events"]),
        learning_rate=float(runtime["training"]["learning_rate"]),
        retention_weight=float(runtime["training"]["retention_weight"]),
        device=device,
        seed=20_000 + seed,
    )
    checkpoint = (
        run_dir / "checkpoints" / f"{variant.value}_seed{seed}.pt"
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "model_class": "MatchedSequenceControlLattice",
            "variant": variant.value,
            "seed": seed,
            "initialization_sha256": initialization_sha256,
            "protocol_lock_sha256": lock["sha256"],
            "config": runtime,
        },
        checkpoint,
    )
    checkpoint_sha256 = file_sha256(checkpoint)

    eval_updates = [int(value) for value in runtime["evaluation"]["updates"]]
    eval_gaps = [int(value) for value in runtime["evaluation"]["gap_events"]]
    assay_updates = int(
        runtime["evaluation"]["active_path_assay"]["updates"]
    )
    assay_gap = int(
        runtime["evaluation"]["active_path_assay"]["gap_events"]
    )
    rows: list[dict[str, float | int | str | bool]] = []
    for family_index, family in enumerate(families):
        for updates in eval_updates:
            evaluation_seed = (
                100_000 + 10_000 * seed + 100 * family_index + updates
            )
            for gap_events in eval_gaps:
                metrics = evaluate_sequence_control_lattice(
                    model=model,
                    family=family,
                    batches=int(runtime["evaluation"]["batches"]),
                    batch_size=int(runtime["evaluation"]["batch_size"]),
                    num_entities=int(runtime["data"]["num_entities"]),
                    value_dim=int(runtime["data"]["value_dim"]),
                    updates=updates,
                    gap_events=gap_events,
                    device=device,
                    seed=evaluation_seed,
                )
                row: dict[str, float | int | str | bool] = {
                    "variant": variant.value,
                    "seed": seed,
                    "demand_family": family.value,
                    "updates": updates,
                    "gap_events": gap_events,
                    "evaluation_seed": evaluation_seed,
                    "checkpoint": str(checkpoint.resolve()),
                    "checkpoint_sha256": checkpoint_sha256,
                    "initialization_sha256": initialization_sha256,
                    "protocol_lock_sha256": lock["sha256"],
                    "parameter_count": sequence_lattice_parameter_count(model),
                    "train_final_loss": trace.final_loss,
                    "train_best_loss": trace.best_loss,
                    "examples_per_second": trace.examples_per_second,
                    "peak_memory_bytes": trace.peak_memory_bytes,
                    "optimizer": trace.optimizer,
                    **metrics,
                }
                if updates == assay_updates and gap_events == assay_gap:
                    activated = evaluate_sequence_control_lattice(
                        model=model,
                        family=family,
                        batches=int(runtime["evaluation"]["batches"]),
                        batch_size=int(runtime["evaluation"]["batch_size"]),
                        num_entities=int(runtime["data"]["num_entities"]),
                        value_dim=int(runtime["data"]["value_dim"]),
                        updates=updates,
                        gap_events=gap_events,
                        device=device,
                        seed=evaluation_seed,
                        activate_distractor_verified=True,
                    )
                    if (
                        activated["base_transaction_digest"]
                        != metrics["base_transaction_digest"]
                    ):
                        raise RuntimeError(
                            "E18 active-path assay changed the base transaction"
                        )
                    row.update(
                        _prefixed_metrics(
                            "distractor_activation",
                            activated,
                        )
                    )
                    row["distractor_activation_retention_harm"] = float(
                        activated["retention_mse"]
                    ) - float(metrics["retention_mse"])
                rows.append(row)

    digest_sets: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in rows:
        digest_sets[
            (str(row["demand_family"]), int(row["updates"]))
        ].add(str(row["base_transaction_digest"]))
    base_digest_paired = all(len(values) == 1 for values in digest_sets.values())
    if not base_digest_paired:
        raise RuntimeError("E18 base transaction changed across gap conditions")

    expected_rows = len(families) * len(eval_updates) * len(eval_gaps)
    report = {
        "status": "DRY_RUN" if args.dry_run else "PASS",
        "run_scope": "SEQUENCE_CONTROL_ARCHITECTURE_DEMAND_LATTICE",
        "run_mode": "DRY_RUN" if args.dry_run else "MAIN",
        "variant": variant.value,
        "seed": seed,
        "rows": len(rows),
        "expected_rows": expected_rows,
        "protocol_lock": lock,
        "paired_contract": {
            "maximal_parameter_surface": True,
            "shared_encoder_and_state_size": True,
            "training_data_order_seed": 20_000 + seed,
            "initialization_sha256": initialization_sha256,
            "parameter_count": sequence_lattice_parameter_count(model),
            "optimizer": trace.optimizer,
            "base_transaction_digest_paired_across_gaps": base_digest_paired,
        },
        "distractor_path_contract": structural_contract,
        "claim_gate": {
            "status": "DRY_RUN" if args.dry_run else "PENDING_AGGREGATE",
            "allowed_claim": (
                "Per-run controlled sequence-lattice evidence only after "
                "E18b five-seed paired aggregation."
            ),
            "forbidden_claim": (
                "Natural-language, learned-candidate/address, recurrent-LM, "
                "agent, planning, or official-backend transfer."
            ),
        },
        "evidence_tier": "CONTROLLED_REFERENCE",
        "scientific_evidence": False,
    }
    write_jsonl(run_dir / "sequence_control_lattice_metrics.jsonl", rows)
    finalize_run(
        experiment_id=EXPERIMENT_ID,
        artifact_root=args.artifact_root,
        run_dir=run_dir,
        report=report,
    )
    print(f"[{EXPERIMENT_ID}] {report['status']}: {run_dir}")


if __name__ == "__main__":
    main()
