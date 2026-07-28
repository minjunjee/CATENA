from __future__ import annotations

import csv
import random
from collections import defaultdict
from typing import Any

import numpy as np
import torch

from catena.core.io import write_jsonl
from catena.core.randomness import seed_everything
from catena.core.schema import CandidateMode, Operation
from catena.data.semantic_transactions import (
    SemanticTransaction,
    generate_semantic_transactions,
)
from catena.data.tamp import TAMPConfig, build_episode
from catena.eval.metrics import evaluate_episode
from catena.models.memory import GateOutput, apply_scalar_update
from catena.models.semantic_controllers import (
    MatchedSemanticController,
    SemanticConstraint,
)
from catena.training.semantic_probe import (
    SemanticProbeConfig,
    SemanticProbeExample,
    evaluate_semantic_probe,
    semantic_probe_input_dim,
    train_semantic_probe,
)
from experiments.common import build_parser, finalize_run, initialize_run

EXPERIMENT_ID = "e05_semantic_demand_inference"
DEFAULT_CONFIG = "configs/e05_semantic_demand_inference.yaml"


def _examples(
    count: int,
    seed: int,
    domains: list[str],
    operations: list[Operation],
    styles: list[str],
    tamp: TAMPConfig,
    hide_old: bool,
) -> list[SemanticProbeExample]:
    transactions = generate_semantic_transactions(
        count=count,
        seed=seed,
        domains=domains,
        operations=operations,
        styles=styles,
    )
    result: list[SemanticProbeExample] = []
    for index, transaction in enumerate(transactions):
        episode = build_episode(
            seed=seed * 100000 + index,
            operation=transaction.operation,
            candidate_mode=CandidateMode.ORACLE,
            config=tamp,
            episode_index=index,
        )
        text = (
            transaction.text.replace(transaction.old_value, "the prior value")
            if hide_old
            else transaction.text
        )
        redacted = SemanticTransaction(
            transaction.transaction_id,
            transaction.domain,
            transaction.operation,
            text,
            transaction.entity,
            transaction.old_value,
            transaction.new_value,
            transaction.template_family,
            transaction.surface_style,
        )
        result.append(SemanticProbeExample(redacted, episode))
    return result


def _rows(
    model_name: str,
    split: str,
    outputs: list[torch.Tensor],
    examples: list[SemanticProbeExample],
    seed: int,
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for output, example in zip(outputs, examples, strict=True):
        result.append(
            {
                "seed": seed,
                "model": model_name,
                "split": split,
                "episode_id": example.episode.episode_id,
                "operation": example.episode.operation.value,
                "domain": example.transaction.domain,
                "surface_style": example.transaction.surface_style,
                **evaluate_episode(output, example.episode).to_dict(),
            }
        )
    return result


def _shuffled(
    examples: list[SemanticProbeExample], seed: int
) -> list[SemanticProbeExample]:
    rng = random.Random(seed)
    transactions = [example.transaction for example in examples]
    rng.shuffle(transactions)
    return [
        SemanticProbeExample(transaction, example.episode)
        for transaction, example in zip(transactions, examples, strict=True)
    ]


def _oracle_outputs(examples: list[SemanticProbeExample]) -> list[torch.Tensor]:
    outputs = []
    for example in examples:
        erase, write = example.episode.operation.demand
        outputs.append(
            apply_scalar_update(
                example.episode,
                torch.tensor(erase, dtype=example.episode.state.dtype),
                torch.tensor(write, dtype=example.episode.state.dtype),
            )
        )
    return outputs


def _stratified_audit_sample(
    candidates: list[tuple[str, SemanticProbeExample]], size: int, seed: int
) -> list[tuple[str, SemanticProbeExample]]:
    by_split: dict[str, list[SemanticProbeExample]] = defaultdict(list)
    for split, example in candidates:
        by_split[split].append(example)
    rng = random.Random(seed)
    selected: list[tuple[str, SemanticProbeExample]] = []
    splits = sorted(by_split)
    quota = max(size // max(len(splits), 1), 1)
    for split in splits:
        pool = list(by_split[split])
        rng.shuffle(pool)
        selected.extend((split, example) for example in pool[:quota])
    if len(selected) < size:
        selected_ids = {example.episode.episode_id for _, example in selected}
        remainder = [
            item
            for item in candidates
            if item[1].episode.episode_id not in selected_ids
        ]
        rng.shuffle(remainder)
        selected.extend(remainder[: size - len(selected)])
    return selected[:size]


def main() -> None:
    parser = build_parser(EXPERIMENT_ID, DEFAULT_CONFIG)
    args = parser.parse_args()
    config, run_dir, device = initialize_run(
        experiment_id=EXPERIMENT_ID,
        config_path=args.config,
        artifact_root=args.artifact_root,
        device_request=args.device,
    )
    seeds = [int(value) for value in config["seeds"]]
    train_count = int(config["data"]["train_count"])
    test_count = int(config["data"]["test_count"])
    steps = int(config["training"]["steps"])
    if args.dry_run:
        seeds = seeds[:1]
        train_count = min(train_count, 64)
        test_count = min(test_count, 24)
        steps = min(steps, 60)

    tamp = TAMPConfig(**config["tamp"])
    probe = SemanticProbeConfig(
        int(config["model"]["bow_dim"]),
        int(config["model"]["hidden_dim"]),
        bool(config["model"]["include_state_read"]),
        steps,
        float(config["training"]["learning_rate"]),
    )
    input_dim = semantic_probe_input_dim(probe, tamp.value_dim)
    rows: list[dict[str, object]] = []
    primary: dict[int, float] = {}
    summary: dict[str, list[dict[str, float | int]]] = {}
    audit_candidates: list[tuple[str, SemanticProbeExample]] = []

    for seed_index, seed in enumerate(seeds):
        seed_everything(seed)
        train = _examples(
            train_count,
            seed,
            list(config["data"]["train_domains"]),
            [Operation.PRESERVE, Operation.ADD, Operation.INVALIDATE],
            list(config["data"]["train_styles"]),
            tamp,
            bool(config["data"]["hide_old_value"]),
        )
        splits = {
            "heldout_supersede_seen_domain": _examples(
                test_count,
                seed + 100,
                list(config["data"]["train_domains"]),
                [Operation.SUPERSEDE],
                list(config["data"]["train_styles"]),
                tamp,
                bool(config["data"]["hide_old_value"]),
            ),
            "heldout_paraphrase": _examples(
                test_count,
                seed + 200,
                list(config["data"]["train_domains"]),
                [Operation.PRESERVE, Operation.ADD, Operation.INVALIDATE],
                ["paraphrase"],
                tamp,
                bool(config["data"]["hide_old_value"]),
            ),
            "heldout_domain": _examples(
                test_count,
                seed + 300,
                [str(config["data"]["heldout_domain"])],
                [Operation.PRESERVE, Operation.ADD, Operation.INVALIDATE],
                list(config["data"]["train_styles"]),
                tamp,
                bool(config["data"]["hide_old_value"]),
            ),
            "combined_stress": _examples(
                test_count,
                seed + 400,
                [str(config["data"]["heldout_domain"])],
                [Operation.SUPERSEDE],
                ["paraphrase"],
                tamp,
                bool(config["data"]["hide_old_value"]),
            ),
        }
        if seed_index == 0:
            for split, examples in splits.items():
                audit_candidates.extend((split, example) for example in examples)

        models = {
            constraint: MatchedSemanticController(input_dim, probe.hidden_dim, constraint)
            for constraint in (SemanticConstraint.FACTORIZED, SemanticConstraint.SHARED)
        }
        parameter_counts = {
            constraint.value: sum(parameter.numel() for parameter in model.parameters())
            for constraint, model in models.items()
        }
        if len(set(parameter_counts.values())) != 1:
            raise AssertionError("Semantic controllers must be parameter matched.")
        initial_state = models[SemanticConstraint.FACTORIZED].state_dict()
        models[SemanticConstraint.SHARED].load_state_dict(initial_state)
        for model in models.values():
            train_semantic_probe(
                model=model, examples=train, config=probe, device=device
            )

        for split, examples in splits.items():
            split_errors: dict[str, float] = {}
            for constraint, model in models.items():
                outputs = evaluate_semantic_probe(
                    model=model, examples=examples, config=probe, device=device
                )
                model_rows = _rows(
                    constraint.value, split, outputs, examples, seed
                )
                rows.extend(model_rows)
                split_errors[constraint.value] = float(
                    np.mean([float(row["affected_read_mse"]) for row in model_rows])
                )

            shuffled_examples = _shuffled(examples, seed + 999)
            shuffled_outputs = evaluate_semantic_probe(
                model=models[SemanticConstraint.FACTORIZED],
                examples=shuffled_examples,
                config=probe,
                device=device,
            )
            rows.extend(
                _rows(
                    "factorized_shuffled_text",
                    split,
                    shuffled_outputs,
                    examples,
                    seed,
                )
            )
            oracle_rows = _rows(
                "oracle_demand",
                split,
                _oracle_outputs(examples),
                examples,
                seed,
            )
            rows.extend(oracle_rows)
            split_errors["oracle_demand"] = float(
                np.mean([float(row["affected_read_mse"]) for row in oracle_rows])
            )

            summary.setdefault(split, []).append(
                {
                    "seed": seed,
                    "factorized_mse": split_errors["factorized"],
                    "shared_mse": split_errors["shared"],
                    "oracle_demand_mse": split_errors["oracle_demand"],
                }
            )
            if split == "heldout_supersede_seen_domain":
                primary[seed] = (
                    split_errors["shared"] - split_errors["factorized"]
                )

        for constraint, model in models.items():
            torch.save(model.state_dict(), run_dir / f"seed{seed}_{constraint.value}.pt")

    audit_size = min(int(config["data"]["audit_export_size"]), len(audit_candidates))
    audit_sample = _stratified_audit_sample(
        audit_candidates, audit_size, seed=int(config["data"].get("audit_seed", 2026))
    )
    with (run_dir / "naturalization_audit.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "split",
                "episode_id",
                "domain",
                "operation_hidden_from_model",
                "text",
                "reviewer_a_meaning_preserved",
                "reviewer_a_answer_leakage",
                "reviewer_b_meaning_preserved",
                "reviewer_b_answer_leakage",
                "adjudication_meaning_preserved",
                "adjudication_answer_leakage",
                "notes",
            ]
        )
        for split, example in audit_sample:
            writer.writerow(
                [
                    split,
                    example.episode.episode_id,
                    example.transaction.domain,
                    example.episode.operation.value,
                    example.transaction.text,
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                ]
            )

    primary_values = np.asarray(list(primary.values()))
    report = {
        "status": "PASS",
        "split_summary": summary,
        "primary": {
            "split": "heldout_supersede_seen_domain",
            "seed_effects_shared_minus_factorized": primary,
            "mean": float(primary_values.mean()),
            "exploratory": True,
        },
        "input_controls": {
            "operation_label_provided": False,
            "oracle_address_primary": True,
            "old_value_hidden_in_primary": bool(config["data"]["hide_old_value"]),
            "addressing_and_old_value_recovery_are_not_jointly_claimed": True,
            "oracle_demand_is_upper_bound_only": True,
        },
        "audit": {
            "exported_items": len(audit_sample),
            "stratified_across_splits": True,
            "requires_two_independent_reviewers_and_adjudication": True,
        },
        "claim_gate": {
            "external_validity_anchor_only": True,
            "anchor_direction_positive": float(primary_values.mean()) > 0,
            "semantic_claim_requires_completed_two_reviewer_audit": True,
            "combined_stress_is_not_primary": True,
        },
    }
    write_jsonl(run_dir / "semantic_metrics.jsonl", rows)
    finalize_run(
        experiment_id=EXPERIMENT_ID,
        artifact_root=args.artifact_root,
        run_dir=run_dir,
        report=report,
    )
    print(f"[{EXPERIMENT_ID}] PASS: {run_dir}")


if __name__ == "__main__":
    main()
