from __future__ import annotations

from collections import defaultdict

import torch

from catena.core.io import file_sha256, write_jsonl
from catena.data.learned_rank import (
    best_rank_errors,
    make_low_rank_family,
    sample_descriptors,
)
from catena.eval.postcore_metrics import exact_sign_flip, monotonic_fraction
from catena.eval.rank_scaling import (
    aggregate_intrinsic_rank_effects_by_seed,
    evaluate_minimum_rank_tracking,
    minimum_sufficient_rank_from_exact_target_recovery,
    oracle_normalized_rank_recovery,
    rank_cell_seed_provenance,
)
from catena.models.operator_controllers import LowRankOperatorController, parameter_count
from catena.training.postcore import evaluate_matrix_controller, train_matrix_controller
from experiments.common import build_parser, finalize_run, initialize_run

EXPERIMENT_ID = "e10_learned_rank_scaling"
DEFAULT_CONFIG = "configs/e10_learned_rank_scaling.yaml"


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
    seeds = [int(value) for value in config["seeds"]]
    intrinsic_ranks = [int(value) for value in config["data"]["intrinsic_ranks"]]
    learned_ranks = [int(value) for value in config["model"]["learned_ranks"]]
    steps = int(config["training"]["steps"])
    if args.dry_run:
        seeds = seeds[:1]
        intrinsic_ranks = intrinsic_ranks[:2]
        learned_ranks = learned_ranks[:3]
        steps = min(steps, 12)

    dimension = int(config["data"]["dimension"])
    descriptor_dim = int(config["data"]["descriptor_dim"])
    train_count = int(config["data"]["train_count"])
    test_count = int(config["data"]["test_count"])
    hidden_dim = int(config["model"]["hidden_dim"])
    rows: list[dict[str, float | int | str | None]] = []
    by_seed_intrinsic: dict[
        tuple[int, int],
        list[dict[str, float | int | str | None]],
    ] = defaultdict(list)

    for seed in seeds:
        for intrinsic_rank in intrinsic_ranks:
            seed_provenance = rank_cell_seed_provenance(seed, intrinsic_rank)
            family = make_low_rank_family(
                dimension=dimension,
                descriptor_dim=descriptor_dim,
                intrinsic_rank=intrinsic_rank,
                seed=seed_provenance["family_seed"],
            )
            train_descriptor = sample_descriptors(
                count=train_count,
                descriptor_dim=descriptor_dim,
                seed=seed_provenance["train_descriptor_seed"],
            )
            test_descriptor = sample_descriptors(
                count=test_count,
                descriptor_dim=descriptor_dim,
                seed=seed_provenance["test_descriptor_seed"],
            )
            train_target = family.operator(train_descriptor)
            test_target = family.operator(test_descriptor)
            baseline_error = float(test_target.square().mean())
            oracle_ranks = list(dict.fromkeys([*learned_ranks, intrinsic_rank]))
            oracle_by_rank = best_rank_errors(test_target, oracle_ranks)
            oracle_means = {
                rank: float(per_example.mean())
                for rank, per_example in oracle_by_rank.items()
            }
            ordered_oracle = [oracle_means[rank] for rank in sorted(set(oracle_ranks))]
            if any(
                right > left + 1e-8
                for left, right in zip(ordered_oracle[:-1], ordered_oracle[1:], strict=True)
            ):
                raise RuntimeError("best-rank oracle error must be non-increasing with rank")
            exact_target_oracle_error = oracle_means[intrinsic_rank]

            for learned_rank in learned_ranks:
                torch.manual_seed(seed_provenance["model_seed"])
                model = LowRankOperatorController(
                    descriptor_dim=descriptor_dim,
                    dimension=dimension,
                    rank=learned_rank,
                    hidden_dim=hidden_dim,
                )
                trace = train_matrix_controller(
                    model=model,
                    descriptors=train_descriptor,
                    targets=train_target,
                    steps=steps,
                    batch_size=int(config["training"]["batch_size"]),
                    learning_rate=float(config["training"]["learning_rate"]),
                    weight_decay=float(config["training"]["weight_decay"]),
                    device=device,
                    seed=seed_provenance["optimizer_sampling_seed"],
                )
                test_error, per_example = evaluate_matrix_controller(
                    model=model,
                    descriptors=test_descriptor,
                    targets=test_target,
                    device=device,
                    batch_size=int(config["evaluation"]["batch_size"]),
                )
                checkpoint = (
                    run_dir
                    / "checkpoints"
                    / (
                        f"seed{seed}_intrinsic{intrinsic_rank}_"
                        f"learned{learned_rank}.pt"
                    )
                )
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model": model.state_dict(),
                        "seed": seed,
                        "intrinsic_rank": intrinsic_rank,
                        "learned_rank": learned_rank,
                        "config": config,
                    },
                    checkpoint,
                )
                oracle = oracle_by_rank[learned_rank]
                oracle_error = float(oracle.mean())
                reachable_floor_recovery = oracle_normalized_rank_recovery(
                    baseline_error=baseline_error,
                    model_error=test_error,
                    oracle_error=oracle_error,
                )
                exact_target_recovery = oracle_normalized_rank_recovery(
                    baseline_error=baseline_error,
                    model_error=test_error,
                    oracle_error=exact_target_oracle_error,
                )
                row: dict[str, float | int | str | None] = {
                    "seed": seed,
                    "intrinsic_rank": intrinsic_rank,
                    "learned_rank": learned_rank,
                    "test_error": test_error,
                    "oracle_error": oracle_error,
                    "exact_target_oracle_error": exact_target_oracle_error,
                    "baseline_error": baseline_error,
                    "reachable_floor_recovery": reachable_floor_recovery,
                    "exact_target_recovery": exact_target_recovery,
                    "excess_over_oracle": test_error - oracle_error,
                    "per_example_error_std": float(per_example.std(unbiased=False)),
                    "parameter_count": parameter_count(model),
                    "initial_loss": trace.initial_loss,
                    "final_loss": trace.final_loss,
                    "best_loss": trace.best_loss,
                    "checkpoint": str(checkpoint.resolve()),
                    "checkpoint_sha256": file_sha256(checkpoint),
                    **seed_provenance,
                    "model_initialization_pairing": (
                        "same_rng_base_and_identical_shared_backbone_tensors_within_"
                        "seed_intrinsic_cell"
                    ),
                    "training_schedule_pairing": (
                        "same_minibatch_index_stream_within_seed_intrinsic_cell"
                    ),
                    "oracle_method": "single_batched_svdvals_reused_across_learned_ranks",
                    "oracle_normalization": "squared_frobenius_per_matrix_entry",
                }
                rows.append(row)
                by_seed_intrinsic[(seed, intrinsic_rank)].append(row)

    quality_threshold = float(config["claim_gate"]["oracle_normalized_recovery"])
    max_rank_factor = float(config["claim_gate"]["max_rank_factor"])
    monotonic_threshold = float(config["claim_gate"]["monotonic_fraction"])
    monotonic_scores: list[float] = []
    cell_effects: dict[tuple[int, int], float] = {}
    minimum_qualifying_ranks: dict[tuple[int, int], int | None] = {}
    for (seed, intrinsic_rank), group in sorted(by_seed_intrinsic.items()):
        group = sorted(group, key=lambda item: int(item["learned_rank"]))
        errors = [float(item["test_error"]) for item in group]
        monotonic_scores.append(monotonic_fraction(errors, decreasing=True))
        exact_target_recoveries = {
            int(item["learned_rank"]): float(item["exact_target_recovery"])
            for item in group
        }
        minimum = minimum_sufficient_rank_from_exact_target_recovery(
            exact_target_recoveries,
            threshold=quality_threshold,
        )
        minimum_qualifying_ranks[(seed, intrinsic_rank)] = minimum
        low = min(group, key=lambda item: int(item["learned_rank"]))
        high = max(group, key=lambda item: int(item["learned_rank"]))
        cell_effects[(seed, intrinsic_rank)] = float(low["test_error"]) - float(
            high["test_error"]
        )

    rank_tracking_cell_rows, rank_tracking_seed_rows = evaluate_minimum_rank_tracking(
        minimum_qualifying_ranks,
        seeds=seeds,
        intrinsic_ranks=intrinsic_ranks,
        max_rank_factor=max_rank_factor,
        max_available_rank=max(learned_ranks),
    )
    rank_matches = [
        float(bool(item["rank_tracking_matched"]))
        for item in rank_tracking_cell_rows
    ]
    mean_rank_match = sum(rank_matches) / max(len(rank_matches), 1)
    mean_monotonic = sum(monotonic_scores) / max(len(monotonic_scores), 1)
    seedwise_rank_nondecreasing = all(
        bool(item["minimum_qualifying_rank_nondecreasing"])
        for item in rank_tracking_seed_rows
    )
    seedwise_rank_nondecreasing_fraction = sum(
        float(bool(item["minimum_qualifying_rank_nondecreasing"]))
        for item in rank_tracking_seed_rows
    ) / len(rank_tracking_seed_rows)
    seed_effect_rows = aggregate_intrinsic_rank_effects_by_seed(
        cell_effects,
        seeds=seeds,
        intrinsic_ranks=intrinsic_ranks,
    )
    seed_effects = [
        float(item["mean_low_vs_high_rank_gain"]) for item in seed_effect_rows
    ]
    sign_flip_p = exact_sign_flip(seed_effects, alternative="greater")
    supported = (
        mean_rank_match >= float(config["claim_gate"]["minimum_rank_match_fraction"])
        and seedwise_rank_nondecreasing
        and mean_monotonic >= monotonic_threshold
        and sign_flip_p <= float(config["statistics"]["alpha"])
    )
    report = {
        "status": "PASS",
        "run_scope": "CONTROLLED_REFERENCE",
        "summary": {
            "rank_match_fraction": mean_rank_match,
            "rank_match_definition": (
                "intrinsic_rank <= minimum_qualifying_rank <= "
                "min(max_rank_factor × intrinsic_rank, maximum_registered_learned_rank)"
            ),
            "seedwise_minimum_rank_nondecreasing_fraction": (
                seedwise_rank_nondecreasing_fraction
            ),
            "mean_monotonic_fraction": mean_monotonic,
            "low_vs_high_rank_sign_flip_p": sign_flip_p,
            "low_vs_high_rank_seed_mean_gain": sum(seed_effects) / len(seed_effects),
            "statistical_unit": "training_seed",
            "statistical_unit_count": len(seed_effects),
            "intrinsic_rank_cells_per_seed": len(intrinsic_ranks),
            "rows": len(rows),
        },
        "provenance": {
            "seed_effect_aggregation": (
                "equal_weight_mean_of_registered_intrinsic_rank_cell_gains_within_seed"
            ),
            "oracle_computation": (
                "one CPU batched singular-value decomposition per seed × intrinsic-rank "
                "test target, reused across all registered learned ranks"
            ),
            "oracle_estimand": "per-example normalized Frobenius best-rank approximation error",
            "recovery_metric_roles": {
                "reachable_floor_recovery": (
                    "optimization diagnostic relative to each learned rank's own "
                    "best-rank reachable floor"
                ),
                "exact_target_recovery": (
                    "minimum-sufficient-rank qualification metric relative to the "
                    "intrinsic-rank exact-target oracle"
                ),
                "registered_threshold": quality_threshold,
            },
            "model_initialization_pairing": (
                "shared backbone initialized identically across learned ranks within each "
                "seed × intrinsic-rank cell; rank-specific heads retain natural shapes"
            ),
            "training_schedule_pairing": (
                "identical sampled-example order across learned ranks within each "
                "seed × intrinsic-rank cell"
            ),
            "descriptive_cell_count": len(rank_matches),
            "inferential_seed_count": len(seed_effects),
        },
        "protocol_repairs": [
            {
                "id": "prospective_pre_evaluation_gate_identifiability_repair",
                "timing": "before_any_evaluable_e10_main_report",
                "reason": (
                    "The original upper-bound-only rank gate could pass if rank 1 "
                    "qualified for every intrinsic-rank family."
                ),
                "repair": (
                    "Require the minimum qualifying rank to lie at or above intrinsic "
                    "rank and at or below the pre-existing max-rank-factor bound, and "
                    "require seedwise nondecreasing minimum qualifying rank. Separate "
                    "best-rank reachable-floor recovery (optimization diagnostic) from "
                    "exact-target recovery (minimum-rank qualification)."
                ),
                "thresholds_or_rank_grids_changed": False,
                "incomplete_prior_manifests_claim_eligible": False,
            }
        ],
        "claim_gate": {
            "supported": supported,
            "conditions": {
                "rank_tracking_fraction_passed": (
                    mean_rank_match
                    >= float(config["claim_gate"]["minimum_rank_match_fraction"])
                ),
                "seedwise_minimum_rank_nondecreasing_passed": (
                    seedwise_rank_nondecreasing
                ),
                "learned_rank_error_monotonicity_passed": (
                    mean_monotonic >= monotonic_threshold
                ),
                "seed_level_low_vs_high_rank_sign_flip_passed": (
                    sign_flip_p <= float(config["statistics"]["alpha"])
                ),
            },
            "allowed_claim": (
                "In the registered smooth operator families, the minimum learned "
                "control rank needed to recover the exact target tracks intrinsic rank, "
                "while each controller is evaluated against its best-rank reachable floor."
            ),
            "forbidden_claim": (
                "Demand algebra universally determines rank in pretrained language models."
            ),
        },
    }
    write_jsonl(run_dir / "rank_scaling_metrics.jsonl", rows)
    write_jsonl(run_dir / "rank_scaling_seed_effects.jsonl", seed_effect_rows)
    write_jsonl(
        run_dir / "rank_scaling_cell_tracking.jsonl",
        rank_tracking_cell_rows,
    )
    write_jsonl(
        run_dir / "rank_scaling_seed_tracking.jsonl",
        rank_tracking_seed_rows,
    )
    finalize_run(
        experiment_id=EXPERIMENT_ID,
        artifact_root=args.artifact_root,
        run_dir=run_dir,
        report=report,
    )
    print(f"[{EXPERIMENT_ID}] PASS: {run_dir}")


if __name__ == "__main__":
    main()
