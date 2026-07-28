from __future__ import annotations

from typing import Any

import torch

from catena.core.io import file_sha256, write_jsonl
from catena.data.representation_dynamics import (
    RepresentationFamily,
    make_representation_generator,
    sample_representation_descriptors,
)
from catena.eval.postcore_metrics import exact_sign_flip
from catena.models.coadaptation import (
    FixedBasisDiagonalController,
    LearnedBasisDiagonalController,
    LowRankCoadaptationController,
)
from catena.models.operator_controllers import parameter_count
from catena.training.postcore import evaluate_matrix_controller, train_matrix_controller
from experiments.common import build_parser, finalize_run, initialize_run

EXPERIMENT_ID = "e11b_scale_normalized_coadaptation"
DEFAULT_CONFIG = "configs/e11b_scale_normalized_coadaptation.yaml"


def _make_model(
    kind: str,
    *,
    descriptor_dim: int,
    dimension: int,
    hidden_dim: int,
    low_rank: int,
) -> torch.nn.Module:
    if kind == "fixed_diagonal":
        return FixedBasisDiagonalController(
            descriptor_dim=descriptor_dim,
            dimension=dimension,
            hidden_dim=hidden_dim,
        )
    if kind == "learned_basis_diagonal":
        return LearnedBasisDiagonalController(
            descriptor_dim=descriptor_dim,
            dimension=dimension,
            hidden_dim=hidden_dim,
        )
    if kind == "low_rank":
        return LowRankCoadaptationController(
            descriptor_dim=descriptor_dim,
            dimension=dimension,
            rank=low_rank,
            hidden_dim=hidden_dim,
        )
    raise ValueError(f"Unknown controller kind: {kind}")


def normalized_seed_contrasts(
    family_errors: dict[tuple[str, str], float],
    target_energy: dict[str, float],
) -> dict[str, float]:
    eps = 1e-12
    axis = RepresentationFamily.AXIS_COMMUTING.value
    common = RepresentationFamily.COMMON_ROTATED_COMMUTING.value
    noncommuting = RepresentationFamily.NONCOMMUTING.value
    axis_fixed = family_errors[(axis, "fixed_diagonal")]
    axis_shared = family_errors[(axis, "learned_basis_diagonal")]
    common_fixed = family_errors[(common, "fixed_diagonal")]
    common_shared = family_errors[(common, "learned_basis_diagonal")]
    noncommuting_shared = family_errors[(noncommuting, "learned_basis_diagonal")]
    noncommuting_low_rank = family_errors[(noncommuting, "low_rank")]
    return {
        "axis_equivalence_fraction": abs(axis_fixed - axis_shared)
        / max(target_energy[axis], eps),
        "common_rotation_raw_gain": common_fixed - common_shared,
        "common_rotation_recovery_fraction": (common_fixed - common_shared)
        / max(common_fixed, eps),
        "common_shared_residual_fraction": common_shared
        / max(target_energy[common], eps),
        "noncommuting_raw_gap": noncommuting_shared - common_shared,
        "noncommuting_gap_fraction": (noncommuting_shared - common_shared)
        / max(target_energy[noncommuting], eps),
        "noncommuting_shared_residual_fraction": noncommuting_shared
        / max(target_energy[noncommuting], eps),
        "low_rank_raw_gain": noncommuting_shared - noncommuting_low_rank,
        "low_rank_recovery_fraction": (
            noncommuting_shared - noncommuting_low_rank
        )
        / max(noncommuting_shared, eps),
        "low_rank_residual_fraction": noncommuting_low_rank
        / max(target_energy[noncommuting], eps),
    }


def _mean(rows: list[dict[str, float]], key: str) -> float:
    return sum(row[key] for row in rows) / len(rows)


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
    original_seeds = {int(value) for value in config["protocol"]["original_seeds"]}
    if original_seeds.intersection(seeds):
        raise ValueError("E11b seeds must be disjoint from the original E11 seeds")
    families = [RepresentationFamily(value) for value in config["data"]["families"]]
    controller_kinds = [str(value) for value in config["model"]["controllers"]]
    steps = int(config["training"]["steps"])
    if args.dry_run:
        seeds = seeds[:1]
        steps = min(steps, 12)

    dimension = int(config["data"]["dimension"])
    active_rank = int(config["data"]["active_rank"])
    descriptor_dim = int(config["data"]["descriptor_dim"])
    hidden_dim = int(config["model"]["hidden_dim"])
    low_rank = int(config["model"]["low_rank"])
    metric_rows: list[dict[str, float | int | str]] = []
    seed_contrasts: list[dict[str, float | int]] = []

    for seed in seeds:
        train_descriptor = sample_representation_descriptors(
            count=int(config["data"]["train_count"]),
            descriptor_dim=descriptor_dim,
            seed=10_000 + seed,
        )
        test_descriptor = sample_representation_descriptors(
            count=int(config["data"]["test_count"]),
            descriptor_dim=descriptor_dim,
            seed=20_000 + seed,
        )
        family_errors: dict[tuple[str, str], float] = {}
        target_energy: dict[str, float] = {}
        for family in families:
            generator = make_representation_generator(
                family=family,
                dimension=dimension,
                active_rank=active_rank,
                descriptor_dim=descriptor_dim,
                rotation_scale=float(config["data"]["rotation_scale"]),
                seed=30_000 + seed,
            )
            train_target = generator.operators(train_descriptor)
            test_target = generator.operators(test_descriptor)
            target_energy[family.value] = float(test_target.square().mean())
            for kind_index, kind in enumerate(controller_kinds):
                torch.manual_seed(40_000 + seed + 100 * kind_index)
                model = _make_model(
                    kind,
                    descriptor_dim=descriptor_dim,
                    dimension=dimension,
                    hidden_dim=hidden_dim,
                    low_rank=low_rank,
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
                    seed=50_000 + seed,
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
                    / f"seed{seed}_{family.value}_{kind}.pt"
                )
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model": model.state_dict(),
                        "seed": seed,
                        "family": family.value,
                        "controller": kind,
                        "config": config,
                    },
                    checkpoint,
                )
                family_errors[(family.value, kind)] = test_error
                metric_rows.append(
                    {
                        "seed": seed,
                        "family": family.value,
                        "controller": kind,
                        "test_error": test_error,
                        "target_energy": target_energy[family.value],
                        "test_error_std": float(per_example.std(unbiased=False)),
                        "parameter_count": parameter_count(model),
                        "initial_loss": trace.initial_loss,
                        "final_loss": trace.final_loss,
                        "best_loss": trace.best_loss,
                        "checkpoint": str(checkpoint.resolve()),
                    }
                )
        seed_contrasts.append(
            {
                "seed": seed,
                **normalized_seed_contrasts(family_errors, target_energy),
            }
        )

    if args.dry_run:
        supported = False
        gates: dict[str, bool] = {"dry_run_not_evidence": False}
        summary: dict[str, Any] = {"seeds": len(seeds)}
    else:
        thresholds = config["claim_gate"]
        common_raw = [
            float(row["common_rotation_raw_gain"]) for row in seed_contrasts
        ]
        noncommuting_raw = [
            float(row["noncommuting_raw_gap"]) for row in seed_contrasts
        ]
        low_rank_raw = [float(row["low_rank_raw_gain"]) for row in seed_contrasts]
        gates = {
            "axis_equivalence": max(
                float(row["axis_equivalence_fraction"]) for row in seed_contrasts
            )
            <= float(thresholds["axis_equivalence_fraction"]),
            "common_rotation_recovery": _mean(
                seed_contrasts, "common_rotation_recovery_fraction"
            )
            >= float(thresholds["minimum_common_rotation_recovery_fraction"]),
            "common_shared_floor": _mean(
                seed_contrasts, "common_shared_residual_fraction"
            )
            <= float(thresholds["maximum_common_shared_residual_fraction"]),
            "common_direction": exact_sign_flip(
                common_raw, alternative="greater"
            )
            <= float(config["statistics"]["alpha"]),
            "noncommuting_gap": _mean(
                seed_contrasts, "noncommuting_gap_fraction"
            )
            >= float(thresholds["minimum_noncommuting_gap_fraction"]),
            "noncommuting_residual": _mean(
                seed_contrasts, "noncommuting_shared_residual_fraction"
            )
            >= float(thresholds["minimum_noncommuting_shared_residual_fraction"]),
            "noncommuting_direction": exact_sign_flip(
                noncommuting_raw, alternative="greater"
            )
            <= float(config["statistics"]["alpha"]),
            "low_rank_recovery": _mean(
                seed_contrasts, "low_rank_recovery_fraction"
            )
            >= float(thresholds["minimum_low_rank_recovery_fraction"]),
            "low_rank_floor": _mean(
                seed_contrasts, "low_rank_residual_fraction"
            )
            <= float(thresholds["maximum_low_rank_residual_fraction"]),
            "low_rank_direction": exact_sign_flip(
                low_rank_raw, alternative="greater"
            )
            <= float(config["statistics"]["alpha"]),
        }
        supported = all(gates.values())
        summary = {
            "seeds": len(seeds),
            "axis_equivalence_fraction_max": max(
                float(row["axis_equivalence_fraction"]) for row in seed_contrasts
            ),
            "common_rotation_recovery_fraction_mean": _mean(
                seed_contrasts, "common_rotation_recovery_fraction"
            ),
            "common_shared_residual_fraction_mean": _mean(
                seed_contrasts, "common_shared_residual_fraction"
            ),
            "noncommuting_gap_fraction_mean": _mean(
                seed_contrasts, "noncommuting_gap_fraction"
            ),
            "noncommuting_shared_residual_fraction_mean": _mean(
                seed_contrasts, "noncommuting_shared_residual_fraction"
            ),
            "low_rank_recovery_fraction_mean": _mean(
                seed_contrasts, "low_rank_recovery_fraction"
            ),
            "low_rank_residual_fraction_mean": _mean(
                seed_contrasts, "low_rank_residual_fraction"
            ),
            "common_sign_flip_p": exact_sign_flip(
                common_raw, alternative="greater"
            ),
            "noncommuting_sign_flip_p": exact_sign_flip(
                noncommuting_raw, alternative="greater"
            ),
            "low_rank_sign_flip_p": exact_sign_flip(
                low_rank_raw, alternative="greater"
            ),
        }

    write_jsonl(run_dir / "coadaptation_metrics.jsonl", metric_rows)
    write_jsonl(run_dir / "seed_normalized_contrasts.jsonl", seed_contrasts)
    report = {
        "status": "DRY_RUN" if args.dry_run else "PASS",
        "run_scope": "CONTROLLED_REFERENCE",
        "protocol": {
            "repair": "PROSPECTIVE_SCALE_NORMALIZED_GATE",
            "original_e11_remains_not_opened": True,
            "original_rows_reused": False,
            "config_sha256": file_sha256(args.config),
        },
        "summary": summary,
        "gates": gates,
        "claim_gate": {
            "supported": supported,
            "allowed_claim": (
                "On fresh controlled-reference families, target-energy-normalized "
                "tests assess whether a learned shared basis absorbs common rotation "
                "but retains a noncommuting residual removed by richer low-rank control."
            ),
            "forbidden_claims": [
                "The original E11 gate was supported.",
                "The low-rank controller is parameter matched to diagonal controllers.",
                (
                    "This controlled reference establishes language-model or "
                    "official-backend transfer."
                ),
            ],
        },
    }
    finalize_run(
        experiment_id=EXPERIMENT_ID,
        artifact_root=args.artifact_root,
        run_dir=run_dir,
        report=report,
    )
    print(f"[{EXPERIMENT_ID}] {report['status']}: {run_dir}")


if __name__ == "__main__":
    main()
