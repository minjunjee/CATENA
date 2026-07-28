from __future__ import annotations

from collections import defaultdict

import torch

from catena.core.io import write_jsonl
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

EXPERIMENT_ID = "e11_representation_control_coadaptation"
DEFAULT_CONFIG = "configs/e11_representation_control_coadaptation.yaml"


def _make_model(kind: str, *, config: dict, descriptor_dim: int, dimension: int):
    hidden_dim = int(config["model"]["hidden_dim"])
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
            rank=int(config["model"]["low_rank"]),
            hidden_dim=hidden_dim,
        )
    raise ValueError(f"Unknown controller kind: {kind}")


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
    families = [RepresentationFamily(value) for value in config["data"]["families"]]
    controller_kinds = [str(value) for value in config["model"]["controllers"]]
    steps = int(config["training"]["steps"])
    if args.dry_run:
        seeds = seeds[:1]
        steps = min(steps, 12)

    dimension = int(config["data"]["dimension"])
    active_rank = int(config["data"]["active_rank"])
    descriptor_dim = int(config["data"]["descriptor_dim"])
    rows: list[dict[str, float | int | str]] = []
    metrics: dict[tuple[int, str, str], float] = {}

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
            for kind in controller_kinds:
                torch.manual_seed(40_000 + seed + 100 * controller_kinds.index(kind))
                model = _make_model(
                    kind,
                    config=config,
                    descriptor_dim=descriptor_dim,
                    dimension=dimension,
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
                metrics[(seed, family.value, kind)] = test_error
                rows.append(
                    {
                        "seed": seed,
                        "family": family.value,
                        "controller": kind,
                        "test_error": test_error,
                        "test_error_std": float(per_example.std(unbiased=False)),
                        "parameter_count": parameter_count(model),
                        "initial_loss": trace.initial_loss,
                        "final_loss": trace.final_loss,
                        "best_loss": trace.best_loss,
                    }
                )

    axis_equivalence: list[float] = []
    common_recovery: list[float] = []
    noncommuting_gap: list[float] = []
    low_rank_recovery: list[float] = []
    for seed in seeds:
        axis_fixed = metrics[(seed, RepresentationFamily.AXIS_COMMUTING.value, "fixed_diagonal")]
        axis_learned = metrics[(seed, RepresentationFamily.AXIS_COMMUTING.value, "learned_basis_diagonal")]
        common_fixed = metrics[(seed, RepresentationFamily.COMMON_ROTATED_COMMUTING.value, "fixed_diagonal")]
        common_learned = metrics[(seed, RepresentationFamily.COMMON_ROTATED_COMMUTING.value, "learned_basis_diagonal")]
        noncommuting_learned = metrics[(seed, RepresentationFamily.NONCOMMUTING.value, "learned_basis_diagonal")]
        noncommuting_low_rank = metrics[(seed, RepresentationFamily.NONCOMMUTING.value, "low_rank")]
        axis_equivalence.append(abs(axis_fixed - axis_learned))
        common_recovery.append(common_fixed - common_learned)
        noncommuting_gap.append(noncommuting_learned - common_learned)
        low_rank_recovery.append(noncommuting_learned - noncommuting_low_rank)

    alpha = float(config["statistics"]["alpha"])
    common_threshold = float(config["claim_gate"]["common_rotation_recovery"])
    noncommuting_threshold = float(config["claim_gate"]["noncommuting_gap"])
    low_rank_threshold = float(config["claim_gate"]["low_rank_recovery"])
    axis_margin = float(config["claim_gate"]["axis_equivalence_margin"])
    tests = {
        "axis_equivalence_max": max(axis_equivalence) if axis_equivalence else 0.0,
        "common_rotation_recovery_mean": sum(common_recovery) / len(common_recovery),
        "common_rotation_recovery_p": exact_sign_flip(common_recovery, alternative="greater"),
        "noncommuting_gap_mean": sum(noncommuting_gap) / len(noncommuting_gap),
        "noncommuting_gap_p": exact_sign_flip(noncommuting_gap, alternative="greater"),
        "low_rank_recovery_mean": sum(low_rank_recovery) / len(low_rank_recovery),
        "low_rank_recovery_p": exact_sign_flip(low_rank_recovery, alternative="greater"),
    }
    supported = (
        tests["axis_equivalence_max"] <= axis_margin
        and tests["common_rotation_recovery_mean"] >= common_threshold
        and tests["common_rotation_recovery_p"] <= alpha
        and tests["noncommuting_gap_mean"] >= noncommuting_threshold
        and tests["noncommuting_gap_p"] <= alpha
        and tests["low_rank_recovery_mean"] >= low_rank_threshold
        and tests["low_rank_recovery_p"] <= alpha
    )
    report = {
        "status": "PASS",
        "run_scope": "CONTROLLED_REFERENCE",
        "contrasts": tests,
        "claim_gate": {
            "supported": supported,
            "allowed_claim": (
                "A learned shared representation can absorb a common rotation, "
                "but not a genuinely noncommuting demand family; richer control is then required."
            ),
            "forbidden_claim": (
                "Representation learning removes all control-geometry constraints in recurrent language models."
            ),
        },
    }
    write_jsonl(run_dir / "coadaptation_metrics.jsonl", rows)
    finalize_run(
        experiment_id=EXPERIMENT_ID,
        artifact_root=args.artifact_root,
        run_dir=run_dir,
        report=report,
    )
    print(f"[{EXPERIMENT_ID}] PASS: {run_dir}")


if __name__ == "__main__":
    main()
