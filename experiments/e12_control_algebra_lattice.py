from __future__ import annotations

import torch

from catena.core.io import file_sha256, write_jsonl
from catena.data.control_lattice import DemandAxis
from catena.eval.postcore_metrics import exact_sign_flip
from catena.models.lattice_controllers import (
    ControlFreedom,
    MatchedControlLatticeController,
    controller_parameter_count,
)
from catena.training.lattice_training import (
    evaluate_lattice_controller,
    train_lattice_controller,
)
from experiments.common import build_parser, finalize_run, initialize_run

EXPERIMENT_ID = "e12_control_algebra_lattice"
DEFAULT_CONFIG = "configs/e12_control_algebra_lattice.yaml"


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
    freedoms = [ControlFreedom(value) for value in config["model"]["freedoms"]]
    families = [DemandAxis(value) for value in config["data"]["families"]]
    steps = int(config["training"]["steps"])
    if args.dry_run:
        seeds = seeds[:1]
        steps = min(steps, 16)

    rows: list[dict[str, float | int | str]] = []
    metric: dict[tuple[int, str, str], float] = {}
    slots = int(config["data"]["slots"])
    value_dim = int(config["data"]["value_dim"])
    descriptor_dim = int(config["data"]["descriptor_dim"])

    for seed in seeds:
        for freedom in freedoms:
            # Every freedom uses the same maximal parameter surface.  Pair its
            # initial tensors within seed so the contrast isolates only the
            # projection constraint.
            torch.manual_seed(10_000 * seed)
            model = MatchedControlLatticeController(
                freedom=freedom,
                descriptor_dim=descriptor_dim,
                value_dim=value_dim,
                hidden_dim=int(config["model"]["hidden_dim"]),
            )
            trace = train_lattice_controller(
                model=model,
                families=families,
                steps=steps,
                batch_size=int(config["training"]["batch_size"]),
                slots=slots,
                value_dim=value_dim,
                learning_rate=float(config["training"]["learning_rate"]),
                device=device,
                seed=20_000 + seed,
            )
            checkpoint = run_dir / "checkpoints" / f"seed{seed}_{freedom.value}.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "seed": seed,
                    "freedom": freedom.value,
                    "config": config,
                },
                checkpoint,
            )
            checkpoint_sha256 = file_sha256(checkpoint)
            for family in families:
                evaluation = evaluate_lattice_controller(
                    model=model,
                    family=family,
                    episodes=int(config["evaluation"]["episodes_per_family"]),
                    batch_size=int(config["evaluation"]["batch_size"]),
                    slots=slots,
                    value_dim=value_dim,
                    device=device,
                    seed=30_000 + seed + 100 * families.index(family),
                )
                metric[(seed, freedom.value, family.value)] = evaluation["affected_mse"]
                rows.append(
                    {
                        "seed": seed,
                        "freedom": freedom.value,
                        "family": family.value,
                        "parameter_count": controller_parameter_count(model),
                        "train_final_loss": trace.final_loss,
                        "train_best_loss": trace.best_loss,
                        "checkpoint": str(checkpoint.resolve()),
                        "checkpoint_sha256": checkpoint_sha256,
                        **evaluation,
                    }
                )

    adjacent = [
        (ControlFreedom.TIED, ControlFreedom.DUAL, DemandAxis.MAGNITUDE, "magnitude_factorization"),
        (ControlFreedom.DUAL, ControlFreedom.DIAGONAL, DemandAxis.GRANULARITY, "value_granularity"),
        (
            ControlFreedom.DIAGONAL,
            ControlFreedom.SEPARATE_ADDRESS,
            DemandAxis.ADDRESS,
            "address_decoupling",
        ),
        (
            ControlFreedom.SEPARATE_ADDRESS,
            ControlFreedom.STATE_AWARE,
            DemandAxis.STATE_CONDITIONED,
            "state_conditioning",
        ),
    ]
    alpha = float(config["statistics"]["alpha"])
    sesoi = float(config["claim_gate"]["selective_gain"])
    noninferiority = float(config["claim_gate"]["simpler_task_noninferiority"])
    contrasts: dict[str, dict[str, float | bool]] = {}
    all_supported = True
    for baseline, treatment, target_family, name in adjacent:
        gains = [
            metric[(seed, baseline.value, target_family.value)]
            - metric[(seed, treatment.value, target_family.value)]
            for seed in seeds
        ]
        simpler = families[: families.index(target_family)]
        collateral = []
        for seed in seeds:
            for family in simpler:
                collateral.append(
                    metric[(seed, treatment.value, family.value)]
                    - metric[(seed, baseline.value, family.value)]
                )
        mean_gain = sum(gains) / len(gains)
        p = exact_sign_flip(gains, alternative="greater")
        max_collateral = max(collateral) if collateral else 0.0
        passed = mean_gain >= sesoi and p <= alpha and max_collateral <= noninferiority
        all_supported = all_supported and passed
        contrasts[name] = {
            "mean_selective_gain": mean_gain,
            "sign_flip_p": p,
            "max_simpler_task_degradation": max_collateral,
            "passed": passed,
        }

    report = {
        "status": "PASS",
        "run_scope": "CONTROLLED_REFERENCE",
        "contrasts": contrasts,
        "claim_gate": {
            "supported": all_supported,
            "allowed_claim": (
                "Each added memory-control freedom yields a selective advantage "
                "only on the registered demand family that requires it."
            ),
            "forbidden_claim": (
                "Any one reference controller is a universally superior "
                "language-model architecture."
            ),
        },
    }
    write_jsonl(run_dir / "control_lattice_metrics.jsonl", rows)
    finalize_run(
        experiment_id=EXPERIMENT_ID,
        artifact_root=args.artifact_root,
        run_dir=run_dir,
        report=report,
    )
    print(f"[{EXPERIMENT_ID}] PASS: {run_dir}")


if __name__ == "__main__":
    main()
