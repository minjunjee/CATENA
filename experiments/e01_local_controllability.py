from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import torch

from catena.core.io import write_jsonl
from catena.core.randomness import seed_everything
from catena.core.schema import CandidateMode, ControllerKind
from catena.data.tamp import TAMPConfig, generate_episodes
from catena.eval.metrics import evaluate_episode
from catena.eval.statistics import hierarchical_seed_episode_slope_bootstrap
from catena.models.controllers import ControllerSpec, GateController
from catena.theory.control_geometry import analytic_optimal_controls, local_control_geometry
from catena.training.probe import TrainConfig, apply_trained_controller, train_probe_controller
from experiments.common import build_parser, finalize_run, initialize_run

EXPERIMENT_ID = "e01_local_controllability"
DEFAULT_CONFIG = "configs/e01_local_controllability.yaml"


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


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
    tamp_cfg = TAMPConfig(**config["tamp"])
    count = int(config["data"]["count_per_operation"])
    train_steps = int(config["training"]["steps"])
    if args.dry_run:
        seeds = seeds[:2]
        count = min(count, 8)
        train_steps = min(train_steps, 40)

    rows: list[dict[str, object]] = []
    seed_xy: dict[tuple[str, str], dict[int, tuple[np.ndarray, np.ndarray]]] = defaultdict(dict)
    summary_values: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)

    for seed in seeds:
        seed_everything(seed)
        for mode_index, mode in enumerate(CandidateMode):
            episodes = generate_episodes(
                count_per_operation=count,
                seed=seed * 10_000 + mode_index * 1000,
                candidate_mode=mode,
                config=tamp_cfg,
            )
            permutation = torch.randperm(
                len(episodes), generator=torch.Generator().manual_seed(seed + mode_index)
            ).tolist()
            episodes = [episodes[index] for index in permutation]
            split = max(4, int(0.7 * len(episodes)))
            train_episodes, test_episodes = episodes[:split], episodes[split:]

            for kind in [ControllerKind.TIED_SCALAR, ControllerKind.DUAL_SCALAR]:
                model = GateController(
                    ControllerSpec(
                        kind=kind,
                        input_dim=4,
                        value_dim=tamp_cfg.value_dim,
                        hidden_dim=int(config["model"]["hidden_dim"]),
                    )
                )
                train_probe_controller(
                    model=model,
                    episodes=train_episodes,
                    config=TrainConfig(
                        steps=train_steps,
                        learning_rate=float(config["training"]["learning_rate"]),
                        affected_weight=1.0,
                        retention_weight=1.0,
                        state_weight=0.25,
                    ),
                    device=device,
                )
                regrets: list[float] = []
                learned_errors: list[float] = []
                analytic_errors: list[float] = []
                for episode in test_episodes:
                    geometry = local_control_geometry(episode, kind)
                    _, analytic_mse = analytic_optimal_controls(episode, kind)
                    output = apply_trained_controller(model, episode.to(device)).cpu()
                    metrics = evaluate_episode(output, episode)
                    regrets.append(geometry.projection_regret)
                    learned_errors.append(metrics.target_state_mse)
                    analytic_errors.append(analytic_mse)
                    rows.append(
                        {
                            "seed": seed,
                            "candidate_mode": mode.value,
                            "controller": kind.value,
                            "episode_id": episode.episode_id,
                            "operation": episode.operation.value,
                            "rank": geometry.rank,
                            "condition_number": geometry.condition_number,
                            "principal_angle_deg": geometry.principal_angle_deg,
                            "projection_regret": geometry.projection_regret,
                            "analytic_target_state_mse": analytic_mse,
                            **metrics.to_dict(),
                        }
                    )
                x = np.asarray(regrets, dtype=np.float64)
                y = np.asarray(learned_errors, dtype=np.float64)
                seed_xy[(mode.value, kind.value)][seed] = (x, y)
                summary_values[(mode.value, kind.value)].append(
                    {
                        "seed": float(seed),
                        "projection_regret_mean": float(x.mean()),
                        "analytic_target_state_mse_mean": float(np.mean(analytic_errors)),
                        "learned_target_state_mse_mean": float(y.mean()),
                        "regret_error_pearson": _pearson(x, y),
                    }
                )
                torch.save(model.state_dict(), run_dir / f"seed{seed}_{mode.value}_{kind.value}.pt")

    candidate_summary: dict[str, dict[str, object]] = {}
    slope_reports: dict[tuple[str, str], object] = {}
    for mode in CandidateMode:
        mode_summary: dict[str, object] = {}
        for kind in [ControllerKind.TIED_SCALAR, ControllerKind.DUAL_SCALAR]:
            key = (mode.value, kind.value)
            slope = hierarchical_seed_episode_slope_bootstrap(
                seed_xy[key],
                samples=int(config["statistics"]["bootstrap_samples"]),
                seed=901 + len(slope_reports),
            )
            slope_reports[key] = slope
            values = summary_values[key]
            mode_summary[kind.value] = {
                "projection_regret_mean": float(
                    np.mean([value["projection_regret_mean"] for value in values])
                ),
                "analytic_target_state_mse_mean": float(
                    np.mean([value["analytic_target_state_mse_mean"] for value in values])
                ),
                "learned_target_state_mse_mean": float(
                    np.mean([value["learned_target_state_mse_mean"] for value in values])
                ),
                "regret_error_pearson_mean": float(
                    np.nanmean([value["regret_error_pearson"] for value in values])
                ),
                "regret_error_slope": {
                    "estimate": slope.estimate,
                    "ci95": [slope.low, slope.high],
                },
            }
        candidate_summary[mode.value] = mode_summary

    oracle_tied_slope = slope_reports[(CandidateMode.ORACLE.value, ControllerKind.TIED_SCALAR.value)]
    recurrent_tied_slope = slope_reports[
        (CandidateMode.RECURRENT_READ.value, ControllerKind.TIED_SCALAR.value)
    ]
    supported = bool(math.isfinite(oracle_tied_slope.low) and oracle_tied_slope.low > 0.0)
    report = {
        "status": "PASS" if math.isfinite(oracle_tied_slope.estimate) else "WARN",
        "primary": {
            "oracle_tied_regret_error_slope": {
                "estimate": oracle_tied_slope.estimate,
                "ci95": [oracle_tied_slope.low, oracle_tied_slope.high],
                "supported": supported,
            },
            "recurrent_read_tied_regret_error_slope": {
                "estimate": recurrent_tied_slope.estimate,
                "ci95": [recurrent_tied_slope.low, recurrent_tied_slope.high],
            },
            "criterion": (
                "The hierarchical seed/episode bootstrap CI for the OracleCandidate tied "
                "projection-regret slope must lie above zero."
            ),
        },
        "candidate_modes": candidate_summary,
        "claim_gate": {
            "supported": supported,
            "oracle_candidate_only_for_pure_geometry": True,
        },
        "interpretation": {
            "oracle_candidate": "isolates local control-space geometry",
            "recurrent_read": (
                "adds candidate recovery, content interference, and addressing error; it is "
                "reported as an external-validity decomposition, not as the pure geometry test"
            ),
        },
    }
    write_jsonl(run_dir / "episode_metrics.jsonl", rows)
    finalize_run(
        experiment_id=EXPERIMENT_ID,
        artifact_root=args.artifact_root,
        run_dir=run_dir,
        report=report,
    )
    print(f"[{EXPERIMENT_ID}] {report['status']}: {run_dir}")


if __name__ == "__main__":
    main()
