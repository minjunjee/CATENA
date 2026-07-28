from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from catena.core.provenance_v61 import sha256_file, write_jsonl_strict
from catena.core.randomness import seed_everything
from catena.core.schema import CandidateMode, Operation
from catena.data.geometry_sweep import build_geometry_episode
from catena.eval.metrics import evaluate_episode
from catena.eval.seed_inference import exact_sign_flip_test
from catena.eval.statistics_v61 import (
    Interval,
    fixed_seed_operation_stratified_bootstrap,
)
from catena.models.matched_controllers import MatchedScalarController, ScalarConstraint
from catena.theory.reachability import behavioral_mse
from catena.training.matched_probe import (
    MatchedTrainConfig,
    apply_matched_controller,
    train_matched_controller,
)
from experiments.common import build_parser
from experiments.v61_common import (
    finalize_v61_run,
    initialize_v61_run,
    validate_e01b_dependency,
)

EXPERIMENT_ID = "e02_magnitude_factorization"
DEFAULT_CONFIG = "configs/e02_magnitude_factorization.yaml"


def _episodes(seed: int, count: int, data: dict[str, Any]) -> list[Any]:
    rows: list[Any] = []
    cursor = 0
    # Index-major construction keeps every prefix balanced across operations.
    # Training cycles deterministically through this list.
    for index in range(count):
        for operation in Operation:
            rows.append(
                build_geometry_episode(
                    seed=seed + cursor,
                    operation=operation,
                    candidate_mode=CandidateMode.ORACLE,
                    key_dim=int(data["key_dim"]),
                    value_dim=int(data["value_dim"]),
                    num_associations=int(data["num_associations"]),
                    key_correlation=float(data["key_correlation"]),
                    old_scale=float(data["old_scale"]),
                    new_scale=float(data["new_scale"]),
                    old_new_cosine=float(data["old_new_cosine"]),
                    episode_index=index,
                )
            )
            cursor += 1
    return rows


def _fresh_initial_state(seed: int, hidden: int) -> dict[str, torch.Tensor]:
    seed_everything(seed)
    template = MatchedScalarController(10, hidden, ScalarConstraint.DUAL)
    return {name: value.detach().clone() for name, value in template.state_dict().items()}


def _state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(b"\0")
        digest.update(repr(tuple(tensor.shape)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _require_finite(value: Any, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise FloatingPointError(f"{name} is non-finite: {result}")
    return result


def _finite_array(
    values: Any,
    name: str,
    *,
    allow_empty: bool = False,
) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got {result.shape}.")
    if not allow_empty and len(result) == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.isfinite(result).all():
        raise FloatingPointError(f"{name} contains a non-finite value.")
    return result


def _assert_finite_state_dict(model: MatchedScalarController, name: str) -> None:
    for tensor_name, tensor in model.state_dict().items():
        if not bool(torch.isfinite(tensor).all().item()):
            raise FloatingPointError(f"{name}.{tensor_name} contains a non-finite value.")


def _assert_finite_payload(payload: Any, name: str = "payload") -> None:
    """Reject JSON NaN/Infinity while permitting explicit unavailable values."""

    if payload is None or isinstance(payload, (str, bool)):
        return
    if isinstance(payload, (int, np.integer)):
        return
    if isinstance(payload, (float, np.floating)):
        _require_finite(float(payload), name)
        return
    if isinstance(payload, dict):
        for key, value in payload.items():
            _assert_finite_payload(value, f"{name}.{key}")
        return
    if isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            _assert_finite_payload(value, f"{name}[{index}]")
        return
    raise TypeError(f"{name} has unsupported JSON value type {type(payload).__name__}.")


def _train_model(
    *,
    constraint: ScalarConstraint,
    initial_state: dict[str, torch.Tensor],
    train: list[Any],
    hidden: int,
    steps: int,
    learning_rate: float,
    device: torch.device,
) -> MatchedScalarController:
    model = MatchedScalarController(10, hidden, constraint)
    model.load_state_dict(initial_state)
    losses = train_matched_controller(
        model=model,
        episodes=train,
        config=MatchedTrainConfig(steps=steps, learning_rate=learning_rate),
        device=device,
    )
    if len(losses) != steps:
        raise RuntimeError(
            f"{constraint.value} training returned {len(losses)} losses for {steps} steps."
        )
    _finite_array(losses, f"{constraint.value} training losses")
    _assert_finite_state_dict(model, f"{constraint.value} trained model")
    return model


def _mean_behavioral_error(
    model: MatchedScalarController,
    episodes: list[Any],
) -> float:
    """Predeclared tuning objective with equal correction/retention weights."""

    values: list[float] = []
    for episode in episodes:
        output = apply_matched_controller(model, episode).cpu()
        values.append(
            _require_finite(
                behavioral_mse(output, episode).item(),
                f"{episode.episode_id} behavioral MSE",
            )
        )
    return _require_finite(
        _finite_array(values, "behavioral validation scores").mean(),
        "mean behavioral validation score",
    )


def _normalized_asymmetric_gain(
    tied: MatchedScalarController,
    dual: MatchedScalarController,
    episodes: list[Any],
    *,
    minimum_headroom: float,
) -> tuple[float | None, int, int]:
    gains: list[float] = []
    excluded = 0
    for episode in episodes:
        if not episode.operation.is_asymmetric:
            continue
        tied_metrics = evaluate_episode(
            apply_matched_controller(tied, episode).cpu(),
            episode,
        )
        dual_metrics = evaluate_episode(
            apply_matched_controller(dual, episode).cpu(),
            episode,
        )
        oracle_metrics = evaluate_episode(episode.target_state, episode)
        tied_headroom = _require_finite(
            tied_metrics.affected_read_mse - oracle_metrics.affected_read_mse,
            f"{episode.episode_id} tuned tied-to-oracle headroom",
        )
        if tied_headroom < minimum_headroom:
            excluded += 1
            continue
        gains.append(
            _require_finite(
                (tied_metrics.affected_read_mse - dual_metrics.affected_read_mse) / tied_headroom,
                f"{episode.episode_id} tuned normalized gain",
            )
        )
    if not gains:
        return None, 0, excluded
    return (
        _require_finite(
            np.mean(gains),
            "tuned mean asymmetric normalized gain",
        ),
        len(gains),
        excluded,
    )


def _validated_seed_effects(
    seed_episode_effects: dict[int, np.ndarray],
    name: str,
) -> dict[int, np.ndarray]:
    if not seed_episode_effects:
        raise ValueError(f"{name} has no seed-level effects.")
    return {
        int(seed): _finite_array(values, f"{name}[seed={seed}]")
        for seed, values in sorted(seed_episode_effects.items())
    }


def _fixed_seed_episode_bootstrap(
    seed_episode_effects: dict[int, np.ndarray],
    seed_operations: dict[int, np.ndarray],
    *,
    samples: int,
    seed: int,
) -> Interval:
    """Bootstrap episodes within fixed checkpoints, never training seeds.

    This local E02 API boundary is intended for later migration to the common
    statistics module. Training-replicate uncertainty is tested separately
    with the exact seed-level sign-flip test.
    """

    if samples <= 0:
        raise ValueError("Bootstrap samples must be positive.")
    effects = _validated_seed_effects(
        seed_episode_effects,
        "fixed-seed episode effects",
    )
    if set(seed_operations) != set(effects):
        raise ValueError("Bootstrap operations must cover the same fixed seeds.")
    operations = {
        training_seed: np.asarray(seed_operations[training_seed]).astype(str)
        for training_seed in effects
    }
    if any(
        len(operations[training_seed]) != len(effects[training_seed])
        for training_seed in effects
    ):
        raise ValueError("Bootstrap operation labels must align with episode effects.")

    def statistic(indices_by_seed: Mapping[int, np.ndarray]) -> float:
        return float(
            np.mean(
                [
                    effects[training_seed][indices].mean()
                    for training_seed, indices in indices_by_seed.items()
                ]
            )
        )

    return fixed_seed_operation_stratified_bootstrap(
        operations,
        statistic,
        samples=samples,
        seed=seed,
    )


def _fixed_seed_did_episode_bootstrap(
    seed_episode_effects: dict[int, np.ndarray],
    seed_operations: dict[int, np.ndarray],
    *,
    samples: int,
    seed: int,
) -> Interval:
    """Episode CI for asymmetric-minus-symmetric DID with fixed checkpoints."""

    if not seed_episode_effects:
        raise ValueError("DID bootstrap has no seed-level effects.")
    if samples <= 0:
        raise ValueError("Bootstrap samples must be positive.")
    effects = _validated_seed_effects(seed_episode_effects, "DID episode effects")
    if set(seed_operations) != set(effects):
        raise ValueError("DID operation labels must cover the same fixed seeds.")
    operations = {
        training_seed: np.asarray(seed_operations[training_seed]).astype(str)
        for training_seed in effects
    }

    def statistic(indices_by_seed: Mapping[int, np.ndarray]) -> float:
        per_seed = []
        for training_seed, indices in indices_by_seed.items():
            values = effects[training_seed][indices]
            labels = operations[training_seed][indices]
            asymmetric = np.isin(
                labels,
                [Operation.ADD.value, Operation.INVALIDATE.value],
            )
            per_seed.append(
                float(values[asymmetric].mean() - values[~asymmetric].mean())
            )
        return float(np.mean(per_seed))

    return fixed_seed_operation_stratified_bootstrap(
        operations,
        statistic,
        samples=samples,
        seed=seed,
    )


def _evaluable(
    seed_effects: dict[int, np.ndarray],
    seeds: list[int],
) -> bool:
    return set(seed_effects) == set(seeds) and all(
        np.asarray(seed_effects[seed]).size > 0 for seed in seeds
    )


def _interval_fields(interval: Interval | None) -> dict[str, object]:
    if interval is None:
        return {"estimate": None, "ci95": None}
    return {
        "estimate": interval.estimate,
        "ci95": [interval.low, interval.high],
    }


def _equivalence_within(interval: Interval, margin: float) -> bool:
    return interval.low >= -margin and interval.high <= margin


def _string_seed_keys(values: dict[int, Any]) -> dict[str, Any]:
    return {str(seed): value for seed, value in sorted(values.items())}


def _confirmatory_eligibility(
    *,
    dry_run: bool,
    seeds: list[int],
    configured_seeds: list[int],
    run_tuning: bool,
    tuning_record_count: int,
    asymmetric_evaluable: bool,
    supersede_evaluable: bool,
) -> bool:
    return bool(
        not dry_run
        and seeds == configured_seeds
        and len(seeds) == 8
        and len(set(seeds)) == 8
        and run_tuning
        and tuning_record_count == 8
        and asymmetric_evaluable
        and supersede_evaluable
    )


def _h2_claim_supported(
    *,
    confirmatory_eligible: bool,
    asymmetric_gain: bool,
    preserve_equivalence: bool,
    supersede_equivalence: bool,
    positive_interaction: bool,
    retention_noninferiority: bool,
    tuned_direction_consistency: bool,
) -> bool:
    return all(
        (
            confirmatory_eligible,
            asymmetric_gain,
            preserve_equivalence,
            supersede_equivalence,
            positive_interaction,
            retention_noninferiority,
            tuned_direction_consistency,
        )
    )


def main() -> None:
    parser = build_parser(EXPERIMENT_ID, DEFAULT_CONFIG)
    args = parser.parse_args()
    e01b = validate_e01b_dependency(
        args.artifact_root,
        require_main_supported=not args.dry_run,
    )
    dependencies = [e01b.dependency_record()]
    config, run_dir, device, run_context = initialize_v61_run(
        experiment_id=EXPERIMENT_ID,
        config_path=args.config,
        artifact_root=args.artifact_root,
        device_request=args.device,
        dry_run=args.dry_run,
        dependencies=dependencies,
    )
    configured_seeds = [int(value) for value in config["seeds"]]
    seeds = list(configured_seeds)
    train_count = int(config["data"]["train_count_per_operation"])
    test_count = int(config["data"]["test_count_per_operation"])
    steps = int(config["training"]["steps"])
    if args.dry_run:
        seeds = seeds[:1]
        train_count = min(train_count, 8)
        test_count = min(test_count, 8)
        steps = min(steps, 40)

    hidden = int(config["model"]["hidden_dim"])
    strict_lr = float(config["training"]["learning_rate"])
    run_tuning = bool(config["training"]["run_equal_budget_tuning"]) and not args.dry_run
    lr_candidates = [float(value) for value in config["training"]["tuning_learning_rates"]]
    tuning_objective = str(config["training"]["tuning_objective"])
    if tuning_objective != "equal_weight_behavioral_mse":
        raise ValueError(
            "E02 tuning_objective must remain equal_weight_behavioral_mse."
        )
    minimum_headroom = float(
        config["statistics"].get(
            "minimum_tied_oracle_headroom",
            1e-8,
        )
    )
    _require_finite(strict_lr, "strict learning rate")
    _require_finite(
        minimum_headroom,
        "minimum tied-to-oracle headroom",
    )
    if strict_lr <= 0.0:
        raise ValueError("Strict learning rate must be positive.")
    if minimum_headroom <= 0.0:
        raise ValueError("Minimum tied-to-oracle headroom must be positive.")
    _finite_array(lr_candidates, "tuning learning rates")
    if any(value <= 0.0 for value in lr_candidates):
        raise ValueError("Every tuning learning rate must be positive.")
    if len(set(lr_candidates)) != len(lr_candidates):
        raise ValueError("Tuning learning-rate candidates must be unique.")

    rows: list[dict[str, object]] = []
    asym_by_seed: dict[int, np.ndarray] = {}
    asym_operations_by_seed: dict[int, np.ndarray] = {}
    preserve_by_seed: dict[int, np.ndarray] = {}
    preserve_operations_by_seed: dict[int, np.ndarray] = {}
    supersede_by_seed: dict[int, np.ndarray] = {}
    supersede_operations_by_seed: dict[int, np.ndarray] = {}
    retention_by_seed: dict[int, np.ndarray] = {}
    retention_operations_by_seed: dict[int, np.ndarray] = {}
    did_by_seed: dict[int, float] = {}
    did_effects_by_seed: dict[int, np.ndarray] = {}
    did_operations_by_seed: dict[int, np.ndarray] = {}
    noop_gain_by_seed: dict[int, np.ndarray] = {}
    noop_operations_by_seed: dict[int, np.ndarray] = {}
    tuned_direction: dict[int, float | None] = {}
    tuning_records: dict[int, dict[str, object]] = {}
    parameter_counts: dict[str, int] = {}
    eligibility_counts: dict[int, dict[str, dict[str, int]]] = {}
    operation_counts_by_seed: dict[int, dict[str, int]] = {}
    checkpoint_entries: list[dict[str, object]] = []

    for seed in seeds:
        train = _episodes(
            seed * 100000,
            train_count,
            config["data"],
        )
        test = _episodes(
            seed * 100000 + 50000,
            test_count,
            config["data"],
        )
        expected_cycle = list(Operation)
        if [episode.operation for episode in train[: len(expected_cycle)]] != expected_cycle:
            raise AssertionError("Training episodes are not operation-interleaved.")
        train_operation_counts = {
            operation.value: sum(episode.operation is operation for episode in train)
            for operation in Operation
        }
        if set(train_operation_counts.values()) != {train_count}:
            raise AssertionError(f"Seed {seed} training episodes are not operation-balanced.")

        initial_state = _fresh_initial_state(seed, hidden)
        tied = _train_model(
            constraint=ScalarConstraint.TIED,
            initial_state=initial_state,
            train=train,
            hidden=hidden,
            steps=steps,
            learning_rate=strict_lr,
            device=device,
        )
        dual = _train_model(
            constraint=ScalarConstraint.DUAL,
            initial_state=initial_state,
            train=train,
            hidden=hidden,
            steps=steps,
            learning_rate=strict_lr,
            device=device,
        )
        parameter_counts = {
            "tied": sum(parameter.numel() for parameter in tied.parameters()),
            "dual": sum(parameter.numel() for parameter in dual.parameters()),
        }
        if len(set(parameter_counts.values())) != 1:
            raise AssertionError("Matched controllers must have identical parameter counts.")
        initial_state_sha256 = _state_dict_sha256(initial_state)
        for constraint_name, model in (("tied", tied), ("dual", dual)):
            _assert_finite_state_dict(
                model,
                f"strict seed {seed} {constraint_name}",
            )
            checkpoint_path = run_dir / f"seed{seed}_{constraint_name}.pt"
            if checkpoint_path.exists():
                raise FileExistsError(f"Refusing to overwrite {checkpoint_path}.")
            torch.save(model.state_dict(), checkpoint_path)
            if not checkpoint_path.is_file() or checkpoint_path.stat().st_size <= 0:
                raise RuntimeError(f"Checkpoint was not written correctly: {checkpoint_path}")
            round_trip = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            final_state_sha256 = _state_dict_sha256(
                {name: tensor for name, tensor in round_trip.items()}
            )
            if final_state_sha256 != _state_dict_sha256(model.state_dict()):
                raise RuntimeError(f"Checkpoint round-trip mismatch: {checkpoint_path}")
            checkpoint_entries.append(
                {
                    "seed": seed,
                    "constraint": constraint_name,
                    "filename": checkpoint_path.name,
                    "sha256": sha256_file(checkpoint_path),
                    "bytes": checkpoint_path.stat().st_size,
                    "parameter_count": parameter_counts[constraint_name],
                    "initial_state_sha256": initial_state_sha256,
                    "state_dict_sha256": final_state_sha256,
                    "round_trip_match": True,
                    "input_dim": 10,
                    "hidden_dim": hidden,
                    "candidate_mode": CandidateMode.ORACLE.value,
                    "recipe": "strict",
                }
            )

        operation_improvements: dict[Operation, list[float]] = defaultdict(list)
        asym_gains: list[float] = []
        asym_gain_operations: list[str] = []
        noop_gains: list[float] = []
        noop_gain_operations: list[str] = []
        preserve_effects: list[float] = []
        supersede_effects: list[float] = []
        retention_effects: list[float] = []
        asymmetric_excluded = 0
        supersede_excluded = 0
        noop_excluded = 0
        test_operation_counts = {operation.value: 0 for operation in Operation}

        for episode in test:
            tied_metrics = evaluate_episode(
                apply_matched_controller(tied, episode).cpu(),
                episode,
            )
            dual_metrics = evaluate_episode(
                apply_matched_controller(dual, episode).cpu(),
                episode,
            )
            noop_metrics = evaluate_episode(episode.state, episode)
            oracle_metrics = evaluate_episode(
                episode.target_state,
                episode,
            )

            test_operation_counts[episode.operation.value] += 1
            improvement = _require_finite(
                (tied_metrics.affected_read_mse - dual_metrics.affected_read_mse),
                f"{episode.episode_id} tied-minus-dual affected MSE",
            )
            operation_improvements[episode.operation].append(improvement)
            tied_headroom = _require_finite(
                (tied_metrics.affected_read_mse - oracle_metrics.affected_read_mse),
                f"{episode.episode_id} tied-to-oracle headroom",
            )
            noop_headroom = _require_finite(
                (noop_metrics.affected_read_mse - oracle_metrics.affected_read_mse),
                f"{episode.episode_id} noop-to-oracle headroom",
            )

            normalized_gain: float | None = None
            normalized_eligible = False
            normalized_exclusion_reason: str | None
            if not episode.operation.is_asymmetric:
                normalized_exclusion_reason = "operation_not_add_or_invalidate"
            elif tied_headroom < minimum_headroom:
                normalized_exclusion_reason = "tied_to_oracle_headroom_below_minimum"
                asymmetric_excluded += 1
            else:
                normalized_eligible = True
                normalized_exclusion_reason = None
                normalized_gain = _require_finite(
                    improvement / tied_headroom,
                    f"{episode.episode_id} normalized gain",
                )
                asym_gains.append(normalized_gain)
                asym_gain_operations.append(episode.operation.value)

            noop_normalized_gain: float | None = None
            noop_gain_eligible = False
            noop_gain_exclusion_reason: str | None
            if not episode.operation.is_asymmetric:
                noop_gain_exclusion_reason = "operation_not_add_or_invalidate"
            elif noop_headroom < minimum_headroom:
                noop_gain_exclusion_reason = "noop_to_oracle_headroom_below_minimum"
                noop_excluded += 1
            else:
                noop_gain_eligible = True
                noop_gain_exclusion_reason = None
                noop_normalized_gain = _require_finite(
                    (noop_metrics.affected_read_mse - dual_metrics.affected_read_mse)
                    / noop_headroom,
                    f"{episode.episode_id} noop normalized gain",
                )
                noop_gains.append(noop_normalized_gain)
                noop_gain_operations.append(episode.operation.value)

            if episode.operation is Operation.PRESERVE:
                preserve_effects.append(improvement)

            supersede_relative_effect: float | None = None
            supersede_effect_eligible = False
            supersede_exclusion_reason: str | None
            if episode.operation is Operation.SUPERSEDE:
                if tied_headroom < minimum_headroom:
                    supersede_exclusion_reason = "tied_to_oracle_headroom_below_minimum"
                    supersede_excluded += 1
                else:
                    supersede_effect_eligible = True
                    supersede_exclusion_reason = None
                    supersede_relative_effect = _require_finite(
                        improvement / tied_headroom,
                        f"{episode.episode_id} supersede relative effect",
                    )
                    supersede_effects.append(supersede_relative_effect)
            else:
                supersede_exclusion_reason = "operation_not_supersede"

            retention_effect = _require_finite(
                (dual_metrics.unaffected_retention_mse - tied_metrics.unaffected_retention_mse),
                f"{episode.episode_id} retention effect",
            )
            retention_effects.append(retention_effect)
            rows.append(
                {
                    "seed": seed,
                    "split": "test",
                    "candidate_mode": CandidateMode.ORACLE.value,
                    "episode_id": episode.episode_id,
                    "operation": episode.operation.value,
                    "tied_affected_mse": _require_finite(
                        tied_metrics.affected_read_mse,
                        f"{episode.episode_id} tied affected MSE",
                    ),
                    "dual_affected_mse": _require_finite(
                        dual_metrics.affected_read_mse,
                        f"{episode.episode_id} dual affected MSE",
                    ),
                    "noop_affected_mse": _require_finite(
                        noop_metrics.affected_read_mse,
                        f"{episode.episode_id} noop affected MSE",
                    ),
                    "oracle_affected_mse": _require_finite(
                        oracle_metrics.affected_read_mse,
                        f"{episode.episode_id} oracle affected MSE",
                    ),
                    "tied_minus_dual_affected_mse": improvement,
                    "tied_to_oracle_headroom": tied_headroom,
                    "normalized_gain": normalized_gain,
                    "normalized_gain_eligible": normalized_eligible,
                    "normalized_gain_exclusion_reason": (normalized_exclusion_reason),
                    "noop_to_oracle_headroom": noop_headroom,
                    "noop_normalized_gain": noop_normalized_gain,
                    "noop_normalized_gain_eligible": noop_gain_eligible,
                    "noop_normalized_gain_exclusion_reason": (noop_gain_exclusion_reason),
                    "supersede_relative_effect": (supersede_relative_effect),
                    "supersede_relative_effect_eligible": (supersede_effect_eligible),
                    "supersede_relative_effect_exclusion_reason": (supersede_exclusion_reason),
                    "tied_retention_mse": _require_finite(
                        tied_metrics.unaffected_retention_mse,
                        f"{episode.episode_id} tied retention MSE",
                    ),
                    "dual_retention_mse": _require_finite(
                        dual_metrics.unaffected_retention_mse,
                        f"{episode.episode_id} dual retention MSE",
                    ),
                    "dual_minus_tied_retention_mse": retention_effect,
                }
            )

        if set(test_operation_counts.values()) != {test_count}:
            raise AssertionError(f"Seed {seed} test episodes are not operation-balanced.")
        operation_counts_by_seed[seed] = test_operation_counts
        asym_by_seed[seed] = _finite_array(
            asym_gains,
            f"asymmetric normalized gains[seed={seed}]",
            allow_empty=True,
        )
        asym_operations_by_seed[seed] = np.asarray(asym_gain_operations)
        noop_gain_by_seed[seed] = _finite_array(
            noop_gains,
            f"noop normalized gains[seed={seed}]",
            allow_empty=True,
        )
        noop_operations_by_seed[seed] = np.asarray(noop_gain_operations)
        preserve_by_seed[seed] = _finite_array(
            preserve_effects,
            f"preserve effects[seed={seed}]",
        )
        preserve_operations_by_seed[seed] = np.full(
            len(preserve_effects),
            Operation.PRESERVE.value,
        )
        supersede_by_seed[seed] = _finite_array(
            supersede_effects,
            f"supersede effects[seed={seed}]",
            allow_empty=True,
        )
        supersede_operations_by_seed[seed] = np.full(
            len(supersede_effects),
            Operation.SUPERSEDE.value,
        )
        retention_by_seed[seed] = _finite_array(
            retention_effects,
            f"retention effects[seed={seed}]",
        )
        retention_operations_by_seed[seed] = np.asarray(
            [
                episode.operation.value
                for episode in test
            ]
        )
        eligibility_counts[seed] = {
            "asymmetric_normalized_gain": {
                "eligible": len(asym_gains),
                "excluded_low_headroom": asymmetric_excluded,
            },
            "supersede_relative_effect": {
                "eligible": len(supersede_effects),
                "excluded_low_headroom": supersede_excluded,
            },
            "noop_normalized_gain": {
                "eligible": len(noop_gains),
                "excluded_low_headroom": noop_excluded,
            },
        }

        asymmetric_improvements = _finite_array(
            (operation_improvements[Operation.ADD] + operation_improvements[Operation.INVALIDATE]),
            f"asymmetric raw improvements[seed={seed}]",
        )
        symmetric_improvements = _finite_array(
            (
                operation_improvements[Operation.PRESERVE]
                + operation_improvements[Operation.SUPERSEDE]
            ),
            f"symmetric raw improvements[seed={seed}]",
        )
        did_effects_by_seed[seed] = np.concatenate(
            [
                np.asarray(operation_improvements[operation])
                for operation in Operation
            ]
        )
        did_operations_by_seed[seed] = np.concatenate(
            [
                np.full(
                    len(operation_improvements[operation]),
                    operation.value,
                )
                for operation in Operation
            ]
        )
        did_by_seed[seed] = _require_finite(
            (asymmetric_improvements.mean() - symmetric_improvements.mean()),
            f"DID[seed={seed}]",
        )

        if run_tuning:
            validation_count = max(8, test_count // 2)
            validation = _episodes(
                seed * 100000 + 25000,
                validation_count,
                config["data"],
            )
            strict_models = {
                ScalarConstraint.TIED: tied,
                ScalarConstraint.DUAL: dual,
            }
            selected_models: dict[
                ScalarConstraint,
                MatchedScalarController,
            ] = {}
            constraint_records: dict[str, object] = {}
            for constraint in (
                ScalarConstraint.TIED,
                ScalarConstraint.DUAL,
            ):
                trials: list[dict[str, Any]] = []
                candidate_models: dict[
                    float,
                    MatchedScalarController,
                ] = {}
                for learning_rate in lr_candidates:
                    if learning_rate == strict_lr:
                        # This is the identical full-budget fit from the strict
                        # recipe; reuse avoids a deterministic duplicate run.
                        model = strict_models[constraint]
                        trial_source = "strict_full_budget_reuse"
                    else:
                        model = _train_model(
                            constraint=constraint,
                            initial_state=initial_state,
                            train=train,
                            hidden=hidden,
                            steps=steps,
                            learning_rate=learning_rate,
                            device=device,
                        )
                        trial_source = "equal_budget_tuning_fit"
                    score = _mean_behavioral_error(model, validation)
                    candidate_models[learning_rate] = model
                    trials.append(
                        {
                            "learning_rate": learning_rate,
                            "validation_equal_weight_behavioral_mse": score,
                            "model_source": trial_source,
                        }
                    )
                selected_trial = min(
                    trials,
                    key=lambda trial: (
                        float(trial["validation_equal_weight_behavioral_mse"]),
                        float(trial["learning_rate"]),
                    ),
                )
                selected_lr = float(selected_trial["learning_rate"])
                selected_models[constraint] = candidate_models[selected_lr]
                constraint_records[constraint.value] = {
                    "trials": trials,
                    "selected_learning_rate": selected_lr,
                    "selected_validation_score": selected_trial[
                        "validation_equal_weight_behavioral_mse"
                    ],
                }

            tuned_gain, tuned_eligible, tuned_excluded = _normalized_asymmetric_gain(
                selected_models[ScalarConstraint.TIED],
                selected_models[ScalarConstraint.DUAL],
                test,
                minimum_headroom=minimum_headroom,
            )
            tuned_direction[seed] = tuned_gain
            tuning_records[seed] = {
                "objective": (
                    "equal_weight_behavioral_mse=0.5*affected_read_mse+0.5*unaffected_retention_mse"
                ),
                "selection_split": "validation",
                "validation_count_per_operation": validation_count,
                "training_steps_per_trial": steps,
                "constraints": constraint_records,
                "selected_models_reused_from_full_budget_trials": True,
                "test_direction_metric": (
                    "mean_add_invalidate_fraction_of_tied_to_oracle_gap_closed"
                ),
                "test_direction_value": tuned_gain,
                "test_direction_eligible_episodes": tuned_eligible,
                "test_direction_excluded_low_headroom": tuned_excluded,
                "direction_positive": (tuned_gain is not None and tuned_gain > 0.0),
            }

    bootstrap_samples = int(config["statistics"]["bootstrap_samples"])
    if bootstrap_samples <= 0:
        raise ValueError("Bootstrap samples must be positive.")

    asym_evaluable = _evaluable(asym_by_seed, seeds)
    noop_evaluable = _evaluable(noop_gain_by_seed, seeds)
    supersede_evaluable = _evaluable(supersede_by_seed, seeds)
    asym_interval = (
        _fixed_seed_episode_bootstrap(
            asym_by_seed,
            asym_operations_by_seed,
            samples=bootstrap_samples,
            seed=201,
        )
        if asym_evaluable
        else None
    )
    noop_interval = (
        _fixed_seed_episode_bootstrap(
            noop_gain_by_seed,
            noop_operations_by_seed,
            samples=bootstrap_samples,
            seed=205,
        )
        if noop_evaluable
        else None
    )
    preserve_interval = _fixed_seed_episode_bootstrap(
        preserve_by_seed,
        preserve_operations_by_seed,
        samples=bootstrap_samples,
        seed=202,
    )
    supersede_interval = (
        _fixed_seed_episode_bootstrap(
            supersede_by_seed,
            supersede_operations_by_seed,
            samples=bootstrap_samples,
            seed=203,
        )
        if supersede_evaluable
        else None
    )
    retention_interval = _fixed_seed_episode_bootstrap(
        retention_by_seed,
        retention_operations_by_seed,
        samples=bootstrap_samples,
        seed=204,
    )
    did_interval = _fixed_seed_did_episode_bootstrap(
        did_effects_by_seed,
        did_operations_by_seed,
        samples=bootstrap_samples,
        seed=206,
    )

    did_values = _finite_array(
        [did_by_seed[seed] for seed in seeds],
        "DID seed values",
    )
    asym_seed = (
        _finite_array(
            [asym_by_seed[seed].mean() for seed in seeds],
            "asymmetric seed values",
        )
        if asym_evaluable
        else None
    )
    retention_seed = _finite_array(
        [retention_by_seed[seed].mean() for seed in seeds],
        "retention seed values",
    )
    stats = config["statistics"]
    alpha = float(stats["alpha"])
    sesoi = float(stats["asymmetric_normalized_gain_sesoi"])
    preserve_margin = float(stats["preserve_absolute_equivalence_margin"])
    supersede_margin = float(stats["supersede_relative_equivalence_margin"])
    retention_margin = float(stats["retention_noninferiority_margin"])
    _finite_array(
        [
            alpha,
            sesoi,
            preserve_margin,
            supersede_margin,
            retention_margin,
        ],
        "statistical thresholds",
    )

    asym_sign_flip_p = (
        exact_sign_flip_test(asym_seed - sesoi, "greater") if asym_seed is not None else None
    )
    did_sign_flip_p = exact_sign_flip_test(did_values, "greater")
    retention_sign_flip_p = exact_sign_flip_test(
        retention_seed - retention_margin,
        "less",
    )
    asymmetric_supported = (
        asym_interval is not None
        and asym_sign_flip_p is not None
        and asym_interval.low > sesoi
        and asym_sign_flip_p <= alpha
    )
    preserve_equivalent = _equivalence_within(
        preserve_interval,
        preserve_margin,
    )
    supersede_equivalent = supersede_interval is not None and _equivalence_within(
        supersede_interval,
        supersede_margin,
    )
    did_supported = did_interval.low > 0.0 and did_sign_flip_p <= alpha
    retention_supported = (
        retention_interval.high <= retention_margin and retention_sign_flip_p <= alpha
    )
    tuned_consistent = (
        run_tuning
        and set(tuned_direction) == set(seeds)
        and all(value is not None and value > 0.0 for value in tuned_direction.values())
    )
    exact_main_seed_design = (
        not args.dry_run and len(seeds) == 8 and len(set(seeds)) == 8 and seeds == configured_seeds
    )
    confirmatory_eligible = _confirmatory_eligibility(
        dry_run=args.dry_run,
        seeds=seeds,
        configured_seeds=configured_seeds,
        run_tuning=run_tuning,
        tuning_record_count=len(tuning_records),
        asymmetric_evaluable=asym_evaluable,
        supersede_evaluable=supersede_evaluable,
    )
    supported = _h2_claim_supported(
        confirmatory_eligible=confirmatory_eligible,
        asymmetric_gain=asymmetric_supported,
        preserve_equivalence=preserve_equivalent,
        supersede_equivalence=supersede_equivalent,
        positive_interaction=did_supported,
        retention_noninferiority=retention_supported,
        tuned_direction_consistency=tuned_consistent,
    )

    expected_rows = len(seeds) * len(Operation) * test_count
    if len(rows) != expected_rows:
        raise AssertionError(f"Expected {expected_rows} test rows, got {len(rows)}.")
    expected_run_checkpoint_names = sorted(
        f"seed{seed}_{constraint}.pt" for seed in seeds for constraint in ("tied", "dual")
    )
    actual_checkpoint_names = sorted(str(entry["filename"]) for entry in checkpoint_entries)
    if actual_checkpoint_names != expected_run_checkpoint_names:
        raise AssertionError("Strict checkpoint filenames violate the E04 contract.")
    main_checkpoint_names = sorted(
        f"seed{seed}_{constraint}.pt"
        for seed in configured_seeds
        for constraint in ("tied", "dual")
    )
    exact_main_checkpoint_contract = (
        exact_main_seed_design
        and len(checkpoint_entries) == 16
        and actual_checkpoint_names == main_checkpoint_names
    )
    if exact_main_seed_design and not exact_main_checkpoint_contract:
        raise AssertionError("Main E02 run did not produce all 16 strict checkpoints.")

    episode_metrics_path = run_dir / "episode_metrics.jsonl"
    _assert_finite_payload(rows, "episode_metrics")
    write_jsonl_strict(episode_metrics_path, rows)
    report = {
        "status": "PASS",
        "scientific_evidence": False,
        "execution": {
            "dry_run": bool(args.dry_run),
            "row_count": len(rows),
            "expected_row_count": expected_rows,
            "strict_checkpoint_count": len(checkpoint_entries),
            "unique_seeds": seeds,
            "tuning_executed": run_tuning,
        },
        "execution_integrity": {
            "run_mode": "dry_run" if args.dry_run else "main",
            "configured_seeds": configured_seeds,
            "executed_seeds": seeds,
            "exact_eight_paired_seed_design": exact_main_seed_design,
            "confirmatory_inference_eligible": confirmatory_eligible,
            "train_count_per_operation": train_count,
            "test_count_per_operation": test_count,
            "training_steps": steps,
            "episode_schedule": ("episode_index_major_operation_interleaved"),
            "operation_cycle": [operation.value for operation in Operation],
            "test_operation_counts_by_seed": _string_seed_keys(
                operation_counts_by_seed
            ),
            "expected_episode_metric_rows": expected_rows,
            "actual_episode_metric_rows": len(rows),
            "nonfinite_values": 0,
        },
        "parameter_matching": parameter_counts,
        "eligibility_counts_by_seed": _string_seed_keys(eligibility_counts),
        "inference_contract": {
            "episode_uncertainty": (
                "within_training_seed_episode_bootstrap_with_fixed_checkpoints"
            ),
            "training_replicate_uncertainty": (
                "seed_level_exact_sign_flip_without_episode_pseudoreplication"
            ),
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_confidence": 0.95,
            "seed_sign_patterns": 2 ** len(seeds),
            "exact_main_requires_unique_seed_count": 8,
        },
        "asymmetric_absolute_gain": {
            "metric": "fraction_of_tied_to_oracle_gap_closed",
            **_interval_fields(asym_interval),
            "sesoi": sesoi,
            "episode_ci_evaluable": asym_evaluable,
            "seed_values": (
                {
                    str(seed): float(asym_by_seed[seed].mean())
                    for seed in seeds
                }
                if asym_evaluable
                else None
            ),
            "seed_exact_sign_flip_p_greater_than_sesoi": (asym_sign_flip_p),
            "supported": asymmetric_supported,
        },
        "practical_gain_over_noop": {
            "metric": ("fraction_of_noop_to_oracle_gap_closed_by_dual"),
            **_interval_fields(noop_interval),
            "episode_ci_evaluable": noop_evaluable,
            "confirmatory": False,
        },
        "symmetric_guardrails": {
            "preserve_raw_effect": {
                **_interval_fields(preserve_interval),
                "equivalence_margin": preserve_margin,
                "equivalent": preserve_equivalent,
            },
            "supersede_relative_effect": {
                **_interval_fields(supersede_interval),
                "denominator": ("tied_to_oracle_affected_mse_headroom"),
                "minimum_headroom": minimum_headroom,
                "episode_ci_evaluable": supersede_evaluable,
                "equivalence_margin": supersede_margin,
                "equivalent": supersede_equivalent,
            },
        },
        "operation_interaction": {
            "seed_values": _string_seed_keys(did_by_seed),
            "metric": (
                "mean_raw_add_invalidate_improvement_minus_mean_raw_preserve_supersede_improvement"
            ),
            **_interval_fields(did_interval),
            "seed_exact_sign_flip_p_greater_than_zero": (did_sign_flip_p),
            "supported": did_supported,
        },
        "retention_noninferiority": {
            "estimate_dual_minus_tied": retention_interval.estimate,
            "ci95": [
                retention_interval.low,
                retention_interval.high,
            ],
            "margin": retention_margin,
            "seed_values": {
                str(seed): float(retention_by_seed[seed].mean())
                for seed in seeds
            },
            "seed_exact_sign_flip_p_less_than_margin": (retention_sign_flip_p),
            "supported": retention_supported,
        },
        "optimization_robustness": {
            "strict_recipe_is_confirmatory": True,
            "tuning_executed": run_tuning,
            "selection_objective": (
                "equal_weight_behavioral_mse=0.5*affected_read_mse+0.5*unaffected_retention_mse"
            ),
            "equal_budget_records_by_seed": _string_seed_keys(tuning_records),
            "tuned_add_invalidate_normalized_gain_by_seed": _string_seed_keys(
                tuned_direction
            ),
            "direction_consistent": tuned_consistent,
        },
        "strict_checkpoint_contract": {
            "schema_version": 1,
            "consumer": "e04_functional_mediation",
            "recipe": "strict",
            "filename_template": "seed{seed}_{tied|dual}.pt",
            "run_expected_count": len(seeds) * 2,
            "actual_count": len(checkpoint_entries),
            "main_expected_count": 16,
            "main_contract_complete": (exact_main_checkpoint_contract),
            "expected_main_filenames": main_checkpoint_names,
            "entries": checkpoint_entries,
        },
        "claim_gate": {
            "supported": supported,
            "confirmatory_run_eligible": confirmatory_eligible,
            "requires_asymmetric_gain": asymmetric_supported,
            "requires_symmetric_equivalence": (preserve_equivalent and supersede_equivalent),
            "requires_positive_interaction": did_supported,
            "requires_retention_noninferiority": retention_supported,
            "requires_tuned_direction_consistency": tuned_consistent,
        },
        "artifacts": {
            "episode_metrics": {
                "path": episode_metrics_path.name,
                "rows": len(rows),
                "sha256": sha256_file(episode_metrics_path),
            },
        },
        "dependency_lineage": dependencies,
    }
    _assert_finite_payload(report, "report")
    finalize_v61_run(
        context=run_context,
        report=report,
        main_eligible=confirmatory_eligible,
        full_eligible=(
            confirmatory_eligible
            and len(rows) == 16384
            and exact_main_checkpoint_contract
        ),
    )
    print(f"[{EXPERIMENT_ID}] PASS: {run_dir}")


if __name__ == "__main__":
    main()
