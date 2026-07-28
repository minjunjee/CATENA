"""Controlled approximate-rank stress used by the prospective E24a protocol."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch

from catena.models.operator_controllers import (
    LowRankOperatorController,
    parameter_count,
)
from catena.training.postcore import train_matrix_controller

SPECTRUM_FAMILIES = (
    "exponential",
    "power_law",
    "low_rank_plus_noise",
)


@dataclass(frozen=True, slots=True)
class SpectrumSpec:
    """One outcome-independent spectrum construction."""

    spectrum_id: str
    family: str
    parameter: float
    base_rank: int | None
    split: str

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON-safe registered descriptor."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class ApproximateRankResult:
    """Rows and factors produced by the direct per-target diagnostic."""

    raw_rows: list[dict[str, Any]]
    seed_rows: list[dict[str, Any]]
    learned_factors: dict[str, dict[str, torch.Tensor]]


@dataclass(frozen=True, slots=True)
class SpectrumInstance:
    """One descriptor/target pair registered before learner optimization."""

    instance_id: str
    seed: int
    spec: SpectrumSpec
    descriptor: torch.Tensor
    singular_values: torch.Tensor
    target: torch.Tensor

    def manifest_dict(self) -> dict[str, Any]:
        """Return outcome-independent instance metadata for the data manifest."""

        return {
            "instance_id": self.instance_id,
            "seed": self.seed,
            **self.spec.as_dict(),
            "descriptor": [float(value) for value in self.descriptor],
            "shared_basis_namespace": f"seed_{self.seed}/shared_spectrum_basis",
        }


@dataclass(frozen=True, slots=True)
class SpectrumFamilyFold:
    """One leave-one-spectrum-family-out split for one registered seed."""

    fold_id: str
    seed: int
    held_out_family: str
    training_families: tuple[str, ...]
    train_instance_ids: tuple[str, ...]
    test_instance_ids: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "fold_id": self.fold_id,
            "seed": self.seed,
            "held_out_family": self.held_out_family,
            "training_families": list(self.training_families),
            "train_instance_ids": list(self.train_instance_ids),
            "test_instance_ids": list(self.test_instance_ids),
            "outcome_independent": True,
        }


@dataclass(frozen=True, slots=True)
class OodPredictionBundle:
    """Predictions/checkpoints created before held-out target scoring."""

    prediction_rows: list[dict[str, Any]]
    predictions: dict[str, torch.Tensor]
    checkpoint_payloads: dict[str, dict[str, Any]]


@dataclass(frozen=True, slots=True)
class OodScoreResult:
    """Artifact-ready primary OOD learned-controller evaluation."""

    raw_rows: list[dict[str, Any]]
    seed_rows: list[dict[str, Any]]
    fold_rows: list[dict[str, Any]]
    assessment: dict[str, Any]


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _stable_seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def registered_spectrum_specs(
    config: Mapping[str, Any],
    *,
    dry_run: bool,
) -> tuple[SpectrumSpec, ...]:
    """Build the frozen spectrum grid without consulting any outcomes."""

    design = _mapping(config["design"], label="design")
    spectra = _mapping(design["spectra"], label="design.spectra")
    low_rank = _mapping(
        spectra["low_rank_plus_noise"],
        label="design.spectra.low_rank_plus_noise",
    )
    if dry_run:
        override = _mapping(config["dry_run_overrides"], label="dry_run_overrides")
        exponential = tuple(float(value) for value in override["exponential_tau"])
        power_law = tuple(float(value) for value in override["power_law_exponent"])
        base_ranks = tuple(int(value) for value in override["low_rank_base_ranks"])
        tail_noise = tuple(float(value) for value in override["low_rank_tail_noise"])
    else:
        exponential = tuple(float(value) for value in spectra["exponential_tau"])
        power_law = tuple(float(value) for value in spectra["power_law_exponent"])
        base_ranks = tuple(int(value) for value in low_rank["base_ranks"])
        tail_noise = tuple(float(value) for value in low_rank["tail_noise"])

    specs: list[SpectrumSpec] = []
    for tau in exponential:
        specs.append(
            SpectrumSpec(
                spectrum_id=f"exponential_tau_{tau:g}",
                family="exponential",
                parameter=tau,
                base_rank=None,
                split="construction_spectrum_stress",
            )
        )
    for exponent in power_law:
        specs.append(
            SpectrumSpec(
                spectrum_id=f"power_law_exponent_{exponent:g}",
                family="power_law",
                parameter=exponent,
                base_rank=None,
                split="construction_spectrum_stress",
            )
        )
    for base_rank in base_ranks:
        for noise in tail_noise:
            split = "exact_rank_reference" if noise == 0.0 else "construction_spectrum_stress"
            specs.append(
                SpectrumSpec(
                    spectrum_id=f"low_rank_{base_rank}_tail_{noise:g}",
                    family="low_rank_plus_noise",
                    parameter=noise,
                    base_rank=base_rank,
                    split=split,
                )
            )
    if not specs or len({spec.spectrum_id for spec in specs}) != len(specs):
        raise RuntimeError("E24a spectrum registry is empty or contains duplicate IDs")
    return tuple(specs)


def singular_values(
    *,
    dimension: int,
    spec: SpectrumSpec,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Construct one Frobenius-normalized registered singular-value profile."""

    if dimension <= 0:
        raise ValueError("dimension must be positive")
    index = torch.arange(dimension, dtype=torch.float64, device=device)
    if spec.family == "exponential":
        if spec.parameter <= 0.0:
            raise ValueError("exponential tau must be positive")
        values = torch.exp(-index / spec.parameter)
    elif spec.family == "power_law":
        if spec.parameter <= 0.0:
            raise ValueError("power-law exponent must be positive")
        values = torch.pow(index + 1.0, -spec.parameter)
    elif spec.family == "low_rank_plus_noise":
        if spec.base_rank is None or not 0 < spec.base_rank <= dimension:
            raise ValueError("low-rank construction requires a valid base rank")
        if spec.parameter < 0.0:
            raise ValueError("tail noise cannot be negative")
        rank = spec.base_rank
        head = torch.exp(-index[:rank] / float(rank))
        tail_index = index[rank:] - float(rank)
        tail = spec.parameter * torch.exp(-tail_index / float(rank))
        values = torch.cat((head, tail))
    else:
        raise ValueError(f"Unknown spectrum family: {spec.family!r}")
    norm = torch.linalg.vector_norm(values)
    if not bool(torch.isfinite(norm)) or float(norm) <= 0.0:
        raise RuntimeError("E24a produced an invalid singular-value profile")
    normalized: torch.Tensor = values / norm
    return normalized


def spectrum_statistics(values: torch.Tensor) -> dict[str, float]:
    """Return energy-entropy effective rank and classical stable rank."""

    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("singular values must be a nonempty vector")
    energy = values.square()
    total = energy.sum()
    maximum = energy.max()
    if float(total) <= 0.0 or float(maximum) <= 0.0:
        raise ValueError("singular values must contain positive energy")
    probabilities = energy / total
    positive = probabilities[probabilities > 0.0]
    effective_rank = torch.exp(-(positive * positive.log()).sum())
    stable_rank = total / maximum
    return {
        "effective_rank": float(effective_rank),
        "stable_rank": float(stable_rank),
    }


def normalized_oracle_floor(values: torch.Tensor, rank: int) -> float:
    """Return the Eckart--Young tail energy relative to total target energy."""

    if rank < 0:
        raise ValueError("rank cannot be negative")
    total = values.square().sum()
    if float(total) <= 0.0:
        raise ValueError("singular values have no energy")
    bounded_rank = min(rank, int(values.numel()))
    return float(values[bounded_rank:].square().sum() / total)


def _orthogonal_matrix(dimension: int, *, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    draw = torch.randn(
        (dimension, dimension),
        dtype=torch.float64,
        generator=generator,
    )
    q_matrix, r_matrix = torch.linalg.qr(draw)
    signs = torch.where(
        torch.diagonal(r_matrix) >= 0.0,
        torch.ones(dimension, dtype=torch.float64),
        -torch.ones(dimension, dtype=torch.float64),
    )
    result: torch.Tensor = q_matrix * signs
    return result


def population_operator(values: torch.Tensor, *, seed: int) -> torch.Tensor:
    """Embed a spectrum in deterministic independent left/right bases."""

    dimension = int(values.numel())
    left = _orthogonal_matrix(dimension, seed=_stable_seed(seed, "left"))
    right = _orthogonal_matrix(dimension, seed=_stable_seed(seed, "right"))
    result: torch.Tensor = (left * values.to(dtype=torch.float64).unsqueeze(0)) @ right.mT
    return result


def learned_truncated_approximations(
    target: torch.Tensor,
    *,
    ranks: Sequence[int],
    observation_count: int,
    relative_observation_noise: float,
    seed: int,
) -> dict[int, torch.Tensor]:
    """Fit truncated SVDs to a noisy empirical population estimate."""

    if target.ndim != 2 or target.shape[0] != target.shape[1]:
        raise ValueError("target operator must be square")
    if observation_count <= 0:
        raise ValueError("observation_count must be positive")
    if relative_observation_noise <= 0.0:
        raise ValueError("relative_observation_noise must be positive")
    dimension = int(target.shape[0])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    noise = torch.randn(
        target.shape,
        dtype=target.dtype,
        device=target.device,
        generator=generator,
    )
    noise = noise / torch.linalg.matrix_norm(noise)
    empirical = target + (
        noise
        * torch.linalg.matrix_norm(target)
        * relative_observation_noise
        / math.sqrt(observation_count)
    )
    left, singular, right_h = torch.linalg.svd(empirical, full_matrices=False)
    approximations: dict[int, torch.Tensor] = {}
    for rank in ranks:
        if not 0 < rank <= dimension:
            raise ValueError(f"rank {rank} is outside [1, {dimension}]")
        approximations[rank] = (left[:, :rank] * singular[:rank].unsqueeze(0)) @ right_h[:rank, :]
    return approximations


def normalized_error(estimate: torch.Tensor, target: torch.Tensor) -> float:
    """Return squared Frobenius error normalized by target energy."""

    denominator = target.square().sum()
    if float(denominator) <= 0.0:
        raise ValueError("target operator has no energy")
    return float((estimate - target).square().sum() / denominator)


def epsilon_minimal_rank(
    errors: Mapping[int, float],
    *,
    epsilon: float,
) -> int | None:
    """Return the smallest registered rank meeting relative residual epsilon."""

    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must lie strictly between zero and one")
    for rank in sorted(errors):
        error = float(errors[rank])
        if error < 0.0 or not math.isfinite(error):
            raise ValueError("epsilon-minimal errors must be finite and nonnegative")
        if math.sqrt(error) <= epsilon:
            return rank
    return None


def spectrum_descriptor(
    values: torch.Tensor,
    *,
    config: Mapping[str, Any],
) -> torch.Tensor:
    """Build the frozen family-agnostic 16-D spectrum descriptor."""

    learning = _mapping(config["learning"], label="learning")
    descriptor_config = _mapping(
        learning["descriptor"],
        label="learning.descriptor",
    )
    indices = tuple(int(value) for value in descriptor_config["log_singular_value_indices"])
    energy_ranks = tuple(int(value) for value in descriptor_config["cumulative_energy_ranks"])
    if max(indices) >= values.numel() or max(energy_ranks) > values.numel():
        raise ValueError("E24a descriptor anchors exceed spectrum dimension")
    log_values = torch.log(values.clamp_min(1e-12))
    energy = values.square()
    total = energy.sum()
    cumulative = torch.cumsum(energy, dim=0) / total
    statistics = spectrum_statistics(values)
    descriptor = torch.tensor(
        [
            *[float(log_values[index]) for index in indices],
            *[float(cumulative[rank - 1]) for rank in energy_ranks],
            statistics["effective_rank"] / float(values.numel()),
            statistics["stable_rank"] / float(values.numel()),
        ],
        dtype=torch.float32,
    )
    expected_dimension = int(descriptor_config["dimension"])
    if descriptor.numel() != expected_dimension:
        raise RuntimeError(
            f"E24a descriptor dimension changed: {descriptor.numel()} != {expected_dimension}"
        )
    if not bool(torch.isfinite(descriptor).all()):
        raise RuntimeError("E24a produced a non-finite spectrum descriptor")
    return descriptor


def build_spectrum_instances(
    config: Mapping[str, Any],
    *,
    dry_run: bool,
    device: torch.device,
) -> tuple[SpectrumInstance, ...]:
    """Build descriptors and clean targets with one shared basis per seed."""

    if device.type != "cpu":
        raise ValueError("E24a spectrum instances require deterministic CPU")
    design = _mapping(config["design"], label="design")
    dimension = int(design["dimension"])
    seeds: tuple[int, ...]
    if dry_run:
        override = _mapping(config["dry_run_overrides"], label="dry_run_overrides")
        seeds = (int(override["seed"]),)
    else:
        seeds = tuple(int(value) for value in config["seeds"])
    specs = registered_spectrum_specs(config, dry_run=dry_run)
    instances: list[SpectrumInstance] = []
    for seed in seeds:
        basis_seed = _stable_seed(seed, "shared_spectrum_basis")
        for spec in specs:
            values = singular_values(
                dimension=dimension,
                spec=spec,
                device=device,
            )
            descriptor = spectrum_descriptor(values, config=config)
            target = population_operator(values, seed=basis_seed).to(
                dtype=torch.float32,
                device=device,
            )
            instances.append(
                SpectrumInstance(
                    instance_id=f"seed_{seed}/{spec.spectrum_id}",
                    seed=seed,
                    spec=spec,
                    descriptor=descriptor,
                    singular_values=values.cpu(),
                    target=target.cpu(),
                )
            )
    if len({instance.instance_id for instance in instances}) != len(instances):
        raise RuntimeError("E24a spectrum instances contain duplicate IDs")
    return tuple(instances)


def build_spectrum_family_folds(
    instances: Sequence[SpectrumInstance],
) -> tuple[SpectrumFamilyFold, ...]:
    """Precompute leave-one-family-out membership without target outcomes."""

    seeds = sorted({instance.seed for instance in instances})
    folds: list[SpectrumFamilyFold] = []
    for seed in seeds:
        seed_instances = [instance for instance in instances if instance.seed == seed]
        observed_families = {instance.spec.family for instance in seed_instances}
        if observed_families != set(SPECTRUM_FAMILIES):
            raise ValueError(f"E24a seed {seed} lacks a registered spectrum family")
        for held_out in SPECTRUM_FAMILIES:
            training_families = tuple(family for family in SPECTRUM_FAMILIES if family != held_out)
            train_ids = tuple(
                instance.instance_id
                for instance in seed_instances
                if instance.spec.family in training_families
            )
            test_ids = tuple(
                instance.instance_id
                for instance in seed_instances
                if instance.spec.family == held_out
            )
            if not train_ids or not test_ids or set(train_ids) & set(test_ids):
                raise RuntimeError("E24a family fold is empty or overlapping")
            folds.append(
                SpectrumFamilyFold(
                    fold_id=f"seed_{seed}/holdout_{held_out}",
                    seed=seed,
                    held_out_family=held_out,
                    training_families=training_families,
                    train_instance_ids=train_ids,
                    test_instance_ids=test_ids,
                )
            )
    return tuple(folds)


def _tensor_sha256(tensor: torch.Tensor) -> str:
    contiguous = tensor.detach().cpu().contiguous()
    return hashlib.sha256(contiguous.numpy().tobytes()).hexdigest()


def train_ood_spectrum_predictors(
    config: Mapping[str, Any],
    *,
    instances: Sequence[SpectrumInstance],
    folds: Sequence[SpectrumFamilyFold],
    dry_run: bool,
    device: torch.device,
) -> OodPredictionBundle:
    """Train rank-factor learners and predict families excluded from training."""

    if device.type != "cpu":
        raise ValueError("E24a learned OOD path requires deterministic CPU")
    design = _mapping(config["design"], label="design")
    learning = _mapping(config["learning"], label="learning")
    model_config = _mapping(learning["model"], label="learning.model")
    training = _mapping(learning["training"], label="learning.training")
    evaluation = _mapping(learning["evaluation"], label="learning.evaluation")
    override = _mapping(config["dry_run_overrides"], label="dry_run_overrides")
    ranks = (
        tuple(int(value) for value in override["learned_ranks"])
        if dry_run
        else tuple(int(value) for value in design["controller_ranks"])
    )
    steps = int(override["training_steps"]) if dry_run else int(training["steps"])
    descriptor_dimension = int(
        _mapping(learning["descriptor"], label="learning.descriptor")["dimension"]
    )
    dimension = int(design["dimension"])
    by_id = {instance.instance_id: instance for instance in instances}
    prediction_rows: list[dict[str, Any]] = []
    predictions: dict[str, torch.Tensor] = {}
    checkpoint_payloads: dict[str, dict[str, Any]] = {}
    for fold in folds:
        training_instances = [by_id[instance_id] for instance_id in fold.train_instance_ids]
        test_instances = [by_id[instance_id] for instance_id in fold.test_instance_ids]
        train_descriptor = torch.stack([instance.descriptor for instance in training_instances])
        descriptor_mean = train_descriptor.mean(dim=0)
        descriptor_scale = train_descriptor.std(
            dim=0,
            unbiased=False,
        ).clamp_min(1e-6)
        standardized_train = (train_descriptor - descriptor_mean) / descriptor_scale
        train_targets = torch.stack([instance.target for instance in training_instances])
        test_descriptor = torch.stack([instance.descriptor for instance in test_instances])
        standardized_test = (test_descriptor - descriptor_mean) / descriptor_scale
        for rank in ranks:
            model_seed = _stable_seed(
                fold.seed,
                fold.held_out_family,
                rank,
                "model",
            )
            torch.manual_seed(model_seed)
            model = LowRankOperatorController(
                descriptor_dim=descriptor_dimension,
                dimension=dimension,
                rank=rank,
                hidden_dim=int(model_config["hidden_dim"]),
            )
            trace = train_matrix_controller(
                model=model,
                descriptors=standardized_train,
                targets=train_targets,
                steps=steps,
                batch_size=int(training["batch_size"]),
                learning_rate=float(training["learning_rate"]),
                weight_decay=float(training["weight_decay"]),
                device=device,
                seed=_stable_seed(
                    fold.seed,
                    fold.held_out_family,
                    rank,
                    "optimizer",
                ),
            )
            model.eval()
            with torch.inference_mode():
                predicted = model(standardized_test.to(device)).cpu()
            checkpoint_name = f"seed_{fold.seed}__holdout_{fold.held_out_family}__rank_{rank}.pt"
            checkpoint_payloads[checkpoint_name] = {
                "schema_version": 1,
                "experiment_id": "e24a_approximate_rank_stress",
                "run_mode": "DRY_RUN" if dry_run else "MAIN",
                "model_class": "LowRankOperatorController",
                "model_state_dict": {
                    name: value.detach().cpu().clone() for name, value in model.state_dict().items()
                },
                "seed": fold.seed,
                "held_out_family": fold.held_out_family,
                "training_families": list(fold.training_families),
                "controller_rank": rank,
                "descriptor_mean": descriptor_mean.clone(),
                "descriptor_scale": descriptor_scale.clone(),
                "training_instance_ids": list(fold.train_instance_ids),
                "test_instance_ids": list(fold.test_instance_ids),
                "test_family_seen_during_training": False,
                "test_outcomes_used_for_training": False,
                "optimizer": "AdamW",
                "optimizer_trace": {
                    "initial_loss": trace.initial_loss,
                    "final_loss": trace.final_loss,
                    "best_loss": trace.best_loss,
                    "steps": trace.steps,
                },
                "parameter_count": parameter_count(model),
                "model_seed": model_seed,
            }
            for index, instance in enumerate(test_instances):
                row_id = f"{fold.fold_id}/rank_{rank}/{instance.spec.spectrum_id}"
                prediction = predicted[index].clone()
                predictions[row_id] = prediction
                prediction_rows.append(
                    {
                        "schema_version": 1,
                        "row_id": row_id,
                        "fold_id": fold.fold_id,
                        "seed": fold.seed,
                        "held_out_family": fold.held_out_family,
                        "training_families": list(fold.training_families),
                        "instance_id": instance.instance_id,
                        "spectrum_id": instance.spec.spectrum_id,
                        "controller_rank": rank,
                        "checkpoint_name": checkpoint_name,
                        "prediction_tensor_sha256": _tensor_sha256(prediction),
                        "test_family_seen_during_training": False,
                        "test_outcome_used": False,
                        "prediction_written_before_outcome_join": True,
                        "evaluation_batch_size": int(evaluation["batch_size"]),
                    }
                )
    if len(predictions) != len(prediction_rows):
        raise RuntimeError("E24a OOD prediction rows are incomplete or duplicate")
    return OodPredictionBundle(
        prediction_rows=prediction_rows,
        predictions=predictions,
        checkpoint_payloads=checkpoint_payloads,
    )


def score_ood_spectrum_predictions(
    config: Mapping[str, Any],
    *,
    instances: Sequence[SpectrumInstance],
    folds: Sequence[SpectrumFamilyFold],
    bundle: OodPredictionBundle,
    dry_run: bool,
) -> OodScoreResult:
    """Join held-out targets after prediction and score learned rank transfer."""

    epsilon = float(
        _mapping(config["epsilon_minimal"], label="epsilon_minimal")["relative_residual_epsilon"]
    )
    gate = _mapping(config["claim_gate"], label="claim_gate")
    by_id = {instance.instance_id: instance for instance in instances}
    ranks = sorted({int(row["controller_rank"]) for row in bundle.prediction_rows})
    by_test_instance: dict[
        tuple[int, str, str],
        list[dict[str, Any]],
    ] = {}
    raw_rows: list[dict[str, Any]] = []
    for prediction_row in bundle.prediction_rows:
        if (
            prediction_row.get("test_outcome_used") is not False
            or prediction_row.get("test_family_seen_during_training") is not False
        ):
            raise ValueError("E24a OOD prediction violates family isolation")
        row_id = str(prediction_row["row_id"])
        instance = by_id[str(prediction_row["instance_id"])]
        rank = int(prediction_row["controller_rank"])
        prediction = bundle.predictions[row_id].to(dtype=torch.float64)
        target = instance.target.to(dtype=torch.float64)
        learned_error = normalized_error(prediction, target)
        oracle = normalized_oracle_floor(instance.singular_values, rank)
        if learned_error + 1e-8 < oracle:
            raise RuntimeError("E24a OOD learner fell below its best-rank oracle")
        statistics = spectrum_statistics(instance.singular_values)
        row = {
            "schema_version": 1,
            "experiment_id": "e24a_approximate_rank_stress",
            "run_mode": "DRY_RUN" if dry_run else "MAIN",
            **prediction_row,
            **instance.spec.as_dict(),
            "dimension": int(instance.target.shape[-1]),
            "descriptor": [float(value) for value in instance.descriptor],
            **statistics,
            "primary_estimand": True,
            "learner": "descriptor_conditioned_low_rank_controller",
            "fold_rule": "leave_one_spectrum_family_out",
            "normalized_oracle_floor": oracle,
            "normalized_ood_learned_error": learned_error,
            "ood_learned_excess_over_oracle": max(
                0.0,
                learned_error - oracle,
            ),
            "relative_oracle_residual": math.sqrt(oracle),
            "relative_ood_learned_residual": math.sqrt(learned_error),
            "epsilon": epsilon,
        }
        raw_rows.append(row)
        key = (
            instance.seed,
            instance.spec.family,
            instance.spec.spectrum_id,
        )
        by_test_instance.setdefault(key, []).append(row)

    instance_summaries: list[dict[str, Any]] = []
    for (seed, family, spectrum_id), rows in sorted(by_test_instance.items()):
        errors = {
            int(row["controller_rank"]): float(row["normalized_ood_learned_error"]) for row in rows
        }
        oracle_errors = {
            int(row["controller_rank"]): float(row["normalized_oracle_floor"]) for row in rows
        }
        if sorted(errors) != ranks:
            raise RuntimeError("E24a OOD instance lacks a registered rank")
        learned_minimal = epsilon_minimal_rank(errors, epsilon=epsilon)
        oracle_minimal = epsilon_minimal_rank(
            oracle_errors,
            epsilon=epsilon,
        )
        match = (
            learned_minimal is not None
            and oracle_minimal is not None
            and learned_minimal == oracle_minimal
        )
        instance_summaries.append(
            {
                "seed": seed,
                "held_out_family": family,
                "spectrum_id": spectrum_id,
                "oracle_epsilon_minimal_rank": oracle_minimal,
                "ood_learned_epsilon_minimal_rank": learned_minimal,
                "ood_epsilon_minimal_rank_match": match,
                "oracle_unresolved": oracle_minimal is None,
                "learner_unresolved": learned_minimal is None,
                "mean_normalized_excess_over_oracle": (
                    sum(float(row["ood_learned_excess_over_oracle"]) for row in rows) / len(rows)
                ),
            }
        )
        for row in rows:
            row.update(
                {
                    "oracle_epsilon_minimal_rank": oracle_minimal,
                    "ood_learned_epsilon_minimal_rank": learned_minimal,
                    "ood_epsilon_minimal_rank_match": match,
                }
            )

    fold_rows: list[dict[str, Any]] = []
    family_gate_status: dict[str, bool] = {}
    for fold in folds:
        summaries = [
            summary
            for summary in instance_summaries
            if summary["seed"] == fold.seed and summary["held_out_family"] == fold.held_out_family
        ]
        prediction_metrics = [row for row in raw_rows if row["fold_id"] == fold.fold_id]
        match_fraction = sum(
            bool(summary["ood_epsilon_minimal_rank_match"]) for summary in summaries
        ) / len(summaries)
        mean_excess = sum(
            float(row["ood_learned_excess_over_oracle"]) for row in prediction_metrics
        ) / len(prediction_metrics)
        fold_pass = match_fraction >= float(
            gate["minimum_ood_epsilon_rank_match_fraction"]
        ) and mean_excess <= float(gate["maximum_mean_normalized_excess_over_oracle"])
        fold_rows.append(
            {
                "schema_version": 1,
                "seed": fold.seed,
                "fold_id": fold.fold_id,
                "held_out_family": fold.held_out_family,
                "training_families": list(fold.training_families),
                "train_instance_count": len(fold.train_instance_ids),
                "test_instance_count": len(fold.test_instance_ids),
                "learned_rank_count": len(ranks),
                "ood_epsilon_minimal_rank_match_fraction": match_fraction,
                "mean_normalized_excess_over_oracle": mean_excess,
                "oracle_unresolved_count": sum(
                    bool(summary["oracle_unresolved"]) for summary in summaries
                ),
                "learner_unresolved_count": sum(
                    bool(summary["learner_unresolved"]) for summary in summaries
                ),
                "prospective_gate_pass": fold_pass,
                "test_family_seen_during_training": False,
            }
        )

    for family in SPECTRUM_FAMILIES:
        relevant = [row for row in fold_rows if row["held_out_family"] == family]
        family_gate_status[family] = bool(relevant) and all(
            bool(row["prospective_gate_pass"]) for row in relevant
        )
    all_families_present = set(family_gate_status) == set(SPECTRUM_FAMILIES)
    supported = all_families_present and all(family_gate_status.values())
    seed_rows: list[dict[str, Any]] = []
    for seed in sorted({instance.seed for instance in instances}):
        summaries = [summary for summary in instance_summaries if summary["seed"] == seed]
        learned_rows = [row for row in raw_rows if int(row["seed"]) == seed]
        seed_rows.append(
            {
                "schema_version": 1,
                "experiment_id": "e24a_approximate_rank_stress",
                "run_mode": "DRY_RUN" if dry_run else "MAIN",
                "seed": seed,
                "held_out_family_count": len(
                    {str(summary["held_out_family"]) for summary in summaries}
                ),
                "test_spectrum_instance_count": len(summaries),
                "ood_epsilon_minimal_rank_match_fraction": sum(
                    bool(summary["ood_epsilon_minimal_rank_match"]) for summary in summaries
                )
                / len(summaries),
                "mean_normalized_ood_learned_error": sum(
                    float(row["normalized_ood_learned_error"]) for row in learned_rows
                )
                / len(learned_rows),
                "mean_ood_learned_excess_over_oracle": sum(
                    float(row["ood_learned_excess_over_oracle"]) for row in learned_rows
                )
                / len(learned_rows),
                "oracle_unresolved_count": sum(
                    bool(summary["oracle_unresolved"]) for summary in summaries
                ),
                "learner_unresolved_count": sum(
                    bool(summary["learner_unresolved"]) for summary in summaries
                ),
            }
        )
    assessment = {
        "primary_estimand": (
            "descriptor-conditioned learned rank-controller transfer to a "
            "spectrum family absent from training"
        ),
        "fold_rule": "leave_one_spectrum_family_out",
        "family_gate_status": family_gate_status,
        "all_three_family_folds_present": all_families_present,
        "computed_supported": supported,
        "claim_disposition": (
            "DRY_RUN_NON_EVIDENCE"
            if dry_run
            else (
                "OOD_SPECTRUM_FAMILY_TRANSFER_SUPPORTED"
                if supported
                else "OOD_SPECTRUM_FAMILY_TRANSFER_NOT_SUPPORTED"
            )
        ),
        "test_outcomes_used_for_training": False,
        "predictions_written_before_test_outcome_join": True,
        "geometry_scope": "shared left/right basis within each registered seed",
    }
    return OodScoreResult(
        raw_rows=raw_rows,
        seed_rows=seed_rows,
        fold_rows=fold_rows,
        assessment=assessment,
    )


def run_approximate_rank_stress(
    config: Mapping[str, Any],
    *,
    dry_run: bool,
    device: torch.device,
) -> ApproximateRankResult:
    """Run the non-primary direct empirical-SVD diagnostic."""

    if device.type != "cpu":
        raise ValueError("E24a currently supports deterministic CPU evaluation only")
    design = _mapping(config["design"], label="design")
    learning = _mapping(config["learning"], label="learning")
    diagnostic = _mapping(
        learning["direct_empirical_svd_diagnostic"],
        label="learning.direct_empirical_svd_diagnostic",
    )
    epsilon_config = _mapping(config["epsilon_minimal"], label="epsilon_minimal")
    ranks = tuple(int(value) for value in design["controller_ranks"])
    dimension = int(design["dimension"])
    epsilon = float(epsilon_config["relative_residual_epsilon"])
    relative_noise = float(diagnostic["relative_observation_noise"])
    seeds: tuple[int, ...]
    if dry_run:
        override = _mapping(config["dry_run_overrides"], label="dry_run_overrides")
        seeds = (int(override["seed"]),)
        observation_count = int(override["diagnostic_observation_count"])
    else:
        seeds = tuple(int(value) for value in config["seeds"])
        observation_count = int(diagnostic["observation_count"])
    specs = registered_spectrum_specs(config, dry_run=dry_run)

    raw_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    factors: dict[str, dict[str, torch.Tensor]] = {}
    for seed in seeds:
        seed_spec_summaries: list[dict[str, Any]] = []
        for spec in specs:
            values = singular_values(dimension=dimension, spec=spec, device=device)
            statistics = spectrum_statistics(values)
            target = population_operator(
                values,
                seed=_stable_seed(seed, spec.spectrum_id, "population"),
            ).to(device=device)
            approximations = learned_truncated_approximations(
                target,
                ranks=ranks,
                observation_count=observation_count,
                relative_observation_noise=relative_noise,
                seed=_stable_seed(seed, spec.spectrum_id, "observations"),
            )
            oracle_errors = {rank: normalized_oracle_floor(values, rank) for rank in ranks}
            learned_errors = {
                rank: normalized_error(approximations[rank], target) for rank in ranks
            }
            for rank in ranks:
                if learned_errors[rank] + 1e-10 < oracle_errors[rank]:
                    raise RuntimeError(
                        "E24a learned rank-r approximation fell below its oracle floor"
                    )
            oracle_minimal = epsilon_minimal_rank(oracle_errors, epsilon=epsilon)
            learned_minimal = epsilon_minimal_rank(learned_errors, epsilon=epsilon)
            minimal_match = (
                oracle_minimal is not None
                and learned_minimal is not None
                and oracle_minimal == learned_minimal
            )
            for rank in ranks:
                oracle = oracle_errors[rank]
                learned = learned_errors[rank]
                raw_rows.append(
                    {
                        "schema_version": 1,
                        "experiment_id": "e24a_approximate_rank_stress",
                        "run_mode": "DRY_RUN" if dry_run else "MAIN",
                        "seed": seed,
                        **spec.as_dict(),
                        "dimension": dimension,
                        "controller_rank": rank,
                        "observation_count": observation_count,
                        "relative_observation_noise": relative_noise,
                        "primary_estimand": False,
                        "diagnostic_only": True,
                        "empirical_estimator_scope": (
                            "direct_per_target_factorization_no_family_transfer"
                        ),
                        **statistics,
                        "normalized_oracle_floor": oracle,
                        "normalized_learned_error": learned,
                        "learned_excess_over_oracle": max(0.0, learned - oracle),
                        "relative_oracle_residual": math.sqrt(oracle),
                        "relative_learned_residual": math.sqrt(learned),
                        "epsilon": epsilon,
                        "oracle_epsilon_minimal_rank": oracle_minimal,
                        "learned_epsilon_minimal_rank": learned_minimal,
                        "epsilon_minimal_rank_match": minimal_match,
                    }
                )
            key = f"seed_{seed}/{spec.spectrum_id}"
            highest_rank = max(ranks)
            left, singular, right_h = torch.linalg.svd(
                approximations[highest_rank],
                full_matrices=False,
            )
            factors[key] = {
                "left": left[:, :highest_rank].cpu(),
                "singular_values": singular[:highest_rank].cpu(),
                "right_h": right_h[:highest_rank, :].cpu(),
            }
            seed_spec_summaries.append(
                {
                    "split": spec.split,
                    "match": minimal_match,
                    "oracle_unresolved": oracle_minimal is None,
                    "learned_unresolved": learned_minimal is None,
                    "mean_excess": sum(
                        max(0.0, learned_errors[rank] - oracle_errors[rank]) for rank in ranks
                    )
                    / len(ranks),
                }
            )
        construction_stress = [
            summary
            for summary in seed_spec_summaries
            if summary["split"] == "construction_spectrum_stress"
        ]
        seed_rows.append(
            {
                "schema_version": 1,
                "experiment_id": "e24a_approximate_rank_stress",
                "run_mode": "DRY_RUN" if dry_run else "MAIN",
                "seed": seed,
                "spectrum_count": len(seed_spec_summaries),
                "construction_spectrum_count": len(construction_stress),
                "mean_learned_excess_over_oracle": sum(
                    float(summary["mean_excess"]) for summary in seed_spec_summaries
                )
                / len(seed_spec_summaries),
                "epsilon_minimal_rank_match_fraction": sum(
                    bool(summary["match"]) for summary in seed_spec_summaries
                )
                / len(seed_spec_summaries),
                "construction_spectrum_match_fraction": (
                    sum(bool(summary["match"]) for summary in construction_stress)
                    / len(construction_stress)
                    if construction_stress
                    else 0.0
                ),
                "learned_spectrum_family_transfer_evaluated": False,
                "oracle_unresolved_count": sum(
                    bool(summary["oracle_unresolved"]) for summary in seed_spec_summaries
                ),
                "learned_unresolved_count": sum(
                    bool(summary["learned_unresolved"]) for summary in seed_spec_summaries
                ),
            }
        )
    return ApproximateRankResult(
        raw_rows=raw_rows,
        seed_rows=seed_rows,
        learned_factors=factors,
    )
