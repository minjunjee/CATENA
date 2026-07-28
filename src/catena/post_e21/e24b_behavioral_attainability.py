"""Behavioral-attainability stress and outcome-isolated OOS prediction for E24b."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch

from catena.core.provenance_v61 import sha256_canonical_json
from catena.post_e21.e24_protocol import REGISTERED_PREDICTOR_FEATURES


@dataclass(frozen=True, slots=True)
class BehavioralCell:
    """One outcome-independent cell in the registered E24b Cartesian grid."""

    row_id: str
    seed: int
    demand_family: str
    controller_class: str
    geometry_block: str
    key_correlation: float
    target_operator_norm: float
    key_load_fraction: float
    noise_condition: str
    target_noise_factor: float
    teacher_noise_factor: float
    readout_lambda: float
    horizon: int
    readout: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class HoldoutFold:
    """A precomputed leave-one-level-out split containing no outcomes."""

    fold_id: str
    axis: str
    held_out_value: str
    train_row_ids: tuple[str, ...]
    test_row_ids: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "fold_id": self.fold_id,
            "axis": self.axis,
            "held_out_value": self.held_out_value,
            "train_row_ids": list(self.train_row_ids),
            "test_row_ids": list(self.test_row_ids),
        }


@dataclass(frozen=True, slots=True)
class PredictorOutput:
    """Precomputed test predictions and serializable fitted fold parameters."""

    predictions: list[dict[str, Any]]
    checkpoints: list[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class ReadoutMetrics:
    """Aggregate and component metrics for complementary readout row blocks."""

    behavioral_mse: float
    affected_mse: float
    unaffected_mse: float
    linearized_regret: float
    affected_linearized_regret: float
    unaffected_linearized_regret: float
    lipschitz_upper_bound: float
    affected_lipschitz_upper_bound: float
    unaffected_lipschitz_upper_bound: float


@dataclass(frozen=True, slots=True)
class TargetTeacherSequence:
    """Nominal, clean-application, and noisy-teacher operator sequences."""

    nominal_targets: tuple[torch.Tensor, ...]
    clean_targets: tuple[torch.Tensor, ...]
    teachers: tuple[torch.Tensor, ...]
    shared_basis: torch.Tensor
    key_transform: torch.Tensor
    affected_rows: int
    realized_mean_key_correlation: float
    realized_high_load_fraction: float
    clean_target_sha256: str
    teacher_sha256: str


@dataclass(frozen=True, slots=True)
class BehavioralSimulationResult:
    """Teacher-side predictor features separated from clean evaluation outcomes."""

    feature_rows: list[dict[str, Any]]
    bound_rows: list[dict[str, Any]]
    outcome_rows: list[dict[str, Any]]


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _stable_seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def _cell_id(
    *,
    seed: int,
    demand: str,
    controller: str,
    geometry: str,
    noise_condition: str,
    readout_lambda: float,
    horizon: int,
    readout: str,
) -> str:
    return (
        f"s{seed}__d-{demand}__c-{controller}__g-{geometry}"
        f"__n-{noise_condition}__l-{readout_lambda:g}"
        f"__h-{horizon}__r-{readout}"
    )


def _registered_noise_conditions(
    config: Mapping[str, Any],
    *,
    dry_run: bool,
) -> tuple[tuple[str, float, float], ...]:
    design = _mapping(config["design"], label="design")
    raw_conditions = design["noise_conditions"]
    conditions = tuple(
        (
            str(_mapping(item, label="design.noise_conditions[]")["label"]),
            float(_mapping(item, label="design.noise_conditions[]")["target_noise_factor"]),
            float(_mapping(item, label="design.noise_conditions[]")["teacher_noise_factor"]),
        )
        for item in raw_conditions
    )
    if dry_run:
        override = _mapping(config["dry_run_overrides"], label="dry_run_overrides")
        selected = tuple(str(value) for value in override["noise_conditions"])
        by_label = {condition[0]: condition for condition in conditions}
        if set(selected) - set(by_label):
            raise ValueError("E24b dry-run selected an unknown noise condition")
        conditions = tuple(by_label[label] for label in selected)
    if not conditions or len({condition[0] for condition in conditions}) != len(conditions):
        raise RuntimeError("E24b noise-condition registry is empty or duplicated")
    return conditions


def _registered_geometry_profiles(
    config: Mapping[str, Any],
) -> dict[str, tuple[float, float, float]]:
    design = _mapping(config["design"], label="design")
    profiles: dict[str, tuple[float, float, float]] = {}
    for raw_profile in design["geometry_profiles"]:
        profile = _mapping(
            raw_profile,
            label="design.geometry_profiles[]",
        )
        label = str(profile["label"])
        if label in profiles:
            raise ValueError("E24b geometry labels must be unique")
        profiles[label] = (
            float(profile["key_correlation"]),
            float(profile["target_operator_norm"]),
            float(profile["key_load_fraction"]),
        )
    if not profiles:
        raise ValueError("E24b geometry-profile registry is empty")
    return profiles


def registered_behavioral_cells(
    config: Mapping[str, Any],
    *,
    dry_run: bool,
) -> tuple[BehavioralCell, ...]:
    """Construct the complete frozen cell registry before any outcomes exist."""

    design = _mapping(config["design"], label="design")
    seeds: tuple[int, ...]
    if dry_run:
        source = _mapping(config["dry_run_overrides"], label="dry_run_overrides")
        seeds = (int(source["seed"]),)
    else:
        source = design
        seeds = tuple(int(value) for value in config["seeds"])
    noise_conditions = _registered_noise_conditions(config, dry_run=dry_run)
    lambdas = tuple(float(value) for value in source["readout_lambda"])
    horizons = tuple(int(value) for value in source["horizon"])
    demands = tuple(str(value) for value in source["demand_families"])
    controllers = tuple(str(value) for value in source["controller_classes"])
    geometries = tuple(str(value) for value in source["geometry_blocks"])
    geometry_profiles = _registered_geometry_profiles(config)
    if set(geometries) - set(geometry_profiles):
        raise ValueError("E24b cell registry selected an unknown geometry")
    readouts = tuple(str(value) for value in design["readout"])
    cells: list[BehavioralCell] = []
    for seed in seeds:
        for demand in demands:
            for controller in controllers:
                for geometry in geometries:
                    key_correlation, target_norm, key_load = geometry_profiles[geometry]
                    for (
                        noise_label,
                        target_noise,
                        teacher_noise,
                    ) in noise_conditions:
                        for readout_lambda in lambdas:
                            for horizon in horizons:
                                for readout in readouts:
                                    cells.append(
                                        BehavioralCell(
                                            row_id=_cell_id(
                                                seed=seed,
                                                demand=demand,
                                                controller=controller,
                                                geometry=geometry,
                                                noise_condition=noise_label,
                                                readout_lambda=readout_lambda,
                                                horizon=horizon,
                                                readout=readout,
                                            ),
                                            seed=seed,
                                            demand_family=demand,
                                            controller_class=controller,
                                            geometry_block=geometry,
                                            key_correlation=key_correlation,
                                            target_operator_norm=target_norm,
                                            key_load_fraction=key_load,
                                            noise_condition=noise_label,
                                            target_noise_factor=target_noise,
                                            teacher_noise_factor=teacher_noise,
                                            readout_lambda=readout_lambda,
                                            horizon=horizon,
                                            readout=readout,
                                        )
                                    )
    if not cells or len({cell.row_id for cell in cells}) != len(cells):
        raise RuntimeError("E24b cell registry is empty or contains duplicate IDs")
    return tuple(cells)


_HOLDOUT_ATTRIBUTES = {
    "demand_family": "demand_family",
    "controller_class": "controller_class",
    "geometry_block": "geometry_block",
}


def build_holdout_plan(
    cells: Sequence[BehavioralCell],
    *,
    holdout_axes: Sequence[str],
) -> tuple[HoldoutFold, ...]:
    """Precompute all leave-one-level-out folds using cell metadata only."""

    if not cells:
        raise ValueError("holdout planning requires at least one cell")
    row_ids = {cell.row_id for cell in cells}
    if len(row_ids) != len(cells):
        raise ValueError("holdout planning received duplicate row IDs")
    folds: list[HoldoutFold] = []
    for axis in holdout_axes:
        attribute = _HOLDOUT_ATTRIBUTES.get(axis)
        if attribute is None:
            raise ValueError(f"Unknown holdout axis: {axis!r}")
        values = sorted({str(getattr(cell, attribute)) for cell in cells})
        if len(values) < 2:
            raise ValueError(f"Holdout axis {axis!r} needs at least two levels")
        for value in values:
            test = tuple(cell.row_id for cell in cells if str(getattr(cell, attribute)) == value)
            train = tuple(cell.row_id for cell in cells if str(getattr(cell, attribute)) != value)
            if not train or not test or set(train) & set(test):
                raise RuntimeError("E24b holdout fold is empty or overlapping")
            if set(train) | set(test) != row_ids:
                raise RuntimeError("E24b holdout fold does not cover the registry")
            folds.append(
                HoldoutFold(
                    fold_id=f"leave_one_{axis}__{value}",
                    axis=axis,
                    held_out_value=value,
                    train_row_ids=train,
                    test_row_ids=test,
                )
            )
    return tuple(folds)


def holdout_plan_payload(folds: Sequence[HoldoutFold]) -> dict[str, Any]:
    """Return the immutable, outcome-free holdout plan."""

    payload = {
        "schema_version": 1,
        "outcome_independent": True,
        "fold_membership_precomputed_without_outcomes": True,
        "folds": [fold.as_dict() for fold in folds],
    }
    payload["plan_sha256"] = sha256_canonical_json(payload)
    return payload


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


def _sequence_sha256(sequence: Sequence[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for tensor in sequence:
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def _affected_row_count(dimension: int, fraction: float) -> int:
    affected = int(round(dimension * fraction))
    if not 0 < affected < dimension:
        raise ValueError("E24b needs nonempty affected and retained row blocks")
    return affected


def _equicorrelated_key_mixer(size: int, correlation: float) -> torch.Tensor:
    if size <= 0 or not 0.0 <= correlation < 1.0:
        raise ValueError("E24b key geometry is invalid")
    gram = (1.0 - correlation) * torch.eye(size, dtype=torch.float64) + correlation * torch.ones(
        (size, size), dtype=torch.float64
    )
    result: torch.Tensor = torch.linalg.cholesky(gram)
    return result


def _input_key_transform(
    *,
    dimension: int,
    correlation: float,
    load_fraction: float,
) -> tuple[torch.Tensor, float, float]:
    load_count = max(1, min(dimension, int(round(load_fraction * dimension))))
    correlation_factor = _equicorrelated_key_mixer(
        dimension,
        correlation,
    )
    loads = torch.full((dimension,), 0.25, dtype=torch.float64)
    loads[:load_count] = 1.0
    transform = loads[:, None] * correlation_factor
    covariance = transform @ transform.mT
    standard_deviation = torch.sqrt(torch.diagonal(covariance))
    correlation_matrix = covariance / (standard_deviation[:, None] * standard_deviation[None, :])
    off_diagonal = correlation_matrix[~torch.eye(dimension, dtype=torch.bool)]
    return (
        transform,
        float(off_diagonal.mean()),
        load_count / dimension,
    )


def _nominal_target_sequence(
    *,
    cell: BehavioralCell,
    dimension: int,
    affected_rows: int,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    common_left = _orthogonal_matrix(
        affected_rows,
        seed=_stable_seed(
            cell.seed,
            cell.demand_family,
            "common_left",
        ),
    )
    shared_basis = torch.eye(dimension, dtype=torch.float64)
    if cell.demand_family != "axis_commuting":
        shared_basis[:affected_rows, :affected_rows] = common_left
    targets: list[torch.Tensor] = []
    for step in range(cell.horizon):
        index = torch.arange(affected_rows, dtype=torch.float64)
        diagonal_values = torch.linspace(
            1.0,
            0.35,
            affected_rows,
            dtype=torch.float64,
        ) * (1.0 + 0.12 * torch.sin((step + 1.0) * (index + 1.0)))
        diagonal = torch.diag(diagonal_values)
        if cell.demand_family == "axis_commuting":
            left_basis = torch.eye(affected_rows, dtype=torch.float64)
        elif cell.demand_family == "common_rotated_commuting":
            left_basis = common_left
        elif cell.demand_family == "noncommuting":
            left_basis = _orthogonal_matrix(
                affected_rows,
                seed=_stable_seed(
                    cell.seed,
                    step,
                    "noncommuting_left",
                ),
            )
        else:
            raise ValueError(f"Unknown E24b demand family: {cell.demand_family!r}")
        active = left_basis @ diagonal @ left_basis.mT
        nominal = torch.zeros((dimension, dimension), dtype=torch.float64)
        nominal[:affected_rows, :affected_rows] = active
        desired_norm = cell.target_operator_norm
        nominal *= desired_norm / torch.linalg.matrix_norm(nominal)
        targets.append(nominal)
    return tuple(targets), shared_basis


def build_target_teacher_sequence(
    cell: BehavioralCell,
    *,
    dimension: int,
    affected_row_fraction: float,
) -> TargetTeacherSequence:
    """Build a clean row-sparse target and a separately corrupted teacher."""

    affected_rows = _affected_row_count(dimension, affected_row_fraction)
    nominal_targets, shared_basis = _nominal_target_sequence(
        cell=cell,
        dimension=dimension,
        affected_rows=affected_rows,
    )
    (
        key_transform,
        realized_correlation,
        realized_load,
    ) = _input_key_transform(
        dimension=dimension,
        correlation=cell.key_correlation,
        load_fraction=cell.key_load_fraction,
    )
    clean_targets: list[torch.Tensor] = []
    teachers: list[torch.Tensor] = []
    for step, nominal in enumerate(nominal_targets):
        clean = nominal.clone()
        if cell.target_noise_factor > 0.0:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(
                _stable_seed(
                    cell.seed,
                    cell.demand_family,
                    cell.target_noise_factor,
                    step,
                    "application_target_noise",
                )
            )
            perturbation = torch.randn(
                (affected_rows, dimension),
                dtype=torch.float64,
                generator=generator,
            )
            perturbation *= (
                cell.target_noise_factor
                * torch.linalg.matrix_norm(nominal)
                / torch.linalg.matrix_norm(perturbation)
            )
            clean[:affected_rows] += perturbation
        clean[affected_rows:] = 0.0
        teacher = clean.clone()
        if cell.teacher_noise_factor > 0.0:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(
                _stable_seed(
                    cell.seed,
                    cell.demand_family,
                    cell.target_noise_factor,
                    cell.teacher_noise_factor,
                    step,
                    "teacher_corruption",
                )
            )
            corruption = torch.randn(
                teacher.shape,
                dtype=torch.float64,
                generator=generator,
            )
            corruption *= (
                cell.teacher_noise_factor
                * torch.linalg.matrix_norm(clean)
                / torch.linalg.matrix_norm(corruption)
            )
            teacher += corruption
        clean_targets.append(clean)
        teachers.append(teacher)
    if any(float(target[affected_rows:].abs().max()) != 0.0 for target in clean_targets):
        raise RuntimeError("E24b clean target changed a retained row")
    return TargetTeacherSequence(
        nominal_targets=nominal_targets,
        clean_targets=tuple(clean_targets),
        teachers=tuple(teachers),
        shared_basis=shared_basis,
        key_transform=key_transform,
        affected_rows=affected_rows,
        realized_mean_key_correlation=realized_correlation,
        realized_high_load_fraction=realized_load,
        clean_target_sha256=_sequence_sha256(clean_targets),
        teacher_sha256=_sequence_sha256(teachers),
    )


def _row_metric_scales(
    *,
    dimension: int,
    affected_rows: int,
    readout_lambda: float,
) -> torch.Tensor:
    retained_rows = dimension - affected_rows
    values = torch.empty(dimension, dtype=torch.float64)
    values[:affected_rows] = math.sqrt(readout_lambda / affected_rows)
    values[affected_rows:] = math.sqrt((1.0 - readout_lambda) / retained_rows)
    return values


def _project_controller_sequence(
    targets: Sequence[torch.Tensor],
    *,
    controller_class: str,
    shared_basis: torch.Tensor,
    key_transform: torch.Tensor,
    affected_rows: int,
    readout_lambda: float,
) -> tuple[torch.Tensor, ...]:
    """Return the analytic row-weighted projection in one controller class."""

    if not targets:
        raise ValueError("E24b controller projection needs a target sequence")
    dimension = int(targets[0].shape[0])
    row_scales = _row_metric_scales(
        dimension=dimension,
        affected_rows=affected_rows,
        readout_lambda=readout_lambda,
    )
    projected: list[torch.Tensor] = []
    for target in targets:
        if controller_class == "fixed_diagonal":
            identity = torch.eye(dimension, dtype=torch.float64)
            components = torch.stack(
                [torch.outer(identity[:, index], identity[:, index]) for index in range(dimension)],
                dim=-1,
            )
        elif controller_class == "shared_basis_diagonal":
            components = torch.stack(
                [
                    torch.outer(
                        shared_basis[:, index],
                        shared_basis[:, index],
                    )
                    for index in range(dimension)
                ],
                dim=-1,
            )
        else:
            components = None
        if components is not None:
            right_transformed_components = torch.einsum(
                "ijk,jl->ilk",
                components,
                key_transform,
            )
            design = (row_scales[:, None, None] * right_transformed_components).reshape(
                dimension * dimension, dimension
            )
            response = ((row_scales[:, None] * target) @ key_transform).reshape(
                dimension * dimension
            )
            gram = design.mT @ design
            coefficients = torch.linalg.solve(
                gram,
                design.mT @ response,
            )
            controller = torch.einsum(
                "ijk,k->ij",
                components,
                coefficients,
            )
        elif controller_class == "rank8":
            weighted = (row_scales[:, None] * target) @ key_transform
            left, singular, right_h = torch.linalg.svd(
                weighted,
                full_matrices=False,
            )
            rank = min(8, dimension)
            weighted_projection = (left[:, :rank] * singular[:rank].unsqueeze(0)) @ right_h[
                :rank, :
            ]
            right_unweighted = torch.linalg.solve(
                key_transform.mT,
                weighted_projection.mT,
            ).mT
            controller = right_unweighted / row_scales[:, None]
        elif controller_class == "full":
            controller = target.clone()
        else:
            raise ValueError(f"Unknown E24b controller class: {controller_class!r}")
        projected.append(controller)
    return tuple(projected)


def _temporally_whitened_inputs(
    *,
    horizon: int,
    batch_size: int,
    dimension: int,
    seed: int,
    key_transform: torch.Tensor,
) -> torch.Tensor:
    if batch_size < horizon * dimension:
        raise ValueError("E24b exact temporal whitening requires batch >= horizon * dimension")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    permutation = torch.randperm(batch_size, generator=generator)
    sign_draw = torch.rand(
        horizon * dimension,
        dtype=torch.float64,
        generator=generator,
    )
    signs = torch.where(
        sign_draw >= 0.5,
        torch.ones_like(sign_draw),
        -torch.ones_like(sign_draw),
    )
    inputs = torch.zeros(
        (horizon, batch_size, dimension),
        dtype=torch.float64,
    )
    columns = torch.arange(dimension)
    scale = math.sqrt(batch_size)
    for step in range(horizon):
        start = step * dimension
        rows = permutation[start : start + dimension]
        inputs[step, rows, columns] = scale * signs[start : start + dimension]
    transformed: torch.Tensor = inputs @ key_transform.mT
    return transformed


def _roll_state(
    operators: Sequence[torch.Tensor],
    *,
    inputs: torch.Tensor,
    update_scale: float,
    recurrence_decay: float,
) -> torch.Tensor:
    if not operators or len(operators) != int(inputs.shape[0]):
        raise ValueError("E24b operator/input sequences must align")
    batch_size = int(inputs.shape[1])
    dimension = int(inputs.shape[2])
    state = torch.zeros((batch_size, dimension), dtype=torch.float64)
    for step, operator in enumerate(operators):
        state = recurrence_decay * state + update_scale * inputs[step] @ operator.mT
    return state


def _normalized_operator_residual(
    estimate: Sequence[torch.Tensor],
    reference: Sequence[torch.Tensor],
) -> float:
    if not estimate or len(estimate) != len(reference):
        raise ValueError("E24b operator residual requires aligned sequences")
    numerator = sum(
        float((left - right).square().sum())
        for left, right in zip(estimate, reference, strict=True)
    )
    denominator = sum(float(value.square().sum()) for value in reference)
    return numerator / max(
        denominator,
        torch.finfo(torch.float64).tiny,
    )


def _readout_metrics(
    target_state: torch.Tensor,
    controller_state: torch.Tensor,
    *,
    seed: int,
    readout: str,
    readout_lambda: float,
    affected_rows: int,
    nonlinear_residual_scale: float,
) -> ReadoutMetrics:
    dimension = int(target_state.shape[1])
    if not 0.0 <= readout_lambda <= 1.0:
        raise ValueError("readout lambda must lie in [0, 1]")
    retained_rows = dimension - affected_rows
    if affected_rows <= 0 or retained_rows <= 0:
        raise ValueError("E24b readout blocks must both be nonempty")
    affected_target_state = target_state[:, :affected_rows]
    affected_controller_state = controller_state[:, :affected_rows]
    retained_target_state = target_state[:, affected_rows:]
    retained_controller_state = controller_state[:, affected_rows:]
    affected_whitener = _orthogonal_matrix(
        affected_rows,
        seed=_stable_seed(seed, "affected_whitener"),
    )
    retained_whitener = _orthogonal_matrix(
        retained_rows,
        seed=_stable_seed(seed, "retained_whitener"),
    )
    affected_target_linear = affected_target_state @ affected_whitener.mT
    affected_controller_linear = affected_controller_state @ affected_whitener.mT
    retained_target_linear = retained_target_state @ retained_whitener.mT
    retained_controller_linear = retained_controller_state @ retained_whitener.mT
    if readout == "linear":
        affected_target = affected_target_linear
        affected_controller = affected_controller_linear
        unaffected_target = retained_target_linear
        unaffected_controller = retained_controller_linear
        affected_delta = affected_controller - affected_target
        unaffected_delta = unaffected_controller - unaffected_target
        affected_linearized = float(affected_delta.square().mean())
        unaffected_linearized = float(unaffected_delta.square().mean())
        affected_lipschitz = 1.0
        unaffected_lipschitz = 1.0
    elif readout == "fixed_nonlinear_mlp":
        if nonlinear_residual_scale <= 0.0:
            raise ValueError("E24b nonlinear residual scale must be positive")
        affected_first = _orthogonal_matrix(
            affected_rows,
            seed=_stable_seed(seed, "affected_mlp_first"),
        )
        affected_second = affected_first.mT
        retained_first = _orthogonal_matrix(
            retained_rows,
            seed=_stable_seed(seed, "retained_mlp_first"),
        )
        retained_second = retained_first.mT
        affected_generator = torch.Generator(device="cpu")
        affected_generator.manual_seed(_stable_seed(seed, "affected_mlp_bias"))
        retained_generator = torch.Generator(device="cpu")
        retained_generator.manual_seed(_stable_seed(seed, "retained_mlp_bias"))
        affected_bias = 0.1 * torch.randn(
            affected_rows,
            dtype=torch.float64,
            generator=affected_generator,
        )
        retained_bias = 0.1 * torch.randn(
            retained_rows,
            dtype=torch.float64,
            generator=retained_generator,
        )
        affected_target_hidden = torch.tanh(
            affected_target_linear @ affected_first.mT + affected_bias
        )
        affected_controller_hidden = torch.tanh(
            affected_controller_linear @ affected_first.mT + affected_bias
        )
        retained_target_hidden = torch.tanh(
            retained_target_linear @ retained_first.mT + retained_bias
        )
        retained_controller_hidden = torch.tanh(
            retained_controller_linear @ retained_first.mT + retained_bias
        )
        affected_target = (
            affected_target_linear
            + nonlinear_residual_scale * affected_target_hidden @ affected_second.mT
        )
        affected_controller = (
            affected_controller_linear
            + nonlinear_residual_scale * affected_controller_hidden @ affected_second.mT
        )
        unaffected_target = (
            retained_target_linear
            + nonlinear_residual_scale * retained_target_hidden @ retained_second.mT
        )
        unaffected_controller = (
            retained_controller_linear
            + nonlinear_residual_scale * retained_controller_hidden @ retained_second.mT
        )
        affected_midpoint_hidden = (
            0.5 * (affected_target_linear + affected_controller_linear) @ affected_first.mT
            + affected_bias
        )
        retained_midpoint_hidden = (
            0.5 * (retained_target_linear + retained_controller_linear) @ retained_first.mT
            + retained_bias
        )
        affected_linear_delta = affected_controller_linear - affected_target_linear
        retained_linear_delta = retained_controller_linear - retained_target_linear
        affected_hidden_delta = affected_linear_delta @ affected_first.mT
        retained_hidden_delta = retained_linear_delta @ retained_first.mT
        affected_jacobian_delta = (
            affected_linear_delta
            + nonlinear_residual_scale
            * (affected_hidden_delta * (1.0 - torch.tanh(affected_midpoint_hidden).square()))
            @ affected_second.mT
        )
        unaffected_jacobian_delta = (
            retained_linear_delta
            + nonlinear_residual_scale
            * (retained_hidden_delta * (1.0 - torch.tanh(retained_midpoint_hidden).square()))
            @ retained_second.mT
        )
        affected_linearized = float(affected_jacobian_delta.square().mean())
        unaffected_linearized = float(unaffected_jacobian_delta.square().mean())
        affected_lipschitz = 1.0 + nonlinear_residual_scale
        unaffected_lipschitz = 1.0 + nonlinear_residual_scale
    else:
        raise ValueError(f"Unknown E24b readout: {readout!r}")
    affected_mse = float((affected_controller - affected_target).square().mean())
    unaffected_mse = float((unaffected_controller - unaffected_target).square().mean())
    outcome = readout_lambda * affected_mse + (1.0 - readout_lambda) * unaffected_mse
    linearized = (
        readout_lambda * affected_linearized + (1.0 - readout_lambda) * unaffected_linearized
    )
    mean_state_error = float((controller_state - target_state).square().mean())
    affected_state_error = float(
        (affected_controller_state - affected_target_state).square().mean()
    )
    retained_state_error = float(
        (retained_controller_state - retained_target_state).square().mean()
    )
    affected_upper_bound = affected_lipschitz**2 * affected_state_error
    unaffected_upper_bound = unaffected_lipschitz**2 * retained_state_error
    upper_bound = (
        readout_lambda * affected_upper_bound + (1.0 - readout_lambda) * unaffected_upper_bound
    )
    if not all(
        math.isfinite(value) and value >= 0.0
        for value in (
            outcome,
            affected_mse,
            unaffected_mse,
            linearized,
            affected_linearized,
            unaffected_linearized,
            upper_bound,
            affected_upper_bound,
            unaffected_upper_bound,
            mean_state_error,
        )
    ):
        raise RuntimeError("E24b produced a non-finite behavioral metric")
    return ReadoutMetrics(
        behavioral_mse=outcome,
        affected_mse=affected_mse,
        unaffected_mse=unaffected_mse,
        linearized_regret=linearized,
        affected_linearized_regret=affected_linearized,
        unaffected_linearized_regret=unaffected_linearized,
        lipschitz_upper_bound=upper_bound,
        affected_lipschitz_upper_bound=affected_upper_bound,
        unaffected_lipschitz_upper_bound=unaffected_upper_bound,
    )


def precompute_controller_bounds(
    config: Mapping[str, Any],
    *,
    cells: Sequence[BehavioralCell],
    dry_run: bool,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Freeze clean-only controller-class bounds before any observed outcome."""

    if device.type != "cpu":
        raise ValueError("E24b currently supports deterministic CPU evaluation only")
    design = _mapping(config["design"], label="design")
    simulation = _mapping(config["simulation"], label="simulation")
    override = _mapping(config["dry_run_overrides"], label="dry_run_overrides")
    dimension = int(override["dimension"]) if dry_run else int(design["dimension"])
    batch_size = int(override["batch_size"]) if dry_run else int(design["batch_size"])
    update_scale = float(simulation["update_scale"])
    recurrence_decay = float(simulation["recurrence_decay"])
    affected_fraction = float(simulation["affected_row_fraction"])
    nonlinear_scale = float(simulation["nonlinear_residual_scale"])
    clean_sequence_cache: dict[
        tuple[int, str, float, int, float, float, float],
        tuple[TargetTeacherSequence, torch.Tensor, torch.Tensor],
    ] = {}
    clean_projection_cache: dict[
        tuple[int, str, float, int, float, float, float, str, float],
        tuple[torch.Tensor, float],
    ] = {}
    bound_rows: list[dict[str, Any]] = []
    for cell in cells:
        clean_key = (
            cell.seed,
            cell.demand_family,
            cell.target_noise_factor,
            cell.horizon,
            cell.key_correlation,
            cell.target_operator_norm,
            cell.key_load_fraction,
        )
        cached_sequence = clean_sequence_cache.get(clean_key)
        if cached_sequence is None:
            target_teacher = build_target_teacher_sequence(
                cell,
                dimension=dimension,
                affected_row_fraction=affected_fraction,
            )
            inputs = _temporally_whitened_inputs(
                horizon=cell.horizon,
                batch_size=batch_size,
                dimension=dimension,
                seed=_stable_seed(
                    cell.seed,
                    cell.demand_family,
                    cell.horizon,
                    "exact_temporal_inputs",
                ),
                key_transform=target_teacher.key_transform,
            )
            clean_state = _roll_state(
                target_teacher.clean_targets,
                inputs=inputs,
                update_scale=update_scale,
                recurrence_decay=recurrence_decay,
            )
            cached_sequence = target_teacher, inputs, clean_state
            clean_sequence_cache[clean_key] = cached_sequence
        target_teacher, inputs, clean_state = cached_sequence
        projection_key = (
            *clean_key,
            cell.controller_class,
            cell.readout_lambda,
        )
        cached_projection = clean_projection_cache.get(projection_key)
        if cached_projection is None:
            analytic_clean_controllers = _project_controller_sequence(
                target_teacher.clean_targets,
                controller_class=cell.controller_class,
                shared_basis=target_teacher.shared_basis,
                key_transform=target_teacher.key_transform,
                affected_rows=target_teacher.affected_rows,
                readout_lambda=cell.readout_lambda,
            )
            analytic_clean_state = _roll_state(
                analytic_clean_controllers,
                inputs=inputs,
                update_scale=update_scale,
                recurrence_decay=recurrence_decay,
            )
            affected_state_floor = float(
                (
                    analytic_clean_state[:, : target_teacher.affected_rows]
                    - clean_state[:, : target_teacher.affected_rows]
                )
                .square()
                .mean()
            )
            retained_state_floor = float(
                (
                    analytic_clean_state[:, target_teacher.affected_rows :]
                    - clean_state[:, target_teacher.affected_rows :]
                )
                .square()
                .mean()
            )
            controller_state_lower_bound = (
                cell.readout_lambda * affected_state_floor
                + (1.0 - cell.readout_lambda) * retained_state_floor
            )
            cached_projection = (
                analytic_clean_state,
                controller_state_lower_bound,
            )
            clean_projection_cache[projection_key] = cached_projection
        analytic_clean_state, controller_state_lower_bound = cached_projection
        controller_lower_bound = controller_state_lower_bound * (
            (1.0 - nonlinear_scale) ** 2 if cell.readout == "fixed_nonlinear_mlp" else 1.0
        )
        readout_seed = _stable_seed(
            cell.seed,
            cell.readout,
            "structural_readout",
        )
        analytic_clean_metrics = _readout_metrics(
            clean_state,
            analytic_clean_state,
            seed=readout_seed,
            readout=cell.readout,
            readout_lambda=cell.readout_lambda,
            affected_rows=target_teacher.affected_rows,
            nonlinear_residual_scale=nonlinear_scale,
        )
        analytic_error = analytic_clean_metrics.behavioral_mse
        if analytic_error + 1e-10 < controller_lower_bound:
            raise RuntimeError("E24b clean same-class projection violated its analytic bound")
        bound_rows.append(
            {
                "schema_version": 1,
                "experiment_id": "e24b_behavioral_attainability_stress",
                "run_mode": "DRY_RUN" if dry_run else "MAIN",
                "row_id": cell.row_id,
                "seed": cell.seed,
                "demand_family": cell.demand_family,
                "controller_class": cell.controller_class,
                "geometry_block": cell.geometry_block,
                "target_noise_factor": cell.target_noise_factor,
                "readout": cell.readout,
                "readout_lambda": cell.readout_lambda,
                "horizon": cell.horizon,
                "affected_row_count": target_teacher.affected_rows,
                "retained_row_count": dimension - target_teacher.affected_rows,
                "key_correlation": cell.key_correlation,
                "target_operator_norm": cell.target_operator_norm,
                "key_load_fraction": cell.key_load_fraction,
                "clean_target_sequence_sha256": (target_teacher.clean_target_sha256),
                "controller_specific_state_mse_lower_bound": (controller_state_lower_bound),
                "controller_specific_clean_target_analytic_behavioral_lower_bound": (
                    controller_lower_bound
                ),
                "clean_oracle_attainable_behavioral_error": analytic_error,
                "lower_bound_source": ("clean_target_same_class_row_key_weighted_projection"),
                "bound_frozen_before_observed_application_outcome": True,
                "observed_application_outcome_used_in_lower_bound": False,
                "predictor_feature_used": False,
            }
        )
    if len(bound_rows) != len(cells):
        raise RuntimeError("E24b did not precompute every controller bound")
    if len({str(row["row_id"]) for row in bound_rows}) != len(bound_rows):
        raise RuntimeError("E24b precomputed duplicate controller-bound IDs")
    full_oracle_error = [
        float(row["clean_oracle_attainable_behavioral_error"])
        for row in bound_rows
        if row["controller_class"] == "full"
    ]
    if full_oracle_error and max(full_oracle_error) > 1e-20:
        raise RuntimeError("E24b clean-target full controller failed to attain zero error")
    return bound_rows


def simulate_behavioral_rows(
    config: Mapping[str, Any],
    *,
    cells: Sequence[BehavioralCell],
    dry_run: bool,
    device: torch.device,
    precomputed_bound_rows: Sequence[Mapping[str, Any]] | None = None,
) -> BehavioralSimulationResult:
    """Build teacher-only features and separately score clean-target outcomes."""

    if device.type != "cpu":
        raise ValueError("E24b currently supports deterministic CPU evaluation only")
    design = _mapping(config["design"], label="design")
    simulation = _mapping(config["simulation"], label="simulation")
    predictor = _mapping(config["predictor"], label="predictor")
    override = _mapping(config["dry_run_overrides"], label="dry_run_overrides")
    dimension = int(override["dimension"]) if dry_run else int(design["dimension"])
    batch_size = int(override["batch_size"]) if dry_run else int(design["batch_size"])
    update_scale = float(simulation["update_scale"])
    recurrence_decay = float(simulation["recurrence_decay"])
    affected_fraction = float(simulation["affected_row_fraction"])
    nonlinear_scale = float(simulation["nonlinear_residual_scale"])
    outcome_floor = float(predictor["outcome_floor"])
    sequence_cache: dict[
        tuple[int, str, str, float, float, int],
        tuple[
            TargetTeacherSequence,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ] = {}
    controller_cache: dict[
        tuple[int, str, str, float, float, int, str],
        tuple[torch.Tensor, float],
    ] = {}
    feature_rows: list[dict[str, Any]] = []
    if precomputed_bound_rows is None:
        precomputed_bound_rows = precompute_controller_bounds(
            config,
            cells=cells,
            dry_run=dry_run,
            device=device,
        )
    bound_rows = [dict(row) for row in precomputed_bound_rows]
    bound_by_id = {str(row["row_id"]): row for row in bound_rows}
    expected_row_ids = {cell.row_id for cell in cells}
    if len(bound_by_id) != len(bound_rows) or set(bound_by_id) != expected_row_ids:
        raise ValueError("E24b frozen controller bounds do not match the registered cells")
    if any(
        row.get("bound_frozen_before_observed_application_outcome") is not True
        or row.get("observed_application_outcome_used_in_lower_bound") is not False
        or "observed_application_error" in row
        or "behavioral_mse" in row
        or "log_behavioral_mse" in row
        for row in bound_rows
    ):
        raise ValueError("E24b frozen bound artifact contains an observed outcome")
    outcome_rows: list[dict[str, Any]] = []
    for cell in cells:
        frozen_bound = bound_by_id[cell.row_id]
        controller_state_lower_bound = float(
            frozen_bound["controller_specific_state_mse_lower_bound"]
        )
        controller_lower_bound = float(
            frozen_bound["controller_specific_clean_target_analytic_behavioral_lower_bound"]
        )
        analytic_error = float(frozen_bound["clean_oracle_attainable_behavioral_error"])
        sequence_key = (
            cell.seed,
            cell.demand_family,
            cell.geometry_block,
            cell.target_noise_factor,
            cell.teacher_noise_factor,
            cell.horizon,
        )
        cached_sequence = sequence_cache.get(sequence_key)
        if cached_sequence is None:
            target_teacher = build_target_teacher_sequence(
                cell,
                dimension=dimension,
                affected_row_fraction=affected_fraction,
            )
            inputs = _temporally_whitened_inputs(
                horizon=cell.horizon,
                batch_size=batch_size,
                dimension=dimension,
                seed=_stable_seed(
                    cell.seed,
                    cell.demand_family,
                    cell.horizon,
                    "exact_temporal_inputs",
                ),
                key_transform=target_teacher.key_transform,
            )
            clean_state = _roll_state(
                target_teacher.clean_targets,
                inputs=inputs,
                update_scale=update_scale,
                recurrence_decay=recurrence_decay,
            )
            teacher_state = _roll_state(
                target_teacher.teachers,
                inputs=inputs,
                update_scale=update_scale,
                recurrence_decay=recurrence_decay,
            )
            cached_sequence = (
                target_teacher,
                inputs,
                clean_state,
                teacher_state,
            )
            sequence_cache[sequence_key] = cached_sequence
        target_teacher, inputs, clean_state, teacher_state = cached_sequence
        controller_key = (
            *sequence_key,
            cell.controller_class,
        )
        cached_controller = controller_cache.get(controller_key)
        if cached_controller is None:
            observed_controllers = _project_controller_sequence(
                target_teacher.teachers,
                controller_class=cell.controller_class,
                shared_basis=target_teacher.shared_basis,
                key_transform=target_teacher.key_transform,
                affected_rows=target_teacher.affected_rows,
                readout_lambda=(target_teacher.affected_rows / dimension),
            )
            observed_state = _roll_state(
                observed_controllers,
                inputs=inputs,
                update_scale=update_scale,
                recurrence_decay=recurrence_decay,
            )
            teacher_operator_residual = _normalized_operator_residual(
                observed_controllers,
                target_teacher.teachers,
            )
            cached_controller = (
                observed_state,
                teacher_operator_residual,
            )
            controller_cache[controller_key] = cached_controller
        observed_state, teacher_operator_residual = cached_controller
        if (
            str(frozen_bound["clean_target_sequence_sha256"]) != target_teacher.clean_target_sha256
            or str(frozen_bound["controller_class"]) != cell.controller_class
            or str(frozen_bound["readout"]) != cell.readout
            or float(frozen_bound["readout_lambda"]) != cell.readout_lambda
        ):
            raise RuntimeError("E24b frozen bound changed before outcome evaluation")
        readout_seed = _stable_seed(
            cell.seed,
            cell.readout,
            "structural_readout",
        )
        teacher_proxy = _readout_metrics(
            teacher_state,
            observed_state,
            seed=readout_seed,
            readout=cell.readout,
            readout_lambda=cell.readout_lambda,
            affected_rows=target_teacher.affected_rows,
            nonlinear_residual_scale=nonlinear_scale,
        )
        observed_metrics = _readout_metrics(
            clean_state,
            observed_state,
            seed=readout_seed,
            readout=cell.readout,
            readout_lambda=cell.readout_lambda,
            affected_rows=target_teacher.affected_rows,
            nonlinear_residual_scale=nonlinear_scale,
        )
        outcome = observed_metrics.behavioral_mse
        if outcome + 1e-10 < controller_lower_bound:
            raise RuntimeError("E24b controller-specific behavioral lower bound was violated")
        retained_clean_norm = max(
            float(target[target_teacher.affected_rows :].abs().max())
            for target in target_teacher.clean_targets
        )
        teacher_retained_norm = math.sqrt(
            sum(
                float(teacher[target_teacher.affected_rows :].square().sum())
                for teacher in target_teacher.teachers
            )
        )
        feature_values = {
            "log_teacher_linearized_behavioral_regret": math.log(
                max(teacher_proxy.linearized_regret, outcome_floor)
            ),
            "log_teacher_lipschitz_upper_bound": math.log(
                max(teacher_proxy.lipschitz_upper_bound, outcome_floor)
            ),
            "log_teacher_operator_residual": math.log(
                max(teacher_operator_residual, outcome_floor)
            ),
            "log_horizon": math.log(float(cell.horizon)),
            "target_noise_factor": cell.target_noise_factor,
            "teacher_noise_factor": cell.teacher_noise_factor,
            "readout_lambda": cell.readout_lambda,
            "key_correlation": cell.key_correlation,
            "target_operator_norm": cell.target_operator_norm,
            "key_load_fraction": cell.key_load_fraction,
            "nonlinear_readout_indicator": float(cell.readout == "fixed_nonlinear_mlp"),
        }
        feature_rows.append(
            {
                "schema_version": 1,
                "experiment_id": "e24b_behavioral_attainability_stress",
                "run_mode": "DRY_RUN" if dry_run else "MAIN",
                **cell.as_dict(),
                "dimension": dimension,
                "batch_size": batch_size,
                "affected_row_count": target_teacher.affected_rows,
                "retained_row_count": dimension - target_teacher.affected_rows,
                "teacher_sequence_sha256": target_teacher.teacher_sha256,
                "input_key_covariance_sha256": _sequence_sha256(
                    [target_teacher.key_transform @ target_teacher.key_transform.mT]
                ),
                "realized_mean_key_correlation": (target_teacher.realized_mean_key_correlation),
                "realized_high_load_fraction": (target_teacher.realized_high_load_fraction),
                "controller_fit_source": "noisy_teacher_only",
                "clean_target_features_included": False,
                "teacher_linearized_behavioral_regret": (teacher_proxy.linearized_regret),
                "teacher_lipschitz_upper_bound": (teacher_proxy.lipschitz_upper_bound),
                "teacher_normalized_operator_residual": (teacher_operator_residual),
                **feature_values,
            }
        )
        outcome_rows.append(
            {
                "schema_version": 1,
                "experiment_id": "e24b_behavioral_attainability_stress",
                "run_mode": "DRY_RUN" if dry_run else "MAIN",
                **cell.as_dict(),
                "dimension": dimension,
                "batch_size": batch_size,
                "affected_row_count": target_teacher.affected_rows,
                "retained_row_count": dimension - target_teacher.affected_rows,
                "clean_target_sequence_sha256": (target_teacher.clean_target_sha256),
                "teacher_sequence_sha256": target_teacher.teacher_sha256,
                "input_key_covariance_sha256": _sequence_sha256(
                    [target_teacher.key_transform @ target_teacher.key_transform.mT]
                ),
                "realized_mean_key_correlation": (target_teacher.realized_mean_key_correlation),
                "realized_high_load_fraction": (target_teacher.realized_high_load_fraction),
                "clean_target_retained_row_max_abs": retained_clean_norm,
                "teacher_retained_row_frobenius_norm": teacher_retained_norm,
                "controller_fit_source": "noisy_teacher_only",
                "evaluation_target": "clean_application_target_only",
                "behavioral_mse": outcome,
                "affected_behavioral_mse": observed_metrics.affected_mse,
                "unaffected_behavioral_mse": observed_metrics.unaffected_mse,
                "readout_weighting_formula": (
                    "lambda * affected_mse + (1 - lambda) * unaffected_mse"
                ),
                "readout_row_block_method": (
                    "structural_affected_and_retained_state_rows_with_within_block_qr_whitening"
                ),
                "observed_application_error": outcome,
                "controller_specific_behavioral_lower_bound": (controller_lower_bound),
                "controller_specific_state_mse_lower_bound": (controller_state_lower_bound),
                "controller_specific_clean_target_analytic_behavioral_lower_bound": (
                    controller_lower_bound
                ),
                "lower_bound_source": ("clean_target_same_class_row_key_weighted_projection"),
                "observed_application_outcome_used_in_lower_bound": False,
                "bound_frozen_before_observed_application_outcome": True,
                "excess_over_controller_specific_lower_bound": max(
                    0.0,
                    outcome - controller_lower_bound,
                ),
                "lower_bound_controller_class": cell.controller_class,
                "lower_bound_independent_of_predictor": True,
                "clean_oracle_attainable_behavioral_error": analytic_error,
                "clean_oracle_gap_above_lower_bound": max(
                    0.0,
                    analytic_error - controller_lower_bound,
                ),
                "log_behavioral_mse": math.log(max(outcome, outcome_floor)),
                "linearized_behavioral_regret": (observed_metrics.linearized_regret),
                "affected_linearized_behavioral_regret": (
                    observed_metrics.affected_linearized_regret
                ),
                "unaffected_linearized_behavioral_regret": (
                    observed_metrics.unaffected_linearized_regret
                ),
                "lipschitz_upper_bound": (observed_metrics.lipschitz_upper_bound),
                "affected_lipschitz_upper_bound": (observed_metrics.affected_lipschitz_upper_bound),
                "unaffected_lipschitz_upper_bound": (
                    observed_metrics.unaffected_lipschitz_upper_bound
                ),
                "teacher_normalized_operator_residual": (teacher_operator_residual),
                **feature_values,
            }
        )
    if (
        len(feature_rows) != len(cells)
        or len(bound_rows) != len(cells)
        or len(outcome_rows) != len(cells)
    ):
        raise RuntimeError("E24b did not evaluate the complete registered grid")
    full_oracle_error = [
        float(row["clean_oracle_attainable_behavioral_error"])
        for row in outcome_rows
        if row["controller_class"] == "full"
    ]
    if full_oracle_error and max(full_oracle_error) > 1e-20:
        raise RuntimeError("E24b clean-target full controller failed to attain zero error")
    return BehavioralSimulationResult(
        feature_rows=feature_rows,
        bound_rows=bound_rows,
        outcome_rows=outcome_rows,
    )


def _feature_vector(row: Mapping[str, Any]) -> list[float]:
    values = [float(row[name]) for name in REGISTERED_PREDICTOR_FEATURES]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("E24b predictor features must be finite")
    return values


def fit_fold_predictor(
    *,
    feature_rows: Mapping[str, Mapping[str, Any]],
    fold: HoldoutFold,
    training_outcomes: Mapping[str, float],
    ridge: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fit on exactly one fold's training outcomes, then predict its held-out IDs."""

    required_train = set(fold.train_row_ids)
    if set(training_outcomes) != required_train:
        raise ValueError("E24b fold fitter accepts exactly the registered training outcomes")
    required_rows = required_train | set(fold.test_row_ids)
    if not required_rows <= set(feature_rows):
        raise ValueError("E24b fold fitter is missing registered feature rows")
    if ridge <= 0.0:
        raise ValueError("ridge must be positive")
    train_x = torch.tensor(
        [_feature_vector(feature_rows[row_id]) for row_id in fold.train_row_ids],
        dtype=torch.float64,
    )
    train_y = torch.tensor(
        [float(training_outcomes[row_id]) for row_id in fold.train_row_ids],
        dtype=torch.float64,
    )
    if not bool(torch.isfinite(train_y).all()):
        raise ValueError("E24b training outcomes must be finite")
    mean = train_x.mean(dim=0)
    scale = train_x.std(dim=0, unbiased=False).clamp_min(1e-8)
    standardized = (train_x - mean) / scale
    design = torch.cat(
        (
            torch.ones((standardized.shape[0], 1), dtype=torch.float64),
            standardized,
        ),
        dim=1,
    )
    penalty = torch.eye(design.shape[1], dtype=torch.float64) * ridge
    penalty[0, 0] = 0.0
    coefficients = torch.linalg.solve(
        design.mT @ design + penalty,
        design.mT @ train_y,
    )
    test_x = torch.tensor(
        [_feature_vector(feature_rows[row_id]) for row_id in fold.test_row_ids],
        dtype=torch.float64,
    )
    test_design = torch.cat(
        (
            torch.ones((test_x.shape[0], 1), dtype=torch.float64),
            (test_x - mean) / scale,
        ),
        dim=1,
    )
    predicted = test_design @ coefficients
    predictions = [
        {
            "schema_version": 1,
            "fold_id": fold.fold_id,
            "holdout_axis": fold.axis,
            "held_out_value": fold.held_out_value,
            "row_id": row_id,
            "prediction_scale": "log_behavioral_mse",
            "predicted_log_behavioral_mse": float(predicted[index]),
            "test_outcome_used": False,
        }
        for index, row_id in enumerate(fold.test_row_ids)
    ]
    training_payload = {
        row_id: float(training_outcomes[row_id]) for row_id in sorted(training_outcomes)
    }
    checkpoint = {
        "schema_version": 1,
        "fold_id": fold.fold_id,
        "holdout_axis": fold.axis,
        "held_out_value": fold.held_out_value,
        "feature_order": list(REGISTERED_PREDICTOR_FEATURES),
        "feature_mean": [float(value) for value in mean],
        "feature_scale": [float(value) for value in scale],
        "intercept": float(coefficients[0]),
        "standardized_coefficients": [float(value) for value in coefficients[1:]],
        "ridge": ridge,
        "training_row_count": len(fold.train_row_ids),
        "test_row_count": len(fold.test_row_ids),
        "training_outcome_sha256": sha256_canonical_json(training_payload),
        "test_outcome_used": False,
    }
    return predictions, checkpoint


def precompute_holdout_predictions(
    *,
    feature_rows: Sequence[Mapping[str, Any]],
    outcome_rows: Sequence[Mapping[str, Any]],
    folds: Sequence[HoldoutFold],
    ridge: float,
) -> PredictorOutput:
    """Fit all folds while exposing only their own training outcomes to each fit."""

    features_by_id = {str(row["row_id"]): row for row in feature_rows}
    outcomes_by_id = {str(row["row_id"]): row for row in outcome_rows}
    if len(features_by_id) != len(feature_rows):
        raise ValueError("E24b prediction input contains duplicate row IDs")
    if len(outcomes_by_id) != len(outcome_rows):
        raise ValueError("E24b training outcomes contain duplicate row IDs")
    if set(features_by_id) != set(outcomes_by_id):
        raise ValueError("E24b feature and outcome registries do not align")
    all_predictions: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    for fold in folds:
        training_outcomes = {
            row_id: float(outcomes_by_id[row_id]["log_behavioral_mse"])
            for row_id in fold.train_row_ids
        }
        predictions, checkpoint = fit_fold_predictor(
            feature_rows=features_by_id,
            fold=fold,
            training_outcomes=training_outcomes,
            ridge=ridge,
        )
        all_predictions.extend(predictions)
        checkpoints.append(checkpoint)
    return PredictorOutput(
        predictions=all_predictions,
        checkpoints=checkpoints,
    )


def family_level_scatter_rows(
    *,
    predictions: Sequence[Mapping[str, Any]],
    outcome_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Join outcomes into seed-by-held-out-level scatter points after prediction."""

    outcomes = {
        str(row["row_id"]): (
            int(row["seed"]),
            float(row["log_behavioral_mse"]),
        )
        for row in outcome_rows
    }
    groups: dict[
        tuple[str, str, int],
        list[tuple[float, float]],
    ] = {}
    for prediction in predictions:
        if prediction.get("test_outcome_used") is not False:
            raise ValueError("E24b scatter input violated outcome isolation")
        row_id = str(prediction["row_id"])
        if row_id not in outcomes:
            raise ValueError("E24b scatter prediction lacks a clean outcome")
        seed, actual = outcomes[row_id]
        key = (
            str(prediction["holdout_axis"]),
            str(prediction["held_out_value"]),
            seed,
        )
        groups.setdefault(key, []).append(
            (
                actual,
                float(prediction["predicted_log_behavioral_mse"]),
            )
        )
    return [
        {
            "schema_version": 1,
            "aggregation": "held_out_level_by_seed",
            "holdout_axis": axis,
            "held_out_value": held_out,
            "seed": seed,
            "cell_prediction_count": len(values),
            "mean_actual_log_behavioral_mse": (
                sum(actual for actual, _predicted in values) / len(values)
            ),
            "mean_predicted_log_behavioral_mse": (
                sum(predicted for _actual, predicted in values) / len(values)
            ),
            "outcome_join_after_prediction_artifact": True,
            "upper_unit": "seed",
        }
        for (axis, held_out, seed), values in sorted(groups.items())
    ]


def factor_sensitivity_rows(
    *,
    outcome_rows: Sequence[Mapping[str, Any]],
    factors: Sequence[str],
) -> list[dict[str, Any]]:
    """Summarize registered lambda/noise/horizon sensitivity within each seed."""

    allowed = {"readout_lambda", "noise_condition", "horizon"}
    if tuple(factors) != (
        "readout_lambda",
        "noise_condition",
        "horizon",
    ):
        raise ValueError("E24b sensitivity factors changed")
    groups: dict[
        tuple[str, str, int],
        tuple[object, list[Mapping[str, Any]]],
    ] = {}
    for row in outcome_rows:
        seed = int(row["seed"])
        for factor in factors:
            if factor not in allowed or factor not in row:
                raise ValueError(f"E24b sensitivity factor is invalid: {factor!r}")
            level = row[factor]
            key = factor, str(level), seed
            if key not in groups:
                groups[key] = level, []
            groups[key][1].append(row)
    sensitivity_rows: list[dict[str, Any]] = []
    for (factor, _level_key, seed), (level, rows) in sorted(
        groups.items(),
        key=lambda item: item[0],
    ):
        count = len(rows)
        sensitivity_rows.append(
            {
                "schema_version": 1,
                "aggregation": "registered_factor_level_by_seed",
                "factor": factor,
                "factor_level": level,
                "seed": seed,
                "cell_count": count,
                "mean_observed_application_error": (
                    sum(float(row["observed_application_error"]) for row in rows) / count
                ),
                "mean_controller_specific_lower_bound": (
                    sum(
                        float(
                            row["controller_specific_clean_target_analytic_behavioral_lower_bound"]
                        )
                        for row in rows
                    )
                    / count
                ),
                "mean_excess_over_controller_specific_lower_bound": (
                    sum(float(row["excess_over_controller_specific_lower_bound"]) for row in rows)
                    / count
                ),
                "mean_affected_behavioral_mse": (
                    sum(float(row["affected_behavioral_mse"]) for row in rows) / count
                ),
                "mean_retained_behavioral_mse": (
                    sum(float(row["unaffected_behavioral_mse"]) for row in rows) / count
                ),
                "upper_unit": "seed",
                "descriptive_sensitivity_only": True,
            }
        )
    return sensitivity_rows


def _average_ranks(values: torch.Tensor) -> torch.Tensor:
    """Return one-based average ranks with deterministic tie handling."""

    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("rank computation requires a nonempty vector")
    order = torch.argsort(values, stable=True)
    ranks = torch.empty_like(values)
    start = 0
    while start < int(values.numel()):
        end = start + 1
        while end < int(values.numel()) and float(values[order[end]]) == float(
            values[order[start]]
        ):
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def _prediction_metrics(
    actual: Sequence[float],
    predicted: Sequence[float],
) -> dict[str, float]:
    if len(actual) != len(predicted) or not actual:
        raise ValueError("OOS metrics require aligned nonempty values")
    target = torch.tensor(actual, dtype=torch.float64)
    estimate = torch.tensor(predicted, dtype=torch.float64)
    residual = estimate - target
    mse = residual.square().mean()
    rmse = torch.sqrt(mse)
    mae = residual.abs().mean()
    centered_target = target - target.mean()
    total = centered_target.square().sum()
    r2 = (
        1.0 - residual.square().sum() / total
        if float(total) > 0.0
        else torch.tensor(float(float(mse) == 0.0), dtype=torch.float64)
    )
    centered_estimate = estimate - estimate.mean()
    correlation_denominator = torch.sqrt(
        centered_target.square().sum() * centered_estimate.square().sum()
    )
    pearson = (
        (centered_target * centered_estimate).sum() / correlation_denominator
        if float(correlation_denominator) > 0.0
        else torch.tensor(0.0, dtype=torch.float64)
    )
    target_ranks = _average_ranks(target)
    estimate_ranks = _average_ranks(estimate)
    centered_target_ranks = target_ranks - target_ranks.mean()
    centered_estimate_ranks = estimate_ranks - estimate_ranks.mean()
    rank_denominator = torch.sqrt(
        centered_target_ranks.square().sum() * centered_estimate_ranks.square().sum()
    )
    spearman = (
        (centered_target_ranks * centered_estimate_ranks).sum() / rank_denominator
        if float(rank_denominator) > 0.0
        else torch.tensor(0.0, dtype=torch.float64)
    )
    prediction_variance = centered_estimate.square().sum()
    slope = (
        (centered_estimate * centered_target).sum() / prediction_variance
        if float(prediction_variance) > 0.0
        else torch.tensor(0.0, dtype=torch.float64)
    )
    intercept = target.mean() - slope * estimate.mean()
    target_std = torch.sqrt(centered_target.square().mean())
    normalized_rmse = rmse / target_std.clamp_min(1e-12)
    return {
        "r2": float(r2),
        "rmse": float(rmse),
        "mae": float(mae),
        "pearson_r": float(pearson),
        "spearman_r": float(spearman),
        "calibration_slope": float(slope),
        "calibration_intercept": float(intercept),
        "normalized_rmse": float(normalized_rmse),
    }


_BOOTSTRAP_METRIC_NAMES = (
    "r2",
    "rmse",
    "mae",
    "pearson_r",
    "spearman_r",
    "calibration_slope",
    "calibration_intercept",
    "normalized_rmse",
)


def _seed_cluster_metric_summary(
    *,
    predictions: Sequence[Mapping[str, Any]],
    outcomes: Mapping[str, float],
    seed_by_row: Mapping[str, int],
    replicates: int,
    confidence_level: float,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Average seed-level metrics and bootstrap only the registered seed unit."""

    if replicates <= 0 or not 0.0 < confidence_level < 1.0:
        raise ValueError("E24b cluster-bootstrap configuration is invalid")
    by_seed: dict[int, list[Mapping[str, Any]]] = {}
    for prediction in predictions:
        row_id = str(prediction["row_id"])
        by_seed.setdefault(seed_by_row[row_id], []).append(prediction)
    if not by_seed:
        raise ValueError("E24b seed-cluster metrics require predictions")
    per_seed: dict[int, dict[str, float]] = {}
    for seed, rows in sorted(by_seed.items()):
        per_seed[seed] = _prediction_metrics(
            [outcomes[str(row["row_id"])] for row in rows],
            [float(row["predicted_log_behavioral_mse"]) for row in rows],
        )
    seeds = sorted(per_seed)
    point = {
        name: sum(per_seed[seed][name] for seed in seeds) / len(seeds)
        for name in _BOOTSTRAP_METRIC_NAMES
    }
    generator = torch.Generator(device="cpu")
    generator.manual_seed(bootstrap_seed)
    samples: dict[str, list[float]] = {name: [] for name in _BOOTSTRAP_METRIC_NAMES}
    for _replicate in range(replicates):
        indices = torch.randint(
            len(seeds),
            (len(seeds),),
            generator=generator,
        )
        selected = [seeds[int(index)] for index in indices]
        for name in _BOOTSTRAP_METRIC_NAMES:
            samples[name].append(sum(per_seed[seed][name] for seed in selected) / len(selected))
    alpha = (1.0 - confidence_level) / 2.0
    intervals = {
        name: {
            "lower": float(
                torch.quantile(
                    torch.tensor(values, dtype=torch.float64),
                    alpha,
                )
            ),
            "upper": float(
                torch.quantile(
                    torch.tensor(values, dtype=torch.float64),
                    1.0 - alpha,
                )
            ),
        }
        for name, values in samples.items()
    }
    return {
        **point,
        "seed_cluster_count": len(seeds),
        "cluster_bootstrap_replicates": replicates,
        "cluster_bootstrap_confidence_level": confidence_level,
        "cluster_bootstrap_resample_unit": "seed",
        "episode_row_resampling_used": False,
        "cluster_bootstrap_ci": intervals,
    }


def _passes_prospective_gate(
    metrics: Mapping[str, Any],
    gates: Mapping[str, Any],
) -> bool:
    intervals = metrics.get("cluster_bootstrap_ci")
    if not isinstance(intervals, Mapping):
        raise ValueError("E24b prospective gates require seed-cluster intervals")
    r2_interval = _mapping(intervals["r2"], label="r2 bootstrap interval")
    pearson_interval = _mapping(
        intervals["pearson_r"],
        label="pearson bootstrap interval",
    )
    nrmse_interval = _mapping(
        intervals["normalized_rmse"],
        label="normalized-rmse bootstrap interval",
    )
    return (
        float(r2_interval["lower"]) >= float(gates["minimum_r2"])
        and float(pearson_interval["lower"]) >= float(gates["minimum_pearson_r"])
        and float(nrmse_interval["upper"]) <= float(gates["maximum_normalized_rmse"])
    )


def _claim_subset_member(row: Mapping[str, Any], *, subset: str) -> bool:
    if subset == "broad_noisy_nonlinear_multistep":
        return (
            float(row["teacher_noise_factor"]) > 0.0
            and row["readout"] == "fixed_nonlinear_mlp"
            and int(row["horizon"]) > 1
        )
    if subset == "linear_h1":
        return row["readout"] == "linear" and int(row["horizon"]) == 1
    raise ValueError(f"Unknown E24b claim subset: {subset!r}")


def _assess_claim_subset(
    *,
    subset: str,
    predictions: Sequence[Mapping[str, Any]],
    outcomes: Mapping[str, float],
    outcome_rows: Mapping[str, Mapping[str, Any]],
    seed_by_row: Mapping[str, int],
    required_axes: Sequence[str],
    gates: Mapping[str, Any],
    bootstrap_replicates: int,
    bootstrap_confidence: float,
    bootstrap_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected = [
        prediction
        for prediction in predictions
        if _claim_subset_member(
            outcome_rows[str(prediction["row_id"])],
            subset=subset,
        )
    ]
    selected_ids = {str(prediction["row_id"]) for prediction in selected}
    if not selected or not selected_ids:
        raise RuntimeError(f"E24b registered claim subset is empty: {subset}")
    axis_status: dict[str, bool] = {}
    axis_metrics: dict[str, dict[str, Any]] = {}
    metric_rows: list[dict[str, Any]] = []
    for axis in required_axes:
        group = [prediction for prediction in selected if prediction["holdout_axis"] == axis]
        if not group:
            raise RuntimeError(f"E24b claim subset {subset!r} lacks holdout axis {axis!r}")
        metrics = _seed_cluster_metric_summary(
            predictions=group,
            outcomes=outcomes,
            seed_by_row=seed_by_row,
            replicates=bootstrap_replicates,
            confidence_level=bootstrap_confidence,
            bootstrap_seed=_stable_seed(
                bootstrap_seed,
                subset,
                axis,
            ),
        )
        gate_pass = _passes_prospective_gate(metrics, gates)
        axis_status[axis] = gate_pass
        axis_metrics[axis] = {
            "row_count": len(group),
            "prospective_gate_pass": gate_pass,
            **metrics,
        }
        metric_rows.append(
            {
                "schema_version": 1,
                "aggregation": "claim_subset_holdout_axis",
                "claim_subset": subset,
                "holdout_axis": axis,
                "held_out_value": None,
                "prediction_scale": "log_behavioral_mse",
                "row_count": len(group),
                "prospective_gate_pass": gate_pass,
                **metrics,
            }
        )
    return (
        {
            "definition": (
                "teacher_noise_factor > 0 AND fixed_nonlinear_mlp AND horizon > 1"
                if subset == "broad_noisy_nonlinear_multistep"
                else "linear readout AND horizon == 1"
            ),
            "cell_count": len(selected_ids),
            "prediction_count": len(selected),
            "axis_gate_status": axis_status,
            "axis_metrics": axis_metrics,
            "all_holdout_axes_pass": (
                set(axis_status) == set(required_axes) and all(axis_status.values())
            ),
        },
        metric_rows,
    )


def score_precomputed_predictions(
    *,
    predictions: Sequence[Mapping[str, Any]],
    outcome_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    dry_run: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Join test outcomes only after prediction and calculate registered OOS metrics."""

    outcomes = {str(row["row_id"]): float(row["log_behavioral_mse"]) for row in outcome_rows}
    outcome_metadata = {str(row["row_id"]): row for row in outcome_rows}
    seeds = {str(row["row_id"]): int(row["seed"]) for row in outcome_rows}
    if len(outcomes) != len(outcome_rows):
        raise ValueError("E24b scoring input contains duplicate outcome IDs")
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    axis_groups: dict[str, list[Mapping[str, Any]]] = {}
    seed_groups: dict[int, list[Mapping[str, Any]]] = {}
    for prediction in predictions:
        if prediction.get("test_outcome_used") is not False:
            raise ValueError("E24b prediction lacks the test-outcome isolation marker")
        row_id = str(prediction["row_id"])
        if row_id not in outcomes:
            raise ValueError(f"E24b prediction has no matching outcome: {row_id}")
        key = (str(prediction["holdout_axis"]), str(prediction["held_out_value"]))
        groups.setdefault(key, []).append(prediction)
        axis_groups.setdefault(key[0], []).append(prediction)
        seed_groups.setdefault(seeds[row_id], []).append(prediction)

    metric_rows: list[dict[str, Any]] = []
    for (axis, held_out), group in sorted(groups.items()):
        actual = [outcomes[str(row["row_id"])] for row in group]
        predicted = [float(row["predicted_log_behavioral_mse"]) for row in group]
        metric_rows.append(
            {
                "schema_version": 1,
                "aggregation": "fold",
                "holdout_axis": axis,
                "held_out_value": held_out,
                "prediction_scale": "log_behavioral_mse",
                "row_count": len(group),
                **_prediction_metrics(actual, predicted),
            }
        )
    gates = _mapping(
        _mapping(config["oos_metrics"], label="oos_metrics")["gates"],
        label="oos_metrics.gates",
    )
    inference = _mapping(config["inference"], label="inference")
    bootstrap = _mapping(
        inference["cluster_bootstrap"],
        label="inference.cluster_bootstrap",
    )
    bootstrap_replicates = (
        int(
            _mapping(
                config["dry_run_overrides"],
                label="dry_run_overrides",
            )["bootstrap_replicates"]
        )
        if dry_run
        else int(bootstrap["replicates"])
    )
    bootstrap_confidence = float(bootstrap["confidence_level"])
    bootstrap_seed = int(bootstrap["seed"])
    axis_gate_status: dict[str, bool] = {}
    for axis, group in sorted(axis_groups.items()):
        metrics = _seed_cluster_metric_summary(
            predictions=group,
            outcomes=outcomes,
            seed_by_row=seeds,
            replicates=bootstrap_replicates,
            confidence_level=bootstrap_confidence,
            bootstrap_seed=_stable_seed(
                bootstrap_seed,
                "overall",
                axis,
            ),
        )
        gate_pass = _passes_prospective_gate(metrics, gates)
        axis_gate_status[axis] = gate_pass
        metric_rows.append(
            {
                "schema_version": 1,
                "aggregation": "holdout_axis",
                "holdout_axis": axis,
                "held_out_value": None,
                "prediction_scale": "log_behavioral_mse",
                "row_count": len(group),
                "prospective_gate_pass": gate_pass,
                **metrics,
            }
        )
    seed_rows: list[dict[str, Any]] = []
    for seed, group in sorted(seed_groups.items()):
        actual = [outcomes[str(row["row_id"])] for row in group]
        predicted = [float(row["predicted_log_behavioral_mse"]) for row in group]
        unique_seed_rows = [row for row in outcome_rows if int(row["seed"]) == seed]
        seed_rows.append(
            {
                "schema_version": 1,
                "experiment_id": "e24b_behavioral_attainability_stress",
                "run_mode": "DRY_RUN" if dry_run else "MAIN",
                "seed": seed,
                "joined_oos_prediction_count": len(group),
                "prediction_scale": "log_behavioral_mse",
                "mean_controller_specific_analytic_behavioral_lower_bound": (
                    sum(
                        float(
                            row["controller_specific_clean_target_analytic_behavioral_lower_bound"]
                        )
                        for row in unique_seed_rows
                    )
                    / len(unique_seed_rows)
                ),
                "mean_observed_application_error": (
                    sum(float(row["observed_application_error"]) for row in unique_seed_rows)
                    / len(unique_seed_rows)
                ),
                "mean_excess_over_controller_specific_lower_bound": (
                    sum(
                        float(row["excess_over_controller_specific_lower_bound"])
                        for row in unique_seed_rows
                    )
                    / len(unique_seed_rows)
                ),
                "maximum_excess_over_controller_specific_lower_bound": max(
                    float(row["excess_over_controller_specific_lower_bound"])
                    for row in unique_seed_rows
                ),
                **_prediction_metrics(actual, predicted),
            }
        )
    required_axes = tuple(
        str(value) for value in _mapping(config["design"], label="design")["holdouts"]
    )
    overall_gate = set(axis_gate_status) == set(required_axes) and all(axis_gate_status.values())
    broad_assessment, broad_metric_rows = _assess_claim_subset(
        subset="broad_noisy_nonlinear_multistep",
        predictions=predictions,
        outcomes=outcomes,
        outcome_rows=outcome_metadata,
        seed_by_row=seeds,
        required_axes=required_axes,
        gates=gates,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_confidence=bootstrap_confidence,
        bootstrap_seed=bootstrap_seed,
    )
    linear_assessment, linear_metric_rows = _assess_claim_subset(
        subset="linear_h1",
        predictions=predictions,
        outcomes=outcomes,
        outcome_rows=outcome_metadata,
        seed_by_row=seeds,
        required_axes=required_axes,
        gates=gates,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_confidence=bootstrap_confidence,
        bootstrap_seed=bootstrap_seed,
    )
    metric_rows.extend(broad_metric_rows)
    metric_rows.extend(linear_metric_rows)
    if overall_gate and bool(broad_assessment["all_holdout_axes_pass"]):
        computed_disposition = "BROAD_NOISY_NONLINEAR_MULTISTEP_PASS"
        allowed_claim = (
            "Registered finite-dimensional OOS calibration including the "
            "predeclared noisy nonlinear multistep subset."
        )
    elif bool(linear_assessment["all_holdout_axes_pass"]):
        computed_disposition = "ONLY_LINEAR_H1_PASS"
        allowed_claim = (
            "Registered finite-dimensional OOS calibration limited to the linear H=1 subset."
        )
    else:
        computed_disposition = "CONSTRUCTION_ROBUST_PREDICTION_FAILURE"
        allowed_claim = "No construction-robust behavioral prediction claim."
    claim_assessment = {
        "overall_oos_gate_pass": overall_gate,
        "broad_noisy_nonlinear_multistep": broad_assessment,
        "linear_h1": linear_assessment,
        "computed_disposition": computed_disposition,
        "claim_disposition": ("DRY_RUN_NON_EVIDENCE" if dry_run else computed_disposition),
        "allowed_claim": "None; dry-run is non-evidence." if dry_run else allowed_claim,
    }
    summary = {
        "prediction_scale": "log_behavioral_mse",
        "prediction_count": len(predictions),
        "fold_count": len(groups),
        "holdout_axis_count": len(axis_groups),
        "axis_gate_status": axis_gate_status,
        "all_holdout_axes_required": True,
        "prospective_gate_unit": "seed",
        "cluster_bootstrap_replicates": bootstrap_replicates,
        "episode_row_resampling_used": False,
        "prospective_gate_pass": overall_gate,
        "claim_assessment": claim_assessment,
        "scientific_status": ("NOT_EVALUATED_DRY_RUN" if dry_run else "PROSPECTIVE_GATE_EVALUATED"),
    }
    return metric_rows, seed_rows, summary
