from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import torch


@dataclass(slots=True)
class JointDiagonalizationReport:
    """Numerical shared-basis fit and held-out operator-application diagnostics."""

    commutator_norm: float
    fixed_diagonal_regret: float
    learned_basis_diagonal_regret: float
    low_rank_regret: float
    full_matrix_regret: float
    fixed_diagonal_rjd_objective: float
    learned_basis_rjd_objective: float
    train_fixed_diagonal_regret: float
    train_learned_basis_diagonal_regret: float
    fixed_diagonal_empirical_error: float
    learned_basis_diagonal_empirical_error: float
    low_rank_empirical_error: float
    full_matrix_empirical_error: float
    learned_basis_orthogonality_error: float
    optimizer_best_restart: int
    optimizer_best_step: int
    optimizer_identity_candidate_regret: float
    optimizer_restart_initial_regrets: list[float]
    optimizer_restart_best_regrets: list[float]
    optimizer_restart_final_regrets: list[float]


@dataclass(slots=True)
class _SharedBasisFit:
    basis: torch.Tensor
    best_regret: float
    best_restart: int
    best_step: int
    identity_regret: float
    restart_initial_regrets: list[float]
    restart_best_regrets: list[float]
    restart_final_regrets: list[float]


def _validate_positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _validate_projectors(
    projectors: list[torch.Tensor],
    *,
    name: str,
) -> tuple[int, torch.dtype, torch.device]:
    if not projectors:
        raise ValueError(f"{name} must contain at least one projector.")
    first = projectors[0]
    if (
        first.ndim != 2
        or first.shape[0] != first.shape[1]
        or first.shape[0] < 2
    ):
        raise ValueError(f"{name}[0] must be a square matrix of dimension at least 2.")
    if not first.dtype.is_floating_point:
        raise TypeError(f"{name} must use a floating-point dtype.")
    dim = int(first.shape[0])
    dtype = first.dtype
    device = first.device
    tolerance = 1e-8 if dtype == torch.float64 else 2e-5

    for index, projector in enumerate(projectors):
        if projector.shape != (dim, dim):
            raise ValueError(f"{name}[{index}] has an inconsistent shape.")
        if projector.dtype != dtype or projector.device != device:
            raise ValueError(f"{name}[{index}] has an inconsistent dtype or device.")
        if not bool(torch.isfinite(projector).all().item()):
            raise ValueError(f"{name}[{index}] contains a non-finite value.")
        symmetry_error = float(
            torch.linalg.matrix_norm(
                projector - projector.transpose(0, 1), ord="fro"
            ).item()
        )
        idempotence_error = float(
            torch.linalg.matrix_norm(projector @ projector - projector, ord="fro").item()
        )
        if symmetry_error > tolerance:
            raise ValueError(
                f"{name}[{index}] is not symmetric within tolerance: "
                f"{symmetry_error} > {tolerance}."
            )
        if idempotence_error > tolerance:
            raise ValueError(
                f"{name}[{index}] is not idempotent within tolerance: "
                f"{idempotence_error} > {tolerance}."
            )
    return dim, dtype, device


def _offdiag_energy(matrix: torch.Tensor) -> torch.Tensor:
    diagonal = torch.diag(torch.diag(matrix))
    return torch.sum((matrix - diagonal) ** 2)


def _basis_objective(
    basis: torch.Tensor,
    projectors: list[torch.Tensor],
) -> torch.Tensor:
    return torch.sum(
        torch.stack(
            [
                _offdiag_energy(basis.transpose(0, 1) @ projector @ basis)
                for projector in projectors
            ]
        )
    )


def _normalized_basis_regret(
    basis: torch.Tensor,
    projectors: list[torch.Tensor],
) -> torch.Tensor:
    dim = int(projectors[0].shape[0])
    return _basis_objective(basis, projectors) / (len(projectors) * dim * dim)


def joint_diagonalization_objective(
    basis: torch.Tensor,
    projectors: list[torch.Tensor],
) -> float:
    """Return the paper's sum of squared off-diagonal Frobenius norms."""

    dim, dtype, device = _validate_projectors(projectors, name="projectors")
    if basis.shape != (dim, dim) or basis.dtype != dtype or basis.device != device:
        raise ValueError("basis must match the projector shape, dtype, and device.")
    if not bool(torch.isfinite(basis).all().item()):
        raise ValueError("basis contains a non-finite value.")
    orthogonality_error = torch.linalg.matrix_norm(
        basis.transpose(0, 1) @ basis
        - torch.eye(dim, dtype=dtype, device=device),
        ord="fro",
    )
    if float(orthogonality_error.item()) > (1e-8 if dtype == torch.float64 else 2e-5):
        raise ValueError("basis must be orthogonal.")
    return float(_basis_objective(basis, projectors).item())


def commutator_norm(projectors: list[torch.Tensor]) -> float:
    """Return mean pairwise Frobenius norm, matching the registered κ metric."""

    _validate_projectors(projectors, name="projectors")
    values: list[torch.Tensor] = []
    for index in range(len(projectors)):
        for other in range(index + 1, len(projectors)):
            commutator = (
                projectors[index] @ projectors[other]
                - projectors[other] @ projectors[index]
            )
            values.append(torch.linalg.matrix_norm(commutator, ord="fro"))
    return float(torch.mean(torch.stack(values)).item()) if values else 0.0


def _fit_shared_basis(
    projectors: list[torch.Tensor],
    *,
    steps: int,
    learning_rate: float,
    restarts: int,
    seed: int,
) -> _SharedBasisFit:
    dim, dtype, device = _validate_projectors(
        projectors,
        name="training_projectors",
    )
    _validate_positive_integer(steps, "steps")
    _validate_positive_integer(restarts, "restarts")
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive and finite.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer.")

    identity = torch.eye(dim, dtype=dtype, device=device)
    identity_regret = float(_normalized_basis_regret(identity, projectors).item())
    best_regret = identity_regret
    best_basis = identity.clone()
    best_restart = -1
    best_step = -1
    initial_regrets: list[float] = []
    restart_best_regrets: list[float] = []
    final_regrets: list[float] = []

    generator = torch.Generator(device=device).manual_seed(seed)
    for restart in range(restarts):
        if restart == 0:
            initialization = identity
        else:
            random_matrix = torch.randn(
                dim,
                dim,
                generator=generator,
                dtype=dtype,
                device=device,
            )
            initialization, _ = torch.linalg.qr(random_matrix)
        raw_basis = initialization.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([raw_basis], lr=learning_rate)

        with torch.no_grad():
            initial_basis, _ = torch.linalg.qr(raw_basis)
            initial_regret = float(
                _normalized_basis_regret(initial_basis, projectors).item()
            )
        initial_regrets.append(initial_regret)
        restart_best = initial_regret

        for step in range(steps):
            basis, _ = torch.linalg.qr(raw_basis)
            loss = _normalized_basis_regret(basis, projectors)
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError(
                    f"Joint-diagonalization loss became non-finite at "
                    f"restart={restart}, step={step}."
                )
            current = float(loss.detach().item())
            if current < restart_best:
                restart_best = current
            if current < best_regret:
                best_regret = current
                best_basis = basis.detach().clone()
                best_restart = restart
                best_step = step
            optimizer.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            if raw_basis.grad is None or not bool(torch.isfinite(raw_basis.grad).all().item()):
                raise FloatingPointError(
                    f"Joint-diagonalization gradient became non-finite at "
                    f"restart={restart}, step={step}."
                )
            optimizer.step()

        with torch.no_grad():
            final_basis, _ = torch.linalg.qr(raw_basis)
            final_regret = float(
                _normalized_basis_regret(final_basis, projectors).item()
            )
        if not math.isfinite(final_regret):
            raise FloatingPointError(
                f"Joint-diagonalization final loss is non-finite at restart={restart}."
            )
        if final_regret < restart_best:
            restart_best = final_regret
        if final_regret < best_regret:
            best_regret = final_regret
            best_basis = final_basis.detach().clone()
            best_restart = restart
            best_step = steps
        restart_best_regrets.append(restart_best)
        final_regrets.append(final_regret)

    return _SharedBasisFit(
        basis=best_basis,
        best_regret=best_regret,
        best_restart=best_restart,
        best_step=best_step,
        identity_regret=identity_regret,
        restart_initial_regrets=initial_regrets,
        restart_best_regrets=restart_best_regrets,
        restart_final_regrets=final_regrets,
    )


def _diagonal_approximation(
    projector: torch.Tensor,
    basis: torch.Tensor,
) -> torch.Tensor:
    transformed = basis.transpose(0, 1) @ projector @ basis
    return basis @ torch.diag(torch.diag(transformed)) @ basis.transpose(0, 1)


def _low_rank_approximation(
    projector: torch.Tensor,
    low_rank: int,
) -> torch.Tensor:
    u, singular_values, vh = torch.linalg.svd(projector)
    approximation = (u[:, :low_rank] * singular_values[:low_rank]) @ vh[:low_rank]
    return cast(torch.Tensor, approximation)


def _mean_operator_mse(
    targets: list[torch.Tensor],
    approximations: list[torch.Tensor],
) -> float:
    return float(
        torch.mean(
            torch.stack(
                [
                    torch.mean((target - approximation) ** 2)
                    for target, approximation in zip(
                        targets,
                        approximations,
                        strict=True,
                    )
                ]
            )
        ).item()
    )


def _empirical_application_mse(
    targets: list[torch.Tensor],
    approximations: list[torch.Tensor],
    *,
    probe_count: int,
    probe_seed: int,
) -> float:
    _validate_positive_integer(probe_count, "probe_count")
    if isinstance(probe_seed, bool) or not isinstance(probe_seed, int):
        raise TypeError("probe_seed must be an integer.")
    dim = int(targets[0].shape[0])
    generator = torch.Generator(device=targets[0].device).manual_seed(probe_seed)
    errors: list[torch.Tensor] = []
    for target, approximation in zip(targets, approximations, strict=True):
        probes = torch.randn(
            probe_count,
            dim,
            generator=generator,
            dtype=target.dtype,
            device=target.device,
        ) / math.sqrt(dim)
        target_outputs = probes @ target.transpose(0, 1)
        approximate_outputs = probes @ approximation.transpose(0, 1)
        errors.append(torch.mean((target_outputs - approximate_outputs) ** 2))
    return float(torch.mean(torch.stack(errors)).item())


def analyze_joint_diagonalization(
    training_projectors: list[torch.Tensor],
    *,
    evaluation_projectors: list[torch.Tensor] | None = None,
    steps: int,
    learning_rate: float,
    low_rank: int,
    restarts: int = 4,
    seed: int = 0,
    probe_count: int = 1024,
    probe_seed: int | None = None,
) -> JointDiagonalizationReport:
    """Fit one shared diagonal basis and evaluate four operator control classes.

    The same target projector is the demand descriptor for every control class.
    Diagonal coefficients are transaction-conditioned projections in a fixed or
    learned shared basis. The low-rank and full-matrix conditions are oracle
    upper bounds; they are not parameter-matched learned models.
    """

    dim, dtype, device = _validate_projectors(
        training_projectors,
        name="training_projectors",
    )
    evaluation = (
        training_projectors
        if evaluation_projectors is None
        else evaluation_projectors
    )
    evaluation_dim, evaluation_dtype, evaluation_device = _validate_projectors(
        evaluation,
        name="evaluation_projectors",
    )
    if (evaluation_dim, evaluation_dtype, evaluation_device) != (dim, dtype, device):
        raise ValueError(
            "training and evaluation projectors must share shape, dtype, and device."
        )
    _validate_positive_integer(low_rank, "low_rank")
    if low_rank > dim:
        raise ValueError("low_rank cannot exceed the operator dimension.")

    fit = _fit_shared_basis(
        training_projectors,
        steps=steps,
        learning_rate=learning_rate,
        restarts=restarts,
        seed=seed,
    )
    identity = torch.eye(dim, dtype=dtype, device=device)
    fixed_approximations = [
        _diagonal_approximation(projector, identity) for projector in evaluation
    ]
    learned_approximations = [
        _diagonal_approximation(projector, fit.basis) for projector in evaluation
    ]
    low_rank_approximations = [
        _low_rank_approximation(projector, low_rank) for projector in evaluation
    ]
    full_approximations = [projector.clone() for projector in evaluation]

    fixed = _mean_operator_mse(evaluation, fixed_approximations)
    learned = _mean_operator_mse(evaluation, learned_approximations)
    low_rank_error = _mean_operator_mse(evaluation, low_rank_approximations)
    full_matrix_error = _mean_operator_mse(evaluation, full_approximations)
    fixed_objective = joint_diagonalization_objective(identity, evaluation)
    learned_objective = joint_diagonalization_objective(fit.basis, evaluation)
    orthogonality_error = float(
        torch.linalg.matrix_norm(
            fit.basis.transpose(0, 1) @ fit.basis - identity,
            ord="fro",
        ).item()
    )
    effective_probe_seed = seed + 1_000_003 if probe_seed is None else probe_seed

    values = (
        fixed,
        learned,
        low_rank_error,
        full_matrix_error,
        fixed_objective,
        learned_objective,
        fit.identity_regret,
        fit.best_regret,
        orthogonality_error,
    )
    if not all(math.isfinite(value) for value in values):
        raise FloatingPointError("Joint-diagonalization analysis produced a non-finite value.")

    return JointDiagonalizationReport(
        commutator_norm=commutator_norm(evaluation),
        fixed_diagonal_regret=fixed,
        learned_basis_diagonal_regret=learned,
        low_rank_regret=low_rank_error,
        full_matrix_regret=full_matrix_error,
        fixed_diagonal_rjd_objective=fixed_objective,
        learned_basis_rjd_objective=learned_objective,
        train_fixed_diagonal_regret=fit.identity_regret,
        train_learned_basis_diagonal_regret=fit.best_regret,
        fixed_diagonal_empirical_error=_empirical_application_mse(
            evaluation,
            fixed_approximations,
            probe_count=probe_count,
            probe_seed=effective_probe_seed,
        ),
        learned_basis_diagonal_empirical_error=_empirical_application_mse(
            evaluation,
            learned_approximations,
            probe_count=probe_count,
            probe_seed=effective_probe_seed,
        ),
        low_rank_empirical_error=_empirical_application_mse(
            evaluation,
            low_rank_approximations,
            probe_count=probe_count,
            probe_seed=effective_probe_seed,
        ),
        full_matrix_empirical_error=_empirical_application_mse(
            evaluation,
            full_approximations,
            probe_count=probe_count,
            probe_seed=effective_probe_seed,
        ),
        learned_basis_orthogonality_error=orthogonality_error,
        optimizer_best_restart=fit.best_restart,
        optimizer_best_step=fit.best_step,
        optimizer_identity_candidate_regret=fit.identity_regret,
        optimizer_restart_initial_regrets=fit.restart_initial_regrets,
        optimizer_restart_best_regrets=fit.restart_best_regrets,
        optimizer_restart_final_regrets=fit.restart_final_regrets,
    )
