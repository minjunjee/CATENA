from __future__ import annotations

import math

import pytest
import torch

from catena.data.operator_families import OperatorFamily, generate_operator_set
from catena.theory.joint_diagonalization import (
    analyze_joint_diagonalization,
    commutator_norm,
    joint_diagonalization_objective,
)


def _with_small_thread_pool() -> int:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    return previous


def test_registered_metrics_match_hand_computable_projectors() -> None:
    first = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
    second = 0.5 * torch.ones((2, 2), dtype=torch.float64)
    identity = torch.eye(2, dtype=torch.float64)

    assert commutator_norm([first, second]) == pytest.approx(math.sqrt(0.5))
    assert joint_diagonalization_objective(identity, [first, second]) == pytest.approx(
        0.5
    )


@pytest.mark.parametrize("seed", [13, 17])
@pytest.mark.parametrize("family", list(OperatorFamily))
def test_operator_families_are_valid_deterministic_rank_projectors(
    seed: int,
    family: OperatorFamily,
) -> None:
    first = generate_operator_set(
        family=family,
        dim=8,
        rank=2,
        count=6,
        seed=seed,
    )
    second = generate_operator_set(
        family=family,
        dim=8,
        rank=2,
        count=6,
        seed=seed,
    )
    assert len(first.projectors) == 6
    for left, right in zip(first.projectors, second.projectors, strict=True):
        assert left.dtype == torch.float64
        assert torch.equal(left, right)
        assert torch.allclose(left, left.T, atol=1e-12, rtol=0.0)
        assert torch.allclose(left @ left, left, atol=1e-12, rtol=0.0)
        assert int(torch.linalg.matrix_rank(left).item()) == 2


def test_commuting_and_noncommuting_family_certificates() -> None:
    axis = generate_operator_set(
        family=OperatorFamily.AXIS_COMMUTING,
        dim=8,
        rank=2,
        count=6,
        seed=19,
    )
    common = generate_operator_set(
        family=OperatorFamily.COMMON_ROTATED_COMMUTING,
        dim=8,
        rank=2,
        count=6,
        seed=19,
    )
    noncommuting = generate_operator_set(
        family=OperatorFamily.NONCOMMUTING,
        dim=8,
        rank=2,
        count=6,
        seed=19,
    )
    assert commutator_norm(axis.projectors) < 1e-12
    assert commutator_norm(common.projectors) < 1e-12
    assert commutator_norm(noncommuting.projectors) > 1e-3
    assert axis.certified_shared_basis is not None
    assert common.certified_shared_basis is not None
    assert noncommuting.certified_shared_basis is None


def test_shared_basis_fit_recovers_common_rotation_and_preserves_nesting() -> None:
    previous_threads = _with_small_thread_pool()
    try:
        operator_set = generate_operator_set(
            family=OperatorFamily.COMMON_ROTATED_COMMUTING,
            dim=8,
            rank=2,
            count=20,
            seed=23,
        )
        report = analyze_joint_diagonalization(
            operator_set.projectors[:16],
            evaluation_projectors=operator_set.projectors[16:],
            steps=500,
            learning_rate=0.03,
            low_rank=2,
            restarts=3,
            seed=29,
            probe_count=2048,
            probe_seed=31,
        )
    finally:
        torch.set_num_threads(previous_threads)

    assert report.train_learned_basis_diagonal_regret <= (
        report.optimizer_identity_candidate_regret + 1e-15
    )
    assert report.learned_basis_diagonal_regret < 1e-8
    assert report.low_rank_regret < 1e-20
    assert report.full_matrix_regret == 0.0
    assert report.learned_basis_orthogonality_error < 1e-10
    assert report.learned_basis_diagonal_empirical_error == pytest.approx(
        report.learned_basis_diagonal_regret,
        abs=1e-8,
    )
    assert len(report.optimizer_restart_best_regrets) == 3


def test_noncommuting_family_retains_shared_basis_residual() -> None:
    previous_threads = _with_small_thread_pool()
    try:
        operator_set = generate_operator_set(
            family=OperatorFamily.NONCOMMUTING,
            dim=8,
            rank=2,
            count=12,
            seed=37,
        )
        report = analyze_joint_diagonalization(
            operator_set.projectors[:8],
            evaluation_projectors=operator_set.projectors[8:],
            steps=150,
            learning_rate=0.03,
            low_rank=2,
            restarts=2,
            seed=41,
            probe_count=1024,
            probe_seed=43,
        )
    finally:
        torch.set_num_threads(previous_threads)

    assert report.commutator_norm > 1e-3
    assert report.learned_basis_diagonal_regret > 1e-3
    assert report.low_rank_regret < 1e-20
    assert report.learned_basis_diagonal_regret > report.low_rank_regret


@pytest.mark.parametrize(
    ("kwargs", "exception"),
    [
        ({"dim": 1, "rank": 1, "count": 2}, ValueError),
        ({"dim": 8, "rank": 0, "count": 2}, ValueError),
        ({"dim": 8, "rank": 8, "count": 2}, ValueError),
        ({"dim": 8, "rank": 2, "count": 1}, ValueError),
    ],
)
def test_operator_family_rejects_invalid_contract(
    kwargs: dict[str, int],
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        generate_operator_set(
            family=OperatorFamily.AXIS_COMMUTING,
            seed=1,
            **kwargs,
        )


def test_analysis_rejects_non_projector_and_bad_hyperparameters() -> None:
    bad = [torch.ones((3, 3), dtype=torch.float64)]
    with pytest.raises(ValueError, match="idempotent"):
        analyze_joint_diagonalization(
            bad,
            steps=10,
            learning_rate=0.03,
            low_rank=1,
        )

    valid = generate_operator_set(
        family=OperatorFamily.AXIS_COMMUTING,
        dim=4,
        rank=1,
        count=2,
        seed=47,
    )
    with pytest.raises(ValueError, match="steps"):
        analyze_joint_diagonalization(
            valid.projectors,
            steps=0,
            learning_rate=0.03,
            low_rank=1,
        )
