from __future__ import annotations

import inspect
import math
from collections.abc import Iterator

import pytest
import torch

from catena.data.graded_operator_families import (
    GENERATOR_NORMALIZATION,
    MASK_SAMPLING_CONVENTION,
    GradedOperatorFamilySpec,
    OperatorSplit,
    generate_graded_operator_family,
    tensor_sha256,
)
from catena.theory.joint_diagonalization import commutator_norm


@pytest.fixture(scope="module", autouse=True)
def _small_torch_thread_pool() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _family(alpha: float, *, seed: int = 101, max_rotation: float = math.pi):
    return generate_graded_operator_family(
        dim=8,
        rank=2,
        train_count=6,
        heldout_count=3,
        seed=seed,
        alpha=alpha,
        max_rotation_radians=max_rotation,
    )


def test_formula_uses_spectral_normalized_skew_generators() -> None:
    family = _family(0.4)
    assert family.spec.max_rotation_radians == pytest.approx(math.pi)
    assert family.rotation_magnitude_radians == pytest.approx(0.4 * math.pi)
    assert family.generator_normalization == GENERATOR_NORMALIZATION
    assert family.mask_sampling_convention == MASK_SAMPLING_CONVENTION

    for candidate in family.candidates:
        unit = candidate.unit_skew_generator
        assert unit.dtype == torch.float64
        assert torch.allclose(unit + unit.T, torch.zeros_like(unit), atol=0.0, rtol=0.0)
        assert float(torch.linalg.matrix_norm(unit, ord=2).item()) == pytest.approx(
            1.0,
            abs=1e-12,
        )
        assert torch.allclose(
            candidate.skew_generator,
            math.pi * unit,
            atol=1e-14,
            rtol=0.0,
        )
        rotation = torch.matrix_exp(family.alpha * candidate.skew_generator)
        basis = family.base_basis @ rotation
        expected = basis @ candidate.diagonal_mask @ basis.T
        assert torch.allclose(candidate.projector, expected, atol=1e-14, rtol=0.0)
        assert candidate.rotation_magnitude_radians == pytest.approx(0.4 * math.pi)


def test_regeneration_is_deterministic_and_does_not_use_global_rng() -> None:
    torch.manual_seed(777)
    state_before = torch.random.get_rng_state().clone()
    first = _family(0.35)
    assert torch.equal(torch.random.get_rng_state(), state_before)

    torch.manual_seed(123456)
    _ = torch.randn(17)
    second = first.regenerate(0.35)
    assert first.identity_record() == second.identity_record()
    assert torch.equal(first.base_basis, second.base_basis)
    for left, right in zip(first.candidates, second.candidates, strict=True):
        assert left.candidate_id == right.candidate_id
        assert left.base_sha256 == right.base_sha256
        assert left.operator_sha256 == right.operator_sha256
        assert torch.equal(left.diagonal_mask, right.diagonal_mask)
        assert torch.equal(left.unit_skew_generator, right.unit_skew_generator)
        assert torch.equal(left.projector, right.projector)


def test_alpha_sweep_preserves_paired_construction_and_changes_realization() -> None:
    zero = _family(0.0)
    rotated = zero.regenerate(0.5)
    assert zero.construction_sha256 == rotated.construction_sha256
    assert zero.base_candidate_id == rotated.base_candidate_id
    assert zero.base_basis_sha256 == rotated.base_basis_sha256
    assert zero.realization_sha256 != rotated.realization_sha256
    assert zero.realization_id != rotated.realization_id

    changed_projectors = 0
    for base, perturbed in zip(zero.candidates, rotated.candidates, strict=True):
        assert base.candidate_id == perturbed.candidate_id
        assert base.base_sha256 == perturbed.base_sha256
        assert torch.equal(base.diagonal_mask, perturbed.diagonal_mask)
        assert torch.equal(base.unit_skew_generator, perturbed.unit_skew_generator)
        assert base.operator_sha256 != perturbed.operator_sha256
        changed_projectors += int(not torch.equal(base.projector, perturbed.projector))
    assert changed_projectors == len(zero.candidates)


def test_seed_and_max_rotation_are_bound_into_construction_hash() -> None:
    registered = _family(0.25)
    changed_seed = _family(0.25, seed=102)
    changed_rotation = _family(0.25, max_rotation=math.pi / 2.0)
    assert registered.construction_sha256 != changed_seed.construction_sha256
    assert registered.construction_sha256 != changed_rotation.construction_sha256
    assert registered.base_candidate_id != changed_seed.base_candidate_id
    assert registered.base_candidate_id != changed_rotation.base_candidate_id


def test_train_and_heldout_splits_are_explicit_and_disjoint() -> None:
    family = _family(0.2)
    assert len(family.train_candidates) == 6
    assert len(family.heldout_candidates) == 3
    assert family.projectors(OperatorSplit.TRAIN) == tuple(
        candidate.projector for candidate in family.train_candidates
    )
    assert family.projectors("heldout") == tuple(
        candidate.projector for candidate in family.heldout_candidates
    )
    train_ids = {candidate.candidate_id for candidate in family.train_candidates}
    heldout_ids = {candidate.candidate_id for candidate in family.heldout_candidates}
    assert train_ids.isdisjoint(heldout_ids)
    assert len({tensor_sha256(candidate.diagonal_mask) for candidate in family.candidates}) == len(
        family.candidates
    )
    assert [candidate.split_index for candidate in family.train_candidates] == list(range(6))
    assert [candidate.split_index for candidate in family.heldout_candidates] == list(range(3))
    with pytest.raises(ValueError, match="Unknown operator split"):
        family.projectors("validation")


def test_uniform_unique_masks_have_approximately_exchangeable_coordinates() -> None:
    inclusion_counts = torch.zeros(8, dtype=torch.int64)
    for seed in range(20_000, 20_064):
        family = _family(0.0, seed=seed)
        for candidate in family.candidates:
            inclusion_counts += torch.diag(candidate.diagonal_mask).to(torch.int64)

    expected = 64 * 9 * 2 / 8
    assert int(inclusion_counts.sum().item()) == 64 * 9 * 2
    assert bool(torch.all(inclusion_counts > 0).item())
    assert max(abs(float(value) - expected) for value in inclusion_counts) < 25
    assert int((inclusion_counts.max() - inclusion_counts.min()).item()) < 40


def test_projectors_are_float64_symmetric_idempotent_and_rank_fixed() -> None:
    family = _family(0.75)
    identity = torch.eye(family.spec.dim, dtype=torch.float64)
    assert family.base_basis.dtype == torch.float64
    assert torch.allclose(
        family.base_basis.T @ family.base_basis,
        identity,
        atol=1e-12,
        rtol=0.0,
    )
    for candidate in family.candidates:
        projector = candidate.projector
        assert projector.dtype == torch.float64
        assert torch.allclose(projector, projector.T, atol=1e-12, rtol=0.0)
        assert torch.allclose(projector @ projector, projector, atol=1e-12, rtol=0.0)
        assert int(torch.linalg.matrix_rank(projector).item()) == family.spec.rank


def test_zero_alpha_commutes_and_endpoint_is_nonzero() -> None:
    zero = _family(0.0)
    endpoint = _family(1.0)
    assert commutator_norm(list(zero.projectors())) < 1e-12
    assert commutator_norm(list(endpoint.projectors())) > 1e-2
    for candidate in zero.candidates:
        diagonalized = zero.base_basis.T @ candidate.projector @ zero.base_basis
        assert torch.allclose(
            diagonalized,
            candidate.diagonal_mask,
            atol=1e-12,
            rtol=0.0,
        )


def test_small_alpha_curve_has_an_increasing_commutator_diagnostic() -> None:
    # This is a regression diagnostic for the registered construction, not a
    # general theorem that kappa or R_JD is globally monotone in alpha.
    values = [commutator_norm(list(_family(alpha).projectors())) for alpha in (0.0, 0.05, 0.1)]
    assert values[0] < 1e-12
    assert values[0] < values[1] < values[2]


def test_construction_is_probe_independent_by_api_and_identity() -> None:
    signature = inspect.signature(generate_graded_operator_family)
    assert all("probe" not in parameter for parameter in signature.parameters)
    family = _family(0.3)
    record = family.identity_record()
    assert "probe" not in repr(record).lower()
    assert record["construction_sha256"] == family.construction_sha256
    assert record["realization_sha256"] == family.realization_sha256


@pytest.mark.parametrize(
    ("overrides", "exception"),
    [
        ({"dim": 1}, ValueError),
        ({"rank": 0}, ValueError),
        ({"rank": 8}, ValueError),
        ({"train_count": 0}, ValueError),
        ({"heldout_count": 0}, ValueError),
        (
            {
                "dim": 2,
                "rank": 1,
                "train_count": 2,
                "heldout_count": 1,
            },
            ValueError,
        ),
        ({"seed": -1}, ValueError),
        ({"seed": True}, TypeError),
        ({"max_rotation_radians": 0.0}, ValueError),
        ({"max_rotation_radians": float("nan")}, ValueError),
    ],
)
def test_spec_rejects_invalid_construction(
    overrides: dict[str, object],
    exception: type[Exception],
) -> None:
    values: dict[str, object] = {
        "dim": 8,
        "rank": 2,
        "train_count": 6,
        "heldout_count": 3,
        "seed": 101,
        "max_rotation_radians": math.pi,
    }
    values.update(overrides)
    with pytest.raises(exception):
        GradedOperatorFamilySpec(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("alpha", [-0.1, 1.1, float("nan"), float("inf")])
def test_generator_rejects_alpha_outside_registered_domain(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        _family(alpha)


def test_tensor_hash_includes_shape_dtype_and_content() -> None:
    base = torch.arange(4, dtype=torch.float64).reshape(2, 2)
    assert tensor_sha256(base) == tensor_sha256(base.clone())
    assert tensor_sha256(base) != tensor_sha256(base.reshape(4))
    assert tensor_sha256(base) != tensor_sha256(base.to(torch.float32))
    changed = base.clone()
    changed[0, 0] = 1.0
    assert tensor_sha256(base) != tensor_sha256(changed)
