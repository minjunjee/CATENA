"""Probe-independent graded projector families for the E03b theory branch.

For candidate ``tau`` this module constructs

    Q_tau(alpha) = Q exp(alpha A_tau)
    P_tau(alpha) = Q_tau(alpha) D_tau Q_tau(alpha)^T,

where ``D_tau`` is a rank-fixed diagonal mask.  Each stored unit skew generator
has spectral norm one.  The effective generator is

    A_tau = max_rotation_radians * unit_generator_tau.

Candidate masks, generators, identifiers, and construction hashes do not depend
on ``alpha``.  This makes an alpha sweep exactly paired.  No empirical probe is
accepted or generated here; probe construction belongs to the evaluation layer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, cast

import torch

from catena.core.provenance_v61 import (
    canonical_json_bytes,
    sha256_bytes,
    sha256_canonical_json,
)

GENERATOR_NORMALIZATION: Final[str] = (
    "unit_skew_generator has spectral norm 1; "
    "A_tau=max_rotation_radians*unit_skew_generator; "
    "Q_tau(alpha)=Q@matrix_exp(alpha*A_tau)"
)
MASK_SAMPLING_CONVENTION: Final[str] = (
    "uniform rank-r diagonal masks sampled without replacement across "
    "the combined train and heldout construction"
)
_HASH_SCHEMA_VERSION: Final[int] = 1
_MAX_TORCH_SEED: Final[int] = 2**63 - 1


class OperatorSplit(StrEnum):
    TRAIN = "train"
    HELDOUT = "heldout"


@dataclass(frozen=True, slots=True)
class GradedOperatorFamilySpec:
    """Alpha-invariant construction specification."""

    dim: int
    rank: int
    train_count: int
    heldout_count: int
    seed: int
    max_rotation_radians: float = math.pi

    def __post_init__(self) -> None:
        for name in ("dim", "rank", "train_count", "heldout_count", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
        if self.dim < 2:
            raise ValueError("dim must be at least 2.")
        if not 0 < self.rank < self.dim:
            raise ValueError("rank must lie strictly between zero and dim.")
        if self.train_count < 1:
            raise ValueError("train_count must be positive.")
        if self.heldout_count < 1:
            raise ValueError("heldout_count must be positive.")
        if self.total_count > math.comb(self.dim, self.rank):
            raise ValueError(
                "train_count + heldout_count exceeds the number of unique rank-r masks."
            )
        if not 0 <= self.seed <= _MAX_TORCH_SEED:
            raise ValueError(f"seed must lie in [0, {_MAX_TORCH_SEED}].")
        if isinstance(self.max_rotation_radians, bool) or not isinstance(
            self.max_rotation_radians, (int, float)
        ):
            raise TypeError("max_rotation_radians must be numeric.")
        if not math.isfinite(float(self.max_rotation_radians)) or self.max_rotation_radians <= 0.0:
            raise ValueError("max_rotation_radians must be positive and finite.")

    @property
    def total_count(self) -> int:
        return self.train_count + self.heldout_count

    def hash_payload(self) -> dict[str, int | float | str]:
        return {
            "schema_version": _HASH_SCHEMA_VERSION,
            "construction": "graded_operator_family",
            "dim": self.dim,
            "rank": self.rank,
            "train_count": self.train_count,
            "heldout_count": self.heldout_count,
            "seed": self.seed,
            "max_rotation_radians": float(self.max_rotation_radians),
            "generator_normalization": GENERATOR_NORMALIZATION,
            "mask_sampling_convention": MASK_SAMPLING_CONVENTION,
            "dtype": "torch.float64",
        }

    def generate(self, alpha: float) -> GradedOperatorFamily:
        """Regenerate this exact paired construction at another alpha."""

        return _generate_from_spec(self, alpha)


@dataclass(frozen=True, slots=True)
class GradedOperatorCandidate:
    """One deterministic train or held-out projector in a graded family."""

    candidate_id: str
    split: OperatorSplit
    split_index: int
    global_index: int
    diagonal_mask: torch.Tensor
    unit_skew_generator: torch.Tensor
    skew_generator: torch.Tensor
    rotation_magnitude_radians: float
    projector: torch.Tensor
    base_sha256: str
    operator_sha256: str


@dataclass(frozen=True, slots=True)
class GradedOperatorFamily:
    """A realized alpha slice with paired construction and realization identities."""

    spec: GradedOperatorFamilySpec
    alpha: float
    base_basis: torch.Tensor
    base_basis_sha256: str
    candidates: tuple[GradedOperatorCandidate, ...]
    construction_sha256: str
    realization_sha256: str
    base_candidate_id: str
    realization_id: str
    generator_normalization: str = GENERATOR_NORMALIZATION
    mask_sampling_convention: str = MASK_SAMPLING_CONVENTION

    @property
    def rotation_magnitude_radians(self) -> float:
        return self.alpha * float(self.spec.max_rotation_radians)

    @property
    def train_candidates(self) -> tuple[GradedOperatorCandidate, ...]:
        return tuple(
            candidate for candidate in self.candidates if candidate.split is OperatorSplit.TRAIN
        )

    @property
    def heldout_candidates(self) -> tuple[GradedOperatorCandidate, ...]:
        return tuple(
            candidate for candidate in self.candidates if candidate.split is OperatorSplit.HELDOUT
        )

    def projectors(
        self,
        split: OperatorSplit | str | None = None,
    ) -> tuple[torch.Tensor, ...]:
        selected = self.candidates
        if split is not None:
            try:
                normalized_split = OperatorSplit(split)
            except ValueError as exc:
                raise ValueError(f"Unknown operator split: {split!r}.") from exc
            selected = tuple(
                candidate for candidate in self.candidates if candidate.split is normalized_split
            )
        return tuple(candidate.projector for candidate in selected)

    def regenerate(self, alpha: float) -> GradedOperatorFamily:
        """Regenerate the same base masks and generators at ``alpha``."""

        return self.spec.generate(alpha)

    def identity_record(self) -> dict[str, object]:
        """Return a JSON-safe identity record for manifests and candidate registries."""

        return {
            "schema_version": _HASH_SCHEMA_VERSION,
            "base_candidate_id": self.base_candidate_id,
            "realization_id": self.realization_id,
            "construction_sha256": self.construction_sha256,
            "realization_sha256": self.realization_sha256,
            "base_basis_sha256": self.base_basis_sha256,
            "alpha": self.alpha,
            "spec": self.spec.hash_payload(),
            "mask_sampling_convention": self.mask_sampling_convention,
            "candidate_ids": [candidate.candidate_id for candidate in self.candidates],
            "operator_sha256": [candidate.operator_sha256 for candidate in self.candidates],
        }


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash tensor shape, dtype, and contiguous CPU bytes."""

    value = tensor.detach().cpu().contiguous()
    header = canonical_json_bytes(
        {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }
    )
    return sha256_bytes(header + b"\0" + value.view(torch.uint8).numpy().tobytes())


def _validate_alpha(alpha: float) -> float:
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise TypeError("alpha must be numeric.")
    result = float(alpha)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("alpha must be finite and lie in [0, 1].")
    return result


def _derived_seed(seed: int, domain: str) -> int:
    digest = sha256_canonical_json(
        {
            "schema_version": _HASH_SCHEMA_VERSION,
            "seed": seed,
            "domain": domain,
        }
    )
    return int(digest[:16], 16) % _MAX_TORCH_SEED


def _canonical_orthogonal_basis(dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(_derived_seed(seed, "base_basis"))
    raw = torch.randn(dim, dim, generator=generator, dtype=torch.float64)
    basis, upper = torch.linalg.qr(raw)
    diagonal = torch.diag(upper)
    signs = torch.where(
        diagonal < 0.0,
        -torch.ones_like(diagonal),
        torch.ones_like(diagonal),
    )
    return cast(torch.Tensor, (basis * signs.unsqueeze(0)).contiguous())


def _unit_skew_generator(
    spec: GradedOperatorFamilySpec,
    split: OperatorSplit,
    split_index: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(
        _derived_seed(spec.seed, f"skew:{split.value}:{split_index}")
    )
    raw = torch.randn(
        spec.dim,
        spec.dim,
        generator=generator,
        dtype=torch.float64,
    )
    skew = raw - raw.transpose(0, 1)
    spectral_norm = torch.linalg.matrix_norm(skew, ord=2)
    if not bool(torch.isfinite(spectral_norm).item()) or float(spectral_norm.item()) <= 0.0:
        raise AssertionError("Internal random skew generator is degenerate.")
    return cast(torch.Tensor, (skew / spectral_norm).contiguous())


def _unique_diagonal_masks(
    spec: GradedOperatorFamilySpec,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cpu").manual_seed(
        _derived_seed(spec.seed, "uniform_unique_masks")
    )
    signatures: set[tuple[int, ...]] = set()
    masks: list[torch.Tensor] = []
    while len(masks) < spec.total_count:
        indices = torch.randperm(spec.dim, generator=generator)[: spec.rank]
        signature = tuple(sorted(int(index) for index in indices.tolist()))
        if signature in signatures:
            continue
        signatures.add(signature)
        diagonal = torch.zeros(spec.dim, dtype=torch.float64)
        diagonal[indices] = 1.0
        masks.append(torch.diag(diagonal))
    return tuple(masks)


def _candidate_base_payload(
    *,
    spec: GradedOperatorFamilySpec,
    split: OperatorSplit,
    split_index: int,
    global_index: int,
    base_basis_sha256: str,
    diagonal_mask: torch.Tensor,
    unit_skew_generator: torch.Tensor,
    skew_generator: torch.Tensor,
) -> dict[str, object]:
    return {
        "schema_version": _HASH_SCHEMA_VERSION,
        "spec": spec.hash_payload(),
        "split": split.value,
        "split_index": split_index,
        "global_index": global_index,
        "base_basis_sha256": base_basis_sha256,
        "diagonal_mask_sha256": tensor_sha256(diagonal_mask),
        "unit_skew_generator_sha256": tensor_sha256(unit_skew_generator),
        "skew_generator_sha256": tensor_sha256(skew_generator),
    }


def _generate_from_spec(
    spec: GradedOperatorFamilySpec,
    alpha: float,
) -> GradedOperatorFamily:
    alpha_value = _validate_alpha(alpha)
    base_basis = _canonical_orthogonal_basis(spec.dim, spec.seed)
    base_basis_sha256 = tensor_sha256(base_basis)
    diagonal_masks = _unique_diagonal_masks(spec)

    candidate_inputs: list[
        tuple[
            OperatorSplit,
            int,
            int,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            str,
            str,
        ]
    ] = []
    global_index = 0
    for split, count in (
        (OperatorSplit.TRAIN, spec.train_count),
        (OperatorSplit.HELDOUT, spec.heldout_count),
    ):
        for split_index in range(count):
            diagonal_mask = diagonal_masks[global_index]
            unit_generator = _unit_skew_generator(spec, split, split_index)
            skew_generator = float(spec.max_rotation_radians) * unit_generator
            base_payload = _candidate_base_payload(
                spec=spec,
                split=split,
                split_index=split_index,
                global_index=global_index,
                base_basis_sha256=base_basis_sha256,
                diagonal_mask=diagonal_mask,
                unit_skew_generator=unit_generator,
                skew_generator=skew_generator,
            )
            base_sha256 = sha256_canonical_json(base_payload)
            candidate_id = f"{split.value}-{split_index:04d}-{base_sha256[:16]}"
            candidate_inputs.append(
                (
                    split,
                    split_index,
                    global_index,
                    diagonal_mask,
                    unit_generator,
                    skew_generator,
                    base_sha256,
                    candidate_id,
                )
            )
            global_index += 1

    construction_sha256 = sha256_canonical_json(
        {
            "schema_version": _HASH_SCHEMA_VERSION,
            "spec": spec.hash_payload(),
            "base_basis_sha256": base_basis_sha256,
            "candidate_base_sha256": [candidate_input[6] for candidate_input in candidate_inputs],
        }
    )
    base_candidate_id = f"graded-s{spec.seed}-{construction_sha256[:20]}"

    candidates: list[GradedOperatorCandidate] = []
    for (
        split,
        split_index,
        candidate_global_index,
        diagonal_mask,
        unit_generator,
        skew_generator,
        base_sha256,
        candidate_id,
    ) in candidate_inputs:
        rotation = torch.matrix_exp(alpha_value * skew_generator)
        candidate_basis = base_basis @ rotation
        projector = (candidate_basis @ diagonal_mask @ candidate_basis.transpose(0, 1)).contiguous()
        projector_sha256 = tensor_sha256(projector)
        operator_sha256 = sha256_canonical_json(
            {
                "schema_version": _HASH_SCHEMA_VERSION,
                "candidate_id": candidate_id,
                "base_sha256": base_sha256,
                "alpha": alpha_value,
                "projector_sha256": projector_sha256,
            }
        )
        candidates.append(
            GradedOperatorCandidate(
                candidate_id=candidate_id,
                split=split,
                split_index=split_index,
                global_index=candidate_global_index,
                diagonal_mask=diagonal_mask,
                unit_skew_generator=unit_generator,
                skew_generator=skew_generator,
                rotation_magnitude_radians=(alpha_value * float(spec.max_rotation_radians)),
                projector=projector,
                base_sha256=base_sha256,
                operator_sha256=operator_sha256,
            )
        )

    realization_sha256 = sha256_canonical_json(
        {
            "schema_version": _HASH_SCHEMA_VERSION,
            "construction_sha256": construction_sha256,
            "alpha": alpha_value,
            "operator_sha256": [candidate.operator_sha256 for candidate in candidates],
        }
    )
    return GradedOperatorFamily(
        spec=spec,
        alpha=alpha_value,
        base_basis=base_basis,
        base_basis_sha256=base_basis_sha256,
        candidates=tuple(candidates),
        construction_sha256=construction_sha256,
        realization_sha256=realization_sha256,
        base_candidate_id=base_candidate_id,
        realization_id=f"graded-realization-{realization_sha256[:20]}",
    )


def generate_graded_operator_family(
    *,
    dim: int,
    rank: int,
    train_count: int,
    heldout_count: int,
    seed: int,
    alpha: float,
    max_rotation_radians: float = math.pi,
) -> GradedOperatorFamily:
    """Generate a deterministic, paired graded operator-family realization."""

    spec = GradedOperatorFamilySpec(
        dim=dim,
        rank=rank,
        train_count=train_count,
        heldout_count=heldout_count,
        seed=seed,
        max_rotation_radians=max_rotation_radians,
    )
    return spec.generate(alpha)
