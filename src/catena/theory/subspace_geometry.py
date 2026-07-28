from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class OrientationResult:
    scalar_regret: float
    diagonal_regret: float
    matrix_regret: float


def orientation_regrets(projector: torch.Tensor) -> OrientationResult:
    """Approximation error of a target linear projector by control families.

    Scalar control uses cI, channel-wise control uses a diagonal matrix, and
    matrix control uses the full operator. The target is evaluated as an
    operator, not on one vector, so a rotated projector cannot be disguised by
    per-example coordinate scaling.
    """
    dim = projector.shape[0]
    identity = torch.eye(dim, dtype=projector.dtype, device=projector.device)
    scalar = torch.trace(projector) / dim
    scalar_operator = scalar * identity
    diagonal_operator = torch.diag(torch.diag(projector))
    return OrientationResult(
        scalar_regret=float(torch.linalg.norm(projector - scalar_operator).item()),
        diagonal_regret=float(torch.linalg.norm(projector - diagonal_operator).item()),
        matrix_regret=0.0,
    )
