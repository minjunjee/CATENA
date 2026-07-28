from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch


class Operation(str, Enum):
    PRESERVE = "preserve"
    ADD = "add"
    INVALIDATE = "invalidate"
    SUPERSEDE = "supersede"

    @property
    def demand(self) -> tuple[float, float]:
        return {
            Operation.PRESERVE: (0.0, 0.0),
            Operation.ADD: (0.0, 1.0),
            Operation.INVALIDATE: (1.0, 0.0),
            Operation.SUPERSEDE: (1.0, 1.0),
        }[self]

    @property
    def is_asymmetric(self) -> bool:
        erase, write = self.demand
        return erase != write


class CandidateMode(str, Enum):
    ORACLE = "oracle_candidate"
    RECURRENT_READ = "recurrent_read"


class ControllerKind(str, Enum):
    TIED_SCALAR = "tied_scalar"
    DUAL_SCALAR = "dual_scalar"
    VECTOR = "vector"
    LOW_RANK = "low_rank"


class DemandOrientation(str, Enum):
    AXIS_CONTIGUOUS = "axis_contiguous"
    AXIS_SPARSE = "axis_sparse"
    ROTATED = "rotated"
    ORACLE_MASK = "oracle_mask"


@dataclass(slots=True)
class MemoryEpisode:
    episode_id: str
    operation: Operation
    keys: torch.Tensor
    values: torch.Tensor
    state: torch.Tensor
    target_state: torch.Tensor
    affected_index: int
    unaffected_indices: torch.Tensor
    old_value: torch.Tensor
    new_value: torch.Tensor
    erase_candidate: torch.Tensor
    write_candidate: torch.Tensor
    operation_features: torch.Tensor
    metadata: dict[str, Any]

    def to(self, device: torch.device | str) -> "MemoryEpisode":
        tensor_fields = {
            "keys",
            "values",
            "state",
            "target_state",
            "unaffected_indices",
            "old_value",
            "new_value",
            "erase_candidate",
            "write_candidate",
            "operation_features",
        }
        kwargs: dict[str, Any] = {}
        for field_name in self.__dataclass_fields__:  # type: ignore[attr-defined]
            value = getattr(self, field_name)
            if field_name in tensor_fields:
                kwargs[field_name] = value.to(device)
            else:
                kwargs[field_name] = value
        return MemoryEpisode(**kwargs)


@dataclass(slots=True)
class ExperimentResult:
    experiment_id: str
    status: str
    summary: dict[str, Any]
    artifact_dir: str
