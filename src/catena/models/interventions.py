from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch

from catena.models.memory import GateOutput


class InterventionKind(str, Enum):
    NONE = "none"
    ERASE_CLAMP = "erase_clamp"
    WRITE_CLAMP = "write_clamp"
    SCALARIZE = "scalarize"
    SWAP = "swap"
    CHANNEL_PERMUTE = "channel_permute"
    TRANSPLANT = "transplant"
    RESCUE_ORACLE = "rescue_oracle"


@dataclass(slots=True)
class Intervention:
    kind: InterventionKind
    dose: float = 1.0
    permutation: torch.Tensor | None = None
    donor: GateOutput | None = None
    oracle: GateOutput | None = None


def apply_intervention(gates: GateOutput, intervention: Intervention) -> GateOutput:
    if intervention.kind is InterventionKind.NONE:
        return gates
    if intervention.kind is InterventionKind.ERASE_CLAMP:
        return GateOutput(erase=gates.erase * intervention.dose, write=gates.write)
    if intervention.kind is InterventionKind.WRITE_CLAMP:
        return GateOutput(erase=gates.erase, write=gates.write * intervention.dose)
    if intervention.kind is InterventionKind.SCALARIZE:
        erase = gates.erase.mean().expand_as(gates.erase)
        write = gates.write.mean().expand_as(gates.write)
        return GateOutput(erase=erase, write=write)
    if intervention.kind is InterventionKind.SWAP:
        return GateOutput(erase=gates.write, write=gates.erase)
    if intervention.kind is InterventionKind.CHANNEL_PERMUTE:
        if intervention.permutation is None:
            raise ValueError("CHANNEL_PERMUTE requires a permutation tensor.")
        return GateOutput(
            erase=gates.erase[..., intervention.permutation],
            write=gates.write[..., intervention.permutation],
        )
    if intervention.kind is InterventionKind.TRANSPLANT:
        if intervention.donor is None:
            raise ValueError("TRANSPLANT requires donor gates.")
        return intervention.donor
    if intervention.kind is InterventionKind.RESCUE_ORACLE:
        if intervention.oracle is None:
            raise ValueError("RESCUE_ORACLE requires oracle gates.")
        return intervention.oracle
    raise ValueError(f"Unknown intervention: {intervention.kind}")
