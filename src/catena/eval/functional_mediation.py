from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import torch

from catena.core.schema import Operation
from catena.models.memory import GateOutput


class GateChannel(StrEnum):
    ERASE = "erase"
    WRITE = "write"
    JOINT = "joint"


@dataclass(frozen=True, slots=True)
class NormMatchResult:
    gates: GateOutput
    donor_norm: float
    recipient_norm: float
    matched_norm: float
    absolute_mismatch: float
    scale: float


def gate_vector(gates: GateOutput) -> torch.Tensor:
    erase = gates.erase.reshape(())
    write = gates.write.reshape(())
    if erase.device != write.device:
        raise ValueError("Erase and write gates must be on the same device.")
    if erase.dtype != write.dtype:
        raise ValueError("Erase and write gates must have the same dtype.")
    vector = torch.stack((erase, write))
    if not bool(torch.isfinite(vector).all().item()):
        raise FloatingPointError("Gate vector contains a non-finite value.")
    return vector


def gate_from_vector(vector: torch.Tensor) -> GateOutput:
    if vector.shape != (2,):
        raise ValueError(f"Gate vector must have shape (2,), got {tuple(vector.shape)}.")
    if not bool(torch.isfinite(vector).all().item()):
        raise FloatingPointError("Gate vector contains a non-finite value.")
    return GateOutput(erase=vector[0], write=vector[1])


def exact_feasible_l2_norm_match(
    donor: GateOutput,
    recipient: GateOutput,
    *,
    tolerance: float,
) -> NormMatchResult:
    """Scale a donor without clipping and require an in-box exact L2 match."""

    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("Norm-match tolerance must be positive and finite.")
    donor_vector = gate_vector(donor)
    recipient_vector = gate_vector(recipient)
    if bool(((donor_vector < 0.0) | (donor_vector > 1.0)).any().item()):
        raise ValueError("Donor gate lies outside [0, 1].")
    if bool(((recipient_vector < 0.0) | (recipient_vector > 1.0)).any().item()):
        raise ValueError("Recipient gate lies outside [0, 1].")

    donor_norm_tensor = torch.linalg.vector_norm(donor_vector)
    recipient_norm_tensor = torch.linalg.vector_norm(recipient_vector)
    donor_norm = float(donor_norm_tensor.item())
    recipient_norm = float(recipient_norm_tensor.item())
    if donor_norm <= tolerance:
        if recipient_norm <= tolerance:
            scale = 0.0
            matched_vector = torch.zeros_like(donor_vector)
        else:
            raise ValueError("A near-zero donor cannot be norm-matched to this recipient.")
    else:
        scale = recipient_norm / donor_norm
        matched_vector = donor_vector * scale

    if bool(((matched_vector < -tolerance) | (matched_vector > 1.0 + tolerance)).any().item()):
        raise ValueError("Exact donor norm matching would leave the feasible gate box.")
    matched_vector = torch.where(
        torch.abs(matched_vector) <= tolerance,
        torch.zeros_like(matched_vector),
        matched_vector,
    )
    matched_vector = torch.where(
        torch.abs(matched_vector - 1.0) <= tolerance,
        torch.ones_like(matched_vector),
        matched_vector,
    )
    matched_norm = float(torch.linalg.vector_norm(matched_vector).item())
    mismatch = abs(matched_norm - recipient_norm)
    if mismatch > tolerance:
        raise ValueError(
            f"Donor norm mismatch {mismatch} exceeds tolerance {tolerance}."
        )
    return NormMatchResult(
        gates=gate_from_vector(matched_vector),
        donor_norm=donor_norm,
        recipient_norm=recipient_norm,
        matched_norm=matched_norm,
        absolute_mismatch=mismatch,
        scale=scale,
    )


def dose_gate(gates: GateOutput, channel: GateChannel, dose: float) -> GateOutput:
    if not np.isfinite(dose) or not 0.0 <= dose <= 1.0:
        raise ValueError("Dose must be finite and lie in [0, 1].")
    erase = gates.erase.clone()
    write = gates.write.clone()
    if channel in {GateChannel.ERASE, GateChannel.JOINT}:
        erase = erase * dose
    if channel in {GateChannel.WRITE, GateChannel.JOINT}:
        write = write * dose
    return GateOutput(erase=erase, write=write)


def relevant_channels(operation: Operation) -> tuple[GateChannel, ...]:
    return {
        Operation.PRESERVE: (),
        Operation.ADD: (GateChannel.WRITE,),
        Operation.INVALIDATE: (GateChannel.ERASE,),
        Operation.SUPERSEDE: (GateChannel.ERASE, GateChannel.WRITE),
    }[operation]


def restore_relevant(
    damaged: GateOutput,
    donor: GateOutput,
    operation: Operation,
) -> GateOutput:
    erase = damaged.erase.clone()
    write = damaged.write.clone()
    if GateChannel.ERASE in relevant_channels(operation):
        erase = donor.erase
    if GateChannel.WRITE in relevant_channels(operation):
        write = donor.write
    return GateOutput(erase=erase, write=write)


def scalarize_gate(gates: GateOutput) -> GateOutput:
    beta = (gates.erase + gates.write) / 2.0
    return GateOutput(erase=beta, write=beta)


def recovery_fraction_from_means(
    *,
    damaged_error: float,
    rescued_error: float,
    baseline_error: float,
    minimum_headroom: float,
) -> float:
    values = np.asarray(
        [damaged_error, rescued_error, baseline_error, minimum_headroom],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise ValueError("Recovery inputs must be finite.")
    if minimum_headroom <= 0.0:
        raise ValueError("minimum_headroom must be positive.")
    denominator = damaged_error - baseline_error
    if denominator <= minimum_headroom:
        raise ValueError(
            f"Rescue headroom {denominator} does not exceed {minimum_headroom}."
        )
    return float((damaged_error - rescued_error) / denominator)


def monotonic_nonincreasing_fraction(
    errors_by_ascending_dose: np.ndarray,
    *,
    tolerance: float,
) -> float:
    errors = np.asarray(errors_by_ascending_dose, dtype=np.float64)
    if errors.ndim != 1 or len(errors) < 2:
        raise ValueError("A dose curve requires at least two errors.")
    if not np.isfinite(errors).all():
        raise ValueError("Dose errors must be finite.")
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("Monotonic tolerance must be non-negative and finite.")
    return float(np.mean(np.diff(errors) <= tolerance))
