from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum
from typing import cast

import torch


class StructuredTransferCondition(StrEnum):
    A_ORACLE_ADDRESS_ORACLE_CANDIDATE = "A_oracle_address_oracle_candidate"
    B_LEARNED_ADDRESS_ORACLE_CANDIDATE = "B_learned_address_oracle_candidate"
    C_ORACLE_ADDRESS_STATE_READ_CANDIDATE = (
        "C_oracle_address_state_read_candidate"
    )
    D_LEARNED_ADDRESS_STATE_READ_CANDIDATE = (
        "D_learned_address_state_read_candidate"
    )

    @property
    def uses_oracle_address(self) -> bool:
        return self in {
            self.A_ORACLE_ADDRESS_ORACLE_CANDIDATE,
            self.C_ORACLE_ADDRESS_STATE_READ_CANDIDATE,
        }

    @property
    def uses_oracle_candidate(self) -> bool:
        return self in {
            self.A_ORACLE_ADDRESS_ORACLE_CANDIDATE,
            self.B_LEARNED_ADDRESS_ORACLE_CANDIDATE,
        }


class StructuredTransferDemand(StrEnum):
    MAGNITUDE = "magnitude_factorization"
    VALUE_GRANULARITY = "value_granularity"
    ADDRESS_DECOUPLING = "address_decoupling"
    STATE_CONDITIONING = "state_conditioning"


@dataclass(slots=True)
class StructuredSequenceTransferInput:
    """Model-visible fields; integer slots, old values and update_mask are absent."""

    initial_state: torch.Tensor
    identifier_features: torch.Tensor
    new_candidates: torch.Tensor
    family_one_hot: torch.Tensor
    operation_one_hot: torch.Tensor
    channel_masks: torch.Tensor
    relation_features: torch.Tensor
    verified_flags: torch.Tensor


@dataclass(slots=True)
class StructuredSequenceTransferBatch:
    inputs: StructuredSequenceTransferInput
    update_mask: torch.Tensor
    erase_addresses: torch.Tensor
    write_addresses: torch.Tensor
    old_candidates: torch.Tensor
    target_erase_gates: torch.Tensor
    target_write_gates: torch.Tensor
    target_state: torch.Tensor
    affected_entities: torch.Tensor
    demand_family: StructuredTransferDemand


def structured_event_feature_dim(identifier_dim: int, value_dim: int) -> int:
    if identifier_dim <= 0 or value_dim <= 0:
        raise ValueError("identifier_dim and value_dim must be positive")
    # source/destination codes + new value + family + operation + channel mask
    # + same/different relation + verified provenance bit
    return 2 * int(identifier_dim) + 2 * int(value_dim) + 4 + 4 + 2 + 1


def make_structured_identifier_codebook(
    *,
    slots: int,
    code_dim: int,
    seed: int,
) -> torch.Tensor:
    if slots < 2:
        raise ValueError("slots must be at least 2")
    if code_dim < 2:
        raise ValueError("code_dim must be at least 2")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    codebook = torch.randn(slots, code_dim, generator=generator)
    return cast(
        torch.Tensor,
        codebook / codebook.norm(dim=-1, keepdim=True).clamp_min(1e-8),
    )


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(repr(tuple(tensor.shape)).encode("utf-8"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def indexed_structured_sequence_seed(
    seed: int,
    namespace: str,
    index: int,
) -> int:
    if seed < 0 or index < 0:
        raise ValueError("seed and index must be non-negative")
    digest = hashlib.sha256(
        f"{int(seed)}\0{namespace}\0{int(index)}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def _generator(seed: int, namespace: str) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(
        indexed_structured_sequence_seed(seed, namespace, 0)
    )


def _normalized_values(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    value = torch.randn(*shape, generator=generator)
    return cast(
        torch.Tensor,
        value / value.norm(dim=-1, keepdim=True).clamp_min(1e-8),
    )


def _contiguous_masks(
    *,
    batch_size: int,
    events: int,
    value_dim: int,
    width_generator: torch.Generator,
    start_generator: torch.Generator,
) -> torch.Tensor:
    widths = torch.randint(
        1,
        value_dim,
        (batch_size, events),
        generator=width_generator,
    )
    starts = (
        torch.rand(batch_size, events, generator=start_generator)
        * (value_dim - widths + 1)
    ).to(torch.long)
    coordinates = torch.arange(value_dim).view(1, 1, value_dim)
    return (
        (coordinates >= starts.unsqueeze(-1))
        & (coordinates < (starts + widths).unsqueeze(-1))
    ).to(torch.float32)


def _verified_positions(updates: int, gap_events: int) -> torch.Tensor:
    if updates == 1:
        return torch.zeros(1, dtype=torch.long)
    return torch.cat(
        (
            torch.zeros(1, dtype=torch.long),
            torch.arange(1, updates, dtype=torch.long) + gap_events,
        )
    )


def _family_one_hot(
    family: StructuredTransferDemand,
    *,
    batch_size: int,
    events: int,
) -> torch.Tensor:
    result = torch.zeros(batch_size, events, 4)
    result[:, :, list(StructuredTransferDemand).index(family)] = 1.0
    return result


def _relation_features(
    erase_addresses: torch.Tensor,
    write_addresses: torch.Tensor,
) -> torch.Tensor:
    different = erase_addresses != write_addresses
    return torch.stack((~different, different), dim=-1).to(torch.float32)


def _target_gates(
    *,
    family: StructuredTransferDemand,
    old_candidate: torch.Tensor,
    operation: torch.Tensor,
    channel_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, value_dim = old_candidate.shape
    if family is StructuredTransferDemand.MAGNITUDE:
        erase = ((operation == 2) | (operation == 3)).to(torch.float32)
        write = ((operation == 1) | (operation == 3)).to(torch.float32)
        return (
            erase[:, None].expand(batch_size, value_dim),
            write[:, None].expand(batch_size, value_dim),
        )
    if family is StructuredTransferDemand.VALUE_GRANULARITY:
        return channel_mask, channel_mask
    if family is StructuredTransferDemand.ADDRESS_DECOUPLING:
        ones = torch.ones_like(old_candidate)
        return ones, ones
    if family is StructuredTransferDemand.STATE_CONDITIONING:
        erase = (old_candidate[:, 0] > 0.0).to(torch.float32)
        write = 1.0 - erase
        return (
            erase[:, None].expand(batch_size, value_dim),
            write[:, None].expand(batch_size, value_dim),
        )
    raise AssertionError(f"Unhandled demand family: {family}")


def _apply_update(
    *,
    state: torch.Tensor,
    erase_address: torch.Tensor,
    write_address: torch.Tensor,
    old_candidate: torch.Tensor,
    new_candidate: torch.Tensor,
    erase_gate: torch.Tensor,
    write_gate: torch.Tensor,
) -> torch.Tensor:
    batch_size = state.shape[0]
    row = torch.arange(batch_size)
    result = state.clone()
    result[row, erase_address] -= erase_gate * old_candidate
    result[row, write_address] += write_gate * new_candidate
    return result


def generate_structured_sequence_transfer_batch(
    *,
    family: StructuredTransferDemand,
    batch_size: int,
    slots: int,
    value_dim: int,
    updates: int,
    gap_events: int,
    state_scale: float,
    identifier_codebook: torch.Tensor,
    seed: int,
    base_namespace: str,
    distractor_namespace: str,
    device: torch.device,
) -> StructuredSequenceTransferBatch:
    if batch_size <= 0 or updates <= 0:
        raise ValueError("batch_size and updates must be positive")
    if slots < 2 or value_dim < 2 or gap_events < 0:
        raise ValueError("invalid slots, value_dim or gap_events")
    if identifier_codebook.shape[0] != slots:
        raise ValueError("identifier codebook and slot count differ")

    initial_state = float(state_scale) * torch.randn(
        batch_size,
        slots,
        value_dim,
        generator=_generator(seed, f"{base_namespace}:initial-state"),
    )
    base_erase = torch.randint(
        slots,
        (batch_size, updates),
        generator=_generator(seed, f"{base_namespace}:erase-address"),
    )
    base_write = base_erase.clone()
    if family is StructuredTransferDemand.ADDRESS_DECOUPLING:
        offsets = torch.randint(
            1,
            slots,
            (batch_size, updates),
            generator=_generator(seed, f"{base_namespace}:write-offset"),
        )
        base_write = (base_erase + offsets) % slots
    base_new = _normalized_values(
        (batch_size, updates, value_dim),
        generator=_generator(seed, f"{base_namespace}:new-candidate"),
    )
    base_operation = torch.zeros(batch_size, updates, dtype=torch.long)
    if family is StructuredTransferDemand.MAGNITUDE:
        base_operation = torch.randint(
            4,
            (batch_size, updates),
            generator=_generator(seed, f"{base_namespace}:operation"),
        )
    base_channel_mask = torch.ones(batch_size, updates, value_dim)
    if family is StructuredTransferDemand.VALUE_GRANULARITY:
        base_channel_mask = _contiguous_masks(
            batch_size=batch_size,
            events=updates,
            value_dim=value_dim,
            width_generator=_generator(
                seed,
                f"{base_namespace}:mask-width",
            ),
            start_generator=_generator(
                seed,
                f"{base_namespace}:mask-start",
            ),
        )

    target = initial_state.clone()
    affected = torch.zeros(batch_size, slots, dtype=torch.bool)
    old_rows: list[torch.Tensor] = []
    erase_gate_rows: list[torch.Tensor] = []
    write_gate_rows: list[torch.Tensor] = []
    row = torch.arange(batch_size)
    for update_index in range(updates):
        old = target[row, base_erase[:, update_index]].clone()
        erase_gate, write_gate = _target_gates(
            family=family,
            old_candidate=old,
            operation=base_operation[:, update_index],
            channel_mask=base_channel_mask[:, update_index],
        )
        target = _apply_update(
            state=target,
            erase_address=base_erase[:, update_index],
            write_address=base_write[:, update_index],
            old_candidate=old,
            new_candidate=base_new[:, update_index],
            erase_gate=erase_gate,
            write_gate=write_gate,
        )
        affected[row, base_erase[:, update_index]] = True
        affected[row, base_write[:, update_index]] = True
        old_rows.append(old)
        erase_gate_rows.append(erase_gate)
        write_gate_rows.append(write_gate)
    base_old = torch.stack(old_rows, dim=1)
    base_erase_gates = torch.stack(erase_gate_rows, dim=1)
    base_write_gates = torch.stack(write_gate_rows, dim=1)

    total_events = updates + gap_events
    verified_positions = _verified_positions(updates, gap_events)
    update_mask = torch.zeros(batch_size, total_events, dtype=torch.bool)
    update_mask[:, verified_positions] = True
    erase_addresses = torch.zeros(batch_size, total_events, dtype=torch.long)
    write_addresses = torch.zeros(batch_size, total_events, dtype=torch.long)
    old_candidates = torch.zeros(batch_size, total_events, value_dim)
    new_candidates = torch.zeros(batch_size, total_events, value_dim)
    operations = torch.zeros(batch_size, total_events, dtype=torch.long)
    channel_masks = torch.ones(batch_size, total_events, value_dim)
    target_erase_gates = torch.zeros(batch_size, total_events, value_dim)
    target_write_gates = torch.zeros(batch_size, total_events, value_dim)

    erase_addresses[:, verified_positions] = base_erase
    write_addresses[:, verified_positions] = base_write
    old_candidates[:, verified_positions] = base_old
    new_candidates[:, verified_positions] = base_new
    operations[:, verified_positions] = base_operation
    channel_masks[:, verified_positions] = base_channel_mask
    target_erase_gates[:, verified_positions] = base_erase_gates
    target_write_gates[:, verified_positions] = base_write_gates

    distractor_positions = (~update_mask[0]).nonzero(as_tuple=False).flatten()
    if distractor_positions.numel():
        count = int(distractor_positions.numel())
        distractor_erase = torch.randint(
            slots,
            (batch_size, count),
            generator=_generator(seed, f"{distractor_namespace}:erase"),
        )
        distractor_write = torch.randint(
            slots,
            (batch_size, count),
            generator=_generator(seed, f"{distractor_namespace}:write"),
        )
        distractor_new = _normalized_values(
            (batch_size, count, value_dim),
            generator=_generator(seed, f"{distractor_namespace}:candidate"),
        )
        distractor_old = _normalized_values(
            (batch_size, count, value_dim),
            generator=_generator(seed, f"{distractor_namespace}:old-decoy"),
        )
        distractor_operation = torch.randint(
            4,
            (batch_size, count),
            generator=_generator(seed, f"{distractor_namespace}:operation"),
        )
        distractor_masks = _contiguous_masks(
            batch_size=batch_size,
            events=count,
            value_dim=value_dim,
            width_generator=_generator(
                seed,
                f"{distractor_namespace}:mask-width",
            ),
            start_generator=_generator(
                seed,
                f"{distractor_namespace}:mask-start",
            ),
        )
        erase_addresses[:, distractor_positions] = distractor_erase
        write_addresses[:, distractor_positions] = distractor_write
        old_candidates[:, distractor_positions] = distractor_old
        new_candidates[:, distractor_positions] = distractor_new
        operations[:, distractor_positions] = distractor_operation
        channel_masks[:, distractor_positions] = distractor_masks

    identifier_features = torch.cat(
        (
            identifier_codebook[erase_addresses],
            identifier_codebook[write_addresses],
        ),
        dim=-1,
    )
    family_features = _family_one_hot(
        family,
        batch_size=batch_size,
        events=total_events,
    )
    operation_one_hot = torch.nn.functional.one_hot(
        operations,
        num_classes=4,
    ).to(torch.float32)
    relation = _relation_features(erase_addresses, write_addresses)
    verified = update_mask.to(torch.float32).unsqueeze(-1)
    inputs = StructuredSequenceTransferInput(
        initial_state=initial_state.to(device),
        identifier_features=identifier_features.to(device),
        new_candidates=new_candidates.to(device),
        family_one_hot=family_features.to(device),
        operation_one_hot=operation_one_hot.to(device),
        channel_masks=channel_masks.to(device),
        relation_features=relation.to(device),
        verified_flags=verified.to(device),
    )
    return StructuredSequenceTransferBatch(
        inputs=inputs,
        update_mask=update_mask.to(device),
        erase_addresses=erase_addresses.to(device),
        write_addresses=write_addresses.to(device),
        old_candidates=old_candidates.to(device),
        target_erase_gates=target_erase_gates.to(device),
        target_write_gates=target_write_gates.to(device),
        target_state=target.to(device),
        affected_entities=affected.to(device),
        demand_family=family,
    )


def structured_base_transaction_digest(
    batch: StructuredSequenceTransferBatch,
) -> str:
    digest = hashlib.sha256()
    mask = batch.update_mask
    tensors = (
        batch.inputs.initial_state,
        batch.erase_addresses[mask],
        batch.write_addresses[mask],
        batch.old_candidates[mask],
        batch.inputs.new_candidates[mask],
        batch.inputs.family_one_hot[mask],
        batch.inputs.operation_one_hot[mask],
        batch.inputs.channel_masks[mask],
        batch.inputs.relation_features[mask],
        batch.target_erase_gates[mask],
        batch.target_write_gates[mask],
        batch.target_state,
        batch.affected_entities,
    )
    for value in tensors:
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(repr(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def concatenate_structured_event_features(
    inputs: StructuredSequenceTransferInput,
) -> torch.Tensor:
    return torch.cat(
        (
            inputs.identifier_features,
            inputs.new_candidates,
            inputs.family_one_hot,
            inputs.operation_one_hot,
            inputs.channel_masks,
            inputs.relation_features,
            inputs.verified_flags,
        ),
        dim=-1,
    )
