from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum

import torch


class LocalizationCandidateCondition(StrEnum):
    A_ORACLE_ADDRESS_ORACLE_CANDIDATE = "A_oracle_address_oracle_candidate"
    B_LEARNED_ADDRESS_ORACLE_CANDIDATE = "B_learned_address_oracle_candidate"
    C_ORACLE_ADDRESS_STATE_READ_CANDIDATE = "C_oracle_address_state_read_candidate"
    D_LEARNED_ADDRESS_STATE_READ_CANDIDATE = "D_learned_address_state_read_candidate"

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


@dataclass(slots=True)
class LocalizationCandidateBatch:
    state: torch.Tensor
    descriptor: torch.Tensor
    erase_address: torch.Tensor
    write_address: torch.Tensor
    old_candidate: torch.Tensor
    new_candidate: torch.Tensor
    target: torch.Tensor


def make_address_codebook(
    *,
    slots: int,
    code_dim: int,
    seed: int,
) -> torch.Tensor:
    if slots < 2:
        raise ValueError("slots must be at least 2")
    if code_dim < 2:
        raise ValueError("code_dim must be at least 2")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    codebook = torch.randn(slots, code_dim, generator=generator)
    return codebook / codebook.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def address_codebook_sha256(codebook: torch.Tensor) -> str:
    value = codebook.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(repr(tuple(value.shape)).encode("utf-8"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def generate_localization_candidate_batch(
    *,
    batch_size: int,
    slots: int,
    value_dim: int,
    state_scale: float,
    address_codebook: torch.Tensor,
    generator: torch.Generator,
    device: torch.device,
) -> LocalizationCandidateBatch:
    if address_codebook.shape[0] != slots:
        raise ValueError("address codebook and slot count differ")
    if batch_size <= 0 or value_dim <= 0:
        raise ValueError("batch_size and value_dim must be positive")

    state = float(state_scale) * torch.randn(
        batch_size,
        slots,
        value_dim,
        generator=generator,
    )
    erase_address = torch.randint(slots, (batch_size,), generator=generator)
    offset = torch.randint(1, slots, (batch_size,), generator=generator)
    write_address = (erase_address + offset) % slots
    if torch.any(erase_address == write_address):
        raise AssertionError("erase and write addresses must differ")

    batch_index = torch.arange(batch_size)
    old_candidate = state[batch_index, erase_address].clone()
    new_candidate = torch.randn(batch_size, value_dim, generator=generator)
    new_candidate = new_candidate / new_candidate.norm(
        dim=-1,
        keepdim=True,
    ).clamp_min(1e-8)

    descriptor = torch.cat(
        (
            address_codebook[erase_address],
            address_codebook[write_address],
            new_candidate,
        ),
        dim=-1,
    )
    target = state.clone()
    target[batch_index, erase_address] -= old_candidate
    target[batch_index, write_address] += new_candidate
    return LocalizationCandidateBatch(
        state=state.to(device),
        descriptor=descriptor.to(device),
        erase_address=erase_address.to(device),
        write_address=write_address.to(device),
        old_candidate=old_candidate.to(device),
        new_candidate=new_candidate.to(device),
        target=target.to(device),
    )
