from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol

import torch


class StructuredSemanticRecord(Protocol):
    entity_description: str
    domain: str
    current_relation: str
    incoming_evidence: str
    prior_version: int
    evidence_version: int
    observation_day: int
    evidence_timestamp_day: int
    prior_valid_from_day: int
    prior_valid_to_day: int
    evidence_valid_from_day: int
    evidence_valid_to_day: int
    scope: str
    source: str
    provenance: str
    incoming_value_token: str
    template_surface: str


@dataclass(frozen=True, slots=True)
class SemanticFeatureConfigV61:
    categorical_fields: tuple[str, ...]
    numeric_fields: tuple[str, ...]
    categorical_bins_per_field: int
    version_scale: float
    day_scale: float
    state_read_dim: int

    def __post_init__(self) -> None:
        if not self.categorical_fields or len(set(self.categorical_fields)) != len(
            self.categorical_fields
        ):
            raise ValueError("categorical_fields must be nonempty and unique.")
        if not self.numeric_fields or len(set(self.numeric_fields)) != len(
            self.numeric_fields
        ):
            raise ValueError("numeric_fields must be nonempty and unique.")
        if set(self.categorical_fields) & set(self.numeric_fields):
            raise ValueError("categorical and numeric fields must be disjoint.")
        if self.categorical_bins_per_field <= 0:
            raise ValueError("categorical_bins_per_field must be positive.")
        if self.version_scale <= 0.0 or self.day_scale <= 0.0:
            raise ValueError("numeric scales must be positive.")
        if self.state_read_dim <= 0:
            raise ValueError("state_read_dim must be positive.")

    @property
    def semantic_dim(self) -> int:
        return (
            len(self.categorical_fields) * self.categorical_bins_per_field
            + len(self.numeric_fields)
        )

    @property
    def input_dim(self) -> int:
        return self.semantic_dim + self.state_read_dim


class FrozenSemanticFieldEncoderV61:
    """Deterministic field-aware encoder with no fitted main-set state."""

    _VERSION_FIELDS = frozenset({"prior_version", "evidence_version"})

    def __init__(self, config: SemanticFeatureConfigV61) -> None:
        self.config = config

    @staticmethod
    def _signed_bucket(field: str, value: str, bins: int) -> tuple[int, float]:
        digest = hashlib.sha256(f"{field}\0{value}".encode()).digest()
        bucket = int.from_bytes(digest[:8], "big") % bins
        sign = 1.0 if digest[8] & 1 else -1.0
        return bucket, sign

    def encode_semantics(self, record: StructuredSemanticRecord) -> torch.Tensor:
        result = torch.zeros(self.config.semantic_dim, dtype=torch.float32)
        offset = 0
        for field in self.config.categorical_fields:
            value = getattr(record, field)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field} must be a nonempty string.")
            bucket, sign = self._signed_bucket(
                field,
                value,
                self.config.categorical_bins_per_field,
            )
            result[offset + bucket] = sign
            offset += self.config.categorical_bins_per_field
        for field in self.config.numeric_fields:
            value = getattr(record, field)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field} must be an integer.")
            scale = (
                self.config.version_scale
                if field in self._VERSION_FIELDS
                else self.config.day_scale
            )
            result[offset] = float(value) / scale
            offset += 1
        if offset != self.config.semantic_dim:
            raise AssertionError("semantic feature layout is inconsistent.")
        if not bool(torch.isfinite(result).all().item()):
            raise FloatingPointError("semantic field encoding is non-finite.")
        return result

    def encode(
        self,
        record: StructuredSemanticRecord,
        state_read: torch.Tensor,
        *,
        mask_semantics: bool = False,
        mask_state_read: bool = False,
    ) -> torch.Tensor:
        state_read = state_read.detach().to(dtype=torch.float32, device="cpu").reshape(-1)
        if state_read.numel() != self.config.state_read_dim:
            raise ValueError(
                f"state_read must have {self.config.state_read_dim} values."
            )
        if not bool(torch.isfinite(state_read).all().item()):
            raise FloatingPointError("state_read is non-finite.")
        semantic = (
            torch.zeros(self.config.semantic_dim, dtype=torch.float32)
            if mask_semantics
            else self.encode_semantics(record)
        )
        visible_state_read = (
            torch.zeros_like(state_read) if mask_state_read else state_read
        )
        return torch.cat([semantic, visible_state_read])
