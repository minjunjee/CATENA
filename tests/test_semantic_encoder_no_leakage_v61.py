from dataclasses import dataclass

import torch

from catena.models.semantic_encoder_v61 import (
    FrozenSemanticFieldEncoderV61,
    SemanticFeatureConfigV61,
)


@dataclass(frozen=True)
class Record:
    entity_description: str = "subject 17"
    domain: str = "api"
    current_relation: str = "setting record"
    incoming_evidence: str = "filing 4"
    prior_version: int = 3
    evidence_version: int = 4
    observation_day: int = 80
    evidence_timestamp_day: int = 70
    prior_valid_from_day: int = 20
    prior_valid_to_day: int = 79
    evidence_valid_from_day: int = 78
    evidence_valid_to_day: int = 120
    scope: str = "tenant"
    source: str = "registry"
    provenance: str = "signed filing"
    incoming_value_token: str = "item 91"
    template_surface: str = "record"


def _encoder() -> FrozenSemanticFieldEncoderV61:
    return FrozenSemanticFieldEncoderV61(
        SemanticFeatureConfigV61(
            categorical_fields=(
                "entity_description",
                "domain",
                "current_relation",
                "incoming_evidence",
                "scope",
                "source",
                "provenance",
                "incoming_value_token",
                "template_surface",
            ),
            numeric_fields=(
                "prior_version",
                "evidence_version",
                "observation_day",
                "evidence_timestamp_day",
                "prior_valid_from_day",
                "prior_valid_to_day",
                "evidence_valid_from_day",
                "evidence_valid_to_day",
            ),
            categorical_bins_per_field=8,
            version_scale=16.0,
            day_scale=256.0,
            state_read_dim=32,
        )
    )


def test_frozen_field_encoder_shape_and_independent_masks():
    encoder = _encoder()
    state_read = torch.arange(32, dtype=torch.float32) / 32.0
    full = encoder.encode(Record(), state_read)
    no_semantics = encoder.encode(Record(), state_read, mask_semantics=True)
    no_state = encoder.encode(Record(), state_read, mask_state_read=True)
    assert full.shape == (112,)
    assert torch.equal(no_semantics[:80], torch.zeros(80))
    assert torch.equal(no_semantics[80:], state_read)
    assert torch.equal(no_state[:80], full[:80])
    assert torch.equal(no_state[80:], torch.zeros(32))


def test_encoder_has_no_operation_or_oracle_demand_argument():
    fields = set(SemanticFeatureConfigV61.__dataclass_fields__)
    assert "operation" not in fields
    assert "erase" not in fields
    assert "write" not in fields
    assert "target" not in fields
    assert "mask" not in fields
