from __future__ import annotations

from catena.data.generator import generate_episode
from catena.methods.encoder_inputs import render_encoder_text


def _episode():
    return generate_episode(
        split="test",
        index=0,
        seed=13,
        history_token_target=256,
        domain="access",
        operation="AMEND",
        dependency_depth=2,
        query_gap_tokens=0,
        schema_family="finance-console",
    )


def test_encoder_modes_preserve_values_but_change_structure():
    episode = _episode()
    typed = render_encoder_text(episode, mode="typed_transaction")
    untyped = render_encoder_text(episode, mode="untyped_transaction")
    generic = render_encoder_text(episode, mode="generic_soft_slot")
    assert str(episode.transaction.new_value) in typed.text
    assert str(episode.transaction.new_value) in untyped.text
    assert str(episode.transaction.new_value) in generic.text
    assert typed.field_spans
    assert untyped.field_spans
    assert generic.field_spans == []
    assert "<operation>" in typed.text
    assert "<operation>" not in untyped.text
