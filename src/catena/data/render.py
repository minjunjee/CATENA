from __future__ import annotations

import json
from typing import Iterable

from .schema import ClosureItem, Episode, HistorySegment, Transaction


def render_segments(segments: Iterable[HistorySegment]) -> str:
    return "\n".join(segment.text for segment in segments)


def render_plain_correction(tx: Transaction) -> str:
    return f"Correction: {tx.target} is now {tx.new_value}."


def render_typed_transaction(tx: Transaction) -> str:
    payload = {
        "operation": tx.operation,
        "target": tx.target,
        "old_value": tx.old_value,
        "new_value": tx.new_value,
        "old_version": tx.old_version,
        "new_version": tx.new_version,
        "valid_from": tx.valid_from,
        "scope": tx.scope,
    }
    return "[MEMORY TRANSACTION]\n" + json.dumps(payload, ensure_ascii=False, sort_keys=True)


def render_typed_closure(tx: Transaction, closure: list[ClosureItem]) -> str:
    closure_payload = [
        {"node_id": item.node_id, "relation": item.relation, "text": item.text}
        for item in closure
    ]
    return (
        render_typed_transaction(tx)
        + "\n[DEPENDENCY CLOSURE]\n"
        + json.dumps(closure_payload, ensure_ascii=False, sort_keys=True)
    )


def render_compact_snapshot(episode: Episode) -> str:
    payload = {
        "domain": episode.domain,
        "schema_family": episode.schema_family,
        "current_state": episode.current_state,
        "invalidates": episode.transaction.invalidates,
    }
    return "[CURRENT CANONICAL SNAPSHOT]\n" + json.dumps(payload, ensure_ascii=False, sort_keys=True)


def render_refresh_prompt(episode: Episode) -> str:
    return render_segments(episode.refresh_segments)


def render_history_prompt(episode: Episode) -> str:
    return render_segments(episode.history_segments)
