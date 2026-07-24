from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable

from catena.data.schema import ChainEpisode, ClosureItem, Episode, Transaction

FIELD_TYPE_IDS = {
    "operation": 1,
    "target": 2,
    "old_value": 3,
    "new_value": 4,
    "old_version": 5,
    "new_version": 6,
    "valid_from": 7,
    "scope": 8,
    "closure_relation": 9,
    "closure_text": 10,
    "transaction_boundary": 11,
}

EncoderMode = str


@dataclass(frozen=True)
class EncoderText:
    """Text and character-level field spans supplied to the transaction encoder.

    ``field_spans`` contains ``(start_char, end_char, field_type_id)`` tuples.  The
    trainer maps these spans to tokenizer offsets when a fast tokenizer supports
    ``return_offsets_mapping``.  The explicit spans let the typed encoder use the
    same lexical content as an untyped encoder while receiving field identities as
    an auxiliary signal.
    """

    text: str
    field_spans: list[tuple[int, int, int]]


def _structured_text(
    tx: Transaction,
    closure: Iterable[ClosureItem],
    *,
    typed_tags: bool,
    include_closure: bool,
    transaction_index: int | None = None,
) -> EncoderText:
    parts: list[str] = []
    spans: list[tuple[int, int, int]] = []

    def add(field: str, value: Any) -> None:
        prefix = f"<{field}> " if typed_tags else ""
        body = json.dumps(value, ensure_ascii=False, sort_keys=True)
        text = prefix + body + "\n"
        start = sum(len(p) for p in parts)
        parts.append(text)
        spans.append((start, start + len(text), FIELD_TYPE_IDS.get(field, 0)))

    if transaction_index is not None:
        add("transaction_boundary", transaction_index)
    add("operation", tx.operation)
    add("target", tx.target)
    add("old_value", tx.old_value)
    add("new_value", tx.new_value)
    add("old_version", tx.old_version)
    add("new_version", tx.new_version)
    add("valid_from", tx.valid_from)
    add("scope", tx.scope)
    if include_closure:
        for item in closure:
            add("closure_relation", item.relation)
            add("closure_text", item.text)
    return EncoderText("".join(parts), spans)


def _generic_text(
    tx: Transaction,
    closure: Iterable[ClosureItem],
    *,
    include_closure: bool,
    transaction_index: int | None = None,
) -> EncoderText:
    """Natural-language control with the same underlying information.

    This is the parameter-matched generic soft-slot baseline.  It does not receive
    field-type IDs or machine-readable tags, but it does receive the transaction
    values and, when enabled, the same dependency-closure statements.
    """

    prefix = "" if transaction_index is None else f"Update number {transaction_index}. "
    lines = [
        (
            f"{prefix}The verified memory update uses operation {tx.operation}. "
            f"It changes {tx.target} from {tx.old_value} to {tx.new_value}, "
            f"moving version {tx.old_version} to version {tx.new_version}. "
            f"It is valid from {tx.valid_from}. Scope: {json.dumps(tx.scope, sort_keys=True)}."
        )
    ]
    if include_closure:
        lines.append("Consequences of this update:")
        lines.extend(f"- {item.text}" for item in closure)
    return EncoderText("\n".join(lines) + "\n", [])


def render_transaction_encoder_text(
    tx: Transaction,
    closure: Iterable[ClosureItem],
    *,
    mode: EncoderMode = "typed_transaction",
    include_closure: bool = True,
    transaction_index: int | None = None,
) -> EncoderText:
    if mode == "typed_transaction":
        return _structured_text(
            tx,
            closure,
            typed_tags=True,
            include_closure=include_closure,
            transaction_index=transaction_index,
        )
    if mode == "untyped_transaction":
        return _structured_text(
            tx,
            closure,
            typed_tags=False,
            include_closure=include_closure,
            transaction_index=transaction_index,
        )
    if mode == "generic_soft_slot":
        return _generic_text(
            tx,
            closure,
            include_closure=include_closure,
            transaction_index=transaction_index,
        )
    raise ValueError(f"Unsupported encoder mode: {mode}")


def render_encoder_text(
    episode: Episode,
    *,
    mode: EncoderMode = "typed_transaction",
    include_closure: bool = True,
) -> EncoderText:
    return render_transaction_encoder_text(
        episode.transaction,
        episode.closure,
        mode=mode,
        include_closure=include_closure,
    )


def render_chain_encoder_text(
    episode: ChainEpisode,
    *,
    mode: EncoderMode = "typed_transaction",
    include_closure: bool = True,
) -> EncoderText:
    text_parts: list[str] = []
    spans: list[tuple[int, int, int]] = []
    offset = 0
    for index, (tx, closure) in enumerate(zip(episode.transactions, episode.closures), start=1):
        rendered = render_transaction_encoder_text(
            tx,
            closure,
            mode=mode,
            include_closure=include_closure,
            transaction_index=index,
        )
        text_parts.append(rendered.text)
        spans.extend((start + offset, end + offset, field_id) for start, end, field_id in rendered.field_spans)
        offset += len(rendered.text)
    return EncoderText("".join(text_parts), spans)
