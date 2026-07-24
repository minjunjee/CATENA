from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from catena.methods.encoder_inputs import EncoderText


@dataclass
class PreparedEncoderInput:
    embeddings: Any
    attention_mask: Any | None
    field_type_ids: Any | None


def _field_ids_from_offsets(
    offsets: list[tuple[int, int]],
    spans: list[tuple[int, int, int]],
) -> list[int]:
    result: list[int] = []
    for token_start, token_end in offsets:
        if token_start == token_end:
            result.append(0)
            continue
        field_id = 0
        for span_start, span_end, candidate_id in spans:
            if token_start < span_end and token_end > span_start:
                field_id = candidate_id
                break
        result.append(field_id)
    return result


def prepare_encoder_input(model, rendered: EncoderText) -> PreparedEncoderInput:
    """Tokenize transaction text and materialize frozen backbone embeddings.

    Fast tokenizers yield character offsets, allowing the typed encoder to receive
    token-level field IDs.  Remote-code tokenizers that do not expose offsets fall
    back to lexical type tags in the input text, so the experiment remains runnable.
    """

    import torch

    tokenizer_kwargs = {
        "add_special_tokens": False,
        "return_tensors": "pt",
    }
    encoding = None
    if rendered.field_spans:
        try:
            encoding = model.tokenizer(
                rendered.text,
                return_offsets_mapping=True,
                **tokenizer_kwargs,
            )
        except (TypeError, NotImplementedError, ValueError):
            encoding = None
    if encoding is None:
        encoding = model.tokenizer(rendered.text, **tokenizer_kwargs)

    input_ids = encoding.input_ids.to(model.device)
    attention_mask = getattr(encoding, "attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)
    with torch.no_grad():
        embeddings = model.get_input_embeddings()(input_ids)

    field_type_ids = None
    offsets = getattr(encoding, "offset_mapping", None)
    if offsets is not None and rendered.field_spans:
        raw_offsets = [(int(a), int(b)) for a, b in offsets[0].tolist()]
        ids = _field_ids_from_offsets(raw_offsets, rendered.field_spans)
        field_type_ids = torch.tensor([ids], dtype=torch.long, device=model.device)

    return PreparedEncoderInput(
        embeddings=embeddings,
        attention_mask=attention_mask,
        field_type_ids=field_type_ids,
    )
