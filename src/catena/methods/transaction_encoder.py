from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EncoderSpec:
    input_dim: int
    hidden_dim: int = 512
    num_slots: int = 8
    num_layers: int = 2
    num_heads: int = 8
    dropout: float = 0.1
    num_field_types: int = 16


def build_encoder(spec: EncoderSpec):
    import torch
    from torch import nn

    class TransactionSlotEncoder(nn.Module):
        """Compress token embeddings for a memory transaction into K soft slots.

        The backbone embedding matrix remains frozen. This module learns a small
        bottleneck representation, optional field-type embeddings, and K learned
        queries that cross-attend to the transaction sequence.
        """

        def __init__(self) -> None:
            super().__init__()
            self.spec = spec
            self.input_proj = nn.Linear(spec.input_dim, spec.hidden_dim, bias=False)
            self.field_type_embedding = nn.Embedding(spec.num_field_types, spec.hidden_dim)
            layer = nn.TransformerEncoderLayer(
                d_model=spec.hidden_dim,
                nhead=spec.num_heads,
                dim_feedforward=spec.hidden_dim * 4,
                dropout=spec.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.sequence_encoder = nn.TransformerEncoder(layer, num_layers=spec.num_layers)
            self.slot_queries = nn.Parameter(torch.empty(spec.num_slots, spec.hidden_dim))
            nn.init.normal_(self.slot_queries, mean=0.0, std=0.02)
            self.cross_attention = nn.MultiheadAttention(
                spec.hidden_dim,
                spec.num_heads,
                dropout=spec.dropout,
                batch_first=True,
            )
            self.output_norm = nn.LayerNorm(spec.hidden_dim)
            self.output_proj = nn.Linear(spec.hidden_dim, spec.input_dim, bias=False)

        def forward(
            self,
            token_embeddings,
            attention_mask=None,
            field_type_ids=None,
        ):
            x = self.input_proj(token_embeddings)
            if field_type_ids is not None:
                x = x + self.field_type_embedding(field_type_ids)
            key_padding_mask = None
            if attention_mask is not None:
                key_padding_mask = ~attention_mask.bool()
            x = self.sequence_encoder(x, src_key_padding_mask=key_padding_mask)
            queries = self.slot_queries.unsqueeze(0).expand(x.shape[0], -1, -1)
            slots, _ = self.cross_attention(
                queries,
                x,
                x,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            slots = self.output_proj(self.output_norm(slots))
            return slots

    return TransactionSlotEncoder()
