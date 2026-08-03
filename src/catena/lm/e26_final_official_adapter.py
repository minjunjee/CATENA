"""Additive cache adapter for the pinned pure-recurrent official GDN-2 GPT.

The upstream GPT wrapper does not pass an FLA cache through its GDN-2 blocks.
This adapter leaves upstream source untouched and mirrors the exact official
embedding, block residual/MLP, final-normalization, and LM-head order while
adding only explicit ``past_key_values``/``use_cache`` plumbing to every
GDN-2 layer.  It supports the pinned pure-recurrent model only and fails
closed for mixed attention architectures.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from copy import deepcopy
from typing import Any, cast

import torch


class E26FinalOfficialAdapterError(RuntimeError):
    """Raised when the pinned pure-recurrent adapter contract is violated."""


def walk_cache_tensors(
    value: object,
    prefix: str = "root",
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield a stable path for every tensor in a nested FLA cache state."""

    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, Mapping):
        for key in sorted(value, key=str):
            yield from walk_cache_tensors(value[key], f"{prefix}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from walk_cache_tensors(item, f"{prefix}[{index}]")


def cache_tensor_equality_and_no_alias(left: object, right: object) -> dict[str, Any]:
    """Compare cache tensor trees exactly and require disjoint storage."""

    left_rows = list(walk_cache_tensors(left))
    right_rows = list(walk_cache_tensors(right))
    paths_match = [path for path, _ in left_rows] == [path for path, _ in right_rows]
    shapes_match = paths_match and all(
        first.shape == second.shape
        for (_, first), (_, second) in zip(left_rows, right_rows, strict=True)
    )
    values_equal = shapes_match and all(
        torch.equal(first, second)
        for (_, first), (_, second) in zip(left_rows, right_rows, strict=True)
    )
    no_alias = shapes_match and bool(left_rows) and all(
        first.data_ptr() != second.data_ptr()
        for (_, first), (_, second) in zip(left_rows, right_rows, strict=True)
    )
    return {
        "tensor_count": len(left_rows),
        "paths_match": paths_match,
        "shapes_match": shapes_match,
        "values_equal": values_equal,
        "no_alias": no_alias,
        "passed": bool(left_rows) and paths_match and shapes_match and values_equal and no_alias,
    }


class PureRecurrentOfficialGptCacheAdapter(torch.nn.Module):
    """Cache-aware wrapper for the pinned all-GDN-2 official GPT only."""

    def __init__(
        self,
        official_gpt: torch.nn.Module,
        *,
        cache_factory: Callable[[], Any],
    ) -> None:
        super().__init__()
        self.official_gpt = official_gpt
        self._cache_factory = cache_factory
        transformer = getattr(official_gpt, "transformer", None)
        blocks = getattr(transformer, "h", None)
        if not isinstance(blocks, torch.nn.ModuleList) or not blocks:
            raise E26FinalOfficialAdapterError("Official GPT lacks transformer.h blocks")
        if not all(getattr(block, "use_gdn2", False) is True for block in blocks):
            raise E26FinalOfficialAdapterError(
                "Cache adapter is restricted to the pinned pure-recurrent GDN-2 GPT"
            )
        indices = [getattr(getattr(block, "attn", None), "layer_idx", None) for block in blocks]
        if indices != list(range(len(blocks))):
            raise E26FinalOfficialAdapterError(
                "Every official GDN-2 layer must have an explicit unique layer_idx"
            )

    @property
    def layer_count(self) -> int:
        official = cast(Any, self.official_gpt)
        blocks = cast(torch.nn.ModuleList, official.transformer.h)
        return len(blocks)

    def new_cache(self) -> Any:
        cache = self._cache_factory()
        if not hasattr(cache, "update") or not hasattr(cache, "__len__"):
            raise E26FinalOfficialAdapterError("FLA cache factory returned an invalid object")
        return cache

    def clone_cache(self, cache: Any) -> Any:
        """Deep-clone cache tensors and metadata, then verify exact no-alias equality."""

        tensor_memo: dict[int, torch.Tensor] = {}
        for layer_index in range(len(cache)):
            for _, tensor in walk_cache_tensors(cache[layer_index]):
                tensor_memo[id(tensor)] = tensor.detach().clone()
        if not tensor_memo:
            raise E26FinalOfficialAdapterError("Cannot clone an empty tensor cache")
        cloned = deepcopy(cache, cast(dict[int, Any], tensor_memo))
        if len(cloned) != len(cache):
            raise E26FinalOfficialAdapterError("Cache clone changed layer count")
        for layer_index in range(len(cache)):
            comparison = cache_tensor_equality_and_no_alias(
                cache[layer_index], cloned[layer_index]
            )
            if comparison["passed"] is not True:
                raise E26FinalOfficialAdapterError(
                    f"Cache clone equality/no-alias failed at layer {layer_index}"
                )
            if hasattr(cache, "get_seq_length") and (
                int(cache.get_seq_length(layer_index))
                != int(cloned.get_seq_length(layer_index))
            ):
                raise E26FinalOfficialAdapterError(
                    f"Cache clone changed sequence metadata at layer {layer_index}"
                )
        return cloned

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        past_key_values: Any | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Any | None]:
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise E26FinalOfficialAdapterError(
                "input_ids must be a rank-2 torch.long tensor"
            )
        if past_key_values is not None and not use_cache:
            raise E26FinalOfficialAdapterError(
                "A cache may only be supplied when use_cache=True"
            )
        cache = past_key_values
        if use_cache and cache is None:
            cache = self.new_cache()

        official = cast(Any, self.official_gpt)
        transformer = cast(Any, official.transformer)
        hidden = transformer.wte(input_ids)
        for layer_index, block in enumerate(transformer.h):
            attention = block.attn
            if getattr(attention, "layer_idx", None) != layer_index:
                raise E26FinalOfficialAdapterError(
                    f"GDN-2 layer_idx drift at block {layer_index}"
                )
            normalized = block.norm_1(hidden)
            mixed, attention_map, returned_cache = attention(
                normalized,
                attention_mask=None,
                past_key_values=cache,
                use_cache=use_cache,
            )
            if attention_map is not None:
                raise E26FinalOfficialAdapterError("GDN-2 unexpectedly returned attention")
            if use_cache and returned_cache is not cache:
                raise E26FinalOfficialAdapterError("GDN-2 replaced the shared FLA cache")
            if block.config.parallel_residual:
                if block.config.shared_attention_norm is not True:
                    raise E26FinalOfficialAdapterError(
                        "Official parallel residual contract changed"
                    )
                if block.config.mlp:
                    mixed = mixed + block.mlp(normalized)
                hidden = hidden + mixed
            else:
                hidden = hidden + mixed
                if block.config.mlp:
                    normalized_second = block.norm_2(hidden)
                    hidden = hidden + block.mlp(normalized_second)
        logits = cast(torch.Tensor, official.lm_head(transformer.ln_f(hidden)))
        return logits, cache
