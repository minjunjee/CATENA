from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from catena.lm.e26_final_official_adapter import (
    E26FinalOfficialAdapterError,
    PureRecurrentOfficialGptCacheAdapter,
    cache_tensor_equality_and_no_alias,
)


class _TinyCache:
    def __init__(self) -> None:
        self.states: list[dict[str, torch.Tensor]] = []
        self.lengths: list[int] = []

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.states[index]

    def get_seq_length(self, index: int) -> int:
        return self.lengths[index]

    def update(self, *, layer_idx: int, state: torch.Tensor, offset: int) -> None:
        if len(self.states) <= layer_idx:
            self.states.append({"recurrent_state": state})
            self.lengths.append(offset)
        else:
            self.states[layer_idx]["recurrent_state"] = state
            self.lengths[layer_idx] += offset


class _TinyAttention(torch.nn.Module):
    def __init__(self, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.projection = torch.nn.Linear(4, 4, bias=False)
        torch.nn.init.eye_(self.projection.weight)

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        attention_mask: None,
        past_key_values: _TinyCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, None, _TinyCache | None]:
        del attention_mask
        previous = torch.zeros_like(hidden[:, :1])
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            previous = past_key_values[self.layer_idx]["recurrent_state"]
        output = self.projection(hidden) + previous
        if use_cache:
            if past_key_values is None:
                raise AssertionError("test cache missing")
            final = previous + hidden.sum(dim=1, keepdim=True)
            past_key_values.update(
                layer_idx=self.layer_idx,
                state=final,
                offset=hidden.shape[1],
            )
        return output, None, past_key_values


class _TinyBlock(torch.nn.Module):
    def __init__(self, layer_idx: int, *, use_gdn2: bool = True) -> None:
        super().__init__()
        self.use_gdn2 = use_gdn2
        self.norm_1 = torch.nn.Identity()
        self.norm_2 = torch.nn.Identity()
        self.attn = _TinyAttention(layer_idx)
        self.mlp = torch.nn.Linear(4, 4, bias=False)
        torch.nn.init.zeros_(self.mlp.weight)
        self.config = SimpleNamespace(
            parallel_residual=False,
            shared_attention_norm=False,
            mlp=True,
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        normalized = self.norm_1(hidden)
        mixed, _, _ = self.attn(normalized, attention_mask=None)
        hidden = hidden + mixed
        return cast(torch.Tensor, hidden + self.mlp(self.norm_2(hidden)))


class _TinyTransformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.wte = torch.nn.Embedding(11, 4)
        self.h = torch.nn.ModuleList([_TinyBlock(0), _TinyBlock(1)])
        self.ln_f = torch.nn.Identity()


class _TinyGpt(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(17)
        self.transformer = _TinyTransformer()
        self.lm_head = torch.nn.Linear(4, 11, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.transformer.wte(input_ids)
        for block in self.transformer.h:
            hidden = block(hidden)
        return cast(torch.Tensor, self.lm_head(self.transformer.ln_f(hidden)))


def test_adapter_no_cache_is_function_identical_to_official_block_loop() -> None:
    model = _TinyGpt()
    adapter = PureRecurrentOfficialGptCacheAdapter(model, cache_factory=_TinyCache)
    tokens = torch.tensor([[1, 2, 3, 4]])
    expected = model(tokens)
    observed, cache = adapter(tokens, use_cache=False)
    torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    assert cache is None


def test_adapter_carries_and_deep_clones_full_gpt_cache_without_alias() -> None:
    model = _TinyGpt()
    adapter = PureRecurrentOfficialGptCacheAdapter(model, cache_factory=_TinyCache)
    prefix = torch.tensor([[1, 2, 3]])
    query = torch.tensor([[4, 5]])
    _, cache = adapter(prefix, use_cache=True)
    assert isinstance(cache, _TinyCache)
    assert len(cache) == 2
    assert [cache.get_seq_length(index) for index in range(2)] == [3, 3]

    clone = adapter.clone_cache(cache)
    assert isinstance(clone, _TinyCache)
    for layer_index in range(2):
        comparison = cache_tensor_equality_and_no_alias(
            cache[layer_index], clone[layer_index]
        )
        assert comparison["passed"] is True

    first, cache = adapter(query, past_key_values=cache, use_cache=True)
    second, clone = adapter(query, past_key_values=clone, use_cache=True)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert [cache.get_seq_length(index) for index in range(2)] == [5, 5]
    assert [clone.get_seq_length(index) for index in range(2)] == [5, 5]
    for layer_index in range(2):
        assert cache_tensor_equality_and_no_alias(
            cache[layer_index], clone[layer_index]
        )["passed"] is True


def test_adapter_rejects_non_recurrent_or_unindexed_model() -> None:
    mixed = _TinyGpt()
    mixed_block = cast(Any, mixed.transformer.h[1])
    mixed_block.use_gdn2 = False
    with pytest.raises(E26FinalOfficialAdapterError, match="pure-recurrent"):
        PureRecurrentOfficialGptCacheAdapter(mixed, cache_factory=_TinyCache)

    unindexed = _TinyGpt()
    unindexed_block = cast(Any, unindexed.transformer.h[1])
    unindexed_block.attn.layer_idx = None
    with pytest.raises(E26FinalOfficialAdapterError, match="explicit unique layer_idx"):
        PureRecurrentOfficialGptCacheAdapter(unindexed, cache_factory=_TinyCache)


def test_adapter_rejects_cache_without_use_cache() -> None:
    model = _TinyGpt()
    adapter = PureRecurrentOfficialGptCacheAdapter(model, cache_factory=_TinyCache)
    with pytest.raises(E26FinalOfficialAdapterError, match="use_cache=True"):
        adapter(torch.tensor([[1]]), past_key_values=_TinyCache(), use_cache=False)
