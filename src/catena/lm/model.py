from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, cast

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig
from .hashing import named_parameter_signature, state_dict_digest
from .interventions import AddressIntervention, GateIntervention
from .recurrent_mixer import GateTrace, MixerState, TransactionalDeltaMixer


class RMSNorm(nn.Module):
    def __init__(self, dimension: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.eps = float(eps)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(value.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (value.float() * scale).to(value.dtype) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hidden = config.ffn_hidden_dim
        self.in_proj = nn.Linear(config.d_model, 2 * hidden, bias=False)
        self.out_proj = nn.Linear(hidden, config.d_model, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        gate, content = self.in_proj(value).chunk(2, dim=-1)
        return cast(torch.Tensor, self.out_proj(F.silu(gate) * content))


@dataclass
class LocalAttentionState:
    """Fixed-capacity K/V ring and absolute-position metadata for one layer."""

    key: torch.Tensor
    value: torch.Tensor
    positions: torch.Tensor
    length: int
    write_index: int

    def clone(self, *, detach: bool = False) -> LocalAttentionState:
        key = self.key.detach().clone() if detach else self.key.clone()
        value = self.value.detach().clone() if detach else self.value.clone()
        return LocalAttentionState(
            key=key,
            value=value,
            positions=self.positions.clone(),
            length=self.length,
            write_index=self.write_index,
        )

    def to(self, *args: Any, **kwargs: Any) -> LocalAttentionState:
        # Positions remain integral even when a dtype conversion is requested
        # for the floating-point cache tensors.
        key = self.key.to(*args, **kwargs)
        value = self.value.to(*args, **kwargs)
        return LocalAttentionState(
            key=key,
            value=value,
            positions=self.positions.to(device=key.device),
            length=self.length,
            write_index=self.write_index,
        )

    def storage_ptrs(self) -> list[int]:
        return [
            self.key.untyped_storage().data_ptr(),
            self.value.untyped_storage().data_ptr(),
            self.positions.untyped_storage().data_ptr(),
        ]


class LocalCausalSelfAttention(nn.Module):
    """SDPA reference with a cloneable, fixed-capacity local K/V ring.

    The cache is deliberately part of the correctness reference rather than a
    performance optimization.  It makes prefix-prefill plus incremental
    continuation semantically identical to a single full-sequence call, which
    is required when several query branches share one transactional prefix.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def _mask(self, sequence: int, device: torch.device) -> torch.Tensor:
        row = torch.arange(sequence, device=device)[:, None]
        col = torch.arange(sequence, device=device)[None, :]
        allowed = (col <= row) & (col > row - self.config.local_attention_window)
        mask = torch.full((sequence, sequence), float("-inf"), device=device)
        return mask.masked_fill(allowed, 0.0)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> LocalAttentionState:
        shape = (
            batch_size,
            self.config.n_heads,
            self.config.local_attention_window,
            self.config.head_dim,
        )
        return LocalAttentionState(
            key=torch.zeros(shape, device=device, dtype=dtype),
            value=torch.zeros(shape, device=device, dtype=dtype),
            positions=torch.full(
                (self.config.local_attention_window,),
                -1,
                device=device,
                dtype=torch.long,
            ),
            length=0,
            write_index=0,
        )

    def _validate_state(
        self,
        state: LocalAttentionState,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        position_offset: int,
    ) -> None:
        expected_shape = (
            batch_size,
            self.config.n_heads,
            self.config.local_attention_window,
            self.config.head_dim,
        )
        if state.key.shape != expected_shape or state.value.shape != expected_shape:
            raise ValueError(
                "Unexpected local-attention cache shape: "
                f"key={tuple(state.key.shape)}, value={tuple(state.value.shape)}, "
                f"expected={expected_shape}"
            )
        if state.positions.shape != (self.config.local_attention_window,):
            raise ValueError(
                "Unexpected local-attention position metadata shape: "
                f"{tuple(state.positions.shape)}"
            )
        if state.key.device != device or state.value.device != device:
            raise ValueError("Local-attention cache and hidden states must share a device")
        if state.positions.device != device:
            raise ValueError("Local-attention positions and hidden states must share a device")
        if state.key.dtype != dtype or state.value.dtype != dtype:
            raise ValueError("Local-attention cache and hidden states must share a dtype")
        if not 0 <= state.length <= self.config.local_attention_window:
            raise ValueError("Local-attention cache length is out of range")
        if not 0 <= state.write_index < self.config.local_attention_window:
            raise ValueError("Local-attention write index is out of range")
        if state.write_index != position_offset % self.config.local_attention_window:
            raise ValueError("Local-attention write index is inconsistent with runtime position")

        valid_positions = torch.sort(state.positions[state.positions >= 0]).values
        if valid_positions.numel() != state.length:
            raise ValueError("Local-attention cache length and position metadata disagree")
        expected_start = max(0, position_offset - self.config.local_attention_window)
        expected_positions = torch.arange(
            expected_start,
            position_offset,
            device=device,
            dtype=torch.long,
        )
        if not torch.equal(valid_positions, expected_positions):
            raise ValueError(
                "Local-attention position metadata is not a contiguous suffix "
                "of the runtime history"
            )

    def _chronological_cache(
        self,
        state: LocalAttentionState,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        valid_slots = torch.nonzero(state.positions >= 0, as_tuple=False).flatten()
        if valid_slots.numel() == 0:
            return (
                state.key[:, :, :0],
                state.value[:, :, :0],
                state.positions[:0],
            )
        order = torch.argsort(state.positions.index_select(0, valid_slots))
        chronological_slots = valid_slots.index_select(0, order)
        return (
            state.key.index_select(2, chronological_slots),
            state.value.index_select(2, chronological_slots),
            state.positions.index_select(0, chronological_slots),
        )

    def _updated_ring(
        self,
        *,
        cached_key: torch.Tensor,
        cached_value: torch.Tensor,
        cached_positions: torch.Tensor,
        current_key: torch.Tensor,
        current_value: torch.Tensor,
        current_positions: torch.Tensor,
        next_position: int,
    ) -> LocalAttentionState:
        capacity = self.config.local_attention_window
        combined_key = torch.cat((cached_key, current_key), dim=2)
        combined_value = torch.cat((cached_value, current_value), dim=2)
        combined_positions = torch.cat((cached_positions, current_positions), dim=0)

        keep = min(capacity, combined_positions.numel())
        recent_key = combined_key[:, :, -keep:]
        recent_value = combined_value[:, :, -keep:]
        recent_positions = combined_positions[-keep:]
        slots = torch.remainder(recent_positions, capacity)

        # Functional index_copy keeps input RuntimeState immutable and retains
        # autograd connections for callers that carry state across train chunks.
        cache_shape = (
            cached_key.shape[0],
            cached_key.shape[1],
            capacity,
            cached_key.shape[3],
        )
        key = cached_key.new_zeros(cache_shape).index_copy(2, slots, recent_key)
        value = cached_value.new_zeros(cache_shape).index_copy(2, slots, recent_value)
        positions = torch.full(
            (capacity,),
            -1,
            device=current_positions.device,
            dtype=torch.long,
        ).index_copy(0, slots, recent_positions)
        return LocalAttentionState(
            key=key,
            value=value,
            positions=positions,
            length=keep,
            write_index=next_position % capacity,
        )

    def forward_with_state(
        self,
        hidden: torch.Tensor,
        state: LocalAttentionState | None = None,
        *,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, LocalAttentionState]:
        if hidden.ndim != 3:
            raise ValueError("hidden must have shape [batch, sequence, d_model]")
        batch, sequence, _ = hidden.shape
        if sequence <= 0:
            raise ValueError("local attention requires a non-empty sequence")
        if position_offset < 0:
            raise ValueError("position_offset must be non-negative")
        if state is None:
            if position_offset != 0:
                raise ValueError("A nonzero position_offset requires a cache state")
            state = self.initial_state(batch, device=hidden.device, dtype=hidden.dtype)
        self._validate_state(
            state,
            batch_size=batch,
            device=hidden.device,
            dtype=hidden.dtype,
            position_offset=position_offset,
        )

        qkv = self.qkv(hidden)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch, sequence, self.config.n_heads, self.config.head_dim).transpose(1, 2)
        k = k.view(batch, sequence, self.config.n_heads, self.config.head_dim).transpose(1, 2)
        v = v.view(batch, sequence, self.config.n_heads, self.config.head_dim).transpose(1, 2)
        cached_k, cached_v, cached_positions = self._chronological_cache(state)
        current_positions = torch.arange(
            position_offset,
            position_offset + sequence,
            device=hidden.device,
            dtype=torch.long,
        )
        all_k = torch.cat((cached_k, k), dim=2)
        all_v = torch.cat((cached_v, v), dim=2)
        all_positions = torch.cat((cached_positions, current_positions), dim=0)
        allowed = (all_positions[None, :] <= current_positions[:, None]) & (
            all_positions[None, :] > current_positions[:, None] - self.config.local_attention_window
        )
        mask = torch.full(
            (sequence, all_positions.numel()),
            float("-inf"),
            device=hidden.device,
        ).masked_fill(allowed, 0.0)
        result = F.scaled_dot_product_attention(q, all_k, all_v, attn_mask=mask)
        result = result.transpose(1, 2).contiguous().view(batch, sequence, self.config.d_model)
        output = self.out_proj(result)
        updated_state = self._updated_ring(
            cached_key=cached_k,
            cached_value=cached_v,
            cached_positions=cached_positions,
            current_key=k,
            current_value=v,
            current_positions=current_positions,
            next_position=position_offset + sequence,
        )
        return output, updated_state

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Preserve the original complete-sequence module interface."""

        output, _ = self.forward_with_state(hidden)
        return output


class CatenaBlock(nn.Module):
    def __init__(self, config: ModelConfig, layer_index: int) -> None:
        super().__init__()
        self.layer_index = layer_index
        self.backend_id = config.backend_id
        self.optimized_chunk_size = config.optimized_chunk_size
        self.is_recurrent = layer_index in config.recurrent_layers
        self.norm1 = RMSNorm(config.d_model, config.rms_norm_eps)
        self.norm2 = RMSNorm(config.d_model, config.rms_norm_eps)
        if self.is_recurrent:
            self.mixer: nn.Module = TransactionalDeltaMixer(config, layer_index)
        else:
            self.mixer = LocalCausalSelfAttention(config)
        self.ffn = SwiGLU(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden: torch.Tensor,
        recurrent_state: MixerState | None,
        attention_state: LocalAttentionState | None,
        *,
        position_offset: int,
        chunked_reference: bool,
        gate_intervention: GateIntervention | None,
        address_intervention: AddressIntervention | None,
        return_gate_trace: bool,
    ) -> tuple[
        torch.Tensor,
        MixerState | None,
        LocalAttentionState | None,
        GateTrace | None,
    ]:
        normalized = self.norm1(hidden)
        if self.is_recurrent:
            mixer = self.mixer
            assert isinstance(mixer, TransactionalDeltaMixer)
            if chunked_reference:
                update, state, trace = mixer.forward_chunked_reference(
                    normalized,
                    recurrent_state,
                    gate_intervention=gate_intervention,
                    address_intervention=address_intervention,
                    return_gate_trace=return_gate_trace,
                    token_offset=position_offset,
                )
            elif self.backend_id == "compiled_scan":
                update, state, trace = mixer.forward_optimized(
                    normalized,
                    recurrent_state,
                    chunk_size=self.optimized_chunk_size,
                    gate_intervention=gate_intervention,
                    address_intervention=address_intervention,
                    return_gate_trace=return_gate_trace,
                    token_offset=position_offset,
                )
            else:
                update, state, trace = mixer.forward_reference(
                    normalized,
                    recurrent_state,
                    gate_intervention=gate_intervention,
                    address_intervention=address_intervention,
                    return_gate_trace=return_gate_trace,
                    token_offset=position_offset,
                )
            next_attention_state = None
        else:
            mixer = self.mixer
            assert isinstance(mixer, LocalCausalSelfAttention)
            update, next_attention_state = mixer.forward_with_state(
                normalized,
                attention_state,
                position_offset=position_offset,
            )
            state = None
            trace = None
        hidden = hidden + self.dropout(update)
        hidden = hidden + self.dropout(self.ffn(self.norm2(hidden)))
        return hidden, state, next_attention_state, trace


@dataclass
class RuntimeState:
    """Cloneable recurrent and local-attention state for branch evaluation."""

    recurrent: list[MixerState]
    position: int = 0
    attention: list[LocalAttentionState] = field(default_factory=list)

    def clone(self, *, detach: bool = False) -> RuntimeState:
        return RuntimeState(
            recurrent=[state.clone(detach=detach) for state in self.recurrent],
            attention=[state.clone(detach=detach) for state in self.attention],
            position=self.position,
        )

    def storage_ptrs(self) -> list[int]:
        pointers = [state.matrix.untyped_storage().data_ptr() for state in self.recurrent]
        for state in self.attention:
            pointers.extend(state.storage_ptrs())
        return pointers


@dataclass
class ModelOutput:
    logits: torch.Tensor
    runtime_state: RuntimeState
    gate_traces: dict[int, GateTrace]
    hidden: torch.Tensor | None = None


class CatenaLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(
            CatenaBlock(config, layer_index) for layer_index in range(config.n_layers)
        )
        self.final_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.weight_tying:
            self.lm_head.weight = self.embedding.weight
        self.apply(self._initialize)

    def _initialize(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @property
    def recurrent_layer_indices(self) -> tuple[int, ...]:
        return self.config.recurrent_layers

    @property
    def has_local_attention(self) -> bool:
        return bool(self.config.local_attention_layers)

    def initial_runtime_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> RuntimeState:
        recurrent_states: list[MixerState] = []
        for layer_index in self.recurrent_layer_indices:
            mixer = self.blocks[layer_index].mixer
            assert isinstance(mixer, TransactionalDeltaMixer)
            recurrent_states.append(mixer.initial_state(batch_size, device=device, dtype=dtype))
        attention_states: list[LocalAttentionState] = []
        for layer_index in self.config.local_attention_layers:
            mixer = self.blocks[layer_index].mixer
            assert isinstance(mixer, LocalCausalSelfAttention)
            attention_states.append(mixer.initial_state(batch_size, device=device, dtype=dtype))
        return RuntimeState(
            recurrent=recurrent_states,
            attention=attention_states,
            position=0,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        runtime_state: RuntimeState | None = None,
        *,
        chunked_reference: bool = False,
        gate_intervention: GateIntervention | None = None,
        address_intervention: AddressIntervention | None = None,
        return_gate_trace: bool = False,
        return_hidden: bool = False,
    ) -> ModelOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.shape[1] > self.config.context_length:
            raise ValueError("input sequence exceeds configured context length")
        hidden = self.embedding(input_ids)
        if runtime_state is None:
            runtime_state = self.initial_runtime_state(
                input_ids.shape[0], device=hidden.device, dtype=hidden.dtype
            )
        if runtime_state.position < 0:
            raise ValueError("runtime_state.position must be non-negative")
        if len(runtime_state.recurrent) != len(self.recurrent_layer_indices):
            raise ValueError("runtime_state has the wrong number of recurrent layer states")
        if len(runtime_state.attention) != len(self.config.local_attention_layers):
            raise ValueError("runtime_state has the wrong number of attention layer states")

        recurrent_cursor = 0
        attention_cursor = 0
        output_states: list[MixerState] = []
        output_attention_states: list[LocalAttentionState] = []
        traces: dict[int, GateTrace] = {}
        for layer_index, block in enumerate(self.blocks):
            recurrent_state = None
            attention_state = None
            if block.is_recurrent:
                recurrent_state = runtime_state.recurrent[recurrent_cursor]
            else:
                attention_state = runtime_state.attention[attention_cursor]
            hidden, recurrent_state, attention_state, trace = block(
                hidden,
                recurrent_state,
                attention_state,
                position_offset=runtime_state.position,
                chunked_reference=chunked_reference,
                gate_intervention=gate_intervention,
                address_intervention=address_intervention,
                return_gate_trace=return_gate_trace,
            )
            if block.is_recurrent:
                assert recurrent_state is not None
                output_states.append(recurrent_state)
                recurrent_cursor += 1
            else:
                assert attention_state is not None
                output_attention_states.append(attention_state)
                attention_cursor += 1
            if trace is not None:
                traces[layer_index] = trace
        normalized = self.final_norm(hidden)
        logits = self.lm_head(normalized)
        return ModelOutput(
            logits=logits,
            runtime_state=RuntimeState(
                recurrent=output_states,
                attention=output_attention_states,
                position=runtime_state.position + input_ids.shape[1],
            ),
            gate_traces=traces,
            hidden=normalized if return_hidden else None,
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def parameter_signature(self) -> list[dict[str, Any]]:
        return named_parameter_signature(self)

    def initialization_digest(self) -> str:
        return state_dict_digest(self)

    def assert_reference_only_for_main(self) -> None:
        if not self.config.backend_scientific_main_capable:
            raise RuntimeError(
                "Scientific MAIN is blocked: the selected backend is the reference "
                "Python recurrence. Integrate and validate an optimized backend first."
            )


def build_paired_models(
    base_config: ModelConfig,
    *,
    seed: int,
    device: torch.device | str = "cpu",
) -> tuple[CatenaLM, CatenaLM]:
    torch.manual_seed(seed)
    base = CatenaLM(ModelConfig.from_mapping({**base_config.to_dict(), "variant": "dual_delta_lm"}))
    state = copy.deepcopy(base.state_dict())
    dual = base
    tied_config = ModelConfig.from_mapping(
        {**base_config.to_dict(), "variant": "projected_tied_delta_lm"}
    )
    tied = CatenaLM(tied_config)
    tied.load_state_dict(state, strict=True)
    dual.to(device)
    tied.to(device)
    return tied, dual


def assert_matched_models(left: CatenaLM, right: CatenaLM) -> None:
    if left.parameter_signature() != right.parameter_signature():
        raise AssertionError("Named parameter signatures do not match")
    if left.parameter_count() != right.parameter_count():
        raise AssertionError("Parameter counts do not match")
    for (left_name, left_parameter), (right_name, right_parameter) in zip(
        left.named_parameters(), right.named_parameters(), strict=True
    ):
        if left_name != right_name:
            raise AssertionError(f"Parameter name mismatch: {left_name} vs {right_name}")
        if not torch.equal(left_parameter.detach().cpu(), right_parameter.detach().cpu()):
            raise AssertionError(f"Initial tensor mismatch at {left_name}")


def cross_entropy_loss(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    if logits.shape[:2] != input_ids.shape:
        raise ValueError("logits and input_ids sequence shapes differ")
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        input_ids[:, 1:].reshape(-1),
    )
