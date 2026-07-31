from __future__ import annotations

import hashlib
import threading
import types
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig, canonical_variant
from .interventions import AddressIntervention, GateIntervention


@dataclass
class MixerState:
    matrix: torch.Tensor

    def clone(self, *, detach: bool = False) -> MixerState:
        tensor = self.matrix.detach().clone() if detach else self.matrix.clone()
        return MixerState(matrix=tensor)

    def to(self, *args: Any, **kwargs: Any) -> MixerState:
        return MixerState(matrix=self.matrix.to(*args, **kwargs))


@dataclass
class GateTrace:
    raw_erase_logits: torch.Tensor
    raw_write_logits: torch.Tensor
    erase: torch.Tensor
    write: torch.Tensor
    decay: torch.Tensor


class OptimizedBackendUnsupported(RuntimeError):
    """Raised instead of silently falling back to the Python reference path."""


@dataclass
class _BackendDiagnostics:
    graph_compilations: int = 0
    graph_invocations: int = 0
    optimized_calls: int = 0
    chunks_executed: int = 0
    padded_tokens: int = 0
    fallback_count: int = 0
    graph_break_count: int = 0
    last_graph_node_count: int = 0
    last_graph_code_sha256: str | None = None


_DIAGNOSTICS = _BackendDiagnostics()
_DIAGNOSTICS_LOCK = threading.Lock()
_COMPILED_SCANS: dict[
    tuple[Any, ...],
    Callable[..., tuple[torch.Tensor, torch.Tensor]],
] = {}
_COMPILED_SCANS_LOCK = threading.Lock()


def _fixed_chunk_scan(
    matrix: torch.Tensor,
    q: torch.Tensor,
    erase_key: torch.Tensor,
    write_key: torch.Tensor,
    value: torch.Tensor,
    erase: torch.Tensor,
    write: torch.Tensor,
    decay: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed-shape recurrence captured and unrolled by ``torch.compile``.

    ``q`` has shape ``[batch, fixed_chunk, heads, head_dim]``. The Python
    ``range`` is evaluated only while Dynamo captures a static full graph; the
    emitted runtime graph contains no Python token loop. Tail chunks are padded
    with identity transitions by the caller.
    """

    outputs: list[torch.Tensor] = []
    for token_index in range(q.shape[1]):
        q_t = q[:, token_index]
        erase_key_t = erase_key[:, token_index]
        write_key_t = write_key[:, token_index]
        value_t = value[:, token_index]
        erase_t = erase[:, token_index]
        write_t = write[:, token_index]
        decay_t = decay[:, token_index]
        old_value = torch.einsum("bhd,bhdv->bhv", erase_key_t, matrix)
        erase_outer = erase_key_t.unsqueeze(-1) * old_value.unsqueeze(-2)
        write_outer = write_key_t.unsqueeze(-1) * value_t.unsqueeze(-2)
        matrix = (
            decay_t[..., None, None] * matrix
            - erase_t[..., None, None] * erase_outer
            + write_t[..., None, None] * write_outer
        )
        outputs.append(torch.einsum("bhd,bhdv->bhv", q_t, matrix))
    return torch.stack(outputs, dim=1), matrix


def _compiler_backend(
    compiler: str,
) -> Callable[[torch.fx.GraphModule, list[torch.Tensor]], Callable[..., Any]]:
    def compile_graph(
        graph_module: torch.fx.GraphModule,
        example_inputs: list[torch.Tensor],
    ) -> Callable[..., Any]:
        code = graph_module.code.encode("utf-8")
        with _DIAGNOSTICS_LOCK:
            _DIAGNOSTICS.graph_compilations += 1
            _DIAGNOSTICS.last_graph_node_count = sum(1 for _ in graph_module.graph.nodes)
            _DIAGNOSTICS.last_graph_code_sha256 = hashlib.sha256(code).hexdigest()
        if compiler == "eager":
            return graph_module.forward
        if compiler != "inductor":
            raise OptimizedBackendUnsupported(
                f"Unsupported optimized compiler {compiler!r}; expected 'eager' or 'inductor'"
            )
        # Import lazily so reference-only CPU validation does not initialize
        # Inductor or require a working C++/CUDA toolchain.
        from torch._inductor import compile as compile_inductor

        inductor_inputs = cast(
            list[torch.Tensor | int | torch.SymInt | None],
            example_inputs,
        )
        return cast(
            Callable[..., Any],
            compile_inductor(graph_module, inductor_inputs),
        )

    return compile_graph


def _compiled_scan(
    compiler: str,
    inputs: tuple[torch.Tensor, ...],
) -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
    signature: tuple[Any, ...] = (
        compiler,
        tuple((str(item.device), str(item.dtype), tuple(item.shape)) for item in inputs),
        torch.is_grad_enabled(),
        tuple(item.requires_grad for item in inputs),
    )
    with _COMPILED_SCANS_LOCK:
        compiled = _COMPILED_SCANS.get(signature)
        if compiled is None:
            # Dynamo's recompile cache is attached to a Python code object.
            # Give every registered tensor signature its own code object so the
            # preregistered device/dtype/batch parity grid cannot exhaust the
            # global per-code recompile limit. No generated function changes
            # the equation; only its cache identity is distinct.
            digest = hashlib.sha256(repr(signature).encode("utf-8")).hexdigest()[:16]
            name = f"_fixed_chunk_scan_{digest}"
            code = _fixed_chunk_scan.__code__.replace(
                co_name=name,
                co_qualname=name,
            )
            specialized = types.FunctionType(
                code,
                _fixed_chunk_scan.__globals__,
                name,
                _fixed_chunk_scan.__defaults__,
                _fixed_chunk_scan.__closure__,
            )
            compiled = torch.compile(
                specialized,
                backend=_compiler_backend(compiler),
                fullgraph=True,
                dynamic=False,
            )
            _COMPILED_SCANS[signature] = compiled
        return compiled


def reset_optimized_backend_diagnostics() -> None:
    """Reset counters without invalidating already compiled code."""

    with _DIAGNOSTICS_LOCK:
        _DIAGNOSTICS.graph_compilations = 0
        _DIAGNOSTICS.graph_invocations = 0
        _DIAGNOSTICS.optimized_calls = 0
        _DIAGNOSTICS.chunks_executed = 0
        _DIAGNOSTICS.padded_tokens = 0
        _DIAGNOSTICS.fallback_count = 0
        _DIAGNOSTICS.graph_break_count = 0
        _DIAGNOSTICS.last_graph_node_count = 0
        _DIAGNOSTICS.last_graph_code_sha256 = None


def optimized_backend_diagnostics() -> dict[str, int | str | None]:
    with _DIAGNOSTICS_LOCK:
        return {
            "graph_compilations": _DIAGNOSTICS.graph_compilations,
            "graph_invocations": _DIAGNOSTICS.graph_invocations,
            "optimized_calls": _DIAGNOSTICS.optimized_calls,
            "chunks_executed": _DIAGNOSTICS.chunks_executed,
            "padded_tokens": _DIAGNOSTICS.padded_tokens,
            "fallback_count": _DIAGNOSTICS.fallback_count,
            "graph_break_count": _DIAGNOSTICS.graph_break_count,
            "last_graph_node_count": _DIAGNOSTICS.last_graph_node_count,
            "last_graph_code_sha256": _DIAGNOSTICS.last_graph_code_sha256,
        }


def optimized_backend_metadata(
    *,
    device: torch.device | str,
    compiler: str | None = None,
    chunk_size: int,
    parity_verified: bool = False,
) -> dict[str, Any]:
    resolved_device = torch.device(device)
    resolved_compiler = compiler or ("inductor" if resolved_device.type == "cuda" else "eager")
    return {
        "backend_id": "torch_compile_fixed_chunk_scan_v1",
        "algorithm": "static_chunk_unrolled_delta_recurrence",
        "compiler": resolved_compiler,
        "fullgraph": True,
        "dynamic_shapes": False,
        "chunk_size": int(chunk_size),
        "python_token_loop_at_runtime": False,
        "python_outer_chunk_loop": True,
        "tail_policy": "identity_pad_no_fallback",
        "silent_fallback_allowed": False,
        "device_type": resolved_device.type,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "candidate_codegen_capable": (
            resolved_device.type == "cuda" and resolved_compiler == "inductor"
        ),
        "parity_verified": bool(parity_verified),
        "scientific_main_capable": (
            resolved_device.type == "cuda"
            and resolved_compiler == "inductor"
            and bool(parity_verified)
        ),
        "accumulation_policy": (
            "bf16/fp16 input projections with fp32 recurrent-state, recurrent-output, "
            "and FFN-output projection accumulation; fp32 path matches reference"
        ),
        "supported_gate_interventions": [
            "scale",
            "override",
            "force_tied",
            "token_mask",
        ],
        "unsupported_gate_interventions": ["custom_python_callback"],
        "supported_address_interventions": [
            "erase_address",
            "write_address",
            "token_mask",
        ],
    }


class TransactionalDeltaMixer(nn.Module):
    """Slow, explicit CATENA recurrence used as a correctness oracle.

    Scientific MAIN is intentionally blocked when this reference implementation
    is the selected backend. The live repository must provide an optimized scan
    that passes parity against this module.
    """

    scientific_main_capable = False

    def __init__(self, config: ModelConfig, layer_index: int) -> None:
        super().__init__()
        self.config = config
        self.layer_index = int(layer_index)
        d_model = config.d_model
        n_heads = config.n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        # Both variants register this identical maximal two-output head.
        self.gate_head = nn.Linear(d_model, 2 * n_heads, bias=True)
        self.decay_head = nn.Linear(d_model, n_heads, bias=True)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> MixerState:
        shape = (
            batch_size,
            self.config.n_heads,
            self.config.head_dim,
            self.config.head_dim,
        )
        return MixerState(matrix=torch.zeros(shape, device=device, dtype=dtype))

    def _shape_heads(self, value: torch.Tensor) -> torch.Tensor:
        batch, sequence, _ = value.shape
        return value.view(batch, sequence, self.config.n_heads, self.config.head_dim)

    def _project_gates(self, hidden: torch.Tensor) -> tuple[torch.Tensor, ...]:
        logits = self.gate_head(hidden)
        z_e, z_w = logits.chunk(2, dim=-1)
        variant = canonical_variant(self.config.variant)
        if variant == "dual_delta_lm":
            erase = torch.sigmoid(z_e)
            write = torch.sigmoid(z_w)
        else:
            tied = torch.sigmoid(0.5 * (z_e + z_w))
            erase = tied
            write = tied
        # A stable carry near one at initialization. Decay is not an
        # experimental difference and uses identical parameters in both models.
        decay = torch.sigmoid(self.decay_head(hidden) + 4.0)
        return z_e, z_w, erase, write, decay

    @staticmethod
    def _absolute_token_mask(
        token_mask: torch.Tensor | None,
        *,
        token_offset: int,
        sequence: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if token_mask is None:
            return None
        if token_mask.ndim != 1:
            raise ValueError("token_mask must be one-dimensional")
        positions = torch.arange(sequence, device=device) + int(token_offset)
        in_range = positions < token_mask.numel()
        safe_positions = positions.clamp(max=max(token_mask.numel() - 1, 0))
        if token_mask.numel() == 0:
            return torch.zeros(sequence, dtype=torch.bool, device=device)
        selected = token_mask.to(device=device, dtype=torch.bool)[safe_positions]
        return in_range & selected

    def _apply_gate_intervention_vectorized(
        self,
        erase: torch.Tensor,
        write: torch.Tensor,
        intervention: GateIntervention | None,
        *,
        token_offset: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if intervention is None:
            return erase, write
        if intervention.custom is not None:
            raise OptimizedBackendUnsupported(
                "Custom Python gate callbacks are not supported by the optimized backend"
            )
        selected = self._absolute_token_mask(
            intervention.token_mask,
            token_offset=token_offset,
            sequence=erase.shape[1],
            device=erase.device,
        )
        candidate_erase = erase * float(intervention.erase_scale)
        candidate_write = write * float(intervention.write_scale)
        broadcast_shape = (erase.shape[0], erase.shape[2])
        if intervention.erase_override is not None:
            override = torch.broadcast_to(
                intervention.erase_override.to(erase),
                broadcast_shape,
            )
            candidate_erase = override[:, None, :].expand_as(erase)
        if intervention.write_override is not None:
            override = torch.broadcast_to(
                intervention.write_override.to(write),
                broadcast_shape,
            )
            candidate_write = override[:, None, :].expand_as(write)
        if intervention.force_tied:
            tied = 0.5 * (candidate_erase + candidate_write)
            candidate_erase = tied
            candidate_write = tied
        candidate_erase = candidate_erase.clamp(0.0, 1.0)
        candidate_write = candidate_write.clamp(0.0, 1.0)
        if selected is None:
            return candidate_erase, candidate_write
        mask = selected[None, :, None]
        return (
            torch.where(mask, candidate_erase, erase),
            torch.where(mask, candidate_write, write),
        )

    def _apply_address_intervention_vectorized(
        self,
        key: torch.Tensor,
        intervention: AddressIntervention | None,
        *,
        token_offset: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if intervention is None:
            return key, key
        selected = self._absolute_token_mask(
            intervention.token_mask,
            token_offset=token_offset,
            sequence=key.shape[1],
            device=key.device,
        )
        erase_key = key
        write_key = key
        broadcast_shape = (key.shape[0], key.shape[2], key.shape[3])
        if intervention.erase_address is not None:
            address = torch.broadcast_to(
                intervention.erase_address.to(key),
                broadcast_shape,
            )
            normalized = F.normalize(
                address,
                dim=-1,
                eps=self.config.key_norm_eps,
            )[:, None, :, :].expand_as(key)
            erase_key = (
                normalized
                if selected is None
                else torch.where(
                    selected[None, :, None, None],
                    normalized,
                    key,
                )
            )
        if intervention.write_address is not None:
            address = torch.broadcast_to(
                intervention.write_address.to(key),
                broadcast_shape,
            )
            normalized = F.normalize(
                address,
                dim=-1,
                eps=self.config.key_norm_eps,
            )[:, None, :, :].expand_as(key)
            write_key = (
                normalized
                if selected is None
                else torch.where(
                    selected[None, :, None, None],
                    normalized,
                    key,
                )
            )
        return erase_key, write_key

    def forward_optimized(
        self,
        hidden: torch.Tensor,
        state: MixerState | None = None,
        *,
        chunk_size: int | None = None,
        compiler: str | None = None,
        gate_intervention: GateIntervention | None = None,
        address_intervention: AddressIntervention | None = None,
        return_gate_trace: bool = False,
        token_offset: int = 0,
    ) -> tuple[torch.Tensor, MixerState, GateTrace | None]:
        """Run a fullgraph-compiled fixed-chunk recurrence.

        The only Python iteration is over fixed-size chunks. Each chunk's token
        recurrence is statically unrolled into an FX graph and compiled with
        Inductor on CUDA. There is no automatic reference fallback.
        """

        if hidden.ndim != 3:
            raise ValueError("hidden must have shape [batch, sequence, d_model]")
        batch, sequence, _ = hidden.shape
        if sequence <= 0:
            raise ValueError("optimized recurrence requires a non-empty sequence")
        size = int(chunk_size or self.config.optimized_chunk_size)
        if size <= 0:
            raise ValueError("chunk_size must be positive")
        resolved_compiler = compiler or ("inductor" if hidden.device.type == "cuda" else "eager")
        if resolved_compiler not in {"eager", "inductor"}:
            raise OptimizedBackendUnsupported(
                f"Unsupported compiler {resolved_compiler!r}; expected 'eager' or 'inductor'"
            )
        if state is None:
            state = self.initial_state(batch, device=hidden.device, dtype=hidden.dtype)
        matrix = state.matrix
        expected_shape = (
            batch,
            self.config.n_heads,
            self.config.head_dim,
            self.config.head_dim,
        )
        if matrix.shape != expected_shape:
            raise ValueError(f"Unexpected recurrent state shape: {tuple(matrix.shape)}")
        allowed_state_dtypes = {hidden.dtype}
        if hidden.dtype in {torch.bfloat16, torch.float16}:
            allowed_state_dtypes.add(torch.float32)
        if matrix.device != hidden.device or matrix.dtype not in allowed_state_dtypes:
            raise ValueError(
                "recurrent state device must match hidden; dtype must match hidden "
                "or be float32 for a low-precision hidden state"
            )

        q = F.normalize(
            self._shape_heads(self.q_proj(hidden)),
            dim=-1,
            eps=self.config.key_norm_eps,
        )
        key = F.normalize(
            self._shape_heads(self.k_proj(hidden)),
            dim=-1,
            eps=self.config.key_norm_eps,
        )
        value = self._shape_heads(self.v_proj(hidden))
        z_e, z_w, erase, write, decay = self._project_gates(hidden)
        erase, write = self._apply_gate_intervention_vectorized(
            erase,
            write,
            gate_intervention,
            token_offset=token_offset,
        )
        erase_key, write_key = self._apply_address_intervention_vectorized(
            key,
            address_intervention,
            token_offset=token_offset,
        )
        trace_erase = erase
        trace_write = write
        trace_decay = decay
        low_precision_dtypes = {torch.bfloat16, torch.float16}
        low_precision_scan = any(
            item.dtype in low_precision_dtypes for item in (q, key, value, erase, write, decay)
        )
        if low_precision_scan:
            # Keep the learned projections and gates under autocast, but prevent
            # recurrent rounding from accumulating over the token chain. This
            # is the registered candidate accumulation policy for E26a; it does
            # not change the recurrence equation.
            matrix = matrix.float()
            q = q.float()
            erase_key = erase_key.float()
            write_key = write_key.float()
            value = value.float()
            erase = erase.float()
            write = write.float()
            decay = decay.float()

        pieces: list[torch.Tensor] = []
        padded_tokens = 0
        chunks = 0
        for start in range(0, sequence, size):
            stop = min(start + size, sequence)
            valid = stop - start
            q_chunk = q[:, start:stop]
            erase_key_chunk = erase_key[:, start:stop]
            write_key_chunk = write_key[:, start:stop]
            value_chunk = value[:, start:stop]
            erase_chunk = erase[:, start:stop]
            write_chunk = write[:, start:stop]
            decay_chunk = decay[:, start:stop]
            if valid < size:
                padding = size - valid
                vector_padding = torch.zeros(
                    (
                        batch,
                        padding,
                        self.config.n_heads,
                        self.config.head_dim,
                    ),
                    device=hidden.device,
                    dtype=q.dtype,
                )
                gate_padding = torch.zeros(
                    (batch, padding, self.config.n_heads),
                    device=hidden.device,
                    dtype=erase.dtype,
                )
                q_chunk = torch.cat((q_chunk, vector_padding), dim=1)
                erase_key_chunk = torch.cat((erase_key_chunk, vector_padding), dim=1)
                write_key_chunk = torch.cat((write_key_chunk, vector_padding), dim=1)
                value_chunk = torch.cat((value_chunk, vector_padding), dim=1)
                erase_chunk = torch.cat((erase_chunk, gate_padding), dim=1)
                write_chunk = torch.cat((write_chunk, gate_padding), dim=1)
                decay_chunk = torch.cat((decay_chunk, torch.ones_like(gate_padding)), dim=1)
                padded_tokens += padding
            scan_inputs = (
                matrix,
                q_chunk,
                erase_key_chunk,
                write_key_chunk,
                value_chunk,
                erase_chunk,
                write_chunk,
                decay_chunk,
            )
            scan = _compiled_scan(resolved_compiler, scan_inputs)
            if low_precision_scan:
                with torch.autocast(device_type=hidden.device.type, enabled=False):
                    output_chunk, matrix = scan(*scan_inputs)
            else:
                output_chunk, matrix = scan(*scan_inputs)
            pieces.append(output_chunk[:, :valid].reshape(batch, valid, self.config.d_model))
            chunks += 1
        with _DIAGNOSTICS_LOCK:
            _DIAGNOSTICS.graph_invocations += chunks
            _DIAGNOSTICS.optimized_calls += 1
            _DIAGNOSTICS.chunks_executed += chunks
            _DIAGNOSTICS.padded_tokens += padded_tokens

        mixed_input = torch.cat(pieces, dim=1)
        if low_precision_scan:
            # F.linear on explicit float views avoids autocast rounding at the
            # final recurrent projection while retaining gradients to the
            # registered parameter tensor.
            with torch.autocast(device_type=hidden.device.type, enabled=False):
                mixed = F.linear(
                    mixed_input.float(),
                    self.out_proj.weight.float(),
                    self.out_proj.bias.float() if self.out_proj.bias is not None else None,
                ).to(hidden.dtype)
        else:
            mixed = self.out_proj(mixed_input)
        trace = None
        if return_gate_trace:
            trace = GateTrace(
                raw_erase_logits=z_e,
                raw_write_logits=z_w,
                erase=trace_erase,
                write=trace_write,
                decay=trace_decay,
            )
        return mixed, MixerState(matrix=matrix), trace

    def forward_reference(
        self,
        hidden: torch.Tensor,
        state: MixerState | None = None,
        *,
        gate_intervention: GateIntervention | None = None,
        address_intervention: AddressIntervention | None = None,
        return_gate_trace: bool = False,
        token_offset: int = 0,
    ) -> tuple[torch.Tensor, MixerState, GateTrace | None]:
        if hidden.ndim != 3:
            raise ValueError("hidden must have shape [batch, sequence, d_model]")
        batch, sequence, _ = hidden.shape
        if state is None:
            state = self.initial_state(batch, device=hidden.device, dtype=hidden.dtype)
        matrix = state.matrix
        if matrix.shape != (
            batch,
            self.config.n_heads,
            self.config.head_dim,
            self.config.head_dim,
        ):
            raise ValueError(f"Unexpected recurrent state shape: {tuple(matrix.shape)}")

        q = F.normalize(
            self._shape_heads(self.q_proj(hidden)), dim=-1, eps=self.config.key_norm_eps
        )
        k = F.normalize(
            self._shape_heads(self.k_proj(hidden)), dim=-1, eps=self.config.key_norm_eps
        )
        v = self._shape_heads(self.v_proj(hidden))
        z_e, z_w, erase, write, decay = self._project_gates(hidden)

        outputs: list[torch.Tensor] = []
        erase_trace: list[torch.Tensor] = []
        write_trace: list[torch.Tensor] = []
        for local_index in range(sequence):
            absolute_index = token_offset + local_index
            q_t = q[:, local_index]
            k_t = k[:, local_index]
            v_t = v[:, local_index]
            e_t = erase[:, local_index]
            w_t = write[:, local_index]
            d_t = decay[:, local_index]

            if gate_intervention is not None:
                e_t, w_t = gate_intervention.apply(
                    e_t,
                    w_t,
                    layer_index=self.layer_index,
                    token_index=absolute_index,
                )
            erase_key = k_t
            write_key = k_t
            if address_intervention is not None and address_intervention.applies(absolute_index):
                if address_intervention.erase_address is not None:
                    erase_key = F.normalize(
                        torch.broadcast_to(address_intervention.erase_address.to(k_t), k_t.shape),
                        dim=-1,
                        eps=self.config.key_norm_eps,
                    )
                if address_intervention.write_address is not None:
                    write_key = F.normalize(
                        torch.broadcast_to(address_intervention.write_address.to(k_t), k_t.shape),
                        dim=-1,
                        eps=self.config.key_norm_eps,
                    )

            old_value = torch.einsum("bhd,bhdv->bhv", erase_key, matrix)
            erase_outer = erase_key.unsqueeze(-1) * old_value.unsqueeze(-2)
            write_outer = write_key.unsqueeze(-1) * v_t.unsqueeze(-2)
            matrix = (
                d_t[..., None, None] * matrix
                - e_t[..., None, None] * erase_outer
                + w_t[..., None, None] * write_outer
            )
            output_t = torch.einsum("bhd,bhdv->bhv", q_t, matrix)
            outputs.append(output_t.reshape(batch, self.config.d_model))
            if return_gate_trace:
                erase_trace.append(e_t)
                write_trace.append(w_t)

        mixed = self.out_proj(torch.stack(outputs, dim=1))
        trace = None
        if return_gate_trace:
            trace = GateTrace(
                raw_erase_logits=z_e,
                raw_write_logits=z_w,
                erase=torch.stack(erase_trace, dim=1),
                write=torch.stack(write_trace, dim=1),
                decay=decay,
            )
        return mixed, MixerState(matrix=matrix), trace

    def forward_chunked_reference(
        self,
        hidden: torch.Tensor,
        state: MixerState | None = None,
        *,
        chunk_size: int | None = None,
        gate_intervention: GateIntervention | None = None,
        address_intervention: AddressIntervention | None = None,
        return_gate_trace: bool = False,
        token_offset: int = 0,
    ) -> tuple[torch.Tensor, MixerState, GateTrace | None]:
        """Chunked wrapper around the reference recurrence.

        This method exists to test state-carry semantics. It is not a scientific
        performance backend because the inner recurrence remains a Python loop.
        """

        size = int(chunk_size or self.config.reference_chunk_size)
        if size <= 0:
            raise ValueError("chunk_size must be positive")
        pieces: list[torch.Tensor] = []
        gate_pieces: list[GateTrace] = []
        current = state
        for start in range(0, hidden.shape[1], size):
            result, current, trace = self.forward_reference(
                hidden[:, start : start + size],
                current,
                gate_intervention=gate_intervention,
                address_intervention=address_intervention,
                return_gate_trace=return_gate_trace,
                token_offset=token_offset + start,
            )
            pieces.append(result)
            if trace is not None:
                gate_pieces.append(trace)
        merged_trace = None
        if gate_pieces:
            merged_trace = GateTrace(
                raw_erase_logits=torch.cat([item.raw_erase_logits for item in gate_pieces], dim=1),
                raw_write_logits=torch.cat([item.raw_write_logits for item in gate_pieces], dim=1),
                erase=torch.cat([item.erase for item in gate_pieces], dim=1),
                write=torch.cat([item.write for item in gate_pieces], dim=1),
                decay=torch.cat([item.decay for item in gate_pieces], dim=1),
            )
        assert current is not None
        return torch.cat(pieces, dim=1), current, merged_trace

    def forward(
        self,
        hidden: torch.Tensor,
        state: MixerState | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, MixerState, GateTrace | None]:
        return self.forward_reference(hidden, state, **kwargs)
