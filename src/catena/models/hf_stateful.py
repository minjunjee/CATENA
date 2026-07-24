from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Sequence

from .protocol import CandidateScores
from .state_utils import clone_tree, tensor_tree_bytes


@dataclass
class HFState:
    cache: Any
    sequence_length: int


class HFStatefulAdapter:
    """Generic Hugging Face adapter for Qwen and FLA-format RWKV models.

    The adapter intentionally uses the model's standard forward path. For Transformer
    models, ``cache`` is a KV cache. For FLA RWKV models it is expected to be the
    recurrent cache object returned as ``past_key_values``.
    """

    def __init__(
        self,
        *,
        model_id: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        trust_remote_code: bool = False,
        attn_implementation: str | None = None,
        use_cache: bool = True,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        if dtype not in dtype_map:
            raise ValueError(f"Unsupported dtype: {dtype}")
        kwargs: dict[str, Any] = {
            "torch_dtype": dtype_map[dtype],
            "trust_remote_code": trust_remote_code,
        }
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=trust_remote_code, use_fast=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).to(device)
        self.model.eval()
        self.device = torch.device(device)
        self.use_cache = use_cache
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    @classmethod
    def from_config(cls, config: dict[str, Any], device: str = "cuda") -> "HFStatefulAdapter":
        return cls(
            model_id=str(config["model_id"]),
            device=device,
            dtype=str(config.get("dtype", "bfloat16")),
            trust_remote_code=bool(config.get("trust_remote_code", False)),
            attn_implementation=config.get("attn_implementation"),
            use_cache=bool(config.get("use_cache", True)),
        )

    def freeze_backbone(self) -> None:
        self.model.requires_grad_(False)
        self.model.eval()

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer(text, add_special_tokens=False).input_ids)

    def _cache_seq_len(self, cache: Any, fallback: int = 0) -> int:
        if cache is None:
            return 0
        for attr in ("get_seq_length", "get_usable_length"):
            fn = getattr(cache, attr, None)
            if callable(fn):
                try:
                    return int(fn())
                except TypeError:
                    try:
                        return int(fn(0))
                    except Exception:
                        pass
        if isinstance(cache, (tuple, list)) and cache:
            layer = cache[0]
            if isinstance(layer, (tuple, list)) and layer:
                tensor = layer[0]
                if hasattr(tensor, "shape") and len(tensor.shape) >= 3:
                    return int(tensor.shape[-2])
        return fallback

    def _forward(
        self,
        *,
        input_ids=None,
        inputs_embeds=None,
        state: HFState | None = None,
        grad: bool = False,
    ):
        import torch

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Exactly one of input_ids and inputs_embeds must be provided")
        new_len = int(input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1])
        past_len = 0 if state is None else state.sequence_length
        kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "inputs_embeds": inputs_embeds,
            "past_key_values": None if state is None else state.cache,
            "use_cache": self.use_cache,
            "return_dict": True,
        }
        # Most decoder-only models accept an all-ones mask. Some remote-code linear
        # recurrent models do not need it, so retry without it on a shape/API error.
        kwargs["attention_mask"] = torch.ones(
            (1, past_len + new_len), dtype=torch.long, device=self.device
        )
        context = torch.enable_grad() if grad else torch.no_grad()
        with context:
            try:
                outputs = self.model(**kwargs)
            except (TypeError, ValueError, RuntimeError) as first_error:
                kwargs.pop("attention_mask", None)
                try:
                    outputs = self.model(**kwargs)
                except Exception:
                    raise RuntimeError(
                        "HF stateful forward failed with and without attention_mask. "
                        f"Original error: {first_error!r}"
                    )
        cache = getattr(outputs, "past_key_values", None)
        if self.use_cache and cache is None:
            raise RuntimeError("Model did not return past_key_values/recurrent cache")
        seq_len = self._cache_seq_len(cache, fallback=past_len + new_len)
        return outputs, HFState(cache=cache, sequence_length=seq_len)

    def prefill_token_ids(
        self, token_ids: Sequence[int], state: HFState | None = None, *, grad: bool = False
    ) -> HFState:
        import torch

        ids = list(token_ids)
        if not ids:
            return state if state is not None else HFState(cache=None, sequence_length=0)
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        _, next_state = self._forward(input_ids=input_ids, state=state, grad=grad)
        return next_state

    def prefill_text(self, text: str, state: HFState | None = None) -> HFState:
        return self.prefill_token_ids(self.encode(text), state, grad=False)

    def prefill_embeddings(
        self,
        embeddings,
        state: HFState | None = None,
        *,
        grad: bool = True,
    ) -> HFState:
        _, next_state = self._forward(inputs_embeds=embeddings, state=state, grad=grad)
        return next_state

    def forward_embeddings(
        self,
        embeddings,
        state: HFState | None = None,
        *,
        grad: bool = True,
    ):
        return self._forward(inputs_embeds=embeddings, state=state, grad=grad)

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def _query_state_and_logits(self, state: HFState, query: str):
        import torch

        query_ids = self.encode(query)
        if not query_ids:
            query_ids = [self.tokenizer.eos_token_id]
        input_ids = torch.tensor([query_ids], dtype=torch.long, device=self.device)
        outputs, query_state = self._forward(
            input_ids=input_ids, state=self.clone_state(state), grad=False
        )
        return query_state, outputs.logits[:, -1, :]

    def score_candidates(
        self, state: HFState, query: str, candidates: Sequence[str]
    ) -> CandidateScores:
        import torch
        import torch.nn.functional as F

        query_state, next_logits = self._query_state_and_logits(state, query)
        scores: list[float] = []
        for candidate in candidates:
            token_ids = self.encode(candidate)
            if not token_ids:
                scores.append(float("-inf"))
                continue
            candidate_score = 0.0
            logits = next_logits
            cand_state = self.clone_state(query_state)
            for token_id in token_ids:
                candidate_score += float(F.log_softmax(logits.float(), dim=-1)[0, token_id].item())
                token = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
                outputs, cand_state = self._forward(input_ids=token, state=cand_state, grad=False)
                logits = outputs.logits[:, -1, :]
            scores.append(candidate_score / max(1, len(token_ids)))
        return CandidateScores(list(candidates), scores)


    def continuation_log_likelihood(
        self,
        state: HFState,
        prefix: str,
        continuation: str,
        *,
        grad: bool = True,
    ):
        """Return a differentiable log-likelihood for ``continuation``.

        ``prefix`` must contain at least one token so the first continuation token
        has a prediction position inside this forward call.
        """
        import torch
        import torch.nn.functional as F

        prefix_ids = self.encode(prefix)
        continuation_ids = self.encode(continuation)
        if not continuation_ids:
            return torch.tensor(float("-inf"), device=self.device)
        if not prefix_ids:
            prefix_ids = [self.tokenizer.eos_token_id]
        all_ids = prefix_ids + continuation_ids
        input_ids = torch.tensor([all_ids], dtype=torch.long, device=self.device)
        outputs, _ = self._forward(input_ids=input_ids, state=state, grad=grad)
        logits = outputs.logits[0]
        start = len(prefix_ids) - 1
        stop = start + len(continuation_ids)
        prediction_logits = logits[start:stop, :]
        targets = torch.tensor(continuation_ids, dtype=torch.long, device=self.device)
        token_log_probs = F.log_softmax(prediction_logits.float(), dim=-1)
        return token_log_probs.gather(1, targets[:, None]).mean()

    def clone_state(self, state: HFState) -> HFState:
        cache = state.cache
        # New Hugging Face Cache objects sometimes provide a native copy method.
        for attr in ("clone", "copy"):
            fn = getattr(cache, attr, None)
            if callable(fn):
                try:
                    cloned = fn()
                    return HFState(cloned, state.sequence_length)
                except Exception:
                    pass
        try:
            cloned = copy.deepcopy(cache)
        except Exception:
            cloned = clone_tree(cache)
        return HFState(cache=cloned, sequence_length=state.sequence_length)

    def crop_state(self, state: HFState, max_length: int) -> HFState:
        cache = self.clone_state(state).cache
        crop = getattr(cache, "crop", None)
        if callable(crop):
            crop(max_length)
            return HFState(cache=cache, sequence_length=max_length)
        if isinstance(cache, tuple):
            cropped = tuple(
                tuple(t[..., :max_length, :].contiguous() for t in layer)
                for layer in cache
            )
            return HFState(cache=cropped, sequence_length=max_length)
        raise NotImplementedError("This cache type does not expose a crop operation")

    def generate_greedy(
        self,
        state: HFState,
        prompt: str,
        *,
        max_new_tokens: int = 32,
    ) -> tuple[list[int], HFState]:
        """Greedy decode used only for system profiling and tool-call smoke tests."""
        import torch

        prompt_ids = self.encode(prompt)
        if not prompt_ids:
            prompt_ids = [self.tokenizer.eos_token_id]
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        outputs, current = self._forward(
            input_ids=input_ids, state=self.clone_state(state), grad=False
        )
        logits = outputs.logits[:, -1, :]
        generated: list[int] = []
        for _ in range(max_new_tokens):
            token_id = int(logits.argmax(dim=-1).item())
            generated.append(token_id)
            token = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
            outputs, current = self._forward(input_ids=token, state=current, grad=False)
            logits = outputs.logits[:, -1, :]
            if token_id == self.tokenizer.eos_token_id:
                break
        return generated, current

    def decode(self, token_ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(token_ids), skip_special_tokens=True)

    def state_sequence_length(self, state: HFState) -> int:
        return int(state.sequence_length)

    def state_bytes(self, state: HFState) -> int:
        return tensor_tree_bytes(state.cache)
