from __future__ import annotations

import copy
import os
from typing import Any, Sequence

from .protocol import CandidateScores
from .state_utils import tensor_tree_bytes


class RWKVPipAdapter:
    """Inference-only RWKV pip adapter used for H1/H2 runtime experiments."""

    def __init__(
        self,
        *,
        model_path: str,
        strategy: str = "cuda bf16",
        tokenizer: str = "rwkv_vocab_v20230424",
        rwkv_cuda_on: bool = False,
        rwkv_jit_on: bool = True,
    ) -> None:
        if not model_path:
            raise ValueError("RWKV model_path is empty")
        os.environ["RWKV_V7_ON"] = "1"
        os.environ["RWKV_CUDA_ON"] = "1" if rwkv_cuda_on else "0"
        os.environ["RWKV_JIT_ON"] = "1" if rwkv_jit_on else "0"
        from rwkv.model import RWKV
        from rwkv.utils import PIPELINE

        self.model = RWKV(model=model_path, strategy=strategy)
        self.pipeline = PIPELINE(self.model, tokenizer)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "RWKVPipAdapter":
        model_path = config.get("model_path")
        env_name = config.get("model_path_env")
        if not model_path and env_name:
            model_path = os.getenv(str(env_name), "")
        return cls(
            model_path=str(model_path or ""),
            strategy=str(config.get("strategy", "cuda bf16")),
            tokenizer=str(config.get("tokenizer", "rwkv_vocab_v20230424")),
            rwkv_cuda_on=bool(config.get("rwkv_cuda_on", False)),
            rwkv_jit_on=bool(config.get("rwkv_jit_on", True)),
        )

    def encode(self, text: str) -> list[int]:
        return list(self.pipeline.encode(text))

    def prefill_text(self, text: str, state=None):
        token_ids = self.encode(text)
        if not token_ids:
            return state
        _, next_state = self.model.forward(token_ids, state)
        return next_state

    def prefill_embeddings(self, embeddings, state=None):
        raise NotImplementedError(
            "The rwkv pip backend does not expose differentiable inputs_embeds. "
            "Use the FLA/HF backend for H3/H4."
        )

    def score_candidates(self, state, query: str, candidates: Sequence[str]) -> CandidateScores:
        import torch.nn.functional as F

        query_ids = self.encode(query)
        if query_ids:
            logits, query_state = self.model.forward(query_ids, copy.deepcopy(state))
        else:
            raise ValueError("Query tokenization produced no tokens")
        scores: list[float] = []
        for candidate in candidates:
            ids = self.encode(candidate)
            if not ids:
                scores.append(float("-inf"))
                continue
            candidate_state = copy.deepcopy(query_state)
            next_logits = logits
            total = 0.0
            for token_id in ids:
                total += float(F.log_softmax(next_logits.float(), dim=-1)[token_id].item())
                next_logits, candidate_state = self.model.forward([token_id], candidate_state)
            scores.append(total / max(1, len(ids)))
        return CandidateScores(list(candidates), scores)

    def generate_greedy(
        self,
        state,
        prompt: str,
        *,
        max_new_tokens: int = 32,
    ):
        import torch

        prompt_ids = self.encode(prompt)
        if not prompt_ids:
            raise ValueError("Prompt tokenization produced no tokens")
        logits, current = self.model.forward(prompt_ids, copy.deepcopy(state))
        generated: list[int] = []
        for _ in range(max_new_tokens):
            token_id = int(torch.argmax(logits).item())
            generated.append(token_id)
            logits, current = self.model.forward([token_id], current)
        return generated, current

    def decode(self, token_ids: Sequence[int]) -> str:
        return self.pipeline.decode(list(token_ids))

    def state_sequence_length(self, state) -> int:
        return -1

    def clone_state(self, state):
        return copy.deepcopy(state)

    def state_bytes(self, state) -> int:
        return tensor_tree_bytes(state)
