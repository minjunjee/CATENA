from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from typing import Sequence

from .protocol import CandidateScores


@dataclass
class MockState:
    text: str


class MockStatefulModel:
    """Deterministic CPU-only model used for repo smoke tests."""

    def encode(self, text: str) -> list[int]:
        return [ord(ch) % 256 for ch in text]

    def prefill_text(self, text: str, state: MockState | None = None) -> MockState:
        prefix = "" if state is None else state.text
        return MockState(prefix + text)

    def prefill_embeddings(self, embeddings, state: MockState | None = None) -> MockState:
        prefix = "" if state is None else state.text
        return MockState(prefix + f"<embeddings:{getattr(embeddings, 'shape', '?')}>")

    def score_candidates(
        self, state: MockState, query: str, candidates: Sequence[str]
    ) -> CandidateScores:
        scores: list[float] = []
        haystack = (state.text + "\n" + query).lower()
        for candidate in candidates:
            overlap = sum(token.lower() in haystack for token in candidate.replace("{", " ").split())
            digest = hashlib.sha256((haystack + candidate).encode()).digest()
            scores.append(float(overlap) + int.from_bytes(digest[:2], "big") / 65535.0)
        return CandidateScores(list(candidates), scores)

    def clone_state(self, state: MockState) -> MockState:
        return copy.deepcopy(state)

    def state_bytes(self, state: MockState) -> int:
        return len(state.text.encode("utf-8"))
