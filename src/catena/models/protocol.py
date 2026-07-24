from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence


@dataclass
class CandidateScores:
    candidates: list[str]
    log_likelihoods: list[float]

    @property
    def prediction_index(self) -> int:
        return max(range(len(self.log_likelihoods)), key=self.log_likelihoods.__getitem__)


class StatefulLanguageModel(Protocol):
    """Minimal model contract used by the CATENA experiments."""

    def encode(self, text: str) -> list[int]: ...

    def prefill_text(self, text: str, state: Any | None = None) -> Any: ...

    def prefill_embeddings(self, embeddings: Any, state: Any | None = None) -> Any: ...

    def score_candidates(self, state: Any, query: str, candidates: Sequence[str]) -> CandidateScores: ...

    def clone_state(self, state: Any) -> Any: ...

    def state_bytes(self, state: Any) -> int: ...
