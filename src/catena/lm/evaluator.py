from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch.nn import functional as F

from .model import CatenaLM, RuntimeState
from .tokenizer import Tokenizer
from .transactional_stream import QueryRecord, TransactionEpisode


@dataclass(frozen=True)
class CandidateScore:
    candidate: str
    log_probability: float
    token_count: int


@dataclass(frozen=True)
class QueryEvaluation:
    episode_id: str
    query_type: str
    correct: bool
    predicted_index: int
    gold_index: int
    candidate_scores: tuple[CandidateScore, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "query_type": self.query_type,
            "correct": self.correct,
            "predicted_index": self.predicted_index,
            "gold_index": self.gold_index,
            "candidate_scores": [item.__dict__ for item in self.candidate_scores],
        }


def _continuation_log_probability(
    model: CatenaLM,
    tokenizer: Tokenizer,
    prefix: str,
    continuation: str,
    *,
    device: torch.device | str,
) -> CandidateScore:
    prefix_ids = tokenizer.encode(prefix, add_bos=True)
    continuation_ids = tokenizer.encode(continuation, add_eos=True)
    input_ids = torch.tensor([prefix_ids + continuation_ids], device=device, dtype=torch.long)
    with torch.no_grad():
        logits = model(input_ids).logits
        log_probs = F.log_softmax(logits[:, :-1], dim=-1)
        labels = input_ids[:, 1:]
        token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    start = max(len(prefix_ids) - 1, 0)
    selected = token_log_probs[0, start : start + len(continuation_ids)]
    return CandidateScore(
        candidate=continuation,
        log_probability=float(selected.sum().item()),
        token_count=len(continuation_ids),
    )


def _score_candidate_from_runtime_state(
    model: CatenaLM,
    tokenizer: Tokenizer,
    prompt_last_logits: torch.Tensor,
    prompt_state: RuntimeState,
    continuation: str,
    *,
    device: torch.device | str,
) -> CandidateScore:
    """Score one continuation from an independently cloned branch state.

    The first continuation token is scored by the final prompt position.  The
    remaining tokens are scored by a continuation forward from a deep-cloned
    runtime state.  This avoids re-prefilling the update prefix and, crucially,
    prevents a future query or candidate from altering any sibling branch.
    """

    continuation_ids = tokenizer.encode(continuation, add_eos=True)
    if not continuation_ids:
        raise ValueError("A candidate continuation must contain at least one token")
    token_tensor = torch.tensor([continuation_ids], device=device, dtype=torch.long)
    first_log_probs = F.log_softmax(prompt_last_logits.float(), dim=-1)
    total = first_log_probs[0, continuation_ids[0]]
    if len(continuation_ids) > 1:
        with torch.no_grad():
            continued = model(token_tensor, prompt_state.clone(detach=True))
            subsequent = F.log_softmax(continued.logits[:, :-1].float(), dim=-1)
            labels = token_tensor[:, 1:]
            total = total + subsequent.gather(-1, labels.unsqueeze(-1)).sum()
    return CandidateScore(
        candidate=continuation,
        log_probability=float(total.item()),
        token_count=len(continuation_ids),
    )


def evaluate_episode_branched(
    model: CatenaLM,
    tokenizer: Tokenizer,
    episode: TransactionEpisode,
    *,
    device: torch.device | str = "cpu",
    query_order: Sequence[int] | None = None,
    exact_refresh: bool = False,
) -> tuple[QueryEvaluation, ...]:
    """Evaluate query branches from one frozen update-prefix runtime state.

    Unlike :func:`evaluate_episode_reprefill_reference`, this is the runtime
    contract used by the hybrid recurrent/local-attention model.  Recurrent
    matrices, local K/V buffers, and position metadata are cloned before every
    query and candidate.  The optional query permutation exists solely for the
    branch-contamination audit.
    """

    prefix = episode.exact_refresh_text if exact_refresh else episode.branch_prefix_text
    prefix_ids = tokenizer.encode(prefix, add_bos=True)
    if not prefix_ids:
        raise ValueError("Episode prefix must contain at least one token")
    prefix_tensor = torch.tensor([prefix_ids], device=device, dtype=torch.long)
    with torch.no_grad():
        prefix_output = model(prefix_tensor)
    frozen_prefix_state = prefix_output.runtime_state.clone(detach=True)

    order = tuple(range(len(episode.queries))) if query_order is None else tuple(query_order)
    if sorted(order) != list(range(len(episode.queries))):
        raise ValueError("query_order must be a permutation of all query indices")

    by_index: dict[int, QueryEvaluation] = {}
    for query_index in order:
        query = episode.queries[query_index]
        prompt = f"\n\n{query.prompt}\nANSWER:"
        prompt_ids = tokenizer.encode(prompt)
        if not prompt_ids:
            raise ValueError("Query prompt must contain at least one token")
        prompt_tensor = torch.tensor([prompt_ids], device=device, dtype=torch.long)
        with torch.no_grad():
            prompt_output = model(
                prompt_tensor,
                frozen_prefix_state.clone(detach=True),
            )
        branch_state = prompt_output.runtime_state.clone(detach=True)
        scores = tuple(
            _score_candidate_from_runtime_state(
                model,
                tokenizer,
                prompt_output.logits[:, -1],
                branch_state,
                candidate,
                device=device,
            )
            for candidate in query.candidate_answers
        )
        predicted = max(range(len(scores)), key=lambda index: scores[index].log_probability)
        by_index[query_index] = QueryEvaluation(
            episode_id=episode.episode_id,
            query_type=query.query_type,
            correct=predicted == query.gold_index,
            predicted_index=predicted,
            gold_index=query.gold_index,
            candidate_scores=scores,
        )
    return tuple(by_index[index] for index in range(len(episode.queries)))


def evaluate_query_repefill_reference(
    model: CatenaLM,
    tokenizer: Tokenizer,
    episode: TransactionEpisode,
    query: QueryRecord,
    *,
    device: torch.device | str = "cpu",
) -> QueryEvaluation:
    """Non-evidence candidate evaluation using common-prefix re-prefill.

    Scientific E26d must use cloneable runtime state, including local attention
    caches. This helper is intentionally named and documented as a re-prefill
    fallback for packet smoke tests.
    """

    prompt = f"{episode.branch_prefix_text}\n\n{query.prompt}\nANSWER:"
    scores = tuple(
        _continuation_log_probability(model, tokenizer, prompt, candidate, device=device)
        for candidate in query.candidate_answers
    )
    predicted = max(range(len(scores)), key=lambda index: scores[index].log_probability)
    return QueryEvaluation(
        episode_id=episode.episode_id,
        query_type=query.query_type,
        correct=predicted == query.gold_index,
        predicted_index=predicted,
        gold_index=query.gold_index,
        candidate_scores=scores,
    )


def evaluate_episode_reprefill_reference(
    model: CatenaLM,
    tokenizer: Tokenizer,
    episode: TransactionEpisode,
    *,
    device: torch.device | str = "cpu",
) -> tuple[QueryEvaluation, ...]:
    return tuple(
        evaluate_query_repefill_reference(model, tokenizer, episode, query, device=device)
        for query in episode.queries
    )


def transaction_score(rows: Iterable[QueryEvaluation]) -> float:
    by_type = {row.query_type: float(row.correct) for row in rows}
    required = {"current_state", "derived_action", "stale_probe"}
    if not required.issubset(by_type):
        raise ValueError(f"Missing query types: {sorted(required - set(by_type))}")
    return (
        0.40 * by_type["current_state"]
        + 0.40 * by_type["derived_action"]
        + 0.20 * by_type["stale_probe"]
    )


def exact_json_match(candidate: str, gold: dict[str, Any]) -> bool:
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return False
    return cast(bool, parsed == gold)


def perplexity_from_nll(total_nll: float, token_count: int) -> float:
    if token_count <= 0:
        raise ValueError("token_count must be positive")
    return float(torch.exp(torch.tensor(total_nll / token_count)).item())
