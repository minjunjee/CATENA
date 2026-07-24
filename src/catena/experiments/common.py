from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from catena.data.render import render_segments
from catena.data.schema import Episode, Query
from catena.data.validate import read_jsonl
from catena.eval.metrics import PredictionRecord, stratified_summary
from catena.methods.policies import apply_text_policy, build_base_state
from catena.models.hf_stateful import HFStatefulAdapter
from catena.models.protocol import CandidateScores
from catena.utils.manifest import write_manifest
from catena.utils.timing import TimingResult, measured


def softmax(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr - np.max(arr)
    exp = np.exp(arr)
    return exp / np.sum(exp)


def categorical_kl_from_scores(reference: list[float], candidate: list[float]) -> float:
    p = softmax(reference)
    q = softmax(candidate)
    eps = 1e-12
    return float(np.sum(p * (np.log(p + eps) - np.log(q + eps))))


def score_query(model, state: Any, query: Query) -> CandidateScores:
    return model.score_candidates(state, query.prompt, query.candidates)


def _suffix_reprefill_state(model, episode: Episode, base_state: Any):
    if not isinstance(model, HFStatefulAdapter):
        raise TypeError("oracle_suffix_reprefill requires the HFStatefulAdapter")
    index = int(episode.metadata.get("affected_segment_index", 0))
    prefix_text = render_segments(episode.history_segments[:index])
    # The full history renderer places a newline between segments.  Preserve that
    # boundary so the prefix tokenization is a true prefix of the original prompt.
    if index > 0:
        prefix_text += "\n"
    prefix_len = len(model.encode(prefix_text))
    cropped = model.crop_state(base_state, prefix_len)
    suffix_text = render_segments(episode.refresh_segments[index:])
    return model.prefill_text(suffix_text, cropped)


def policy_state(
    model,
    episode: Episode,
    base_state: Any,
    policy: str,
    query: Query | None = None,
):
    if policy == "oracle_suffix_reprefill":
        return _suffix_reprefill_state(model, episode, base_state)
    return apply_text_policy(model, episode, base_state, policy, query=query)


def _sharded(
    episodes: Iterable[Episode], *, shard_index: int = 0, num_shards: int = 1
) -> Iterator[Episode]:
    if num_shards < 1:
        raise ValueError("num_shards must be positive")
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")
    for index, episode in enumerate(episodes):
        if index % num_shards == shard_index:
            yield episode


def run_policy_evaluation(
    *,
    model,
    episodes: Iterable[Episode],
    policies: list[str],
    output_dir: str | Path,
    max_episodes: int | None = None,
    config: dict[str, Any] | None = None,
    shard_index: int = 0,
    num_shards: int = 1,
) -> dict[str, Any]:
    output = Path(output_dir)
    if num_shards > 1:
        output = output / f"shard_{shard_index:02d}_of_{num_shards:02d}"
    output.mkdir(parents=True, exist_ok=True)
    write_manifest(
        output,
        {**(config or {}), "shard_index": shard_index, "num_shards": num_shards},
    )
    records_by_policy: dict[str, list[PredictionRecord]] = {
        policy: [] for policy in policies
    }
    raw_path = output / "predictions.jsonl"
    processed = 0

    with raw_path.open("w", encoding="utf-8") as writer:
        for episode in _sharded(
            episodes, shard_index=shard_index, num_shards=num_shards
        ):
            if max_episodes is not None and processed >= max_episodes:
                break
            base_state = build_base_state(model, episode)
            exact_state = apply_text_policy(model, episode, base_state, "exact_refresh")
            exact_scores = {
                query.query_id: score_query(model, exact_state, query)
                for query in episode.queries
            }
            for policy in policies:
                shared_state = None
                if policy != "query_time_retrieval":
                    timer = TimingResult()
                    with measured(timer):
                        shared_state = policy_state(model, episode, base_state, policy)
                    update_ms = timer.milliseconds
                else:
                    update_ms = 0.0
                for query in episode.queries:
                    if policy == "query_time_retrieval":
                        timer = TimingResult()
                        with measured(timer):
                            state = policy_state(
                                model, episode, base_state, policy, query=query
                            )
                        this_update_ms = timer.milliseconds
                    else:
                        state = shared_state
                        this_update_ms = update_ms
                    assert state is not None
                    scores = score_query(model, state, query)
                    exact = exact_scores[query.query_id]
                    record = PredictionRecord(
                        episode_id=episode.episode_id,
                        query_id=query.query_id,
                        query_kind=query.kind,
                        policy=policy,
                        prediction_index=scores.prediction_index,
                        gold_index=query.gold_index,
                        exact_prediction_index=exact.prediction_index,
                        teacher_correct=(exact.prediction_index == query.gold_index),
                        logit_kl=categorical_kl_from_scores(
                            exact.log_likelihoods, scores.log_likelihoods
                        ),
                        latency_ms=this_update_ms,
                        state_bytes=model.state_bytes(state),
                        domain=episode.domain,
                        operation=episode.transaction.operation,
                        history_tokens=episode.history_token_target,
                        dependency_depth=episode.dependency_depth,
                        query_gap_tokens=episode.query_gap_tokens,
                    )
                    records_by_policy[policy].append(record)
                    writer.write(
                        json.dumps(
                            {
                                **asdict(record),
                                "scores": scores.log_likelihoods,
                                "exact_scores": exact.log_likelihoods,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            processed += 1

    summary = {
        "episodes": processed,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "policies": {
            policy: stratified_summary(records)
            for policy, records in records_by_policy.items()
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def load_split(data_dir: str | Path, split: str = "test") -> Iterable[Episode]:
    return read_jsonl(Path(data_dir) / f"{split}.jsonl")
