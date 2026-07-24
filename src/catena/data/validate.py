from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .schema import ChainEpisode, Episode


def read_jsonl(path: str | Path) -> Iterable[Episode]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                yield Episode.from_dict(payload)
            except Exception as exc:
                raise ValueError(f"Failed to parse {path}:{line_no}: {exc}") from exc


def read_chain_jsonl(path: str | Path) -> Iterable[ChainEpisode]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                yield ChainEpisode.from_dict(payload)
            except Exception as exc:
                raise ValueError(f"Failed to parse chain {path}:{line_no}: {exc}") from exc


def validate_episode(episode: Episode) -> list[str]:
    errors: list[str] = []
    if episode.transaction.new_version <= episode.transaction.old_version:
        errors.append("transaction version did not increase")
    if episode.current_state.get("version") != episode.transaction.new_version:
        errors.append("current state version differs from transaction")
    if not episode.queries:
        errors.append("episode has no queries")
    for query in episode.queries:
        if not 0 <= query.gold_index < len(query.candidates):
            errors.append(f"invalid gold_index for {query.query_id}")
        closure_text = " ".join(item.text for item in episode.closure)
        if query.kind == "affected_derived" and query.gold in closure_text:
            errors.append("dependency closure contains the full derived gold action")
    if not any(q.kind == "unaffected" for q in episode.queries):
        errors.append("missing retention query")
    if not any(q.kind.startswith("affected") for q in episode.queries):
        errors.append("missing correction query")
    return errors


def validate_file(path: str | Path) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    total = 0
    errors: list[dict[str, Any]] = []
    for episode in read_jsonl(path):
        total += 1
        counts[f"domain:{episode.domain}"] += 1
        counts[f"operation:{episode.transaction.operation}"] += 1
        counts[f"depth:{episode.dependency_depth}"] += 1
        episode_errors = validate_episode(episode)
        if episode_errors:
            errors.append({"episode_id": episode.episode_id, "errors": episode_errors})
    return {"path": str(path), "episodes": total, "counts": dict(counts), "errors": errors}
