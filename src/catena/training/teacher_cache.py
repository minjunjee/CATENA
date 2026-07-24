from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from catena.config import load_yaml
from catena.data.render import render_refresh_prompt, render_segments
from catena.data.schema import ChainEpisode, Episode
from catena.data.validate import read_chain_jsonl, read_jsonl
from catena.models.factory import load_model
from catena.utils.manifest import write_manifest


def _iter_shard(items: Iterable[Any], shard_index: int, num_shards: int):
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("Invalid shard configuration")
    for index, item in enumerate(items):
        if index % num_shards == shard_index:
            yield item


def _episode_id(episode: Episode | ChainEpisode) -> str:
    return episode.episode_id if isinstance(episode, Episode) else episode.chain_id


def _refresh_text(episode: Episode | ChainEpisode) -> str:
    if isinstance(episode, Episode):
        return render_refresh_prompt(episode)
    return render_segments(episode.refresh_segments)


def build_teacher_cache(
    config_path: str,
    *,
    device: str = "cuda",
    split: str = "train",
    shard_index: int = 0,
    num_shards: int = 1,
) -> Path:
    config = load_yaml(config_path)
    model = load_model(str(config["model"]), device=device)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(
        output_dir / f"manifest_shard_{shard_index:02d}_of_{num_shards:02d}",
        {**config, "split": split, "shard_index": shard_index, "num_shards": num_shards},
    )
    dataset_type = str(config.get("dataset_type", "episode"))
    data_path = Path(config["data_dir"]) / f"{split}.jsonl"
    if dataset_type == "chain":
        episodes: Iterable[Episode | ChainEpisode] = read_chain_jsonl(data_path)
    else:
        episodes = read_jsonl(data_path)

    suffix = "" if num_shards == 1 else f"_shard{shard_index:02d}_of_{num_shards:02d}"
    output_path = output_dir / f"{split}_teacher_scores{suffix}.jsonl"
    with output_path.open("w", encoding="utf-8") as writer:
        for episode in _iter_shard(episodes, shard_index, num_shards):
            exact_state = model.prefill_text(_refresh_text(episode), None)
            query_payload: dict[str, Any] = {}
            for query in episode.queries:
                scores = model.score_candidates(exact_state, query.prompt, query.candidates)
                query_payload[query.query_id] = {
                    "scores": scores.log_likelihoods,
                    "prediction_index": scores.prediction_index,
                    "gold_index": query.gold_index,
                }
            writer.write(
                json.dumps(
                    {"episode_id": _episode_id(episode), "queries": query_payload},
                    ensure_ascii=False,
                )
                + "\n"
            )
    return output_path


def merge_teacher_shards(
    output_dir: str | Path,
    *,
    split: str,
    num_shards: int,
    remove_shards: bool = False,
) -> Path:
    output = Path(output_dir)
    target = output / f"{split}_teacher_scores.jsonl"
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for shard_index in range(num_shards):
        path = output / f"{split}_teacher_scores_shard{shard_index:02d}_of_{num_shards:02d}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                payload = json.loads(line)
                episode_id = str(payload["episode_id"])
                if episode_id in seen:
                    raise ValueError(f"Duplicate teacher row: {episode_id}")
                seen.add(episode_id)
                rows.append(payload)
    rows.sort(key=lambda row: str(row["episode_id"]))
    with target.open("w", encoding="utf-8") as writer:
        for row in rows:
            writer.write(json.dumps(row, ensure_ascii=False) + "\n")
    if remove_shards:
        for shard_index in range(num_shards):
            (output / f"{split}_teacher_scores_shard{shard_index:02d}_of_{num_shards:02d}.jsonl").unlink()
    return target


def load_teacher_scores(path: str | Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            payload = json.loads(line)
            result[payload["episode_id"]] = payload["queries"]
    return result
