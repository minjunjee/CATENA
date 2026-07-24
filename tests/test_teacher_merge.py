from __future__ import annotations

import json

from catena.training.teacher_cache import merge_teacher_shards


def test_teacher_shard_merge(tmp_path):
    for index, episode_id in enumerate(["b", "a"]):
        path = tmp_path / f"train_teacher_scores_shard{index:02d}_of_02.jsonl"
        path.write_text(
            json.dumps({"episode_id": episode_id, "queries": {}}) + "\n",
            encoding="utf-8",
        )
    merged = merge_teacher_shards(tmp_path, split="train", num_shards=2)
    rows = [json.loads(line) for line in merged.read_text().splitlines()]
    assert [row["episode_id"] for row in rows] == ["a", "b"]
