from __future__ import annotations

from pathlib import Path
from itertools import chain

from catena.config import load_yaml
from catena.models.factory import load_model

from .common import load_split, run_policy_evaluation


def run(
    config_path: str,
    *,
    device: str = "cuda",
    max_episodes: int | None = None,
    shard_index: int = 0,
    num_shards: int = 1,
):
    config = load_yaml(config_path)
    model = load_model(str(config["model"]), device=device)
    if "data_dirs" in config:
        data_dirs = [str(path) for path in config["data_dirs"]]
    else:
        data_dirs = [str(config["data_dir"])]
    policies = list(config["policies"])
    episodes = chain.from_iterable(load_split(path, "test") for path in data_dirs)
    return run_policy_evaluation(
        model=model,
        episodes=episodes,
        policies=policies,
        output_dir=Path(config["output_dir"]),
        max_episodes=max_episodes,
        config=config,
        shard_index=shard_index,
        num_shards=num_shards,
    )
