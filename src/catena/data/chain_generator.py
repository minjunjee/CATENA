from __future__ import annotations

import json
import random
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable

from catena.config import load_yaml

from .generator import (
    _apply_transaction,
    _make_closure,
    _make_history,
    _make_initial_state,
    _make_queries,
    _stable_id,
    _target_key,
)
from .schema import ChainEpisode, HistorySegment, Transaction
from .templates import SCHEMA_FAMILIES


def generate_chain_episode(
    *,
    split: str,
    index: int,
    seed: int,
    history_token_target: int,
    chain_length: int,
    domain: str,
    operations: list[str],
    dependency_depth: int,
) -> ChainEpisode:
    rng = random.Random(_stable_id(seed, split, index, domain, chain_length))
    schema_family = rng.choice(SCHEMA_FAMILIES[domain])
    initial = _make_initial_state(domain, schema_family, rng)
    state = dict(initial)
    first_target = _target_key(domain, rng)
    history = _make_history(domain, schema_family, initial, first_target, history_token_target, rng)
    transactions: list[Transaction] = []
    closures = []
    touched: list[str] = []
    for step in range(chain_length):
        operation = rng.choice(operations)
        target = _target_key(domain, rng)
        next_state, old_value, new_value, tx_meta = _apply_transaction(state, operation, target, rng)
        target = str(tx_meta["target"])
        old_version = int(state["version"])
        new_version = int(next_state["version"])
        closure = _make_closure(schema_family, target, old_version, dependency_depth)
        tx = Transaction(
            operation=operation,  # type: ignore[arg-type]
            target=target,
            old_value=old_value,
            new_value=new_value,
            old_version=old_version,
            new_version=new_version,
            valid_from=(date(2026, 1, 1) + timedelta(days=rng.randint(0, 365))).isoformat(),
            invalidates=[item.node_id for item in closure if item.relation == "INVALIDATES"],
            affects=[item.node_id for item in closure],
            scope={k: v for k, v in tx_meta.items() if k != "target"},
        )
        transactions.append(tx)
        closures.append(closure)
        touched.append(target)
        state = next_state

    queries = _make_queries(domain, schema_family, initial, state, touched[-1], rng)
    refresh = list(history)
    for tx, closure in zip(transactions, closures):
        refresh.append(
            HistorySegment(
                segment_id=f"tx-{tx.new_version}",
                kind="verified_transaction",
                text=(
                    f"Verified transaction {tx.operation}: {tx.target} changed from {tx.old_value} "
                    f"to {tx.new_value}; canonical version is now {tx.new_version}."
                ),
                entities=[schema_family, tx.target],
                affected=True,
            )
        )
        for item in closure:
            refresh.append(
                HistorySegment(
                    segment_id=item.node_id,
                    kind="invalidation",
                    text=item.text,
                    entities=[schema_family],
                    affected=True,
                )
            )
    refresh.append(
        HistorySegment(
            segment_id=f"snapshot-{state['version']}",
            kind="canonical_current",
            text="Final canonical state: " + json.dumps(state, sort_keys=True),
            entities=[schema_family],
            affected=True,
        )
    )
    return ChainEpisode(
        chain_id=f"{split}-chain-{_stable_id(seed, index, domain, chain_length)}",
        split=split,
        domain=domain,
        schema_family=schema_family,
        seed=seed,
        history_token_target=history_token_target,
        chain_length=chain_length,
        initial_state=initial,
        final_state=state,
        history_segments=history,
        transactions=transactions,
        closures=closures,
        queries=queries,
        refresh_segments=refresh,
        metadata={"touched_keys": touched},
    )


def _iter(config: dict[str, Any], split: str, count: int) -> Iterable[ChainEpisode]:
    seed = int(config["seed"])
    domains = list(config["domains"])
    operations = list(config["operations"])
    lengths = list(config["history_token_targets"])
    depths = list(config["dependency_depths"])
    chain_lengths = list(
        config.get(f"{split}_chain_lengths", config.get("chain_lengths", [1, 2]))
    )
    for index in range(count):
        rng = random.Random(_stable_id(seed, "chain", split, index))
        yield generate_chain_episode(
            split=split,
            index=index,
            seed=seed,
            history_token_target=rng.choice(lengths),
            chain_length=rng.choice(chain_lengths),
            domain=rng.choice(domains),
            operations=operations,
            dependency_depth=rng.choice(depths),
        )


def generate_chains_from_config(config_path: str | Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    output_dir = Path(config["output_dir"]) / "chains"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"config": config, "files": {}}
    for split, count in {
        "train": int(config.get("num_train", 0)),
        "val": int(config.get("num_val", 0)),
        "test": int(config.get("num_test", 0)),
    }.items():
        path = output_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            written = 0
            for episode in _iter(config, split, count):
                f.write(json.dumps(episode.to_dict(), ensure_ascii=False) + "\n")
                written += 1
        manifest["files"][split] = {"path": str(path), "episodes": written}
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest
