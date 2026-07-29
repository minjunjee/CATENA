from __future__ import annotations

import random
from dataclasses import dataclass

from catena.core.schema import Operation


@dataclass(slots=True)
class TransactionStep:
    entity: str
    operation: Operation
    old_value: str
    new_value: str


@dataclass(slots=True)
class TransactionChain:
    chain_id: str
    steps: list[TransactionStep]
    query_count: int
    distractor_count: int
    rollback_target: int | None


def generate_chains(
    *,
    count: int,
    lengths: list[int],
    seed: int,
    query_counts: list[int],
    include_rollback: bool = True,
) -> list[TransactionChain]:
    rng = random.Random(seed)
    chains: list[TransactionChain] = []
    for index in range(count):
        length = rng.choice(lengths)
        entity = f"entity_{rng.randrange(64):03d}"
        current = f"v_{rng.randrange(32):02d}"
        steps: list[TransactionStep] = []
        for _ in range(length):
            operation = rng.choice([Operation.ADD, Operation.INVALIDATE, Operation.SUPERSEDE])
            new_value = f"v_{rng.randrange(32):02d}"
            steps.append(
                TransactionStep(
                    entity=entity,
                    operation=operation,
                    old_value=current,
                    new_value=new_value,
                )
            )
            if operation is Operation.SUPERSEDE:
                current = new_value
        rollback_target = rng.randrange(length) if include_rollback and rng.random() < 0.2 else None
        chains.append(
            TransactionChain(
                chain_id=f"chain-{seed}-{index}",
                steps=steps,
                query_count=rng.choice(query_counts),
                distractor_count=rng.choice([0, 4, 16]),
                rollback_target=rollback_target,
            )
        )
    return chains
