from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Operation = Literal["SUPERSEDE", "AMEND", "INVALIDATE", "ADD_EXCEPTION"]
QueryKind = Literal[
    "affected_direct",
    "affected_derived",
    "unaffected",
    "old_rule_probe",
    "tool_call",
]


@dataclass(slots=True)
class HistorySegment:
    segment_id: str
    kind: str
    text: str
    entities: list[str] = field(default_factory=list)
    affected: bool = False


@dataclass(slots=True)
class Transaction:
    operation: Operation
    target: str
    old_value: Any
    new_value: Any
    old_version: int
    new_version: int
    valid_from: str
    invalidates: list[str] = field(default_factory=list)
    affects: list[str] = field(default_factory=list)
    scope: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ClosureItem:
    node_id: str
    relation: str
    text: str


@dataclass(slots=True)
class Query:
    query_id: str
    kind: QueryKind
    prompt: str
    candidates: list[str]
    gold_index: int
    affected_keys: list[str] = field(default_factory=list)
    tool_schema: dict[str, Any] | None = None

    @property
    def gold(self) -> str:
        return self.candidates[self.gold_index]


@dataclass(slots=True)
class Episode:
    episode_id: str
    split: str
    domain: str
    schema_family: str
    seed: int
    history_token_target: int
    dependency_depth: int
    query_gap_tokens: int
    initial_state: dict[str, Any]
    current_state: dict[str, Any]
    history_segments: list[HistorySegment]
    transaction: Transaction
    closure: list[ClosureItem]
    queries: list[Query]
    refresh_segments: list[HistorySegment]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Episode":
        return cls(
            episode_id=payload["episode_id"],
            split=payload["split"],
            domain=payload["domain"],
            schema_family=payload["schema_family"],
            seed=int(payload["seed"]),
            history_token_target=int(payload["history_token_target"]),
            dependency_depth=int(payload["dependency_depth"]),
            query_gap_tokens=int(payload["query_gap_tokens"]),
            initial_state=dict(payload["initial_state"]),
            current_state=dict(payload["current_state"]),
            history_segments=[HistorySegment(**x) for x in payload["history_segments"]],
            transaction=Transaction(**payload["transaction"]),
            closure=[ClosureItem(**x) for x in payload["closure"]],
            queries=[Query(**x) for x in payload["queries"]],
            refresh_segments=[HistorySegment(**x) for x in payload["refresh_segments"]],
            metadata=dict(payload.get("metadata", {})),
        )

@dataclass(slots=True)
class ChainEpisode:
    chain_id: str
    split: str
    domain: str
    schema_family: str
    seed: int
    history_token_target: int
    chain_length: int
    initial_state: dict[str, Any]
    final_state: dict[str, Any]
    history_segments: list[HistorySegment]
    transactions: list[Transaction]
    closures: list[list[ClosureItem]]
    queries: list[Query]
    refresh_segments: list[HistorySegment]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ChainEpisode":
        return cls(
            chain_id=payload["chain_id"],
            split=payload["split"],
            domain=payload["domain"],
            schema_family=payload["schema_family"],
            seed=int(payload["seed"]),
            history_token_target=int(payload["history_token_target"]),
            chain_length=int(payload["chain_length"]),
            initial_state=dict(payload["initial_state"]),
            final_state=dict(payload["final_state"]),
            history_segments=[HistorySegment(**x) for x in payload["history_segments"]],
            transactions=[Transaction(**x) for x in payload["transactions"]],
            closures=[[ClosureItem(**item) for item in items] for items in payload["closures"]],
            queries=[Query(**x) for x in payload["queries"]],
            refresh_segments=[HistorySegment(**x) for x in payload["refresh_segments"]],
            metadata=dict(payload.get("metadata", {})),
        )
