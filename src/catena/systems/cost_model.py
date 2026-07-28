from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class CostModel:
    assimilation_update_ms: float
    assimilation_query_ms: float
    external_retrieve_ms: float
    external_query_ms: float
    full_refresh_ms: float
    snapshot_encode_ms: float = 0.0
    snapshot_query_ms: float = 0.0

    def assimilation_cost(self, queries: int) -> float:
        return self.assimilation_update_ms + queries * self.assimilation_query_ms

    def external_read_cost(self, queries: int) -> float:
        return queries * (self.external_retrieve_ms + self.external_query_ms)

    def cached_snapshot_cost(self, queries: int) -> float:
        return self.external_retrieve_ms + self.snapshot_encode_ms + queries * self.snapshot_query_ms

    def break_even_queries(self, baseline: str = "external_every_query", max_queries: int = 10_000) -> int | None:
        for queries in range(1, max_queries + 1):
            baseline_cost = self.external_read_cost(queries) if baseline == "external_every_query" else self.cached_snapshot_cost(queries)
            if self.assimilation_cost(queries) <= baseline_cost:
                return queries
        return None
