"""Lazy, canonical PyArrow reader for hash-verified FineWeb Parquet shards."""

from __future__ import annotations

import importlib
from collections.abc import Iterable, Iterator
from pathlib import Path

from .data_lock import SourceDocument


class ParquetDocumentError(RuntimeError):
    """Raised when a pinned shard lacks the locked FineWeb columns."""


def iter_parquet_documents(
    shard_paths: Iterable[str | Path],
    *,
    batch_size: int = 1_024,
) -> Iterator[SourceDocument]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    try:
        parquet = importlib.import_module("pyarrow.parquet")
    except ModuleNotFoundError as error:
        raise ParquetDocumentError(
            "FineWeb preparation requires pinned pyarrow==25.0.0"
        ) from error
    for path_value in sorted(Path(item).resolve(strict=True) for item in shard_paths):
        source = parquet.ParquetFile(path_value)
        names = set(source.schema.names)
        required = {"text", "id", "url"}
        if not required.issubset(names):
            raise ParquetDocumentError(
                f"{path_value} lacks required FineWeb columns {sorted(required - names)}"
            )
        for row_group in range(source.num_row_groups):
            row_index = 0
            for batch in source.iter_batches(
                batch_size=batch_size,
                row_groups=[row_group],
                columns=["text", "id", "url"],
                use_threads=False,
            ):
                payload = batch.to_pydict()
                for text, source_id, source_url in zip(
                    payload["text"],
                    payload["id"],
                    payload["url"],
                    strict=True,
                ):
                    if not isinstance(text, str):
                        raise ParquetDocumentError(
                            f"Non-string text at {path_value}:{row_group}:{row_index}"
                        )
                    yield SourceDocument(
                        text=text,
                        shard_path=path_value.name,
                        row_group=row_group,
                        row_index=row_index,
                        source_id="" if source_id is None else str(source_id),
                        source_url="" if source_url is None else str(source_url),
                    )
                    row_index += 1
