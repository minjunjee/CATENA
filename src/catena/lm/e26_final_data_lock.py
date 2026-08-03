"""Read-only FineWeb source lock and 32K capacity audit for E26 Final.

This module is additive.  It deliberately does not reuse the old E26 16K
token memmaps or mutate the original SQLite document index.  The immutable raw
index, source receipts, and zero-tolerance exclusion set are checked before a
content-SHA-ordered general-train iterator can be opened.
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Protocol

from catena.core.provenance_v61 import (
    SHA256_PATTERN,
    read_json_object_strict,
    sha256_canonical_json,
    sha256_file,
    write_json_strict,
)

from .data_lock import ContentSplit, bucket_for_sha256, normalize_document, split_for_bucket
from .e26_final_tokenizer import (
    DEFAULT_TOKENIZER_EXPECTATION,
    DOCUMENT_SEPARATOR_POLICY,
    E26FinalTokenizerExpectation,
    validate_e26_final_tokenizer_receipt,
)

DATA_SOURCE_RECEIPT_SCHEMA: Final = "catena-e26-final-data-source-lock-v1"
DATA_SOURCE_RECEIPT_TYPE: Final = "E26_FINAL_FINEWEB_READ_ONLY_SOURCE_LOCK"
CAPACITY_RECEIPT_SCHEMA: Final = "catena-e26-final-general-capacity-v1"
CAPACITY_RECEIPT_TYPE: Final = "E26_FINAL_32K_GENERAL_CAPACITY"
HELDOUT_DOMAIN_RECEIPT_SCHEMA: Final = "catena-e26-final-heldout-domain-lock-v1"
HELDOUT_DOMAIN_RECEIPT_TYPE: Final = "E26_FINAL_OUTCOME_INDEPENDENT_HELDOUT_DOMAIN"
FINEWEB_DATASET_ID: Final = "HuggingFaceFW/fineweb-edu"
FINEWEB_SUBSET: Final = "sample-10BT"
FINEWEB_REVISION: Final = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
TRAIN_SELECTION_ORDER: Final = "CONTENT_SHA256_ASCENDING"
EXCLUSION_POLICY: Final = "EXCLUDE_ALL_FLAGGED_TRAIN_DOCUMENTS"
HELDOUT_DOMAIN_CANDIDATES: Final = ("compliance", "inventory", "logistics")
HELDOUT_DOMAIN_SELECTION: Final = "LEXICOGRAPHIC_FIRST_V1"
HELDOUT_DOMAIN: Final = "compliance"
# Large enough to amortize SentencePiece worker-pool setup while remaining a
# bounded read-only scan (roughly tens to low hundreds of MiB for this corpus).
CAPACITY_ENCODING_BATCH_DOCUMENTS: Final = 8_192
CAPACITY_ENCODING_THREADS: Final = 32


class E26FinalDataLockError(RuntimeError):
    """Raised when E26 Final source data or a derived receipt is not immutable."""


@dataclass(frozen=True, slots=True)
class BoundFileExpectation:
    path: str
    sha256: str

    def __post_init__(self) -> None:
        if not SHA256_PATTERN.fullmatch(self.sha256):
            raise ValueError("Bound file SHA-256 must be 64 lowercase hex characters")


@dataclass(frozen=True, slots=True)
class E26FinalDataSourceExpectation:
    """Exact source bytes and population invariants inherited by E26 Final."""

    sqlite: BoundFileExpectation
    dedup_receipt: BoundFileExpectation
    source_inventory: BoundFileExpectation
    download_receipt: BoundFileExpectation
    exclusion_manifest: BoundFileExpectation
    source_revision: str = FINEWEB_REVISION
    dataset_id: str = FINEWEB_DATASET_ID
    subset: str = FINEWEB_SUBSET
    document_selection_sha256: str = (
        "0ea86446b376a83abb64c786d4284201668cd695a45afe3e0f96e4fd47f64f2e"
    )
    expected_unique_documents: int = 3_053_890
    expected_documents_seen: int = 3_093_101
    expected_exact_duplicates: int = 39_211
    expected_exclusion_count: int = 541
    expected_excluded_utf8_bytes: int = 3_892_332
    expected_selected_shard_indices: tuple[int, ...] = (0, 2, 4, 9, 13)
    expected_split_counts: tuple[tuple[str, int], ...] = (
        (ContentSplit.GENERAL_TEST.value, 6_073),
        (ContentSplit.GENERAL_TRAIN.value, 3_011_273),
        (ContentSplit.GENERAL_VALIDATION.value, 5_979),
        (ContentSplit.TOKENIZER_ONLY.value, 30_565),
    )
    expected_split_utf8_bytes: tuple[tuple[str, int], ...] = (
        (ContentSplit.GENERAL_TEST.value, 28_586_930),
        (ContentSplit.GENERAL_TRAIN.value, 14_370_299_182),
        (ContentSplit.GENERAL_VALIDATION.value, 28_085_380),
        (ContentSplit.TOKENIZER_ONLY.value, 147_653_341),
    )

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["expected_selected_shard_indices"] = list(self.expected_selected_shard_indices)
        payload["expected_split_counts"] = dict(self.expected_split_counts)
        payload["expected_split_utf8_bytes"] = dict(self.expected_split_utf8_bytes)
        return payload


DEFAULT_DATA_SOURCE_EXPECTATION: Final = E26FinalDataSourceExpectation(
    sqlite=BoundFileExpectation(
        "/data/minjun_dev/CATENA/e26_data_v1/document_index/expansion1/documents.sqlite3",
        "6f835450270811b0c08dd70df285fa5b64f99598bfc341aa69d389e7377c8f0a",
    ),
    dedup_receipt=BoundFileExpectation(
        "/data/minjun_dev/CATENA/e26_data_v1/document_index/expansion1/dedup_receipt.json",
        "8b69579d93e76b0184dec5b0b3cac818384e1c1627378af86eae2a1c439e6dca",
    ),
    source_inventory=BoundFileExpectation(
        "/data/minjun_dev/CATENA/e26_data_v1/source_manifest/expansion1/fineweb_inventory.json",
        "990bb95b7ed09913bd22e6a2d81cc42ed785edb86a4163060d8d43166e825753",
    ),
    download_receipt=BoundFileExpectation(
        "/data/minjun_dev/CATENA/e26_data_v1/source_manifest/expansion1/"
        "fineweb_download_receipt.json",
        "ed0044dfcdac39e864130b3caa2285c254931925ca69a4b67ffb599811fbb4a6",
    ),
    exclusion_manifest=BoundFileExpectation(
        "/data/minjun_dev/CATENA/e26_data_v2_zero_tolerance_6c6cb0fce46f/"
        "initial_exclusion_manifest.json",
        "76a81f018abe61929895dcfecffe4b2de605d1267f145a10077902c7e7c62521",
    ),
)


@dataclass(frozen=True, slots=True)
class E26FinalGeneralDocument:
    content_sha256: str
    content_sha512: str
    normalized_utf8_bytes: int
    shard_path: str
    row_group: int
    row_index: int
    source_id_sha256: str
    source_url_sha256: str
    text: str


class E26FinalTokenEncoder(Protocol):
    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> Sequence[int]: ...

    def encode_batch(
        self,
        texts: Sequence[str],
        *,
        num_threads: int,
    ) -> Sequence[Sequence[int]]: ...


_EXPECTED_SQLITE_COLUMNS: Final = (
    ("content_sha256", "TEXT", 1),
    ("content_sha512", "TEXT", 0),
    ("split", "TEXT", 0),
    ("bucket", "INTEGER", 0),
    ("normalized_utf8_bytes", "INTEGER", 0),
    ("shard_path", "TEXT", 0),
    ("row_group", "INTEGER", 0),
    ("row_index", "INTEGER", 0),
    ("source_id_sha256", "TEXT", 0),
    ("source_url_sha256", "TEXT", 0),
    ("normalized_utf8", "BLOB", 0),
)


def _regular_file(binding: BoundFileExpectation, label: str) -> Path:
    candidate = Path(binding.path).expanduser()
    if candidate.is_symlink():
        raise E26FinalDataLockError(f"{label} must not be a symlink: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as error:
        raise E26FinalDataLockError(f"{label} is missing: {candidate}") from error
    if not resolved.is_file() or resolved.is_symlink():
        raise E26FinalDataLockError(f"{label} is not a regular file: {resolved}")
    observed = sha256_file(resolved)
    if observed != binding.sha256:
        raise E26FinalDataLockError(
            f"{label} SHA-256 changed: observed={observed}, expected={binding.sha256}"
        )
    return resolved


def _bound(path: Path, verified_sha256: str) -> dict[str, Any]:
    """Record a digest already verified by :func:`_regular_file`.

    The raw index is roughly 18.6 GB, so re-reading it merely to serialize the
    same expected digest would make one admission pass needlessly scan it twice.
    """

    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": verified_sha256,
    }


def _without_internal_sha(payload: Mapping[str, Any], field: str) -> tuple[Any, str | None]:
    copied = deepcopy(dict(payload))
    claimed = copied.pop(field, None)
    return copied, claimed if isinstance(claimed, str) else None


def _readonly_connection(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"{path.as_uri()}?mode=ro&immutable=1", uri=True)
    connection.execute("PRAGMA query_only=ON")
    return connection


def _database_summary(path: Path) -> dict[str, Any]:
    with _readonly_connection(path) as connection:
        table = connection.execute("PRAGMA table_info(documents)").fetchall()
        observed_columns = tuple((str(row[1]), str(row[2]).upper(), int(row[5])) for row in table)
        if observed_columns != _EXPECTED_SQLITE_COLUMNS:
            raise E26FinalDataLockError("Raw SQLite documents schema changed")
        split_rows = connection.execute(
            "SELECT split, COUNT(*), COALESCE(SUM(normalized_utf8_bytes), 0) "
            "FROM documents GROUP BY split ORDER BY split"
        ).fetchall()
        indexes = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = 'documents'"
            )
        }
        query_only = int(connection.execute("PRAGMA query_only").fetchone()[0])
    return {
        "query_only": query_only == 1,
        "schema_columns": [list(row) for row in observed_columns],
        "split_counts": {str(row[0]): int(row[1]) for row in split_rows},
        "split_normalized_utf8_bytes": {str(row[0]): int(row[2]) for row in split_rows},
        "unique_documents": sum(int(row[1]) for row in split_rows),
        "documents_split_sha_index_present": "documents_split_sha" in indexes,
    }


def _load_exclusions(
    path: Path,
    *,
    expectation: E26FinalDataSourceExpectation,
) -> tuple[tuple[str, ...], dict[str, Any]]:
    payload = read_json_object_strict(path)
    without_sha, claimed = _without_internal_sha(payload, "exclusion_manifest_sha256")
    if claimed != sha256_canonical_json(without_sha):
        raise E26FinalDataLockError("Exclusion manifest internal SHA-256 changed")
    raw = payload.get("exclusions")
    if not isinstance(raw, list):
        raise E26FinalDataLockError("Exclusion manifest lacks an exclusion list")
    rows: list[dict[str, Any]] = []
    hashes: list[str] = []
    for row in raw:
        if not isinstance(row, dict):
            raise E26FinalDataLockError("Exclusion manifest contains a malformed row")
        content_sha = row.get("content_sha256")
        byte_count = row.get("normalized_utf8_bytes")
        if (
            not isinstance(content_sha, str)
            or not SHA256_PATTERN.fullmatch(content_sha)
            or isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count < 0
            or row.get("split") != ContentSplit.GENERAL_TRAIN.value
        ):
            raise E26FinalDataLockError("Exclusion manifest row violates the train-only lock")
        hashes.append(content_sha)
        rows.append(dict(row))
    if hashes != sorted(set(hashes)):
        raise E26FinalDataLockError("Exclusion hashes must be unique and sorted")
    if (
        len(hashes) != expectation.expected_exclusion_count
        or payload.get("unique_train_exclusion_count") != len(hashes)
        or payload.get("flagged_pair_count") != len(hashes)
        or payload.get("policy") != EXCLUSION_POLICY
        or payload.get("outcome_independent") is not True
        or payload.get("human_labels_used") is not False
        or payload.get("excluded_normalized_utf8_bytes") != expectation.expected_excluded_utf8_bytes
        or payload.get("exclusion_sorted_json_sha256") != sha256_canonical_json(hashes)
    ):
        raise E26FinalDataLockError("Exclusion manifest population contract changed")
    return tuple(hashes), {"payload": payload, "rows": rows}


def _validate_exclusions_in_database(
    sqlite_path: Path,
    exclusions: Sequence[str],
) -> dict[str, Any]:
    observed: dict[str, tuple[str, int, bytes]] = {}
    with _readonly_connection(sqlite_path) as connection:
        for start in range(0, len(exclusions), 400):
            part = tuple(exclusions[start : start + 400])
            placeholders = ",".join("?" for _ in part)
            query = (
                "SELECT content_sha256, split, normalized_utf8_bytes, normalized_utf8 "
                f"FROM documents WHERE content_sha256 IN ({placeholders})"
            )
            for content_sha, split, byte_count, raw_value in connection.execute(query, part):
                raw = raw_value if isinstance(raw_value, bytes) else bytes(raw_value)
                observed[str(content_sha)] = (str(split), int(byte_count), raw)
    if set(observed) != set(exclusions):
        raise E26FinalDataLockError("Not every exclusion exists in the raw SQLite index")
    normalized_bytes = 0
    raw_digest = hashlib.sha256()
    for content_sha in exclusions:
        split, byte_count, raw = observed[content_sha]
        if (
            split != ContentSplit.GENERAL_TRAIN.value
            or len(raw) != byte_count
            or hashlib.sha256(raw).hexdigest() != content_sha
        ):
            raise E26FinalDataLockError("Excluded raw bytes fail content identity")
        normalized_bytes += byte_count
        raw_digest.update(bytes.fromhex(content_sha))
        raw_digest.update(byte_count.to_bytes(8, "big"))
        raw_digest.update(raw)
    return {
        "verified_count": len(observed),
        "normalized_utf8_bytes": normalized_bytes,
        "raw_boundary_digest_sha256": raw_digest.hexdigest(),
    }


def _validate_source_json(
    *,
    inventory: Mapping[str, Any],
    download: Mapping[str, Any],
    dedup: Mapping[str, Any],
    sqlite_path: Path,
    expectation: E26FinalDataSourceExpectation,
) -> dict[str, bool]:
    inventory_body, inventory_claimed = _without_internal_sha(inventory, "inventory_sha256")
    download_body, download_claimed = _without_internal_sha(download, "receipt_sha256")
    selected = list(expectation.expected_selected_shard_indices)
    return {
        "inventory_internal_sha_exact": inventory_claimed == sha256_canonical_json(inventory_body),
        "download_internal_sha_exact": download_claimed == sha256_canonical_json(download_body),
        "inventory_source_exact": (
            inventory.get("dataset_id") == expectation.dataset_id
            and inventory.get("subset") == expectation.subset
            and inventory.get("revision") == expectation.source_revision
        ),
        "download_source_exact": (
            download.get("dataset_id") == expectation.dataset_id
            and download.get("subset") == expectation.subset
            and download.get("revision") == expectation.source_revision
        ),
        "selected_shards_exact": download.get("selected_indices") == selected,
        "download_all_verified": download.get("all_verified") is True
        and all(
            isinstance(row, Mapping) and row.get("verified") is True
            for row in download.get("shards", ())
        ),
        "dedup_sqlite_binding_exact": (
            dedup.get("sqlite_path") == str(sqlite_path)
            and dedup.get("sqlite_sha256") == expectation.sqlite.sha256
        ),
        "dedup_population_exact": (
            dedup.get("document_selection_sha256") == expectation.document_selection_sha256
            and dedup.get("manifest_sha256") == expectation.document_selection_sha256
            and dedup.get("documents_seen") == expectation.expected_documents_seen
            and dedup.get("unique_documents") == expectation.expected_unique_documents
            and dedup.get("exact_duplicates") == expectation.expected_exact_duplicates
            and dedup.get("split_counts") == dict(expectation.expected_split_counts)
        ),
    }


def build_e26_final_data_source_receipt(
    *,
    expectation: E26FinalDataSourceExpectation = DEFAULT_DATA_SOURCE_EXPECTATION,
) -> dict[str, Any]:
    """Bind the immutable raw corpus and its monotonic train-only exclusions."""

    sqlite_path = _regular_file(expectation.sqlite, "raw document SQLite")
    dedup_path = _regular_file(expectation.dedup_receipt, "dedup receipt")
    inventory_path = _regular_file(expectation.source_inventory, "source inventory")
    download_path = _regular_file(expectation.download_receipt, "download receipt")
    exclusion_path = _regular_file(expectation.exclusion_manifest, "exclusion manifest")

    dedup = read_json_object_strict(dedup_path)
    inventory = read_json_object_strict(inventory_path)
    download = read_json_object_strict(download_path)
    exclusions, exclusion_detail = _load_exclusions(
        exclusion_path,
        expectation=expectation,
    )
    database = _database_summary(sqlite_path)
    exclusion_database = _validate_exclusions_in_database(sqlite_path, exclusions)
    source_checks = _validate_source_json(
        inventory=inventory,
        download=download,
        dedup=dedup,
        sqlite_path=sqlite_path,
        expectation=expectation,
    )
    checks = {
        **source_checks,
        "sqlite_query_only": database["query_only"] is True,
        "sqlite_expected_index_present": database["documents_split_sha_index_present"] is True,
        "sqlite_split_counts_exact": database["split_counts"]
        == dict(expectation.expected_split_counts),
        "sqlite_split_bytes_exact": database["split_normalized_utf8_bytes"]
        == dict(expectation.expected_split_utf8_bytes),
        "sqlite_unique_count_exact": database["unique_documents"]
        == expectation.expected_unique_documents,
        "exclusion_count_exact": len(exclusions) == expectation.expected_exclusion_count,
        "exclusions_exist_as_train": exclusion_database["verified_count"] == len(exclusions),
        "exclusion_bytes_exact": exclusion_database["normalized_utf8_bytes"]
        == expectation.expected_excluded_utf8_bytes,
    }
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise E26FinalDataLockError(f"E26 Final source admission failed: {failed}")

    eligible_train_count = dict(expectation.expected_split_counts)[
        ContentSplit.GENERAL_TRAIN.value
    ] - len(exclusions)
    eligible_train_bytes = dict(expectation.expected_split_utf8_bytes)[
        ContentSplit.GENERAL_TRAIN.value
    ] - int(exclusion_database["normalized_utf8_bytes"])
    payload: dict[str, Any] = {
        "schema_version": DATA_SOURCE_RECEIPT_SCHEMA,
        "manifest_type": DATA_SOURCE_RECEIPT_TYPE,
        "scientific_evidence": False,
        "evidence_tier": "SCIENTIFIC_INPUT_PROVENANCE",
        "claim_ceiling": "RAW_CORPUS_POPULATION_AND_ORDER_ONLY",
        "dataset_id": expectation.dataset_id,
        "subset": expectation.subset,
        "source_revision": expectation.source_revision,
        "train_selection_order": TRAIN_SELECTION_ORDER,
        "read_policy": "SQLITE_MODE_RO_IMMUTABLE_QUERY_ONLY",
        "inputs": {
            "sqlite": _bound(sqlite_path, expectation.sqlite.sha256),
            "dedup_receipt": _bound(dedup_path, expectation.dedup_receipt.sha256),
            "source_inventory": _bound(inventory_path, expectation.source_inventory.sha256),
            "download_receipt": _bound(download_path, expectation.download_receipt.sha256),
            "exclusion_manifest": _bound(exclusion_path, expectation.exclusion_manifest.sha256),
        },
        "database": database,
        "zero_tolerance_exclusions": {
            "policy": EXCLUSION_POLICY,
            "count": len(exclusions),
            "hashes_sha256": sha256_canonical_json(list(exclusions)),
            "manifest_internal_sha256": exclusion_detail["payload"].get(
                "exclusion_manifest_sha256"
            ),
            "normalized_utf8_bytes": exclusion_database["normalized_utf8_bytes"],
            "raw_boundary_digest_sha256": exclusion_database["raw_boundary_digest_sha256"],
        },
        "eligible_general_train": {
            "document_count": eligible_train_count,
            "normalized_utf8_bytes": eligible_train_bytes,
            "token_count_32k": None,
            "token_capacity_audit_required": True,
        },
        "hard_checks": dict(sorted(checks.items())),
        "scientific_main_started": False,
        "passed": True,
    }
    payload["receipt_sha256"] = sha256_canonical_json(payload)
    return payload


def validate_e26_final_data_source_receipt(
    payload: Mapping[str, Any],
    *,
    expectation: E26FinalDataSourceExpectation = DEFAULT_DATA_SOURCE_EXPECTATION,
    verify_bound_inputs: bool = True,
) -> dict[str, Any]:
    """Validate receipt integrity and, by default, reconstruct it from raw bytes."""

    normalized = deepcopy(dict(payload))
    claimed = normalized.pop("receipt_sha256", None)
    if not isinstance(claimed, str) or not SHA256_PATTERN.fullmatch(claimed):
        raise E26FinalDataLockError("Data-source receipt lacks a valid SHA-256")
    if claimed != sha256_canonical_json(normalized):
        raise E26FinalDataLockError("Data-source receipt SHA-256 changed")
    normalized["receipt_sha256"] = claimed
    if (
        normalized.get("schema_version") != DATA_SOURCE_RECEIPT_SCHEMA
        or normalized.get("manifest_type") != DATA_SOURCE_RECEIPT_TYPE
        or normalized.get("scientific_evidence") is not False
        or normalized.get("source_revision") != expectation.source_revision
        or normalized.get("train_selection_order") != TRAIN_SELECTION_ORDER
        or normalized.get("scientific_main_started") is not False
        or normalized.get("passed") is not True
    ):
        raise E26FinalDataLockError("Data-source receipt identity changed")
    checks = normalized.get("hard_checks")
    if (
        not isinstance(checks, Mapping)
        or not checks
        or not all(value is True for value in checks.values())
    ):
        raise E26FinalDataLockError("Data-source hard-check map is not all PASS")
    if verify_bound_inputs:
        reconstructed = build_e26_final_data_source_receipt(expectation=expectation)
        if reconstructed != normalized:
            raise E26FinalDataLockError("Data-source receipt differs from reconstruction")
    return normalized


def write_e26_final_data_source_receipt(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    expectation: E26FinalDataSourceExpectation = DEFAULT_DATA_SOURCE_EXPECTATION,
) -> Path:
    """Write one immutable source receipt, refusing existing paths and symlinks."""

    destination = Path(path).expanduser()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite data-source receipt: {destination}")
    validated = validate_e26_final_data_source_receipt(payload, expectation=expectation)
    write_json_strict(destination, validated)
    return destination


def _exclusions_from_receipt(
    receipt: Mapping[str, Any],
    *,
    expectation: E26FinalDataSourceExpectation,
) -> tuple[str, ...]:
    inputs = receipt.get("inputs")
    if not isinstance(inputs, Mapping):
        raise E26FinalDataLockError("Data-source receipt lacks input bindings")
    record = inputs.get("exclusion_manifest")
    if not isinstance(record, Mapping) or record.get("path") != str(
        Path(expectation.exclusion_manifest.path).expanduser().resolve(strict=True)
    ):
        raise E26FinalDataLockError("Data-source receipt binds another exclusion manifest")
    path = _regular_file(expectation.exclusion_manifest, "exclusion manifest")
    exclusions, _detail = _load_exclusions(path, expectation=expectation)
    return exclusions


def iter_e26_final_general_train(
    source_receipt: Mapping[str, Any],
    *,
    expectation: E26FinalDataSourceExpectation = DEFAULT_DATA_SOURCE_EXPECTATION,
    verify_bound_inputs: bool = True,
) -> Iterator[E26FinalGeneralDocument]:
    """Yield eligible train documents in content-SHA order from a read-only DB."""

    receipt = validate_e26_final_data_source_receipt(
        source_receipt,
        expectation=expectation,
        verify_bound_inputs=verify_bound_inputs,
    )
    exclusions = frozenset(_exclusions_from_receipt(receipt, expectation=expectation))
    sqlite_path = _regular_file(expectation.sqlite, "raw document SQLite")
    query = (
        "SELECT content_sha256, content_sha512, normalized_utf8_bytes, shard_path, "
        "row_group, row_index, source_id_sha256, source_url_sha256, bucket, normalized_utf8 "
        "FROM documents WHERE split = ? ORDER BY content_sha256"
    )
    previous_sha: str | None = None
    with _readonly_connection(sqlite_path) as connection:
        for row in connection.execute(query, (ContentSplit.GENERAL_TRAIN.value,)):
            content_sha = str(row[0])
            if content_sha in exclusions:
                continue
            raw_value = row[9]
            raw = raw_value if isinstance(raw_value, bytes) else bytes(raw_value)
            text = raw.decode("utf-8", errors="strict")
            if (
                previous_sha is not None
                and content_sha <= previous_sha
                or len(raw) != int(row[2])
                or hashlib.sha256(raw).hexdigest() != content_sha
                or hashlib.sha512(raw).hexdigest() != str(row[1])
                or normalize_document(text).encode("utf-8") != raw
                or bucket_for_sha256(content_sha) != int(row[8])
                or split_for_bucket(int(row[8])) is not ContentSplit.GENERAL_TRAIN
            ):
                raise E26FinalDataLockError("General-train row identity or order changed")
            previous_sha = content_sha
            yield E26FinalGeneralDocument(
                content_sha256=content_sha,
                content_sha512=str(row[1]),
                normalized_utf8_bytes=int(row[2]),
                shard_path=str(row[3]),
                row_group=int(row[4]),
                row_index=int(row[5]),
                source_id_sha256=str(row[6]),
                source_url_sha256=str(row[7]),
                text=text,
            )


def _token_ids(encoding: Sequence[int], *, vocab_size: int) -> list[int]:
    if isinstance(encoding, (str, bytes, bytearray)):
        raise E26FinalDataLockError("Tokenizer returned bytes/text instead of token IDs")
    values = [int(value) for value in encoding]
    if any(value < 0 or value >= vocab_size for value in values):
        raise E26FinalDataLockError(f"Tokenizer emitted an ID outside [0,{vocab_size})")
    return values


def count_e26_final_general_train_capacity(
    *,
    source_receipt: Mapping[str, Any],
    tokenizer_receipt: Mapping[str, Any],
    tokenizer: E26FinalTokenEncoder,
    minimum_required_tokens: int,
    source_expectation: E26FinalDataSourceExpectation = DEFAULT_DATA_SOURCE_EXPECTATION,
    tokenizer_expectation: E26FinalTokenizerExpectation = DEFAULT_TOKENIZER_EXPECTATION,
    verify_bound_inputs: bool = True,
    encoding_batch_documents: int = CAPACITY_ENCODING_BATCH_DOCUMENTS,
    encoding_threads: int = CAPACITY_ENCODING_THREADS,
) -> dict[str, Any]:
    """Count the full eligible corpus under the exact 32K tokenizer.

    One EOS token is appended after every document.  The scan never writes a
    token file and never stops early, so the receipt attests total available
    capacity rather than merely the requested prefix.
    """

    if isinstance(minimum_required_tokens, bool) or minimum_required_tokens < 1:
        raise ValueError("minimum_required_tokens must be a positive integer")
    if (
        isinstance(encoding_batch_documents, bool)
        or not isinstance(encoding_batch_documents, int)
        or encoding_batch_documents < 1
        or isinstance(encoding_threads, bool)
        or not isinstance(encoding_threads, int)
        or encoding_threads < 1
    ):
        raise ValueError("Capacity encoding batch size and thread count must be positive integers")
    source = validate_e26_final_data_source_receipt(
        source_receipt,
        expectation=source_expectation,
        verify_bound_inputs=verify_bound_inputs,
    )
    tokenizer_lock = validate_e26_final_tokenizer_receipt(
        tokenizer_receipt,
        expectation=tokenizer_expectation,
        verify_local_files=verify_bound_inputs,
    )
    content_tokens = 0
    documents = 0
    token_min = tokenizer_expectation.eos_token_id
    token_max = tokenizer_expectation.eos_token_id
    order_digest = hashlib.sha256()
    pending: list[E26FinalGeneralDocument] = []

    def consume(batch: Sequence[E26FinalGeneralDocument]) -> None:
        nonlocal content_tokens, documents, token_min, token_max
        encoded = tokenizer.encode_batch(
            [document.text for document in batch],
            num_threads=encoding_threads,
        )
        if len(encoded) != len(batch):
            raise E26FinalDataLockError("Tokenizer changed capacity-scan batch cardinality")
        for document, raw_ids in zip(batch, encoded, strict=True):
            ids = _token_ids(raw_ids, vocab_size=tokenizer_expectation.vocab_size)
            documents += 1
            content_tokens += len(ids)
            if ids:
                token_min = min(token_min, min(ids))
                token_max = max(token_max, max(ids))
            order_digest.update(bytes.fromhex(document.content_sha256))
            order_digest.update(len(ids).to_bytes(8, "big"))

    for document in iter_e26_final_general_train(
        source,
        expectation=source_expectation,
        verify_bound_inputs=False,
    ):
        pending.append(document)
        if len(pending) == encoding_batch_documents:
            consume(pending)
            pending.clear()
    if pending:
        consume(pending)
    separator_tokens = documents
    total_tokens = content_tokens + separator_tokens
    eligible = source.get("eligible_general_train")
    if not isinstance(eligible, Mapping) or documents != eligible.get("document_count"):
        raise E26FinalDataLockError("Capacity scan document count differs from source lock")
    passed = total_tokens >= minimum_required_tokens
    payload: dict[str, Any] = {
        "schema_version": CAPACITY_RECEIPT_SCHEMA,
        "manifest_type": CAPACITY_RECEIPT_TYPE,
        "scientific_evidence": False,
        "evidence_tier": "SCIENTIFIC_INPUT_PROVENANCE",
        "claim_ceiling": "GENERAL_TRAIN_TOKEN_CAPACITY_ONLY",
        "source_receipt_sha256": source["receipt_sha256"],
        "tokenizer_receipt_sha256": tokenizer_lock["receipt_sha256"],
        "tokenizer_revision": tokenizer_expectation.revision,
        "tokenizer_vocab_size": tokenizer_expectation.vocab_size,
        "document_separator_policy": DOCUMENT_SEPARATOR_POLICY,
        "document_separator_id": tokenizer_expectation.document_separator_id,
        "train_selection_order": TRAIN_SELECTION_ORDER,
        "capacity_encoding": {
            "algorithm": "ORDERED_SENTENCEPIECE_BATCH_V1",
            "batch_documents": encoding_batch_documents,
            "threads": encoding_threads,
            "implicit_bos": False,
            "implicit_eos": False,
        },
        "document_count": documents,
        "document_order_and_token_count_sha256": order_digest.hexdigest(),
        "content_token_count": content_tokens,
        "separator_token_count": separator_tokens,
        "total_token_count": total_tokens,
        "minimum_required_tokens": minimum_required_tokens,
        "token_id_min": token_min,
        "token_id_max": token_max,
        "capacity_eligible": passed,
        "disposition": "PASS" if passed else "BLOCKED_SOURCE_CAPACITY",
        "scientific_main_started": False,
    }
    payload["receipt_sha256"] = sha256_canonical_json(payload)
    return validate_e26_final_capacity_receipt(payload)


def validate_e26_final_capacity_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(dict(payload))
    claimed = normalized.pop("receipt_sha256", None)
    if not isinstance(claimed, str) or claimed != sha256_canonical_json(normalized):
        raise E26FinalDataLockError("Capacity receipt SHA-256 changed")
    normalized["receipt_sha256"] = claimed
    if (
        normalized.get("schema_version") != CAPACITY_RECEIPT_SCHEMA
        or normalized.get("manifest_type") != CAPACITY_RECEIPT_TYPE
        or normalized.get("scientific_evidence") is not False
        or normalized.get("scientific_main_started") is not False
        or normalized.get("document_separator_policy") != DOCUMENT_SEPARATOR_POLICY
    ):
        raise E26FinalDataLockError("Capacity receipt identity changed")
    integers = (
        "document_count",
        "content_token_count",
        "separator_token_count",
        "total_token_count",
        "minimum_required_tokens",
        "token_id_min",
        "token_id_max",
    )
    if any(
        isinstance(normalized.get(field), bool) or not isinstance(normalized.get(field), int)
        for field in integers
    ):
        raise E26FinalDataLockError("Capacity receipt counters must be integers")
    expected_total = normalized["content_token_count"] + normalized["separator_token_count"]
    expected_pass = expected_total >= normalized["minimum_required_tokens"]
    if (
        normalized["total_token_count"] != expected_total
        or normalized["separator_token_count"] != normalized["document_count"]
        or normalized.get("capacity_eligible") is not expected_pass
        or normalized.get("disposition") != ("PASS" if expected_pass else "BLOCKED_SOURCE_CAPACITY")
        or normalized["token_id_min"] < 0
        or normalized["token_id_max"] >= int(normalized.get("tokenizer_vocab_size", -1))
    ):
        raise E26FinalDataLockError("Capacity receipt arithmetic or disposition changed")
    for field in (
        "source_receipt_sha256",
        "tokenizer_receipt_sha256",
        "document_order_and_token_count_sha256",
    ):
        value = normalized.get(field)
        if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
            raise E26FinalDataLockError(f"Capacity receipt lacks {field}")
    return normalized


def write_e26_final_capacity_receipt(
    path: str | Path,
    payload: Mapping[str, Any],
) -> Path:
    destination = Path(path).expanduser()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite capacity receipt: {destination}")
    validated = validate_e26_final_capacity_receipt(payload)
    write_json_strict(destination, validated)
    return destination


def build_e26_final_heldout_domain_receipt() -> dict[str, Any]:
    """Lock the held-out domain without inspecting model or evaluation outcomes."""

    candidates = tuple(sorted(HELDOUT_DOMAIN_CANDIDATES))
    selected = min(candidates)
    if candidates != HELDOUT_DOMAIN_CANDIDATES or selected != HELDOUT_DOMAIN:
        raise E26FinalDataLockError("Held-out domain constants violate lexicographic selection")
    payload: dict[str, Any] = {
        "schema_version": HELDOUT_DOMAIN_RECEIPT_SCHEMA,
        "manifest_type": HELDOUT_DOMAIN_RECEIPT_TYPE,
        "scientific_evidence": False,
        "evidence_tier": "SCIENTIFIC_INPUT_PROVENANCE",
        "claim_ceiling": "HELDOUT_DOMAIN_SELECTION_ONLY",
        "candidate_domains": list(candidates),
        "selection_algorithm": HELDOUT_DOMAIN_SELECTION,
        "selected_domain": selected,
        "model_outcomes_accessed": False,
        "evaluation_outcomes_accessed": False,
        "main_test_opened": False,
        "scientific_main_started": False,
        "passed": True,
    }
    payload["receipt_sha256"] = sha256_canonical_json(payload)
    return validate_e26_final_heldout_domain_receipt(payload)


def validate_e26_final_heldout_domain_receipt(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = deepcopy(dict(payload))
    claimed = normalized.pop("receipt_sha256", None)
    if not isinstance(claimed, str) or claimed != sha256_canonical_json(normalized):
        raise E26FinalDataLockError("Held-out-domain receipt SHA-256 changed")
    normalized["receipt_sha256"] = claimed
    expected = {
        "schema_version": HELDOUT_DOMAIN_RECEIPT_SCHEMA,
        "manifest_type": HELDOUT_DOMAIN_RECEIPT_TYPE,
        "scientific_evidence": False,
        "candidate_domains": list(HELDOUT_DOMAIN_CANDIDATES),
        "selection_algorithm": HELDOUT_DOMAIN_SELECTION,
        "selected_domain": HELDOUT_DOMAIN,
        "model_outcomes_accessed": False,
        "evaluation_outcomes_accessed": False,
        "main_test_opened": False,
        "scientific_main_started": False,
        "passed": True,
    }
    for key, expected_value in expected.items():
        if normalized.get(key) != expected_value:
            raise E26FinalDataLockError(f"Held-out-domain receipt field changed: {key}")
    return normalized


def write_e26_final_heldout_domain_receipt(
    path: str | Path,
    payload: Mapping[str, Any],
) -> Path:
    destination = Path(path).expanduser()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite held-out-domain receipt: {destination}")
    validated = validate_e26_final_heldout_domain_receipt(payload)
    write_json_strict(destination, validated)
    return destination


__all__ = [
    "CAPACITY_RECEIPT_SCHEMA",
    "CAPACITY_ENCODING_BATCH_DOCUMENTS",
    "CAPACITY_ENCODING_THREADS",
    "DATA_SOURCE_RECEIPT_SCHEMA",
    "DEFAULT_DATA_SOURCE_EXPECTATION",
    "HELDOUT_DOMAIN",
    "HELDOUT_DOMAIN_CANDIDATES",
    "HELDOUT_DOMAIN_SELECTION",
    "BoundFileExpectation",
    "E26FinalDataLockError",
    "E26FinalDataSourceExpectation",
    "E26FinalGeneralDocument",
    "E26FinalTokenEncoder",
    "build_e26_final_data_source_receipt",
    "build_e26_final_heldout_domain_receipt",
    "count_e26_final_general_train_capacity",
    "iter_e26_final_general_train",
    "validate_e26_final_capacity_receipt",
    "validate_e26_final_data_source_receipt",
    "validate_e26_final_heldout_domain_receipt",
    "write_e26_final_capacity_receipt",
    "write_e26_final_data_source_receipt",
    "write_e26_final_heldout_domain_receipt",
]
