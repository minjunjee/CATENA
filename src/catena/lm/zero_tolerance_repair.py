"""Prospective zero-tolerance repair for the blocked E26 Stage-2 corpus.

The original Stage-2 audit and disposition stay immutable.  This module creates
an additive eligible population by removing every general-train endpoint of the
frozen detector's flags, rebuilding exact deduplication from the same pinned
Parquet bytes, and rerunning the unchanged detector.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sqlite3
import subprocess
import sys
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from catena.core.provenance_v61 import (
    read_json_object_strict,
    sha256_canonical_json,
    sha256_file,
    write_json_strict,
)

from .construction_source import CRITICAL_TOOL_VERSIONS
from .data_lock import (
    ContentSplit,
    LockedDocument,
    NearDuplicateFlag,
    SourceDocument,
    SQLiteDocumentIndex,
    audit_near_duplicates_asymmetric,
    lock_document,
    write_near_duplicate_audit,
)
from .tokenizer import ExternalScientificTokenizer

ZERO_FLAGS = "ZERO_PROTECTED_TRAIN_FLAGS"
BLOCKED_CAPACITY = "BLOCKED_SOURCE_CAPACITY"
BLOCKED_PROVENANCE = "BLOCKED_PROVENANCE"
TERMINAL_DISPOSITIONS = frozenset({ZERO_FLAGS, BLOCKED_CAPACITY, BLOCKED_PROVENANCE})
REQUIRED_STAGE2_BASE_COMMIT = "55975897b441891312e977ce3734c6b9d2e3c36e"
PROTECTED_SPLITS = (
    ContentSplit.TOKENIZER_ONLY,
    ContentSplit.GENERAL_VALIDATION,
    ContentSplit.GENERAL_TEST,
)


class ZeroToleranceRepairError(RuntimeError):
    """Base class for fail-closed repair errors."""


class RepairProvenanceError(ZeroToleranceRepairError):
    """Raised when the prospective repair cannot preserve frozen provenance."""


class RepairCapacityError(ZeroToleranceRepairError):
    """Raised when the frozen source population cannot restore token capacity."""


@dataclass(frozen=True, slots=True)
class FilteredDedupReceipt:
    source_documents_seen: int
    admitted_source_documents: int
    excluded_source_occurrences: int
    unique_documents: int
    exact_duplicates: int
    split_counts: dict[str, int]
    exclusion_count: int
    exclusion_sha256: str
    manifest_path: str
    manifest_sha256: str
    document_selection_sha256: str
    sqlite_path: str
    sqlite_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_repair_protocol(path: str | Path) -> tuple[Path, dict[str, Any], str]:
    resolved = Path(path).expanduser().resolve(strict=True)
    parsed = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise RepairProvenanceError("Zero-tolerance protocol must be a YAML object")
    if parsed.get("schema_version") != "catena-e26-data-lock-v2-zero-tolerance":
        raise RepairProvenanceError("Unexpected zero-tolerance protocol schema")
    repair = parsed.get("repair")
    if not isinstance(repair, dict):
        raise RepairProvenanceError("Protocol lacks repair settings")
    if repair.get("policy") != "EXCLUDE_ALL_FLAGGED_TRAIN_DOCUMENTS":
        raise RepairProvenanceError("Repair policy changed after prospective lock")
    if repair.get("original_stage2_disposition") != "BLOCKED_DATA_SOURCE":
        raise RepairProvenanceError("Original Stage-2 disposition was not preserved")
    if repair.get("original_policy") != "FAIL_PENDING_MANUAL_AUDIT":
        raise RepairProvenanceError("Original Stage-2 policy was not preserved")
    if set(repair.get("terminal_dispositions", ())) != TERMINAL_DISPOSITIONS:
        raise RepairProvenanceError("Terminal disposition set changed")
    source = parsed.get("source")
    if not isinstance(source, dict):
        raise RepairProvenanceError("Protocol lacks source settings")
    if source.get("train_selection_order") != "CONTENT_SHA256_ASCENDING":
        raise RepairProvenanceError("V1 train-selection order changed")
    if source.get("backfill_policy") != "NEXT_ELIGIBLE_CONTENT_SHA256":
        raise RepairProvenanceError("V1 backfill law changed")
    if source.get("additional_shard_download_allowed") is not False:
        raise RepairProvenanceError("Zero-tolerance repair must not add source shards")
    return resolved, parsed, sha256_file(resolved)


def derived_data_root(protocol: Mapping[str, Any], protocol_sha256: str) -> Path:
    output = protocol.get("output")
    if not isinstance(output, Mapping):
        raise RepairProvenanceError("Protocol lacks output settings")
    template = output.get("namespace_template")
    prefix_length = output.get("protocol_sha_prefix_length")
    if not isinstance(template, str) or "<PROTOCOL_SHA_PREFIX>" not in template:
        raise RepairProvenanceError("Output namespace template is invalid")
    if isinstance(prefix_length, bool) or not isinstance(prefix_length, int):
        raise RepairProvenanceError("Protocol SHA prefix length must be an integer")
    if prefix_length < 8 or prefix_length > 64:
        raise RepairProvenanceError("Protocol SHA prefix length is unsafe")
    return Path(template.replace("<PROTOCOL_SHA_PREFIX>", protocol_sha256[:prefix_length]))


def _bound_file(record: Mapping[str, Any], label: str) -> Path:
    path_value = record.get("path")
    expected = record.get("sha256")
    if not isinstance(path_value, str) or not isinstance(expected, str):
        raise RepairProvenanceError(f"{label} binding is incomplete")
    path = Path(path_value).expanduser().resolve(strict=True)
    if not path.is_file() or sha256_file(path) != expected:
        raise RepairProvenanceError(f"{label} bytes differ from the prospective lock")
    return path


def validate_original_bindings(protocol: Mapping[str, Any]) -> dict[str, Path]:
    raw = protocol.get("original_inputs")
    if not isinstance(raw, Mapping):
        raise RepairProvenanceError("Protocol lacks original input bindings")
    bindings: dict[str, Path] = {}
    for label, record in raw.items():
        if label == "data_root":
            continue
        if isinstance(record, Mapping) and "path" in record and "sha256" in record:
            bindings[str(label)] = _bound_file(record, str(label))
    required = {
        "data_lock",
        "corrected_audit",
        "document_index",
        "dedup_receipt",
        "download_receipt",
        "source_inventory",
        "source_metadata",
        "tokenizer_manifest",
        "tokenizer_replay",
        "general_memmap_receipt",
        "transaction_manifest",
        "paired_schedule",
    }
    if set(bindings) != required:
        raise RepairProvenanceError(
            f"Original binding set mismatch: missing={sorted(required - set(bindings))}, "
            f"extra={sorted(set(bindings) - required)}"
        )
    return bindings


def _expected_detector_contract(protocol: Mapping[str, Any]) -> dict[str, Any]:
    detector = protocol.get("detector")
    if not isinstance(detector, Mapping):
        raise RepairProvenanceError("Protocol lacks detector settings")
    return dict(detector)


def _validate_detector_payload(
    audit: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    fields = {
        "algorithm": "algorithm",
        "normalization": "normalization",
        "shingle_width_words": "shingle_width_words",
        "permutations": "permutations",
        "field_prime": "field_prime",
        "shingle_hash": "shingle_hash",
        "field_reduction": "field_reduction",
        "bands": "bands",
        "rows_per_band": "rows_per_band",
        "seed": "seed",
        "estimated_jaccard_flag_threshold": "estimated_jaccard_flag_threshold",
        "candidate_generation": "candidate_generation",
    }
    for audit_field, protocol_field in fields.items():
        if audit.get(audit_field) != expected.get(protocol_field):
            raise RepairProvenanceError(
                f"Frozen detector field changed: {audit_field}="
                f"{audit.get(audit_field)!r} expected {expected.get(protocol_field)!r}"
            )


def load_initial_exclusions(
    audit_path: str | Path,
    *,
    protocol: Mapping[str, Any],
) -> tuple[tuple[str, ...], dict[str, Any], tuple[dict[str, Any], ...]]:
    audit = read_json_object_strict(audit_path)
    expected_record = protocol["original_inputs"]["corrected_audit"]
    expected_count = int(expected_record["flagged_pair_count"])
    if audit.get("pass") is not False:
        raise RepairProvenanceError("Original blocked audit was unexpectedly relabelled")
    if audit.get("flagged_pair_policy") != "FAIL_PENDING_MANUAL_AUDIT":
        raise RepairProvenanceError("Original blocked audit policy changed")
    if audit.get("audit_sha256") != expected_record["internal_audit_sha256"]:
        raise RepairProvenanceError("Original audit internal SHA changed")
    _validate_detector_payload(audit, _expected_detector_contract(protocol))
    raw_flags = audit.get("flagged_pairs")
    if not isinstance(raw_flags, list) or len(raw_flags) != expected_count:
        raise RepairProvenanceError("Original flagged-pair population changed")
    flags: list[dict[str, Any]] = []
    strata: Counter[str] = Counter()
    train_hashes: list[str] = []
    for row in raw_flags:
        if not isinstance(row, dict):
            raise RepairProvenanceError("Original audit contains a malformed flag")
        left_split = row.get("left_split")
        right_split = row.get("right_split")
        left_sha = row.get("left_sha256")
        right_sha = row.get("right_sha256")
        if left_split not in {item.value for item in PROTECTED_SPLITS}:
            raise RepairProvenanceError("Original flag does not have a protected left endpoint")
        if right_split != ContentSplit.GENERAL_TRAIN.value:
            raise RepairProvenanceError("Original flag does not have a train right endpoint")
        if not isinstance(left_sha, str) or not isinstance(right_sha, str):
            raise RepairProvenanceError("Original flag lacks content hashes")
        strata[f"{left_split}__{right_split}"] += 1
        train_hashes.append(right_sha)
        flags.append(dict(row))
    exclusions = tuple(sorted(set(train_hashes)))
    if len(exclusions) != expected_count:
        raise RepairProvenanceError("Flagged train endpoints are not one-to-one as locked")
    expected_strata = expected_record.get("strata_counts")
    if not isinstance(expected_strata, dict):
        raise RepairProvenanceError("Protocol lacks original audit strata counts")
    if dict(strata) != expected_strata:
        raise RepairProvenanceError(f"Original audit strata changed: {dict(strata)}")
    graph = _graph_statistics(flags)
    return exclusions, graph, tuple(flags)


def _graph_statistics(flags: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    adjacency: dict[str, set[str]] = {}
    for row in flags:
        left = str(row["left_sha256"])
        right = str(row["right_sha256"])
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    seen: set[str] = set()
    component_shapes: Counter[str] = Counter()
    for node in sorted(adjacency):
        if node in seen:
            continue
        stack = [node]
        protected = 0
        train = 0
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            is_train = any(row["right_sha256"] == current for row in flags)
            train += int(is_train)
            protected += int(not is_train)
            stack.extend(sorted(adjacency[current] - seen, reverse=True))
        component_shapes[f"protected_{protected}__train_{train}"] += 1
    edges = len(flags)
    nodes = len(adjacency)
    components = sum(component_shapes.values())
    return {
        "edge_count": edges,
        "node_count": nodes,
        "component_count": components,
        "cycle_rank": edges - nodes + components,
        "component_shapes": dict(sorted(component_shapes.items())),
    }


def _manifest_rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            parsed = json.loads(line)
            if not isinstance(parsed, dict):
                raise RepairProvenanceError(f"Document manifest row {line_number} is not an object")
            yield parsed


def build_initial_exclusion_manifest(
    *,
    exclusions: Sequence[str],
    flags: Sequence[Mapping[str, Any]],
    graph: Mapping[str, Any],
    original_index_path: str | Path,
    original_train_documents_path: str | Path,
    tokenizer_manifest_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    exclusion_set = set(exclusions)
    tokenizer = ExternalScientificTokenizer.from_manifest(tokenizer_manifest_path)
    rows: list[dict[str, Any]] = []
    normalized_bytes = 0
    content_tokens = 0
    raw_digest = hashlib.sha256()
    with SQLiteDocumentIndex(original_index_path, create=False) as index:
        placeholders = ",".join("?" for _ in exclusions)
        query = (
            "SELECT content_sha256, split, normalized_utf8_bytes, normalized_utf8 "
            f"FROM documents WHERE content_sha256 IN ({placeholders}) "
            "ORDER BY content_sha256"
        )
        observed = list(index.connection.execute(query, tuple(exclusions)))
    if len(observed) != len(exclusions):
        raise RepairProvenanceError("Not every locked exclusion exists in the original index")
    for content_sha, split, byte_count, raw_value in observed:
        raw = raw_value if isinstance(raw_value, bytes) else bytes(raw_value)
        if split != ContentSplit.GENERAL_TRAIN.value:
            raise RepairProvenanceError("An exclusion resolves to a protected split")
        if len(raw) != byte_count or hashlib.sha256(raw).hexdigest() != content_sha:
            raise RepairProvenanceError("Excluded normalized bytes fail content identity")
        token_count = len(tokenizer.encode(raw.decode("utf-8")))
        row = {
            "content_sha256": str(content_sha),
            "normalized_utf8_bytes": int(byte_count),
            "locked_content_tokens": token_count,
            "split": str(split),
        }
        rows.append(row)
        normalized_bytes += int(byte_count)
        content_tokens += token_count
        raw_digest.update(bytes.fromhex(str(content_sha)))
        raw_digest.update(len(raw).to_bytes(8, "big"))
        raw_digest.update(raw)
    selected_rows: list[dict[str, Any]] = []
    original_train_tokens = 0
    for row in _manifest_rows(Path(original_train_documents_path)):
        row_token_count = row.get("token_count")
        if isinstance(row_token_count, bool) or not isinstance(row_token_count, int):
            raise RepairProvenanceError("Original train manifest token_count is not an integer")
        original_train_tokens += row_token_count
        content_sha = row.get("content_sha256")
        if content_sha in exclusion_set:
            selected_rows.append(dict(row))
    selected_tokens = sum(int(row["token_count"]) for row in selected_rows)
    payload: dict[str, Any] = {
        "schema_version": "catena-e26-zero-tolerance-exclusions-v1",
        "manifest_type": "E26_ZERO_TOLERANCE_EXCLUSION_MANIFEST",
        "scientific_evidence": False,
        "policy": "EXCLUDE_ALL_FLAGGED_TRAIN_DOCUMENTS",
        "outcome_independent": True,
        "human_labels_used": False,
        "flagged_pair_count": len(flags),
        "unique_train_exclusion_count": len(rows),
        "exclusions": rows,
        "exclusion_sorted_json_sha256": sha256_canonical_json(list(exclusions)),
        "excluded_normalized_utf8_bytes": normalized_bytes,
        "excluded_locked_content_tokens": content_tokens,
        "excluded_locked_tokens_with_one_separator_per_document": content_tokens + len(rows),
        "raw_boundary_digest_sha256": raw_digest.hexdigest(),
        "graph": dict(graph),
        "v1_train_exposure": {
            "selected_exclusion_count": len(selected_rows),
            "selected_exclusion_token_count": selected_tokens,
            "selected_exclusion_fraction": selected_tokens / original_train_tokens,
            "original_train_token_count": original_train_tokens,
            "selected_rows_sha256": sha256_canonical_json(selected_rows),
        },
    }
    payload["exclusion_manifest_sha256"] = sha256_canonical_json(payload)
    write_json_strict(output_path, payload)
    return payload


def _record_line(record: LockedDocument) -> str:
    return json.dumps(
        record.as_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def build_filtered_document_index(
    documents: Iterable[SourceDocument],
    *,
    exclusions: Sequence[str],
    sqlite_path: str | Path,
    manifest_path: str | Path,
) -> FilteredDedupReceipt:
    """Re-run exact dedup from source while removing only locked train hashes."""

    exclusion_set = frozenset(exclusions)
    if len(exclusion_set) != len(exclusions):
        raise RepairProvenanceError("Exclusion sequence contains duplicate hashes")
    destination = Path(manifest_path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite filtered manifest: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_seen = 0
    admitted_seen = 0
    excluded_occurrences = 0
    duplicates = 0
    previous_location: tuple[str, int, int] | None = None
    with SQLiteDocumentIndex(sqlite_path, create=True) as index:
        for document in documents:
            if previous_location is not None and document.location_key < previous_location:
                raise RepairProvenanceError("Source order differs from canonical shard/row order")
            previous_location = document.location_key
            source_seen += 1
            record, normalized = lock_document(document)
            if record.content_sha256 in exclusion_set:
                if record.split != ContentSplit.GENERAL_TRAIN.value:
                    raise RepairProvenanceError("Exclusion would remove a protected document")
                excluded_occurrences += 1
                continue
            admitted_seen += 1
            if not index.add(record, normalized):
                duplicates += 1
        index.commit()
        split_counts: Counter[str] = Counter()
        digest = hashlib.sha256()
        unique_documents = 0
        with destination.open("x", encoding="utf-8", newline="\n") as handle:
            for record in index.records():
                unique_documents += 1
                split_counts[record.split] += 1
                line = _record_line(record)
                handle.write(line + "\n")
                digest.update(line.encode() + b"\n")
            handle.flush()
    if excluded_occurrences < len(exclusion_set):
        raise RepairProvenanceError("At least one exclusion was absent from pinned source bytes")
    return FilteredDedupReceipt(
        source_documents_seen=source_seen,
        admitted_source_documents=admitted_seen,
        excluded_source_occurrences=excluded_occurrences,
        unique_documents=unique_documents,
        exact_duplicates=duplicates,
        split_counts=dict(sorted(split_counts.items())),
        exclusion_count=len(exclusion_set),
        exclusion_sha256=sha256_canonical_json(sorted(exclusion_set)),
        manifest_path=str(destination.resolve()),
        manifest_sha256=sha256_file(destination),
        document_selection_sha256=digest.hexdigest(),
        sqlite_path=str(Path(sqlite_path).resolve()),
        sqlite_sha256=sha256_file(sqlite_path),
    )


def _next_record(iterator: Iterator[LockedDocument]) -> LockedDocument | None:
    try:
        return next(iterator)
    except StopIteration:
        return None


def validate_filtered_population(
    *,
    original_index_path: str | Path,
    filtered_index_path: str | Path,
    exclusions: Sequence[str],
) -> dict[str, Any]:
    """Prove the repaired population is exactly V1 minus the monotonic exclusion set."""

    exclusion_set = frozenset(exclusions)
    protected: dict[str, dict[str, Any]] = {}
    with (
        SQLiteDocumentIndex(original_index_path, create=False) as original,
        SQLiteDocumentIndex(filtered_index_path, create=False) as filtered,
    ):
        for split in PROTECTED_SPLITS:
            old_digest = hashlib.sha256()
            new_digest = hashlib.sha256()
            old_count = 0
            new_count = 0
            for record in original.records(split=split):
                old_count += 1
                old_digest.update((_record_line(record) + "\n").encode())
            for record in filtered.records(split=split):
                new_count += 1
                new_digest.update((_record_line(record) + "\n").encode())
            if old_count != new_count or old_digest.digest() != new_digest.digest():
                raise RepairProvenanceError(f"Protected split changed during repair: {split.value}")
            protected[split.value] = {
                "document_count": old_count,
                "record_digest_sha256": old_digest.hexdigest(),
                "byte_identical_records": True,
            }
        old_rows = original.records(split=ContentSplit.GENERAL_TRAIN)
        new_rows = filtered.records(split=ContentSplit.GENERAL_TRAIN)
        old = _next_record(old_rows)
        new = _next_record(new_rows)
        removed: list[str] = []
        retained = 0
        while old is not None:
            if old.content_sha256 in exclusion_set:
                removed.append(old.content_sha256)
                old = _next_record(old_rows)
                continue
            if new is None or old.as_dict() != new.as_dict():
                raise RepairProvenanceError("Filtered train population is not V1 minus exclusions")
            retained += 1
            old = _next_record(old_rows)
            new = _next_record(new_rows)
        if new is not None or tuple(removed) != tuple(sorted(exclusion_set)):
            raise RepairProvenanceError("Filtered train population has missing or extra rows")
    return {
        "protected": protected,
        "removed_train_document_count": len(removed),
        "removed_train_sha256": sha256_canonical_json(removed),
        "retained_train_document_count": retained,
        "exact_subset_relation": True,
    }


def _audit_records(
    index: SQLiteDocumentIndex,
    splits: Sequence[ContentSplit],
) -> Iterator[tuple[str, str, str]]:
    for split in splits:
        for record, text in index.texts(split):
            yield record.content_sha256, split.value, text


def run_locked_near_duplicate_audit(
    document_index_path: str | Path,
) -> tuple[NearDuplicateFlag, ...]:
    with SQLiteDocumentIndex(document_index_path, create=False) as index:
        return audit_near_duplicates_asymmetric(
            _audit_records(index, PROTECTED_SPLITS),
            _audit_records(index, (ContentSplit.GENERAL_TRAIN,)),
        )


def write_zero_tolerance_audit(
    path: str | Path,
    flags: Sequence[NearDuplicateFlag],
    *,
    original_implementation_history: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    # Keep the registered detector writer byte-contract unchanged.  Automatic
    # exclusion is recorded in a separate prospective round ledger, never by
    # relabelling this audit's original FAIL_PENDING_MANUAL_AUDIT field.
    output = write_near_duplicate_audit(
        path,
        flags,
        implementation_history=original_implementation_history,
    )
    return read_json_object_strict(output)


def write_repair_round_ledger(
    path: str | Path,
    *,
    iteration: int,
    exclusion_manifest_sha256: str,
    exclusions: Sequence[str],
    audit_path: str | Path,
    audit: Mapping[str, Any],
    population_validation: Mapping[str, Any],
) -> dict[str, Any]:
    additions = additional_train_exclusions(
        tuple(
            NearDuplicateFlag(
                str(row["left_sha256"]),
                str(row["left_split"]),
                str(row["right_sha256"]),
                str(row["right_split"]),
                float(row["estimated_jaccard"]),
            )
            for row in audit.get("flagged_pairs", ())
            if isinstance(row, Mapping)
        )
    )
    payload: dict[str, Any] = {
        "schema_version": "catena-e26-zero-tolerance-round-v1",
        "manifest_type": "E26_ZERO_TOLERANCE_REPAIR_ROUND",
        "scientific_evidence": False,
        "policy": "EXCLUDE_ALL_FLAGGED_TRAIN_DOCUMENTS",
        "iteration": iteration,
        "input_exclusion_count": len(exclusions),
        "input_exclusion_sha256": sha256_canonical_json(list(exclusions)),
        "seed_exclusion_manifest_sha256": exclusion_manifest_sha256,
        "audit": {
            "path": str(Path(audit_path).resolve(strict=True)),
            "sha256": sha256_file(audit_path),
            "internal_audit_sha256": audit.get("audit_sha256"),
            "flagged_pair_count": audit.get("flagged_pair_count"),
            "registered_flagged_pair_policy_preserved": (
                audit.get("flagged_pair_policy") == "FAIL_PENDING_MANUAL_AUDIT"
            ),
        },
        "automatic_train_side_additions": list(additions),
        "automatic_addition_count": len(additions),
        "population_validation": dict(population_validation),
        "original_stage2_disposition_preserved": True,
    }
    payload["round_sha256"] = sha256_canonical_json(payload)
    write_json_strict(path, payload)
    return payload


def additional_train_exclusions(flags: Sequence[NearDuplicateFlag]) -> tuple[str, ...]:
    additions: set[str] = set()
    for flag in flags:
        if (
            flag.left_split not in {item.value for item in PROTECTED_SPLITS}
            or flag.right_split != ContentSplit.GENERAL_TRAIN.value
        ):
            raise RepairProvenanceError(
                "Detector produced a flag without one removable train endpoint"
            )
        additions.add(flag.right_sha256)
    return tuple(sorted(additions))


def verified_shard_paths(download_receipt_path: str | Path) -> tuple[Path, ...]:
    receipt = read_json_object_strict(download_receipt_path)
    if receipt.get("revision") != "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9":
        raise RepairProvenanceError("Pinned FineWeb-Edu revision changed")
    if receipt.get("selected_indices") != [0, 2, 4, 9, 13]:
        raise RepairProvenanceError("Pinned source shard selection changed")
    rows = receipt.get("shards")
    if not isinstance(rows, list) or len(rows) != 5:
        raise RepairProvenanceError("Download receipt does not contain five locked shards")
    paths: list[Path] = []
    for row in rows:
        if not isinstance(row, Mapping) or row.get("verified") is not True:
            raise RepairProvenanceError("Download receipt contains an unverified shard")
        path = Path(str(row.get("local_path", ""))).resolve(strict=True)
        if path.stat().st_size != row.get("size") or sha256_file(path) != row.get("local_sha256"):
            raise RepairProvenanceError(f"Pinned source shard bytes changed: {path}")
        paths.append(path)
    return tuple(sorted(paths))


REPAIR_SOURCE_FILES = (
    "configs/e26_data_lock_v2_zero_tolerance.yaml",
    "schemas/v8_1/e26_scientific_data_readiness_v3.schema.json",
    "schemas/v8_1/e26_zero_tolerance_repair_receipt.schema.json",
    "scripts/prepare_e26_data_v2_zero_tolerance.sh",
    "scripts/validate_e26_data_v2_zero_tolerance.sh",
    "src/catena/core/provenance_v61.py",
    "src/catena/lm/data_lock.py",
    "src/catena/lm/data_readiness_v3.py",
    "src/catena/lm/construction_source.py",
    "src/catena/lm/frozen_invariance.py",
    "src/catena/lm/general_corpus.py",
    "src/catena/lm/hashing.py",
    "src/catena/lm/memmap_builder.py",
    "src/catena/lm/paired_stream.py",
    "src/catena/lm/parquet_documents.py",
    "src/catena/lm/schedule_manifest.py",
    "src/catena/lm/tokenizer.py",
    "src/catena/lm/transactional_stream.py",
    "src/catena/lm/zero_tolerance_repair.py",
    "tools/repair_e26_zero_tolerance_data.py",
    "tools/validate_e26_data_v2.py",
)


def build_repair_source_receipt(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve(strict=True)
    head = subprocess.run(
        ("git", "-C", str(root), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    branch = subprocess.run(
        ("git", "-C", str(root), "branch", "--show-current"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        (
            "git",
            "-C",
            str(root),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise RepairProvenanceError("Repair source worktree must be clean and committed")
    ancestry = subprocess.run(
        (
            "git",
            "-C",
            str(root),
            "merge-base",
            "--is-ancestor",
            REQUIRED_STAGE2_BASE_COMMIT,
            head,
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    if ancestry.returncode != 0:
        raise RepairProvenanceError("Repair branch does not descend from the locked Stage-2 base")
    files: list[dict[str, Any]] = []
    for relative in REPAIR_SOURCE_FILES:
        path = (root / relative).resolve(strict=True)
        if not path.is_file():
            raise RepairProvenanceError(f"Repair source file is missing: {relative}")
        files.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    observed_versions = {
        package: importlib.metadata.version(package) for package in CRITICAL_TOOL_VERSIONS
    }
    if observed_versions != CRITICAL_TOOL_VERSIONS:
        raise RepairProvenanceError(f"Pinned repair-tool environment changed: {observed_versions}")
    freeze_result = subprocess.run(
        (sys.executable, "-m", "pip", "freeze", "--all"),
        check=True,
        capture_output=True,
        text=True,
    )
    freeze = sorted(line.strip() for line in freeze_result.stdout.splitlines() if line.strip())
    payload: dict[str, Any] = {
        "schema_version": "catena-e26-zero-tolerance-source-v1",
        "manifest_type": "E26_ZERO_TOLERANCE_REPAIR_SOURCE",
        "scientific_evidence": False,
        "claim_ceiling": "SCIENTIFIC_INPUT_PROVENANCE_ONLY",
        "repo_root": str(root),
        "git_head": head,
        "git_branch": branch,
        "required_stage2_base_commit": REQUIRED_STAGE2_BASE_COMMIT,
        "required_stage2_base_is_ancestor": True,
        "git_clean": True,
        "builder_files": files,
        "builder_source_sha256": sha256_canonical_json(files),
        "tool_environment": {
            "python_executable": str(Path(sys.executable).resolve()),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "sqlite_version": sqlite3.sqlite_version,
            "critical_versions": observed_versions,
            "pip_freeze": freeze,
            "pip_freeze_sha256": sha256_canonical_json(freeze),
        },
    }
    payload["receipt_sha256"] = sha256_canonical_json(payload)
    return payload


def validate_repair_source_receipt(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != "catena-e26-zero-tolerance-source-v1":
        raise RepairProvenanceError("Repair source receipt schema changed")
    if payload.get("git_clean") is not True:
        raise RepairProvenanceError("Repair source receipt was not generated from a clean tree")
    if (
        payload.get("required_stage2_base_commit") != REQUIRED_STAGE2_BASE_COMMIT
        or payload.get("required_stage2_base_is_ancestor") is not True
    ):
        raise RepairProvenanceError("Repair source receipt lacks Stage-2 ancestry")
    rows = payload.get("builder_files")
    if not isinstance(rows, list):
        raise RepairProvenanceError("Repair source receipt lacks builder files")
    if tuple(str(row.get("path")) for row in rows if isinstance(row, Mapping)) != (
        REPAIR_SOURCE_FILES
    ):
        raise RepairProvenanceError("Repair source file set/order changed")
    root = Path(str(payload.get("repo_root", ""))).resolve(strict=True)
    for row in rows:
        if not isinstance(row, Mapping):
            raise RepairProvenanceError("Malformed repair builder source row")
        path = (root / str(row["path"])).resolve(strict=True)
        if path.stat().st_size != row.get("bytes") or sha256_file(path) != row.get("sha256"):
            raise RepairProvenanceError(f"Repair builder source changed: {row.get('path')}")
    if payload.get("builder_source_sha256") != sha256_canonical_json(rows):
        raise RepairProvenanceError("Repair builder source aggregate changed")
    environment = payload.get("tool_environment")
    if not isinstance(environment, Mapping):
        raise RepairProvenanceError("Repair source receipt lacks tool environment")
    if environment.get("critical_versions") != CRITICAL_TOOL_VERSIONS:
        raise RepairProvenanceError("Repair critical tool versions changed")
    freeze = environment.get("pip_freeze")
    if not isinstance(freeze, list) or environment.get(
        "pip_freeze_sha256"
    ) != sha256_canonical_json(freeze):
        raise RepairProvenanceError("Repair pip-freeze lock changed")
    without_hash = dict(payload)
    claimed = without_hash.pop("receipt_sha256", None)
    if claimed != sha256_canonical_json(without_hash):
        raise RepairProvenanceError("Repair source receipt internal hash changed")


def write_terminal_status(
    path: str | Path,
    *,
    disposition: str,
    detail: Mapping[str, Any],
) -> None:
    if disposition not in TERMINAL_DISPOSITIONS:
        raise ValueError(f"Unsupported terminal disposition: {disposition}")
    payload: dict[str, Any] = {
        "schema_version": "catena-e26-zero-tolerance-status-v1",
        "scientific_evidence": False,
        "disposition": disposition,
        "gpu_preflight_started": False,
        "scientific_e26a_started": False,
        "scientific_main_started": False,
        "detail": dict(detail),
    }
    payload["status_sha256"] = sha256_canonical_json(payload)
    write_json_strict(path, payload)
