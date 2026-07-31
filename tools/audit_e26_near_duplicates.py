#!/usr/bin/env python3
"""Run the prospectively locked cross-split 5-word MinHash audit."""

from __future__ import annotations

import argparse
from collections.abc import Iterator

from catena.lm.data_lock import (
    ContentSplit,
    SQLiteDocumentIndex,
    audit_near_duplicates_asymmetric,
    normalize_document,
    write_near_duplicate_audit,
)


def _records(
    index: SQLiteDocumentIndex,
    splits: tuple[ContentSplit, ...],
) -> Iterator[tuple[str, str, str]]:
    for split in splits:
        for record, text in index.texts(split):
            yield record.content_sha256, split.value, normalize_document(text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--document-index", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--invalid-attempt-command", required=True)
    parser.add_argument("--invalid-attempt-output", required=True)
    args = parser.parse_args()
    with SQLiteDocumentIndex(args.document_index, create=False) as index:
        flags = audit_near_duplicates_asymmetric(
            _records(
                index,
                (
                    ContentSplit.TOKENIZER_ONLY,
                    ContentSplit.GENERAL_VALIDATION,
                    ContentSplit.GENERAL_TEST,
                ),
            ),
            _records(index, (ContentSplit.GENERAL_TRAIN,)),
        )
    output = write_near_duplicate_audit(
        args.output,
        flags,
        implementation_history=(
            {
                "status": "INVALID_IMPLEMENTATION_ATTEMPT_INTERRUPTED",
                "scientific_input_eligible": False,
                "defect": (
                    "PROTECTED_8_LOWEST_RAW_SHINGLE_HASH_BLOCKER_DID_NOT_MATCH_"
                    "REGISTERED_128_PERMUTATION_32X4_LSH"
                ),
                "command": args.invalid_attempt_command,
                "output_path": (
                    None
                    if args.invalid_attempt_output == "NONE_NO_ARTIFACT"
                    else args.invalid_attempt_output
                ),
                "termination": "SIGINT_AFTER_CONTRACT_AUDIT",
                "eligible_artifact_created": False,
            },
        ),
    )
    print(f"E26 near-duplicate flags: {len(flags)} ({output.resolve()})")
    return 2 if flags else 0


if __name__ == "__main__":
    raise SystemExit(main())
