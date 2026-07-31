"""Exact paired mixed-stream replay receipt for scientific E26 inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from catena.core.provenance_v61 import sha256_canonical_json, sha256_file, write_json_strict

from .general_corpus import TokenMemmap
from .paired_stream import (
    PackedTransactionCursor,
    TokenBalancedPairedTrainingCursor,
    replay_digest,
)
from .tokenizer import ExternalScientificTokenizer


class ScheduleManifestError(RuntimeError):
    """Raised when paired variants or a resumed cursor see different examples."""


def _new_cursor(
    corpus: TokenMemmap,
    tokenizer: ExternalScientificTokenizer,
    *,
    seed: int,
    sequence_length: int,
) -> TokenBalancedPairedTrainingCursor:
    general = corpus.paired_cursor(seed=seed, sequence_length=sequence_length)
    transaction = PackedTransactionCursor(
        tokenizer,
        tokenizer_hash=tokenizer.manifest.manifest_hash,
        seed=seed,
        sequence_length=sequence_length,
        pad_token_id=tokenizer.manifest.special_tokens["pad"],
        split="train",
    )
    return TokenBalancedPairedTrainingCursor(general, transaction)


def _validate_realized_mix(probe: dict[str, Any], label: str) -> None:
    snapshot = probe.get("cursor_snapshot")
    if not isinstance(snapshot, dict):
        raise ScheduleManifestError(f"{label} cursor snapshot is missing")
    general_tokens = snapshot.get("general_unpadded_tokens")
    transaction_tokens = snapshot.get("transaction_unpadded_tokens")
    sequence_length = snapshot.get("sequence_length")
    if (
        not isinstance(general_tokens, int)
        or not isinstance(transaction_tokens, int)
        or not isinstance(sequence_length, int)
        or general_tokens <= 0
        or transaction_tokens <= 0
        or abs(4 * transaction_tokens - general_tokens) > 4 * sequence_length
    ):
        raise ScheduleManifestError(
            f"{label} violates the greedy one-row cumulative token-balance bound"
        )


def write_schedule_manifest(
    path: str | Path,
    *,
    train_corpus_manifest: str | Path,
    tokenizer_manifest: str | Path,
    seed: int = 260_026,
    sequence_length: int = 4_096,
    probe_tokens: int = 1_000_000,
) -> Path:
    tokenizer = ExternalScientificTokenizer.from_manifest(tokenizer_manifest)
    corpus = TokenMemmap.from_scientific_manifest(
        train_corpus_manifest,
        tokenizer_manifest=tokenizer.manifest,
    )
    tied = _new_cursor(corpus, tokenizer, seed=seed, sequence_length=sequence_length)
    dual = _new_cursor(corpus, tokenizer, seed=seed, sequence_length=sequence_length)
    tied_first = replay_digest(tied, minimum_tokens=probe_tokens)
    dual_first = replay_digest(dual, minimum_tokens=probe_tokens)
    if tied_first != dual_first:
        raise ScheduleManifestError("Paired variants differ in their first cursor probe")
    _validate_realized_mix(tied_first, "first probe")
    resumed = TokenBalancedPairedTrainingCursor.from_snapshot(
        corpus,
        tokenizer,
        tokenizer_hash=tokenizer.manifest.manifest_hash,
        snapshot=tied.snapshot(),
    )
    tied_second = replay_digest(tied, minimum_tokens=probe_tokens)
    resumed_second = replay_digest(resumed, minimum_tokens=probe_tokens)
    dual_second = replay_digest(dual, minimum_tokens=probe_tokens)
    if tied_second != resumed_second or tied_second != dual_second:
        raise ScheduleManifestError("Cursor continuation/resume/paired replay differs")
    _validate_realized_mix(tied_second, "post-resume probe")
    payload: dict[str, Any] = {
        "schema_version": "catena-e26-paired-schedule-v2",
        "manifest_type": "E26_PAIRED_TRAINING_SCHEDULE",
        "scientific_evidence": False,
        "evidence_tier": "SCIENTIFIC_INPUT_PROVENANCE",
        "scientific_main_input_eligible": True,
        "algorithm": "token_balanced_complete_example_80_20_v2",
        "transaction_packing": "COMPLETE_EXAMPLES_NO_TRUNCATION_V2",
        "seed": seed,
        "sequence_length": sequence_length,
        "target_general_fraction": 0.8,
        "target_transaction_fraction": 0.2,
        "mix_validation": "ABS_4T_MINUS_G_LE_4_TIMES_SEQUENCE_LENGTH",
        "train_corpus_manifest_sha256": sha256_file(train_corpus_manifest),
        "tokenizer_manifest_sha256": sha256_file(tokenizer_manifest),
        "first_probe": tied_first,
        "post_resume_probe": tied_second,
        "paired_variants_identical": True,
        "resume_identical": True,
        "actual_loss_bearing_mix_valid": True,
        "main_test_opened": False,
    }
    payload["manifest_sha256"] = sha256_canonical_json(payload)
    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite schedule manifest: {destination}")
    write_json_strict(destination, payload)
    return destination
