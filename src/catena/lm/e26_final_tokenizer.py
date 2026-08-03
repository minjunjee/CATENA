"""Exact TinyLlama tokenizer contract for the additive E26 Final pipeline.

The older E26 scientific-data stack is intentionally locked to a CATENA-built
16K tokenizer.  E26 Final instead uses the exact 32K TinyLlama tokenizer files
named by the official GDN2 training path.  This module keeps those contracts
separate and fails closed before a runtime tokenizer is constructed.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

from catena.core.provenance_v61 import (
    SHA256_PATTERN,
    read_json_object_strict,
    sha256_canonical_json,
    sha256_file,
    write_json_strict,
)

from .e26_final_provenance import (
    TOKENIZER_FILES,
    TOKENIZER_PAD_POLICY,
    TOKENIZER_REPO_ID,
    TOKENIZER_REVISION,
    TOKENIZER_SPECIAL_IDS,
    TOKENIZER_VOCAB_SIZE,
    HfFileExpectation,
)

TOKENIZER_BACKEND: Final = "SENTENCEPIECE_TOKENIZER_MODEL"
DOCUMENT_SEPARATOR_POLICY: Final = "EOS_AFTER_EACH_DOCUMENT"
TOKENIZER_RECEIPT_SCHEMA: Final = "catena-e26-final-tokenizer-lock-v1"
TOKENIZER_RECEIPT_TYPE: Final = "E26_FINAL_32K_TOKENIZER_LOCK"
DEFAULT_TOKENIZER_ROOT: Final = Path("/data/minjun_dev/CATENA/checkpoints/e26/tokenizer_ff3c701")


class E26FinalTokenizerError(RuntimeError):
    """Raised when the exact E26 Final tokenizer contract cannot be proven."""


@dataclass(frozen=True, slots=True)
class E26FinalTokenizerExpectation:
    """Immutable expected bytes and semantic identity for one tokenizer bundle."""

    repo_id: str = TOKENIZER_REPO_ID
    revision: str = TOKENIZER_REVISION
    vocab_size: int = TOKENIZER_VOCAB_SIZE
    unk_token_id: int = TOKENIZER_SPECIAL_IDS["unk"]
    bos_token_id: int = TOKENIZER_SPECIAL_IDS["bos"]
    eos_token_id: int = TOKENIZER_SPECIAL_IDS["eos"]
    pad_token_id: int = TOKENIZER_SPECIAL_IDS["eos"]
    document_separator_id: int = TOKENIZER_SPECIAL_IDS["eos"]
    files: tuple[HfFileExpectation, ...] = TOKENIZER_FILES

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["files"] = [asdict(row) for row in self.files]
        return payload


DEFAULT_TOKENIZER_EXPECTATION: Final = E26FinalTokenizerExpectation()


def _regular_bundle_root(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise E26FinalTokenizerError("Tokenizer bundle root must not be a symlink")
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as error:
        raise E26FinalTokenizerError(f"Tokenizer bundle is missing: {candidate}") from error
    if not resolved.is_dir():
        raise E26FinalTokenizerError(f"Tokenizer bundle is not a directory: {resolved}")
    return resolved


def _required_file(root: Path, filename: str) -> Path:
    candidate = root / filename
    if candidate.is_symlink() or not candidate.is_file():
        raise E26FinalTokenizerError(
            f"Tokenizer input must be a regular non-symlink file: {candidate}"
        )
    resolved = candidate.resolve(strict=True)
    if resolved.parent != root:
        raise E26FinalTokenizerError(f"Tokenizer input escaped its bundle root: {candidate}")
    return resolved


def _token_content(value: Any) -> Any:
    return value.get("content") if isinstance(value, Mapping) else None


def _semantic_checks(
    root: Path,
    expectation: E26FinalTokenizerExpectation,
) -> dict[str, bool]:
    tokenizer = read_json_object_strict(root / "tokenizer.json")
    tokenizer_config = read_json_object_strict(root / "tokenizer_config.json")
    special_map = read_json_object_strict(root / "special_tokens_map.json")
    model_config = read_json_object_strict(root / "config.json")
    model = tokenizer.get("model")
    vocab = model.get("vocab") if isinstance(model, Mapping) else None
    ids = {
        "unk": vocab.get("<unk>") if isinstance(vocab, Mapping) else None,
        "bos": vocab.get("<s>") if isinstance(vocab, Mapping) else None,
        "eos": vocab.get("</s>") if isinstance(vocab, Mapping) else None,
    }
    return {
        "tokenizer_model_bpe": isinstance(model, Mapping) and model.get("type") == "BPE",
        "vocab_size_exact": (
            isinstance(vocab, Mapping)
            and len(vocab) == expectation.vocab_size
            and model_config.get("vocab_size") == expectation.vocab_size
        ),
        "special_ids_exact": ids
        == {
            "unk": expectation.unk_token_id,
            "bos": expectation.bos_token_id,
            "eos": expectation.eos_token_id,
        },
        "tokenizer_config_specials_exact": (
            _token_content(tokenizer_config.get("bos_token")) == "<s>"
            and _token_content(tokenizer_config.get("eos_token")) == "</s>"
            and _token_content(tokenizer_config.get("unk_token")) == "<unk>"
            and tokenizer_config.get("pad_token") is None
        ),
        "tokenizer_config_boundary_policy_exact": (
            tokenizer_config.get("add_bos_token") is True
            and tokenizer_config.get("add_eos_token") is False
        ),
        "special_token_population_exact": set(special_map)
        == {"bos_token", "eos_token", "unk_token"},
        "pad_is_eos": expectation.pad_token_id == expectation.eos_token_id,
        "document_separator_is_eos": (
            expectation.document_separator_id == expectation.eos_token_id
        ),
        "silent_id_zero_padding_forbidden": expectation.pad_token_id != 0,
    }


def build_e26_final_tokenizer_receipt(
    tokenizer_root: str | Path,
    *,
    expectation: E26FinalTokenizerExpectation = DEFAULT_TOKENIZER_EXPECTATION,
) -> dict[str, Any]:
    """Audit exact local bytes and return a self-hashed admission receipt.

    Directories such as a Hugging Face ``.cache`` folder are ignored.  Every
    regular file in the bundle root, however, must be one of the six pinned
    tokenizer files.  This prevents an ambiguous runtime file population.
    """

    root = _regular_bundle_root(tokenizer_root)
    expected_names = {row.filename for row in expectation.files}
    if len(expected_names) != len(expectation.files):
        raise E26FinalTokenizerError("Tokenizer expectation contains duplicate filenames")
    unexpected_symlinks = sorted(row.name for row in root.iterdir() if row.is_symlink())
    if unexpected_symlinks:
        raise E26FinalTokenizerError(f"Tokenizer bundle contains symlinks: {unexpected_symlinks}")
    observed_regular = {row.name for row in root.iterdir() if row.is_file()}
    if observed_regular != expected_names:
        raise E26FinalTokenizerError(
            "Tokenizer file population differs; "
            f"missing={sorted(expected_names - observed_regular)}, "
            f"extra={sorted(observed_regular - expected_names)}"
        )

    file_rows: dict[str, dict[str, Any]] = {}
    file_checks: dict[str, bool] = {}
    for expected in expectation.files:
        path = _required_file(root, expected.filename)
        observed_size = path.stat().st_size
        observed_sha = sha256_file(path)
        exact = observed_size == expected.size and observed_sha == expected.sha256
        file_checks[f"file_bytes_exact.{expected.filename}"] = exact
        file_rows[expected.filename] = {
            "path": str(path),
            "bytes": observed_size,
            "sha256": observed_sha,
            "expected_bytes": expected.size,
            "expected_sha256": expected.sha256,
            "hf_blob_id": expected.blob_id,
            "hf_lfs": expected.lfs,
        }

    checks = {**file_checks, **_semantic_checks(root, expectation)}
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise E26FinalTokenizerError(f"Tokenizer admission checks failed: {failed}")

    payload: dict[str, Any] = {
        "schema_version": TOKENIZER_RECEIPT_SCHEMA,
        "manifest_type": TOKENIZER_RECEIPT_TYPE,
        "scientific_evidence": False,
        "evidence_tier": "SCIENTIFIC_INPUT_PROVENANCE",
        "claim_ceiling": "TOKENIZER_IDENTITY_AND_BOUNDARY_POLICY_ONLY",
        "repo_id": expectation.repo_id,
        "revision": expectation.revision,
        "bundle_root": str(root),
        "runtime_backend": TOKENIZER_BACKEND,
        "runtime_primary_file": "tokenizer.model",
        "vocab_size": expectation.vocab_size,
        "special_token_ids": {
            "unk": expectation.unk_token_id,
            "bos": expectation.bos_token_id,
            "eos": expectation.eos_token_id,
            "pad": expectation.pad_token_id,
        },
        "pad_policy": TOKENIZER_PAD_POLICY,
        "document_separator_policy": DOCUMENT_SEPARATOR_POLICY,
        "document_separator_id": expectation.document_separator_id,
        "files": dict(sorted(file_rows.items())),
        "hard_checks": dict(sorted(checks.items())),
        "scientific_main_started": False,
        "passed": True,
    }
    payload["receipt_sha256"] = sha256_canonical_json(payload)
    return validate_e26_final_tokenizer_receipt(payload, expectation=expectation)


def validate_e26_final_tokenizer_receipt(
    payload: Mapping[str, Any],
    *,
    expectation: E26FinalTokenizerExpectation = DEFAULT_TOKENIZER_EXPECTATION,
    verify_local_files: bool = True,
) -> dict[str, Any]:
    """Validate receipt integrity and optionally re-hash every bound local file."""

    normalized = deepcopy(dict(payload))
    claimed = normalized.pop("receipt_sha256", None)
    if not isinstance(claimed, str) or not SHA256_PATTERN.fullmatch(claimed):
        raise E26FinalTokenizerError("Tokenizer receipt lacks a valid SHA-256")
    if claimed != sha256_canonical_json(normalized):
        raise E26FinalTokenizerError("Tokenizer receipt SHA-256 changed")
    normalized["receipt_sha256"] = claimed
    expected_identity = {
        "schema_version": TOKENIZER_RECEIPT_SCHEMA,
        "manifest_type": TOKENIZER_RECEIPT_TYPE,
        "scientific_evidence": False,
        "repo_id": expectation.repo_id,
        "revision": expectation.revision,
        "runtime_backend": TOKENIZER_BACKEND,
        "runtime_primary_file": "tokenizer.model",
        "vocab_size": expectation.vocab_size,
        "pad_policy": TOKENIZER_PAD_POLICY,
        "document_separator_policy": DOCUMENT_SEPARATOR_POLICY,
        "document_separator_id": expectation.document_separator_id,
        "scientific_main_started": False,
        "passed": True,
    }
    for key, expected_value in expected_identity.items():
        if normalized.get(key) != expected_value:
            raise E26FinalTokenizerError(f"Tokenizer receipt field changed: {key}")
    expected_specials = {
        "unk": expectation.unk_token_id,
        "bos": expectation.bos_token_id,
        "eos": expectation.eos_token_id,
        "pad": expectation.pad_token_id,
    }
    if normalized.get("special_token_ids") != expected_specials:
        raise E26FinalTokenizerError("Tokenizer special-token contract changed")
    checks = normalized.get("hard_checks")
    if (
        not isinstance(checks, Mapping)
        or not checks
        or not all(value is True for value in checks.values())
    ):
        raise E26FinalTokenizerError("Tokenizer hard-check map is not all PASS")

    root_value = normalized.get("bundle_root")
    files = normalized.get("files")
    if not isinstance(root_value, str) or not isinstance(files, Mapping):
        raise E26FinalTokenizerError("Tokenizer receipt lacks bound local files")
    root = _regular_bundle_root(root_value)
    expected_names = {row.filename for row in expectation.files}
    if set(files) != expected_names:
        raise E26FinalTokenizerError("Tokenizer receipt file population changed")
    if verify_local_files:
        for expected in expectation.files:
            record = files.get(expected.filename)
            if not isinstance(record, Mapping):
                raise E26FinalTokenizerError(f"Tokenizer receipt lacks {expected.filename} binding")
            path = _required_file(root, expected.filename)
            if (
                record.get("path") != str(path)
                or record.get("bytes") != expected.size
                or record.get("sha256") != expected.sha256
                or path.stat().st_size != expected.size
                or sha256_file(path) != expected.sha256
            ):
                raise E26FinalTokenizerError(f"Tokenizer file bytes changed: {expected.filename}")
    return normalized


def write_e26_final_tokenizer_receipt(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    expectation: E26FinalTokenizerExpectation = DEFAULT_TOKENIZER_EXPECTATION,
) -> Path:
    """Write one immutable tokenizer receipt, refusing any overwrite."""

    destination = Path(path).expanduser()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite tokenizer receipt: {destination}")
    validated = validate_e26_final_tokenizer_receipt(payload, expectation=expectation)
    write_json_strict(destination, validated)
    return destination


class E26FinalSentencePieceTokenizer:
    """Lazy SentencePiece runtime admitted by an exact tokenizer receipt."""

    def __init__(
        self,
        receipt: Mapping[str, Any],
        *,
        expectation: E26FinalTokenizerExpectation = DEFAULT_TOKENIZER_EXPECTATION,
    ) -> None:
        locked = validate_e26_final_tokenizer_receipt(receipt, expectation=expectation)
        try:
            sentencepiece = importlib.import_module("sentencepiece")
        except ModuleNotFoundError as error:
            raise E26FinalTokenizerError(
                "sentencepiece is required for the locked E26 Final tokenizer runtime"
            ) from error
        processor_class = getattr(sentencepiece, "SentencePieceProcessor", None)
        if processor_class is None:
            raise E26FinalTokenizerError("sentencepiece lacks SentencePieceProcessor")
        model_path = Path(str(locked["bundle_root"])) / "tokenizer.model"
        processor = processor_class(model_file=str(model_path))
        if (
            int(processor.vocab_size()) != expectation.vocab_size
            or int(processor.unk_id()) != expectation.unk_token_id
            or int(processor.bos_id()) != expectation.bos_token_id
            or int(processor.eos_id()) != expectation.eos_token_id
        ):
            raise E26FinalTokenizerError("SentencePiece runtime IDs differ from the lock")
        self._processor = processor
        self.receipt = locked
        self.vocab_size = expectation.vocab_size
        self.unk_token_id = expectation.unk_token_id
        self.bos_token_id = expectation.bos_token_id
        self.eos_token_id = expectation.eos_token_id
        self.pad_token_id = expectation.pad_token_id
        self.document_separator_id = expectation.document_separator_id

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("Tokenizer input must be text")
        raw = self._processor.encode(text, out_type=int)
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
            raise E26FinalTokenizerError("SentencePiece returned a non-sequence encoding")
        ids = [int(value) for value in raw]
        if add_bos:
            ids.insert(0, self.bos_token_id)
        if add_eos:
            ids.append(self.eos_token_id)
        if any(value < 0 or value >= self.vocab_size for value in ids):
            raise E26FinalTokenizerError("Tokenizer emitted an out-of-range token ID")
        return ids

    def encode_batch(
        self,
        texts: Sequence[str],
        *,
        num_threads: int,
    ) -> list[list[int]]:
        """Encode a fixed ordered batch without adding implicit boundaries."""

        if isinstance(num_threads, bool) or not isinstance(num_threads, int) or num_threads < 1:
            raise ValueError("num_threads must be a positive integer")
        values = list(texts)
        if any(not isinstance(text, str) for text in values):
            raise TypeError("Every tokenizer batch input must be text")
        raw = self._processor.encode(values, out_type=int, num_threads=num_threads)
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
            raise E26FinalTokenizerError("SentencePiece returned a malformed batch encoding")
        result: list[list[int]] = []
        for row in raw:
            if not isinstance(row, Sequence) or isinstance(row, (str, bytes, bytearray)):
                raise E26FinalTokenizerError("SentencePiece returned a malformed batch row")
            ids = [int(value) for value in row]
            if any(value < 0 or value >= self.vocab_size for value in ids):
                raise E26FinalTokenizerError("Tokenizer emitted an out-of-range token ID")
            result.append(ids)
        if len(result) != len(values):
            raise E26FinalTokenizerError("SentencePiece changed batch cardinality")
        return result


__all__ = [
    "DEFAULT_TOKENIZER_EXPECTATION",
    "DEFAULT_TOKENIZER_ROOT",
    "DOCUMENT_SEPARATOR_POLICY",
    "E26FinalSentencePieceTokenizer",
    "E26FinalTokenizerError",
    "E26FinalTokenizerExpectation",
    "TOKENIZER_BACKEND",
    "TOKENIZER_RECEIPT_SCHEMA",
    "build_e26_final_tokenizer_receipt",
    "validate_e26_final_tokenizer_receipt",
    "write_e26_final_tokenizer_receipt",
]
