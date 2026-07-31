from __future__ import annotations

import importlib
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from catena.core.provenance_v61 import (
    SHA256_PATTERN,
    read_json_object_strict,
    sha256_canonical_json,
    sha256_file,
)

SCIENTIFIC_VOCAB_SIZE = 16_384
SCIENTIFIC_TOKENIZER_FAMILIES = frozenset({"BPE", "UNIGRAM"})


class ScientificTokenizerContractError(ValueError):
    """Raised when an external tokenizer cannot be admitted to E26 MAIN."""


class Tokenizer(Protocol):
    @property
    def vocab_size(self) -> int: ...

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]: ...

    def decode(self, token_ids: Sequence[int]) -> str: ...


@dataclass(frozen=True, slots=True)
class TokenizerManifest:
    tokenizer_id: str
    tokenizer_type: str
    vocab_size: int
    model_path: str | None
    model_sha256: str | None
    training_manifest_sha256: str | None
    special_tokens: dict[str, int]
    schema_version: str = "catena-v8.1"
    tokenizer_family: str | None = None
    training_manifest_path: str | None = None
    manifest_path: str | None = None
    manifest_sha256: str | None = None
    evidence_tier: str = "NON_EVIDENCE_VALIDATION"
    scientific_main_eligible: bool = False
    synthetic: bool = True
    reference_only: bool = True
    shared_across_variants: bool = True
    trained_once: bool = True
    training_document_selection_sha256: str | None = None

    @property
    def manifest_hash(self) -> str:
        if self.manifest_sha256 is not None:
            return self.manifest_sha256
        return sha256_canonical_json(asdict(self))

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def assert_scientific_main(self) -> None:
        if not self.scientific_main_eligible:
            raise ScientificTokenizerContractError(
                "Tokenizer manifest is not eligible for scientific MAIN"
            )
        if self.synthetic or self.reference_only:
            raise ScientificTokenizerContractError(
                "Synthetic/reference tokenizer cannot be used for scientific MAIN"
            )
        if self.vocab_size != SCIENTIFIC_VOCAB_SIZE:
            raise ScientificTokenizerContractError(
                f"E26 MAIN requires exactly {SCIENTIFIC_VOCAB_SIZE} tokens"
            )
        if self.tokenizer_family not in SCIENTIFIC_TOKENIZER_FAMILIES:
            raise ScientificTokenizerContractError(
                f"Tokenizer family must be one of {sorted(SCIENTIFIC_TOKENIZER_FAMILIES)}"
            )
        required_hashes = (
            self.model_sha256,
            self.training_manifest_sha256,
            self.training_document_selection_sha256,
            self.manifest_sha256,
        )
        if any(value is None or not SHA256_PATTERN.fullmatch(value) for value in required_hashes):
            raise ScientificTokenizerContractError(
                "Scientific tokenizer lacks complete SHA-256 provenance"
            )


class ByteTokenizer:
    """Deterministic UTF-8 byte tokenizer for non-evidence validation only."""

    pad_id = 0
    bos_id = 1
    eos_id = 2
    byte_offset = 3

    @property
    def vocab_size(self) -> int:
        return 259

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        output: list[int] = []
        if add_bos:
            output.append(self.bos_id)
        output.extend(self.byte_offset + value for value in text.encode("utf-8"))
        if add_eos:
            output.append(self.eos_id)
        return output

    def decode(self, token_ids: Sequence[int]) -> str:
        values = bytes(
            token_id - self.byte_offset
            for token_id in token_ids
            if self.byte_offset <= token_id < self.byte_offset + 256
        )
        return values.decode("utf-8", errors="replace")

    def manifest(self) -> TokenizerManifest:
        return TokenizerManifest(
            tokenizer_id="byte-tokenizer-non-evidence",
            tokenizer_type="BYTE_REFERENCE_ONLY",
            vocab_size=self.vocab_size,
            model_path=None,
            model_sha256=None,
            training_manifest_sha256=None,
            special_tokens={"pad": self.pad_id, "bos": self.bos_id, "eos": self.eos_id},
        )


class ExternalScientificTokenizer:
    """Runtime adapter for one hash-validated Hugging Face tokenizer JSON.

    Import is lazy so the existing CATENA environment remains untouched during
    reference validation. Scientific E26 execution fails closed when the
    optional ``tokenizers`` runtime is absent or its loaded vocabulary differs
    from the frozen 16K manifest.
    """

    def __init__(self, manifest: TokenizerManifest, runtime: Any) -> None:
        manifest.assert_scientific_main()
        observed_vocab = runtime.get_vocab_size(with_added_tokens=True)
        if observed_vocab != manifest.vocab_size:
            raise ScientificTokenizerContractError(
                "Runtime tokenizer vocabulary differs from the frozen manifest"
            )
        self._manifest = manifest
        self._runtime = runtime

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str | Path,
    ) -> ExternalScientificTokenizer:
        manifest = load_scientific_tokenizer_manifest(manifest_path)
        try:
            module = importlib.import_module("tokenizers")
        except ModuleNotFoundError as error:
            raise ScientificTokenizerContractError(
                "Scientific tokenizer runtime requires the optional 'tokenizers' package"
            ) from error
        runtime_class = getattr(module, "Tokenizer", None)
        if runtime_class is None or not hasattr(runtime_class, "from_file"):
            raise ScientificTokenizerContractError(
                "Installed tokenizers package lacks Tokenizer.from_file"
            )
        runtime = runtime_class.from_file(str(manifest.model_path))
        return cls(manifest, runtime)

    @property
    def vocab_size(self) -> int:
        return self._manifest.vocab_size

    @property
    def manifest(self) -> TokenizerManifest:
        return self._manifest

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        encoding = self._runtime.encode(text, add_special_tokens=False)
        token_ids = [int(value) for value in encoding.ids]
        if add_bos:
            token_ids.insert(0, self._manifest.special_tokens["bos"])
        if add_eos:
            token_ids.append(self._manifest.special_tokens["eos"])
        if any(not 0 <= token_id < self.vocab_size for token_id in token_ids):
            raise ScientificTokenizerContractError(
                "Runtime tokenizer emitted an out-of-vocabulary token ID"
            )
        return token_ids

    def decode(self, token_ids: Sequence[int]) -> str:
        if any(not 0 <= int(token_id) < self.vocab_size for token_id in token_ids):
            raise ScientificTokenizerContractError("Cannot decode an out-of-vocabulary token ID")
        return str(
            self._runtime.decode(
                [int(value) for value in token_ids],
                skip_special_tokens=False,
            )
        )


def _require_bool(payload: dict[str, Any], field: str, expected: bool) -> None:
    value = payload.get(field)
    if value is not expected:
        raise ScientificTokenizerContractError(f"{field} must be {expected!r}, got {value!r}")


def _require_nonempty_string(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ScientificTokenizerContractError(f"{field} must be a non-empty string")
    return value


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
        raise ScientificTokenizerContractError(
            f"{field} must be 64 lowercase hexadecimal characters"
        )
    return value


def _resolve_recorded_file(manifest_path: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ScientificTokenizerContractError(f"{field} must be a non-empty path")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = manifest_path.parent / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as error:
        raise ScientificTokenizerContractError(f"{field} is missing: {candidate}") from error
    if not resolved.is_file():
        raise ScientificTokenizerContractError(f"{field} is not a regular file: {resolved}")
    return resolved


def _validate_training_manifest(path: Path) -> tuple[str, str]:
    payload = read_json_object_strict(path)
    if payload.get("schema_version") != "catena-v8.1":
        raise ScientificTokenizerContractError(
            "Tokenizer training manifest must use schema_version=catena-v8.1"
        )
    if payload.get("manifest_type") != "E26_TOKENIZER_TRAINING_DOCUMENTS":
        raise ScientificTokenizerContractError(
            "Tokenizer training manifest has the wrong manifest_type"
        )
    _require_bool(payload, "selection_frozen", True)
    _require_bool(payload, "synthetic", False)
    _require_bool(payload, "reference_only", False)
    document_count = payload.get("document_count")
    if (
        isinstance(document_count, bool)
        or not isinstance(document_count, int)
        or document_count < 1
    ):
        raise ScientificTokenizerContractError(
            "Tokenizer training manifest requires document_count >= 1"
        )
    source_revisions = payload.get("source_revisions")
    if (
        not isinstance(source_revisions, list)
        or not source_revisions
        or any(not isinstance(item, str) or not item for item in source_revisions)
    ):
        raise ScientificTokenizerContractError(
            "Tokenizer training manifest requires pinned source_revisions"
        )
    selection_sha256 = _require_sha256(
        payload.get("document_selection_sha256"),
        "training_manifest.document_selection_sha256",
    )
    return sha256_file(path), selection_sha256


def load_scientific_tokenizer_manifest(
    manifest_path: str | Path,
    *,
    expected_vocab_size: int = SCIENTIFIC_VOCAB_SIZE,
) -> TokenizerManifest:
    """Validate and pin an already materialized external 16K tokenizer.

    This loader performs no download and no tokenizer training.  Paths may be
    absolute or relative to the manifest, but the manifest pins the exact
    bytes, byte counts, training-document selection, and shared-use policy.
    """

    path = Path(manifest_path).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ScientificTokenizerContractError(f"Tokenizer manifest is not a file: {path}")
    payload = read_json_object_strict(path)
    if payload.get("schema_version") != "catena-v8.1":
        raise ScientificTokenizerContractError(
            "Tokenizer manifest must use schema_version=catena-v8.1"
        )
    if payload.get("manifest_type") != "E26_SCIENTIFIC_TOKENIZER":
        raise ScientificTokenizerContractError("Tokenizer manifest has the wrong manifest_type")
    if payload.get("evidence_tier") != "SCIENTIFIC_INPUT":
        raise ScientificTokenizerContractError("Tokenizer evidence_tier must be SCIENTIFIC_INPUT")
    _require_bool(payload, "scientific_main_eligible", True)
    _require_bool(payload, "synthetic", False)
    _require_bool(payload, "reference_only", False)
    _require_bool(payload, "shared_across_variants", True)
    _require_bool(payload, "trained_once", True)

    tokenizer_id = _require_nonempty_string(payload, "tokenizer_id")
    tokenizer_type = _require_nonempty_string(payload, "tokenizer_type")
    if tokenizer_type != "EXTERNAL_SCIENTIFIC_TOKENIZER":
        raise ScientificTokenizerContractError(
            "tokenizer_type must be EXTERNAL_SCIENTIFIC_TOKENIZER"
        )
    family = _require_nonempty_string(payload, "tokenizer_family").upper()
    if family not in SCIENTIFIC_TOKENIZER_FAMILIES:
        raise ScientificTokenizerContractError(
            f"tokenizer_family must be one of {sorted(SCIENTIFIC_TOKENIZER_FAMILIES)}"
        )
    vocab_size = payload.get("vocab_size")
    if (
        isinstance(vocab_size, bool)
        or not isinstance(vocab_size, int)
        or vocab_size != expected_vocab_size
    ):
        raise ScientificTokenizerContractError(
            f"vocab_size must equal the locked value {expected_vocab_size}"
        )

    model = payload.get("model")
    if not isinstance(model, dict):
        raise ScientificTokenizerContractError("model must be an object")
    model_path = _resolve_recorded_file(path, model.get("path"), "model.path")
    expected_model_hash = _require_sha256(model.get("sha256"), "model.sha256")
    if sha256_file(model_path) != expected_model_hash:
        raise ScientificTokenizerContractError("Tokenizer model SHA-256 mismatch")
    model_bytes = model.get("bytes")
    if isinstance(model_bytes, bool) or not isinstance(model_bytes, int):
        raise ScientificTokenizerContractError("model.bytes must be an integer")
    if model_path.stat().st_size != model_bytes:
        raise ScientificTokenizerContractError("Tokenizer model byte count mismatch")

    training = payload.get("training_manifest")
    if not isinstance(training, dict):
        raise ScientificTokenizerContractError("training_manifest must be an object")
    training_path = _resolve_recorded_file(
        path,
        training.get("path"),
        "training_manifest.path",
    )
    expected_training_hash = _require_sha256(
        training.get("sha256"),
        "training_manifest.sha256",
    )
    training_hash, selection_hash = _validate_training_manifest(training_path)
    if training_hash != expected_training_hash:
        raise ScientificTokenizerContractError("Tokenizer training manifest SHA-256 mismatch")

    special_tokens = payload.get("special_tokens")
    required_specials = {"pad", "bos", "eos", "unk"}
    if not isinstance(special_tokens, dict) or not required_specials.issubset(special_tokens):
        raise ScientificTokenizerContractError(
            f"special_tokens must define {sorted(required_specials)}"
        )
    normalized_specials: dict[str, int] = {}
    for name, value in special_tokens.items():
        if (
            not isinstance(name, str)
            or isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value < vocab_size
        ):
            raise ScientificTokenizerContractError(f"Invalid special token record: {name!r}")
        normalized_specials[name] = value
    if len(set(normalized_specials.values())) != len(normalized_specials):
        raise ScientificTokenizerContractError("Special token IDs must be unique")

    manifest = TokenizerManifest(
        tokenizer_id=tokenizer_id,
        tokenizer_type=tokenizer_type,
        tokenizer_family=family,
        vocab_size=vocab_size,
        model_path=str(model_path),
        model_sha256=expected_model_hash,
        training_manifest_path=str(training_path),
        training_manifest_sha256=training_hash,
        training_document_selection_sha256=selection_hash,
        special_tokens=normalized_specials,
        manifest_path=str(path),
        manifest_sha256=sha256_file(path),
        evidence_tier="SCIENTIFIC_INPUT",
        scientific_main_eligible=True,
        synthetic=False,
        reference_only=False,
        shared_across_variants=True,
        trained_once=True,
    )
    manifest.assert_scientific_main()
    return manifest


@dataclass(frozen=True, slots=True)
class ExternalTokenizerContract:
    """Compatibility adapter plus strict manifest admission for scientific MAIN."""

    model_path: Path
    vocab_size: int
    tokenizer_id: str
    training_manifest_path: Path
    special_tokens: dict[str, int]
    tokenizer_family: str = "BPE"
    manifest_path: Path | None = None

    @classmethod
    def from_manifest(cls, manifest_path: str | Path) -> ExternalTokenizerContract:
        manifest = load_scientific_tokenizer_manifest(manifest_path)
        return cls(
            model_path=Path(manifest.model_path or ""),
            vocab_size=manifest.vocab_size,
            tokenizer_id=manifest.tokenizer_id,
            training_manifest_path=Path(manifest.training_manifest_path or ""),
            special_tokens=dict(manifest.special_tokens),
            tokenizer_family=manifest.tokenizer_family or "",
            manifest_path=Path(manifest.manifest_path or ""),
        )

    def validate(self, *, scientific_main: bool = False) -> TokenizerManifest:
        if scientific_main:
            if self.manifest_path is None:
                raise ScientificTokenizerContractError(
                    "Scientific MAIN requires a frozen tokenizer manifest, not loose paths"
                )
            manifest = load_scientific_tokenizer_manifest(self.manifest_path)
            expected = (
                self.model_path.expanduser().resolve(),
                self.vocab_size,
                self.tokenizer_id,
                self.training_manifest_path.expanduser().resolve(),
                self.special_tokens,
                self.tokenizer_family.upper(),
            )
            observed = (
                Path(manifest.model_path or "").resolve(),
                manifest.vocab_size,
                manifest.tokenizer_id,
                Path(manifest.training_manifest_path or "").resolve(),
                manifest.special_tokens,
                manifest.tokenizer_family,
            )
            if observed != expected:
                raise ScientificTokenizerContractError(
                    "Loose tokenizer contract differs from its frozen manifest"
                )
            return manifest

        if not self.model_path.is_file():
            raise FileNotFoundError(self.model_path)
        if not self.training_manifest_path.is_file():
            raise FileNotFoundError(self.training_manifest_path)
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        return TokenizerManifest(
            tokenizer_id=self.tokenizer_id,
            tokenizer_type="EXTERNAL_VALIDATION_ONLY",
            tokenizer_family=self.tokenizer_family.upper(),
            vocab_size=self.vocab_size,
            model_path=str(self.model_path.resolve()),
            model_sha256=sha256_file(self.model_path),
            training_manifest_path=str(self.training_manifest_path.resolve()),
            training_manifest_sha256=sha256_file(self.training_manifest_path),
            special_tokens=dict(self.special_tokens),
            synthetic=False,
            reference_only=True,
        )
