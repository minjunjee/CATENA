from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from catena.core.provenance_v61 import sha256_canonical_json

from .general_corpus import PairedTokenCursor, ScientificCorpusContractError, TokenMemmap
from .tokenizer import Tokenizer
from .transactional_stream import Operation, QueryType, generate_episode


class PairedStreamContractError(ValueError):
    """Raised when a paired or resumed stream is not an exact replay."""


class StreamSource(StrEnum):
    GENERAL = "general"
    TRANSACTION = "transaction"


@dataclass(frozen=True, slots=True)
class TrainingSequence:
    token_ids: NDArray[np.int64]
    source_type: str
    source_id: str
    source_index: int
    token_offset: int | None
    unpadded_tokens: int
    padding_tokens: int
    reset_state: bool
    query_type: str | None
    packed_examples: int = 1
    component_source_ids: tuple[str, ...] = ()
    component_query_types: tuple[str, ...] = ()

    def audit_record(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "source_id": self.source_id,
            "source_index": self.source_index,
            "token_offset": self.token_offset,
            "sequence_length": int(self.token_ids.size),
            "unpadded_tokens": self.unpadded_tokens,
            "padding_tokens": self.padding_tokens,
            "reset_state": self.reset_state,
            "query_type": self.query_type,
            "packed_examples": self.packed_examples,
            "component_source_ids": list(self.component_source_ids),
            "component_query_types": list(self.component_query_types),
            "token_bytes_sha256": hashlib.sha256(
                np.asarray(self.token_ids, dtype="<u2").tobytes(order="C")
            ).hexdigest(),
        }


@dataclass(frozen=True, slots=True)
class TrainingCursorReceipt:
    start_sequence_index: int
    end_sequence_index: int
    sequences: int
    tokens: int
    general_sequences: int
    transaction_sequences: int
    metadata_sha256: str
    token_bytes_sha256: str
    data_order_sha256: str
    loss_bearing_tokens: int = 0
    general_unpadded_tokens: int = 0
    transaction_unpadded_tokens: int = 0
    padding_tokens: int = 0
    realized_general_fraction: float = 0.0
    realized_transaction_fraction: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _training_receipt(
    *,
    algorithm: str,
    start_sequence_index: int,
    end_sequence_index: int,
    sequence_length: int,
    sequences: Sequence[TrainingSequence],
) -> TrainingCursorReceipt:
    records = [sequence.audit_record() for sequence in sequences]
    metadata_sha256 = sha256_canonical_json(records)
    token_digest = hashlib.sha256()
    for sequence in sequences:
        encoded = np.asarray(sequence.token_ids, dtype="<u2")
        token_digest.update(len(encoded).to_bytes(8, "big"))
        token_digest.update(encoded.tobytes(order="C"))
    token_bytes_sha256 = token_digest.hexdigest()
    general_sequences = sum(
        sequence.source_type == StreamSource.GENERAL.value for sequence in sequences
    )
    transaction_sequences = len(sequences) - general_sequences
    general_tokens = sum(
        sequence.unpadded_tokens
        for sequence in sequences
        if sequence.source_type == StreamSource.GENERAL.value
    )
    transaction_tokens = sum(
        sequence.unpadded_tokens
        for sequence in sequences
        if sequence.source_type == StreamSource.TRANSACTION.value
    )
    loss_bearing_tokens = general_tokens + transaction_tokens
    padding_tokens = sum(sequence.padding_tokens for sequence in sequences)
    general_fraction = general_tokens / loss_bearing_tokens if loss_bearing_tokens else 0.0
    transaction_fraction = transaction_tokens / loss_bearing_tokens if loss_bearing_tokens else 0.0
    receipt_payload = {
        "cursor_algorithm": algorithm,
        "start_sequence_index": start_sequence_index,
        "end_sequence_index": end_sequence_index,
        "sequences": len(sequences),
        "tokens": len(sequences) * sequence_length,
        "general_sequences": general_sequences,
        "transaction_sequences": transaction_sequences,
        "metadata_sha256": metadata_sha256,
        "token_bytes_sha256": token_bytes_sha256,
        "loss_bearing_tokens": loss_bearing_tokens,
        "general_unpadded_tokens": general_tokens,
        "transaction_unpadded_tokens": transaction_tokens,
        "padding_tokens": padding_tokens,
        "realized_general_fraction": general_fraction,
        "realized_transaction_fraction": transaction_fraction,
    }
    return TrainingCursorReceipt(
        start_sequence_index=start_sequence_index,
        end_sequence_index=end_sequence_index,
        sequences=len(sequences),
        tokens=len(sequences) * sequence_length,
        general_sequences=general_sequences,
        transaction_sequences=transaction_sequences,
        metadata_sha256=metadata_sha256,
        token_bytes_sha256=token_bytes_sha256,
        data_order_sha256=sha256_canonical_json(receipt_payload),
        loss_bearing_tokens=loss_bearing_tokens,
        general_unpadded_tokens=general_tokens,
        transaction_unpadded_tokens=transaction_tokens,
        padding_tokens=padding_tokens,
        realized_general_fraction=general_fraction,
        realized_transaction_fraction=transaction_fraction,
    )


class PairedTransactionCursor:
    """Counter-derived transaction stream with no model-variant input."""

    _ALGORITHM = "transaction_counter_grid_v1"

    def __init__(
        self,
        tokenizer: Tokenizer,
        *,
        tokenizer_hash: str,
        seed: int,
        sequence_length: int,
        pad_token_id: int,
        split: str = "train",
        domains: Sequence[str] = (
            "access_control",
            "api_configuration",
            "workflow",
            "versioned_preference",
        ),
        operations: Sequence[Operation | str] = tuple(Operation),
        query_types: Sequence[QueryType | str] = tuple(QueryType),
        distractor_units: int = 1,
        start_sequence_index: int = 0,
    ) -> None:
        if sequence_length <= 1:
            raise ValueError("sequence_length must exceed one")
        if isinstance(start_sequence_index, bool) or start_sequence_index < 0:
            raise ValueError("start_sequence_index must be non-negative")
        if not domains or not operations or not query_types:
            raise ValueError("transaction schedule dimensions cannot be empty")
        if not 0 <= pad_token_id < tokenizer.vocab_size:
            raise ValueError("pad_token_id is outside the tokenizer vocabulary")
        self.tokenizer = tokenizer
        self.tokenizer_hash = str(tokenizer_hash)
        self.seed = int(seed)
        self.sequence_length = int(sequence_length)
        self.pad_token_id = int(pad_token_id)
        self.split = str(split)
        self.domains = tuple(str(value) for value in domains)
        self.operations = tuple(Operation(value).value for value in operations)
        self.query_types = tuple(QueryType(value).value for value in query_types)
        self.distractor_units = int(distractor_units)
        self.sequence_index = int(start_sequence_index)

    def _schedule_record(self, sequence_index: int) -> tuple[str, str, str]:
        domain_index = sequence_index % len(self.domains)
        operation_index = (sequence_index // len(self.domains)) % len(self.operations)
        query_index = (sequence_index // (len(self.domains) * len(self.operations))) % len(
            self.query_types
        )
        return (
            self.domains[domain_index],
            self.operations[operation_index],
            self.query_types[query_index],
        )

    def _config_payload(self) -> dict[str, Any]:
        return {
            "cursor_algorithm": self._ALGORITHM,
            "tokenizer_hash": self.tokenizer_hash,
            "seed": self.seed,
            "sequence_length": self.sequence_length,
            "pad_token_id": self.pad_token_id,
            "split": self.split,
            "domains": list(self.domains),
            "operations": list(self.operations),
            "query_types": list(self.query_types),
            "distractor_units": self.distractor_units,
        }

    def snapshot(self) -> dict[str, Any]:
        payload = {
            "schema_version": "catena-v8.1",
            **self._config_payload(),
            "sequence_index": self.sequence_index,
            "tokens_emitted": self.sequence_index * self.sequence_length,
        }
        payload["snapshot_sha256"] = sha256_canonical_json(payload)
        return payload

    @classmethod
    def from_snapshot(
        cls,
        tokenizer: Tokenizer,
        *,
        tokenizer_hash: str,
        snapshot: Mapping[str, Any],
    ) -> PairedTransactionCursor:
        payload = dict(snapshot)
        observed_hash = payload.pop("snapshot_sha256", None)
        if not isinstance(observed_hash, str) or observed_hash != sha256_canonical_json(payload):
            raise PairedStreamContractError("Transaction cursor snapshot SHA-256 mismatch")
        if payload.get("schema_version") != "catena-v8.1":
            raise PairedStreamContractError("Unsupported transaction cursor snapshot schema")
        if payload.get("cursor_algorithm") != cls._ALGORITHM:
            raise PairedStreamContractError("Unsupported transaction cursor algorithm")
        if payload.get("tokenizer_hash") != tokenizer_hash:
            raise PairedStreamContractError("Transaction cursor tokenizer changed")
        sequence_index = payload.get("sequence_index")
        sequence_length = payload.get("sequence_length")
        tokens_emitted = payload.get("tokens_emitted")
        if (
            isinstance(sequence_index, bool)
            or not isinstance(sequence_index, int)
            or isinstance(sequence_length, bool)
            or not isinstance(sequence_length, int)
            or tokens_emitted != sequence_index * sequence_length
        ):
            raise PairedStreamContractError("Invalid transaction cursor exposure counters")
        return cls(
            tokenizer,
            tokenizer_hash=tokenizer_hash,
            seed=int(payload["seed"]),
            sequence_length=sequence_length,
            pad_token_id=int(payload["pad_token_id"]),
            split=str(payload["split"]),
            domains=tuple(str(value) for value in payload["domains"]),
            operations=tuple(str(value) for value in payload["operations"]),
            query_types=tuple(str(value) for value in payload["query_types"]),
            distractor_units=int(payload["distractor_units"]),
            start_sequence_index=sequence_index,
        )

    def fork(self) -> PairedTransactionCursor:
        return self.from_snapshot(
            self.tokenizer,
            tokenizer_hash=self.tokenizer_hash,
            snapshot=self.snapshot(),
        )

    def take(self, count: int) -> list[TrainingSequence]:
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("count must be a non-negative integer")
        output: list[TrainingSequence] = []
        for sequence_index in range(self.sequence_index, self.sequence_index + count):
            domain, operation, query_type = self._schedule_record(sequence_index)
            episode = generate_episode(
                seed=self.seed,
                split=self.split,
                domain=domain,
                operation=operation,
                index=sequence_index,
                distractor_units=self.distractor_units,
            )
            encoded = self.tokenizer.encode(
                episode.render_training_example(query_type),
                add_bos=True,
                add_eos=True,
            )
            unpadded = min(len(encoded), self.sequence_length)
            fixed = encoded[: self.sequence_length]
            padding = self.sequence_length - len(fixed)
            if padding:
                fixed.extend([self.pad_token_id] * padding)
            token_ids = np.asarray(fixed, dtype=np.int64)
            if token_ids.size != self.sequence_length:
                raise AssertionError("Transaction cursor failed to emit one fixed-length sequence")
            output.append(
                TrainingSequence(
                    token_ids=token_ids,
                    source_type=StreamSource.TRANSACTION.value,
                    source_id=episode.episode_id,
                    source_index=sequence_index,
                    token_offset=None,
                    unpadded_tokens=unpadded,
                    padding_tokens=padding,
                    reset_state=True,
                    query_type=query_type,
                )
            )
        self.sequence_index += count
        return output


class PackedTransactionCursor:
    """Pack complete transaction examples without truncation into fixed contexts."""

    _ALGORITHM = "complete_example_transaction_pack_v2"

    def __init__(
        self,
        tokenizer: Tokenizer,
        *,
        tokenizer_hash: str,
        seed: int,
        sequence_length: int,
        pad_token_id: int,
        split: str = "train",
        domains: Sequence[str] = (
            "access_control",
            "api_configuration",
            "workflow",
            "versioned_preference",
        ),
        operations: Sequence[Operation | str] = tuple(Operation),
        query_types: Sequence[QueryType | str] = tuple(QueryType),
        distractor_units: int = 1,
        start_sequence_index: int = 0,
        start_episode_index: int = 0,
        start_unpadded_tokens: int = 0,
    ) -> None:
        if sequence_length <= 1:
            raise ValueError("sequence_length must exceed one")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (
                start_sequence_index,
                start_episode_index,
                start_unpadded_tokens,
            )
        ):
            raise ValueError("packed cursor counters must be non-negative integers")
        if not domains or not operations or not query_types:
            raise ValueError("transaction schedule dimensions cannot be empty")
        if not 0 <= pad_token_id < tokenizer.vocab_size:
            raise ValueError("pad_token_id is outside the tokenizer vocabulary")
        self.tokenizer = tokenizer
        self.tokenizer_hash = str(tokenizer_hash)
        self.seed = int(seed)
        self.sequence_length = int(sequence_length)
        self.pad_token_id = int(pad_token_id)
        self.split = str(split)
        self.domains = tuple(str(value) for value in domains)
        self.operations = tuple(Operation(value).value for value in operations)
        self.query_types = tuple(QueryType(value).value for value in query_types)
        self.distractor_units = int(distractor_units)
        self.sequence_index = int(start_sequence_index)
        self.episode_index = int(start_episode_index)
        self.unpadded_tokens_emitted = int(start_unpadded_tokens)

    def _schedule_record(self, episode_index: int) -> tuple[str, str, str]:
        domain_index = episode_index % len(self.domains)
        operation_index = (episode_index // len(self.domains)) % len(self.operations)
        query_index = (episode_index // (len(self.domains) * len(self.operations))) % len(
            self.query_types
        )
        return (
            self.domains[domain_index],
            self.operations[operation_index],
            self.query_types[query_index],
        )

    def _config_payload(self) -> dict[str, Any]:
        return {
            "cursor_algorithm": self._ALGORITHM,
            "tokenizer_hash": self.tokenizer_hash,
            "seed": self.seed,
            "sequence_length": self.sequence_length,
            "pad_token_id": self.pad_token_id,
            "split": self.split,
            "domains": list(self.domains),
            "operations": list(self.operations),
            "query_types": list(self.query_types),
            "distractor_units": self.distractor_units,
        }

    def snapshot(self) -> dict[str, Any]:
        payload = {
            "schema_version": "catena-v8.1",
            **self._config_payload(),
            "sequence_index": self.sequence_index,
            "episode_index": self.episode_index,
            "unpadded_tokens_emitted": self.unpadded_tokens_emitted,
            "allocated_tokens_emitted": self.sequence_index * self.sequence_length,
        }
        payload["snapshot_sha256"] = sha256_canonical_json(payload)
        return payload

    @classmethod
    def from_snapshot(
        cls,
        tokenizer: Tokenizer,
        *,
        tokenizer_hash: str,
        snapshot: Mapping[str, Any],
    ) -> PackedTransactionCursor:
        payload = dict(snapshot)
        observed_hash = payload.pop("snapshot_sha256", None)
        if not isinstance(observed_hash, str) or observed_hash != sha256_canonical_json(payload):
            raise PairedStreamContractError("Packed transaction cursor snapshot SHA-256 mismatch")
        if (
            payload.get("schema_version") != "catena-v8.1"
            or payload.get("cursor_algorithm") != cls._ALGORITHM
            or payload.get("tokenizer_hash") != tokenizer_hash
        ):
            raise PairedStreamContractError("Packed transaction cursor configuration changed")
        sequence_index = payload.get("sequence_index")
        sequence_length = payload.get("sequence_length")
        episode_index = payload.get("episode_index")
        unpadded = payload.get("unpadded_tokens_emitted")
        allocated = payload.get("allocated_tokens_emitted")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (sequence_index, sequence_length, episode_index, unpadded)
        ):
            raise PairedStreamContractError("Invalid packed transaction exposure counters")
        sequence_index_int = cast(int, sequence_index)
        sequence_length_int = cast(int, sequence_length)
        episode_index_int = cast(int, episode_index)
        unpadded_int = cast(int, unpadded)
        if allocated != sequence_index_int * sequence_length_int:
            raise PairedStreamContractError("Invalid packed transaction allocation counter")
        return cls(
            tokenizer,
            tokenizer_hash=tokenizer_hash,
            seed=int(payload["seed"]),
            sequence_length=sequence_length_int,
            pad_token_id=int(payload["pad_token_id"]),
            split=str(payload["split"]),
            domains=tuple(str(value) for value in payload["domains"]),
            operations=tuple(str(value) for value in payload["operations"]),
            query_types=tuple(str(value) for value in payload["query_types"]),
            distractor_units=int(payload["distractor_units"]),
            start_sequence_index=sequence_index_int,
            start_episode_index=episode_index_int,
            start_unpadded_tokens=unpadded_int,
        )

    def fork(self) -> PackedTransactionCursor:
        return self.from_snapshot(
            self.tokenizer,
            tokenizer_hash=self.tokenizer_hash,
            snapshot=self.snapshot(),
        )

    def _encoded_example(self, episode_index: int) -> tuple[list[int], str, str]:
        domain, operation, query_type = self._schedule_record(episode_index)
        episode = generate_episode(
            seed=self.seed,
            split=self.split,
            domain=domain,
            operation=operation,
            index=episode_index,
            distractor_units=self.distractor_units,
        )
        encoded = self.tokenizer.encode(
            episode.render_training_example(query_type),
            add_bos=True,
            add_eos=True,
        )
        if not encoded or len(encoded) > self.sequence_length:
            raise PairedStreamContractError(
                "A complete transaction example does not fit the locked context"
            )
        return encoded, episode.episode_id, query_type

    def take(self, count: int) -> list[TrainingSequence]:
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("count must be a non-negative integer")
        output: list[TrainingSequence] = []
        for _ in range(count):
            sequence_index = self.sequence_index
            start_episode = self.episode_index
            tokens: list[int] = []
            episode_ids: list[str] = []
            query_types: list[str] = []
            while True:
                encoded, episode_id, query_type = self._encoded_example(self.episode_index)
                if tokens and len(tokens) + len(encoded) > self.sequence_length:
                    break
                tokens.extend(encoded)
                episode_ids.append(episode_id)
                query_types.append(query_type)
                self.episode_index += 1
                if len(tokens) == self.sequence_length:
                    break
            unpadded = len(tokens)
            padding = self.sequence_length - unpadded
            tokens.extend([self.pad_token_id] * padding)
            self.sequence_index += 1
            self.unpadded_tokens_emitted += unpadded
            source_id = sha256_canonical_json(
                {
                    "first_episode_index": start_episode,
                    "episode_ids": episode_ids,
                    "query_types": query_types,
                }
            )
            output.append(
                TrainingSequence(
                    token_ids=np.asarray(tokens, dtype=np.int64),
                    source_type=StreamSource.TRANSACTION.value,
                    source_id=source_id,
                    source_index=sequence_index,
                    token_offset=None,
                    unpadded_tokens=unpadded,
                    padding_tokens=padding,
                    reset_state=True,
                    query_type="packed_complete_examples",
                    packed_examples=len(episode_ids),
                    component_source_ids=tuple(episode_ids),
                    component_query_types=tuple(query_types),
                )
            )
        return output


class PairedTrainingCursor:
    """Exact 4:1 general/transaction schedule shared by paired variants."""

    _ALGORITHM = "fixed_4_general_1_transaction_v1"
    _CYCLE = (
        StreamSource.GENERAL,
        StreamSource.GENERAL,
        StreamSource.GENERAL,
        StreamSource.GENERAL,
        StreamSource.TRANSACTION,
    )

    def __init__(
        self,
        general: PairedTokenCursor,
        transaction: PairedTransactionCursor,
        *,
        start_sequence_index: int = 0,
    ) -> None:
        if general.sequence_length != transaction.sequence_length:
            raise ValueError("General and transaction sequence lengths must match")
        if isinstance(start_sequence_index, bool) or start_sequence_index < 0:
            raise ValueError("start_sequence_index must be non-negative")
        self.general = general
        self.transaction = transaction
        self.sequence_length = general.sequence_length
        self.sequence_index = int(start_sequence_index)

    @classmethod
    def _source_at(cls, sequence_index: int) -> StreamSource:
        return cls._CYCLE[sequence_index % len(cls._CYCLE)]

    @classmethod
    def _source_counts_before(cls, sequence_index: int) -> tuple[int, int]:
        cycles, remainder = divmod(sequence_index, len(cls._CYCLE))
        prefix = cls._CYCLE[:remainder]
        general = cycles * 4 + sum(item is StreamSource.GENERAL for item in prefix)
        transaction = cycles + sum(item is StreamSource.TRANSACTION for item in prefix)
        return general, transaction

    def snapshot(self) -> dict[str, Any]:
        payload = {
            "schema_version": "catena-v8.1",
            "cursor_algorithm": self._ALGORITHM,
            "sequence_length": self.sequence_length,
            "sequence_index": self.sequence_index,
            "tokens_emitted": self.sequence_index * self.sequence_length,
            "general_cursor": self.general.snapshot(),
            "transaction_cursor": self.transaction.snapshot(),
        }
        payload["snapshot_sha256"] = sha256_canonical_json(payload)
        return payload

    @classmethod
    def from_snapshot(
        cls,
        corpus: TokenMemmap,
        tokenizer: Tokenizer,
        *,
        tokenizer_hash: str,
        snapshot: Mapping[str, Any],
    ) -> PairedTrainingCursor:
        payload = dict(snapshot)
        observed_hash = payload.pop("snapshot_sha256", None)
        if not isinstance(observed_hash, str) or observed_hash != sha256_canonical_json(payload):
            raise PairedStreamContractError("Mixed cursor snapshot SHA-256 mismatch")
        if payload.get("schema_version") != "catena-v8.1":
            raise PairedStreamContractError("Unsupported mixed cursor snapshot schema")
        if payload.get("cursor_algorithm") != cls._ALGORITHM:
            raise PairedStreamContractError("Unsupported mixed cursor schedule")
        sequence_index = payload.get("sequence_index")
        sequence_length = payload.get("sequence_length")
        tokens_emitted = payload.get("tokens_emitted")
        if (
            isinstance(sequence_index, bool)
            or not isinstance(sequence_index, int)
            or isinstance(sequence_length, bool)
            or not isinstance(sequence_length, int)
            or tokens_emitted != sequence_index * sequence_length
        ):
            raise PairedStreamContractError("Invalid mixed cursor exposure counters")
        general_snapshot = payload.get("general_cursor")
        transaction_snapshot = payload.get("transaction_cursor")
        if not isinstance(general_snapshot, Mapping) or not isinstance(
            transaction_snapshot, Mapping
        ):
            raise PairedStreamContractError("Mixed cursor child snapshots are missing")
        general = PairedTokenCursor.from_snapshot(corpus, general_snapshot)
        transaction = PairedTransactionCursor.from_snapshot(
            tokenizer,
            tokenizer_hash=tokenizer_hash,
            snapshot=transaction_snapshot,
        )
        expected_general, expected_transaction = cls._source_counts_before(sequence_index)
        if (
            general.sequence_index != expected_general
            or transaction.sequence_index != expected_transaction
        ):
            raise PairedStreamContractError(
                "Mixed cursor child indices disagree with the locked 4:1 source schedule"
            )
        cursor = cls(general, transaction, start_sequence_index=sequence_index)
        if cursor.sequence_length != sequence_length:
            raise PairedStreamContractError("Mixed cursor sequence length changed")
        return cursor

    def fork(
        self,
        corpus: TokenMemmap,
        tokenizer: Tokenizer,
        *,
        tokenizer_hash: str,
    ) -> PairedTrainingCursor:
        return self.from_snapshot(
            corpus,
            tokenizer,
            tokenizer_hash=tokenizer_hash,
            snapshot=self.snapshot(),
        )

    def take(self, count: int) -> tuple[list[TrainingSequence], TrainingCursorReceipt]:
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("count must be a non-negative integer")
        start = self.sequence_index
        sequences: list[TrainingSequence] = []
        for global_index in range(start, start + count):
            source = self._source_at(global_index)
            if source is StreamSource.GENERAL:
                general_rows, receipt = self.general.take(1)
                if len(general_rows) != 1 or len(receipt.starts) != 1:
                    raise ScientificCorpusContractError("General cursor did not return one row")
                general_token_ids = general_rows[0]
                sequences.append(
                    TrainingSequence(
                        token_ids=general_token_ids,
                        source_type=source.value,
                        source_id=(
                            f"{self.general.corpus.manifest.manifest_hash}:{receipt.starts[0]}"
                        ),
                        source_index=receipt.start_sequence_index,
                        token_offset=receipt.starts[0],
                        unpadded_tokens=self.sequence_length,
                        padding_tokens=0,
                        reset_state=True,
                        query_type=None,
                    )
                )
            else:
                transaction_rows = self.transaction.take(1)
                if len(transaction_rows) != 1:
                    raise PairedStreamContractError("Transaction cursor did not return one row")
                sequences.append(transaction_rows[0])
        self.sequence_index += count
        return sequences, _training_receipt(
            algorithm=self._ALGORITHM,
            start_sequence_index=start,
            end_sequence_index=self.sequence_index,
            sequence_length=self.sequence_length,
            sequences=sequences,
        )


class TokenBalancedPairedTrainingCursor:
    """Deterministic 80/20 scheduler over actual non-padding source tokens."""

    _ALGORITHM = "token_balanced_complete_example_80_20_v2"

    def __init__(
        self,
        general: PairedTokenCursor,
        transaction: PackedTransactionCursor,
        *,
        start_sequence_index: int = 0,
        start_general_unpadded_tokens: int = 0,
        start_transaction_unpadded_tokens: int = 0,
    ) -> None:
        if general.sequence_length != transaction.sequence_length:
            raise ValueError("General and transaction sequence lengths must match")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (
                start_sequence_index,
                start_general_unpadded_tokens,
                start_transaction_unpadded_tokens,
            )
        ):
            raise ValueError("token-balanced cursor counters must be non-negative")
        self.general = general
        self.transaction = transaction
        self.sequence_length = general.sequence_length
        self.sequence_index = int(start_sequence_index)
        self.general_unpadded_tokens = int(start_general_unpadded_tokens)
        self.transaction_unpadded_tokens = int(start_transaction_unpadded_tokens)
        self._transaction_preview: tuple[TrainingSequence, PackedTransactionCursor] | None = None

    def _peek_transaction(
        self,
    ) -> tuple[TrainingSequence, PackedTransactionCursor]:
        if self._transaction_preview is None:
            preview = self.transaction.fork()
            rows = preview.take(1)
            if len(rows) != 1:
                raise PairedStreamContractError("Packed transaction preview did not return one row")
            self._transaction_preview = (rows[0], preview)
        return self._transaction_preview

    def _next_source(self) -> StreamSource:
        if self.sequence_index == 0:
            return StreamSource.GENERAL
        transaction_row, _ = self._peek_transaction()
        general_deviation = abs(
            4 * self.transaction_unpadded_tokens
            - (self.general_unpadded_tokens + self.sequence_length)
        )
        transaction_deviation = abs(
            4 * (self.transaction_unpadded_tokens + transaction_row.unpadded_tokens)
            - self.general_unpadded_tokens
        )
        if transaction_deviation < general_deviation:
            return StreamSource.TRANSACTION
        return StreamSource.GENERAL

    def snapshot(self) -> dict[str, Any]:
        payload = {
            "schema_version": "catena-v8.1",
            "cursor_algorithm": self._ALGORITHM,
            "sequence_length": self.sequence_length,
            "sequence_index": self.sequence_index,
            "allocated_tokens_emitted": self.sequence_index * self.sequence_length,
            "general_unpadded_tokens": self.general_unpadded_tokens,
            "transaction_unpadded_tokens": self.transaction_unpadded_tokens,
            "loss_bearing_tokens_emitted": (
                self.general_unpadded_tokens + self.transaction_unpadded_tokens
            ),
            "target_general_fraction": 0.8,
            "target_transaction_fraction": 0.2,
            "general_cursor": self.general.snapshot(),
            "transaction_cursor": self.transaction.snapshot(),
        }
        payload["snapshot_sha256"] = sha256_canonical_json(payload)
        return payload

    @classmethod
    def from_snapshot(
        cls,
        corpus: TokenMemmap,
        tokenizer: Tokenizer,
        *,
        tokenizer_hash: str,
        snapshot: Mapping[str, Any],
    ) -> TokenBalancedPairedTrainingCursor:
        payload = dict(snapshot)
        observed_hash = payload.pop("snapshot_sha256", None)
        if not isinstance(observed_hash, str) or observed_hash != sha256_canonical_json(payload):
            raise PairedStreamContractError("Token-balanced cursor snapshot SHA-256 mismatch")
        if (
            payload.get("schema_version") != "catena-v8.1"
            or payload.get("cursor_algorithm") != cls._ALGORITHM
            or payload.get("target_general_fraction") != 0.8
            or payload.get("target_transaction_fraction") != 0.2
        ):
            raise PairedStreamContractError("Unsupported token-balanced cursor snapshot")
        general_snapshot = payload.get("general_cursor")
        transaction_snapshot = payload.get("transaction_cursor")
        if not isinstance(general_snapshot, Mapping) or not isinstance(
            transaction_snapshot, Mapping
        ):
            raise PairedStreamContractError("Token-balanced cursor child snapshots are missing")
        general = PairedTokenCursor.from_snapshot(corpus, general_snapshot)
        transaction = PackedTransactionCursor.from_snapshot(
            tokenizer,
            tokenizer_hash=tokenizer_hash,
            snapshot=transaction_snapshot,
        )
        sequence_index = payload.get("sequence_index")
        sequence_length = payload.get("sequence_length")
        general_tokens = payload.get("general_unpadded_tokens")
        transaction_tokens = payload.get("transaction_unpadded_tokens")
        loss_tokens = payload.get("loss_bearing_tokens_emitted")
        allocated_tokens = payload.get("allocated_tokens_emitted")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (
                sequence_index,
                sequence_length,
                general_tokens,
                transaction_tokens,
            )
        ):
            raise PairedStreamContractError("Invalid token-balanced counters")
        sequence_index_int = cast(int, sequence_index)
        sequence_length_int = cast(int, sequence_length)
        general_tokens_int = cast(int, general_tokens)
        transaction_tokens_int = cast(int, transaction_tokens)
        if (
            sequence_index_int != general.sequence_index + transaction.sequence_index
            or general_tokens_int != general.sequence_index * sequence_length_int
            or transaction_tokens_int != transaction.unpadded_tokens_emitted
            or loss_tokens != general_tokens_int + transaction_tokens_int
            or allocated_tokens != sequence_index_int * sequence_length_int
        ):
            raise PairedStreamContractError(
                "Token-balanced child progress disagrees with exposure counters"
            )
        cursor = cls(
            general,
            transaction,
            start_sequence_index=sequence_index_int,
            start_general_unpadded_tokens=general_tokens_int,
            start_transaction_unpadded_tokens=transaction_tokens_int,
        )
        if cursor.sequence_length != sequence_length_int:
            raise PairedStreamContractError("Token-balanced cursor sequence length changed")
        return cursor

    def fork(
        self,
        corpus: TokenMemmap,
        tokenizer: Tokenizer,
        *,
        tokenizer_hash: str,
    ) -> TokenBalancedPairedTrainingCursor:
        return self.from_snapshot(
            corpus,
            tokenizer,
            tokenizer_hash=tokenizer_hash,
            snapshot=self.snapshot(),
        )

    def _take_one(self) -> TrainingSequence:
        source = self._next_source()
        if source is StreamSource.GENERAL:
            general_rows, receipt = self.general.take(1)
            if len(general_rows) != 1 or len(receipt.starts) != 1:
                raise ScientificCorpusContractError("General cursor did not return one row")
            row = TrainingSequence(
                token_ids=general_rows[0],
                source_type=source.value,
                source_id=(f"{self.general.corpus.manifest.manifest_hash}:{receipt.starts[0]}"),
                source_index=receipt.start_sequence_index,
                token_offset=receipt.starts[0],
                unpadded_tokens=self.sequence_length,
                padding_tokens=0,
                reset_state=True,
                query_type=None,
            )
            self.general_unpadded_tokens += row.unpadded_tokens
        else:
            row, preview = self._peek_transaction()
            self.transaction = preview
            self._transaction_preview = None
            self.transaction_unpadded_tokens += row.unpadded_tokens
        self.sequence_index += 1
        return row

    def take(self, count: int) -> tuple[list[TrainingSequence], TrainingCursorReceipt]:
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("count must be a non-negative integer")
        start = self.sequence_index
        rows = [self._take_one() for _ in range(count)]
        return rows, _training_receipt(
            algorithm=self._ALGORITHM,
            start_sequence_index=start,
            end_sequence_index=self.sequence_index,
            sequence_length=self.sequence_length,
            sequences=rows,
        )

    def take_minimum_loss_tokens(
        self,
        minimum_tokens: int,
    ) -> tuple[list[TrainingSequence], TrainingCursorReceipt]:
        if minimum_tokens <= 0:
            raise ValueError("minimum_tokens must be positive")
        start = self.sequence_index
        rows: list[TrainingSequence] = []
        emitted = 0
        while emitted < minimum_tokens:
            row = self._take_one()
            rows.append(row)
            emitted += row.unpadded_tokens
        return rows, _training_receipt(
            algorithm=self._ALGORITHM,
            start_sequence_index=start,
            end_sequence_index=self.sequence_index,
            sequence_length=self.sequence_length,
            sequences=rows,
        )


def replay_digest(
    cursor: PairedTrainingCursor | TokenBalancedPairedTrainingCursor,
    *,
    minimum_tokens: int,
) -> dict[str, Any]:
    """Consume at least ``minimum_tokens`` and return exact metadata/byte digests."""

    if minimum_tokens <= 0:
        raise ValueError("minimum_tokens must be positive")
    if isinstance(cursor, TokenBalancedPairedTrainingCursor):
        _, receipt = cursor.take_minimum_loss_tokens(minimum_tokens)
    else:
        sequence_count = (minimum_tokens + cursor.sequence_length - 1) // cursor.sequence_length
        _, receipt = cursor.take(sequence_count)
    return {
        **receipt.as_dict(),
        "requested_minimum_tokens": minimum_tokens,
        "overrun_tokens": receipt.loss_bearing_tokens - minimum_tokens,
        "cursor_snapshot": cursor.snapshot(),
    }
