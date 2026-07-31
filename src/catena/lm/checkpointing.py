from __future__ import annotations

import os
import random
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from catena.core.provenance_v61 import sha256_canonical_json, sha256_file

from .audit_contract import validate_e26_audit_locked_hashes
from .hashing import tensor_tree_digest
from .model import LocalAttentionState, RuntimeState
from .recurrent_mixer import MixerState


class TrainingCheckpointError(RuntimeError):
    """Raised when a checkpoint cannot prove an exact training continuation."""


@dataclass(frozen=True, slots=True)
class RNGSnapshot:
    python_state: object
    numpy_state: tuple[Any, ...]
    torch_cpu_state: torch.Tensor
    torch_cuda_states: tuple[torch.Tensor, ...]
    visible_cuda_devices: int

    @classmethod
    def capture(cls) -> RNGSnapshot:
        cuda_states: tuple[torch.Tensor, ...] = ()
        visible_devices = 0
        if torch.cuda.is_available():
            visible_devices = torch.cuda.device_count()
            cuda_states = tuple(state.clone() for state in torch.cuda.get_rng_state_all())
        return cls(
            python_state=random.getstate(),
            numpy_state=cast(tuple[Any, ...], np.random.get_state()),
            torch_cpu_state=torch.random.get_rng_state().clone(),
            torch_cuda_states=cuda_states,
            visible_cuda_devices=visible_devices,
        )

    def restore(self) -> None:
        observed_devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if observed_devices != self.visible_cuda_devices:
            raise TrainingCheckpointError(
                "Visible CUDA device count changed across checkpoint resume: "
                f"{self.visible_cuda_devices} -> {observed_devices}"
            )
        random.setstate(self.python_state)  # type: ignore[arg-type]
        np.random.set_state(self.numpy_state)
        torch.random.set_rng_state(self.torch_cpu_state)
        if self.torch_cuda_states:
            torch.cuda.set_rng_state_all(list(self.torch_cuda_states))

    def as_payload(self) -> dict[str, Any]:
        return {
            "python_state": self.python_state,
            "numpy_state": self.numpy_state,
            "torch_cpu_state": self.torch_cpu_state,
            "torch_cuda_states": self.torch_cuda_states,
            "visible_cuda_devices": self.visible_cuda_devices,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> RNGSnapshot:
        required = {
            "python_state",
            "numpy_state",
            "torch_cpu_state",
            "torch_cuda_states",
            "visible_cuda_devices",
        }
        if set(payload) != required:
            raise TrainingCheckpointError("RNG snapshot fields do not match the locked schema")
        cpu_state = payload["torch_cpu_state"]
        cuda_states = payload["torch_cuda_states"]
        visible_devices = payload["visible_cuda_devices"]
        numpy_state = payload["numpy_state"]
        if not torch.is_tensor(cpu_state):
            raise TrainingCheckpointError("Torch CPU RNG state is not a tensor")
        if not isinstance(cuda_states, (list, tuple)) or not all(
            torch.is_tensor(state) for state in cuda_states
        ):
            raise TrainingCheckpointError("Torch CUDA RNG states are invalid")
        if isinstance(visible_devices, bool) or not isinstance(visible_devices, int):
            raise TrainingCheckpointError("Visible CUDA device count is invalid")
        if not isinstance(numpy_state, tuple):
            raise TrainingCheckpointError("NumPy RNG state is invalid")
        return cls(
            python_state=payload["python_state"],
            numpy_state=numpy_state,
            torch_cpu_state=cpu_state,
            torch_cuda_states=tuple(cuda_states),
            visible_cuda_devices=visible_devices,
        )


@dataclass(frozen=True, slots=True)
class TrainingProgress:
    optimizer_step: int
    tokens_seen: int
    general_sequences_seen: int
    transaction_sequences_seen: int
    document_index: int
    episode_index: int
    cursor_snapshot: dict[str, Any]
    last_source_type: str | None = None

    def __post_init__(self) -> None:
        counters = (
            self.optimizer_step,
            self.tokens_seen,
            self.general_sequences_seen,
            self.transaction_sequences_seen,
            self.document_index,
            self.episode_index,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counters
        ):
            raise ValueError("Training progress counters must be non-negative integers")

    def as_payload(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> TrainingProgress:
        return cls(
            optimizer_step=int(payload["optimizer_step"]),
            tokens_seen=int(payload["tokens_seen"]),
            general_sequences_seen=int(payload["general_sequences_seen"]),
            transaction_sequences_seen=int(payload["transaction_sequences_seen"]),
            document_index=int(payload["document_index"]),
            episode_index=int(payload["episode_index"]),
            cursor_snapshot=dict(payload["cursor_snapshot"]),
            last_source_type=(
                None
                if payload.get("last_source_type") is None
                else str(payload["last_source_type"])
            ),
        )


@dataclass(frozen=True, slots=True)
class CheckpointReceipt:
    path: str
    bytes: int
    sha256: str
    semantic_payload_sha256: str
    optimizer_step: int
    tokens_seen: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LoadedTrainingCheckpoint:
    progress: TrainingProgress
    runtime_state: RuntimeState | None
    rng_snapshot: RNGSnapshot
    locked_hashes: dict[str, str]
    amp_policy: dict[str, Any]
    backend_manifest: dict[str, Any] | None
    semantic_payload_sha256: str


def runtime_state_to_payload(state: RuntimeState | None) -> dict[str, Any] | None:
    if state is None:
        return None
    return {
        "position": state.position,
        "recurrent": [item.matrix for item in state.recurrent],
        "attention": [
            {
                "key": item.key,
                "value": item.value,
                "positions": item.positions,
                "length": item.length,
                "write_index": item.write_index,
            }
            for item in state.attention
        ],
    }


def runtime_state_from_payload(payload: Mapping[str, Any] | None) -> RuntimeState | None:
    if payload is None:
        return None
    recurrent = payload.get("recurrent")
    attention = payload.get("attention")
    position = payload.get("position")
    if not isinstance(recurrent, list) or not all(torch.is_tensor(item) for item in recurrent):
        raise TrainingCheckpointError("Checkpoint recurrent state is invalid")
    if not isinstance(attention, list) or not isinstance(position, int):
        raise TrainingCheckpointError("Checkpoint hybrid-state metadata is invalid")
    attention_states: list[LocalAttentionState] = []
    for item in attention:
        if not isinstance(item, Mapping):
            raise TrainingCheckpointError("Checkpoint attention state is invalid")
        key = item.get("key")
        value = item.get("value")
        positions = item.get("positions")
        if not all(torch.is_tensor(tensor) for tensor in (key, value, positions)):
            raise TrainingCheckpointError("Checkpoint attention tensors are invalid")
        attention_states.append(
            LocalAttentionState(
                key=cast(torch.Tensor, key),
                value=cast(torch.Tensor, value),
                positions=cast(torch.Tensor, positions),
                length=int(item["length"]),
                write_index=int(item["write_index"]),
            )
        )
    return RuntimeState(
        recurrent=[MixerState(matrix=item) for item in recurrent],
        attention=attention_states,
        position=position,
    )


def _atomic_torch_save(payload: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {destination}")
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        directory_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_training_checkpoint(
    destination: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    progress: TrainingProgress,
    locked_hashes: Mapping[str, str],
    runtime_state: RuntimeState | None = None,
    amp_policy: Mapping[str, Any] | None = None,
    backend_manifest: Mapping[str, Any] | None = None,
    rng_snapshot: RNGSnapshot | None = None,
) -> CheckpointReceipt:
    """Atomically save every state needed for a deterministic new-process resume."""

    normalized_hashes = validate_e26_audit_locked_hashes(locked_hashes)
    rng = rng_snapshot or RNGSnapshot.capture()
    payload: dict[str, Any] = {
        "format_version": "catena-e26-training-checkpoint-v1",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "progress": progress.as_payload(),
        "runtime_state": runtime_state_to_payload(runtime_state),
        "rng": rng.as_payload(),
        "locked_hashes": normalized_hashes,
        "amp_policy": dict(amp_policy or {"dtype": "float32", "grad_scaler": None}),
        "backend_manifest": (None if backend_manifest is None else dict(backend_manifest)),
    }
    semantic_digest = tensor_tree_digest(payload)
    payload["semantic_payload_sha256"] = semantic_digest
    unresolved = Path(destination).expanduser()
    if unresolved.is_symlink():
        raise TrainingCheckpointError("Checkpoint destination cannot be a symlink")
    path = unresolved.resolve()
    _atomic_torch_save(payload, path)
    return CheckpointReceipt(
        path=str(path),
        bytes=path.stat().st_size,
        sha256=sha256_file(path),
        semantic_payload_sha256=semantic_digest,
        optimizer_step=progress.optimizer_step,
        tokens_seen=progress.tokens_seen,
    )


def load_training_checkpoint(
    source: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    expected_locked_hashes: Mapping[str, str],
    expected_file_sha256: str,
    map_location: torch.device | str = "cpu",
    restore_rng: bool = True,
    expected_amp_policy: Mapping[str, Any] | None = None,
    expected_backend_manifest: Mapping[str, Any] | None = None,
) -> LoadedTrainingCheckpoint:
    """Validate and load a checkpoint, restoring RNG only after all state loads."""

    unresolved = Path(source).expanduser()
    if unresolved.is_symlink():
        raise TrainingCheckpointError("Training checkpoint cannot be a symlink")
    path = unresolved.resolve(strict=True)
    if not path.is_file():
        raise TrainingCheckpointError("Training checkpoint must be a regular non-symlink file")
    observed_file_hash = sha256_file(path)
    if observed_file_hash != expected_file_sha256:
        raise TrainingCheckpointError("Training checkpoint file SHA-256 mismatch")
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise TrainingCheckpointError("Training checkpoint payload is not a mapping")
    if payload.get("format_version") != "catena-e26-training-checkpoint-v1":
        raise TrainingCheckpointError("Unsupported training checkpoint format")
    observed_semantic_digest = payload.get("semantic_payload_sha256")
    semantic_payload = dict(payload)
    semantic_payload.pop("semantic_payload_sha256", None)
    if (
        not isinstance(observed_semantic_digest, str)
        or tensor_tree_digest(semantic_payload) != observed_semantic_digest
    ):
        raise TrainingCheckpointError("Training checkpoint semantic digest mismatch")
    observed_hashes = payload.get("locked_hashes")
    expected_hashes = validate_e26_audit_locked_hashes(expected_locked_hashes)
    if observed_hashes != expected_hashes:
        raise TrainingCheckpointError("Checkpoint config/data/backend hashes changed")
    amp_policy = payload.get("amp_policy")
    if not isinstance(amp_policy, Mapping):
        raise TrainingCheckpointError("Checkpoint AMP policy is missing or invalid")
    normalized_amp_policy = dict(amp_policy)
    if expected_amp_policy is not None and normalized_amp_policy != dict(expected_amp_policy):
        raise TrainingCheckpointError("Checkpoint AMP/BF16 policy changed")
    backend_manifest = payload.get("backend_manifest")
    if backend_manifest is not None and not isinstance(backend_manifest, Mapping):
        raise TrainingCheckpointError("Checkpoint compile/backend manifest is invalid")
    normalized_backend_manifest = None if backend_manifest is None else dict(backend_manifest)
    if expected_backend_manifest is not None and normalized_backend_manifest != dict(
        expected_backend_manifest
    ):
        raise TrainingCheckpointError("Checkpoint compile/backend manifest changed")
    scheduler_state = payload.get("scheduler")
    if (scheduler is None) != (scheduler_state is None):
        raise TrainingCheckpointError("Checkpoint scheduler presence does not match the runner")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(scheduler_state)
    progress_payload = payload.get("progress")
    rng_payload = payload.get("rng")
    if not isinstance(progress_payload, Mapping) or not isinstance(rng_payload, Mapping):
        raise TrainingCheckpointError("Checkpoint progress or RNG state is missing")
    progress = TrainingProgress.from_payload(progress_payload)
    rng_snapshot = RNGSnapshot.from_payload(rng_payload)
    runtime_payload = payload.get("runtime_state")
    if runtime_payload is not None and not isinstance(runtime_payload, Mapping):
        raise TrainingCheckpointError("Checkpoint runtime state payload is invalid")
    runtime_state = runtime_state_from_payload(runtime_payload)
    if restore_rng:
        rng_snapshot.restore()
    return LoadedTrainingCheckpoint(
        progress=progress,
        runtime_state=runtime_state,
        rng_snapshot=rng_snapshot,
        locked_hashes=expected_hashes,
        amp_policy=normalized_amp_policy,
        backend_manifest=normalized_backend_manifest,
        semantic_payload_sha256=observed_semantic_digest,
    )


def restart_audit_receipt(
    *,
    resume_cases: Mapping[str, Mapping[str, Any]],
    cursor_replays: Mapping[str, Mapping[str, Any]],
    expected_candidate_ids: Sequence[str],
    locked_hashes: Mapping[str, str],
    source_inventory: Mapping[str, Any],
) -> dict[str, Any]:
    """Aggregate new-process resume and paired-cursor checks without opening evidence."""

    candidate_ids = tuple(str(value) for value in expected_candidate_ids)
    if not candidate_ids or len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("expected_candidate_ids must be non-empty and unique")
    variants = ("dual_delta_lm", "projected_tied_delta_lm")
    transitions = ("general_to_transaction", "transaction_to_general")
    expected_case_ids = {
        f"{candidate_id}__{variant}__{transition}"
        for candidate_id in candidate_ids
        for variant in variants
        for transition in transitions
    }
    if set(resume_cases) != expected_case_ids:
        raise ValueError("Restart cases do not cover the exact candidate/variant/transition grid")
    if set(cursor_replays) != set(candidate_ids):
        raise ValueError("Cursor replays do not cover the exact locked candidate set")
    normalized_hashes = validate_e26_audit_locked_hashes(locked_hashes)
    if source_inventory.get("source_tree_sha256") != normalized_hashes["source_tree_sha256"]:
        raise ValueError("Restart receipt source inventory differs from locked source hash")
    case_passes: list[bool] = []
    for case_id, value in resume_cases.items():
        if not isinstance(value, Mapping):
            raise ValueError(f"Malformed restart case: {case_id}")
        candidate_id, variant, transition = case_id.split("__", maxsplit=2)
        identity_matches = (
            value.get("candidate_id") == candidate_id
            and value.get("variant") == variant
            and value.get("transition") == transition
        )
        case_passes.append(identity_matches and value.get("passed") is True)
    cursor_passes = [
        isinstance(value, Mapping) and value.get("passed") is True
        for value in cursor_replays.values()
    ]
    all_passed = (
        len(case_passes) == len(resume_cases)
        and all(case_passes)
        and len(cursor_passes) == len(candidate_ids)
        and all(cursor_passes)
    )
    payload: dict[str, Any] = {
        "schema_version": "catena-v8.1",
        "manifest_type": "E26_RESTART_AUDIT",
        "scientific_evidence": False,
        "evidence_tier": "NON_EVIDENCE_VALIDATION",
        "all_passed": all_passed,
        "passed": all_passed,
        "main_test_opened": False,
        "locked_hashes": normalized_hashes,
        "source_inventory": dict(source_inventory),
        "expected_candidate_ids": list(candidate_ids),
        "expected_variants": list(variants),
        "expected_transitions": list(transitions),
        "resume_cases": {key: dict(value) for key, value in sorted(resume_cases.items())},
        "cursor_replays": {key: dict(value) for key, value in sorted(cursor_replays.items())},
    }
    payload["receipt_sha256"] = sha256_canonical_json(payload)
    return payload


def validate_restart_audit_coverage(
    receipt: Mapping[str, Any],
    *,
    expected_candidate_ids: Sequence[str],
) -> dict[str, bool]:
    """Fail closed unless every selectable candidate has an exact restart grid."""

    candidate_ids = tuple(str(value) for value in expected_candidate_ids)
    variants = ("dual_delta_lm", "projected_tied_delta_lm")
    transitions = ("general_to_transaction", "transaction_to_general")
    expected_case_ids = {
        f"{candidate_id}__{variant}__{transition}"
        for candidate_id in candidate_ids
        for variant in variants
        for transition in transitions
    }
    if receipt.get("expected_candidate_ids") != list(candidate_ids):
        raise ValueError("Restart receipt candidate order differs from the locked table")
    if receipt.get("expected_variants") != list(variants):
        raise ValueError("Restart receipt variant grid differs from the locked table")
    if receipt.get("expected_transitions") != list(transitions):
        raise ValueError("Restart receipt transition grid differs from the locked table")
    cases = receipt.get("resume_cases")
    replays = receipt.get("cursor_replays")
    if not isinstance(cases, Mapping) or set(cases) != expected_case_ids:
        raise ValueError("Restart receipt lacks exact candidate restart coverage")
    if not isinstance(replays, Mapping) or set(replays) != set(candidate_ids):
        raise ValueError("Restart receipt lacks exact candidate cursor coverage")
    coverage: dict[str, bool] = {}
    for candidate_id in candidate_ids:
        candidate_case_ids = {
            f"{candidate_id}__{variant}__{transition}"
            for variant in variants
            for transition in transitions
        }
        coverage[candidate_id] = bool(
            isinstance(replays[candidate_id], Mapping)
            and replays[candidate_id].get("passed") is True
            and all(
                isinstance(cases[case_id], Mapping)
                and cases[case_id].get("candidate_id") == candidate_id
                and cases[case_id].get("passed") is True
                for case_id in candidate_case_ids
            )
        )
    if receipt.get("passed") is not all(coverage.values()):
        raise ValueError("Restart receipt top-level PASS contradicts candidate coverage")
    return coverage
