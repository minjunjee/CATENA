"""Fail-closed external-source provenance for the E26 Final admission gate.

This module audits only repository and Hugging Face metadata plus small
tokenizer files.  It never downloads the 17.4 GB checkpoint.  When a local
checkpoint path is explicitly supplied, its existing bytes are streamed
through SHA-256 without loading the tensor object into memory.
"""

from __future__ import annotations

import hashlib
import subprocess
import urllib.request
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from catena.core.provenance_v61 import (
    loads_json_strict,
    sha256_canonical_json,
    sha256_file,
    write_json_strict,
)


class E26FinalProvenanceError(RuntimeError):
    """Raised when an external provenance receipt is malformed or mutable."""


@dataclass(frozen=True)
class OfficialSourceExpectation:
    remote_url: str
    commit: str
    tree: str
    license_blob: str
    license_bytes: int
    license_sha256: str
    key_blobs: Mapping[str, str]


@dataclass(frozen=True)
class HfFileExpectation:
    filename: str
    size: int
    blob_id: str
    sha256: str
    lfs: bool


OFFICIAL_SOURCE: Final = OfficialSourceExpectation(
    remote_url="https://github.com/NVlabs/GatedDeltaNet-2.git",
    commit="95709fc250357c2dd109361c353192f2aa5913f9",
    tree="bec1976e3b1ab0fab519f60c73e36a3c0092da47",
    license_blob="d50677fec39f78e04515c615d30ad16741f8a29c",
    license_bytes=4_126,
    license_sha256="eaff393a7abc4ea7cb05795423b531a212b6d2189bcbe30410587d52d70988bb",
    key_blobs={
        "lit_gpt/config.py": "150497a501b0c70a173d9a5601545e2f326a5bb3",
        "lit_gpt/gdn2.py": "4a0199c69b2097d2637bb8714d707bd53e67e91d",
        "lit_gpt/gdn2_ops/chunk_gdn2.py": "6eb08c941475b5c1ec581d17f0e20136756d57bc",
        "lit_gpt/gdn2_ops/chunk_kda.py": "d3deb1b78d8052c2f4fa24f65c1ef6ce5b7d00a7",
        "lit_gpt/gdn2_ops/fused_recurrent_gdn2.py": (
            "b994d407c6f0bae69c939426cd72dd7196eb0d28"
        ),
        "lit_gpt/tokenizer.py": "d7c08a57cc866859ba6e69a4f3bc1528f092bba1",
        "pretrain.py": "a2ad4fe9079a4127b4e95be90d137031c4a50bfe",
    },
)

CHECKPOINT_REPO_ID: Final = "LLM-OS-Models2/gdn2-1.3b-paper-matched"
CHECKPOINT_REVISION: Final = "8b1f11f6ac0322825120580bbf7ac7133e72a167"
CHECKPOINT_FILE: Final = HfFileExpectation(
    filename="model-100b.pth",
    size=17_401_727_659,
    blob_id="42194d557d389ae7c251dd50381c99ff64172f36",
    sha256="0322ebeefa96badb24d6b4b511c36b02374b704dc1a65b90eab2ee1383a9ce23",
    lfs=True,
)
CHECKPOINT_ALIAS_FILE: Final = "model-95b.pth"

TOKENIZER_REPO_ID: Final = "TinyLlama/TinyLlama_v1.1"
TOKENIZER_REVISION: Final = "ff3c701f2424c7625fdefb9dd470f45ef18b02d6"
TOKENIZER_FILES: Final[tuple[HfFileExpectation, ...]] = (
    HfFileExpectation(
        "tokenizer.json",
        1_842_767,
        "a6e931b92caff4c79c5c56282f1e89569a0ae558",
        "bcd04f0eadf90287bd26e1a183ac487d8a141b09b06aecb7725bbdd343640f2e",
        False,
    ),
    HfFileExpectation(
        "tokenizer.model",
        499_723,
        "6c00c742ce03c627d6cd5b795984876fa49fa899",
        "9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347",
        True,
    ),
    HfFileExpectation(
        "tokenizer_config.json",
        776,
        "2ef41cbc275000b29afe157ba487f0530b8c26dc",
        "f514e7c3008881b6ba7e6a0cdb44c71ce47dc335920dac143ae7bc788197e53a",
        False,
    ),
    HfFileExpectation(
        "special_tokens_map.json",
        414,
        "451134b2ddc2e78555d1e857518c54b4bdc2e87d",
        "6fa06efa2785e450051989a6f8fb4416b10149ded485ddd3f127a40734f5cfd0",
        False,
    ),
    HfFileExpectation(
        "config.json",
        560,
        "6a20305540fec9201e5c28b99dcd32c1000201fd",
        "7ef24df4204405995b8a3936355171b0d80f64fc3bc0ca5991158eaa5c75de50",
        False,
    ),
    HfFileExpectation(
        "generation_config.json",
        129,
        "89c1930ccf07b1ba0c1bf146b2ad2d2666761dfb",
        "eb2d264819d797a5ccbd6bb45430d9017d38a4e21c54ca8400851aa0a6f1e9c4",
        False,
    ),
)
TOKENIZER_VOCAB_SIZE: Final = 32_000
TOKENIZER_SPECIAL_IDS: Final = {"unk": 0, "bos": 1, "eos": 2}
TOKENIZER_PAD_POLICY: Final = "SET_PAD_TO_EOS_ID_2_AT_RUNTIME"

_RECEIPT_SCHEMA = "catena-e26-final-external-provenance-v1"
_RECEIPT_TYPE = "E26_FINAL_EXTERNAL_PROVENANCE_RECEIPT"
_MAX_METADATA_BYTES = 4_000_000


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(root), *arguments),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise E26FinalProvenanceError(
            f"Git command failed ({' '.join(arguments)}): {result.stderr.strip()}"
        )
    return result.stdout.strip()


def _blob_oid(root: Path, commit: str, relative: str) -> str:
    line = _git(root, "ls-tree", commit, "--", relative)
    fields = line.split()
    if len(fields) < 4 or fields[1] != "blob":
        raise E26FinalProvenanceError(f"Missing Git blob at {commit}:{relative}")
    return fields[2]


def _normalize_remote(url: str) -> str:
    return url.rstrip("/")


def parse_ls_remote(output: str) -> dict[str, str]:
    """Parse exact Git refs without accepting abbreviated object IDs."""

    refs: dict[str, str] = {}
    for line in output.splitlines():
        fields = line.split()
        if len(fields) != 2 or len(fields[0]) != 40:
            raise E26FinalProvenanceError("Malformed git ls-remote output")
        refs[fields[1]] = fields[0]
    return refs


def audit_official_source(
    repository: str | Path,
    *,
    remote_refs: Mapping[str, str],
    expectation: OfficialSourceExpectation = OFFICIAL_SOURCE,
) -> dict[str, Any]:
    """Audit one existing official checkout without fetching or modifying it."""

    root = Path(repository).expanduser()
    if root.is_symlink():
        raise E26FinalProvenanceError("Official source checkout must not be a symlink")
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise E26FinalProvenanceError("Official source checkout is not a directory")
    head = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    remote_url = _git(root, "remote", "get-url", "origin")
    license_path = root / "LICENSE"
    if license_path.is_symlink() or not license_path.is_file():
        raise E26FinalProvenanceError("Official source LICENSE is missing or a symlink")
    license_bytes = license_path.read_bytes()
    license_text = license_bytes.decode("utf-8")
    key_blobs = {
        relative: _blob_oid(root, expectation.commit, relative)
        for relative in expectation.key_blobs
    }
    checks = {
        "remote_head_exact": remote_refs.get("HEAD") == expectation.commit,
        "remote_main_exact": remote_refs.get("refs/heads/main") == expectation.commit,
        "origin_url_exact": _normalize_remote(remote_url)
        == _normalize_remote(expectation.remote_url),
        "local_head_exact": head == expectation.commit,
        "local_tree_exact": tree == expectation.tree,
        "local_checkout_clean": status == "",
        "license_git_blob_exact": _blob_oid(root, expectation.commit, "LICENSE")
        == expectation.license_blob,
        "license_size_exact": len(license_bytes) == expectation.license_bytes,
        "license_sha256_exact": hashlib.sha256(license_bytes).hexdigest()
        == expectation.license_sha256,
        "license_noncommercial_terms_present": (
            "Nvidia Source Code License-NC" in license_text
            and "only may be used or intended for use non-commercially" in license_text
        ),
        "key_source_blobs_exact": key_blobs == dict(expectation.key_blobs),
    }
    return {
        "repository": str(root),
        "remote_url": remote_url,
        "expected_commit": expectation.commit,
        "observed_head": head,
        "expected_tree": expectation.tree,
        "observed_tree": tree,
        "git_status_porcelain": status.splitlines(),
        "license": {
            "name": "NVIDIA Source Code License-NC",
            "commercial_use_allowed": False,
            "bytes": len(license_bytes),
            "sha256": hashlib.sha256(license_bytes).hexdigest(),
            "git_blob": _blob_oid(root, expectation.commit, "LICENSE"),
        },
        "key_source_blobs": key_blobs,
        "hard_checks": checks,
        "passed": all(checks.values()),
    }


def _metadata_file(metadata: Mapping[str, Any], filename: str) -> Mapping[str, Any] | None:
    siblings = metadata.get("siblings")
    if not isinstance(siblings, list):
        return None
    matches = [
        row
        for row in siblings
        if isinstance(row, Mapping) and row.get("rfilename") == filename
    ]
    return matches[0] if len(matches) == 1 else None


def _file_metadata_matches(row: Mapping[str, Any] | None, expected: HfFileExpectation) -> bool:
    if row is None or row.get("size") != expected.size or row.get("blobId") != expected.blob_id:
        return False
    lfs = row.get("lfs")
    if expected.lfs:
        return bool(
            isinstance(lfs, Mapping)
            and lfs.get("sha256") == expected.sha256
            and lfs.get("size") == expected.size
        )
    return lfs is None


def _local_checkpoint_audit(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "status": "NOT_PROVIDED",
            "path": None,
            "bytes": None,
            "sha256": None,
            "verified": False,
        }
    candidate = Path(path).expanduser()
    if candidate.is_symlink() or not candidate.is_file():
        return {
            "status": "INVALID_PATH",
            "path": str(candidate),
            "bytes": None,
            "sha256": None,
            "verified": False,
        }
    candidate = candidate.resolve(strict=True)
    size = candidate.stat().st_size
    digest = sha256_file(candidate) if size == CHECKPOINT_FILE.size else None
    verified = size == CHECKPOINT_FILE.size and digest == CHECKPOINT_FILE.sha256
    return {
        "status": "VERIFIED" if verified else "MISMATCH",
        "path": str(candidate),
        "bytes": size,
        "sha256": digest,
        "verified": verified,
    }


def audit_checkpoint_metadata(
    metadata: Mapping[str, Any],
    *,
    local_checkpoint: str | Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate checkpoint metadata and separately report semantic warnings."""

    target = _metadata_file(metadata, CHECKPOINT_FILE.filename)
    alias = _metadata_file(metadata, CHECKPOINT_ALIAS_FILE)
    alias_same = bool(
        isinstance(target, Mapping)
        and isinstance(alias, Mapping)
        and target.get("blobId") == alias.get("blobId")
        and isinstance(target.get("lfs"), Mapping)
        and isinstance(alias.get("lfs"), Mapping)
        and target["lfs"].get("sha256") == alias["lfs"].get("sha256")
        and target.get("size") == alias.get("size")
    )
    checks = {
        "repo_id_exact": metadata.get("id") == CHECKPOINT_REPO_ID,
        "revision_exact": metadata.get("sha") == CHECKPOINT_REVISION,
        "public_ungated_enabled": (
            metadata.get("private") is False
            and metadata.get("gated") is False
            and metadata.get("disabled") is False
        ),
        "model_100b_metadata_exact": _file_metadata_matches(target, CHECKPOINT_FILE),
    }
    local = _local_checkpoint_audit(local_checkpoint)
    if local_checkpoint is not None:
        checks["local_checkpoint_bytes_exact"] = local["verified"] is True
    warnings: list[dict[str, Any]] = []
    if alias_same:
        warnings.append(
            {
                "code": "CHECKPOINT_95B_AND_100B_BYTE_IDENTICAL",
                "severity": "HIGH",
                "protocol_hard_gate": False,
                "detail": (
                    "model-95b.pth and model-100b.pth share the same Git blob, "
                    "LFS SHA-256, and byte count; training-token identity is not established."
                ),
            }
        )
    warnings.extend(
        [
            {
                "code": "COMMUNITY_CHECKPOINT_NOT_NVIDIA_OFFICIAL_RELEASE",
                "severity": "HIGH",
                "protocol_hard_gate": False,
                "detail": "The checkpoint repository is separate from NVlabs/GatedDeltaNet-2.",
            },
            {
                "code": "CHECKPOINT_LICENSE_REQUIRES_MANUAL_REVIEW",
                "severity": "HIGH",
                "protocol_hard_gate": False,
                "detail": (
                    "The model card declares Apache-2.0 while official source uses the "
                    "NVIDIA Source Code License-NC."
                ),
            },
        ]
    )
    return (
        {
            "repo_id": CHECKPOINT_REPO_ID,
            "revision": metadata.get("sha"),
            "file": {
                "filename": CHECKPOINT_FILE.filename,
                "bytes": target.get("size") if isinstance(target, Mapping) else None,
                "blob_id": target.get("blobId") if isinstance(target, Mapping) else None,
                "lfs_sha256": (
                    target["lfs"].get("sha256")
                    if isinstance(target, Mapping) and isinstance(target.get("lfs"), Mapping)
                    else None
                ),
            },
            "local_checkpoint": local,
            "checkpoint_bytes_ready": local["verified"] is True,
            "training_token_label_claim_eligible": not alias_same,
            "hard_checks": checks,
            "passed": all(checks.values()),
        },
        warnings,
    )


def _strict_json_bytes(payload: bytes, label: str) -> Mapping[str, Any]:
    try:
        decoded = loads_json_strict(payload)
    except (UnicodeDecodeError, ValueError) as error:
        raise E26FinalProvenanceError(f"Invalid strict JSON in {label}") from error
    if not isinstance(decoded, Mapping):
        raise E26FinalProvenanceError(f"{label} must contain a JSON object")
    return decoded


def _token_content(value: Any) -> Any:
    return value.get("content") if isinstance(value, Mapping) else None


def audit_tokenizer_metadata(
    metadata: Mapping[str, Any],
    *,
    files: Mapping[str, bytes],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Bind the exact TinyLlama revision, file bytes, IDs, and pad policy."""

    expected_names = {row.filename for row in TOKENIZER_FILES}
    file_metadata_checks = {
        row.filename: _file_metadata_matches(_metadata_file(metadata, row.filename), row)
        for row in TOKENIZER_FILES
    }
    bytes_checks = {
        row.filename: (
            row.filename in files
            and len(files[row.filename]) == row.size
            and hashlib.sha256(files[row.filename]).hexdigest() == row.sha256
        )
        for row in TOKENIZER_FILES
    }
    if set(files) != expected_names:
        missing = sorted(expected_names - set(files))
        extra = sorted(set(files) - expected_names)
        raise E26FinalProvenanceError(
            f"Tokenizer file population differs; missing={missing}, extra={extra}"
        )
    tokenizer = _strict_json_bytes(files["tokenizer.json"], "tokenizer.json")
    tokenizer_config = _strict_json_bytes(
        files["tokenizer_config.json"], "tokenizer_config.json"
    )
    special_map = _strict_json_bytes(
        files["special_tokens_map.json"], "special_tokens_map.json"
    )
    model_config = _strict_json_bytes(files["config.json"], "config.json")
    generation = _strict_json_bytes(
        files["generation_config.json"], "generation_config.json"
    )
    model = tokenizer.get("model")
    vocab = model.get("vocab") if isinstance(model, Mapping) else None
    ids = {
        "unk": vocab.get("<unk>") if isinstance(vocab, Mapping) else None,
        "bos": vocab.get("<s>") if isinstance(vocab, Mapping) else None,
        "eos": vocab.get("</s>") if isinstance(vocab, Mapping) else None,
    }
    checks = {
        "repo_id_exact": metadata.get("id") == TOKENIZER_REPO_ID,
        "revision_exact": metadata.get("sha") == TOKENIZER_REVISION,
        "public_ungated_enabled": (
            metadata.get("private") is False
            and metadata.get("gated") is False
            and metadata.get("disabled") is False
        ),
        "file_metadata_exact": all(file_metadata_checks.values()),
        "small_file_bytes_exact": all(bytes_checks.values()),
        "tokenizer_model_bpe": isinstance(model, Mapping) and model.get("type") == "BPE",
        "vocab_size_32000": (
            isinstance(vocab, Mapping)
            and len(vocab) == TOKENIZER_VOCAB_SIZE
            and model_config.get("vocab_size") == TOKENIZER_VOCAB_SIZE
        ),
        "special_ids_exact": ids == TOKENIZER_SPECIAL_IDS,
        "tokenizer_config_specials_exact": (
            _token_content(tokenizer_config.get("bos_token")) == "<s>"
            and _token_content(tokenizer_config.get("eos_token")) == "</s>"
            and _token_content(tokenizer_config.get("unk_token")) == "<unk>"
            and tokenizer_config.get("pad_token") is None
        ),
        "special_token_map_exact": set(special_map) == {"bos_token", "eos_token", "unk_token"},
        "admission_pad_policy_locked": TOKENIZER_PAD_POLICY
        == "SET_PAD_TO_EOS_ID_2_AT_RUNTIME",
    }
    warnings = [
        {
            "code": "UPSTREAM_PAD_POLICY_CONFLICT",
            "severity": "MEDIUM",
            "protocol_hard_gate": False,
            "detail": (
                f"tokenizer_config has no pad token and generation_config uses "
                f"pad_token_id={generation.get('pad_token_id')}; E26 admission locks "
                "pad to EOS id 2."
            ),
        },
        {
            "code": "TOKENIZER_REVISION_NOT_CRYPTOGRAPHICALLY_LINKED_TO_CHECKPOINT",
            "severity": "HIGH",
            "protocol_hard_gate": False,
            "detail": (
                "Official training source names TinyLlama but does not bind this exact revision "
                "to the community checkpoint."
            ),
        },
    ]
    return (
        {
            "repo_id": TOKENIZER_REPO_ID,
            "revision": metadata.get("sha"),
            "file_metadata": {
                row.filename: {
                    "bytes": row.size,
                    "blob_id": row.blob_id,
                    "sha256": row.sha256,
                    "lfs": row.lfs,
                }
                for row in TOKENIZER_FILES
            },
            "observed": {
                "tokenizer_type": model.get("type") if isinstance(model, Mapping) else None,
                "vocab_size": len(vocab) if isinstance(vocab, Mapping) else None,
                "special_ids": ids,
                "tokenizer_config_pad_token": tokenizer_config.get("pad_token"),
                "generation_config_pad_token_id": generation.get("pad_token_id"),
            },
            "admission_pad_policy": {
                "policy": TOKENIZER_PAD_POLICY,
                "pad_token": "</s>",
                "pad_token_id": 2,
                "silent_id_0_padding_allowed": False,
            },
            "hard_checks": checks,
            "passed": all(checks.values()),
        },
        warnings,
    )


def build_final_provenance_receipt(
    *,
    official_source: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    tokenizer: Mapping[str, Any],
    warnings: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a deterministic strict-JSON receipt with hard checks separated."""

    sections = {
        "official_source": deepcopy(dict(official_source)),
        "checkpoint": deepcopy(dict(checkpoint)),
        "tokenizer": deepcopy(dict(tokenizer)),
    }
    hard_checks: dict[str, bool] = {}
    for section_name, section in sections.items():
        checks = section.get("hard_checks")
        if not isinstance(checks, Mapping) or not all(
            isinstance(value, bool) for value in checks.values()
        ):
            raise E26FinalProvenanceError(f"{section_name} lacks boolean hard checks")
        if section.get("passed") is not all(checks.values()):
            raise E26FinalProvenanceError(
                f"{section_name} disposition differs from its hard checks"
            )
        hard_checks.update({f"{section_name}.{key}": value for key, value in checks.items()})
    warning_rows = [deepcopy(dict(row)) for row in warnings]
    for row in warning_rows:
        if (
            row.get("protocol_hard_gate") is not False
            or not isinstance(row.get("code"), str)
            or not isinstance(row.get("severity"), str)
            or not isinstance(row.get("detail"), str)
        ):
            raise E26FinalProvenanceError("Warnings must not masquerade as protocol hard checks")
    passed = all(hard_checks.values())
    payload: dict[str, Any] = {
        "schema_version": _RECEIPT_SCHEMA,
        "manifest_type": _RECEIPT_TYPE,
        "scientific_evidence": False,
        "evidence_tier": "SCIENTIFIC_INPUT_PROVENANCE",
        "claim_ceiling": "EXTERNAL_SOURCE_AND_CHECKPOINT_ADMISSION_ONLY",
        "network_policy": "METADATA_AND_SMALL_TOKENIZER_FILES_ONLY",
        "large_checkpoint_downloaded": False,
        "official_source": sections["official_source"],
        "checkpoint": sections["checkpoint"],
        "tokenizer": sections["tokenizer"],
        "protocol_hard_checks": hard_checks,
        "warnings": warning_rows,
        "warning_count": len(warning_rows),
        "external_metadata_eligible": passed,
        "checkpoint_bytes_ready": checkpoint.get("checkpoint_bytes_ready") is True,
        "scientific_e26a_started": False,
        "passed": passed,
    }
    payload["receipt_sha256"] = sha256_canonical_json(payload)
    return validate_final_provenance_receipt(payload)


def validate_final_provenance_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(dict(payload))
    claimed = normalized.pop("receipt_sha256", None)
    if claimed != sha256_canonical_json(normalized):
        raise E26FinalProvenanceError("Final provenance receipt SHA-256 changed")
    normalized["receipt_sha256"] = claimed
    if (
        normalized.get("schema_version") != _RECEIPT_SCHEMA
        or normalized.get("manifest_type") != _RECEIPT_TYPE
        or normalized.get("scientific_evidence") is not False
        or normalized.get("large_checkpoint_downloaded") is not False
        or normalized.get("scientific_e26a_started") is not False
    ):
        raise E26FinalProvenanceError("Final provenance receipt contract changed")
    checks = normalized.get("protocol_hard_checks")
    warnings = normalized.get("warnings")
    if not isinstance(checks, Mapping) or not checks or not all(
        isinstance(value, bool) for value in checks.values()
    ):
        raise E26FinalProvenanceError("Final provenance hard-check map is invalid")
    if not isinstance(warnings, list) or normalized.get("warning_count") != len(warnings):
        raise E26FinalProvenanceError("Final provenance warning population is invalid")
    if any(
        not isinstance(row, Mapping)
        or row.get("protocol_hard_gate") is not False
        or not isinstance(row.get("code"), str)
        or not isinstance(row.get("severity"), str)
        or not isinstance(row.get("detail"), str)
        for row in warnings
    ):
        raise E26FinalProvenanceError("Warnings and protocol hard checks were conflated")
    for section_name in ("official_source", "checkpoint", "tokenizer"):
        section = normalized.get(section_name)
        if not isinstance(section, Mapping):
            raise E26FinalProvenanceError(f"Receipt lacks {section_name}")
        section_checks = section.get("hard_checks")
        if not isinstance(section_checks, Mapping) or section.get("passed") is not all(
            section_checks.values()
        ):
            raise E26FinalProvenanceError(
                f"{section_name} disposition differs from its hard checks"
            )
    expected = all(checks.values())
    if normalized.get("passed") is not expected or normalized.get(
        "external_metadata_eligible"
    ) is not expected:
        raise E26FinalProvenanceError("Final provenance disposition is inconsistent")
    checkpoint = normalized.get("checkpoint")
    if not isinstance(checkpoint, Mapping) or normalized.get("checkpoint_bytes_ready") is not (
        checkpoint.get("checkpoint_bytes_ready") is True
    ):
        raise E26FinalProvenanceError("Checkpoint byte-readiness status is inconsistent")
    return normalized


def write_final_provenance_receipt(path: str | Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(path).expanduser()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite final provenance receipt: {destination}")
    validated = validate_final_provenance_receipt(payload)
    write_json_strict(destination, validated)
    return destination


def fetch_json_metadata(url: str, *, timeout_seconds: float = 30.0) -> Mapping[str, Any]:
    payload = fetch_small_bytes(url, timeout_seconds=timeout_seconds)
    return _strict_json_bytes(payload, url)


def fetch_small_bytes(
    url: str,
    *,
    timeout_seconds: float = 30.0,
    maximum_bytes: int = _MAX_METADATA_BYTES,
) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "CATENA-E26-provenance/1"})
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
        length = response.headers.get("Content-Length")
        if length is not None and int(length) > maximum_bytes:
            raise E26FinalProvenanceError(f"Refusing oversized metadata response: {url}")
        payload = bytes(response.read(maximum_bytes + 1))
    if len(payload) > maximum_bytes:
        raise E26FinalProvenanceError(f"Refusing oversized metadata response: {url}")
    return payload


def hf_model_api_url(repo_id: str, revision: str) -> str:
    return f"https://huggingface.co/api/models/{repo_id}/revision/{revision}?blobs=true"


def hf_resolve_url(repo_id: str, revision: str, filename: str) -> str:
    return f"https://huggingface.co/{repo_id}/resolve/{revision}/{filename}"


def git_ls_remote(url: str) -> dict[str, str]:
    result = subprocess.run(
        ("git", "ls-remote", url),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise E26FinalProvenanceError(f"git ls-remote failed: {result.stderr.strip()}")
    return parse_ls_remote(result.stdout)
