from __future__ import annotations

import hashlib
import json
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

import catena.lm.e26_final_provenance as provenance
from catena.core.provenance_v61 import StrictJSONError, read_json_object_strict
from catena.lm.e26_final_provenance import (
    CHECKPOINT_ALIAS_FILE,
    CHECKPOINT_FILE,
    CHECKPOINT_REPO_ID,
    CHECKPOINT_REVISION,
    TOKENIZER_REPO_ID,
    TOKENIZER_REVISION,
    E26FinalProvenanceError,
    HfFileExpectation,
    OfficialSourceExpectation,
    audit_checkpoint_metadata,
    audit_official_source,
    audit_tokenizer_metadata,
    build_final_provenance_receipt,
    validate_final_provenance_receipt,
    write_final_provenance_receipt,
)


def _git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(("git", "-C", str(root), *arguments), text=True).strip()


def test_official_source_requires_exact_clean_commit_tree_license_and_blobs(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "official"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "CATENA Test")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "remote", "add", "origin", "https://example.invalid/official.git")
    license_text = (
        "Nvidia Source Code License-NC\n"
        "The Work only may be used or intended for use non-commercially.\n"
    )
    (repo / "LICENSE").write_text(license_text, encoding="utf-8")
    (repo / "operator.py").write_text("operator = 'fixed'\n", encoding="utf-8")
    _git(repo, "add", "LICENSE", "operator.py")
    _git(repo, "commit", "-m", "official fixture")
    head = _git(repo, "rev-parse", "HEAD")
    expectation = OfficialSourceExpectation(
        remote_url="https://example.invalid/official.git",
        commit=head,
        tree=_git(repo, "rev-parse", "HEAD^{tree}"),
        license_blob=_git(repo, "rev-parse", "HEAD:LICENSE"),
        license_bytes=len(license_text.encode()),
        license_sha256=hashlib.sha256(license_text.encode()).hexdigest(),
        key_blobs={"operator.py": _git(repo, "rev-parse", "HEAD:operator.py")},
    )
    refs = {"HEAD": head, "refs/heads/main": head}
    receipt = audit_official_source(repo, remote_refs=refs, expectation=expectation)
    assert receipt["passed"] is True
    assert all(receipt["hard_checks"].values())

    (repo / "operator.py").write_text("operator = 'dirty'\n", encoding="utf-8")
    dirty = audit_official_source(repo, remote_refs=refs, expectation=expectation)
    assert dirty["passed"] is False
    assert dirty["hard_checks"]["local_checkout_clean"] is False
    assert dirty["hard_checks"]["key_source_blobs_exact"] is True


def _checkpoint_metadata() -> dict[str, Any]:
    lfs = {"size": CHECKPOINT_FILE.size, "sha256": CHECKPOINT_FILE.sha256}
    return {
        "id": CHECKPOINT_REPO_ID,
        "sha": CHECKPOINT_REVISION,
        "private": False,
        "gated": False,
        "disabled": False,
        "siblings": [
            {
                "rfilename": CHECKPOINT_FILE.filename,
                "size": CHECKPOINT_FILE.size,
                "blobId": CHECKPOINT_FILE.blob_id,
                "lfs": dict(lfs),
            },
            {
                "rfilename": CHECKPOINT_ALIAS_FILE,
                "size": CHECKPOINT_FILE.size,
                "blobId": CHECKPOINT_FILE.blob_id,
                "lfs": dict(lfs),
            },
        ],
    }


def test_checkpoint_metadata_passes_hard_contract_but_alias_is_warning() -> None:
    receipt, warnings = audit_checkpoint_metadata(_checkpoint_metadata())
    assert receipt["passed"] is True
    assert receipt["checkpoint_bytes_ready"] is False
    assert receipt["training_token_label_claim_eligible"] is False
    alias = [row for row in warnings if row["code"] == "CHECKPOINT_95B_AND_100B_BYTE_IDENTICAL"]
    assert len(alias) == 1
    assert alias[0]["protocol_hard_gate"] is False


def test_checkpoint_metadata_or_supplied_local_byte_mismatch_blocks(tmp_path: Path) -> None:
    changed = _checkpoint_metadata()
    changed["siblings"][0]["lfs"]["sha256"] = "0" * 64
    receipt, _warnings = audit_checkpoint_metadata(changed)
    assert receipt["passed"] is False
    assert receipt["hard_checks"]["model_100b_metadata_exact"] is False

    local = tmp_path / "model-100b.pth"
    local.write_bytes(b"not the checkpoint")
    receipt, _warnings = audit_checkpoint_metadata(
        _checkpoint_metadata(), local_checkpoint=local
    )
    assert receipt["passed"] is False
    assert receipt["local_checkpoint"]["status"] == "MISMATCH"
    assert receipt["hard_checks"]["local_checkpoint_bytes_exact"] is False


def _tokenizer_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
    vocab.update({f"token_{index}": index for index in range(3, 32_000)})
    payloads = {
        "tokenizer.json": json.dumps(
            {"model": {"type": "BPE", "vocab": vocab}}, sort_keys=True
        ).encode(),
        "tokenizer.model": b"sentencepiece-fixture",
        "tokenizer_config.json": json.dumps(
            {
                "bos_token": {"content": "<s>"},
                "eos_token": {"content": "</s>"},
                "unk_token": {"content": "<unk>"},
                "pad_token": None,
            },
            sort_keys=True,
        ).encode(),
        "special_tokens_map.json": json.dumps(
            {
                "bos_token": {"content": "<s>"},
                "eos_token": {"content": "</s>"},
                "unk_token": {"content": "<unk>"},
            },
            sort_keys=True,
        ).encode(),
        "config.json": json.dumps({"vocab_size": 32_000}, sort_keys=True).encode(),
        "generation_config.json": json.dumps(
            {"pad_token_id": 0}, sort_keys=True
        ).encode(),
    }
    expectations = tuple(
        HfFileExpectation(
            filename=name,
            size=len(payload),
            blob_id=hashlib.sha1(name.encode()).hexdigest(),  # noqa: S324 - Git blob fixture
            sha256=hashlib.sha256(payload).hexdigest(),
            lfs=name == "tokenizer.model",
        )
        for name, payload in payloads.items()
    )
    monkeypatch.setattr(provenance, "TOKENIZER_FILES", expectations)
    siblings = []
    for row in expectations:
        siblings.append(
            {
                "rfilename": row.filename,
                "size": row.size,
                "blobId": row.blob_id,
                "lfs": (
                    {"size": row.size, "sha256": row.sha256} if row.lfs else None
                ),
            }
        )
    metadata = {
        "id": TOKENIZER_REPO_ID,
        "sha": TOKENIZER_REVISION,
        "private": False,
        "gated": False,
        "disabled": False,
        "siblings": siblings,
    }
    return metadata, payloads


def test_tokenizer_binds_revision_files_vocab_special_ids_and_pad_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, files = _tokenizer_fixture(monkeypatch)
    receipt, warnings = audit_tokenizer_metadata(metadata, files=files)
    assert receipt["passed"] is True
    assert receipt["observed"]["vocab_size"] == 32_000
    assert receipt["observed"]["special_ids"] == {"unk": 0, "bos": 1, "eos": 2}
    assert receipt["admission_pad_policy"]["pad_token_id"] == 2
    assert receipt["admission_pad_policy"]["silent_id_0_padding_allowed"] is False
    assert all(row["protocol_hard_gate"] is False for row in warnings)

    malformed = dict(files)
    malformed["tokenizer_config.json"] = malformed["tokenizer_config.json"].replace(
        b'"pad_token": null', b'"pad_token": "<unk>"'
    )
    with pytest.raises(E26FinalProvenanceError, match="strict JSON"):
        audit_tokenizer_metadata(
            metadata, files={**malformed, "config.json": b'{"x":NaN}'}
        )


def test_tokenizer_pad_contract_fails_even_when_changed_bytes_are_rebound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, files = _tokenizer_fixture(monkeypatch)
    changed = dict(files)
    changed["tokenizer_config.json"] = changed["tokenizer_config.json"].replace(
        b'"pad_token": null', b'"pad_token": "<unk>"'
    )
    expectations = []
    for row in provenance.TOKENIZER_FILES:
        payload = changed[row.filename]
        expectations.append(
            HfFileExpectation(
                row.filename,
                len(payload),
                row.blob_id,
                hashlib.sha256(payload).hexdigest(),
                row.lfs,
            )
        )
    monkeypatch.setattr(provenance, "TOKENIZER_FILES", tuple(expectations))
    for sibling in metadata["siblings"]:
        expected = next(row for row in expectations if row.filename == sibling["rfilename"])
        sibling["size"] = expected.size
        if expected.lfs:
            sibling["lfs"] = {"size": expected.size, "sha256": expected.sha256}
    receipt, _warnings = audit_tokenizer_metadata(metadata, files=changed)
    assert receipt["passed"] is False
    assert receipt["hard_checks"]["small_file_bytes_exact"] is True
    assert receipt["hard_checks"]["tokenizer_config_specials_exact"] is False


def _section(passed: bool = True) -> dict[str, Any]:
    return {"hard_checks": {"bound": passed}, "passed": passed}


def test_final_receipt_keeps_warnings_separate_and_output_is_immutable(
    tmp_path: Path,
) -> None:
    checkpoint = {**_section(), "checkpoint_bytes_ready": False}
    warning = {
        "code": "WARNING_ONLY",
        "severity": "HIGH",
        "protocol_hard_gate": False,
        "detail": "does not alter hard admission",
    }
    receipt = build_final_provenance_receipt(
        official_source=_section(),
        checkpoint=checkpoint,
        tokenizer=_section(),
        warnings=[warning],
    )
    assert receipt["passed"] is True
    assert receipt["warning_count"] == 1
    assert validate_final_provenance_receipt(receipt) == receipt
    destination = tmp_path / "receipt.json"
    write_final_provenance_receipt(destination, receipt)
    assert read_json_object_strict(destination) == receipt
    with pytest.raises(FileExistsError):
        write_final_provenance_receipt(destination, receipt)

    tampered = deepcopy(receipt)
    tampered["warnings"][0]["detail"] = "changed"
    with pytest.raises(E26FinalProvenanceError, match="SHA-256"):
        validate_final_provenance_receipt(tampered)


def test_final_receipt_rejects_nonfinite_json() -> None:
    source = {**_section(), "diagnostic": float("nan")}
    with pytest.raises(StrictJSONError):
        build_final_provenance_receipt(
            official_source=source,
            checkpoint={**_section(), "checkpoint_bytes_ready": False},
            tokenizer=_section(),
            warnings=[
                {
                    "code": "BAD",
                    "severity": "HIGH",
                    "protocol_hard_gate": False,
                    "detail": "non-finite source diagnostic is forbidden",
                }
            ],
        )
