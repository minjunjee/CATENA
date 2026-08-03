"""Prospective protocol lock and predecessor guards for E26 Final.

E26 Final is a new official-kernel/pretrained-LM experiment.  It is not a
repair or reinterpretation of E26 Stage-3C/3D, so its lock must prove those
terminal reports still have the registered bytes before any admission work is
allowed to become scientific evidence.
"""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, Final

import yaml

from catena.core.provenance_v61 import sha256_canonical_json, sha256_file

EXPERIMENT_ID: Final = "E26_FINAL_GDN2_1P3B_TRANSACTIONAL_TRANSFER"
STAGE3D_DISPOSITION: Final = "STAGE3D_BLOCKED_FIXED_LAYOUT_NUMERICAL_INSTABILITY"
STAGE3D_REPORT_SHA256: Final = (
    "4c4528bf35052423896b29dbc12944e9ad5df3ec2f87410a9688417297a42650"
)
STAGE3D_SOURCE_COMMIT: Final = "47cbc68636367e32832c66ea57d1a827282ef447"
OFFICIAL_COMMIT: Final = "95709fc250357c2dd109361c353192f2aa5913f9"
CHECKPOINT_SHA256: Final = (
    "0322ebeefa96badb24d6b4b511c36b02374b704dc1a65b90eab2ee1383a9ce23"
)
TOKENIZER_REVISION: Final = "ff3c701f2424c7625fdefb9dd470f45ef18b02d6"


class E26FinalProtocolError(RuntimeError):
    """Raised when prospective or predecessor inputs do not match the lock."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise E26FinalProtocolError(f"{label} must be a mapping")
    return value


def load_protocol(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve(strict=True)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise E26FinalProtocolError("E26 Final protocol must be a YAML mapping")
    return payload


def validate_protocol(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate constants whose drift would alter the registered estimand."""

    evidence = _mapping(payload.get("evidence"), "evidence")
    source = _mapping(payload.get("source"), "source")
    official = _mapping(source.get("official_gdn2"), "source.official_gdn2")
    checkpoint = _mapping(source.get("checkpoint"), "source.checkpoint")
    tokenizer = _mapping(source.get("tokenizer"), "source.tokenizer")
    training = _mapping(payload.get("training"), "training")
    statistics = _mapping(payload.get("statistics"), "statistics")
    speed = _mapping(payload.get("speed"), "speed")
    runtime = _mapping(payload.get("runtime"), "runtime")
    expected = {
        "schema_version": "catena-e26-final-v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "PROSPECTIVE_LOCK",
        "stage3d_disposition": STAGE3D_DISPOSITION,
        "stage3d_report_sha256": STAGE3D_REPORT_SHA256,
        "stage3d_source_commit": STAGE3D_SOURCE_COMMIT,
        "official_commit": OFFICIAL_COMMIT,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "checkpoint_size": 17_401_727_659,
        "tokenizer_revision": TOKENIZER_REVISION,
        "seeds": [26011, 26022, 26033, 26044, 26055],
        "token_budget_candidates": [350_000_000, 500_000_000, 750_000_000, 1_000_000_000],
        "asymmetric_sesoi": 0.02,
        "symmetric_margin": 0.01,
        "retention_margin": 0.01,
        "minimum_throughput": 12_000,
        "peak_vram_limit_gib": 92,
        "maximum_wall_hours": 36,
        "gpu_count": 4,
    }
    observed = {
        "schema_version": payload.get("schema_version"),
        "experiment_id": payload.get("experiment_id"),
        "status": payload.get("status"),
        "stage3d_disposition": evidence.get("previous_stage3d_disposition"),
        "stage3d_report_sha256": evidence.get("previous_stage3d_report_sha256"),
        "stage3d_source_commit": evidence.get("previous_stage3d_source_commit"),
        "official_commit": official.get("commit"),
        "checkpoint_sha256": checkpoint.get("sha256"),
        "checkpoint_size": checkpoint.get("size_bytes"),
        "tokenizer_revision": tokenizer.get("revision"),
        "seeds": training.get("seeds"),
        "token_budget_candidates": training.get("token_budget_candidates"),
        "asymmetric_sesoi": statistics.get("asymmetric_absolute_sesoi"),
        "symmetric_margin": statistics.get("symmetric_equivalence_margin"),
        "retention_margin": statistics.get("retention_noninferiority_margin"),
        "minimum_throughput": speed.get("minimum_train_tokens_per_second_per_gpu"),
        "peak_vram_limit_gib": speed.get("peak_vram_limit_gib"),
        "maximum_wall_hours": speed.get("max_projected_wall_hours"),
        "gpu_count": runtime.get("gpu_count"),
    }
    if observed != expected:
        differing = sorted(key for key in expected if observed.get(key) != expected[key])
        raise E26FinalProtocolError(f"Registered E26 Final fields drifted: {differing}")
    if evidence.get("predecessors_immutable") is not True:
        raise E26FinalProtocolError("Predecessor evidence must remain immutable")
    if runtime.get("automatic_execution_after_gates") is not True:
        raise E26FinalProtocolError("Admission PASS must continue autonomously")
    return deepcopy(dict(payload))


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=repo, text=True).strip()


def build_protocol_receipt(
    *,
    config_path: str | Path,
    repo_root: str | Path,
    stage3d_report: str | Path,
) -> dict[str, Any]:
    config = Path(config_path).expanduser().resolve(strict=True)
    repo = Path(repo_root).expanduser().resolve(strict=True)
    predecessor = Path(stage3d_report).expanduser().resolve(strict=True)
    payload = validate_protocol(load_protocol(config))
    observed_predecessor_sha = sha256_file(predecessor)
    if observed_predecessor_sha != STAGE3D_REPORT_SHA256:
        raise E26FinalProtocolError("Stage-3D canonical report bytes changed")
    head = _git(repo, "rev-parse", "HEAD")
    status = _git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise E26FinalProtocolError("Protocol lock requires a clean committed worktree")
    receipt: dict[str, Any] = {
        "schema_version": "catena-e26-final-protocol-lock-v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "LOCKED",
        "scientific_evidence": False,
        "evidence_tier": "NON_EVIDENCE_ADMISSION",
        "source_commit": head,
        "source_branch": _git(repo, "branch", "--show-current"),
        "source_dirty": False,
        "config_path": str(config),
        "config_sha256": sha256_file(config),
        "protocol_canonical_sha256": sha256_canonical_json(payload),
        "stage3d_report_path": str(predecessor),
        "stage3d_report_sha256": observed_predecessor_sha,
        "stage3d_disposition_preserved": STAGE3D_DISPOSITION,
        "scientific_main_started": False,
    }
    receipt["receipt_sha256"] = sha256_canonical_json(receipt)
    return receipt


__all__ = [
    "E26FinalProtocolError",
    "EXPERIMENT_ID",
    "build_protocol_receipt",
    "load_protocol",
    "validate_protocol",
]
