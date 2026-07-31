from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from catena.core.provenance_v61 import (
    SHA256_PATTERN,
    read_json_object_strict,
    sha256_canonical_json,
    sha256_file,
)

from .audit_contract import e26_execution_source_inventory
from .general_corpus import ScientificDataReadiness, validate_scientific_data_bundle
from .hashing import hash_mapping


class E26ReadinessBlocked(RuntimeError):
    """Raised before any E26a gate run when a frozen dependency is invalid."""


_LOCKED_CONFIG_SECTIONS = (
    "safety",
    "candidate_selection",
    "matching",
    "backend_gates",
    "data",
    "gate_population",
    "floor_gate",
    "throughput",
    "claim_gates",
    "mechanism_gates",
)

_SCIENTIFIC_SOURCE_PATHS = (
    "configs",
    "experiments",
    "schemas",
    "scripts",
    "src",
    "tools",
    "pyproject.toml",
    "requirements.v8_1.txt",
)


def _git(root: Path, *arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *arguments],
            cwd=root,
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
    except subprocess.CalledProcessError as error:
        raise E26ReadinessBlocked(
            f"Git readiness check failed: git {' '.join(arguments)}"
        ) from error


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise E26ReadinessBlocked("E26a config is not a mapping")
    if payload.get("schema_version") != "catena-v8.1":
        raise E26ReadinessBlocked("E26a config schema_version changed")
    if payload.get("experiment") != "e26a_operator_data_gate":
        raise E26ReadinessBlocked("E26a config experiment ID changed")
    return payload


def _locked_sections(config: dict[str, Any]) -> dict[str, Any]:
    return {key: config[key] for key in _LOCKED_CONFIG_SECTIONS if key in config}


def _require_sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
        raise E26ReadinessBlocked(f"{field} is not a SHA-256 digest")
    return value


def validate_e26a_control_inputs(
    *,
    repo_root: str | Path,
    config_path: str | Path,
    protocol_lock_path: str | Path,
    backend_manifest_path: str | Path,
    require_clean_source: bool = True,
    verify_backend_source: bool = True,
) -> dict[str, Any]:
    """Validate prospective protocol and optimized candidate without data access."""

    root = Path(repo_root).expanduser().resolve(strict=True)
    config_source = Path(config_path).expanduser().resolve(strict=True)
    protocol_source = Path(protocol_lock_path).expanduser().resolve(strict=True)
    backend_source = Path(backend_manifest_path).expanduser().resolve(strict=True)
    config = _load_config(config_source)
    protocol = read_json_object_strict(protocol_source)
    backend = read_json_object_strict(backend_source)

    expected_config_hash = hash_mapping(config)
    if protocol.get("schema_version") != "catena-v8.1":
        raise E26ReadinessBlocked("Protocol lock schema_version changed")
    if protocol.get("experiment") != "e26a_operator_data_gate":
        raise E26ReadinessBlocked("Protocol lock experiment ID changed")
    if protocol.get("locked") is not True:
        raise E26ReadinessBlocked("E26a protocol is not locked")
    if protocol.get("config_hash") != expected_config_hash:
        raise E26ReadinessBlocked("E26a config bytes no longer match the protocol lock")
    if protocol.get("full_config_snapshot") != config:
        raise E26ReadinessBlocked("Protocol full_config_snapshot differs from the config")
    if protocol.get("thresholds") != _locked_sections(config):
        raise E26ReadinessBlocked("Protocol thresholds do not snapshot all E26a gates")

    execution_inputs = protocol.get("execution_inputs")
    if not isinstance(execution_inputs, dict):
        raise E26ReadinessBlocked("Protocol lock lacks execution_inputs")
    current_inventory = e26_execution_source_inventory(root)
    source_hash = _require_sha(
        execution_inputs.get("source_tree_sha256"),
        "protocol.execution_inputs.source_tree_sha256",
    )
    if (
        source_hash != current_inventory["source_tree_sha256"]
        or protocol.get("source_hash") != source_hash
    ):
        raise E26ReadinessBlocked("Protocol execution-source inventory changed")

    if backend.get("schema_version") != "catena-v8.1":
        raise E26ReadinessBlocked("Backend manifest schema_version changed")
    if backend.get("backend_type") == "REFERENCE_PYTHON":
        raise E26ReadinessBlocked("Reference backend cannot enter E26a")
    if backend.get("candidate_codegen_capable") is not True:
        raise E26ReadinessBlocked("Backend code generation is not E26a-capable")
    if backend.get("e26a_candidate_capable") is not True:
        raise E26ReadinessBlocked("Stage-2 numerical/restart backend preflight did not pass")
    if backend.get("e26a_gate_capable") is not False:
        raise E26ReadinessBlocked("Candidate manifest improperly pre-opened E26a")
    if backend.get("parity_verified") is not False:
        raise E26ReadinessBlocked("Candidate manifest improperly claims full parity")
    if backend.get("scientific_main_capable") is not False:
        raise E26ReadinessBlocked("Candidate manifest improperly opened scientific MAIN")
    if backend.get("fallback_count") != 0 or backend.get("graph_break_count") != 0:
        raise E26ReadinessBlocked("Backend candidate used a fallback or graph break")

    source_commit = backend.get("source_commit")
    if not isinstance(source_commit, str) or len(source_commit) != 40:
        raise E26ReadinessBlocked("Backend source_commit is not a full Git SHA")
    if verify_backend_source:
        _git(root, "cat-file", "-e", f"{source_commit}^{{commit}}")
        unchanged = subprocess.run(
            ["git", "diff", "--quiet", source_commit, "--", *_SCIENTIFIC_SOURCE_PATHS],
            cwd=root,
            check=False,
        )
        if unchanged.returncode != 0:
            raise E26ReadinessBlocked("Scientific source differs from the backend smoke commit")
    if require_clean_source:
        dirty = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
        if dirty:
            raise E26ReadinessBlocked("E26a requires a clean committed worktree")

    return {
        "config_sha256": sha256_file(config_source),
        "config_hash": expected_config_hash,
        "protocol_lock_sha256": sha256_file(protocol_source),
        "backend_manifest_sha256": sha256_file(backend_source),
        "backend_source_commit": source_commit,
        "candidate_id": backend.get("candidate_id", "d512_ctx4096"),
    }


@dataclass(frozen=True, slots=True)
class E26AReadiness:
    control: dict[str, Any]
    data: ScientificDataReadiness
    readiness_sha256: str

    def payload_without_hash(self) -> dict[str, Any]:
        return {
            "schema_version": "catena-v8.1",
            "manifest_type": "E26A_EXECUTION_READINESS",
            "status": "READY_FOR_EXPLICIT_E26A_APPROVAL",
            "scientific_main_started": False,
            "control": self.control,
            "data": self.data.as_dict(),
        }

    def as_dict(self) -> dict[str, Any]:
        payload = self.payload_without_hash()
        payload["readiness_sha256"] = self.readiness_sha256
        return payload


def validate_e26a_readiness(
    *,
    repo_root: str | Path,
    config_path: str | Path,
    protocol_lock_path: str | Path,
    backend_manifest_path: str | Path,
    tokenizer_manifest_path: str | Path,
    corpus_manifest_path: str | Path,
    require_clean_source: bool = True,
) -> E26AReadiness:
    """Fail closed unless every frozen E26a execution input is ready."""

    control = validate_e26a_control_inputs(
        repo_root=repo_root,
        config_path=config_path,
        protocol_lock_path=protocol_lock_path,
        backend_manifest_path=backend_manifest_path,
        require_clean_source=require_clean_source,
    )
    config = _load_config(Path(config_path).expanduser().resolve(strict=True))
    candidate_id = str(control["candidate_id"])
    candidates = config.get("model_candidates")
    if not isinstance(candidates, list):
        raise E26ReadinessBlocked("E26a config lacks model_candidates")
    candidate = next(
        (item for item in candidates if isinstance(item, dict) and item.get("id") == candidate_id),
        None,
    )
    if candidate is None:
        raise E26ReadinessBlocked("Backend candidate is absent from the locked config")
    data = validate_scientific_data_bundle(
        tokenizer_manifest_path=tokenizer_manifest_path,
        corpus_manifest_path=corpus_manifest_path,
        sequence_length=int(candidate["context_length"]),
    )
    payload = {
        "schema_version": "catena-v8.1",
        "manifest_type": "E26A_EXECUTION_READINESS",
        "status": "READY_FOR_EXPLICIT_E26A_APPROVAL",
        "scientific_main_started": False,
        "control": control,
        "data": data.as_dict(),
    }
    return E26AReadiness(
        control=control,
        data=data,
        readiness_sha256=sha256_canonical_json(payload),
    )
