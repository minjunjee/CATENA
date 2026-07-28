from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from catena.core.config import load_config
from catena.core.provenance_v61 import (
    ProvenanceValidationError,
    ValidatedRun,
    read_json_object_strict,
    sha256_canonical_json,
    sha256_file,
    validate_run_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

PROTOCOL_PATH = REPO_ROOT / "docs/E05A_E05B_PROTOCOL_PREREGISTRATION_FROZEN_KO.md"
PROTOCOL_LOCK_PATH = (
    REPO_ROOT / "docs/E05A_E05B_PROTOCOL_PREREGISTRATION_LOCK_KO.md"
)
E05A_CONFIG_PATH = REPO_ROOT / "configs/e05a_semantic_protocol_lock.yaml"
E05B_CONFIG_PATH = REPO_ROOT / "configs/e05b_semantic_anchor.yaml"

PINNED_PROTOCOL_SHA256 = (
    "e235a351ca84589d92a211e78f6f4ebe6631ce228adad8f8da34dec17527b8e0"
)
PINNED_PROTOCOL_LOCK_SHA256 = (
    "4aaadf16eb02abff19f519191c0c56de0ad383cf01dfeea91fedacfb8a41980e"
)
PINNED_E05A_CONFIG_CANONICAL_SHA256 = (
    "4d95f6afa16ea66488125825a75f97831677a563c0bf3b2f9a09934535e637e7"
)
PINNED_E05A_CONFIG_FILE_SHA256 = (
    "2895bb250f92744e0238d56fcdb85816584195ea6d2aba5a767899377a1ec6c2"
)
PINNED_E05B_CONFIG_CANONICAL_SHA256 = (
    "c7c37a67c978b3cd2e3cf93a54669f57ebed86a6ce59e75194d59d6003b9d7ef"
)
PINNED_E05B_CONFIG_FILE_SHA256 = (
    "67eb6d5efecbb1dc653efc388d8e0d18e386498644f3b6594e63f8d8fdea4eb8"
)
PINNED_E04_FREEZE_SHA256 = (
    "6d225b673da998cef9131af0b2d49fc699f89af2159f40c302898144c2765b30"
)
PINNED_E04_MANIFEST_SHA256 = (
    "7ae767b8fb7226588fd770194783308ff89a24b52b6f67389c55dab0e044ff7d"
)
PINNED_E04_REPORT_SHA256 = (
    "7111e23ab70558a5130dff6937a3264c3ec6e4b5ea342183e52ea28f1bb36444"
)
PINNED_E04_RUN_ID = "20260727T054917.678326Z"


@dataclass(frozen=True, slots=True)
class FrozenE04Dependency:
    artifact_root: Path
    freeze_path: Path
    run_dir: Path
    freeze: dict[str, Any]
    report: dict[str, Any]

    def dependency_record(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "experiment_id": "e04_functional_mediation",
            "run_id": PINNED_E04_RUN_ID,
            "run_dir": str(self.run_dir),
            "status": "PASS",
            "run_mode": "main",
            "eligibility": {"main": True, "full": True},
            "evidence_role": "immutable_supported_h4_dependency",
            "freeze_id": "E04_ARTIFACT_FREEZE_V1",
            "freeze_sha256": PINNED_E04_FREEZE_SHA256,
            "manifest_sha256": PINNED_E04_MANIFEST_SHA256,
            "report_sha256": PINNED_E04_REPORT_SHA256,
            "full_h4_claim_open": True,
            "original_e02_confirmatory_status": "INCONCLUSIVE",
            "e02b_prospective_repair_status": "SUPPORTED",
            "scientific_evidence": False,
        }


def validate_frozen_e05_protocol() -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate both configs and the protocol/lock before any data generation."""

    expected_files = {
        PROTOCOL_PATH: PINNED_PROTOCOL_SHA256,
        PROTOCOL_LOCK_PATH: PINNED_PROTOCOL_LOCK_SHA256,
        E05A_CONFIG_PATH: PINNED_E05A_CONFIG_FILE_SHA256,
        E05B_CONFIG_PATH: PINNED_E05B_CONFIG_FILE_SHA256,
    }
    for path, expected in expected_files.items():
        if path.is_symlink() or not path.is_file():
            raise ProvenanceValidationError(f"Frozen E05 file is missing: {path}.")
        if sha256_file(path) != expected:
            raise ProvenanceValidationError(f"Frozen E05 file hash mismatch: {path}.")

    e05a = load_config(E05A_CONFIG_PATH)
    e05b = load_config(E05B_CONFIG_PATH)
    if sha256_canonical_json(e05a) != PINNED_E05A_CONFIG_CANONICAL_SHA256:
        raise ProvenanceValidationError("E05a canonical config hash mismatch.")
    if sha256_canonical_json(e05b) != PINNED_E05B_CONFIG_CANONICAL_SHA256:
        raise ProvenanceValidationError("E05b canonical config hash mismatch.")
    if e05a.get("experiment_id") != "e05a_semantic_protocol_lock":
        raise ProvenanceValidationError("Frozen E05a experiment identity changed.")
    if e05b.get("experiment_id") != "e05b_semantic_anchor":
        raise ProvenanceValidationError("Frozen E05b experiment identity changed.")
    return e05a, e05b


def _direct_hashed_child(
    directory: Path,
    name: object,
    expected_sha256: object,
) -> Path:
    if not isinstance(name, str) or Path(name).name != name:
        raise ProvenanceValidationError(f"Unsafe frozen artifact name: {name!r}.")
    if not isinstance(expected_sha256, str):
        raise ProvenanceValidationError(f"Missing frozen hash for {name!r}.")
    path = directory / name
    if path.is_symlink() or not path.is_file():
        raise ProvenanceValidationError(f"Frozen artifact is missing: {path}.")
    if sha256_file(path) != expected_sha256:
        raise ProvenanceValidationError(f"Frozen artifact hash mismatch: {path}.")
    return path


def validate_frozen_e04_dependency(
    artifact_root: str | Path,
) -> FrozenE04Dependency:
    root = Path(artifact_root).expanduser().resolve(strict=True)
    freeze_path = root / "E04_ARTIFACT_FREEZE_V1.json"
    if freeze_path.is_symlink() or not freeze_path.is_file():
        raise ProvenanceValidationError(f"E04 additive freeze is missing: {freeze_path}.")
    if sha256_file(freeze_path) != PINNED_E04_FREEZE_SHA256:
        raise ProvenanceValidationError("E04 additive freeze hash mismatch.")
    freeze = read_json_object_strict(freeze_path)
    if (
        freeze.get("freeze_id") != "E04_ARTIFACT_FREEZE_V1"
        or freeze.get("experiment_id") != "e04_functional_mediation"
        or freeze.get("run_id") != PINNED_E04_RUN_ID
        or freeze.get("execution_status") != "PASS"
        or freeze.get("eligibility") != {"main": True, "full": True}
    ):
        raise ProvenanceValidationError("E04 additive freeze identity/status changed.")
    claim = freeze.get("claim_status")
    if not isinstance(claim, dict) or claim.get("full_h4_claim_open") is not True:
        raise ProvenanceValidationError("Frozen E04 does not open H4.")
    disposition = freeze.get("dependency_disposition")
    if not isinstance(disposition, dict) or disposition != {
        "original_e02_confirmatory_status": "INCONCLUSIVE",
        "original_e02_h2_claim_open": False,
        "e02b_prospective_repair_status": "SUPPORTED",
    }:
        raise ProvenanceValidationError("E02/E02b disposition changed in E04 freeze.")

    run_dir = root / "e04_functional_mediation" / PINNED_E04_RUN_ID
    artifacts = freeze.get("run_artifacts")
    if not isinstance(artifacts, dict):
        raise ProvenanceValidationError("E04 freeze lacks run_artifacts.")
    for name, descriptor in artifacts.items():
        if not isinstance(descriptor, dict):
            raise ProvenanceValidationError(f"Invalid E04 descriptor for {name}.")
        _direct_hashed_child(run_dir, name, descriptor.get("sha256"))
    if sha256_file(run_dir / "run_manifest.json") != PINNED_E04_MANIFEST_SHA256:
        raise ProvenanceValidationError("Pinned E04 manifest hash mismatch.")
    if sha256_file(run_dir / "report.json") != PINNED_E04_REPORT_SHA256:
        raise ProvenanceValidationError("Pinned E04 report hash mismatch.")
    report = read_json_object_strict(run_dir / "report.json")
    if (
        report.get("status") != "PASS"
        or report.get("run_mode") != "main"
        or report.get("eligibility") != {"main": True, "full": True}
        or report.get("claim_gate", {}).get("full_h4_claim_open") is not True
    ):
        raise ProvenanceValidationError("Pinned E04 report no longer satisfies H4.")
    return FrozenE04Dependency(root, freeze_path, run_dir, freeze, report)


def validate_completed_e05a_run(
    run_dir: str | Path,
    *,
    require_go: bool,
) -> ValidatedRun:
    resolved = Path(run_dir).expanduser().resolve(strict=True)
    run = validate_run_manifest(resolved, resolved.parents[1])
    if run.experiment_id != "e05a_semantic_protocol_lock":
        raise ProvenanceValidationError("Dependency is not an E05a run.")
    if run.run_mode != "main" or not run.main_eligible or not run.full_eligible:
        raise ProvenanceValidationError("E05a dependency is not a complete main run.")
    if require_go and run.report.get("e05a_design_status") != "GO":
        raise ProvenanceValidationError("E05a design status is not GO.")
    protocol = run.report.get("protocol_lock")
    if not isinstance(protocol, Mapping) or (
        protocol.get("protocol_sha256") != PINNED_PROTOCOL_SHA256
        or protocol.get("protocol_lock_sha256") != PINNED_PROTOCOL_LOCK_SHA256
    ):
        raise ProvenanceValidationError("E05a report does not pin the frozen protocol.")
    return run


def validate_completed_e05a_human_audit(
    run_dir: str | Path,
    *,
    expected_e05a: ValidatedRun,
) -> ValidatedRun:
    resolved = Path(run_dir).expanduser().resolve(strict=True)
    run = validate_run_manifest(resolved, resolved.parents[1])
    if run.experiment_id != "e05a_semantic_audit_adjudication":
        raise ProvenanceValidationError("Dependency is not an E05a human-audit run.")
    if run.run_mode != "main" or not run.main_eligible or not run.full_eligible:
        raise ProvenanceValidationError(
            "Human-audit dependency is not a complete main run."
        )
    if run.report.get("human_audit_status") != "PASSED":
        raise ProvenanceValidationError("Human audit did not pass.")
    gate = run.report.get("claim_gate")
    if not isinstance(gate, dict) or (
        gate.get("opens_h5_claim") is not False
        or gate.get("is_e05b_training_dependency") is not True
    ):
        raise ProvenanceValidationError("Human-audit claim/dependency status changed.")
    dependencies = run.manifest.get("dependencies")
    if not isinstance(dependencies, list):
        raise ProvenanceValidationError("Human-audit run lacks dependencies.")
    matches = [
        dependency
        for dependency in dependencies
        if isinstance(dependency, dict)
        and dependency.get("experiment_id") == expected_e05a.experiment_id
        and dependency.get("run_id") == expected_e05a.run_id
        and dependency.get("manifest_sha256") == expected_e05a.manifest_sha256
        and dependency.get("report_sha256") == expected_e05a.report_sha256
    ]
    if len(matches) != 1:
        raise ProvenanceValidationError(
            "Human audit does not pin the selected E05a GO artifact."
        )
    return run
