from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from catena.core.config import load_config
from catena.core.provenance_v61 import (
    ManifestValidationRequirements,
    ProvenanceValidationError,
    ValidatedRun,
    read_json_object_strict,
    sha256_canonical_json,
    sha256_file,
    validate_run_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

R1_CONFIG_PATH = REPO_ROOT / "configs/e05a_r1_semantic_design_repair.yaml"
R1_PROTOCOL_PATH = (
    REPO_ROOT
    / "docs/E05A_R1_SEMANTIC_DESIGN_REPAIR_PREREGISTRATION_FROZEN_KO.md"
)
R1_PROTOCOL_LOCK_PATH = (
    REPO_ROOT
    / "docs/E05A_R1_SEMANTIC_DESIGN_REPAIR_PREREGISTRATION_LOCK_KO.md"
)

PINNED_R1_CONFIG_FILE_SHA256 = (
    "0f56777e4f8283154e317afa21c60c01f80d138457098155a3bfba49fafe190c"
)
PINNED_R1_CONFIG_CANONICAL_SHA256 = (
    "159880727a3a3e0e10edc876466238022828252118b1ebff0fecb0f4c9984b91"
)
PINNED_R1_PROTOCOL_SHA256 = (
    "5b06d68aa0fe5c3ec9ef0dceb1963b9ed114aa6570d16970df5fa1c1c9a100ee"
)
PINNED_R1_PROTOCOL_LOCK_SHA256 = (
    "cdcf21bc9b88093c9a82d9c08557693a2568288fef9728aa80e7b247d9824d1a"
)

PINNED_ORIGINAL_E05A_FREEZE_SHA256 = (
    "f6e6edebd303fb1b6d48cff9630516a8864dc317386778202da58a2a6c189122"
)
PINNED_ORIGINAL_E05A_CLAIM_SHA256 = (
    "f1c1e1585829048c47ac725ce03ffaa3293bfe67c10a23a1b205aaa4af432ec3"
)
PINNED_ORIGINAL_E05A_MANIFEST_SHA256 = (
    "c2571fa8c4ec184068dff3bb002dc08be1c503c147348bb19ada1fd1199b5e2b"
)
PINNED_ORIGINAL_E05A_REPORT_SHA256 = (
    "34bab0288d5bbe82e1debcfa81e51493f4fa280475b86f0d097cb2a4aff8057c"
)
PINNED_ORIGINAL_E05A_RUN_ID = "20260727T081532.073522Z"

EXPECTED_R1_SEEDS = (1103, 2207, 3301, 4409, 5501, 6607, 7703, 8807)


def validate_frozen_r1_protocol() -> dict[str, Any]:
    expected = {
        R1_CONFIG_PATH: PINNED_R1_CONFIG_FILE_SHA256,
        R1_PROTOCOL_PATH: PINNED_R1_PROTOCOL_SHA256,
        R1_PROTOCOL_LOCK_PATH: PINNED_R1_PROTOCOL_LOCK_SHA256,
    }
    for path, digest in expected.items():
        if path.is_symlink() or not path.is_file():
            raise ProvenanceValidationError(f"Frozen E05a-R1 file is missing: {path}.")
        if sha256_file(path) != digest:
            raise ProvenanceValidationError(
                f"Frozen E05a-R1 file hash mismatch: {path}."
            )
    config = load_config(R1_CONFIG_PATH)
    if sha256_canonical_json(config) != PINNED_R1_CONFIG_CANONICAL_SHA256:
        raise ProvenanceValidationError("E05a-R1 canonical config hash mismatch.")
    if config.get("experiment_id") != "e05a_r1_semantic_design_repair":
        raise ProvenanceValidationError("E05a-R1 experiment identity changed.")
    seeds = tuple(config.get("r1", {}).get("seeds", ()))
    if seeds != EXPECTED_R1_SEEDS:
        raise ProvenanceValidationError("E05a-R1 fixed seed list changed.")
    return config


def _direct_artifact_file(
    artifact_root: Path,
    filename: str,
    expected_sha256: str,
) -> Path:
    path = artifact_root / filename
    if path.is_symlink() or not path.is_file():
        raise ProvenanceValidationError(f"Pinned artifact is missing: {path}.")
    if sha256_file(path) != expected_sha256:
        raise ProvenanceValidationError(f"Pinned artifact hash mismatch: {path}.")
    return path


def validate_original_e05a_dependency(
    artifact_root: str | Path,
) -> ValidatedRun:
    root = Path(artifact_root).expanduser().resolve(strict=True)
    freeze_path = _direct_artifact_file(
        root,
        "E05A_ARTIFACT_FREEZE_V1.json",
        PINNED_ORIGINAL_E05A_FREEZE_SHA256,
    )
    claim_path = _direct_artifact_file(
        root,
        "E05A_CLAIM_STATUS.json",
        PINNED_ORIGINAL_E05A_CLAIM_SHA256,
    )
    freeze = read_json_object_strict(freeze_path)
    claim = read_json_object_strict(claim_path)
    if (
        freeze.get("freeze_id") != "E05A_ARTIFACT_FREEZE_V1"
        or freeze.get("experiment_id") != "e05a_semantic_protocol_lock"
        or freeze.get("run_id") != PINNED_ORIGINAL_E05A_RUN_ID
        or freeze.get("execution_status") != "PASS"
        or freeze.get("e05a_design_status") != "NO_GO"
        or freeze.get("h5_lite_claim_open") is not False
        or freeze.get("e05b_registry_generated") is not False
    ):
        raise ProvenanceValidationError("Original E05a freeze identity/status changed.")
    if (
        claim.get("source_run") != PINNED_ORIGINAL_E05A_RUN_ID
        or claim.get("e05a_original_status") != "NO_GO"
        or claim.get("e05b_execution_allowed") is not False
        or claim.get("h5_claim_open") is not False
        or claim.get("interpretation")
        != "PROMISING_DIRECTION_BUT_DESIGN_VALIDITY_NOT_ESTABLISHED"
    ):
        raise ProvenanceValidationError("Original E05a claim disposition changed.")

    run_dir = (
        root
        / "e05a_semantic_protocol_lock"
        / PINNED_ORIGINAL_E05A_RUN_ID
    )
    run = validate_run_manifest(
        run_dir,
        root,
        requirements=ManifestValidationRequirements(
            expected_experiment_id="e05a_semantic_protocol_lock",
            expected_run_mode="main",
            require_main_eligible=True,
            require_full_eligible=True,
        ),
    )
    if (
        run.manifest_sha256 != PINNED_ORIGINAL_E05A_MANIFEST_SHA256
        or run.report_sha256 != PINNED_ORIGINAL_E05A_REPORT_SHA256
    ):
        raise ProvenanceValidationError("Original E05a run hash changed.")
    if (
        run.report.get("execution_status") != "PASS"
        or run.report.get("e05a_design_status") != "NO_GO"
        or run.report.get("full_h5_lite_claim_open") is not False
        or run.report.get("namespace", {}).get("e05b_registry_generated") is not False
    ):
        raise ProvenanceValidationError("Original E05a report disposition changed.")
    return run


def original_e05a_dependency_record(run: ValidatedRun) -> dict[str, Any]:
    record = dict(run.dependency_record())
    record.update(
        {
            "evidence_role": "immutable_no_go_design_history",
            "artifact_freeze_sha256": PINNED_ORIGINAL_E05A_FREEZE_SHA256,
            "claim_status_sha256": PINNED_ORIGINAL_E05A_CLAIM_SHA256,
            "e05a_original_status": "NO_GO",
            "h5_claim_open": False,
            "rows_reused_in_r1_inference": 0,
            "scientific_evidence": False,
        }
    )
    return record


def validate_r1_config_path(config_path: str | Path) -> None:
    resolved = Path(config_path).resolve(strict=True)
    if resolved != R1_CONFIG_PATH.resolve(strict=True):
        raise ProvenanceValidationError(
            "E05a-R1 requires the frozen default config path."
        )


def require_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProvenanceValidationError(f"{name} must be a mapping.")
    return value
