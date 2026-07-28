from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from catena.core.io import file_sha256
from catena.eval.evidence_freeze import (
    freeze_evidence,
    validate_evidence_contract,
)

SCOPE_FLAGS = (
    "controlled_claim_eligible",
    "structured_sequence_claim_eligible",
    "official_operator_claim_eligible",
    "language_model_claim_eligible",
    "agent_claim_eligible",
)

REQUIRED_RECORDS: dict[str, dict[str, Any]] = {
    "e10_original": {
        "experiment_id": "e10_learned_rank_scaling",
        "run_id": "20260727T184326.484361Z",
        "record_role": "ORIGINAL",
        "claim_disposition": "NOT_OPENED",
        "scope_flags": (False, False, False, False, False),
        "freeze_anchor": "E10_POSTCORE_ARTIFACT_FREEZE_V1.json",
    },
    "e10b_prospective_repair": {
        "experiment_id": "e10b_floor_aware_rank_scaling",
        "run_id": "20260727T190906.272784Z",
        "record_role": "PROSPECTIVE_REPAIR",
        "claim_disposition": "SUPPORTED",
        "scope_flags": (True, False, False, False, False),
        "freeze_anchor": "E10B_POSTCORE_ARTIFACT_FREEZE_V1.json",
    },
    "e11_original": {
        "experiment_id": "e11_representation_control_coadaptation",
        "run_id": "20260727T180703.763554Z",
        "record_role": "ORIGINAL",
        "claim_disposition": "NOT_OPENED_SCALE_RESTRICTION",
        "scope_flags": (False, False, False, False, False),
        "freeze_anchor": "E11_POSTCORE_ARTIFACT_FREEZE_V1.json",
    },
    "e11b_prospective_repair": {
        "experiment_id": "e11b_scale_normalized_coadaptation",
        "run_id": "20260727T183004.928280Z",
        "record_role": "PROSPECTIVE_REPAIR",
        "claim_disposition": "SUPPORTED",
        "scope_flags": (True, False, False, False, False),
        "freeze_anchor": "E11_POSTCORE_ARTIFACT_FREEZE_V1.json",
    },
    "e12_canonical": {
        "experiment_id": "e12_control_algebra_lattice",
        "run_id": "20260727T184511.437394Z",
        "record_role": "CANONICAL_ARTIFACT_COMPLETE",
        "claim_disposition": "SUPPORTED",
        "scope_flags": (True, False, False, False, False),
        "freeze_anchor": "E12_POSTCORE_ARTIFACT_FREEZE_V1.json",
    },
    "e13a_original": {
        "experiment_id": "e13a_sequence_floor_throughput",
        "run_id": "20260727T180703.836996Z",
        "record_role": "ORIGINAL_CALIBRATION_PILOT",
        "claim_disposition": "CALIBRATION_PILOT_ONLY",
        "scope_flags": (False, False, False, False, False),
        "freeze_anchor": "E13A_R1_RESULT_STATUS_AMENDMENT_FREEZE_V1.json",
        "repo_anchor": "docs/E13A_R1_SEQUENCE_CALIBRATION_LOCK.json",
    },
    "e13a_r1": {
        "experiment_id": "e13a_r1_sequence_floor_throughput",
        "run_id": "20260727T183609.755945Z",
        "record_role": "PROSPECTIVE_R1_LEGACY_PIPELINE",
        "claim_disposition": "GO_FOR_ORIGINAL_HARD_MASKED_PIPELINE_ONLY",
        "scope_flags": (False, False, False, False, False),
        "freeze_anchor": "E13A_R1_POSTCORE_ARTIFACT_FREEZE_V1.json",
    },
    "e13a_r2": {
        "experiment_id": "e13a_r2_sequence_floor_throughput",
        "run_id": "20260727T190642.222102Z",
        "record_role": "PROSPECTIVE_R2_REPAIRED_DEPENDENCY",
        "claim_disposition": "GO_FOR_E13B_R1_CALIBRATION_ONLY",
        "scope_flags": (True, False, False, False, False),
        "freeze_anchor": "E13A_R2_POSTCORE_ARTIFACT_FREEZE_V1.json",
    },
    "e13c_r1": {
        "experiment_id": "e13c_r1_transactional_sequence_aggregate",
        "run_id": "20260727T214126.954177Z",
        "record_role": "CANONICAL_SEQUENCE_AGGREGATE",
        "claim_disposition": "SUPPORTED",
        "scope_flags": (True, True, False, False, False),
        "freeze_anchor": "E13BC_R1_POSTCORE_ARTIFACT_FREEZE_V1.json",
    },
    "e14_canonical": {
        "experiment_id": "e14_plan_continuation",
        "run_id": "20260727T214143.455051Z",
        "record_role": "CANONICAL_STRUCTURED_PROXY",
        "claim_disposition": "SUPPORTED_STRUCTURED_PROXY_ONLY",
        "scope_flags": (True, False, False, False, False),
        "freeze_anchor": "E14_POSTCORE_ARTIFACT_FREEZE_V1.json",
    },
    "e15_canonical_dry_gate": {
        "experiment_id": "e15_official_backend_gate",
        "run_id": "20260727T184517.578907Z",
        "record_role": "CANONICAL_DRY_GATE",
        "claim_disposition": "NOT_CONFIGURED_DRY_GATE",
        "scope_flags": (False, False, False, False, False),
        "freeze_anchor": "E15_DRY_GATE_ARTIFACT_FREEZE_V1.json",
    },
}

DISPOSITION_GROUPS = {
    "e10_rank_scaling": {
        "original": "e10_original",
        "prospective_repair": "e10b_prospective_repair",
    },
    "e11_representation_control": {
        "original": "e11_original",
        "prospective_repair": "e11b_prospective_repair",
    },
    "e13a_sequence_calibration": {
        "original": "e13a_original",
        "prospective_r1": "e13a_r1",
        "prospective_r2": "e13a_r2",
    },
}

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def _nested_value(payload: dict[str, Any], dotted_path: str) -> Any:
    value: Any = payload
    for component in dotted_path.split("."):
        if not isinstance(value, dict) or component not in value:
            raise KeyError(f"Missing expected field {dotted_path!r}")
        value = value[component]
    return value


def _validate_expected_fields(
    payload: dict[str, Any],
    expected_fields: dict[str, Any],
    *,
    label: str,
) -> None:
    for dotted_path, expected in expected_fields.items():
        observed = _nested_value(payload, str(dotted_path))
        if observed != expected:
            raise ValueError(
                f"{label} field {dotted_path!r} mismatch: "
                f"expected={expected!r}, observed={observed!r}"
            )


def _validate_digest(value: object, *, label: str) -> str:
    digest = str(value)
    if not _SHA256_PATTERN.fullmatch(digest):
        raise ValueError(f"{label} is not a lowercase SHA-256 digest: {digest!r}")
    return digest


def _contained_repo_path(repo_root: Path, relative_path: str, *, label: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute():
        raise ValueError(f"{label} must be repository-relative: {relative}")
    resolved_root = repo_root.resolve()
    resolved = (resolved_root / relative).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"{label} escapes repository root: {relative}")
    return resolved


def _scope_mapping(values: tuple[bool, ...]) -> dict[str, bool]:
    return dict(zip(SCOPE_FLAGS, values, strict=True))


def validate_postcore_evidence_contract(
    evidence_contract: dict[str, Any],
) -> None:
    validate_evidence_contract(evidence_contract)
    if set(evidence_contract) != set(REQUIRED_RECORDS):
        raise ValueError(
            "E17 record set mismatch: "
            f"expected={sorted(REQUIRED_RECORDS)}, "
            f"observed={sorted(evidence_contract)}"
        )
    for name, protocol in REQUIRED_RECORDS.items():
        specification = evidence_contract[name]
        if not isinstance(specification, dict):
            raise TypeError(f"E17 record {name!r} must be a mapping")
        expected_identity = (
            protocol["experiment_id"],
            protocol["run_id"],
        )
        observed_identity = (
            specification.get("experiment_id"),
            specification.get("run_id"),
        )
        if observed_identity != expected_identity:
            raise ValueError(
                f"E17 record {name!r} identity mismatch: "
                f"expected={expected_identity!r}, observed={observed_identity!r}"
            )
        expected_path = "/".join(expected_identity)
        if specification.get("exact_run_path") != expected_path:
            raise ValueError(
                f"E17 record {name!r} exact_run_path must be {expected_path!r}"
            )
        if specification.get("record_role") != protocol["record_role"]:
            raise ValueError(
                f"E17 record {name!r} role mismatch: "
                f"expected={protocol['record_role']!r}, "
                f"observed={specification.get('record_role')!r}"
            )
        if (
            specification.get("claim_disposition")
            != protocol["claim_disposition"]
        ):
            raise ValueError(
                f"E17 record {name!r} claim disposition mismatch: "
                f"expected={protocol['claim_disposition']!r}, "
                f"observed={specification.get('claim_disposition')!r}"
            )
        scopes = specification.get("scope_flags")
        expected_scopes = _scope_mapping(protocol["scope_flags"])
        if not isinstance(scopes, dict) or scopes != expected_scopes:
            raise ValueError(
                f"E17 record {name!r} scope mismatch: "
                f"expected={expected_scopes!r}, observed={scopes!r}"
            )
        manifest_fields = specification.get("expected_manifest_fields")
        if not isinstance(manifest_fields, dict) or not manifest_fields:
            raise ValueError(
                f"E17 record {name!r} must pin expected_manifest_fields"
            )
        anchors = specification.get("anchors", [])
        if protocol["freeze_anchor"] not in {
            str(anchor.get("path"))
            for anchor in anchors
            if isinstance(anchor, dict)
        }:
            raise ValueError(
                f"E17 record {name!r} must pin freeze "
                f"{protocol['freeze_anchor']!r}"
            )
        repo_anchors = specification.get("repo_anchors", [])
        if not isinstance(repo_anchors, list):
            raise TypeError(f"E17 record {name!r} repo_anchors must be a list")
        required_repo_anchor = protocol.get("repo_anchor")
        if required_repo_anchor is not None and required_repo_anchor not in {
            str(anchor.get("path"))
            for anchor in repo_anchors
            if isinstance(anchor, dict)
        }:
            raise ValueError(
                f"E17 record {name!r} must pin repository lock "
                f"{required_repo_anchor!r}"
            )
        for index, anchor in enumerate(repo_anchors):
            if (
                not isinstance(anchor, dict)
                or "path" not in anchor
                or "sha256" not in anchor
            ):
                raise ValueError(
                    f"E17 record {name!r} repo anchor {index} is incomplete"
                )
            _validate_digest(
                anchor["sha256"],
                label=f"{name}.repo_anchors[{index}].sha256",
            )


def _validate_repo_anchors(
    *,
    repo_root: Path,
    record_name: str,
    raw_anchors: list[dict[str, Any]],
) -> list[dict[str, str]]:
    frozen: list[dict[str, str]] = []
    for index, anchor in enumerate(raw_anchors):
        path = _contained_repo_path(
            repo_root,
            str(anchor["path"]),
            label=f"{record_name}.repo_anchors[{index}]",
        )
        if not path.is_file():
            raise FileNotFoundError(f"Missing pinned repository anchor: {path}")
        expected_digest = _validate_digest(
            anchor["sha256"],
            label=f"{record_name}.repo_anchors[{index}].sha256",
        )
        observed_digest = file_sha256(path)
        if observed_digest != expected_digest:
            raise ValueError(
                f"Pinned repository anchor hash mismatch for {path}: "
                f"expected={expected_digest}, observed={observed_digest}"
            )
        expected_fields = anchor.get("expected_fields", {})
        if not isinstance(expected_fields, dict):
            raise TypeError(
                f"{record_name}.repo_anchors[{index}].expected_fields "
                "must be a mapping"
            )
        _validate_expected_fields(
            _load_json(path),
            expected_fields,
            label=f"{record_name}.repo_anchor[{index}]",
        )
        frozen.append({"path": str(path), "sha256": observed_digest})
    return frozen


def freeze_postcore_evidence(
    *,
    repo_root: str | Path,
    artifact_root: str | Path,
    evidence_contract: dict[str, Any],
    dry_run: bool = False,
) -> dict[str, Any]:
    """Validate and pin the exact post-core evidence chain.

    Unlike E16's schema-only ``validate_only`` mode, E17 dry-run resolves every
    exact run and verifies every declared digest. It differs from a main freeze
    only in the registry mode and the explicit ``canonical_freeze_written``
    marker.
    """

    validate_postcore_evidence_contract(evidence_contract)
    registry = freeze_evidence(
        artifact_root=artifact_root,
        evidence_contract=evidence_contract,
        validate_only=False,
    )
    resolved_repo = Path(repo_root).resolve()
    for name, raw_specification in evidence_contract.items():
        specification = dict(raw_specification)
        item = registry["evidence"][name]
        item["record_role"] = specification["record_role"]
        item["exact_run_path"] = specification["exact_run_path"]
        item["scope_flags"] = dict(specification["scope_flags"])
        if not item.get("valid", False):
            continue
        try:
            manifest_path = Path(
                item["files"]["run_manifest.json"]["path"]
            )
            _validate_expected_fields(
                _load_json(manifest_path),
                dict(specification["expected_manifest_fields"]),
                label=f"{name}.manifest",
            )
            item["repo_anchors"] = _validate_repo_anchors(
                repo_root=resolved_repo,
                record_name=name,
                raw_anchors=[
                    dict(anchor)
                    for anchor in specification.get("repo_anchors", [])
                ],
            )
        except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
            item["valid"] = False
            item["error"] = f"{type(error).__name__}: {error}"

    complete = all(
        bool(item.get("valid", False))
        for item in registry["evidence"].values()
    )
    registry["mode"] = (
        "VALIDATED_DRY_RUN_NO_CANONICAL_FREEZE"
        if dry_run
        else "PINNED_POSTCORE_EVIDENCE_FREEZE"
    )
    registry["canonical_freeze_written"] = bool(not dry_run and complete)
    registry["postcore_registry_complete"] = complete
    registry.pop("core_registry_complete", None)
    registry["scope_index"] = {
        flag: sorted(
            name
            for name, item in registry["evidence"].items()
            if item.get("scope_flags", {}).get(flag) is True
        )
        for flag in SCOPE_FLAGS
    }
    registry["disposition_groups"] = {
        group: {
            role: {
                "record": record,
                "claim_disposition": registry["evidence"][record].get(
                    "claim_disposition"
                ),
                "valid": registry["evidence"][record].get("valid", False),
            }
            for role, record in members.items()
        }
        for group, members in DISPOSITION_GROUPS.items()
    }
    return registry
