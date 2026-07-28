from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from catena.core.io import file_sha256

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def _contained_path(root: Path, relative_path: str, *, label: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute():
        raise ValueError(f"{label} must be relative to {root}: {relative}")
    resolved_root = root.resolve()
    resolved = (resolved_root / relative).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"{label} escapes {resolved_root}: {relative}")
    return resolved


def _validate_sha256(value: object, *, label: str) -> str:
    digest = str(value)
    if not _SHA256_PATTERN.fullmatch(digest):
        raise ValueError(f"{label} is not a lowercase SHA-256 digest: {digest!r}")
    return digest


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


def resolve_report(artifact_root: str | Path, experiment_id: str) -> tuple[Path, dict[str, Any]]:
    """Resolve a legacy ``latest.json`` pointer without allowing path escape."""

    experiment_root = (Path(artifact_root).resolve() / experiment_id).resolve()
    pointer = experiment_root / "latest.json"
    if not pointer.exists():
        raise FileNotFoundError(f"Missing latest pointer: {pointer}")
    run_reference = Path(str(_load_json(pointer)["run_dir"]))
    run_dir = (
        run_reference.resolve()
        if run_reference.is_absolute()
        else (experiment_root / run_reference).resolve()
    )
    if not run_dir.is_relative_to(experiment_root):
        raise ValueError(f"Latest pointer escapes experiment root: {run_reference}")
    report_path = run_dir / "report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"Missing report: {report_path}")
    return report_path, _load_json(report_path)


def validate_evidence_contract(evidence_contract: dict[str, Any]) -> None:
    if not evidence_contract:
        raise ValueError("E16 evidence contract must contain at least one record")
    for claim_name, raw_specification in evidence_contract.items():
        if not isinstance(raw_specification, dict):
            raise TypeError(f"E16 record {claim_name!r} must be a mapping")
        specification = dict(raw_specification)
        for required in ("experiment_id", "run_id", "claim_disposition", "files"):
            if required not in specification:
                raise KeyError(f"E16 record {claim_name!r} is missing {required!r}")
        if not str(specification["experiment_id"]).strip():
            raise ValueError(f"E16 record {claim_name!r} has an empty experiment_id")
        run_id = str(specification["run_id"])
        if not run_id.strip() or Path(run_id).name != run_id:
            raise ValueError(f"E16 record {claim_name!r} has an invalid run_id: {run_id!r}")
        if not str(specification["claim_disposition"]).strip():
            raise ValueError(f"E16 record {claim_name!r} has an empty claim disposition")
        files = specification["files"]
        if (
            not isinstance(files, dict)
            or "report.json" not in files
            or "run_manifest.json" not in files
        ):
            raise ValueError(
                f"E16 record {claim_name!r} must pin report.json and run_manifest.json"
            )
        for relative_path, digest in files.items():
            if Path(str(relative_path)).is_absolute():
                raise ValueError(
                    f"E16 record {claim_name!r} file path must be relative: {relative_path}"
                )
            _validate_sha256(digest, label=f"{claim_name}.{relative_path}")
        anchors = specification.get("anchors", [])
        if not isinstance(anchors, list):
            raise TypeError(f"E16 record {claim_name!r} anchors must be a list")
        for index, anchor in enumerate(anchors):
            if not isinstance(anchor, dict) or "path" not in anchor or "sha256" not in anchor:
                raise ValueError(f"E16 record {claim_name!r} anchor {index} is incomplete")
            _validate_sha256(
                anchor["sha256"],
                label=f"{claim_name}.anchors[{index}].sha256",
            )


def _freeze_record(
    *,
    artifact_root: Path,
    claim_name: str,
    specification: dict[str, Any],
) -> dict[str, Any]:
    experiment_id = str(specification["experiment_id"])
    run_id = str(specification["run_id"])
    run_root = _contained_path(
        artifact_root,
        f"{experiment_id}/{run_id}",
        label=f"{claim_name}.run",
    )
    frozen_files: dict[str, dict[str, Any]] = {}
    loaded_report: dict[str, Any] | None = None
    for relative_path, expected_digest_value in dict(specification["files"]).items():
        expected_digest = _validate_sha256(
            expected_digest_value,
            label=f"{claim_name}.{relative_path}",
        )
        path = _contained_path(
            run_root,
            str(relative_path),
            label=f"{claim_name}.{relative_path}",
        )
        if not path.is_file():
            raise FileNotFoundError(f"Missing pinned evidence file: {path}")
        observed_digest = file_sha256(path)
        if observed_digest != expected_digest:
            raise ValueError(
                f"Pinned evidence hash mismatch for {path}: "
                f"expected={expected_digest}, observed={observed_digest}"
            )
        frozen_files[str(relative_path)] = {
            "path": str(path),
            "sha256": observed_digest,
        }
        if str(relative_path) == "report.json":
            loaded_report = _load_json(path)
    if loaded_report is None:
        raise ValueError(f"E16 record {claim_name!r} did not load report.json")
    expected_report_fields = specification.get("expected_report_fields", {})
    if not isinstance(expected_report_fields, dict):
        raise TypeError(f"E16 record {claim_name!r} expected_report_fields must be a mapping")
    _validate_expected_fields(
        loaded_report,
        dict(expected_report_fields),
        label=f"{claim_name}.report",
    )

    frozen_anchors: list[dict[str, Any]] = []
    for index, raw_anchor in enumerate(specification.get("anchors", [])):
        anchor = dict(raw_anchor)
        path = _contained_path(
            artifact_root,
            str(anchor["path"]),
            label=f"{claim_name}.anchors[{index}]",
        )
        if not path.is_file():
            raise FileNotFoundError(f"Missing pinned evidence anchor: {path}")
        expected_digest = _validate_sha256(
            anchor["sha256"],
            label=f"{claim_name}.anchors[{index}].sha256",
        )
        observed_digest = file_sha256(path)
        if observed_digest != expected_digest:
            raise ValueError(
                f"Pinned anchor hash mismatch for {path}: "
                f"expected={expected_digest}, observed={observed_digest}"
            )
        anchor_payload = _load_json(path)
        expected_fields = anchor.get("expected_fields", {})
        if not isinstance(expected_fields, dict):
            raise TypeError(
                f"E16 record {claim_name!r} anchor {index} expected_fields must be a mapping"
            )
        _validate_expected_fields(
            anchor_payload,
            dict(expected_fields),
            label=f"{claim_name}.anchor[{index}]",
        )
        frozen_anchors.append({"path": str(path), "sha256": observed_digest})

    execution_status = loaded_report.get(
        "execution_status",
        loaded_report.get("status", "UNKNOWN"),
    )
    return {
        "valid": True,
        "experiment_id": experiment_id,
        "run_id": run_id,
        "run_dir": str(run_root),
        "execution_status": execution_status,
        "claim_disposition": str(specification["claim_disposition"]),
        "evidence_tier": str(specification.get("evidence_tier", "CONTROLLED_REFERENCE")),
        "scientific_evidence": bool(loaded_report.get("scientific_evidence", False)),
        "files": frozen_files,
        "anchors": frozen_anchors,
        "claim_gate": loaded_report.get("claim_gate"),
    }


def freeze_evidence(
    *,
    artifact_root: str | Path,
    evidence_contract: dict[str, Any],
    validate_only: bool = False,
) -> dict[str, Any]:
    validate_evidence_contract(evidence_contract)
    root = Path(artifact_root).resolve()
    registry: dict[str, Any] = {
        "artifact_root": str(root),
        "mode": "CONTRACT_VALIDATION_ONLY" if validate_only else "PINNED_EVIDENCE_FREEZE",
        "evidence": {},
    }
    if validate_only:
        for claim_name, specification in evidence_contract.items():
            registry["evidence"][claim_name] = {
                "valid": False,
                "validation_status": "DRY_RUN_NOT_RESOLVED",
                "experiment_id": str(specification["experiment_id"]),
                "run_id": str(specification["run_id"]),
                "claim_disposition": str(specification["claim_disposition"]),
            }
        registry["core_registry_complete"] = False
        return registry

    for claim_name, raw_specification in evidence_contract.items():
        try:
            registry["evidence"][claim_name] = _freeze_record(
                artifact_root=root,
                claim_name=claim_name,
                specification=dict(raw_specification),
            )
        except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
            registry["evidence"][claim_name] = {
                "valid": False,
                "claim_disposition": str(
                    dict(raw_specification).get("claim_disposition", "UNKNOWN")
                ),
                "error": f"{type(error).__name__}: {error}",
            }
    registry["core_registry_complete"] = all(
        bool(item.get("valid", False)) for item in registry["evidence"].values()
    )
    return registry
