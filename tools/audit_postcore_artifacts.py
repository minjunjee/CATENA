from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CHECKPOINT_MANIFEST_RE = re.compile(r"^checkpoint_manifest_(\d+)$")

CheckStatus = Literal["PASS", "FAIL", "WARN", "SKIP"]
ExperimentStatus = Literal["PASS", "FAIL", "WARN", "NOT_COMPLETE"]


@dataclass(frozen=True)
class FlatFreezeSpec:
    name: str
    freeze_name: str
    result_doc: str
    aliases: dict[str, str]
    document_tokens: tuple[str, ...]
    checkpoint_metrics: str | None = None
    checkpoint_scheme: Literal["path_nul_hash", "legacy_sha_path"] = "path_nul_hash"


@dataclass
class AuditCheck:
    name: str
    status: CheckStatus
    detail: str
    path: str | None = None
    expected: Any = None
    observed: Any = None


@dataclass
class ExperimentAudit:
    name: str
    freeze_path: str | None
    result_doc: str | None
    execution_status: str | None = None
    claim_status: str | None = None
    status: ExperimentStatus = "PASS"
    checks: list[AuditCheck] = field(default_factory=list)

    def add(
        self,
        name: str,
        status: CheckStatus,
        detail: str,
        *,
        path: Path | None = None,
        expected: Any = None,
        observed: Any = None,
    ) -> None:
        self.checks.append(
            AuditCheck(
                name=name,
                status=status,
                detail=detail,
                path=None if path is None else str(path),
                expected=expected,
                observed=observed,
            )
        )
        if status == "FAIL":
            self.status = "FAIL"
        elif status == "WARN" and self.status == "PASS":
            self.status = "WARN"

    def not_complete(self, detail: str, *, path: Path) -> None:
        self.status = "NOT_COMPLETE"
        self.add("availability", "SKIP", detail, path=path)


FLAT_SPECS = (
    FlatFreezeSpec(
        name="E10",
        freeze_name="E10_POSTCORE_ARTIFACT_FREEZE_V1.json",
        result_doc="docs/E10_LEARNED_RANK_SCALING_RESULT_KO.md",
        aliases={
            "protocol_lock_v2": "docs/E10_LEARNED_RANK_SCALING_LOCK_V2.json",
            "result_markdown": "docs/E10_LEARNED_RANK_SCALING_RESULT_KO.md",
        },
        document_tokens=(
            "execution_status: PASS",
            "full_e10_claim_open: false",
            "20260727T184326.484361Z",
        ),
        checkpoint_metrics="rank_scaling_metrics.jsonl",
    ),
    FlatFreezeSpec(
        name="E10b",
        freeze_name="E10B_POSTCORE_ARTIFACT_FREEZE_V1.json",
        result_doc="docs/E10B_FLOOR_AWARE_RANK_SCALING_RESULT_KO.md",
        aliases={
            "protocol_lock": "docs/E10B_FLOOR_AWARE_RANK_SCALING_LOCK.json",
            "result_markdown": "docs/E10B_FLOOR_AWARE_RANK_SCALING_RESULT_KO.md",
        },
        document_tokens=(
            "execution_status: PASS",
            "original_e10_claim_status: NOT_OPENED",
            "prospective_e10b_status: SUPPORTED",
            "20260727T190906.272784Z",
        ),
    ),
    FlatFreezeSpec(
        name="E12",
        freeze_name="E12_POSTCORE_ARTIFACT_FREEZE_V1.json",
        result_doc="docs/E12_CONTROL_LATTICE_RESULT_KO.md",
        aliases={
            "artifact_completion_lock": "docs/E12_ARTIFACT_COMPLETION_AMENDMENT_LOCK.json",
            "result_markdown": "docs/E12_CONTROL_LATTICE_RESULT_KO.md",
        },
        document_tokens=(
            "execution_status: PASS",
            "architecture_demand_lattice_status: SUPPORTED",
            "20260727T184511.437394Z",
        ),
        checkpoint_metrics="control_lattice_metrics.jsonl",
    ),
    FlatFreezeSpec(
        name="E13a-R1",
        freeze_name="E13A_R1_POSTCORE_ARTIFACT_FREEZE_V1.json",
        result_doc="docs/E13A_SEQUENCE_CALIBRATION_RESULT_KO.md",
        aliases={
            "protocol_lock": "docs/E13A_R1_SEQUENCE_CALIBRATION_LOCK.json",
            "result_markdown": "docs/E13A_SEQUENCE_CALIBRATION_RESULT_KO.md",
        },
        document_tokens=(
            "20260727T183609.755945Z",
            "PASS / GO_FOR_E13B",
        ),
        checkpoint_metrics="sequence_calibration_repair_metrics.jsonl",
    ),
    FlatFreezeSpec(
        name="E13a-R2",
        freeze_name="E13A_R2_POSTCORE_ARTIFACT_FREEZE_V1.json",
        result_doc="docs/E13A_R2_LEARNED_DISTRACTOR_RESULT_KO.md",
        aliases={
            "protocol_lock": "docs/E13A_R2_LEARNED_DISTRACTOR_LOCK.json",
            "e13b_main_lock": "docs/E13B_R1_MAIN_LOCK.json",
            "result_markdown": "docs/E13A_R2_LEARNED_DISTRACTOR_RESULT_KO.md",
        },
        document_tokens=(
            "execution_status: PASS",
            "e13a_r2_calibration_status: GO_FOR_E13B_R1",
            "e13a_r1_repaired_dependency_eligible: false",
            "20260727T190642.222102Z",
        ),
        checkpoint_metrics="sequence_calibration_repair_metrics.jsonl",
    ),
    FlatFreezeSpec(
        name="E15",
        freeze_name="E15_DRY_GATE_ARTIFACT_FREEZE_V1.json",
        result_doc="docs/E15_OFFICIAL_BACKEND_GATE_RESULT_KO.md",
        aliases={
            "result_markdown": "docs/E15_OFFICIAL_BACKEND_GATE_RESULT_KO.md",
        },
        document_tokens=(
            "execution_status: DRY_RUN",
            "official_backend_ready: false",
            "reference_fallback_used: false",
            "20260727T184517.578907Z",
        ),
    ),
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_manifest_sha256(
    run_dir: Path,
    checkpoint_paths: list[Path],
    *,
    scheme: Literal["path_nul_hash", "legacy_sha_path"],
) -> str:
    digest = hashlib.sha256()
    resolved_run = run_dir.resolve()
    ordered_paths = sorted(
        checkpoint_paths,
        key=lambda item: item.resolve().relative_to(resolved_run).as_posix(),
    )
    for path in ordered_paths:
        resolved = path.resolve()
        relative = resolved.relative_to(resolved_run).as_posix()
        checkpoint_sha = file_sha256(resolved)
        if scheme == "path_nul_hash":
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(checkpoint_sha.encode("ascii"))
        elif scheme == "legacy_sha_path":
            # E11b was frozen from:
            # sha256sum ... | sed "s#  $run/##" | sha256sum
            digest.update(checkpoint_sha.encode("ascii"))
            digest.update(relative.encode("utf-8"))
        else:  # pragma: no cover - Literal prevents normal callers reaching this.
            raise ValueError(f"Unsupported checkpoint manifest scheme: {scheme}")
        digest.update(b"\n")
    return digest.hexdigest()


def _json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _jsonl_objects(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"Expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _execution_status(report: dict[str, Any]) -> str | None:
    value = report.get("execution_status", report.get("status"))
    return None if value is None else str(value)


def _claim_gate(report: dict[str, Any]) -> dict[str, Any]:
    value = report.get("claim_gate")
    return value if isinstance(value, dict) else {}


def _check_file_hash(
    audit: ExperimentAudit,
    *,
    name: str,
    path: Path,
    expected: object,
) -> None:
    if not isinstance(expected, str) or not SHA256_RE.fullmatch(expected):
        audit.add(
            name,
            "FAIL",
            "Declared digest is not a lowercase SHA-256 value.",
            path=path,
            expected="64 lowercase hexadecimal characters",
            observed=expected,
        )
        return
    if not path.is_file():
        audit.add(
            name,
            "FAIL",
            "Declared file does not exist.",
            path=path,
            expected=expected,
            observed=None,
        )
        return
    observed = file_sha256(path)
    audit.add(
        name,
        "PASS" if observed == expected else "FAIL",
        "Declared file hash matches." if observed == expected else "Declared file hash mismatch.",
        path=path,
        expected=expected,
        observed=observed,
    )


def _check_document_tokens(
    audit: ExperimentAudit,
    document: Path,
    tokens: tuple[str, ...],
) -> None:
    if not document.is_file():
        audit.add("result_document", "FAIL", "Result document is missing.", path=document)
        return
    text = document.read_text(encoding="utf-8")
    for token in tokens:
        audit.add(
            f"document_token:{token}",
            "PASS" if token in text else "FAIL",
            (
                "Result document contains the frozen status/path token."
                if token in text
                else "Result document is inconsistent with the frozen status/path token."
            ),
            path=document,
            expected=token,
            observed=token if token in text else None,
        )


def _check_run_identity(
    audit: ExperimentAudit,
    *,
    artifact_root: Path,
    freeze: dict[str, Any],
    run_dir: Path,
    report: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    experiment_id = freeze.get("experiment_id")
    run_id = freeze.get("run_id")
    expected_root = artifact_root / str(experiment_id)
    audit.add(
        "run_dir_containment",
        "PASS" if _is_within(run_dir, expected_root) else "FAIL",
        "Frozen run directory is contained by its experiment root.",
        path=run_dir,
        expected=str(expected_root.resolve()),
        observed=str(run_dir.resolve()),
    )
    audit.add(
        "run_id",
        "PASS" if isinstance(run_id, str) and run_dir.name == run_id else "FAIL",
        "Frozen run ID matches the run-directory basename.",
        path=run_dir,
        expected=run_id,
        observed=run_dir.name,
    )
    manifest_experiment = manifest.get("experiment_id")
    audit.add(
        "manifest_experiment_id",
        "PASS" if manifest_experiment == experiment_id else "FAIL",
        "Run manifest experiment ID matches the freeze.",
        path=run_dir / "run_manifest.json",
        expected=experiment_id,
        observed=manifest_experiment,
    )
    manifest_run = manifest.get("run_id")
    if manifest_run is not None:
        audit.add(
            "manifest_run_id",
            "PASS" if manifest_run == run_id else "FAIL",
            "Run manifest run ID matches the freeze.",
            path=run_dir / "run_manifest.json",
            expected=run_id,
            observed=manifest_run,
        )
    expected_execution = freeze.get("execution_status")
    observed_execution = _execution_status(report)
    audit.execution_status = None if expected_execution is None else str(expected_execution)
    audit.add(
        "execution_status",
        "PASS" if observed_execution == expected_execution else "FAIL",
        "Execution status is checked independently of claim disposition.",
        path=run_dir / "report.json",
        expected=expected_execution,
        observed=observed_execution,
    )


def _check_flat_claim_status(
    audit: ExperimentAudit,
    freeze: dict[str, Any],
    report: dict[str, Any],
) -> None:
    if "claim_status" in freeze:
        disposition = str(freeze["claim_status"])
        audit.claim_status = disposition
        report_supported = _claim_gate(report).get("supported")
        expected_supported: bool | None = None
        if disposition == "SUPPORTED":
            expected_supported = True
        elif disposition in {"NOT_OPENED", "INCONCLUSIVE", "NO_GO"}:
            expected_supported = False
        if expected_supported is not None:
            audit.add(
                "claim_disposition",
                "PASS" if report_supported is expected_supported else "FAIL",
                "Claim disposition is evaluated separately from execution success.",
                path=Path(str(freeze["run_dir"])) / "report.json",
                expected={"claim_status": disposition, "report_supported": expected_supported},
                observed={"report_supported": report_supported},
            )
        return
    if "calibration_status" in freeze:
        audit.claim_status = str(freeze["calibration_status"])
        audit.add(
            "claim_disposition",
            "PASS",
            "Calibration disposition is preserved separately from report execution status.",
            expected=freeze["calibration_status"],
            observed=freeze["calibration_status"],
        )
        if "sequence_claim_status" in freeze:
            audit.add(
                "sequence_claim_status",
                "PASS",
                "Sequence claim remains separate from calibration GO status.",
                expected=freeze["sequence_claim_status"],
                observed=freeze["sequence_claim_status"],
            )
        return
    if freeze.get("experiment_id") == "e15_official_backend_gate":
        ready = bool(freeze.get("official_backend_ready"))
        audit.claim_status = "OFFICIAL_READY" if ready else "NOT_CONFIGURED"
        observed_ready = _claim_gate(report).get("official_backend_ready")
        audit.add(
            "official_backend_claim",
            "PASS" if observed_ready is ready else "FAIL",
            "Official-backend readiness is independent of dry-run execution.",
            expected=ready,
            observed=observed_ready,
        )


def _check_checkpoint_rows(
    audit: ExperimentAudit,
    *,
    run_dir: Path,
    metrics_path: Path,
    checkpoint_paths: list[Path],
) -> None:
    try:
        rows = _jsonl_objects(metrics_path)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        audit.add(
            "checkpoint_metric_rows",
            "FAIL",
            f"Could not read checkpoint metric rows: {type(error).__name__}: {error}",
            path=metrics_path,
        )
        return
    actual_by_path = {path.resolve(): file_sha256(path) for path in checkpoint_paths}
    declared_paths: set[Path] = set()
    rows_with_checkpoint = 0
    rows_with_declared_hash = 0
    invalid_rows = 0
    for row in rows:
        raw_path = row.get("checkpoint")
        if not isinstance(raw_path, str):
            continue
        rows_with_checkpoint += 1
        path = Path(raw_path).resolve()
        declared_paths.add(path)
        if not _is_within(path, run_dir) or path not in actual_by_path:
            invalid_rows += 1
            continue
        expected_hash = row.get("checkpoint_sha256")
        if expected_hash is not None:
            rows_with_declared_hash += 1
            if expected_hash != actual_by_path[path]:
                invalid_rows += 1
    coverage_ok = declared_paths == set(actual_by_path)
    audit.add(
        "checkpoint_metric_coverage",
        "PASS" if coverage_ok and rows_with_checkpoint > 0 else "FAIL",
        "Metric rows cover exactly the checkpoint set.",
        path=metrics_path,
        expected={"unique_checkpoints": len(actual_by_path)},
        observed={
            "rows_with_checkpoint": rows_with_checkpoint,
            "unique_checkpoints": len(declared_paths),
        },
    )
    audit.add(
        "checkpoint_metric_hashes",
        "PASS" if invalid_rows == 0 else "FAIL",
        (
            "All checkpoint hashes declared by metric rows match."
            if invalid_rows == 0
            else "One or more metric checkpoint paths/hashes are invalid."
        ),
        path=metrics_path,
        expected={"invalid_rows": 0},
        observed={
            "invalid_rows": invalid_rows,
            "rows_with_declared_hash": rows_with_declared_hash,
        },
    )


def audit_flat_freeze(
    *,
    repo_root: Path,
    artifact_root: Path,
    spec: FlatFreezeSpec,
) -> ExperimentAudit:
    freeze_path = artifact_root / spec.freeze_name
    result_doc = repo_root / spec.result_doc
    audit = ExperimentAudit(
        name=spec.name,
        freeze_path=str(freeze_path),
        result_doc=str(result_doc),
    )
    if not freeze_path.exists():
        audit.not_complete("Freeze has not been created yet.", path=freeze_path)
        return audit
    try:
        freeze = _json_object(freeze_path)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        audit.add(
            "freeze_json",
            "FAIL",
            f"Could not load freeze: {type(error).__name__}: {error}",
            path=freeze_path,
        )
        return audit
    run_dir_value = freeze.get("run_dir")
    hashes = freeze.get("hashes")
    if not isinstance(run_dir_value, str) or not isinstance(hashes, dict):
        audit.add(
            "freeze_schema",
            "FAIL",
            "Flat freeze requires string run_dir and object hashes.",
            path=freeze_path,
        )
        return audit
    run_dir = Path(run_dir_value)
    report_path = run_dir / "report.json"
    manifest_path = run_dir / "run_manifest.json"
    if not report_path.is_file() or not manifest_path.is_file():
        audit.add(
            "run_files",
            "FAIL",
            "Frozen run is missing report.json or run_manifest.json.",
            path=run_dir,
        )
        return audit
    try:
        report = _json_object(report_path)
        manifest = _json_object(manifest_path)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        audit.add(
            "run_json",
            "FAIL",
            f"Could not load run JSON: {type(error).__name__}: {error}",
            path=run_dir,
        )
        return audit

    _check_run_identity(
        audit,
        artifact_root=artifact_root,
        freeze=freeze,
        run_dir=run_dir,
        report=report,
        manifest=manifest,
    )
    _check_flat_claim_status(audit, freeze, report)
    checkpoints = sorted((run_dir / "checkpoints").glob("*.pt"))

    for key, expected in hashes.items():
        if key in spec.aliases:
            _check_file_hash(
                audit,
                name=f"hash:{key}",
                path=repo_root / spec.aliases[key],
                expected=expected,
            )
            continue
        manifest_match = CHECKPOINT_MANIFEST_RE.fullmatch(str(key))
        if manifest_match:
            expected_count = int(manifest_match.group(1))
            observed_hash = checkpoint_manifest_sha256(
                run_dir,
                checkpoints,
                scheme=spec.checkpoint_scheme,
            )
            count_ok = len(checkpoints) == expected_count
            hash_ok = isinstance(expected, str) and observed_hash == expected
            audit.add(
                f"hash:{key}",
                "PASS" if count_ok and hash_ok else "FAIL",
                "Checkpoint count and aggregate manifest hash match.",
                path=run_dir / "checkpoints",
                expected={"count": expected_count, "sha256": expected},
                observed={"count": len(checkpoints), "sha256": observed_hash},
            )
            continue
        _check_file_hash(
            audit,
            name=f"hash:{key}",
            path=run_dir / str(key),
            expected=expected,
        )

    if spec.checkpoint_metrics is not None:
        _check_checkpoint_rows(
            audit,
            run_dir=run_dir,
            metrics_path=run_dir / spec.checkpoint_metrics,
            checkpoint_paths=checkpoints,
        )
    if spec.name == "E10b":
        _audit_e10b_source_checkpoints(audit, run_dir)
    _check_document_tokens(audit, result_doc, spec.document_tokens)
    if spec.name == "E13a-R1":
        audit_e13a_r1_status_amendment(
            audit,
            repo_root=repo_root,
            artifact_root=artifact_root,
        )
    return audit


def audit_e13a_r1_status_amendment(
    audit: ExperimentAudit,
    *,
    repo_root: Path,
    artifact_root: Path,
) -> None:
    freeze_path = artifact_root / "E13A_R1_RESULT_STATUS_AMENDMENT_FREEZE_V1.json"
    if not freeze_path.is_file():
        audit.add(
            "status_amendment",
            "FAIL",
            "E13a-R1 final status requires the additive amendment freeze.",
            path=freeze_path,
        )
        return
    try:
        amendment = _json_object(freeze_path)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        audit.add(
            "status_amendment",
            "FAIL",
            f"Could not load status amendment: {type(error).__name__}: {error}",
            path=freeze_path,
        )
        return
    paths = amendment.get("paths")
    hashes = amendment.get("hashes")
    r1_status = amendment.get("e13a_r1")
    dependency = amendment.get("repaired_dependency")
    if not all(
        isinstance(value, dict)
        for value in (paths, hashes, r1_status, dependency)
    ):
        audit.add(
            "status_amendment_schema",
            "FAIL",
            "Amendment freeze is missing paths, hashes, or status sections.",
            path=freeze_path,
        )
        return
    assert isinstance(paths, dict)
    assert isinstance(hashes, dict)
    assert isinstance(r1_status, dict)
    assert isinstance(dependency, dict)
    expected_paths = {
        "amendment_markdown": repo_root
        / "docs/E13A_R1_RESULT_STATUS_AMENDMENT_KO.md",
        "original_result_markdown": repo_root
        / "docs/E13A_SEQUENCE_CALIBRATION_RESULT_KO.md",
        "original_e13a_r1_freeze": artifact_root
        / "E13A_R1_POSTCORE_ARTIFACT_FREEZE_V1.json",
        "e13a_r1_report": artifact_root
        / "e13a_r1_sequence_floor_throughput"
        / "20260727T183609.755945Z"
        / "report.json",
        "e13a_r1_run_manifest": artifact_root
        / "e13a_r1_sequence_floor_throughput"
        / "20260727T183609.755945Z"
        / "run_manifest.json",
        "e13a_r2_report": artifact_root
        / "e13a_r2_sequence_floor_throughput"
        / "20260727T190642.222102Z"
        / "report.json",
        "e13a_r2_run_manifest": artifact_root
        / "e13a_r2_sequence_floor_throughput"
        / "20260727T190642.222102Z"
        / "run_manifest.json",
    }
    for name, expected_path in expected_paths.items():
        raw_path = paths.get(name)
        declared_path = Path(raw_path) if isinstance(raw_path, str) else None
        path_matches = bool(
            declared_path is not None
            and declared_path.resolve() == expected_path.resolve()
        )
        audit.add(
            f"status_amendment_path:{name}",
            "PASS" if path_matches else "FAIL",
            "Amendment path points to the intended immutable source.",
            path=freeze_path,
            expected=str(expected_path.resolve()),
            observed=None if declared_path is None else str(declared_path.resolve()),
        )
        if path_matches:
            _check_file_hash(
                audit,
                name=f"status_amendment_hash:{name}",
                path=expected_path,
                expected=hashes.get(name),
            )
    expected_r1_status = {
        "experiment_id": "e13a_r1_sequence_floor_throughput",
        "run_id": "20260727T183609.755945Z",
        "execution_status": "PASS",
        "calibration_status": "GO_FOR_ORIGINAL_HARD_MASKED_PIPELINE_ONLY",
        "repaired_e13b_dependency_eligible": False,
        "diagnosis": "DISTRACTOR_PATH_STRUCTURALLY_HARD_MASKED",
    }
    expected_dependency = {
        "experiment_id": "e13a_r2_sequence_floor_throughput",
        "run_id": "20260727T190642.222102Z",
        "calibration_status": "GO_FOR_E13B_R1",
        "exclusive_for_repaired_e13b_r1": True,
    }
    audit.add(
        "status_amendment_r1_disposition",
        "PASS" if r1_status == expected_r1_status else "FAIL",
        "Additive amendment records the final R1 hard-mask-only disposition.",
        path=freeze_path,
        expected=expected_r1_status,
        observed=r1_status,
    )
    audit.add(
        "status_amendment_repaired_dependency",
        "PASS" if dependency == expected_dependency else "FAIL",
        "Only E13a-R2 is registered as the repaired E13b-R1 dependency.",
        path=freeze_path,
        expected=expected_dependency,
        observed=dependency,
    )
    audit.add(
        "status_amendment_immutability",
        "PASS"
        if amendment.get("original_artifacts_immutable") is True
        and amendment.get("immutable") is True
        else "FAIL",
        "The amendment explicitly preserves original artifacts and is immutable.",
        path=freeze_path,
        expected={
            "original_artifacts_immutable": True,
            "immutable": True,
        },
        observed={
            "original_artifacts_immutable": amendment.get(
                "original_artifacts_immutable"
            ),
            "immutable": amendment.get("immutable"),
        },
    )
    original_freeze_path = expected_paths["original_e13a_r1_freeze"]
    try:
        original_freeze = _json_object(original_freeze_path)
        r1_report = _json_object(expected_paths["e13a_r1_report"])
        r1_manifest = _json_object(expected_paths["e13a_r1_run_manifest"])
        r2_report = _json_object(expected_paths["e13a_r2_report"])
        r2_manifest = _json_object(expected_paths["e13a_r2_run_manifest"])
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        audit.add(
            "status_amendment_source_json",
            "FAIL",
            f"Could not read pinned source JSON: {type(error).__name__}: {error}",
            path=freeze_path,
        )
        return
    original_doc_sha = original_freeze.get("hashes", {}).get("result_markdown")
    audit.add(
        "status_amendment_original_document_chain",
        "PASS"
        if original_doc_sha == hashes.get("original_result_markdown")
        else "FAIL",
        "Amendment pins the same original result-document digest as the R1 freeze.",
        path=original_freeze_path,
        expected=original_doc_sha,
        observed=hashes.get("original_result_markdown"),
    )
    r1_chain = bool(
        _execution_status(r1_report) == "PASS"
        and _claim_gate(r1_report).get("go_for_e13b") is True
        and r1_manifest.get("report_sha256") == hashes.get("e13a_r1_report")
    )
    r2_chain = bool(
        _execution_status(r2_report) == "PASS"
        and _claim_gate(r2_report).get("go_for_e13b_r1") is True
        and r2_manifest.get("report_sha256") == hashes.get("e13a_r2_report")
    )
    audit.add(
        "status_amendment_r1_historical_report",
        "PASS" if r1_chain else "FAIL",
        "R1 historical GO remains pinned as the original hard-masked report value.",
        path=expected_paths["e13a_r1_report"],
        expected={
            "execution": "PASS",
            "go_for_e13b": True,
            "manifest_report_hash": hashes.get("e13a_r1_report"),
        },
        observed={
            "execution": _execution_status(r1_report),
            "go_for_e13b": _claim_gate(r1_report).get("go_for_e13b"),
            "manifest_report_hash": r1_manifest.get("report_sha256"),
        },
    )
    audit.add(
        "status_amendment_r2_dependency_report",
        "PASS" if r2_chain else "FAIL",
        "R2 report and manifest open the repaired E13b-R1 dependency.",
        path=expected_paths["e13a_r2_report"],
        expected={
            "execution": "PASS",
            "go_for_e13b_r1": True,
            "manifest_report_hash": hashes.get("e13a_r2_report"),
        },
        observed={
            "execution": _execution_status(r2_report),
            "go_for_e13b_r1": _claim_gate(r2_report).get("go_for_e13b_r1"),
            "manifest_report_hash": r2_manifest.get("report_sha256"),
        },
    )
    _check_document_tokens(
        audit,
        expected_paths["amendment_markdown"],
        (
            "ff1f13a6955719ada91120891404cbdb43e57d24c4e522f48e007f822e56dd4e",
            "GO_FOR_ORIGINAL_HARD_MASKED_PIPELINE_ONLY",
            "repaired_e13b_dependency_eligible: false",
            "20260727T190642.222102Z",
            "GO_FOR_E13B_R1",
        ),
    )


def _audit_e10b_source_checkpoints(audit: ExperimentAudit, run_dir: Path) -> None:
    verification_path = run_dir / "source_checkpoint_verification.jsonl"
    source_freeze_path = run_dir / "source_freeze.json"
    try:
        rows = _jsonl_objects(verification_path)
        source_freeze = _json_object(source_freeze_path)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        audit.add(
            "e10b_source_checkpoint_contract",
            "FAIL",
            f"Could not load source checkpoint contract: {type(error).__name__}: {error}",
            path=run_dir,
        )
        return
    source_run_value = source_freeze.get("source_run_dir")
    expected_count = source_freeze.get("source_checkpoint_count")
    if not isinstance(source_run_value, str) or not isinstance(expected_count, int):
        audit.add(
            "e10b_source_checkpoint_contract",
            "FAIL",
            "source_freeze.json has an invalid source run/count.",
            path=source_freeze_path,
        )
        return
    source_run = Path(source_run_value)
    invalid = 0
    for row in rows:
        raw_path = row.get("checkpoint")
        expected_sha = row.get("checkpoint_sha256")
        if not isinstance(raw_path, str) or not isinstance(expected_sha, str):
            invalid += 1
            continue
        path = Path(raw_path)
        if (
            not _is_within(path, source_run)
            or not path.is_file()
            or file_sha256(path) != expected_sha
            or row.get("hash_verified") is not True
        ):
            invalid += 1
    audit.add(
        "e10b_source_checkpoint_contract",
        "PASS" if len(rows) == expected_count and invalid == 0 else "FAIL",
        "Frozen E10b source checkpoint rows and file hashes match.",
        path=verification_path,
        expected={"count": expected_count, "invalid": 0},
        observed={"count": len(rows), "invalid": invalid},
    )
    try:
        from catena.eval.rank_saturation import canonical_checkpoint_index_sha256

        observed_index = canonical_checkpoint_index_sha256(rows, source_run_dir=source_run)
    except (ImportError, KeyError, TypeError, ValueError) as error:
        audit.add(
            "e10b_source_checkpoint_index",
            "FAIL",
            f"Could not calculate canonical checkpoint index: {type(error).__name__}: {error}",
            path=source_freeze_path,
        )
        return
    expected_index = source_freeze.get("source_checkpoint_index_sha256")
    audit.add(
        "e10b_source_checkpoint_index",
        "PASS" if observed_index == expected_index else "FAIL",
        "Canonical E10 source checkpoint index matches source_freeze.json.",
        path=source_freeze_path,
        expected=expected_index,
        observed=observed_index,
    )


def audit_e11(*, repo_root: Path, artifact_root: Path) -> ExperimentAudit:
    freeze_path = artifact_root / "E11_POSTCORE_ARTIFACT_FREEZE_V1.json"
    result_doc = repo_root / "docs/E11_REPRESENTATION_COADAPTATION_RESULT_KO.md"
    audit = ExperimentAudit(
        name="E11/E11b",
        freeze_path=str(freeze_path),
        result_doc=str(result_doc),
    )
    if not freeze_path.exists():
        audit.not_complete("Freeze has not been created yet.", path=freeze_path)
        return audit
    try:
        freeze = _json_object(freeze_path)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        audit.add(
            "freeze_json",
            "FAIL",
            f"Could not load freeze: {type(error).__name__}: {error}",
            path=freeze_path,
        )
        return audit
    original = freeze.get("original_e11")
    repair = freeze.get("e11b_prospective_repair")
    if not isinstance(original, dict) or not isinstance(repair, dict):
        audit.add("freeze_schema", "FAIL", "E11 freeze sections are missing.", path=freeze_path)
        return audit

    sections = (
        (
            "original",
            original,
            {
                "report_sha256": "report.json",
                "run_manifest_sha256": "run_manifest.json",
                "metrics_sha256": "coadaptation_metrics.jsonl",
            },
        ),
        (
            "repair",
            repair,
            {
                "report_sha256": "report.json",
                "run_manifest_sha256": "run_manifest.json",
                "metrics_sha256": "coadaptation_metrics.jsonl",
                "seed_contrasts_sha256": "seed_normalized_contrasts.jsonl",
            },
        ),
    )
    execution_parts: list[str] = []
    claim_parts: list[str] = []
    for section_name, section, mapping in sections:
        raw_run_dir = section.get("run_dir")
        if not isinstance(raw_run_dir, str):
            audit.add(
                f"{section_name}:run_dir",
                "FAIL",
                "E11 freeze section has no run directory.",
                path=freeze_path,
            )
            continue
        run_dir = Path(raw_run_dir)
        run_id = section.get("run_id")
        audit.add(
            f"{section_name}:run_id",
            "PASS" if run_dir.name == run_id else "FAIL",
            "Run ID matches run-directory basename.",
            path=run_dir,
            expected=run_id,
            observed=run_dir.name,
        )
        report_path = run_dir / "report.json"
        try:
            report = _json_object(report_path)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            audit.add(
                f"{section_name}:report",
                "FAIL",
                f"Could not load report: {type(error).__name__}: {error}",
                path=report_path,
            )
            continue
        observed_execution = _execution_status(report)
        expected_execution = section.get("execution_status")
        execution_parts.append(f"{section_name}={expected_execution}")
        disposition = str(section.get("claim_disposition"))
        claim_parts.append(f"{section_name}={disposition}")
        audit.add(
            f"{section_name}:execution_status",
            "PASS" if observed_execution == expected_execution else "FAIL",
            "Execution status is independent of claim disposition.",
            path=report_path,
            expected=expected_execution,
            observed=observed_execution,
        )
        expected_supported = disposition == "SUPPORTED"
        observed_supported = _claim_gate(report).get("supported")
        audit.add(
            f"{section_name}:claim_disposition",
            "PASS" if observed_supported is expected_supported else "FAIL",
            "Claim disposition is checked separately from execution.",
            path=report_path,
            expected={"disposition": disposition, "supported": expected_supported},
            observed={"supported": observed_supported},
        )
        for hash_key, relative in mapping.items():
            _check_file_hash(
                audit,
                name=f"{section_name}:hash:{hash_key}",
                path=run_dir / relative,
                expected=section.get(hash_key),
            )

    audit.execution_status = ", ".join(execution_parts)
    audit.claim_status = ", ".join(claim_parts)
    repair_run_value = repair.get("run_dir")
    if isinstance(repair_run_value, str):
        repair_run = Path(repair_run_value)
        checkpoints = sorted((repair_run / "checkpoints").glob("*.pt"))
        expected_count = repair.get("checkpoint_count")
        observed_manifest = checkpoint_manifest_sha256(
            repair_run,
            checkpoints,
            scheme="legacy_sha_path",
        )
        expected_manifest = repair.get("checkpoint_manifest_sha256")
        audit.add(
            "repair:checkpoint_manifest",
            (
                "PASS"
                if len(checkpoints) == expected_count and observed_manifest == expected_manifest
                else "FAIL"
            ),
            "Legacy E11b checkpoint count and aggregate manifest hash match.",
            path=repair_run / "checkpoints",
            expected={"count": expected_count, "sha256": expected_manifest},
            observed={"count": len(checkpoints), "sha256": observed_manifest},
        )
        _check_checkpoint_rows(
            audit,
            run_dir=repair_run,
            metrics_path=repair_run / "coadaptation_metrics.jsonl",
            checkpoint_paths=checkpoints,
        )
    protocol_path = repair.get("protocol_lock_path")
    if isinstance(protocol_path, str):
        path = Path(protocol_path)
        if not path.is_absolute():
            path = repo_root / path
        _check_file_hash(
            audit,
            name="repair:protocol_lock",
            path=path,
            expected=repair.get("protocol_lock_sha256"),
        )
    else:
        audit.add(
            "repair:protocol_lock",
            "FAIL",
            "E11b protocol lock path is not declared.",
            path=freeze_path,
        )
    _check_document_tokens(
        audit,
        result_doc,
        (
            "20260727T180703.763554Z",
            "NOT_OPENED_SCALE_RESTRICTION",
            "20260727T183004.928280Z",
            "SUPPORTED",
            str(freeze_path),
        ),
    )
    audit.add(
        "result_document_seal",
        "WARN",
        "E11 result document is referenced by path but has no digest in the freeze.",
        path=result_doc,
        observed=file_sha256(result_doc) if result_doc.is_file() else None,
    )
    return audit


def _nested_value(payload: dict[str, Any], dotted_path: str) -> Any:
    value: Any = payload
    for part in dotted_path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(dotted_path)
        value = value[part]
    return value


def _resolve_latest_run(experiment_root: Path) -> Path:
    pointer_path = experiment_root / "latest.json"
    pointer = _json_object(pointer_path)
    raw_run = pointer.get("run_dir")
    if not isinstance(raw_run, str):
        raise TypeError(f"latest.json has no string run_dir: {pointer_path}")
    candidate = Path(raw_run)
    if not candidate.is_absolute():
        candidate = experiment_root / candidate
    resolved = candidate.resolve()
    if not _is_within(resolved, experiment_root):
        raise ValueError(f"latest.json run_dir escapes experiment root: {resolved}")
    return resolved


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _audit_e13b_manifest(
    audit: ExperimentAudit,
    *,
    repo_root: Path,
    run_dir: Path,
    manifest: dict[str, Any],
    completed: bool,
) -> dict[str, Any] | None:
    resolved_config_path = run_dir / "config.resolved.yaml"
    config_path_value = manifest.get("config_path")
    if not resolved_config_path.is_file() or not isinstance(config_path_value, str):
        audit.add(
            f"{run_dir.name}:config_provenance",
            "FAIL",
            "Run manifest lacks its resolved or source config path.",
            path=run_dir,
        )
        return None
    config_path = Path(config_path_value)
    try:
        resolved_config = yaml.safe_load(resolved_config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        audit.add(
            f"{run_dir.name}:resolved_config",
            "FAIL",
            f"Could not load resolved config: {type(error).__name__}: {error}",
            path=resolved_config_path,
        )
        return None
    expected_manifest = {
        "schema_version": 2,
        "experiment_id": "e13b_r1_transactional_sequence_memory",
        "run_id": run_dir.name,
        "run_mode": "MAIN",
        "source_fingerprint_phase": "RUN_START",
    }
    observed_manifest = {key: manifest.get(key) for key in expected_manifest}
    audit.add(
        f"{run_dir.name}:manifest_identity",
        "PASS" if observed_manifest == expected_manifest else "FAIL",
        "E13b-R1 manifest identifies a schema-v2 MAIN run-start record.",
        path=run_dir / "run_manifest.json",
        expected=expected_manifest,
        observed=observed_manifest,
    )
    config_checks = {
        "manifest_config": manifest.get("config") == resolved_config,
        "resolved_config_artifact_sha256": (
            manifest.get("resolved_config_artifact_sha256")
            == file_sha256(resolved_config_path)
        ),
        "resolved_config_sha256": (
            manifest.get("resolved_config_sha256") == _canonical_sha256(resolved_config)
        ),
        "config_file_sha256": (
            config_path.is_file()
            and manifest.get("config_file_sha256") == file_sha256(config_path)
        ),
    }
    audit.add(
        f"{run_dir.name}:config_provenance",
        "PASS" if all(config_checks.values()) else "FAIL",
        "Resolved/source config files match every manifest digest and value.",
        path=resolved_config_path,
        expected={key: True for key in config_checks},
        observed=config_checks,
    )
    source = manifest.get("source_fingerprint")
    source_valid = bool(
        isinstance(source, dict)
        and isinstance(source.get("sha256"), str)
        and SHA256_RE.fullmatch(source["sha256"])
        and isinstance(source.get("files"), int)
        and source["files"] > 0
    )
    audit.add(
        f"{run_dir.name}:source_fingerprint_shape",
        "PASS" if source_valid else "FAIL",
        "Run-start source fingerprint has a valid digest and positive file count.",
        path=run_dir / "run_manifest.json",
        observed=source,
    )
    if source_valid:
        try:
            from catena.core.provenance_v61 import source_tree_fingerprint

            live_source = source_tree_fingerprint(repo_root).as_dict()
        except (OSError, ValueError, TypeError) as error:
            audit.add(
                f"{run_dir.name}:live_source_fingerprint",
                "WARN",
                f"Could not compare the live source tree: {type(error).__name__}: {error}",
                path=repo_root,
            )
        else:
            audit.add(
                f"{run_dir.name}:live_source_fingerprint",
                "PASS" if live_source == source else "WARN",
                (
                    "Live source tree still matches the run-start fingerprint."
                    if live_source == source
                    else (
                        "Live source tree has changed since run start. The manifest "
                        "preserves the run-start digest, but no source snapshot is "
                        "available for bytewise reconstruction."
                    )
                ),
                path=repo_root,
                expected=source,
                observed=live_source,
            )
    final_fields = {
        "completed_at_utc": isinstance(manifest.get("completed_at_utc"), str),
        "report_sha256": (
            (run_dir / "report.json").is_file()
            and manifest.get("report_sha256") == file_sha256(run_dir / "report.json")
        ),
    }
    if completed:
        audit.add(
            f"{run_dir.name}:final_manifest",
            "PASS" if all(final_fields.values()) else "FAIL",
            "Completed manifest pins the final report and completion time.",
            path=run_dir / "run_manifest.json",
            expected={"completed_at_utc": True, "report_sha256": True},
            observed=final_fields,
        )
    else:
        no_final_fields = (
            "completed_at_utc" not in manifest and "report_sha256" not in manifest
        )
        audit.add(
            f"{run_dir.name}:run_start_only",
            "PASS" if no_final_fields else "FAIL",
            "Incomplete run has not been mislabeled with final provenance.",
            path=run_dir / "run_manifest.json",
            expected=True,
            observed=no_final_fields,
        )
    return resolved_config if isinstance(resolved_config, dict) else None


def _audit_e13b_checkpoint_payload(
    audit: ExperimentAudit,
    *,
    run_dir: Path,
    checkpoint: Path,
    expected_variant: str,
    expected_seed: int,
    expected_config: dict[str, Any],
) -> None:
    try:
        import torch

        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        audit.add(
            f"{run_dir.name}:checkpoint_payload",
            "FAIL",
            f"Could not load checkpoint safely on CPU: {type(error).__name__}: {error}",
            path=checkpoint,
        )
        return
    if not isinstance(payload, dict):
        audit.add(
            f"{run_dir.name}:checkpoint_payload",
            "FAIL",
            "Checkpoint payload is not a mapping.",
            path=checkpoint,
            observed=type(payload).__name__,
        )
        return
    observed = {
        "model_class": payload.get("model_class"),
        "variant": payload.get("variant"),
        "seed": payload.get("seed"),
        "config_matches": payload.get("config") == expected_config,
        "has_model_state": isinstance(payload.get("model"), dict),
    }
    expected = {
        "model_class": "TransactionalSequenceMemoryV2",
        "variant": expected_variant,
        "seed": expected_seed,
        "config_matches": True,
        "has_model_state": True,
    }
    audit.add(
        f"{run_dir.name}:checkpoint_payload",
        "PASS" if observed == expected else "FAIL",
        "CPU-loaded checkpoint identity/config matches report and metrics.",
        path=checkpoint,
        expected=expected,
        observed=observed,
    )


def _audit_completed_e13b_run(
    audit: ExperimentAudit,
    *,
    run_dir: Path,
    manifest: dict[str, Any],
    config: dict[str, Any],
) -> None:
    report_path = run_dir / "report.json"
    metrics_path = run_dir / "sequence_main_metrics.jsonl"
    if not report_path.is_file() or not metrics_path.is_file():
        audit.add(
            f"{run_dir.name}:scientific_outputs",
            "FAIL",
            "Completed E13b-R1 run is missing report or metrics.",
            path=run_dir,
        )
        return
    try:
        report = _json_object(report_path)
        rows = _jsonl_objects(metrics_path)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        audit.add(
            f"{run_dir.name}:scientific_outputs",
            "FAIL",
            f"Could not load completed outputs: {type(error).__name__}: {error}",
            path=run_dir,
        )
        return
    variants = {str(row.get("variant")) for row in rows}
    seeds = {int(row["seed"]) for row in rows if isinstance(row.get("seed"), int)}
    checkpoints = {Path(str(row.get("checkpoint"))).resolve() for row in rows}
    expected_variant = str(report.get("variant"))
    expected_seed_value = report.get("seed")
    expected_seed = int(expected_seed_value) if isinstance(expected_seed_value, int) else -1
    identity_ok = (
        report.get("status") == "PASS"
        and report.get("run_mode") == "MAIN"
        and len(rows) == report.get("rows")
        and variants == {expected_variant}
        and seeds == {expected_seed}
        and expected_variant in {"tied", "dual"}
        and expected_seed in set(int(seed) for seed in config.get("seeds", []))
        and len(checkpoints) == 1
    )
    audit.add(
        f"{run_dir.name}:report_metric_identity",
        "PASS" if identity_ok else "FAIL",
        "Report, metric rows, variant, seed, and row count agree.",
        path=metrics_path,
        expected={
            "status": "PASS",
            "run_mode": "MAIN",
            "variant": expected_variant,
            "seed": expected_seed,
            "rows": report.get("rows"),
            "checkpoint_count": 1,
        },
        observed={
            "status": report.get("status"),
            "run_mode": report.get("run_mode"),
            "variants": sorted(variants),
            "seeds": sorted(seeds),
            "rows": len(rows),
            "checkpoint_count": len(checkpoints),
        },
    )
    if len(checkpoints) != 1:
        return
    checkpoint = next(iter(checkpoints))
    checkpoint_hashes = {str(row.get("checkpoint_sha256")) for row in rows}
    checkpoint_ok = bool(
        checkpoint.parent == (run_dir / "checkpoints").resolve()
        and checkpoint.is_file()
        and len(checkpoint_hashes) == 1
        and next(iter(checkpoint_hashes)) == file_sha256(checkpoint)
    )
    audit.add(
        f"{run_dir.name}:checkpoint_file",
        "PASS" if checkpoint_ok else "FAIL",
        "Metric rows pin one in-run checkpoint with a matching digest.",
        path=checkpoint,
        expected=next(iter(checkpoint_hashes)) if len(checkpoint_hashes) == 1 else None,
        observed=file_sha256(checkpoint) if checkpoint.is_file() else None,
    )
    if checkpoint_ok:
        _audit_e13b_checkpoint_payload(
            audit,
            run_dir=run_dir,
            checkpoint=checkpoint,
            expected_variant=expected_variant,
            expected_seed=expected_seed,
            expected_config=config,
        )
    report_sha = manifest.get("report_sha256")
    audit.add(
        f"{run_dir.name}:report_hash",
        "PASS" if report_sha == file_sha256(report_path) else "FAIL",
        "Final manifest report digest matches the report.",
        path=report_path,
        expected=report_sha,
        observed=file_sha256(report_path),
    )


def audit_e13b_live(*, repo_root: Path, artifact_root: Path) -> ExperimentAudit:
    source_root = artifact_root / "e13b_r1_transactional_sequence_memory"
    audit = ExperimentAudit(
        name="E13b-R1 live",
        freeze_path=None,
        result_doc=None,
        claim_status="PENDING_AGGREGATE",
    )
    if not source_root.is_dir():
        audit.not_complete("E13b-R1 namespace has not been created yet.", path=source_root)
        return audit
    run_dirs = sorted(path for path in source_root.iterdir() if path.is_dir())
    if not run_dirs:
        audit.not_complete("No E13b-R1 run directories exist yet.", path=source_root)
        return audit
    completed_count = 0
    in_progress_count = 0
    for run_dir in run_dirs:
        manifest_path = run_dir / "run_manifest.json"
        if not manifest_path.is_file():
            audit.add(
                f"{run_dir.name}:availability",
                "WARN",
                "Run directory exists without a manifest and may still be starting.",
                path=run_dir,
            )
            in_progress_count += 1
            continue
        try:
            manifest = _json_object(manifest_path)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            audit.add(
                f"{run_dir.name}:manifest",
                "FAIL",
                f"Could not load manifest: {type(error).__name__}: {error}",
                path=manifest_path,
            )
            continue
        completed = (run_dir / "report.json").is_file()
        config = _audit_e13b_manifest(
            audit,
            repo_root=repo_root,
            run_dir=run_dir,
            manifest=manifest,
            completed=completed,
        )
        if completed:
            completed_count += 1
            if config is not None:
                _audit_completed_e13b_run(
                    audit,
                    run_dir=run_dir,
                    manifest=manifest,
                    config=config,
                )
        else:
            in_progress_count += 1
    audit.execution_status = (
        f"completed={completed_count}, in_progress={in_progress_count}"
    )
    audit.add(
        "run_inventory",
        "PASS" if completed_count > 0 else "SKIP",
        (
            "Completed runs were audited; unfinished runs remain non-evidence."
            if completed_count > 0
            else "All observed runs are still incomplete and were tolerated."
        ),
        path=source_root,
        observed={
            "completed": completed_count,
            "in_progress": in_progress_count,
            "total": len(run_dirs),
        },
    )
    if completed_count == 0 and audit.status != "FAIL":
        audit.status = "NOT_COMPLETE"
    return audit


def _source_row_identity(row: dict[str, Any]) -> tuple[int, str]:
    return int(row["seed"]), str(row["variant"])


def _audit_source_rows(
    audit: ExperimentAudit,
    *,
    rows: list[dict[str, Any]],
    artifact_root: Path,
    check_name: str,
) -> dict[tuple[int, str], dict[str, Any]]:
    indexed: dict[tuple[int, str], dict[str, Any]] = {}
    for index, row in enumerate(rows):
        try:
            key = _source_row_identity(row)
        except (KeyError, TypeError, ValueError) as error:
            audit.add(
                f"{check_name}:{index}:identity",
                "FAIL",
                f"Source row has invalid identity: {type(error).__name__}: {error}",
            )
            continue
        if key in indexed:
            audit.add(
                f"{check_name}:{index}:identity",
                "FAIL",
                "Duplicate source seed/variant identity.",
                expected="unique seed/variant",
                observed={"seed": key[0], "variant": key[1]},
            )
            continue
        indexed[key] = row
        raw_checkpoint = row.get("checkpoint_path")
        raw_run_dir = row.get("source_run_dir", row.get("run_dir"))
        if not isinstance(raw_run_dir, str) and isinstance(raw_checkpoint, str):
            checkpoint_parent = Path(raw_checkpoint).resolve().parent
            if checkpoint_parent.name == "checkpoints":
                raw_run_dir = str(checkpoint_parent.parent)
        if not isinstance(raw_run_dir, str) or not isinstance(raw_checkpoint, str):
            audit.add(
                f"{check_name}:{key}:paths",
                "FAIL",
                "Source row lacks run/checkpoint paths.",
            )
            continue
        run_dir = Path(raw_run_dir).resolve()
        checkpoint = Path(raw_checkpoint).resolve()
        declared_run_id = row.get("source_run_id", row.get("run_id"))
        if declared_run_id is not None:
            audit.add(
                f"{check_name}:{key}:run_id",
                "PASS" if declared_run_id == run_dir.name else "FAIL",
                "Declared source run ID matches the source directory.",
                path=run_dir,
                expected=run_dir.name,
                observed=declared_run_id,
            )
        expected_run_root = (
            artifact_root / "e13b_r1_transactional_sequence_memory"
        ).resolve()
        paths_ok = bool(
            _is_within(run_dir, expected_run_root)
            and checkpoint.parent == (run_dir / "checkpoints").resolve()
        )
        audit.add(
            f"{check_name}:{key}:path_contract",
            "PASS" if paths_ok else "FAIL",
            "Source checkpoint remains inside its E13b-R1 run.",
            path=checkpoint,
            expected=str(run_dir / "checkpoints"),
            observed=str(checkpoint.parent),
        )
        file_contracts = (
            (
                checkpoint,
                row.get("checkpoint_sha256"),
                "checkpoint",
            ),
            (
                Path(
                    str(
                        row.get(
                            "source_report_path",
                            row.get("report_path", run_dir / "report.json"),
                        )
                    )
                ),
                row.get("source_report_sha256", row.get("report_sha256")),
                "report",
            ),
            (
                Path(
                    str(
                        row.get(
                            "source_metrics_path",
                            row.get(
                                "metrics_path",
                                run_dir / "sequence_main_metrics.jsonl",
                            ),
                        )
                    )
                ),
                row.get("source_metrics_sha256", row.get("metrics_sha256")),
                "metrics",
            ),
            (
                Path(
                    str(
                        row.get(
                            "source_manifest_path",
                            row.get(
                                "run_manifest_path",
                                run_dir / "run_manifest.json",
                            ),
                        )
                    )
                ),
                row.get(
                    "source_manifest_sha256",
                    row.get("run_manifest_sha256"),
                ),
                "manifest",
            ),
        )
        for path, digest, label in file_contracts:
            _check_file_hash(
                audit,
                name=f"{check_name}:{key}:hash:{label}",
                path=path,
                expected=digest,
            )
    return indexed


def _source_provenance_projection(row: dict[str, Any]) -> dict[str, Any]:
    raw_run_dir = row.get("source_run_dir", row.get("run_dir"))
    source_run_id = row.get("source_run_id", row.get("run_id"))
    if source_run_id is None and isinstance(raw_run_dir, str):
        source_run_id = Path(raw_run_dir).name
    return {
        "seed": row.get("seed"),
        "variant": row.get("variant"),
        "source_run_id": source_run_id,
        "checkpoint_path": row.get("checkpoint_path"),
        "checkpoint_sha256": row.get("checkpoint_sha256"),
        "source_report_sha256": row.get(
            "source_report_sha256",
            row.get("report_sha256"),
        ),
        "source_metrics_sha256": row.get(
            "source_metrics_sha256",
            row.get("metrics_sha256"),
        ),
        "source_manifest_sha256": row.get(
            "source_manifest_sha256",
            row.get("run_manifest_sha256"),
        ),
    }


def _nested_file_record(
    audit: ExperimentAudit,
    *,
    record: object,
    name: str,
    expected_parent: Path | None = None,
) -> Path | None:
    if not isinstance(record, dict):
        audit.add(name, "FAIL", "Declared artifact record is not an object.")
        return None
    raw_path = record.get("path")
    if not isinstance(raw_path, str):
        audit.add(name, "FAIL", "Declared artifact record has no path.")
        return None
    path = Path(raw_path)
    if expected_parent is not None and path.resolve().parent != expected_parent.resolve():
        audit.add(
            f"{name}:parent",
            "FAIL",
            "Declared artifact is outside its registered run directory.",
            path=path,
            expected=str(expected_parent.resolve()),
            observed=str(path.resolve().parent),
        )
    _check_file_hash(
        audit,
        name=name,
        path=path,
        expected=record.get("sha256"),
    )
    return path


def audit_e13bc_freeze(*, repo_root: Path, artifact_root: Path) -> ExperimentAudit:
    freeze_path = artifact_root / "E13BC_R1_POSTCORE_ARTIFACT_FREEZE_V1.json"
    result_doc = repo_root / "docs/E13BC_TRANSACTIONAL_SEQUENCE_RESULT_KO.md"
    audit = ExperimentAudit(
        name="E13b/c-R1",
        freeze_path=str(freeze_path),
        result_doc=str(result_doc),
    )
    if not freeze_path.is_file():
        audit.not_complete("E13b/c-R1 freeze has not been created yet.", path=freeze_path)
        return audit
    try:
        freeze = _json_object(freeze_path)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        audit.add(
            "freeze_json",
            "FAIL",
            f"Could not load E13b/c freeze: {type(error).__name__}: {error}",
            path=freeze_path,
        )
        return audit
    audit.execution_status = (
        None
        if freeze.get("execution_status") is None
        else str(freeze.get("execution_status"))
    )
    audit.claim_status = (
        None if freeze.get("claim_status") is None else str(freeze.get("claim_status"))
    )
    schema_expected = {
        "schema_version": 1,
        "freeze_id": "E13BC_R1_POSTCORE_ARTIFACT_FREEZE_V1",
        "experiment_ids": [
            "e13b_r1_transactional_sequence_memory",
            "e13c_r1_transactional_sequence_aggregate",
        ],
        "execution_status": "PASS",
        "claim_status": "SUPPORTED",
        "evidence_tier": "CONTROLLED_REFERENCE",
        "scientific_evidence": False,
        "immutable": True,
    }
    schema_observed = {key: freeze.get(key) for key in schema_expected}
    audit.add(
        "freeze_status_schema",
        "PASS" if schema_observed == schema_expected else "FAIL",
        "E13b/c freeze separates completed execution, supported claim, and evidence tier.",
        path=freeze_path,
        expected=schema_expected,
        observed=schema_observed,
    )
    aggregate = freeze.get("aggregate")
    calibration = freeze.get("calibration_dependency")
    source_runs = freeze.get("source_runs")
    excluded = freeze.get("excluded_run_start_manifests")
    protocols = freeze.get("protocol_and_locks")
    result_record = freeze.get("result_document")
    if not (
        isinstance(aggregate, dict)
        and isinstance(calibration, dict)
        and isinstance(source_runs, list)
        and isinstance(excluded, list)
        and isinstance(protocols, dict)
    ):
        audit.add(
            "freeze_schema_sections",
            "FAIL",
            "E13b/c freeze lacks aggregate, dependency, source, exclusion, or lock sections.",
            path=freeze_path,
        )
        return audit
    aggregate_run_value = aggregate.get("run_dir")
    aggregate_files = aggregate.get("files")
    if not isinstance(aggregate_run_value, str) or not isinstance(aggregate_files, dict):
        audit.add(
            "aggregate_schema",
            "FAIL",
            "Aggregate run directory or files are invalid.",
            path=freeze_path,
        )
        return audit
    aggregate_run = Path(aggregate_run_value)
    expected_aggregate_root = (
        artifact_root / "e13c_r1_transactional_sequence_aggregate"
    )
    aggregate_contained = bool(
        _is_within(aggregate_run, expected_aggregate_root)
        and aggregate.get("run_id") == aggregate_run.name
    )
    audit.add(
        "aggregate_identity",
        "PASS" if aggregate_contained else "FAIL",
        "Aggregate run ID/path belongs to the E13c-R1 namespace.",
        path=aggregate_run,
        expected=str(expected_aggregate_root.resolve()),
        observed=str(aggregate_run.resolve()),
    )
    aggregate_paths: dict[str, Path] = {}
    for name, record in aggregate_files.items():
        path = _nested_file_record(
            audit,
            record=record,
            name=f"aggregate:{name}",
            expected_parent=aggregate_run,
        )
        if path is not None:
            aggregate_paths[str(name)] = path
    required_aggregate_files = {
        "report.json",
        "run_manifest.json",
        "sequence_paired_metrics.jsonl",
        "sequence_stress_seed_metrics.jsonl",
        "source_run_provenance.jsonl",
        "excluded_operational_incomplete_runs.jsonl",
    }
    audit.add(
        "aggregate_file_set",
        "PASS" if set(aggregate_files) == required_aggregate_files else "FAIL",
        "Aggregate freeze declares every scientific and provenance output.",
        expected=sorted(required_aggregate_files),
        observed=sorted(aggregate_files),
    )
    try:
        aggregate_report = _json_object(aggregate_paths["report.json"])
        aggregate_manifest = _json_object(aggregate_paths["run_manifest.json"])
        aggregate_source_rows = _jsonl_objects(
            aggregate_paths["source_run_provenance.jsonl"]
        )
        aggregate_excluded_rows = _jsonl_objects(
            aggregate_paths["excluded_operational_incomplete_runs.jsonl"]
        )
    except (
        KeyError,
        OSError,
        ValueError,
        TypeError,
        json.JSONDecodeError,
    ) as error:
        audit.add(
            "aggregate_outputs",
            "FAIL",
            f"Could not load aggregate outputs: {type(error).__name__}: {error}",
            path=aggregate_run,
        )
        return audit
    aggregate_identity = {
        "report_status": _execution_status(aggregate_report),
        "report_supported": _claim_gate(aggregate_report).get("supported"),
        "manifest_schema": aggregate_manifest.get("schema_version"),
        "manifest_experiment": aggregate_manifest.get("experiment_id"),
        "manifest_run_id": aggregate_manifest.get("run_id"),
        "manifest_run_mode": aggregate_manifest.get("run_mode"),
        "manifest_report_sha256": aggregate_manifest.get("report_sha256"),
    }
    expected_aggregate_identity = {
        "report_status": "PASS",
        "report_supported": True,
        "manifest_schema": 2,
        "manifest_experiment": "e13c_r1_transactional_sequence_aggregate",
        "manifest_run_id": aggregate_run.name,
        "manifest_run_mode": "MAIN",
        "manifest_report_sha256": aggregate_files["report.json"].get("sha256"),
    }
    audit.add(
        "aggregate_status_chain",
        "PASS" if aggregate_identity == expected_aggregate_identity else "FAIL",
        "Aggregate manifest and report form one completed MAIN/PASS/SUPPORTED chain.",
        path=aggregate_run,
        expected=expected_aggregate_identity,
        observed=aggregate_identity,
    )
    calibration_paths: dict[str, Path] = {}
    for name in ("report", "run_manifest"):
        path = _nested_file_record(
            audit,
            record=calibration.get(name),
            name=f"calibration:{name}",
        )
        if path is not None:
            calibration_paths[name] = path
    try:
        calibration_report = _json_object(calibration_paths["report"])
        calibration_manifest = _json_object(calibration_paths["run_manifest"])
    except (
        KeyError,
        OSError,
        ValueError,
        TypeError,
        json.JSONDecodeError,
    ) as error:
        audit.add(
            "calibration_dependency",
            "FAIL",
            f"Could not load calibration dependency: {type(error).__name__}: {error}",
        )
        return audit
    calibration_identity = {
        "experiment_id": calibration.get("experiment_id"),
        "run_id": calibration.get("run_id"),
        "status": calibration.get("status"),
        "go_for_e13b_r1": calibration.get("go_for_e13b_r1"),
        "report_status": _execution_status(calibration_report),
        "report_go": _claim_gate(calibration_report).get("go_for_e13b_r1"),
        "manifest_experiment": calibration_manifest.get("experiment_id"),
        "manifest_run_id": calibration_manifest.get("run_id"),
        "manifest_run_mode": calibration_manifest.get("run_mode"),
        "manifest_report_sha256": calibration_manifest.get("report_sha256"),
    }
    calibration_expected = {
        "experiment_id": "e13a_r2_sequence_floor_throughput",
        "run_id": "20260727T190642.222102Z",
        "status": "PASS",
        "go_for_e13b_r1": True,
        "report_status": "PASS",
        "report_go": True,
        "manifest_experiment": "e13a_r2_sequence_floor_throughput",
        "manifest_run_id": "20260727T190642.222102Z",
        "manifest_run_mode": "MAIN",
        "manifest_report_sha256": calibration["report"].get("sha256"),
    }
    audit.add(
        "calibration_dependency_chain",
        "PASS" if calibration_identity == calibration_expected else "FAIL",
        "E13b/c main depends only on the completed E13a-R2 GO report.",
        expected=calibration_expected,
        observed=calibration_identity,
    )
    normalized_source_rows: list[dict[str, Any]] = []
    for row in source_runs:
        if not isinstance(row, dict):
            continue
        report_record = row.get("report")
        raw_report_path = (
            report_record.get("path")
            if isinstance(report_record, dict)
            else None
        )
        normalized = {
            "seed": row.get("seed"),
            "variant": row.get("variant"),
            "run_id": row.get("run_id"),
            "run_dir": (
                str(Path(raw_report_path).parent)
                if isinstance(raw_report_path, str)
                else None
            ),
        }
        for target_name, source_name in (
            ("report", "report"),
            ("run_manifest", "run_manifest"),
            ("metrics", "metrics"),
            ("checkpoint", "checkpoint"),
        ):
            record = row.get(source_name)
            if isinstance(record, dict):
                normalized[f"{target_name}_path"] = record.get("path")
                normalized[f"{target_name}_sha256"] = record.get("sha256")
        normalized_source_rows.append(normalized)
    indexed_frozen_sources = _audit_source_rows(
        audit,
        rows=normalized_source_rows,
        artifact_root=artifact_root,
        check_name="e13bc_source",
    )
    required_source_keys = {
        (seed, variant)
        for seed in (101, 211, 307, 401, 503)
        for variant in ("tied", "dual")
    }
    audit.add(
        "source_run_set",
        "PASS" if set(indexed_frozen_sources) == required_source_keys else "FAIL",
        "Freeze contains exactly the ten paired E13b-R1 source runs.",
        expected=sorted(required_source_keys),
        observed=sorted(indexed_frozen_sources),
    )
    indexed_aggregate_sources = {
        _source_row_identity(row): row
        for row in aggregate_source_rows
    }
    audit.add(
        "aggregate_source_run_set",
        "PASS"
        if len(aggregate_source_rows) == len(indexed_aggregate_sources)
        and set(indexed_aggregate_sources) == required_source_keys
        else "FAIL",
        "Aggregate provenance contains exactly the same ten unique paired sources.",
        path=aggregate_paths["source_run_provenance.jsonl"],
        expected=sorted(required_source_keys),
        observed=sorted(indexed_aggregate_sources),
    )
    source_chain_mismatches: list[dict[str, Any]] = []
    for key, frozen in indexed_frozen_sources.items():
        aggregate_row = indexed_aggregate_sources.get(key)
        if aggregate_row is None:
            source_chain_mismatches.append({"key": key, "reason": "missing"})
            continue
        comparisons = {
            "report": (
                frozen.get("report_sha256")
                == aggregate_row.get("report_sha256")
            ),
            "manifest": (
                frozen.get("run_manifest_sha256")
                == aggregate_row.get("run_manifest_sha256")
            ),
            "metrics": (
                frozen.get("metrics_sha256")
                == aggregate_row.get("metrics_sha256")
            ),
            "checkpoint": (
                frozen.get("checkpoint_sha256")
                == aggregate_row.get("checkpoint_sha256")
            ),
        }
        if not all(comparisons.values()):
            source_chain_mismatches.append(
                {"key": key, "comparisons": comparisons}
            )
    audit.add(
        "source_to_aggregate_chain",
        "PASS" if not source_chain_mismatches else "FAIL",
        "All ten frozen source digests equal E13c source provenance.",
        path=aggregate_paths["source_run_provenance.jsonl"],
        expected=[],
        observed=source_chain_mismatches,
    )
    excluded_index = {
        str(row.get("run_id")): row
        for row in excluded
        if isinstance(row, dict)
    }
    aggregate_excluded_index = {
        Path(str(row.get("run_dir"))).name: row
        for row in aggregate_excluded_rows
    }
    expected_excluded_ids = {
        "20260727T191226.039404Z",
        "20260727T191226.069595Z",
        "20260727T191226.084356Z",
        "20260727T191226.089631Z",
    }
    audit.add(
        "excluded_run_set",
        "PASS"
        if set(excluded_index) == expected_excluded_ids
        and set(aggregate_excluded_index) == expected_excluded_ids
        else "FAIL",
        "Exactly four run-start-only launch remnants are frozen and excluded.",
        expected=sorted(expected_excluded_ids),
        observed={
            "freeze": sorted(excluded_index),
            "aggregate": sorted(aggregate_excluded_index),
        },
    )
    for run_id, row in excluded_index.items():
        path = _nested_file_record(
            audit,
            record=row,
            name=f"excluded:{run_id}:manifest",
        )
        if path is None:
            continue
        run_dir = path.parent
        names = {child.name for child in run_dir.iterdir()}
        expected_names = {
            "config.resolved.yaml",
            "environment.json",
            "run_manifest.json",
        }
        aggregate_row = aggregate_excluded_index.get(run_id, {})
        expected_source_root = (
            artifact_root / "e13b_r1_transactional_sequence_memory"
        )
        excluded_ok = bool(
            _is_within(run_dir, expected_source_root)
            and run_dir.name == run_id
            and names == expected_names
            and Path(str(aggregate_row.get("run_dir"))).resolve()
            == run_dir.resolve()
            and aggregate_row.get("run_manifest_sha256") == row.get("sha256")
            and aggregate_row.get("disposition")
            == "EXCLUDED_OPERATIONAL_INCOMPLETE_RUN_START_ONLY"
        )
        audit.add(
            f"excluded:{run_id}:contract",
            "PASS" if excluded_ok else "FAIL",
            "Excluded directory is strictly run-start-only and matches aggregate provenance.",
            path=run_dir,
            expected={
                "files": sorted(expected_names),
                "disposition": "EXCLUDED_OPERATIONAL_INCOMPLETE_RUN_START_ONLY",
                "sha256": row.get("sha256"),
            },
            observed={
                "files": sorted(names),
                "disposition": aggregate_row.get("disposition"),
                "sha256": aggregate_row.get("run_manifest_sha256"),
            },
        )
    expected_protocols = {
        "configs/e13b_r1_transactional_sequence_memory.yaml",
        "configs/e13c_r1_transactional_sequence_aggregate.yaml",
        "docs/E13_R2_LEARNED_DISTRACTOR_PROTOCOL_KO.md",
        "docs/E13A_R1_RESULT_STATUS_AMENDMENT_KO.md",
        "docs/E13A_R2_LEARNED_DISTRACTOR_LOCK.json",
        "docs/E13A_R2_LEARNED_DISTRACTOR_RESULT_KO.md",
        "docs/E13B_R1_MAIN_LOCK.json",
        "docs/E13C_R1_OPERATIONAL_INCOMPLETE_FILTER_AMENDMENT_LOCK.json",
    }
    audit.add(
        "protocol_lock_set",
        "PASS" if set(protocols) == expected_protocols else "FAIL",
        "Freeze declares the complete E13b/c protocol and amendment lock set.",
        expected=sorted(expected_protocols),
        observed=sorted(protocols),
    )
    for relative, digest in protocols.items():
        candidate = repo_root / str(relative)
        contained = _is_within(candidate, repo_root)
        audit.add(
            f"protocol_path:{relative}",
            "PASS" if contained else "FAIL",
            "Protocol/config path remains inside the repository.",
            path=candidate,
        )
        if contained:
            _check_file_hash(
                audit,
                name=f"protocol_hash:{relative}",
                path=candidate,
                expected=digest,
            )
    result_path = _nested_file_record(
        audit,
        record=result_record,
        name="result_document_hash",
    )
    audit.add(
        "result_document_path",
        "PASS"
        if result_path is not None and result_path.resolve() == result_doc.resolve()
        else "FAIL",
        "Frozen result document path is the registered E13b/c result.",
        expected=str(result_doc.resolve()),
        observed=None if result_path is None else str(result_path.resolve()),
    )
    _check_document_tokens(
        audit,
        result_doc,
        (
            "e13b_r1_execution_status: PASS (10 / 10 runs)",
            "e13c_r1_claim_status: SUPPORTED",
            "20260727T214126.954177Z",
            "E13a-R2",
            "learned-distractor R1 pipeline의 유일한 GO dependency",
        ),
    )
    return audit


def audit_e14_freeze(*, repo_root: Path, artifact_root: Path) -> ExperimentAudit:
    freeze_path = artifact_root / "E14_POSTCORE_ARTIFACT_FREEZE_V1.json"
    result_doc = repo_root / "docs/E14_PLAN_CONTINUATION_RESULT_KO.md"
    audit = ExperimentAudit(
        name="E14",
        freeze_path=str(freeze_path),
        result_doc=str(result_doc),
    )
    if not freeze_path.is_file():
        audit.not_complete("E14 freeze has not been created yet.", path=freeze_path)
        return audit
    try:
        freeze = _json_object(freeze_path)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        audit.add(
            "freeze_json",
            "FAIL",
            f"Could not load E14 freeze: {type(error).__name__}: {error}",
            path=freeze_path,
        )
        return audit
    run_dir_value = freeze.get("run_dir")
    hashes = freeze.get("hashes")
    dependency = freeze.get("dependency")
    selected = freeze.get("selected_checkpoints")
    if not (
        isinstance(run_dir_value, str)
        and isinstance(hashes, dict)
        and isinstance(dependency, dict)
        and isinstance(selected, list)
    ):
        audit.add(
            "freeze_schema",
            "FAIL",
            "E14 freeze lacks run, hashes, dependency, or selected checkpoints.",
            path=freeze_path,
        )
        return audit
    run_dir = Path(run_dir_value)
    report_path = run_dir / "report.json"
    manifest_path = run_dir / "run_manifest.json"
    try:
        report = _json_object(report_path)
        manifest = _json_object(manifest_path)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        audit.add(
            "run_json",
            "FAIL",
            f"Could not load E14 run JSON: {type(error).__name__}: {error}",
            path=run_dir,
        )
        return audit
    _check_run_identity(
        audit,
        artifact_root=artifact_root,
        freeze=freeze,
        run_dir=run_dir,
        report=report,
        manifest=manifest,
    )
    audit.claim_status = str(freeze.get("claim_status"))
    schema_expected = {
        "schema_version": 1,
        "execution_status": "PASS",
        "claim_status": "SUPPORTED_STRUCTURED_PROXY_ONLY",
        "evidence_tier": "CONTROLLED_REFERENCE",
        "scientific_evidence": False,
        "proxy_scope": "STRUCTURED_SYNTHETIC_ENTITY_VALUE",
        "immutable": True,
    }
    schema_observed = {key: freeze.get(key) for key in schema_expected}
    audit.add(
        "freeze_status_schema",
        "PASS" if schema_observed == schema_expected else "FAIL",
        "E14 freeze preserves execution, proxy-only claim, and evidence boundary.",
        path=freeze_path,
        expected=schema_expected,
        observed=schema_observed,
    )
    report_supported = _claim_gate(report).get("supported")
    audit.add(
        "claim_disposition",
        "PASS" if report_supported is True else "FAIL",
        "Structured-proxy claim support is separate from execution status.",
        path=report_path,
        expected=True,
        observed=report_supported,
    )
    alias_paths = {
        "source_experiment": repo_root / "experiments/e14_plan_continuation.py",
        "source_config": repo_root / "configs/e14_plan_continuation.yaml",
        "prospective_identifiability_lock": (
            repo_root / "docs/E14_PROSPECTIVE_IDENTIFIABILITY_REPAIR_LOCK_KO.md"
        ),
        "result_markdown": result_doc,
    }
    expected_hash_names = {
        "config.resolved.yaml",
        "environment.json",
        "plan_continuation_metrics.jsonl",
        "report.json",
        "run_manifest.json",
        "sealed_checkpoint_provenance.jsonl",
        *alias_paths,
    }
    audit.add(
        "freeze_hash_set",
        "PASS" if set(hashes) == expected_hash_names else "FAIL",
        "E14 freeze declares every run, source, protocol, and result artifact.",
        path=freeze_path,
        expected=sorted(expected_hash_names),
        observed=sorted(hashes),
    )
    for name, expected_sha in hashes.items():
        path = alias_paths.get(str(name), run_dir / str(name))
        _check_file_hash(
            audit,
            name=f"hash:{name}",
            path=path,
            expected=expected_sha,
        )
    _check_document_tokens(
        audit,
        result_doc,
        (
            "20260727T214143.455051Z",
            "등록 claim gate | `SUPPORTED`",
            "STRUCTURED_SYNTHETIC_ENTITY_VALUE",
            "20260727T214126.954177Z",
            "Independent plan semantics | 평가하지 않음",
        ),
    )
    dependency_run_value = dependency.get("run_dir")
    if not isinstance(dependency_run_value, str):
        audit.add(
            "e13c_dependency",
            "FAIL",
            "E14 freeze has no E13c dependency run.",
            path=freeze_path,
        )
        return audit
    dependency_run = Path(dependency_run_value)
    dependency_paths = {
        "report_sha256": dependency_run / "report.json",
        "run_manifest_sha256": dependency_run / "run_manifest.json",
        "source_run_provenance_sha256": (
            dependency_run / "source_run_provenance.jsonl"
        ),
    }
    for digest_name, path in dependency_paths.items():
        _check_file_hash(
            audit,
            name=f"e13c_dependency:{digest_name}",
            path=path,
            expected=dependency.get(digest_name),
        )
    try:
        dependency_report = _json_object(dependency_paths["report_sha256"])
        dependency_manifest = _json_object(
            dependency_paths["run_manifest_sha256"]
        )
        dependency_rows = _jsonl_objects(
            dependency_paths["source_run_provenance_sha256"]
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        audit.add(
            "e13c_dependency_json",
            "FAIL",
            f"Could not load E13c dependency: {type(error).__name__}: {error}",
            path=dependency_run,
        )
        return audit
    dependency_identity = {
        "experiment_id": dependency.get("experiment_id"),
        "run_id": dependency.get("run_id"),
        "execution_status": dependency.get("execution_status"),
        "claim_status": dependency.get("claim_status"),
        "manifest_experiment_id": dependency_manifest.get("experiment_id"),
        "manifest_run_id": dependency_manifest.get("run_id"),
        "manifest_run_mode": dependency_manifest.get("run_mode"),
        "manifest_report_sha256": dependency_manifest.get("report_sha256"),
        "report_status": _execution_status(dependency_report),
        "report_supported": _claim_gate(dependency_report).get("supported"),
    }
    dependency_expected = {
        "experiment_id": "e13c_r1_transactional_sequence_aggregate",
        "run_id": dependency_run.name,
        "execution_status": "PASS",
        "claim_status": "SUPPORTED",
        "manifest_experiment_id": "e13c_r1_transactional_sequence_aggregate",
        "manifest_run_id": dependency_run.name,
        "manifest_run_mode": "MAIN",
        "manifest_report_sha256": dependency.get("report_sha256"),
        "report_status": "PASS",
        "report_supported": True,
    }
    audit.add(
        "e13c_dependency_identity",
        "PASS" if dependency_identity == dependency_expected else "FAIL",
        "E14 depends on one completed MAIN/PASS/SUPPORTED E13c-R1 run.",
        path=dependency_run,
        expected=dependency_expected,
        observed=dependency_identity,
    )
    e13bc_freeze_path = (
        artifact_root / "E13BC_R1_POSTCORE_ARTIFACT_FREEZE_V1.json"
    )
    try:
        e13bc_freeze = _json_object(e13bc_freeze_path)
        e13bc_aggregate = e13bc_freeze["aggregate"]
        e13bc_files = e13bc_aggregate["files"]
        e13bc_dependency_projection = {
            "execution_status": e13bc_freeze.get("execution_status"),
            "claim_status": e13bc_freeze.get("claim_status"),
            "run_id": e13bc_aggregate.get("run_id"),
            "run_dir": e13bc_aggregate.get("run_dir"),
            "report_sha256": e13bc_files["report.json"].get("sha256"),
            "run_manifest_sha256": e13bc_files["run_manifest.json"].get(
                "sha256"
            ),
            "source_run_provenance_sha256": e13bc_files[
                "source_run_provenance.jsonl"
            ].get("sha256"),
        }
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        json.JSONDecodeError,
    ) as error:
        audit.add(
            "e13bc_freeze_dependency_chain",
            "FAIL",
            f"Could not load E13b/c freeze: {type(error).__name__}: {error}",
            path=e13bc_freeze_path,
        )
    else:
        expected_e13bc_projection = {
            "execution_status": "PASS",
            "claim_status": "SUPPORTED",
            "run_id": dependency.get("run_id"),
            "run_dir": dependency.get("run_dir"),
            "report_sha256": dependency.get("report_sha256"),
            "run_manifest_sha256": dependency.get("run_manifest_sha256"),
            "source_run_provenance_sha256": dependency.get(
                "source_run_provenance_sha256"
            ),
        }
        audit.add(
            "e13bc_freeze_dependency_chain",
            (
                "PASS"
                if e13bc_dependency_projection == expected_e13bc_projection
                else "FAIL"
            ),
            "E14 dependency equals the immutable E13b/c aggregate freeze.",
            path=e13bc_freeze_path,
            expected=expected_e13bc_projection,
            observed=e13bc_dependency_projection,
        )
    report_dependency = report.get("dependency")
    report_dependency_ok = bool(
        isinstance(report_dependency, dict)
        and report_dependency.get("e13c_run_dir") == str(dependency_run.resolve())
        and report_dependency.get("e13c_report_sha256")
        == dependency.get("report_sha256")
        and report_dependency.get("e13c_manifest_sha256")
        == dependency.get("run_manifest_sha256")
        and report_dependency.get("e13c_source_provenance_sha256")
        == dependency.get("source_run_provenance_sha256")
    )
    audit.add(
        "e14_report_dependency_chain",
        "PASS" if report_dependency_ok else "FAIL",
        "E14 report cites the same E13c artifacts as its freeze.",
        path=report_path,
        expected=dependency,
        observed=report_dependency,
    )
    indexed_e13c = _audit_source_rows(
        audit,
        rows=dependency_rows,
        artifact_root=artifact_root,
        check_name="e13c_source",
    )
    selected_dicts = [row for row in selected if isinstance(row, dict)]
    indexed_selected = _audit_source_rows(
        audit,
        rows=selected_dicts,
        artifact_root=artifact_root,
        check_name="e14_selected",
    )
    expected_selected_keys = {
        (seed, "dual") for seed in (101, 211, 307, 401, 503)
    }
    audit.add(
        "e14_selected_checkpoint_set",
        "PASS" if set(indexed_selected) == expected_selected_keys else "FAIL",
        "E14 selects exactly five dual checkpoints, one per registered seed.",
        path=freeze_path,
        expected=sorted(expected_selected_keys),
        observed=sorted(indexed_selected),
    )
    chain_mismatches: list[dict[str, Any]] = []
    for key, selected_row in indexed_selected.items():
        source_row = indexed_e13c.get(key)
        if source_row is None:
            chain_mismatches.append({"key": key, "reason": "missing_from_e13c"})
            continue
        comparisons = {
            "checkpoint_sha256": (
                selected_row.get("checkpoint_sha256")
                == source_row.get("checkpoint_sha256")
            ),
            "source_report_sha256": (
                selected_row.get("source_report_sha256")
                == source_row.get("report_sha256")
            ),
            "source_metrics_sha256": (
                selected_row.get("source_metrics_sha256")
                == source_row.get("metrics_sha256")
            ),
            "source_manifest_sha256": (
                selected_row.get("source_manifest_sha256")
                == source_row.get("run_manifest_sha256")
            ),
        }
        if not all(comparisons.values()):
            chain_mismatches.append({"key": key, "comparisons": comparisons})
    audit.add(
        "e13c_to_e14_checkpoint_chain",
        "PASS" if not chain_mismatches else "FAIL",
        "Every E14 checkpoint and source digest matches E13c sealed provenance.",
        path=dependency_paths["source_run_provenance_sha256"],
        expected=[],
        observed=chain_mismatches,
    )
    sealed_path = run_dir / "sealed_checkpoint_provenance.jsonl"
    try:
        sealed_rows = _jsonl_objects(sealed_path)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        audit.add(
            "e14_sealed_checkpoint_rows",
            "FAIL",
            f"Could not load E14 sealed rows: {type(error).__name__}: {error}",
            path=sealed_path,
        )
    else:
        sealed_index = {
            _source_row_identity(row): _source_provenance_projection(row)
            for row in sealed_rows
        }
        selected_projection = {
            key: _source_provenance_projection(row)
            for key, row in indexed_selected.items()
        }
        expected_rows = [
            selected_projection[key] for key in sorted(selected_projection)
        ]
        observed_rows = [sealed_index[key] for key in sorted(sealed_index)]
        audit.add(
            "e14_sealed_checkpoint_rows",
            "PASS" if sealed_index == selected_projection else "FAIL",
            "E14 freeze selected-checkpoint records equal the sealed JSONL rows.",
            path=sealed_path,
            expected=expected_rows,
            observed=observed_rows,
        )
    summary = freeze.get("summary")
    summary_ok = bool(
        isinstance(summary, dict)
        and summary.get("required_cells") == 60
        and summary.get("observed_cells") == 60
        and summary.get("passing_cells") == 60
        and summary.get("required_training_seeds") == 5
        and summary.get("positive_seed_directions") == 5
    )
    audit.add(
        "e14_summary_grid",
        "PASS" if summary_ok else "FAIL",
        "Freeze summary records the complete five-seed, sixty-cell grid.",
        path=freeze_path,
        observed=summary,
    )
    return audit


def audit_e16(*, repo_root: Path, artifact_root: Path) -> ExperimentAudit:
    experiment_root = artifact_root / "e16_core_evidence_freeze"
    pointer = experiment_root / "latest.json"
    result_doc = repo_root / "docs/E16_EVIDENCE_FREEZE_KO.md"
    audit = ExperimentAudit(
        name="E16",
        freeze_path=None,
        result_doc=str(result_doc),
    )
    if not pointer.exists():
        audit.not_complete("E16 latest pointer has not been created yet.", path=pointer)
        return audit
    try:
        run_dir = _resolve_latest_run(experiment_root)
        report = _json_object(run_dir / "report.json")
        registry = _json_object(run_dir / "evidence_registry.json")
        manifest = _json_object(run_dir / "run_manifest.json")
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        audit.add(
            "e16_load",
            "FAIL",
            f"Could not load E16 freeze: {type(error).__name__}: {error}",
            path=pointer,
        )
        return audit
    audit.freeze_path = str(run_dir / "evidence_registry.json")
    observed_execution = _execution_status(report)
    audit.execution_status = observed_execution
    audit.claim_status = (
        "CORE_REGISTRY_COMPLETE"
        if registry.get("core_registry_complete") is True
        else "CORE_REGISTRY_INCOMPLETE"
    )
    audit.add(
        "execution_status",
        "PASS" if observed_execution == "PASS" else "FAIL",
        "E16 execution status is distinct from every frozen claim disposition.",
        path=run_dir / "report.json",
        expected="PASS",
        observed=observed_execution,
    )
    report_complete = _claim_gate(report).get("core_registry_complete")
    registry_complete = registry.get("core_registry_complete")
    audit.add(
        "core_registry_complete",
        "PASS" if report_complete is registry_complete else "FAIL",
        "Report and evidence registry agree on registry completeness.",
        expected=registry_complete,
        observed=report_complete,
    )
    required_outputs = (
        "report.json",
        "run_manifest.json",
        "evidence_registry.json",
        "results_macros.tex",
    )
    for required in required_outputs:
        path = run_dir / required
        audit.add(
            f"output:{required}",
            "PASS" if path.is_file() else "FAIL",
            "Required E16 output exists.",
            path=path,
        )

    evidence = registry.get("evidence")
    manifest_config = manifest.get("config")
    contract = (
        manifest_config.get("evidence")
        if isinstance(manifest_config, dict)
        else None
    )
    if not isinstance(evidence, dict) or not isinstance(contract, dict):
        audit.add(
            "registry_schema",
            "FAIL",
            "E16 registry or frozen input contract is missing.",
            path=run_dir,
        )
        return audit
    audit.add(
        "claim_set",
        "PASS" if set(evidence) == set(contract) else "FAIL",
        "Registry claim set matches the input evidence contract.",
        expected=sorted(contract),
        observed=sorted(evidence),
    )
    for claim_name, item in evidence.items():
        if not isinstance(item, dict):
            audit.add(
                f"{claim_name}:schema",
                "FAIL",
                "Evidence item is not an object.",
                path=run_dir / "evidence_registry.json",
            )
            continue
        claim_contract = contract.get(claim_name)
        if not isinstance(claim_contract, dict):
            continue
        disposition = item.get("claim_disposition")
        audit.add(
            f"{claim_name}:claim_disposition",
            "PASS" if disposition == claim_contract.get("claim_disposition") else "FAIL",
            "Claim disposition matches the frozen contract independently of execution.",
            expected=claim_contract.get("claim_disposition"),
            observed=disposition,
        )
        audit.add(
            f"{claim_name}:identity",
            (
                "PASS"
                if item.get("experiment_id") == claim_contract.get("experiment_id")
                and item.get("run_id") == claim_contract.get("run_id")
                else "FAIL"
            ),
            "Frozen experiment and run identities match the contract.",
            expected={
                "experiment_id": claim_contract.get("experiment_id"),
                "run_id": claim_contract.get("run_id"),
            },
            observed={
                "experiment_id": item.get("experiment_id"),
                "run_id": item.get("run_id"),
            },
        )
        files = item.get("files")
        contract_files = claim_contract.get("files")
        if not isinstance(files, dict) or not isinstance(contract_files, dict):
            audit.add(
                f"{claim_name}:files",
                "FAIL",
                "Evidence files or contract files are missing.",
                path=run_dir / "evidence_registry.json",
            )
            continue
        for relative, expected_sha in contract_files.items():
            record = files.get(relative)
            if not isinstance(record, dict) or not isinstance(record.get("path"), str):
                audit.add(
                    f"{claim_name}:file:{relative}",
                    "FAIL",
                    "Registry file record is missing.",
                    path=run_dir / "evidence_registry.json",
                )
                continue
            declared_sha = record.get("sha256")
            audit.add(
                f"{claim_name}:contract_hash:{relative}",
                "PASS" if declared_sha == expected_sha else "FAIL",
                "Registry digest matches the E16 input contract.",
                expected=expected_sha,
                observed=declared_sha,
            )
            _check_file_hash(
                audit,
                name=f"{claim_name}:file:{relative}",
                path=Path(record["path"]),
                expected=declared_sha,
            )
        anchors = item.get("anchors", [])
        contract_anchors = claim_contract.get("anchors", [])
        if not isinstance(anchors, list) or not isinstance(contract_anchors, list):
            audit.add(
                f"{claim_name}:anchors",
                "FAIL",
                "Registry or contract anchors are invalid.",
                path=run_dir / "evidence_registry.json",
            )
            continue
        audit.add(
            f"{claim_name}:anchor_count",
            "PASS" if len(anchors) == len(contract_anchors) else "FAIL",
            "Registry anchor count matches the E16 contract.",
            expected=len(contract_anchors),
            observed=len(anchors),
        )
        for index, (record, expected_record) in enumerate(
            zip(anchors, contract_anchors, strict=False)
        ):
            if not isinstance(record, dict) or not isinstance(expected_record, dict):
                audit.add(
                    f"{claim_name}:anchor:{index}",
                    "FAIL",
                    "Anchor record is invalid.",
                    path=run_dir / "evidence_registry.json",
                )
                continue
            _check_file_hash(
                audit,
                name=f"{claim_name}:anchor:{index}",
                path=Path(str(record.get("path"))),
                expected=expected_record.get("sha256"),
            )
        report_record = files.get("report.json")
        if isinstance(report_record, dict) and isinstance(report_record.get("path"), str):
            try:
                source_report = _json_object(Path(report_record["path"]))
                source_execution = _execution_status(source_report)
                expected_execution = item.get("execution_status")
                audit.add(
                    f"{claim_name}:execution_status",
                    "PASS" if source_execution == expected_execution else "FAIL",
                    "Source execution status is checked separately from claim disposition.",
                    path=Path(report_record["path"]),
                    expected=expected_execution,
                    observed=source_execution,
                )
                for dotted_path, expected_value in claim_contract.get(
                    "expected_report_fields", {}
                ).items():
                    try:
                        observed_value = _nested_value(source_report, dotted_path)
                    except KeyError:
                        observed_value = None
                    audit.add(
                        f"{claim_name}:report_field:{dotted_path}",
                        "PASS" if observed_value == expected_value else "FAIL",
                        "Source report field matches the frozen E16 contract.",
                        path=Path(report_record["path"]),
                        expected=expected_value,
                        observed=observed_value,
                    )
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
                audit.add(
                    f"{claim_name}:source_report",
                    "FAIL",
                    f"Could not load source report: {type(error).__name__}: {error}",
                    path=Path(report_record["path"]),
                )
    _check_document_tokens(
        audit,
        result_doc,
        ("evidence_registry.json", "results_macros.tex", "report.json"),
    )
    audit.add(
        "result_document_seal",
        "WARN",
        "E16 guide is not itself declared by a freeze digest.",
        path=result_doc,
        observed=file_sha256(result_doc) if result_doc.is_file() else None,
    )
    return audit


def audit_postcore_artifacts(
    *,
    repo_root: Path,
    artifact_root: Path,
) -> dict[str, Any]:
    audits = [
        audit_flat_freeze(repo_root=repo_root, artifact_root=artifact_root, spec=spec)
        for spec in FLAT_SPECS
    ]
    audits.insert(2, audit_e11(repo_root=repo_root, artifact_root=artifact_root))
    audits.append(audit_e13bc_freeze(repo_root=repo_root, artifact_root=artifact_root))
    audits.append(audit_e13b_live(repo_root=repo_root, artifact_root=artifact_root))
    audits.append(audit_e14_freeze(repo_root=repo_root, artifact_root=artifact_root))
    audits.append(audit_e16(repo_root=repo_root, artifact_root=artifact_root))
    counts = {
        status: sum(audit.status == status for audit in audits)
        for status in ("PASS", "WARN", "FAIL", "NOT_COMPLETE")
    }
    return {
        "schema_version": 1,
        "mode": "READ_ONLY_AUDIT",
        "repo_root": str(repo_root.resolve()),
        "artifact_root": str(artifact_root.resolve()),
        "overall_ok": counts["FAIL"] == 0,
        "summary": counts,
        "experiments": [asdict(audit) for audit in audits],
    }


def _print_text(result: dict[str, Any]) -> None:
    print("CATENA post-core artifact audit (read-only)")
    print(f"Repository: {result['repo_root']}")
    print(f"Artifacts:  {result['artifact_root']}")
    print()
    print(f"{'Experiment':14} {'Audit':14} {'Execution':26} Claim")
    print("-" * 92)
    for experiment in result["experiments"]:
        execution = experiment.get("execution_status") or "-"
        claim = experiment.get("claim_status") or "-"
        print(
            f"{experiment['name']:14} {experiment['status']:14} "
            f"{execution[:26]:26} {claim}"
        )
        for check in experiment["checks"]:
            if check["status"] in {"FAIL", "WARN"}:
                print(f"  [{check['status']}] {check['name']}: {check['detail']}")
                if check.get("path"):
                    print(f"    path: {check['path']}")
                if check.get("expected") is not None or check.get("observed") is not None:
                    print(f"    expected={check.get('expected')!r}")
                    print(f"    observed={check.get('observed')!r}")
    print()
    summary = result["summary"]
    print(
        "Summary: "
        f"PASS={summary['PASS']} WARN={summary['WARN']} "
        f"FAIL={summary['FAIL']} NOT_COMPLETE={summary['NOT_COMPLETE']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only consistency audit for CATENA post-core freezes and result docs."
    )
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[1]),
    )
    parser.add_argument(
        "--artifact-root",
        default="/data/minjun_dev/CATENA/artifacts",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = audit_postcore_artifacts(
        repo_root=Path(args.repo_root),
        artifact_root=Path(args.artifact_root),
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        _print_text(result)
    raise SystemExit(0 if result["overall_ok"] else 1)


if __name__ == "__main__":
    main()
