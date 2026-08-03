#!/usr/bin/env python3
"""Finalize the immutable E26 Final terminal admission failure.

This reporting command cannot launch CUDA, construct data, or reinterpret a
scientific outcome.  It authenticates the prerequisite and blocked receipts,
writes one terminal artifact namespace, and refuses every overwrite.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from catena.core.provenance_v61 import (
    read_json_object_strict,
    sha256_canonical_json,
    sha256_file,
    write_json_strict,
)

EXPERIMENT_ID: Final = "E26_FINAL_GDN2_1P3B_TRANSACTIONAL_TRANSFER"
DISPOSITION: Final = "BLOCKED_OFFICIAL_RUNTIME_NAMESPACE_PROVENANCE_VALIDATION"
INITIAL_ERROR: Final = "Official runtime source binding failed: only_gate_source_modified"
R1_ERROR: Final = "Refusing preloaded non-official Python module: lit_gpt.gdn2_ops"
STAGE3D_SHA256: Final = (
    "4c4528bf35052423896b29dbc12944e9ad5df3ec2f87410a9688417297a42650"
)
FROZEN_AGGREGATE: Final = (
    "46a779594c760fc1103833810bcc5227d7ca4526111911d7856cdfe2ff2d202b"
)


class E26FinalizationError(RuntimeError):
    """Raised when terminal evidence differs from the registered failure."""


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(root), *arguments),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise E26FinalizationError(result.stderr.strip())
    return result.stdout.strip()


def _canonical_receipt(path: Path) -> dict[str, Any]:
    payload = read_json_object_strict(path)
    claimed = payload.get("receipt_sha256")
    body = dict(payload)
    body.pop("receipt_sha256", None)
    if claimed != sha256_canonical_json(body):
        raise E26FinalizationError(f"Receipt canonical SHA changed: {path}")
    return payload


def _binding(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path.resolve(strict=True)),
        "bytes": path.stat().st_size,
        "file_sha256": sha256_file(path),
        "receipt_sha256": payload.get("receipt_sha256"),
    }


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite terminal artifact: {path}")
    write_json_strict(path, payload)


def _status_stub(stage: str, reason: str) -> dict[str, Any]:
    return {
        "schema_version": "catena-e26-final-not-run-v1",
        "experiment_id": EXPERIMENT_ID,
        "stage": stage,
        "status": "NOT_RUN",
        "reason": reason,
        "scientific_evidence": False,
        "scientific_main_started": False,
    }


def _manifest_bytes(root: Path) -> bytes:
    rows: list[bytes] = []
    for path in sorted(row for row in root.rglob("*") if row.is_file()):
        if path.name == "artifact_manifest.sha256":
            continue
        relative = path.relative_to(root).as_posix()
        rows.append(f"{sha256_file(path)}  {relative}\n".encode())
    return b"".join(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--frozen-reaudit", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--checkpoint-receipt", type=Path, required=True)
    parser.add_argument("--runtime-dependency-receipt", type=Path, required=True)
    parser.add_argument("--initial-dual", type=Path, required=True)
    parser.add_argument("--initial-tied", type=Path, required=True)
    parser.add_argument("--r1-dual", type=Path, required=True)
    parser.add_argument("--r1-tied", type=Path, required=True)
    args = parser.parse_args()

    root = args.repo_root.resolve(strict=True)
    run_dir = args.run_dir.resolve(strict=True)
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise E26FinalizationError("Final source worktree must be clean")

    protocol = _canonical_receipt(args.protocol_lock)
    source = _canonical_receipt(args.source_receipt)
    checkpoint = _canonical_receipt(args.checkpoint_receipt)
    dependency = _canonical_receipt(args.runtime_dependency_receipt)
    initial_dual = _canonical_receipt(args.initial_dual)
    initial_tied = _canonical_receipt(args.initial_tied)
    r1_dual = _canonical_receipt(args.r1_dual)
    r1_tied = _canonical_receipt(args.r1_tied)
    frozen = _canonical_receipt(args.frozen_reaudit)

    if (
        protocol.get("scientific_main_started") is not False
        or source.get("passed") is not True
        or checkpoint.get("passed") is not True
        or dependency.get("passed") is not True
        or frozen.get("passed") is not True
    ):
        raise E26FinalizationError("A prerequisite receipt is not in its registered state")
    frozen_artifacts = frozen.get("frozen_artifacts")
    if not isinstance(frozen_artifacts, Mapping) or (
        frozen_artifacts.get("observed_file_count") != 2062
        or frozen_artifacts.get("observed_aggregate_sha256") != FROZEN_AGGREGATE
    ):
        raise E26FinalizationError("Frozen E00-E25 re-audit changed")

    attempts = (initial_dual, initial_tied, r1_dual, r1_tied)
    expected_variants = (
        "dual_gdn2",
        "projected_tied_gdn2",
        "dual_gdn2",
        "projected_tied_gdn2",
    )
    for index, (row, variant) in enumerate(zip(attempts, expected_variants, strict=True)):
        expected_error = INITIAL_ERROR if index < 2 else R1_ERROR
        if (
            row.get("passed") is not False
            or row.get("scientific_e26a_started") is not False
            or row.get("overall_disposition") != "BLOCKED_OFFICIAL_RUNTIME"
            or row.get("variant") != variant
            or row.get("error") != expected_error
        ):
            raise E26FinalizationError("Official-runtime blocked receipt changed")

    bindings = {
        "protocol_lock": _binding(args.protocol_lock, protocol),
        "frozen_e00_e25_reaudit": _binding(args.frozen_reaudit, frozen),
        "external_source": _binding(args.source_receipt, source),
        "checkpoint": _binding(args.checkpoint_receipt, checkpoint),
        "runtime_dependency": _binding(args.runtime_dependency_receipt, dependency),
        "initial_dual": _binding(args.initial_dual, initial_dual),
        "initial_tied": _binding(args.initial_tied, initial_tied),
        "r1_dual": _binding(args.r1_dual, r1_dual),
        "r1_tied": _binding(args.r1_tied, r1_tied),
    }
    source_lock = {
        "schema_version": "catena-e26-final-terminal-source-lock-v1",
        "experiment_id": EXPERIMENT_ID,
        "source_commit": _git(root, "rev-parse", "HEAD"),
        "source_branch": _git(root, "branch", "--show-current"),
        "source_dirty": False,
        "runtime_audit_source": {
            "path": str((root / "tools/audit_e26_final_official_runtime.py").resolve()),
            "sha256": sha256_file(root / "tools/audit_e26_final_official_runtime.py"),
        },
        "stage3d_report_sha256_preserved": STAGE3D_SHA256,
        "bindings": bindings,
        "scientific_main_started": False,
    }
    source_lock["receipt_sha256"] = sha256_canonical_json(source_lock)
    _write(run_dir / "source_lock.json", source_lock)

    checkpoint_link = {
        "schema_version": "catena-e26-final-checkpoint-binding-v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS_PREREQUISITE_ONLY",
        "community_not_official_nvidia": True,
        "binding": bindings["checkpoint"],
        "scientific_main_started": False,
    }
    checkpoint_link["receipt_sha256"] = sha256_canonical_json(checkpoint_link)
    _write(run_dir / "checkpoint_audit.json", checkpoint_link)

    not_run_reason = "BLOCKED_BEFORE_SPEED_PREFLIGHT_AT_OFFICIAL_RUNTIME_R1"
    _write(
        run_dir / "data_manifest.json",
        _status_stub("DATA_MATERIALIZATION", not_run_reason),
    )
    _write(
        run_dir / "speed_preflight.json",
        _status_stub("SPEED_PREFLIGHT", not_run_reason),
    )
    _write(
        run_dir / "bridge_report.json",
        _status_stub("COMMON_FUNCTION_BRIDGE", not_run_reason),
    )
    _write(run_dir / "evaluation_status.json", _status_stub("FROZEN_EVALUATION", not_run_reason))
    _write(
        run_dir / "mechanism_status.json",
        _status_stub("MECHANISM_INTERVENTION", not_run_reason),
    )

    claim = {
        "schema_version": "catena-e26-final-claim-disposition-v1",
        "experiment_id": EXPERIMENT_ID,
        "execution_status": "BLOCKED_ADMISSION",
        "scientific_disposition": DISPOSITION,
        "scientific_evidence": False,
        "claim_eligible": False,
        "lm_transfer_claim_open": False,
        "allowed_claims": [
            (
                "Pinned source/checkpoint prerequisite provenance passed before "
                "runtime admission failed."
            ),
            "E26 Final produced no pretrained-LM effect estimate.",
        ],
        "prohibited_claims": [
            "pretrained recurrent LM transfer",
            "official GDN2 superiority",
            "transaction effect or null effect",
            "gate mechanism mediation",
            "quality, locality, throughput, or production superiority",
        ],
    }
    claim["receipt_sha256"] = sha256_canonical_json(claim)
    _write(run_dir / "claim_disposition.json", claim)

    report = {
        "schema_version": "catena-e26-final-terminal-report-v1",
        "experiment_id": EXPERIMENT_ID,
        "execution_status": "BLOCKED_ADMISSION",
        "scientific_disposition": DISPOSITION,
        "failure_stage": "OFFICIAL_RUNTIME_R1",
        "failure_classification": "ADMISSION_TOOL_NAMESPACE_PROVENANCE_VALIDATION",
        "hypothesis_evaluated": False,
        "scientific_evidence": False,
        "claim_eligible": False,
        "scientific_main_started": False,
        "gpu_kernel_audit_completed": False,
        "speed_preflight_started": False,
        "common_bridge_started": False,
        "main_training_started": False,
        "main_test_opened": False,
        "initial_attempt_error": INITIAL_ERROR,
        "r1_attempt_error": R1_ERROR,
        "additional_retry_allowed": False,
        "threshold_or_protocol_changed": False,
        "stage3c_disposition_preserved": (
            "BLOCKED_NUMERICAL_GRADIENT_ACCUMULATION_LAYOUT_INVARIANCE"
        ),
        "stage3d_disposition_preserved": "STAGE3D_BLOCKED_FIXED_LAYOUT_NUMERICAL_INSTABILITY",
        "stage3d_report_sha256": STAGE3D_SHA256,
        "frozen_e00_e25": {
            "file_count": 2062,
            "aggregate_sha256": FROZEN_AGGREGATE,
            "passed": True,
        },
        "bindings": bindings,
        "claim_disposition_sha256": claim["receipt_sha256"],
    }
    report["report_sha256"] = sha256_canonical_json(report)
    _write(run_dir / "report.json", report)

    manifest = {
        "schema_version": "catena-e26-final-run-manifest-v1",
        "experiment_id": EXPERIMENT_ID,
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "run_mode": "ADMISSION_ONLY",
        "source_commit": source_lock["source_commit"],
        "execution_status": "BLOCKED_ADMISSION",
        "scientific_evidence": False,
        "scientific_main_started": False,
        "report_sha256": report["report_sha256"],
        "claim_disposition_sha256": claim["receipt_sha256"],
    }
    manifest["receipt_sha256"] = sha256_canonical_json(manifest)
    _write(run_dir / "run_manifest.json", manifest)

    summary = (
        "# E26 Final 결과 요약\n\n"
        "- Execution: `BLOCKED_ADMISSION`\n"
        f"- Disposition: `{DISPOSITION}`\n"
        "- Scientific main / bridge / speed preflight: **NOT STARTED**\n"
        "- First attempt: source-cleanliness failure from validation bytecode cache\n"
        "- Fresh R1: pinned `lit_gpt.gdn2_ops` namespace rejected by module-origin validator\n"
        "- Hypothesis status: **NOT EVALUATED**\n"
        "- E00--E25 frozen re-audit: 2,062 files PASS\n"
        "- Allowed claim: prerequisite provenance passed; no LM effect estimate exists\n"
        "- Forbidden claim: LM transfer, official superiority, mechanism, quality/locality, speed\n"
    )
    summary_path = run_dir / "RESULTS_SUMMARY_KO.md"
    if summary_path.exists():
        raise FileExistsError(summary_path)
    summary_path.write_text(summary, encoding="utf-8")

    manifest_bytes = _manifest_bytes(run_dir)
    manifest_path = run_dir / "artifact_manifest.sha256"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    manifest_path.write_bytes(manifest_bytes)
    manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()

    latest = run_dir.parent / "latest.json"
    if latest.exists() or latest.is_symlink():
        raise FileExistsError(f"Refusing to overwrite latest pointer: {latest}")
    latest_payload = {
        "experiment_id": EXPERIMENT_ID,
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "execution_status": "BLOCKED_ADMISSION",
        "scientific_disposition": DISPOSITION,
        "report_sha256": report["report_sha256"],
        "artifact_manifest_sha256": manifest_digest,
    }
    write_json_strict(latest, latest_payload)
    print(run_dir)
    print(report["report_sha256"])
    print(manifest_digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
