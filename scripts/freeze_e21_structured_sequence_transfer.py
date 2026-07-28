#!/usr/bin/env python3
"""Freeze E21 sources, invalid original aggregate, and valid R1 aggregate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catena.core.config import load_config
from experiments.e21_structured_sequence_localization_transfer import (
    _read_json_object,
    _validate_source_run,
    validate_e21_protocol_lock,
)
from experiments.e21b_r1_structured_sequence_localization_aggregate import (
    _validate_repair_lock,
)

FREEZE_FILENAME = "E21_STRUCTURED_SEQUENCE_TRANSFER_FREEZE_V1.json"
SOURCE_EXPERIMENT_ID = "e21a_structured_sequence_localization_transfer"
ORIGINAL_AGGREGATE_ID = "e21b_structured_sequence_localization_aggregate"
R1_AGGREGATE_ID = "e21b_r1_structured_sequence_localization_aggregate"
ORIGINAL_DISPOSITION = "INCONCLUSIVE_GATE_IMPLEMENTATION"
RUN_ID_PATTERN = re.compile(r"\d{8}T\d{6}\.\d{6}Z")

SOURCE_CONFIG = "configs/e21_structured_sequence_localization_transfer.yaml"
R1_CONFIG = "configs/e21b_r1_structured_sequence_localization_aggregate.yaml"
SOURCE_LOCK = "docs/E21_STRUCTURED_SEQUENCE_LOCALIZATION_TRANSFER_LOCK.json"
R1_LOCK = "docs/E21B_R1_STRUCTURED_SEQUENCE_AGGREGATE_LOCK.json"

SOURCE_FILES = (
    "report.json",
    "run_manifest.json",
    "structured_sequence_transfer_metrics.jsonl",
    "RESULTS_SUMMARY_KO.md",
)
ORIGINAL_AGGREGATE_FILES = (
    "report.json",
    "run_manifest.json",
    "structured_sequence_paired_metrics.jsonl",
    "structured_sequence_seed_contrasts.jsonl",
    "source_run_provenance.jsonl",
    "RESULTS_SUMMARY_KO.md",
)
R1_AGGREGATE_FILES = (
    "report.json",
    "run_manifest.json",
    "structured_sequence_paired_metrics.jsonl",
    "structured_sequence_seed_contrasts_r1.jsonl",
    "source_run_provenance.jsonl",
    "RESULTS_SUMMARY_KO.md",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant: {value}")
        ),
    )
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(
                line,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON constant: {value}")
                ),
            )
            if not isinstance(payload, dict):
                raise TypeError(f"expected object at {path}:{line_number}")
            rows.append(payload)
    return rows


def _canonical_json_counter(rows: list[dict[str, Any]]) -> Counter[str]:
    return Counter(
        json.dumps(
            row,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for row in rows
    )


def _canonical_run_dir(
    run_dir: str | Path,
    *,
    artifact_root: Path,
    experiment_id: str,
) -> Path:
    candidate = Path(run_dir).resolve()
    if (
        candidate.parent != (artifact_root / experiment_id).resolve()
        or RUN_ID_PATTERN.fullmatch(candidate.name) is None
        or not candidate.is_dir()
        or candidate.is_symlink()
    ):
        raise RuntimeError(f"non-canonical {experiment_id} run directory: {candidate}")
    return candidate


def _validate_manifest(
    run_dir: Path,
    *,
    experiment_id: str,
) -> dict[str, Any]:
    report_path = run_dir / "report.json"
    manifest_path = run_dir / "run_manifest.json"
    for path in (report_path, manifest_path):
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"missing or unsafe E21 artifact: {path}")
    manifest = _read_json(manifest_path)
    fingerprint = manifest.get("source_fingerprint")
    if (
        manifest.get("schema_version") != 2
        or manifest.get("experiment_id") != experiment_id
        or manifest.get("run_id") != run_dir.name
        or manifest.get("run_mode") != "MAIN"
        or manifest.get("report_sha256") != _sha256(report_path)
        or not isinstance(fingerprint, dict)
        or not isinstance(fingerprint.get("files"), int)
        or not isinstance(fingerprint.get("sha256"), str)
        or len(fingerprint["sha256"]) != 64
        or manifest.get("source_fingerprint_phase") != "RUN_START"
    ):
        raise RuntimeError(f"E21 manifest/report contract failed: {run_dir}")
    return manifest


def _hash_inventory(run_dir: Path, filenames: tuple[str, ...]) -> dict[str, str]:
    result: dict[str, str] = {}
    for filename in filenames:
        path = run_dir / filename
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"missing or unsafe E21 frozen file: {path}")
        result[filename] = _sha256(path)
    return result


def _source_descriptor(
    run_dir: Path,
    *,
    seed: int,
    source_config: dict[str, Any],
    source_lock: dict[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows, provenance = _validate_source_run(
        run_dir,
        expected_seed=seed,
        expected_mode="MAIN",
        config=source_config,
        lock=source_lock,
        dry_run=False,
    )
    report = _read_json(run_dir / "report.json")
    manifest = _validate_manifest(run_dir, experiment_id=SOURCE_EXPERIMENT_ID)
    if (
        report.get("status") != "PASS"
        or report.get("run_mode") != "MAIN"
        or report.get("evidence_tier") != "CONTROLLED_REFERENCE"
        or report.get("scientific_evidence") is not False
        or report.get("claim_gate", {}).get("status") != "PENDING_AGGREGATE"
        or report.get("protocol", {}).get("h5_reopened") is not False
        or len(rows) != 768
    ):
        raise RuntimeError(f"invalid E21 source claim boundary: {run_dir}")
    checkpoint_hashes = report.get("artifacts", {}).get("checkpoint_hashes")
    if not isinstance(checkpoint_hashes, dict) or set(checkpoint_hashes) != {
        "base",
        "separate_address",
        "state_aware",
        "full",
    }:
        raise RuntimeError("E21 source checkpoint inventory is incomplete")
    source_summary = run_dir / "RESULTS_SUMMARY_KO.md"
    source_summary_lines = len(source_summary.read_text(encoding="utf-8").splitlines())
    if (
        source_summary_lines > 55
        or report.get("artifacts", {}).get("results_summary_ko", {}).get("line_count")
        != source_summary_lines
    ):
        raise RuntimeError("E21 source result summary contract failed")
    for variant, expected_hash in checkpoint_hashes.items():
        checkpoint = run_dir / "checkpoints" / f"seed{seed}_{variant}.pt"
        if (
            not checkpoint.is_file()
            or checkpoint.is_symlink()
            or _sha256(checkpoint) != expected_hash
        ):
            raise RuntimeError(f"E21 source checkpoint changed: {checkpoint}")
    descriptor = {
        "seed": seed,
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "execution_status": "PASS",
        "claim_status": "PENDING_AGGREGATE",
        "evidence_tier": "CONTROLLED_REFERENCE",
        "scientific_evidence": False,
        "row_count": len(rows),
        "hashes": _hash_inventory(run_dir, SOURCE_FILES),
        "checkpoint_hashes": checkpoint_hashes,
        "protocol_lock_sha256": source_lock["sha256"],
        "source_config_sha256": source_lock["config_sha256"],
        "manifest_source_fingerprint": manifest.get("source_fingerprint"),
        "claim_boundary": {
            "source_run_alone_claim_eligible": False,
            "h5_reopened": False,
            "semantic_claim_eligible": False,
            "novel_identifier_claim_eligible": False,
            "official_backend_claim_eligible": False,
        },
        "validated_provenance": provenance,
    }
    return descriptor, rows


def _provenance_by_seed(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        seed = int(row.get("seed", -1))
        if seed in result:
            raise RuntimeError("duplicate E21 aggregate source provenance seed")
        result[seed] = row
    return result


def _validate_aggregate_source_provenance(
    report: dict[str, Any],
    provenance_rows: list[dict[str, Any]],
    *,
    sources: list[dict[str, Any]],
) -> None:
    expected = {int(source["seed"]): source for source in sources}
    file_rows = _provenance_by_seed(provenance_rows)
    report_rows_raw = report.get("source_contract", {}).get("source_runs")
    if not isinstance(report_rows_raw, list):
        raise RuntimeError("E21 aggregate report source provenance is missing")
    report_rows = _provenance_by_seed(report_rows_raw)
    if set(file_rows) != set(expected) or set(report_rows) != set(expected):
        raise RuntimeError("E21 aggregate source seed provenance mismatch")
    for seed, source in expected.items():
        for candidate in (file_rows[seed], report_rows[seed]):
            if (
                Path(str(candidate.get("run_dir", ""))).resolve()
                != Path(str(source["run_dir"])).resolve()
                or candidate.get("report_sha256") != source["hashes"]["report.json"]
                or candidate.get("metrics_sha256")
                != source["hashes"]["structured_sequence_transfer_metrics.jsonl"]
                or candidate.get("results_summary_sha256")
                != source["hashes"]["RESULTS_SUMMARY_KO.md"]
                or candidate.get("checkpoint_hashes") != source["checkpoint_hashes"]
            ):
                raise RuntimeError(f"E21 aggregate source provenance changed for seed {seed}")


def _aggregate_descriptor(
    run_dir: Path,
    *,
    experiment_id: str,
    filenames: tuple[str, ...],
    metrics_filename: str,
    contrasts_filename: str,
    sources: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    source_lock: dict[str, str],
    repair_lock: dict[str, str] | None,
) -> dict[str, Any]:
    report = _read_json(run_dir / "report.json")
    manifest = _validate_manifest(run_dir, experiment_id=experiment_id)
    hashes = _hash_inventory(run_dir, filenames)
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, dict):
        raise RuntimeError("E21 aggregate artifact index is missing")
    if (
        artifacts.get("paired_metrics_sha256") != hashes[metrics_filename]
        or artifacts.get("seed_contrasts_sha256") != hashes[contrasts_filename]
        or artifacts.get("source_provenance_sha256") != hashes["source_run_provenance.jsonl"]
        or artifacts.get("results_summary_ko", {}).get("sha256") != hashes["RESULTS_SUMMARY_KO.md"]
    ):
        raise RuntimeError("E21 aggregate report/file hash contract failed")
    summary_lines = len(
        (run_dir / "RESULTS_SUMMARY_KO.md").read_text(encoding="utf-8").splitlines()
    )
    if (
        summary_lines > 55
        or artifacts.get("results_summary_ko", {}).get("line_count") != summary_lines
    ):
        raise RuntimeError("E21 aggregate result summary contract failed")
    paired_rows = _read_jsonl(run_dir / metrics_filename)
    contrast_rows = _read_jsonl(run_dir / contrasts_filename)
    provenance_rows = _read_jsonl(run_dir / "source_run_provenance.jsonl")
    if (
        len(paired_rows) != 3840
        or len(contrast_rows) != 5
        or len(provenance_rows) != 5
        or _canonical_json_counter(paired_rows) != _canonical_json_counter(source_rows)
    ):
        raise RuntimeError("E21 aggregate row/source contract failed")
    _validate_aggregate_source_provenance(
        report,
        provenance_rows,
        sources=sources,
    )
    if (
        report.get("status") != "PASS"
        or report.get("run_mode") != "MAIN"
        or report.get("evidence_tier") != "CONTROLLED_REFERENCE"
        or report.get("scientific_evidence") is not False
        or not isinstance(report.get("claim_gate", {}).get("supported"), bool)
    ):
        raise RuntimeError("E21 aggregate execution/evidence boundary is invalid")

    supported = bool(report["claim_gate"]["supported"])
    observed_status = "SUPPORTED" if supported else "NOT_SUPPORTED"
    if report["claim_gate"].get("status") != observed_status:
        raise RuntimeError("E21 aggregate claim status/boolean mismatch")
    source_contract = report.get("source_contract", {})
    if experiment_id == ORIGINAL_AGGREGATE_ID:
        if (
            source_contract.get("protocol_lock_sha256") != source_lock["sha256"]
            or source_contract.get("source_config_sha256") != source_lock["config_sha256"]
        ):
            raise RuntimeError("Original E21b source lock/config hash mismatch")
        disposition = ORIGINAL_DISPOSITION
        claim_eligible = False
    else:
        if repair_lock is None:
            raise RuntimeError("E21b-R1 repair lock was not supplied")
        if (
            report.get("original_e21b_disposition") != ORIGINAL_DISPOSITION
            or source_contract.get("source_protocol_lock_sha256") != source_lock["sha256"]
            or source_contract.get("repair_protocol_lock_sha256") != repair_lock["sha256"]
            or source_contract.get("repair_config_sha256") != repair_lock["config_sha256"]
            or report.get("summary", {}).get("supported") is not supported
            or report.get("summary", {}).get("repair", {}).get("original_e21b_disposition")
            != ORIGINAL_DISPOSITION
        ):
            raise RuntimeError("E21b-R1 repaired claim/lock boundary is invalid")
        disposition = observed_status
        claim_eligible = supported

    return {
        "experiment_id": experiment_id,
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "execution_status": "PASS",
        "observed_report_claim_status": observed_status,
        "frozen_disposition": disposition,
        "claim_eligible": claim_eligible,
        "evidence_tier": "CONTROLLED_REFERENCE",
        "scientific_evidence": False,
        "summary": report.get("summary"),
        "claim_gate": report.get("claim_gate"),
        "hashes": hashes,
        "manifest_source_fingerprint": manifest.get("source_fingerprint"),
    }


def _claim_boundary(r1_supported: bool) -> dict[str, bool]:
    return {
        "original_e21b_claim_eligible": False,
        "e21b_r1_controlled_structured_sequence_claim_eligible": r1_supported,
        "h5_semantic_factorization_claim_eligible": False,
        "semantic_or_natural_language_claim_eligible": False,
        "novel_identifier_generalization_claim_eligible": False,
        "pretrained_or_recurrent_lm_claim_eligible": False,
        "agent_or_planning_claim_eligible": False,
        "official_backend_claim_eligible": False,
        "runtime_superiority_claim_eligible": False,
    }


def build_freeze_payload(
    *,
    repo_root: str | Path,
    artifact_root: str | Path,
    source_runs: list[str | Path],
    original_aggregate_run: str | Path,
    r1_aggregate_run: str | Path,
    frozen_at_utc: str | None = None,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    artifacts = Path(artifact_root).resolve()
    source_config_path = root / SOURCE_CONFIG
    source_config = load_config(source_config_path)
    source_lock = validate_e21_protocol_lock(source_config_path)
    repair_lock = _validate_repair_lock(root / R1_CONFIG)
    required_seeds = [int(value) for value in source_config["seeds"]]
    if len(source_runs) != len(required_seeds):
        raise RuntimeError("E21 freeze requires exactly five explicit sources")

    descriptors: list[dict[str, Any]] = []
    all_source_rows: list[dict[str, Any]] = []
    observed_seeds: set[int] = set()
    for source_value in source_runs:
        source = _canonical_run_dir(
            source_value,
            artifact_root=artifacts,
            experiment_id=SOURCE_EXPERIMENT_ID,
        )
        report = _read_json_object(source / "report.json")
        seed = int(report.get("seed", -1))
        if seed not in required_seeds or seed in observed_seeds:
            raise RuntimeError("E21 freeze source seed is duplicate/unregistered")
        descriptor, rows = _source_descriptor(
            source,
            seed=seed,
            source_config=source_config,
            source_lock=source_lock,
        )
        descriptors.append(descriptor)
        all_source_rows.extend(rows)
        observed_seeds.add(seed)
    if observed_seeds != set(required_seeds):
        raise RuntimeError("E21 freeze exact source seed grid is incomplete")
    descriptors.sort(key=lambda row: int(row["seed"]))

    original_dir = _canonical_run_dir(
        original_aggregate_run,
        artifact_root=artifacts,
        experiment_id=ORIGINAL_AGGREGATE_ID,
    )
    r1_dir = _canonical_run_dir(
        r1_aggregate_run,
        artifact_root=artifacts,
        experiment_id=R1_AGGREGATE_ID,
    )
    original = _aggregate_descriptor(
        original_dir,
        experiment_id=ORIGINAL_AGGREGATE_ID,
        filenames=ORIGINAL_AGGREGATE_FILES,
        metrics_filename="structured_sequence_paired_metrics.jsonl",
        contrasts_filename="structured_sequence_seed_contrasts.jsonl",
        sources=descriptors,
        source_rows=all_source_rows,
        source_lock=source_lock,
        repair_lock=None,
    )
    repair = _aggregate_descriptor(
        r1_dir,
        experiment_id=R1_AGGREGATE_ID,
        filenames=R1_AGGREGATE_FILES,
        metrics_filename="structured_sequence_paired_metrics.jsonl",
        contrasts_filename="structured_sequence_seed_contrasts_r1.jsonl",
        sources=descriptors,
        source_rows=all_source_rows,
        source_lock=source_lock,
        repair_lock=repair_lock,
    )
    if repair["frozen_disposition"] not in {"SUPPORTED", "NOT_SUPPORTED"}:
        raise RuntimeError("E21b-R1 disposition is invalid")
    r1_supported = bool(repair["claim_eligible"])
    timestamp = frozen_at_utc or datetime.now(UTC).isoformat().replace(
        "+00:00",
        "Z",
    )
    return {
        "schema_version": 1,
        "frozen_at_utc": timestamp,
        "experiment_family": "E21",
        "source_experiment_id": SOURCE_EXPERIMENT_ID,
        "original_aggregate_experiment_id": ORIGINAL_AGGREGATE_ID,
        "repair_aggregate_experiment_id": R1_AGGREGATE_ID,
        "source_runs": descriptors,
        "original_aggregate": original,
        "repair_aggregate": repair,
        "claim_status": repair["frozen_disposition"],
        "evidence_tier": "CONTROLLED_REFERENCE",
        "scientific_evidence": False,
        "locks_and_configs": {
            "source_protocol_lock": {
                "path": source_lock["path"],
                "sha256": source_lock["sha256"],
            },
            "repair_protocol_lock": {
                "path": repair_lock["path"],
                "sha256": repair_lock["sha256"],
            },
            "source_config": {
                "path": str(source_config_path.resolve()),
                "sha256": source_lock["config_sha256"],
            },
            "repair_config": {
                "path": str((root / R1_CONFIG).resolve()),
                "sha256": repair_lock["config_sha256"],
            },
        },
        "claim_boundary": _claim_boundary(r1_supported),
        "immutable": True,
    }


def validate_freeze(
    payload: dict[str, Any],
    *,
    repo_root: str | Path,
    artifact_root: str | Path,
) -> None:
    required_identity = {
        "schema_version": 1,
        "experiment_family": "E21",
        "source_experiment_id": SOURCE_EXPERIMENT_ID,
        "original_aggregate_experiment_id": ORIGINAL_AGGREGATE_ID,
        "repair_aggregate_experiment_id": R1_AGGREGATE_ID,
        "evidence_tier": "CONTROLLED_REFERENCE",
        "scientific_evidence": False,
        "immutable": True,
    }
    for key, expected in required_identity.items():
        if payload.get(key) != expected:
            raise RuntimeError(f"invalid E21 freeze field: {key}")
    timestamp = payload.get("frozen_at_utc")
    if not isinstance(timestamp, str):
        raise RuntimeError("invalid E21 freeze timestamp")
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as error:
        raise RuntimeError("invalid E21 freeze timestamp") from error
    if parsed.tzinfo is None:
        raise RuntimeError("E21 freeze timestamp must include UTC offset")

    sources = payload.get("source_runs")
    original = payload.get("original_aggregate")
    repair = payload.get("repair_aggregate")
    if (
        not isinstance(sources, list)
        or len(sources) != 5
        or not isinstance(original, dict)
        or not isinstance(repair, dict)
    ):
        raise RuntimeError("invalid E21 freeze run inventory")
    if (
        original.get("frozen_disposition") != ORIGINAL_DISPOSITION
        or original.get("claim_eligible") is not False
        or not isinstance(repair.get("claim_eligible"), bool)
    ):
        raise RuntimeError("invalid E21 original/R1 disposition boundary")
    r1_supported = bool(repair["claim_eligible"])
    expected_status = "SUPPORTED" if r1_supported else "NOT_SUPPORTED"
    if (
        payload.get("claim_status") != expected_status
        or repair.get("frozen_disposition") != expected_status
        or payload.get("claim_boundary") != _claim_boundary(r1_supported)
    ):
        raise RuntimeError("invalid E21 freeze claim boundary")

    rebuilt = build_freeze_payload(
        repo_root=repo_root,
        artifact_root=artifact_root,
        source_runs=[str(row["run_dir"]) for row in sources],
        original_aggregate_run=str(original["run_dir"]),
        r1_aggregate_run=str(repair["run_dir"]),
        frozen_at_utc=timestamp,
    )
    if payload != rebuilt:
        raise RuntimeError("E21 freeze does not reproduce from frozen artifacts")


def _write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    content = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
    )
    content += "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default="/home/minjun_dev/CATENA")
    parser.add_argument(
        "--artifact-root",
        default=os.getenv(
            "CATENA_ARTIFACT_ROOT",
            "/data/minjun_dev/CATENA/artifacts",
        ),
    )
    parser.add_argument("--source-run", action="append", default=[])
    parser.add_argument("--original-aggregate-run")
    parser.add_argument("--r1-aggregate-run")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-existing", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    artifact_root = Path(args.artifact_root).resolve()
    output_path = artifact_root / FREEZE_FILENAME
    try:
        if args.validate_existing:
            validate_freeze(
                _read_json(output_path),
                repo_root=args.repo_root,
                artifact_root=artifact_root,
            )
            print(f"[E21 FREEZE] VALID {output_path} sha256={_sha256(output_path)}")
            return 0
        if not args.original_aggregate_run or not args.r1_aggregate_run:
            raise ValueError("creation requires --original-aggregate-run and --r1-aggregate-run")
        payload = build_freeze_payload(
            repo_root=args.repo_root,
            artifact_root=artifact_root,
            source_runs=args.source_run,
            original_aggregate_run=args.original_aggregate_run,
            r1_aggregate_run=args.r1_aggregate_run,
        )
        if args.dry_run:
            print(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2))
            return 0
        _write_exclusive(output_path, payload)
        print(f"[E21 FREEZE] CREATED {output_path} sha256={_sha256(output_path)}")
        return 0
    except Exception as error:
        print(f"[BLOCKED] {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
