#!/usr/bin/env python3
"""Create one immutable E10–E21 result-summary audit index.

The tool only indexes existing ``RESULTS_SUMMARY_KO.md`` files from completed
run directories.  It never writes inside a run directory and never modifies
the historical E10–E16 index.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

OUTPUT_FILENAME = "POSTCORE_E10_E21_RESULTS_SUMMARY_INDEX_KO.md"
LEGACY_INDEX_FILENAME = "POSTCORE_E10_E16_RESULTS_SUMMARY_INDEX_KO.md"
SUMMARY_FILENAME = "RESULTS_SUMMARY_KO.md"
MAX_SUMMARY_LINES = 60
MAX_SUMMARY_BYTES = 8_000
RUN_ID_PATTERN = re.compile(r"^\d{8}T\d{6}\.\d{6}Z$")
EXPERIMENT_PATTERN = re.compile(r"^e(?P<number>\d{2})(?:[a-z0-9_]*)$")
REQUIRED_NUMBERS = tuple(range(10, 21))
OPTIONAL_NUMBER = 21
REQUIRED_E18_AGGREGATE = "e18b_sequence_control_lattice_aggregate"


@dataclass(frozen=True)
class SummaryRecord:
    experiment_number: int
    experiment_id: str
    run_id: str
    relative_path: str
    sha256: str
    lines: int
    utf8_bytes: int
    execution_status: str
    run_mode: str


@dataclass(frozen=True)
class IndexPlan:
    artifact_root: Path
    output_path: Path
    records: tuple[SummaryRecord, ...]
    content: str
    e21_present: bool

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant: {value}")
        ),
    )
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _experiment_number(name: str) -> int | None:
    match = EXPERIMENT_PATTERN.fullmatch(name)
    if match is None:
        return None
    number = int(match.group("number"))
    if number not in {*REQUIRED_NUMBERS, OPTIONAL_NUMBER}:
        return None
    return number


def _execution_status(report: dict[str, Any]) -> str:
    for key in ("status", "execution_status"):
        value = report.get(key)
        if isinstance(value, str) and value:
            return value
    claim_gate = report.get("claim_gate")
    if isinstance(claim_gate, dict):
        value = claim_gate.get("status")
        if isinstance(value, str) and value:
            return value
    raise RuntimeError("summary run report has no explicit execution status")


def _run_mode(
    report: dict[str, Any],
    manifest: dict[str, Any],
) -> str:
    for payload in (manifest, report):
        value = payload.get("run_mode")
        if isinstance(value, str) and value:
            return value
    return "LEGACY"


def _validate_record(
    *,
    artifact_root: Path,
    summary_path: Path,
    experiment_number: int,
) -> SummaryRecord:
    if summary_path.is_symlink():
        raise RuntimeError(f"summary symlinks are not allowed: {summary_path}")
    resolved = summary_path.resolve()
    if not resolved.is_relative_to(artifact_root):
        raise RuntimeError(f"summary escapes artifact root: {summary_path}")
    run_dir = summary_path.parent
    experiment_dir = run_dir.parent
    experiment_id = experiment_dir.name
    run_id = run_dir.name
    if (
        summary_path.name != SUMMARY_FILENAME
        or _experiment_number(experiment_id) != experiment_number
        or RUN_ID_PATTERN.fullmatch(run_id) is None
        or experiment_dir.parent.resolve() != artifact_root
    ):
        raise RuntimeError(f"noncanonical summary path: {summary_path}")

    report_path = run_dir / "report.json"
    manifest_path = run_dir / "run_manifest.json"
    if not report_path.is_file() or not manifest_path.is_file():
        raise RuntimeError(
            f"summary lacks completed report/manifest pair: {summary_path}"
        )
    report = _read_json_object(report_path)
    manifest = _read_json_object(manifest_path)
    if manifest.get("experiment_id") != experiment_id:
        raise RuntimeError(f"summary manifest experiment mismatch: {summary_path}")
    manifest_run_id = manifest.get("run_id")
    if manifest_run_id is not None and manifest_run_id != run_id:
        raise RuntimeError(f"summary manifest run mismatch: {summary_path}")
    expected_report_hash = manifest.get("report_sha256")
    if expected_report_hash is not None and (
        not isinstance(expected_report_hash, str)
        or expected_report_hash != _sha256(report_path)
    ):
        raise RuntimeError(f"summary report hash mismatch: {summary_path}")

    content = summary_path.read_bytes()
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError(f"summary is not UTF-8: {summary_path}") from error
    lines = len(text.splitlines())
    utf8_bytes = len(content)
    if lines > MAX_SUMMARY_LINES or utf8_bytes > MAX_SUMMARY_BYTES:
        raise RuntimeError(
            "summary exceeds one-page audit contract: "
            f"{summary_path} lines={lines}/{MAX_SUMMARY_LINES} "
            f"bytes={utf8_bytes}/{MAX_SUMMARY_BYTES}"
        )
    relative_path = summary_path.relative_to(artifact_root).as_posix()
    return SummaryRecord(
        experiment_number=experiment_number,
        experiment_id=experiment_id,
        run_id=run_id,
        relative_path=relative_path,
        sha256=hashlib.sha256(content).hexdigest(),
        lines=lines,
        utf8_bytes=utf8_bytes,
        execution_status=_execution_status(report),
        run_mode=_run_mode(report, manifest),
    )


def collect_records(artifact_root: str | Path) -> tuple[SummaryRecord, ...]:
    root = Path(artifact_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"artifact root is not a directory: {root}")
    records: list[SummaryRecord] = []
    for experiment_dir in sorted(root.iterdir()):
        if not experiment_dir.is_dir():
            continue
        number = _experiment_number(experiment_dir.name)
        if number is None:
            continue
        for summary_path in sorted(
            experiment_dir.glob(f"*/{SUMMARY_FILENAME}")
        ):
            records.append(
                _validate_record(
                    artifact_root=root,
                    summary_path=summary_path,
                    experiment_number=number,
                )
            )
    records.sort(
        key=lambda record: (
            record.experiment_number,
            record.experiment_id,
            record.run_id,
        )
    )
    coverage = {record.experiment_number for record in records}
    missing = sorted(set(REQUIRED_NUMBERS) - coverage)
    if missing:
        raise RuntimeError(
            "post-core summary coverage is incomplete: "
            + ", ".join(f"E{number}" for number in missing)
        )
    if not any(
        record.experiment_id == REQUIRED_E18_AGGREGATE
        for record in records
    ):
        raise RuntimeError(
            "E18b aggregate summary is required before creating the index"
        )
    return tuple(records)


def _render(records: tuple[SummaryRecord, ...]) -> str:
    by_number: dict[int, list[SummaryRecord]] = {}
    for record in records:
        by_number.setdefault(record.experiment_number, []).append(record)
    lines = [
        "# CATENA E10–E21 결과 요약 감사 index",
        "",
        "이 파일은 완료된 run directory의 `RESULTS_SUMMARY_KO.md`만 "
        "감사해 연결한다.",
        "각 summary는 UTF-8, 최대 60 lines·8,000 bytes 조건을 통과했으며 "
        "SHA-256은 summary 원문 기준이다.",
        "기존 `POSTCORE_E10_E16_RESULTS_SUMMARY_INDEX_KO.md`와 report, "
        "metric, checkpoint, manifest는 변경하지 않는다.",
        "",
        "## Audit overview",
        "",
        "| 범위 | Summary 수 |",
        "|---|---:|",
        f"| E10–E20 | {sum(len(by_number.get(n, [])) for n in REQUIRED_NUMBERS)} |",
        f"| E21 (optional) | {len(by_number.get(OPTIONAL_NUMBER, []))} |",
        f"| 전체 | {len(records)} |",
        "",
    ]
    for number in (*REQUIRED_NUMBERS, OPTIONAL_NUMBER):
        group = by_number.get(number, [])
        lines.extend([f"## E{number}", ""])
        if not group:
            lines.extend(
                [
                    (
                        "이 index 생성 시점에 검증 가능한 completed summary가 "
                        "없다."
                    ),
                    "",
                ]
            )
            continue
        lines.extend(
            [
                "| Experiment | Run | Status / mode | Lines / bytes | "
                "Summary SHA-256 | Link |",
                "|---|---|---|---:|---|---|",
            ]
        )
        for record in group:
            lines.append(
                f"| `{record.experiment_id}` | `{record.run_id}` | "
                f"`{record.execution_status}` / `{record.run_mode}` | "
                f"{record.lines} / {record.utf8_bytes} | "
                f"`{record.sha256}` | "
                f"[summary]({record.relative_path}) |"
            )
        lines.append("")
    return "\n".join(lines)


def build_index_plan(artifact_root: str | Path) -> IndexPlan:
    root = Path(artifact_root).resolve()
    output_path = root / OUTPUT_FILENAME
    legacy_path = root / LEGACY_INDEX_FILENAME
    records = collect_records(root)
    if output_path == legacy_path:
        raise RuntimeError("new index path collides with the legacy index")
    return IndexPlan(
        artifact_root=root,
        output_path=output_path,
        records=records,
        content=_render(records),
        e21_present=any(
            record.experiment_number == OPTIONAL_NUMBER for record in records
        ),
    )


def _write_exclusive(path: Path, content: str) -> None:
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


def validate_existing(plan: IndexPlan) -> None:
    if not plan.output_path.is_file():
        raise FileNotFoundError(f"index does not exist: {plan.output_path}")
    if plan.output_path.read_text(encoding="utf-8") != plan.content:
        raise RuntimeError("existing E10–E21 index does not reproduce")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        default=os.getenv(
            "CATENA_ARTIFACT_ROOT",
            "/data/minjun_dev/CATENA/artifacts",
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-existing", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        plan = build_index_plan(args.artifact_root)
        if args.validate_existing:
            validate_existing(plan)
            disposition = "VALID"
        elif args.dry_run:
            disposition = "WOULD_CREATE"
        else:
            _write_exclusive(plan.output_path, plan.content)
            disposition = "CREATED"
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "disposition": disposition,
                    "output_path": str(plan.output_path),
                    "content_sha256": plan.content_sha256,
                    "summary_count": len(plan.records),
                    "e21_present": plan.e21_present,
                },
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except Exception as error:
        print(f"[BLOCKED] {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
