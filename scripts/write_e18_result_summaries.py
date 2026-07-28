#!/usr/bin/env python3
"""Create missing, compact Korean result summaries for completed E18 runs.

This is additive documentation tooling.  It does not select scientific inputs,
run an aggregate, or edit reports, manifests, metrics, checkpoints, configs, or
the prospective protocol lock.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import check_e18_status

SOURCE_EXPERIMENT_ID = check_e18_status.SOURCE_EXPERIMENT_ID
AGGREGATE_EXPERIMENT_ID = check_e18_status.AGGREGATE_EXPERIMENT_ID
SUMMARY_FILENAME = "RESULTS_SUMMARY_KO.md"
EXPECTED_PROTOCOL_LOCK_SHA256 = (
    "7c465ceb60b6979e717d85599533bd7c0dd884f10b191fa29c42771ccc9c9989"
)
RUN_ID_PATTERN = re.compile(r"^\d{8}T\d{6}\.\d{6}Z$")
MAX_SUMMARY_LINES = 60
MAX_SUMMARY_UTF8_BYTES = 8_000

KOREAN_VARIANTS = {
    "tied_scalar": "tied scalar",
    "dual_scalar": "dual erase/write",
    "diagonal_value": "diagonal value",
    "separate_address": "separate address",
    "state_aware": "state-aware",
}
KOREAN_DEMANDS = {
    "magnitude_factorization": "Magnitude factorization",
    "value_granularity": "Value granularity",
    "address_decoupling": "Address decoupling",
    "state_conditioning": "State conditioning",
}


@dataclass(frozen=True)
class SummaryAction:
    kind: str
    run_dir: Path
    output_path: Path
    content: str

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SummaryPlan:
    protocol_lock_sha256: str
    actions: tuple[SummaryAction, ...]
    aggregate_available: bool


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    payload = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_nonfinite,
    )
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(
                line,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON constant: {value}")
                ),
            )
        except (json.JSONDecodeError, ValueError) as error:
            raise RuntimeError(f"invalid JSONL {path}:{line_number}") from error
        if not isinstance(row, dict):
            raise TypeError(f"expected JSON object at {path}:{line_number}")
        rows.append(row)
    return rows


def _finite(row: dict[str, Any], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"missing/invalid metric {key!r}") from error
    if not math.isfinite(value):
        raise RuntimeError(f"non-finite metric {key!r}")
    return value


def _format_number(value: float) -> str:
    magnitude = abs(value)
    if magnitude != 0.0 and (magnitude < 1e-4 or magnitude >= 1e4):
        return f"{value:.3e}"
    return f"{value:.7f}"


def _validate_summary_size(content: str) -> None:
    lines = len(content.splitlines())
    encoded = len(content.encode("utf-8"))
    if lines > MAX_SUMMARY_LINES or encoded > MAX_SUMMARY_UTF8_BYTES:
        raise RuntimeError(
            "generated E18 summary exceeds the compact one-page contract: "
            f"lines={lines}/{MAX_SUMMARY_LINES}, "
            f"bytes={encoded}/{MAX_SUMMARY_UTF8_BYTES}"
        )


def _uniform_metric(rows: list[dict[str, Any]], key: str) -> float:
    values = {_finite(row, key) for row in rows}
    if len(values) != 1:
        raise RuntimeError(f"E18a per-run metric is not uniform: {key}")
    return next(iter(values))


def _render_source_summary(
    *,
    run_dir: Path,
    report: dict[str, Any],
    rows: list[dict[str, Any]],
    report_sha256: str,
    protocol_lock_sha256: str,
) -> str:
    seed = int(report["seed"])
    variant = str(report["variant"])
    if len(rows) != 48:
        raise RuntimeError(f"E18a source summary requires 48 rows: {run_dir}")
    mean_affected = sum(_finite(row, "affected_mse") for row in rows) / len(rows)
    max_retention = max(_finite(row, "retention_mse") for row in rows)
    examples_per_second = _uniform_metric(rows, "examples_per_second")
    train_final_loss = _uniform_metric(rows, "train_final_loss")
    train_best_loss = _uniform_metric(rows, "train_best_loss")
    stress_rows = sorted(
        (
            row
            for row in rows
            if int(row["updates"]) == 8 and int(row["gap_events"]) == 2048
        ),
        key=lambda row: tuple(KOREAN_DEMANDS).index(
            str(row["demand_family"])
        ),
    )
    if len(stress_rows) != 4:
        raise RuntimeError(f"E18a source stress grid is incomplete: {run_dir}")
    minimum_active_harm = min(
        _finite(row, "distractor_activation_retention_harm")
        for row in stress_rows
    )
    stress_lines = "\n".join(
        "| "
        f"{KOREAN_DEMANDS[str(row['demand_family'])]} | "
        f"{_format_number(_finite(row, 'affected_mse'))} | "
        f"{_format_number(_finite(row, 'retention_mse'))} |"
        for row in stress_rows
    )
    content = f"""# E18a Sequence Control Lattice — Seed {seed}

**Run:** `{run_dir.name}` · **Controller:** `{variant}` ({KOREAN_VARIANTS[variant]})  
**판정:** `PASS / PENDING_AGGREGATE` · **Evidence:** `CONTROLLED_REFERENCE`

## Run 요약

| 지표 | 값 |
|---|---:|
| 전체 48-cell affected MSE 평균 | {_format_number(mean_affected)} |
| 전체 retention MSE 최댓값 | {_format_number(max_retention)} |
| Train final / best loss | {_format_number(train_final_loss)} / {_format_number(train_best_loss)} |
| 학습 처리량 | {examples_per_second:.1f} examples/s |
| Active-path retention harm 최솟값 | {_format_number(minimum_active_harm)} |

## Long-gap·multi-update stress (`updates=8`, `gap=2048`)

| Demand family | Affected MSE | Retention MSE |
|---|---:|---:|
{stress_lines}

이 문서는 단일 controller×seed run의 개발·검증 기록이다. Architecture 간
registered-grid mean 이득, 5-seed stress 방향성 및 최종 claim은 E18b
paired aggregate에서만 판정한다. 입력은 oracle address/candidate,
explicit demand descriptor와 model-visible verified bit를 포함한다. 따라서
semantic demand/relevance inference, every-cell·uniform persistence,
stress SESOI, accurate preservation, 자연어, learned localization/candidate,
recurrent LM, agent/planning 또는 official-backend claim을 열지 않는다.

**Protocol lock SHA-256:** `{protocol_lock_sha256}`  
**Report SHA-256:** `{report_sha256}`
"""
    _validate_summary_size(content)
    return content


def _render_aggregate_summary(
    *,
    run_dir: Path,
    report: dict[str, Any],
    report_sha256: str,
    protocol_lock_sha256: str,
) -> str:
    supported = bool(report["claim_gate"]["supported"])
    disposition = "SUPPORTED" if supported else "NOT OPENED"
    contrast_lines: list[str] = []
    for name, contrast in report["contrasts"].items():
        contrast_lines.append(
            "| "
            f"{name} | "
            f"{contrast['baseline']}→{contrast['treatment']} | "
            f"{_format_number(float(contrast['mean_corresponding_demand_gain']))} | "
            f"{int(round(float(contrast['stress_positive_seed_fraction']) * 5))}/5 | "
            f"{float(contrast['stress_sign_flip_p']):.5f} | "
            f"{'PASS' if contrast['passed'] else 'FAIL'} |"
        )
    conditions = report["claim_gate"]["conditions"]
    minimum_active_harm = _format_number(
        float(report["summary"]["minimum_active_path_retention_harm"])
    )
    condition_text = ", ".join(
        f"`{name}={'PASS' if passed else 'FAIL'}`"
        for name, passed in conditions.items()
    )
    summary = report["summary"]
    contrast_header = (
        "| Freedom contrast | Adjacent controller | Grid-mean affected gain "
        "| Stress 양수 seed | Exact p | 판정 |"
    )
    content = f"""# E18b Sequence Architecture–Demand Lattice — 결과 요약

**Run:** `{run_dir.name}`  
**판정:** `{disposition}` · **Evidence:** `CONTROLLED_REFERENCE`

## 핵심 paired contrast

{contrast_header}
|---|---|---:|---:|---:|---|
{chr(10).join(contrast_lines)}

## 계약·guardrail

| 항목 | 값 |
|---|---:|
| Source run / metric rows | {int(summary['source_runs'])} / {int(summary['metric_rows'])} |
| Paired seed contrast rows | {int(summary['paired_contrast_seed_rows'])} |
| Active-path assay rows | {int(summary['active_path_rows'])} |
| 최소 active-path retention harm | {minimum_active_harm} |

등록 조건: {condition_text}.

위 gain은 12개 update×gap cell의 등록 평균이며 every-cell 개선을 뜻하지
않는다. Stress는 5/5 방향성과 exact p를 검사하지만 별도 SESOI는 없다.
Simpler-demand/retention은 adjacent cell-mean non-inferiority이지 absolute
accuracy가 아니다. 입력은 oracle address/candidate, explicit demand
descriptor와 model-visible verified bit를 포함하므로 semantic demand/relevance
inference, learned localization/candidate, 자연어, recurrent LM, agent/planning
또는 official-backend transfer로 확장하지 않는다.

**Protocol lock SHA-256:** `{protocol_lock_sha256}`  
**Report SHA-256:** `{report_sha256}`
"""
    _validate_summary_size(content)
    return content


def _running_e18_aggregate_processes() -> list[str]:
    own_pid = os.getpid()
    matches: list[str] = []
    for proc in sorted(Path("/proc").glob("[0-9]*")):
        try:
            pid = int(proc.name)
            if pid == own_pid:
                continue
            args = [
                item.decode("utf-8", errors="replace")
                for item in (proc / "cmdline").read_bytes().split(b"\0")
                if item
            ]
        except (OSError, ValueError):
            continue
        is_aggregate = (
            AGGREGATE_EXPERIMENT_ID in args
            or f"experiments.{AGGREGATE_EXPERIMENT_ID}" in args
            or any(
                item.endswith(f"{AGGREGATE_EXPERIMENT_ID}.py")
                for item in args
            )
        )
        if is_aggregate and "--dry-run" not in args:
            matches.append(f"pid={pid} command={' '.join(args)}")
    return matches


def _aggregate_runtime_config(
    aggregate_config: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    runtime = json.loads(json.dumps(aggregate_config))
    runtime["source"]["config_path"] = str(
        (repo_root / check_e18_status.SOURCE_CONFIG_RELATIVE_PATH).resolve()
    )
    return runtime


def _validate_aggregate_run(
    *,
    run_dir: Path,
    repo_root: Path,
    artifact_root: Path,
    aggregate_config: dict[str, Any],
    protocol_lock_sha256: str,
) -> tuple[dict[str, Any], str]:
    if not RUN_ID_PATTERN.fullmatch(run_dir.name):
        raise RuntimeError(f"noncanonical E18b run id: {run_dir}")
    report_path = run_dir / "report.json"
    manifest_path = run_dir / "run_manifest.json"
    report = _read_json_object(report_path)
    manifest = _read_json_object(manifest_path)
    aggregate_config_path = (
        repo_root / check_e18_status.AGGREGATE_CONFIG_RELATIVE_PATH
    )
    report_sha256 = _sha256(report_path)
    if (
        manifest.get("schema_version") != 2
        or manifest.get("experiment_id") != AGGREGATE_EXPERIMENT_ID
        or manifest.get("run_id") != run_dir.name
        or manifest.get("run_mode") != "MAIN"
        or not isinstance(manifest.get("completed_at_utc"), str)
        or manifest.get("report_sha256") != report_sha256
        or manifest.get("config") != aggregate_config
        or manifest.get("config_file_sha256") != _sha256(aggregate_config_path)
        or report.get("status") != "PASS"
        or report.get("run_scope")
        != "SEQUENCE_CONTROL_ARCHITECTURE_DEMAND_LATTICE_AGGREGATE"
        or report.get("protocol_lock", {}).get("sha256")
        != protocol_lock_sha256
        or report.get("evidence_tier") != "CONTROLLED_REFERENCE"
        or report.get("scientific_evidence") is not False
    ):
        raise RuntimeError(f"invalid completed E18b MAIN provenance: {run_dir}")

    module = importlib.import_module(
        "experiments.e18b_sequence_control_lattice_aggregate"
    )
    runtime = _aggregate_runtime_config(aggregate_config, repo_root)
    rows, provenance = module.collect_main_sources(
        artifact_root=artifact_root,
        config=runtime,
        protocol_lock_sha256=protocol_lock_sha256,
    )
    grid_passed = module.paired_grid_contract(rows=rows, config=runtime)
    contrasts, paired_rows, active_rows = module.aggregate_contrasts(
        rows=rows,
        config=runtime,
    )
    minimum_active_harm = min(
        float(row["active_path_retention_harm"]) for row in active_rows
    )
    active_path_passed = minimum_active_harm >= float(
        runtime["claim_gate"]["minimum_active_path_retention_harm"]
    )
    conditions = {
        "all_adjacent_contrasts_passed": all(
            bool(contrast["passed"]) for contrast in contrasts.values()
        ),
        "full_paired_grid_passed": grid_passed,
        "source_provenance_passed": bool(
            len(provenance) == 25
            and all(
                row["distractor_path_contract_passed"]
                for row in provenance
            )
        ),
        "model_visible_active_path_assay_passed": active_path_passed,
    }
    expected_summary = {
        "source_runs": len(provenance),
        "metric_rows": len(rows),
        "paired_contrast_seed_rows": len(paired_rows),
        "active_path_rows": len(active_rows),
        "minimum_active_path_retention_harm": minimum_active_harm,
    }
    if (
        report.get("source_runs") != provenance
        or report.get("contrasts") != contrasts
        or report.get("summary") != expected_summary
        or report.get("claim_gate", {}).get("conditions") != conditions
        or report.get("claim_gate", {}).get("supported")
        != bool(all(conditions.values()))
    ):
        raise RuntimeError(f"E18b report does not reproduce from sources: {run_dir}")

    exact_files: tuple[tuple[str, list[dict[str, Any]]], ...] = (
        ("sequence_control_lattice_paired_metrics.jsonl", paired_rows),
        ("sequence_control_lattice_active_path_metrics.jsonl", active_rows),
        ("source_run_provenance.jsonl", provenance),
    )
    for filename, expected_rows in exact_files:
        if _read_jsonl(run_dir / filename) != expected_rows:
            raise RuntimeError(
                f"E18b derived artifact does not reproduce: {run_dir / filename}"
            )
    return report, report_sha256


def _find_aggregate(
    *,
    repo_root: Path,
    artifact_root: Path,
    aggregate_config: dict[str, Any],
    protocol_lock_sha256: str,
) -> tuple[Path, dict[str, Any], str] | None:
    namespace = artifact_root / AGGREGATE_EXPERIMENT_ID
    if not namespace.exists():
        return None
    if not namespace.is_dir():
        raise RuntimeError(f"E18b artifact namespace is not a directory: {namespace}")
    completed: list[tuple[Path, dict[str, Any], str]] = []
    for run_dir in sorted(path for path in namespace.iterdir() if path.is_dir()):
        manifest_path = run_dir / "run_manifest.json"
        try:
            manifest = _read_json_object(manifest_path)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise RuntimeError(f"incomplete/invalid E18b run: {run_dir}") from error
        mode = manifest.get("run_mode")
        if mode == "DRY_RUN":
            continue
        if mode != "MAIN":
            raise RuntimeError(f"unexpected E18b run mode {mode!r}: {run_dir}")
        completed.append(
            (
                run_dir,
                *_validate_aggregate_run(
                    run_dir=run_dir,
                    repo_root=repo_root,
                    artifact_root=artifact_root,
                    aggregate_config=aggregate_config,
                    protocol_lock_sha256=protocol_lock_sha256,
                ),
            )
        )
    if len(completed) > 1:
        raise RuntimeError(
            "duplicate completed E18b MAIN runs: "
            + ", ".join(str(item[0]) for item in completed)
        )
    return completed[0] if completed else None


def build_summary_plan(
    *,
    repo_root: str | Path,
    artifact_root: str | Path,
    scope: str = "all",
    include_live: bool = True,
) -> SummaryPlan:
    root = Path(repo_root).resolve()
    artifacts = Path(artifact_root).resolve()
    source_config, aggregate_config, lock_sha256 = (
        check_e18_status.load_locked_contract(root)
    )
    if lock_sha256 != EXPECTED_PROTOCOL_LOCK_SHA256:
        raise RuntimeError(
            "E18 protocol lock differs from the final pre-main lock: "
            f"{lock_sha256}"
        )
    snapshot = check_e18_status.scan_e18_status(
        repo_root=root,
        artifact_root=artifacts,
        include_live=include_live,
    )
    if snapshot.blockers:
        raise RuntimeError(
            "E18 source summary generation blocked: "
            + ", ".join(snapshot.blockers)
        )
    if include_live:
        aggregate_processes = _running_e18_aggregate_processes()
        if aggregate_processes:
            raise RuntimeError(
                "E18 aggregate summary generation blocked by live MAIN: "
                + "; ".join(aggregate_processes)
            )

    module = importlib.import_module(
        "experiments.e18b_sequence_control_lattice_aggregate"
    )
    seeds, variants, demands, updates, gaps = module._source_contract(
        aggregate_config
    )
    actions: list[SummaryAction] = []
    if scope in {"all", "source"}:
        for cell in snapshot.canonical_cells:
            paths = snapshot.completed_runs.get(cell, [])
            if not paths:
                continue
            run_dir = Path(paths[0])
            result = module._validate_source_run(
                run_dir=run_dir,
                expected_config=source_config,
                source_config_path=(
                    root / check_e18_status.SOURCE_CONFIG_RELATIVE_PATH
                ),
                variants=variants,
                seeds=seeds,
                demands=demands,
                updates=updates,
                gaps=gaps,
                protocol_lock_sha256=lock_sha256,
            )
            if result is None:
                raise RuntimeError(f"E18a MAIN validation returned empty: {run_dir}")
            _, rows, provenance = result
            report = _read_json_object(run_dir / "report.json")
            manifest = _read_json_object(run_dir / "run_manifest.json")
            if (
                not RUN_ID_PATTERN.fullmatch(run_dir.name)
                or report.get("status") != "PASS"
                or report.get("run_mode") != "MAIN"
                or report.get("run_scope")
                != "SEQUENCE_CONTROL_ARCHITECTURE_DEMAND_LATTICE"
                or report.get("claim_gate", {}).get("status")
                != "PENDING_AGGREGATE"
                or report.get("evidence_tier") != "CONTROLLED_REFERENCE"
                or report.get("scientific_evidence") is not False
                or not isinstance(manifest.get("completed_at_utc"), str)
            ):
                raise RuntimeError(
                    f"invalid E18a summary identity/claim contract: {run_dir}"
                )
            content = _render_source_summary(
                run_dir=run_dir,
                report=report,
                rows=rows,
                report_sha256=str(provenance["report_sha256"]),
                protocol_lock_sha256=lock_sha256,
            )
            actions.append(
                SummaryAction(
                    kind="SOURCE",
                    run_dir=run_dir,
                    output_path=run_dir / SUMMARY_FILENAME,
                    content=content,
                )
            )

    aggregate = _find_aggregate(
        repo_root=root,
        artifact_root=artifacts,
        aggregate_config=aggregate_config,
        protocol_lock_sha256=lock_sha256,
    )
    if scope in {"all", "aggregate"} and aggregate is not None:
        run_dir, report, report_sha256 = aggregate
        actions.append(
            SummaryAction(
                kind="AGGREGATE",
                run_dir=run_dir,
                output_path=run_dir / SUMMARY_FILENAME,
                content=_render_aggregate_summary(
                    run_dir=run_dir,
                    report=report,
                    report_sha256=report_sha256,
                    protocol_lock_sha256=lock_sha256,
                ),
            )
        )
    return SummaryPlan(
        protocol_lock_sha256=lock_sha256,
        actions=tuple(actions),
        aggregate_available=aggregate is not None,
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


def execute_summary_plan(
    plan: SummaryPlan,
    *,
    dry_run: bool,
) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    for action in plan.actions:
        if action.output_path.exists():
            disposition = "SKIP_EXISTING"
        elif dry_run:
            disposition = "WOULD_CREATE"
        else:
            try:
                _write_exclusive(action.output_path, action.content)
                disposition = "CREATED"
            except FileExistsError:
                disposition = "SKIP_RACE_EXISTING"
        results.append(
            {
                "kind": action.kind,
                "run_dir": str(action.run_dir),
                "output_path": str(action.output_path),
                "content_sha256": action.content_sha256,
                "disposition": disposition,
            }
        )
    return results


def _print_human(plan: SummaryPlan, results: Iterable[dict[str, str]]) -> None:
    print(
        "[E18 SUMMARY] "
        f"protocol_lock_sha256={plan.protocol_lock_sha256} "
        f"aggregate_available={str(plan.aggregate_available).lower()}"
    )
    for result in results:
        print(
            f"[{result['disposition']}] {result['kind']} "
            f"{result['output_path']} sha256={result['content_sha256']}"
        )


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
    parser.add_argument(
        "--scope",
        choices=("all", "source", "aggregate"),
        default="all",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--require-aggregate", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        plan = build_summary_plan(
            repo_root=args.repo_root,
            artifact_root=args.artifact_root,
            scope=args.scope,
        )
        if args.require_aggregate and not plan.aggregate_available:
            raise RuntimeError("no completed, provenance-valid E18b MAIN exists")
        results = execute_summary_plan(plan, dry_run=args.dry_run)
    except Exception as error:
        print(f"[BLOCKED] {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    if args.json:
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "protocol_lock_sha256": plan.protocol_lock_sha256,
                    "aggregate_available": plan.aggregate_available,
                    "dry_run": args.dry_run,
                    "results": results,
                },
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
        )
    else:
        _print_human(plan, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
