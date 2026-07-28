from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SemanticAuditThresholds:
    total_items: int = 300
    minimum_meaning_preservation: float = 0.95
    maximum_answer_leakage: float = 0.02
    minimum_raw_agreement_each_label: float = 0.80


_DEFAULT_THRESHOLDS = SemanticAuditThresholds()


@dataclass(frozen=True, slots=True)
class SemanticAuditReport:
    items: int
    adjudicated_meaning_preservation: float
    adjudicated_answer_leakage: float
    meaning_raw_agreement: float
    leakage_raw_agreement: float
    passed: bool
    failures: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "items": self.items,
            "adjudicated_meaning_preservation": (
                self.adjudicated_meaning_preservation
            ),
            "adjudicated_answer_leakage": self.adjudicated_answer_leakage,
            "meaning_raw_agreement": self.meaning_raw_agreement,
            "leakage_raw_agreement": self.leakage_raw_agreement,
            "passed": self.passed,
            "failures": list(self.failures),
        }


def _read_csv(path: str | Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    resolved = Path(path).expanduser().resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file():
        raise ValueError(f"Audit input must be a direct regular file: {resolved}.")
    with resolved.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Audit CSV has no header: {resolved}.")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"Audit CSV has duplicate columns: {resolved}.")
        return tuple(reader.fieldnames), list(reader)


def _indexed_binary_rows(
    rows: list[dict[str, str]],
    *,
    path_name: str,
    meaning_column: str,
    leakage_column: str,
) -> dict[str, tuple[int, int]]:
    result: dict[str, tuple[int, int]] = {}
    for index, row in enumerate(rows, start=2):
        audit_id = row.get("audit_id", "")
        if not audit_id:
            raise ValueError(f"{path_name} row {index} has no audit_id.")
        if audit_id in result:
            raise ValueError(f"{path_name} duplicates audit_id {audit_id!r}.")
        values: list[int] = []
        for column in (meaning_column, leakage_column):
            raw = row.get(column, "")
            if raw not in {"0", "1"}:
                raise ValueError(
                    f"{path_name} row {index} column {column} must be 0 or 1."
                )
            values.append(int(raw))
        result[audit_id] = (values[0], values[1])
    return result


def evaluate_semantic_human_audit(
    *,
    audit_items_path: str | Path,
    reviewer_a_path: str | Path,
    reviewer_b_path: str | Path,
    adjudication_path: str | Path,
    thresholds: SemanticAuditThresholds = _DEFAULT_THRESHOLDS,
) -> SemanticAuditReport:
    """Validate immutable two-reviewer files and a complete adjudication.

    Review files are intentionally separate from the locked audit-item registry.
    This function never edits any input and rejects outcome-bearing audit columns.
    """

    item_header, item_rows = _read_csv(audit_items_path)
    forbidden_fragments = ("mse", "error", "loss", "gate", "model", "prediction")
    outcome_columns = [
        column
        for column in item_header
        if any(fragment in column.lower() for fragment in forbidden_fragments)
    ]
    if outcome_columns:
        raise ValueError(
            f"Audit items expose model outcome columns: {sorted(outcome_columns)}."
        )
    item_ids = [row.get("audit_id", "") for row in item_rows]
    if any(not value for value in item_ids) or len(item_ids) != len(set(item_ids)):
        raise ValueError("Audit item IDs must be nonempty and unique.")
    if len(item_ids) != thresholds.total_items:
        raise ValueError(
            f"Expected {thresholds.total_items} audit items, got {len(item_ids)}."
        )

    _, reviewer_a_rows = _read_csv(reviewer_a_path)
    _, reviewer_b_rows = _read_csv(reviewer_b_path)
    _, adjudication_rows = _read_csv(adjudication_path)
    reviewer_a = _indexed_binary_rows(
        reviewer_a_rows,
        path_name="reviewer A",
        meaning_column="meaning_preserved",
        leakage_column="answer_leakage",
    )
    reviewer_b = _indexed_binary_rows(
        reviewer_b_rows,
        path_name="reviewer B",
        meaning_column="meaning_preserved",
        leakage_column="answer_leakage",
    )
    adjudication = _indexed_binary_rows(
        adjudication_rows,
        path_name="adjudication",
        meaning_column="adjudicated_meaning_preserved",
        leakage_column="adjudicated_answer_leakage",
    )
    expected = set(item_ids)
    for name, values in (
        ("reviewer A", reviewer_a),
        ("reviewer B", reviewer_b),
        ("adjudication", adjudication),
    ):
        if set(values) != expected:
            missing = sorted(expected - set(values))
            extra = sorted(set(values) - expected)
            raise ValueError(
                f"{name} audit IDs differ; missing={missing[:3]}, extra={extra[:3]}."
            )

    meaning_rate = sum(adjudication[item][0] for item in item_ids) / len(item_ids)
    leakage_rate = sum(adjudication[item][1] for item in item_ids) / len(item_ids)
    meaning_agreement = (
        sum(reviewer_a[item][0] == reviewer_b[item][0] for item in item_ids)
        / len(item_ids)
    )
    leakage_agreement = (
        sum(reviewer_a[item][1] == reviewer_b[item][1] for item in item_ids)
        / len(item_ids)
    )
    failures: list[str] = []
    if meaning_rate < thresholds.minimum_meaning_preservation:
        failures.append("ADJUDICATED_MEANING_PRESERVATION")
    if leakage_rate > thresholds.maximum_answer_leakage:
        failures.append("ADJUDICATED_ANSWER_LEAKAGE")
    if meaning_agreement < thresholds.minimum_raw_agreement_each_label:
        failures.append("MEANING_REVIEWER_AGREEMENT")
    if leakage_agreement < thresholds.minimum_raw_agreement_each_label:
        failures.append("LEAKAGE_REVIEWER_AGREEMENT")
    return SemanticAuditReport(
        items=len(item_ids),
        adjudicated_meaning_preservation=float(meaning_rate),
        adjudicated_answer_leakage=float(leakage_rate),
        meaning_raw_agreement=float(meaning_agreement),
        leakage_raw_agreement=float(leakage_agreement),
        passed=not failures,
        failures=tuple(failures),
    )
