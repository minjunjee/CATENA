from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable
from html import escape
from pathlib import Path
from typing import Any

import numpy as np

PAPER_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PAPER_ROOT / "data/source_manifest.json"
DEFAULT_OUTPUT = PAPER_ROOT / "figures"

NAVY = "#1f4e79"
BLUE = "#0072b2"
ORANGE = "#d55e00"
GREEN = "#009e73"
PURPLE = "#6a51a3"
GRAY = "#6b7280"
LIGHT_GRAY = "#e5e7eb"
INK = "#17212b"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise TypeError(f"Expected JSON object at {path}:{line_number}")
            rows.append(payload)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _resolve_source(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError(f"Source path escapes artifact root: {relative}")
    return candidate


def validate_sources(
    artifact_root: Path,
    manifest: dict[str, Any],
) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    for section in ("data_sources", "provenance_anchors"):
        records = manifest.get(section)
        if not isinstance(records, dict) or not records:
            raise ValueError(f"Manifest section {section!r} is empty or invalid")
        for name, record in records.items():
            if not isinstance(record, dict):
                raise TypeError(f"Manifest record {section}.{name} is invalid")
            path = _resolve_source(artifact_root, str(record["path"]))
            if not path.is_file():
                raise FileNotFoundError(f"Missing canonical source: {path}")
            expected = str(record["sha256"])
            observed = file_sha256(path)
            if observed != expected:
                raise ValueError(
                    f"Canonical source hash mismatch for {path}: "
                    f"expected={expected}, observed={observed}"
                )
            resolved[name] = path
    return resolved


def _nested(payload: dict[str, Any], dotted_path: str) -> Any:
    value: Any = payload
    for component in dotted_path.split("."):
        if not isinstance(value, dict) or component not in value:
            raise KeyError(f"Missing field {dotted_path!r}")
        value = value[component]
    return value


def derive_figure1(sources: dict[str, Path]) -> dict[str, Any]:
    h1_report = load_json(sources["h1_report"])
    reachability = dict(h1_report["reachability_comparison"])
    h1 = {
        "predictor_conditional_oos_r2": {
            "behavioral feasible regret": float(
                reachability["r_beh_conditional_oos_r2"]
            ),
            "state feasible regret": float(
                reachability["r_feas_conditional_oos_r2"]
            ),
            "state span regret": float(
                reachability["r_span_conditional_oos_r2"]
            ),
        },
        "calibration_slope": float(
            _nested(
                h1_report,
                "episode_uncertainty.calibration_slope_descriptive.estimate",
            )
        ),
        "operation_adjusted_slope": float(
            _nested(
                h1_report,
                "episode_uncertainty.unseen_operation_adjusted_slope.estimate",
            )
        ),
    }

    e10_report = load_json(sources["e10b_report"])
    rank_rows = load_jsonl(sources["e10b_rank_tracking_cells"])
    rank_points = [
        {
            "seed": int(row["seed"]),
            "intrinsic_rank": int(row["intrinsic_rank"]),
            "minimum_qualifying_rank": int(row["minimum_qualifying_rank"]),
            "matched": bool(row["rank_tracking_matched"]),
        }
        for row in rank_rows
    ]
    rank_points.sort(key=lambda row: (row["seed"], row["intrinsic_rank"]))
    e10 = {
        "points": rank_points,
        "rank_match_fraction": float(
            _nested(e10_report, "summary.rank_match_fraction")
        ),
        "seed_count": int(
            _nested(e10_report, "summary.statistical_unit_count")
        ),
    }

    e03_report = load_json(sources["e03b_report"])
    family_rows = load_jsonl(sources["e03b_family_calibration"])
    calibration_points = [
        {
            "bin": str(row["bin"]),
            "candidate_id": str(row["candidate_id"]),
            "analytic_regret": float(row["heldout_analytic_regret"]),
            "empirical_application_error": float(
                row["heldout_empirical_application_error"]
            ),
        }
        for row in family_rows
    ]
    calibration_points.sort(
        key=lambda row: (
            row["bin"],
            row["analytic_regret"],
            row["candidate_id"],
        )
    )
    expected_n = int(_nested(e03_report, "empirical_regret_prediction.n"))
    if len(calibration_points) != expected_n:
        raise ValueError(
            "E03b family count disagrees with the canonical report: "
            f"rows={len(calibration_points)}, report={expected_n}"
        )
    e03 = {
        "points": calibration_points,
        "r2": float(
            _nested(e03_report, "empirical_regret_prediction.r2")
        ),
        "slope": float(
            _nested(e03_report, "empirical_regret_prediction.slope")
        ),
        "intercept": float(
            _nested(e03_report, "empirical_regret_prediction.intercept")
        ),
    }
    return {"h1": h1, "e10b": e10, "e03b": e03}


E12_CONTRASTS = (
    ("magnitude_factorization", "tied_scalar", "dual_scalar", "Magnitude"),
    ("value_granularity", "dual_scalar", "diagonal_value", "Value channels"),
    (
        "address_decoupling",
        "diagonal_value",
        "separate_address",
        "Address",
    ),
    (
        "state_conditioning",
        "separate_address",
        "state_aware",
        "State-aware",
    ),
)

E18_SOURCE_KEYS = frozenset({"e18b_report", "e18b_paired_metrics"})


def derive_e18_sequence_lattice(
    sources: dict[str, Path],
) -> dict[str, Any] | None:
    """Derive the optional E18 sequence-lattice panel.

    The source manifest must provide both E18 sources or neither.  Keeping the
    optional branch closed until both hash-pinned records exist lets the
    current E12-only paper regenerate byte-for-byte while E18 is still live.
    """

    present = E18_SOURCE_KEYS.intersection(sources)
    if not present:
        return None
    if present != E18_SOURCE_KEYS:
        missing = sorted(E18_SOURCE_KEYS - present)
        raise ValueError(
            "E18 paper integration requires report and paired metrics "
            f"together; missing={missing}"
        )

    report = load_json(sources["e18b_report"])
    rows = load_jsonl(sources["e18b_paired_metrics"])
    if (
        report.get("status") != "PASS"
        or report.get("evidence_tier") != "CONTROLLED_REFERENCE"
        or report.get("scientific_evidence") is not False
    ):
        raise ValueError("E18 report boundary/status is inconsistent")
    claim_gate = report.get("claim_gate")
    if not isinstance(claim_gate, dict) or not isinstance(
        claim_gate.get("supported"),
        bool,
    ):
        raise ValueError("E18 report lacks a Boolean aggregate disposition")
    conditions = claim_gate.get("conditions")
    if (
        not isinstance(conditions, dict)
        or not conditions
        or any(not isinstance(value, bool) for value in conditions.values())
        or claim_gate["supported"] != all(conditions.values())
    ):
        raise ValueError("E18 aggregate disposition is internally inconsistent")
    if (
        int(_nested(report, "summary.source_runs")) != 25
        or int(_nested(report, "summary.metric_rows")) != 1200
        or int(_nested(report, "summary.paired_contrast_seed_rows")) != 20
    ):
        raise ValueError("E18 report does not satisfy the registered source grid")

    expected_names = tuple(item[0] for item in E12_CONTRASTS)
    report_contrasts = report.get("contrasts")
    if not isinstance(report_contrasts, dict) or set(
        report_contrasts
    ) != set(expected_names):
        raise ValueError("E18 report contrast identity is not canonical")

    expected_rows = int(
        _nested(report, "summary.paired_contrast_seed_rows")
    )
    if len(rows) != expected_rows:
        raise ValueError(
            "E18 paired row count disagrees with the canonical report: "
            f"rows={len(rows)}, report={expected_rows}"
        )
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["contrast"]), int(row["seed"]))
        if key in indexed:
            raise ValueError(f"Duplicate E18 paired contrast row: {key}")
        indexed[key] = row
    seeds = sorted({seed for _, seed in indexed})
    if len(seeds) != 5:
        raise ValueError(f"E18 paper panel requires five paired seeds: {seeds}")
    expected_grid = {
        (contrast, seed)
        for contrast in expected_names
        for seed in seeds
    }
    if set(indexed) != expected_grid:
        raise ValueError("E18 paired contrast grid is incomplete")

    contrasts: list[dict[str, Any]] = []
    for family, simpler, richer, label in E12_CONTRASTS:
        frozen = report_contrasts[family]
        gains: list[float] = []
        stress_gains: list[float] = []
        for seed in seeds:
            row = indexed[(family, seed)]
            identity = (
                str(row["baseline"]),
                str(row["treatment"]),
                str(row["target_demand"]),
            )
            if identity != (simpler, richer, family):
                raise ValueError(
                    f"E18 paired identity mismatch for {family}, seed={seed}"
                )
            gains.append(float(row["mean_corresponding_demand_gain"]))
            stress_gains.append(float(row["stress_gain"]))
        observed_mean = float(np.mean(np.asarray(gains, dtype=float)))
        report_mean = float(frozen["mean_corresponding_demand_gain"])
        if not np.isclose(observed_mean, report_mean, rtol=0.0, atol=1e-12):
            raise ValueError(
                f"E18 derived mean mismatch for {family}: "
                f"derived={observed_mean}, report={report_mean}"
            )
        observed_stress_fraction = float(
            np.mean(np.asarray(stress_gains, dtype=float) > 0.0)
        )
        frozen_stress_fraction = float(
            frozen["stress_positive_seed_fraction"]
        )
        if not np.isclose(
            observed_stress_fraction,
            frozen_stress_fraction,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"E18 stress direction mismatch for {family}")
        contrasts.append(
            {
                "family": family,
                "label": label,
                "simpler": simpler,
                "richer": richer,
                "seed_gains": gains,
                "stress_seed_gains": stress_gains,
                "mean_selective_gain": report_mean,
                "max_simpler_task_degradation": float(
                    frozen["maximum_simpler_demand_degradation"]
                ),
                "max_retention_degradation": float(
                    frozen["maximum_retention_degradation"]
                ),
                "stress_positive_seed_fraction": frozen_stress_fraction,
                "sign_flip_p": float(frozen["stress_sign_flip_p"]),
                "passed": bool(frozen["passed"]),
            }
        )
    return {
        "seeds": seeds,
        "contrasts": contrasts,
        "claim_supported": bool(claim_gate["supported"]),
        "source_runs": int(_nested(report, "summary.source_runs")),
        "metric_rows": int(_nested(report, "summary.metric_rows")),
        "minimum_active_path_retention_harm": float(
            _nested(report, "summary.minimum_active_path_retention_harm")
        ),
        "evidence_tier": str(report["evidence_tier"]),
        "scientific_evidence": bool(report["scientific_evidence"]),
    }


def derive_figure2(sources: dict[str, Path]) -> dict[str, Any]:
    report = load_json(sources["e12_report"])
    rows = load_jsonl(sources["e12_metrics"])
    by_key = {
        (int(row["seed"]), str(row["family"]), str(row["freedom"])): float(
            row["affected_mse"]
        )
        for row in rows
    }
    seeds = sorted({int(row["seed"]) for row in rows})
    contrasts: list[dict[str, Any]] = []
    for family, simpler, richer, label in E12_CONTRASTS:
        gains = [
            by_key[(seed, family, simpler)]
            - by_key[(seed, family, richer)]
            for seed in seeds
        ]
        frozen = dict(report["contrasts"][family])
        observed_mean = float(np.mean(np.asarray(gains, dtype=float)))
        report_mean = float(frozen["mean_selective_gain"])
        if not np.isclose(observed_mean, report_mean, rtol=0.0, atol=1e-12):
            raise ValueError(
                f"E12 derived mean mismatch for {family}: "
                f"derived={observed_mean}, report={report_mean}"
            )
        contrasts.append(
            {
                "family": family,
                "label": label,
                "simpler": simpler,
                "richer": richer,
                "seed_gains": gains,
                "mean_selective_gain": report_mean,
                "max_simpler_task_degradation": float(
                    frozen["max_simpler_task_degradation"]
                ),
                "sign_flip_p": float(frozen["sign_flip_p"]),
                "passed": bool(frozen["passed"]),
            }
        )
    result: dict[str, Any] = {"seeds": seeds, "contrasts": contrasts}
    sequence = derive_e18_sequence_lattice(sources)
    if sequence is not None:
        result["sequence"] = sequence
    return result


def derive_figure3(sources: dict[str, Path]) -> dict[str, Any]:
    report = load_json(sources["e13c_report"])
    rows = load_jsonl(sources["e13c_paired_metrics"])
    updates = sorted({int(row["updates"]) for row in rows})
    gaps = sorted({int(row["gap_events"]) for row in rows})
    seeds = sorted({int(row["seed"]) for row in rows})
    grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["updates"]), int(row["gap_events"]))].append(
            float(row["affected_gain_tied_minus_dual"])
        )
    cells: list[dict[str, Any]] = []
    for update_count in updates:
        for gap in gaps:
            values = grouped[(update_count, gap)]
            if len(values) != len(seeds):
                raise ValueError(
                    "Incomplete E13c cell: "
                    f"updates={update_count}, gap={gap}, "
                    f"rows={len(values)}, seeds={len(seeds)}"
                )
            array = np.asarray(values, dtype=float)
            cells.append(
                {
                    "updates": update_count,
                    "gap_events": gap,
                    "seed_gains": values,
                    "mean_affected_gain": float(np.mean(array)),
                    "minimum_seed_gain": float(np.min(array)),
                    "maximum_seed_gain": float(np.max(array)),
                }
            )
    paired_cells = int(_nested(report, "summary.paired_cells"))
    if len(rows) != paired_cells:
        raise ValueError(
            "E13c paired row count disagrees with canonical report: "
            f"rows={len(rows)}, report={paired_cells}"
        )
    return {
        "updates": updates,
        "gap_events": gaps,
        "seeds": seeds,
        "cells": cells,
        "overall_mean_affected_gain": float(
            _nested(report, "summary.mean_affected_gain")
        ),
        "stress_mean_affected_gain": float(
            _nested(report, "summary.stress_mean_affected_gain")
        ),
        "sign_flip_p": float(_nested(report, "summary.sign_flip_p")),
    }


def derive_e20(sources: dict[str, Path]) -> dict[str, Any]:
    report = load_json(sources["e20_report"])
    rows = load_jsonl(sources["e20_metrics"])
    if (
        report["claim_gate"]["status"]
        != "SUPPORTED_CONTROLLED_SYSTEMS_PROXY"
        or report["evidence_tier"] != "CONTROLLED_SYSTEMS_PROXY"
        or report["scientific_evidence"] is not False
    ):
        raise ValueError("E20 report boundary/status is inconsistent")
    indexed = {
        (str(row["policy"]), int(row["query_count"])): row
        for row in rows
    }
    if len(indexed) != 28:
        raise ValueError("E20 source must contain 28 unique policy×m rows")
    policies = tuple(str(value) for value in report["registered_policies"])
    query_counts = tuple(
        int(value) for value in report["registered_query_counts"]
    )
    expected = {
        (policy, query_count)
        for policy in policies
        for query_count in query_counts
    }
    if set(indexed) != expected:
        raise ValueError("E20 registered source grid is incomplete")
    return {
        "minimum_m_by_baseline": {
            str(key): int(value)
            for key, value in report["primary_estimand"][
                "by_baseline"
            ].items()
        },
        "query_counts": list(query_counts),
        "policies": list(policies),
        "latency_total_seconds_median": {
            policy: {
                str(query_count): float(
                    indexed[(policy, query_count)][
                        "latency_total_seconds_median"
                    ]
                )
                for query_count in query_counts
            }
            for policy in policies
        },
        "maximum_affected_correction_mse": max(
            float(row["affected_correction_mse"]) for row in rows
        ),
        "maximum_retention_mse": max(
            float(row["retention_mse"]) for row in rows
        ),
        "evidence_tier": str(report["evidence_tier"]),
        "scientific_evidence": bool(report["scientific_evidence"]),
    }


def derive_e19(sources: dict[str, Path]) -> dict[str, Any]:
    report = load_json(sources["e19_report"])
    rows = load_jsonl(sources["e19_seed_contrasts"])
    if (
        report["status"] != "PASS"
        or report["claim_gate"]["status"] != "SUPPORTED"
        or report["evidence_tier"] != "CONTROLLED_REFERENCE"
        or report["scientific_evidence"] is not False
    ):
        raise ValueError("E19 report boundary/status is inconsistent")
    expected_seeds = tuple(
        int(seed) for seed in report["source_contract"]["required_seeds"]
    )
    indexed = {int(row["seed"]): row for row in rows}
    if tuple(sorted(indexed)) != tuple(sorted(expected_seeds)):
        raise ValueError("E19 seed-level contrast grid is incomplete")
    pattern = report["summary"]["pattern"]
    mappings = {
        "b_separate_address_recovery": "b_separate_address_gain",
        "c_state_read_recovery": "c_state_read_gain",
        "d_full_only_maintenance": "d_full_only_gain",
    }
    gains: dict[str, list[float]] = {}
    for name, metric in mappings.items():
        seed_values = [float(indexed[seed][metric]) for seed in expected_seeds]
        observed = float(np.mean(np.asarray(seed_values, dtype=float)))
        frozen = float(pattern[name]["mean_gain"])
        if not np.isclose(observed, frozen, rtol=0.0, atol=1e-12):
            raise ValueError(f"E19 derived mean mismatch for {name}")
        gains[name] = seed_values
    return {
        "seeds": list(expected_seeds),
        "seed_gains": gains,
        "mean_gains": {
            name: float(pattern[name]["mean_gain"]) for name in mappings
        },
        "maximum_capable_affected_mse": float(
            report["summary"]["maximum_capable_affected_mse"]
        ),
        "minimum_capable_address_accuracy": float(
            report["summary"]["minimum_capable_address_accuracy"]
        ),
        "maximum_capable_candidate_mse": float(
            report["summary"]["maximum_capable_candidate_mse"]
        ),
        "maximum_retention_degradation": float(
            report["summary"]["maximum_retention_degradation"]
        ),
        "evidence_tier": str(report["evidence_tier"]),
        "scientific_evidence": bool(report["scientific_evidence"]),
    }


class SVG:
    def __init__(self, width: int, height: int, *, title: str) -> None:
        self.width = width
        self.height = height
        self.items = [
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" '
                f'width="{width}" height="{height}" '
                f'viewBox="0 0 {width} {height}" role="img" '
                f'aria-labelledby="title desc">'
            ),
            f"<title id=\"title\">{escape(title)}</title>",
            (
                "<desc id=\"desc\">Generated from hash-verified canonical "
                "CATENA artifacts.</desc>"
            ),
            '<rect width="100%" height="100%" fill="white"/>',
        ]

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        stroke: str = INK,
        width: float = 1.0,
        dash: str | None = None,
    ) -> None:
        dash_attr = "" if dash is None else f' stroke-dasharray="{dash}"'
        self.items.append(
            f'<line x1="{x1:.3f}" y1="{y1:.3f}" '
            f'x2="{x2:.3f}" y2="{y2:.3f}" '
            f'stroke="{stroke}" stroke-width="{width:.3f}"{dash_attr}/>'
        )

    def rect(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        *,
        fill: str,
        stroke: str = "none",
        stroke_width: float = 0.0,
        radius: float = 0.0,
    ) -> None:
        self.items.append(
            f'<rect x="{x:.3f}" y="{y:.3f}" width="{width:.3f}" '
            f'height="{height:.3f}" rx="{radius:.3f}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{stroke_width:.3f}"/>'
        )

    def circle(
        self,
        x: float,
        y: float,
        radius: float,
        *,
        fill: str,
        stroke: str = "white",
        stroke_width: float = 0.8,
        opacity: float = 1.0,
    ) -> None:
        self.items.append(
            f'<circle cx="{x:.3f}" cy="{y:.3f}" r="{radius:.3f}" '
            f'fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{stroke_width:.3f}" opacity="{opacity:.3f}"/>'
        )

    def text(
        self,
        x: float,
        y: float,
        value: object,
        *,
        size: float = 12.0,
        anchor: str = "start",
        weight: str = "normal",
        fill: str = INK,
        rotate: float | None = None,
    ) -> None:
        transform = (
            ""
            if rotate is None
            else f' transform="rotate({rotate:.3f} {x:.3f} {y:.3f})"'
        )
        self.items.append(
            f'<text x="{x:.3f}" y="{y:.3f}" font-family="Arial, '
            f'DejaVu Sans, sans-serif" font-size="{size:.3f}" '
            f'font-weight="{weight}" fill="{fill}" '
            f'text-anchor="{anchor}"{transform}>{escape(str(value))}</text>'
        )

    def finish(self) -> str:
        return "\n".join([*self.items, "</svg>", ""])


def _scale(
    value: float,
    lower: float,
    upper: float,
    output_lower: float,
    output_upper: float,
) -> float:
    if upper <= lower:
        return (output_lower + output_upper) / 2.0
    ratio = (value - lower) / (upper - lower)
    return output_lower + ratio * (output_upper - output_lower)


def _axes(
    svg: SVG,
    left: float,
    top: float,
    width: float,
    height: float,
) -> None:
    svg.line(left, top + height, left + width, top + height, width=1.2)
    svg.line(left, top, left, top + height, width=1.2)


def render_figure1(data: dict[str, Any]) -> str:
    svg = SVG(1440, 440, title="Geometry predicts minimal control architecture")
    panel_offsets = (20.0, 500.0, 980.0)

    # Panel A: H1 predictor comparison.
    x0 = panel_offsets[0]
    svg.text(x0, 24, "A", size=18, weight="bold")
    svg.text(x0 + 26, 24, "Behavioral reachability", size=16, weight="bold")
    left, top, width, height = x0 + 58, 58, 350, 280
    _axes(svg, left, top, width, height)
    values = data["h1"]["predictor_conditional_oos_r2"]
    labels = list(values)
    array = np.asarray([values[label] for label in labels], dtype=float)
    lower = max(0.0, math.floor((float(np.min(array)) - 0.01) * 100) / 100)
    upper = 1.0
    for tick in np.linspace(lower, upper, 3):
        y = _scale(float(tick), lower, upper, top + height, top)
        svg.line(left, y, left + width, y, stroke=LIGHT_GRAY)
        svg.text(left - 8, y + 4, f"{tick:.3f}", size=10, anchor="end")
    colors = (NAVY, ORANGE, GRAY)
    bar_width = 72
    for index, (label, color) in enumerate(zip(labels, colors, strict=True)):
        center = left + (index + 0.5) * width / len(labels)
        y = _scale(float(values[label]), lower, upper, top + height, top)
        svg.rect(
            center - bar_width / 2,
            y,
            bar_width,
            top + height - y,
            fill=color,
            radius=2,
        )
        svg.text(center, y - 8, f"{values[label]:.4f}", size=11, anchor="middle")
        short = (
            "Behavioral"
            if label.startswith("behavioral")
            else "State feasible"
            if label.startswith("state feasible")
            else "State span"
        )
        svg.text(center, top + height + 20, short, size=10, anchor="middle")
    svg.text(
        left + width / 2,
        top + height + 48,
        (
            "unseen-geometry conditional R²; "
            f"calibration slope={data['h1']['calibration_slope']:.4f}"
        ),
        size=10,
        anchor="middle",
        fill=GRAY,
    )

    # Panel B: E10b rank tracking.
    x0 = panel_offsets[1]
    svg.text(x0, 24, "B", size=18, weight="bold")
    svg.text(x0 + 26, 24, "Learned control-rank scaling", size=16, weight="bold")
    left, top, width, height = x0 + 58, 58, 350, 280
    _axes(svg, left, top, width, height)
    points = data["e10b"]["points"]
    rank_values = sorted(
        {
            int(point["intrinsic_rank"])
            for point in points
        }
        | {
            int(point["minimum_qualifying_rank"])
            for point in points
        }
    )
    log_values = [math.log2(value) for value in rank_values]
    log_min, log_max = min(log_values), max(log_values)
    svg.line(left, top + height, left + width, top, stroke=GRAY, dash="5,4")
    seed_order = {
        seed: index
        for index, seed in enumerate(sorted({point["seed"] for point in points}))
    }
    seed_count = max(len(seed_order), 1)
    for point in points:
        log_x = math.log2(point["intrinsic_rank"])
        log_y = math.log2(point["minimum_qualifying_rank"])
        jitter = (seed_order[point["seed"]] - (seed_count - 1) / 2) * 1.2
        x = _scale(log_x, log_min, log_max, left, left + width) + jitter
        y = _scale(log_y, log_min, log_max, top + height, top)
        svg.circle(x, y, 4.2, fill=BLUE, opacity=0.72)
    for rank, log_rank in zip(rank_values, log_values, strict=True):
        x = _scale(log_rank, log_min, log_max, left, left + width)
        y = _scale(log_rank, log_min, log_max, top + height, top)
        svg.text(x, top + height + 19, rank, size=10, anchor="middle")
        svg.text(left - 8, y + 4, rank, size=10, anchor="end")
    svg.text(left + width / 2, top + height + 42, "Intrinsic rank", size=11, anchor="middle")
    svg.text(
        left - 43,
        top + height / 2,
        "Minimum learned rank",
        size=11,
        anchor="middle",
        rotate=-90,
    )
    svg.text(
        left + width / 2,
        top + 16,
        (
            f"match fraction={data['e10b']['rank_match_fraction']:.3f}; "
            f"seeds={data['e10b']['seed_count']}"
        ),
        size=10,
        anchor="middle",
        fill=GRAY,
    )

    # Panel C: E03b JD calibration.
    x0 = panel_offsets[2]
    svg.text(x0, 24, "C", size=18, weight="bold")
    svg.text(x0 + 26, 24, "Joint-diagonalization calibration", size=16, weight="bold")
    left, top, width, height = x0 + 62, 58, 340, 280
    _axes(svg, left, top, width, height)
    points = data["e03b"]["points"]
    maximum = max(
        max(point["analytic_regret"], point["empirical_application_error"])
        for point in points
    ) * 1.04
    svg.line(left, top + height, left + width, top, stroke=GRAY, dash="5,4")
    bins = sorted({point["bin"] for point in points})
    palette = (NAVY, BLUE, GREEN, ORANGE, PURPLE, GRAY)
    colors_by_bin = {
        bin_name: palette[index % len(palette)]
        for index, bin_name in enumerate(bins)
    }
    for point in points:
        x = _scale(point["analytic_regret"], 0.0, maximum, left, left + width)
        y = _scale(
            point["empirical_application_error"],
            0.0,
            maximum,
            top + height,
            top,
        )
        svg.circle(x, y, 4.2, fill=colors_by_bin[point["bin"]], opacity=0.82)
    for tick in np.linspace(0.0, maximum, 4):
        x = _scale(float(tick), 0.0, maximum, left, left + width)
        y = _scale(float(tick), 0.0, maximum, top + height, top)
        svg.text(x, top + height + 19, f"{tick:.3f}", size=9, anchor="middle")
        svg.text(left - 8, y + 3, f"{tick:.3f}", size=9, anchor="end")
    svg.text(left + width / 2, top + height + 42, "Analytic JD regret", size=11, anchor="middle")
    svg.text(
        left - 46,
        top + height / 2,
        "Application error",
        size=11,
        anchor="middle",
        rotate=-90,
    )
    svg.text(
        left + width / 2,
        top + 16,
        (
            f"R²={data['e03b']['r2']:.6f}; "
            f"slope={data['e03b']['slope']:.5f}"
        ),
        size=10,
        anchor="middle",
        fill=GRAY,
    )
    svg.text(
        720,
        425,
        "Dashed lines denote identity; every plotted value is read from a hash-pinned artifact.",
        size=10,
        anchor="middle",
        fill=GRAY,
    )
    return svg.finish()


def _render_lattice_panel(
    *,
    svg: SVG,
    contrasts: list[dict[str, Any]],
    x0: float,
    panel_label: str,
    title: str,
    upper: float,
    footer: str,
) -> None:
    svg.text(x0, 28, panel_label, size=18, weight="bold")
    svg.text(x0 + 27, 28, title, size=15, weight="bold")
    left, top, width, height = x0 + 58, 66.0, 570.0, 330.0
    _axes(svg, left, top, width, height)
    for tick in np.linspace(0.0, upper, 5):
        y = _scale(float(tick), 0.0, upper, top + height, top)
        svg.line(left, y, left + width, y, stroke=LIGHT_GRAY)
        svg.text(left - 8, y + 4, f"{tick:.3f}", size=9, anchor="end")
    slot = width / len(contrasts)
    bar_width = slot * 0.48
    palette = (NAVY, BLUE, GREEN, ORANGE)
    for index, (contrast, color) in enumerate(
        zip(contrasts, palette, strict=True)
    ):
        center = left + (index + 0.5) * slot
        value = float(contrast["mean_selective_gain"])
        y = _scale(value, 0.0, upper, top + height, top)
        svg.rect(
            center - bar_width / 2,
            y,
            bar_width,
            top + height - y,
            fill=color,
            radius=3,
        )
        gains = contrast["seed_gains"]
        for seed_index, seed_gain in enumerate(gains):
            jitter = (seed_index - (len(gains) - 1) / 2) * 4.5
            dot_y = _scale(
                float(seed_gain),
                0.0,
                upper,
                top + height,
                top,
            )
            svg.circle(center + jitter, dot_y, 3.0, fill=INK, opacity=0.72)
        svg.text(center, y - 9, f"{value:.5f}", size=10, anchor="middle")
        svg.text(
            center,
            top + height + 22,
            contrast["label"],
            size=10,
            anchor="middle",
            weight="bold",
        )
        svg.text(
            center,
            top + height + 40,
            f"{contrast['simpler']} →",
            size=8,
            anchor="middle",
            fill=GRAY,
        )
        svg.text(
            center,
            top + height + 54,
            contrast["richer"],
            size=8,
            anchor="middle",
            fill=GRAY,
        )
        svg.text(
            center,
            top + height + 72,
            (
                "simpler Δ≤"
                f"{contrast['max_simpler_task_degradation']:.1e}"
            ),
            size=8,
            anchor="middle",
            fill=GRAY,
        )
    svg.text(
        left + width / 2,
        top + height + 102,
        footer,
        size=9,
        anchor="middle",
        fill=GRAY,
    )


def _render_figure2_with_sequence(data: dict[str, Any]) -> str:
    sequence = data["sequence"]
    static_contrasts = data["contrasts"]
    sequence_contrasts = sequence["contrasts"]
    all_means = np.asarray(
        [
            float(contrast["mean_selective_gain"])
            for contrast in (*static_contrasts, *sequence_contrasts)
        ],
        dtype=float,
    )
    upper = max(float(np.max(all_means)) * 1.18, 1e-12)
    svg = SVG(
        1440,
        560,
        title="Static and sequence architecture-demand control lattice",
    )
    _render_lattice_panel(
        svg=svg,
        contrasts=static_contrasts,
        x0=24.0,
        panel_label="A",
        title="Static controlled lattice (E12)",
        upper=upper,
        footer="Dots: paired E12 seeds; bars: frozen seed means.",
    )
    _render_lattice_panel(
        svg=svg,
        contrasts=sequence_contrasts,
        x0=734.0,
        panel_label="B",
        title="Repeated structured sequences (E18b)",
        upper=upper,
        footer=(
            "Dots: paired E18 seeds; stress direction is a report-level "
            "guardrail without a separate SESOI."
        ),
    )
    svg.text(
        20,
        245,
        "Affected-MSE gain (simpler − richer)",
        size=12,
        anchor="middle",
        rotate=-90,
    )
    svg.text(
        720,
        548,
        (
            "Panels share one y-axis scale. E18 supplies oracle "
            "address/candidate/demand descriptors and a model-visible "
            "verified-event bit."
        ),
        size=10,
        anchor="middle",
        fill=GRAY,
    )
    return svg.finish()


def render_figure2(data: dict[str, Any]) -> str:
    if "sequence" in data:
        return _render_figure2_with_sequence(data)
    svg = SVG(960, 540, title="Architecture-demand control lattice")
    svg.text(30, 30, "Architecture freedom is selectively useful", size=18, weight="bold")
    left, top, width, height = 92.0, 66.0, 820.0, 340.0
    _axes(svg, left, top, width, height)
    contrasts = data["contrasts"]
    means = np.asarray(
        [contrast["mean_selective_gain"] for contrast in contrasts],
        dtype=float,
    )
    upper = float(np.max(means)) * 1.18
    for tick in np.linspace(0.0, upper, 5):
        y = _scale(float(tick), 0.0, upper, top + height, top)
        svg.line(left, y, left + width, y, stroke=LIGHT_GRAY)
        svg.text(left - 9, y + 4, f"{tick:.3f}", size=10, anchor="end")
    slot = width / len(contrasts)
    bar_width = slot * 0.48
    palette = (NAVY, BLUE, GREEN, ORANGE)
    for index, (contrast, color) in enumerate(
        zip(contrasts, palette, strict=True)
    ):
        center = left + (index + 0.5) * slot
        value = float(contrast["mean_selective_gain"])
        y = _scale(value, 0.0, upper, top + height, top)
        svg.rect(
            center - bar_width / 2,
            y,
            bar_width,
            top + height - y,
            fill=color,
            radius=3,
        )
        gains = contrast["seed_gains"]
        for seed_index, seed_gain in enumerate(gains):
            jitter = (seed_index - (len(gains) - 1) / 2) * 5.0
            dot_y = _scale(
                float(seed_gain),
                0.0,
                upper,
                top + height,
                top,
            )
            svg.circle(center + jitter, dot_y, 3.2, fill=INK, opacity=0.72)
        svg.text(center, y - 10, f"{value:.5f}", size=11, anchor="middle")
        svg.text(
            center,
            top + height + 22,
            contrast["label"],
            size=11,
            anchor="middle",
            weight="bold",
        )
        svg.text(
            center,
            top + height + 40,
            f"{contrast['simpler']} →",
            size=9,
            anchor="middle",
            fill=GRAY,
        )
        svg.text(
            center,
            top + height + 55,
            contrast["richer"],
            size=9,
            anchor="middle",
            fill=GRAY,
        )
        svg.text(
            center,
            top + height + 77,
            (
                "max simpler-task Δ="
                f"{contrast['max_simpler_task_degradation']:.2e}"
            ),
            size=9,
            anchor="middle",
            fill=GRAY,
        )
    svg.text(
        24,
        top + height / 2,
        "Affected-MSE gain (simpler − richer)",
        size=12,
        anchor="middle",
        rotate=-90,
    )
    svg.text(
        480,
        522,
        "Bars are canonical seed means; black points are the paired training seeds.",
        size=10,
        anchor="middle",
        fill=GRAY,
    )
    return svg.finish()


def _mix_color(low: tuple[int, int, int], high: tuple[int, int, int], t: float) -> str:
    clipped = min(max(t, 0.0), 1.0)
    rgb = tuple(
        round(low[index] + clipped * (high[index] - low[index]))
        for index in range(3)
    )
    return "#" + "".join(f"{value:02x}" for value in rgb)


def render_figure3(data: dict[str, Any]) -> str:
    svg = SVG(920, 540, title="Structured sequence transfer across updates and gaps")
    svg.text(
        28,
        30,
        "Sequence transfer across repeated updates and distractor gaps",
        size=18,
        weight="bold",
    )
    updates = data["updates"]
    gaps = data["gap_events"]
    cells = {
        (cell["updates"], cell["gap_events"]): cell
        for cell in data["cells"]
    }
    values = np.asarray(
        [cell["mean_affected_gain"] for cell in data["cells"]],
        dtype=float,
    )
    lower, upper = float(np.min(values)), float(np.max(values))
    left, top = 164.0, 88.0
    cell_width, cell_height = 150.0, 104.0
    for column, gap in enumerate(gaps):
        x = left + (column + 0.5) * cell_width
        svg.text(x, top - 22, gap, size=12, anchor="middle", weight="bold")
    svg.text(
        left + len(gaps) * cell_width / 2,
        top - 48,
        "Distractor events",
        size=12,
        anchor="middle",
    )
    for row_index, update_count in enumerate(updates):
        y_center = top + (row_index + 0.5) * cell_height
        svg.text(left - 20, y_center + 4, update_count, size=12, anchor="end", weight="bold")
        for column, gap in enumerate(gaps):
            cell = cells[(update_count, gap)]
            value = float(cell["mean_affected_gain"])
            ratio = 0.5 if upper <= lower else (value - lower) / (upper - lower)
            fill = _mix_color((232, 242, 247), (31, 78, 121), ratio)
            x = left + column * cell_width
            y = top + row_index * cell_height
            svg.rect(
                x,
                y,
                cell_width - 4,
                cell_height - 4,
                fill=fill,
                stroke="white",
                stroke_width=2,
                radius=4,
            )
            text_color = "white" if ratio > 0.52 else INK
            svg.text(
                x + (cell_width - 4) / 2,
                y + 43,
                f"{value * 1000:.4f}",
                size=16,
                anchor="middle",
                weight="bold",
                fill=text_color,
            )
            svg.text(
                x + (cell_width - 4) / 2,
                y + 67,
                f"range {cell['minimum_seed_gain'] * 1000:.3f}–"
                f"{cell['maximum_seed_gain'] * 1000:.3f}",
                size=9,
                anchor="middle",
                fill=text_color,
            )
    svg.text(
        54,
        top + len(updates) * cell_height / 2,
        "Sequential updates",
        size=12,
        anchor="middle",
        rotate=-90,
    )
    legend_x = left + len(gaps) * cell_width + 34
    legend_y = top
    steps = 80
    for index in range(steps):
        ratio = index / max(steps - 1, 1)
        y = legend_y + (steps - 1 - index) * 2.6
        svg.rect(
            legend_x,
            y,
            18,
            2.8,
            fill=_mix_color((232, 242, 247), (31, 78, 121), ratio),
        )
    svg.text(legend_x + 27, legend_y + 8, f"{upper * 1000:.4f}", size=9)
    svg.text(
        legend_x + 27,
        legend_y + steps * 2.6,
        f"{lower * 1000:.4f}",
        size=9,
    )
    svg.text(
        legend_x + 9,
        legend_y + steps * 2.6 + 26,
        "gain ×10⁻³",
        size=9,
        anchor="middle",
        rotate=-90,
    )
    svg.text(
        left + len(gaps) * cell_width / 2,
        438,
        (
            f"Overall mean={data['overall_mean_affected_gain'] * 1000:.4f}×10⁻³; "
            f"registered stress mean={data['stress_mean_affected_gain'] * 1000:.4f}×10⁻³"
        ),
        size=11,
        anchor="middle",
        fill=GRAY,
    )
    svg.text(
        460,
        515,
        (
            "Cell labels are five-seed means of tied − dual affected MSE; "
            "ranges show the seed extrema."
        ),
        size=10,
        anchor="middle",
        fill=GRAY,
    )
    return svg.finish()


def write_results_macros(
    path: Path,
    figure1: dict[str, Any],
    figure2: dict[str, Any],
    figure3: dict[str, Any],
    e19: dict[str, Any],
    e20: dict[str, Any],
) -> None:
    h1 = figure1["h1"]
    e10 = figure1["e10b"]
    e03 = figure1["e03b"]
    e12_means = {
        item["family"]: item["mean_selective_gain"]
        for item in figure2["contrasts"]
    }
    e20_break_even = e20["minimum_m_by_baseline"]
    e20_latency = e20["latency_total_seconds_median"]
    e19_gains = e19["mean_gains"]
    lines = [
        "# Generated Result Macros",
        "",
        "Do not edit manually. Values are regenerated from the pinned source manifest.",
        "",
        "| Token | Value |",
        "|---|---:|",
        (
            "| `H1_BEHAVIORAL_OOS_R2` | "
            f"{h1['predictor_conditional_oos_r2']['behavioral feasible regret']:.9f} |"
        ),
        f"| `H1_CALIBRATION_SLOPE` | {h1['calibration_slope']:.9f} |",
        f"| `E10B_RANK_MATCH_FRACTION` | {e10['rank_match_fraction']:.9f} |",
        f"| `E03B_JD_R2` | {e03['r2']:.9f} |",
        f"| `E03B_JD_SLOPE` | {e03['slope']:.9f} |",
        (
            "| `E12_MAGNITUDE_GAIN` | "
            f"{e12_means['magnitude_factorization']:.12f} |"
        ),
        (
            "| `E12_VALUE_GAIN` | "
            f"{e12_means['value_granularity']:.12f} |"
        ),
        (
            "| `E12_ADDRESS_GAIN` | "
            f"{e12_means['address_decoupling']:.12f} |"
        ),
        (
            "| `E12_STATE_GAIN` | "
            f"{e12_means['state_conditioning']:.12f} |"
        ),
        (
            "| `E13C_OVERALL_GAIN` | "
            f"{figure3['overall_mean_affected_gain']:.12f} |"
        ),
        (
            "| `E13C_STRESS_GAIN` | "
            f"{figure3['stress_mean_affected_gain']:.12f} |"
        ),
        (
            "| `E19_SEPARATE_ADDRESS_GAIN` | "
            f"{e19_gains['b_separate_address_recovery']:.12f} |"
        ),
        (
            "| `E19_STATE_READ_GAIN` | "
            f"{e19_gains['c_state_read_recovery']:.12f} |"
        ),
        (
            "| `E19_FULL_ONLY_GAIN` | "
            f"{e19_gains['d_full_only_maintenance']:.12f} |"
        ),
        (
            "| `E20_BREAK_EVEN_EXTERNAL_PER_QUERY` | "
            f"{e20_break_even['external_canonical_state_per_query']} |"
        ),
        (
            "| `E20_BREAK_EVEN_COMPACT_CACHE` | "
            f"{e20_break_even['retrieve_once_cached_compact_snapshot']} |"
        ),
        (
            "| `E20_BREAK_EVEN_FULL_REFRESH` | "
            f"{e20_break_even['full_refresh']} |"
        ),
        (
            "| `E20_INTERNAL_M1_MICROSECONDS` | "
            f"{e20_latency['one_time_internal_assimilation']['1'] * 1e6:.6f} |"
        ),
        (
            "| `E20_INTERNAL_M64_MICROSECONDS` | "
            f"{e20_latency['one_time_internal_assimilation']['64'] * 1e6:.6f} |"
        ),
        "",
    ]
    sequence = figure2.get("sequence")
    if sequence is not None:
        sequence_means = {
            item["family"]: item["mean_selective_gain"]
            for item in sequence["contrasts"]
        }
        sequence_macros = [
            (
                "| `E18_MAGNITUDE_GAIN` | "
                f"{sequence_means['magnitude_factorization']:.12f} |"
            ),
            (
                "| `E18_VALUE_GAIN` | "
                f"{sequence_means['value_granularity']:.12f} |"
            ),
            (
                "| `E18_ADDRESS_GAIN` | "
                f"{sequence_means['address_decoupling']:.12f} |"
            ),
            (
                "| `E18_STATE_GAIN` | "
                f"{sequence_means['state_conditioning']:.12f} |"
            ),
        ]
        lines[-1:-1] = sequence_macros
    path.write_text("\n".join(lines), encoding="utf-8")


def _freeze_record(
    artifact_root: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    resolved_sources: dict[str, Path],
    output_dir: Path,
    generated_paths: Iterable[Path],
) -> dict[str, Any]:
    source_records: dict[str, Any] = {}
    for section in ("data_sources", "provenance_anchors"):
        for name, record in manifest[section].items():
            path = resolved_sources[name]
            source_records[name] = {
                "section": section,
                "path": str(path),
                "relative_path": str(record["path"]),
                "sha256": file_sha256(path),
            }
    outputs = {
        path.relative_to(output_dir).as_posix(): {
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(generated_paths)
    }
    return {
        "schema_version": 1,
        "mode": "DETERMINISTIC_FIGURE_SOURCE_FREEZE",
        "artifact_root": str(artifact_root.resolve()),
        "source_manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": file_sha256(manifest_path),
        },
        "generator": {
            "path": str(Path(__file__).resolve()),
            "sha256": file_sha256(Path(__file__).resolve()),
            "numpy_version": np.__version__,
        },
        "validated_sources": source_records,
        "generated_outputs": outputs,
    }


def generate(
    *,
    artifact_root: Path,
    manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported source manifest schema")
    sources = validate_sources(artifact_root, manifest)
    figure1 = derive_figure1(sources)
    figure2 = derive_figure2(sources)
    figure3 = derive_figure3(sources)
    e19 = derive_e19(sources)
    e20 = derive_e20(sources)

    output_dir.mkdir(parents=True, exist_ok=True)
    source_data_dir = output_dir / "source_data"
    source_data_dir.mkdir(parents=True, exist_ok=True)
    generated_paths = [
        source_data_dir / "figure1.json",
        source_data_dir / "figure2.json",
        source_data_dir / "figure3.json",
        source_data_dir / "e19.json",
        source_data_dir / "e20.json",
        output_dir / "figure1_geometry.svg",
        output_dir / "figure2_control_lattice.svg",
        output_dir / "figure3_sequence_transfer.svg",
        output_dir / "RESULTS_MACROS.md",
    ]
    write_json(generated_paths[0], figure1)
    write_json(generated_paths[1], figure2)
    write_json(generated_paths[2], figure3)
    write_json(generated_paths[3], e19)
    write_json(generated_paths[4], e20)
    generated_paths[5].write_text(render_figure1(figure1), encoding="utf-8")
    generated_paths[6].write_text(render_figure2(figure2), encoding="utf-8")
    generated_paths[7].write_text(render_figure3(figure3), encoding="utf-8")
    write_results_macros(
        generated_paths[8],
        figure1,
        figure2,
        figure3,
        e19,
        e20,
    )

    freeze = _freeze_record(
        artifact_root,
        manifest_path,
        manifest,
        sources,
        output_dir,
        generated_paths,
    )
    freeze_path = output_dir / "source_data_freeze.json"
    write_json(freeze_path, freeze)
    return {
        "output_dir": str(output_dir.resolve()),
        "source_data_freeze": str(freeze_path.resolve()),
        "generated_outputs": freeze["generated_outputs"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate CATENA long-paper SVG figures from canonical artifacts."
    )
    parser.add_argument(
        "--artifact-root",
        default="/data/minjun_dev/CATENA/artifacts",
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    result = generate(
        artifact_root=Path(args.artifact_root),
        manifest_path=Path(args.manifest),
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
