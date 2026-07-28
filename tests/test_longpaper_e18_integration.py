from __future__ import annotations

import json
from pathlib import Path

import pytest

from papers.transactional_control_algebra_long.scripts import (
    generate_main_figures as figures,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, allow_nan=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _e12_sources(tmp_path: Path) -> dict[str, Path]:
    rows: list[dict] = []
    contrasts: dict[str, dict] = {}
    for index, (family, simpler, richer, _) in enumerate(
        figures.E12_CONTRASTS,
        start=1,
    ):
        gain = index * 0.001
        rows.extend(
            [
                {
                    "seed": 101,
                    "family": family,
                    "freedom": simpler,
                    "affected_mse": 0.02,
                },
                {
                    "seed": 101,
                    "family": family,
                    "freedom": richer,
                    "affected_mse": 0.02 - gain,
                },
            ]
        )
        contrasts[family] = {
            "mean_selective_gain": gain,
            "max_simpler_task_degradation": 0.0,
            "sign_flip_p": 0.5,
            "passed": True,
        }
    report_path = tmp_path / "e12_report.json"
    metrics_path = tmp_path / "e12_metrics.jsonl"
    _write_json(report_path, {"contrasts": contrasts})
    _write_jsonl(metrics_path, rows)
    return {
        "e12_report": report_path,
        "e12_metrics": metrics_path,
    }


def _add_e18_sources(
    tmp_path: Path,
    sources: dict[str, Path],
) -> tuple[Path, Path]:
    seeds = (101, 211, 307, 401, 503)
    rows: list[dict] = []
    contrasts: dict[str, dict] = {}
    for index, (family, simpler, richer, _) in enumerate(
        figures.E12_CONTRASTS,
        start=1,
    ):
        gains = [index * 0.002 + offset * 1e-5 for offset in range(5)]
        stress_gains = [value + 0.0002 for value in gains]
        for seed, gain, stress_gain in zip(
            seeds,
            gains,
            stress_gains,
            strict=True,
        ):
            rows.append(
                {
                    "contrast": family,
                    "seed": seed,
                    "baseline": simpler,
                    "treatment": richer,
                    "target_demand": family,
                    "mean_corresponding_demand_gain": gain,
                    "stress_gain": stress_gain,
                }
            )
        contrasts[family] = {
            "baseline": simpler,
            "treatment": richer,
            "target_demand": family,
            "mean_corresponding_demand_gain": sum(gains) / len(gains),
            "maximum_simpler_demand_degradation": 0.0001,
            "maximum_retention_degradation": 0.00001,
            "stress_positive_seed_fraction": 1.0,
            "stress_sign_flip_p": 0.03125,
            "passed": True,
        }
    report = {
        "status": "PASS",
        "contrasts": contrasts,
        "summary": {
            "source_runs": 25,
            "metric_rows": 1200,
            "paired_contrast_seed_rows": 20,
            "active_path_rows": 100,
            "minimum_active_path_retention_harm": 0.01,
        },
        "claim_gate": {
            "supported": True,
            "conditions": {
                "all_adjacent_contrasts_passed": True,
                "full_paired_grid_passed": True,
                "source_provenance_passed": True,
                "model_visible_active_path_assay_passed": True,
            },
        },
        "evidence_tier": "CONTROLLED_REFERENCE",
        "scientific_evidence": False,
    }
    report_path = tmp_path / "e18_report.json"
    metrics_path = tmp_path / "e18_paired.jsonl"
    _write_json(report_path, report)
    _write_jsonl(metrics_path, rows)
    sources["e18b_report"] = report_path
    sources["e18b_paired_metrics"] = metrics_path
    return report_path, metrics_path


def test_e12_only_branch_remains_deterministic(
    tmp_path: Path,
) -> None:
    data = figures.derive_figure2(_e12_sources(tmp_path))
    rendered = figures.render_figure2(data)

    assert "sequence" not in data
    assert rendered == figures.render_figure2(data)
    assert "Repeated structured sequences (E18b)" not in rendered
    assert 'width="960" height="540"' in rendered


def test_future_e18_pair_adds_deterministic_second_panel(
    tmp_path: Path,
) -> None:
    sources = _e12_sources(tmp_path)
    _add_e18_sources(tmp_path, sources)
    data = figures.derive_figure2(sources)
    assert data["sequence"]["claim_supported"] is True
    assert data["sequence"]["seeds"] == [101, 211, 307, 401, 503]
    assert len(data["sequence"]["contrasts"]) == 4
    rendered = figures.render_figure2(data)
    assert rendered == figures.render_figure2(data)
    assert "Static controlled lattice (E12)" in rendered
    assert "Repeated structured sequences (E18b)" in rendered
    assert "without a separate SESOI" in rendered
    assert "model-visible verified-event bit" in rendered
    assert 'width="1440" height="560"' in rendered


def test_e18_report_json_object_order_is_not_semantic(
    tmp_path: Path,
) -> None:
    sources = _e12_sources(tmp_path)
    report_path, _ = _add_e18_sources(tmp_path, sources)
    report = figures.load_json(report_path)
    report["contrasts"] = {
        key: report["contrasts"][key]
        for key in sorted(report["contrasts"])
    }
    _write_json(report_path, report)

    data = figures.derive_figure2(sources)

    assert [
        contrast["family"] for contrast in data["sequence"]["contrasts"]
    ] == [item[0] for item in figures.E12_CONTRASTS]


def test_e18_sources_are_all_or_none(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    _write_json(report_path, {})
    with pytest.raises(ValueError, match="requires report and paired metrics"):
        figures.derive_e18_sequence_lattice(
            {"e18b_report": report_path}
        )


def test_e18_derived_mean_must_match_report(tmp_path: Path) -> None:
    sources = _e12_sources(tmp_path)
    report_path, _ = _add_e18_sources(tmp_path, sources)
    report = figures.load_json(report_path)
    report["contrasts"]["magnitude_factorization"][
        "mean_corresponding_demand_gain"
    ] += 0.001
    _write_json(report_path, report)
    with pytest.raises(ValueError, match="derived mean mismatch"):
        figures.derive_figure2(sources)
