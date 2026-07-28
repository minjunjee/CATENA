from __future__ import annotations

import json
from pathlib import Path

import pytest

from catena.core.config import load_config
from experiments.e13c_transactional_sequence_aggregate import (
    FIXED_GAPS,
    FIXED_SEEDS,
    FIXED_UPDATES,
    FIXED_VARIANTS,
    aggregate_paired_rows,
    collect_e13b_sources,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_source_run(
    root: Path,
    *,
    seed: int,
    variant: str,
    run_suffix: str = "",
    status: str = "PASS",
    drop_last_cell: bool = False,
) -> None:
    run_dir = (
        root
        / "e13b_transactional_sequence_memory"
        / f"run-{seed}-{variant}{run_suffix}"
    )
    checkpoint = run_dir / "checkpoints" / f"{variant}_seed{seed}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(f"checkpoint:{seed}:{variant}".encode())
    rows = []
    for updates in FIXED_UPDATES:
        for gap in FIXED_GAPS:
            rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "updates": updates,
                    "gap_events": gap,
                    "checkpoint": str(checkpoint.resolve()),
                    "affected_mse": 0.01 if variant == "tied" else 0.005,
                    "retention_mse": 0.0,
                    "old_rule_residual": 0.01 if variant == "tied" else 0.005,
                    "entity_exact_match": 0.90 if variant == "tied" else 0.99,
                }
            )
    if drop_last_cell:
        rows.pop()
    metrics = run_dir / "sequence_main_metrics.jsonl"
    metrics.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    _write_json(
        run_dir / "run_manifest.json",
        {
            "experiment_id": "e13b_transactional_sequence_memory",
            "config": load_config(
                "configs/e13b_transactional_sequence_memory.yaml"
            ),
        },
    )
    _write_json(
        run_dir / "report.json",
        {
            "status": status,
            "rows": len(rows),
            "claim_gate": {"status": "PENDING_AGGREGATE"},
        },
    )


def _complete_sources(root: Path) -> None:
    for seed in FIXED_SEEDS:
        for variant in FIXED_VARIANTS:
            _write_source_run(root, seed=seed, variant=variant)


def test_e13c_requires_five_complete_unique_paired_source_runs(
    tmp_path: Path,
) -> None:
    _complete_sources(tmp_path)
    config = load_config("configs/e13c_transactional_sequence_aggregate.yaml")
    rows, provenance = collect_e13b_sources(
        artifact_root=tmp_path,
        config=config,
        dry_run=False,
    )
    paired, gains, retention = aggregate_paired_rows(
        rows,
        required_seeds=FIXED_SEEDS,
        required_updates=FIXED_UPDATES,
        required_gaps=FIXED_GAPS,
    )

    assert len(provenance) == len(FIXED_SEEDS) * len(FIXED_VARIANTS)
    assert len(paired) == len(FIXED_SEEDS) * len(FIXED_UPDATES) * len(
        FIXED_GAPS
    )
    assert gains == pytest.approx([0.005] * len(FIXED_SEEDS))
    assert retention == pytest.approx([0.0] * len(FIXED_SEEDS))
    assert all(item["checkpoint_sha256"] for item in provenance)


def test_e13c_rejects_duplicate_eligible_seed_variant_run(
    tmp_path: Path,
) -> None:
    _complete_sources(tmp_path)
    _write_source_run(
        tmp_path,
        seed=FIXED_SEEDS[0],
        variant=FIXED_VARIANTS[0],
        run_suffix="-duplicate",
    )
    config = load_config("configs/e13c_transactional_sequence_aggregate.yaml")
    with pytest.raises(RuntimeError, match="Duplicate eligible"):
        collect_e13b_sources(
            artifact_root=tmp_path,
            config=config,
            dry_run=False,
        )


def test_e13c_rejects_incomplete_or_ineligible_source_runs(
    tmp_path: Path,
) -> None:
    _complete_sources(tmp_path)
    broken = (
        tmp_path
        / "e13b_transactional_sequence_memory"
        / f"run-{FIXED_SEEDS[0]}-{FIXED_VARIANTS[0]}"
    )
    rows = (broken / "sequence_main_metrics.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    (broken / "sequence_main_metrics.jsonl").write_text(
        "\n".join(rows[:-1]) + "\n",
        encoding="utf-8",
    )
    report = json.loads((broken / "report.json").read_text(encoding="utf-8"))
    report["rows"] -= 1
    _write_json(broken / "report.json", report)
    config = load_config("configs/e13c_transactional_sequence_aggregate.yaml")
    with pytest.raises(RuntimeError, match="Incomplete E13b grid"):
        collect_e13b_sources(
            artifact_root=tmp_path,
            config=config,
            dry_run=False,
        )

    _write_json(
        broken / "report.json",
        {
            "status": "FAIL",
            "rows": len(rows) - 1,
            "claim_gate": {"status": "PENDING_AGGREGATE"},
        },
    )
    with pytest.raises(RuntimeError, match="status='FAIL'"):
        collect_e13b_sources(
            artifact_root=tmp_path,
            config=config,
            dry_run=False,
        )


def test_five_seed_exact_sign_flip_has_alpha_05_resolution() -> None:
    from catena.eval.postcore_metrics import exact_sign_flip

    assert exact_sign_flip([1.0] * 5, alternative="greater") == pytest.approx(
        1 / 32
    )
    assert exact_sign_flip([1.0] * 3, alternative="greater") == pytest.approx(
        1 / 8
    )
