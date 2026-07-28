from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import pytest
import torch

from catena.core.config import load_config
from catena.core.io import file_sha256
from catena.data.transactional_sequence_v2 import (
    base_transaction_digest_v2,
    generate_transactional_sequence_batch_v2,
)
from catena.models.sequence_memory_v2 import (
    SequenceControlV2,
    TransactionalSequenceMemoryV2,
)
from experiments.e14_plan_continuation import (
    FIXED_GAPS,
    FIXED_SEEDS,
    FIXED_UPDATES,
    FIXED_VARIANTS,
    STATIC_EVIDENCE_BOUNDARY,
    _canonical_sha256,
    assess_plan_gate,
    resolve_e13c_dependency,
)

E14_CONFIG = load_config("configs/e14_plan_continuation.yaml")
E13A_R1_EXPERIMENT_ID = str(
    E14_CONFIG["dependency"]["calibration_experiment_id"]
)
E13B_EXPERIMENT_ID = str(
    E14_CONFIG["dependency"]["e13b_experiment_id"]
)
E13C_EXPERIMENT_ID = str(
    E14_CONFIG["dependency"]["e13c_experiment_id"]
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _checkpoint_payload(
    *,
    seed: int,
    variant: str,
    source_config: dict,
) -> dict[str, object]:
    torch.manual_seed(seed)
    model = TransactionalSequenceMemoryV2(
        control=SequenceControlV2(variant),
        num_entities=int(source_config["data"]["num_entities"]),
        value_vocab=int(source_config["data"]["value_vocab"]),
        embedding_dim=int(source_config["model"]["embedding_dim"]),
        hidden_dim=int(source_config["model"]["hidden_dim"]),
    )
    return {
        "model": model.state_dict(),
        "variant": variant,
        "seed": seed,
        "config": source_config,
        "model_class": "TransactionalSequenceMemoryV2",
    }


def _build_dependency_tree(
    root: Path,
    *,
    dry_run: bool = False,
    supported: bool | None = None,
) -> tuple[Path, list[dict[str, object]]]:
    e13b_config = load_config(
        str(E14_CONFIG["dependency"]["e13b_config_path"])
    )
    e13c_config = load_config(
        str(E14_CONFIG["dependency"]["e13c_config_path"])
    )
    seeds = FIXED_SEEDS[:1] if dry_run else FIXED_SEEDS
    updates = FIXED_UPDATES[:1] if dry_run else FIXED_UPDATES
    gaps = FIXED_GAPS[:1] if dry_run else FIXED_GAPS
    run_mode = "DRY_RUN" if dry_run else "MAIN"
    status = "DRY_RUN" if dry_run else "PASS"
    gate_status = "DRY_RUN" if dry_run else "PENDING_AGGREGATE"
    provenance: list[dict[str, object]] = []
    calibration_run = root / E13A_R1_EXPERIMENT_ID / "calibration-run"
    calibration_report = calibration_run / "report.json"
    calibration_gate_name = str(
        E14_CONFIG["dependency"]["calibration_gate_field"]
    ).removeprefix("claim_gate.")
    _write_json(
        calibration_report,
        {
            "status": "PASS",
            "claim_gate": {calibration_gate_name: True},
        },
    )
    calibration_manifest = calibration_run / "run_manifest.json"
    _write_json(
        calibration_manifest,
        {
            "experiment_id": E13A_R1_EXPERIMENT_ID,
            "run_mode": "MAIN",
        },
    )
    calibration_dependency = {
        "experiment_id": E13A_R1_EXPERIMENT_ID,
        "run_dir": str(calibration_run.resolve()),
        "report_path": str(calibration_report.resolve()),
        "report_sha256": file_sha256(calibration_report),
        "run_manifest_path": str(calibration_manifest.resolve()),
        "run_manifest_sha256": file_sha256(calibration_manifest),
        "source_config_sha256": file_sha256(
            str(E14_CONFIG["dependency"]["e13b_config_path"])
        ),
    }

    for seed, variant in product(seeds, FIXED_VARIANTS):
        run_dir = root / E13B_EXPERIMENT_ID / f"run-{seed}-{variant}"
        checkpoint = (
            run_dir / "checkpoints" / f"{variant}_seed{seed}.pt"
        )
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            _checkpoint_payload(
                seed=seed,
                variant=variant,
                source_config=e13b_config,
            ),
            checkpoint,
        )
        metrics = run_dir / "sequence_main_metrics.jsonl"
        _write_jsonl(metrics, [{"sealed": True}])
        manifest = run_dir / "run_manifest.json"
        _write_json(
            manifest,
            {
                "experiment_id": E13B_EXPERIMENT_ID,
                "run_mode": run_mode,
                "config": e13b_config,
            },
        )
        report = run_dir / "report.json"
        _write_json(
            report,
            {
                "status": status,
                "variant": variant,
                "seed": seed,
                "calibration_dependency": (
                    None if dry_run else calibration_dependency
                ),
                "claim_gate": {"status": gate_status},
            },
        )
        provenance.append(
            {
                "run_dir": str(run_dir.resolve()),
                "seed": seed,
                "variant": variant,
                "report_path": str(report.resolve()),
                "report_sha256": file_sha256(report),
                "metrics_path": str(metrics.resolve()),
                "metrics_sha256": file_sha256(metrics),
                "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_sha256": file_sha256(checkpoint),
                "source_config_canonical_sha256": _canonical_sha256(
                    e13b_config
                ),
            }
        )

    e13c_run = root / E13C_EXPERIMENT_ID / "sealed-run"
    provenance_path = e13c_run / "source_run_provenance.jsonl"
    _write_jsonl(provenance_path, provenance)
    _write_json(
        e13c_run / "run_manifest.json",
        {
            "experiment_id": E13C_EXPERIMENT_ID,
            "run_mode": run_mode,
            "config": e13c_config,
        },
    )
    _write_json(
        e13c_run / "report.json",
        {
            "status": status,
            "source_runs": provenance,
            "calibration_dependency": (
                None if dry_run else calibration_dependency
            ),
            "source_contract": {
                "fixed_seeds": list(seeds),
                "variants": list(FIXED_VARIANTS),
                "updates": list(updates),
                "gap_events": list(gaps),
                "complete_grid_required": True,
                "unique_run_per_seed_variant_required": True,
                "same_base_transaction_across_gaps_required": True,
                "source_config_path": str(
                    E14_CONFIG["dependency"]["e13b_config_path"]
                ),
                "source_config_sha256": file_sha256(
                    str(E14_CONFIG["dependency"]["e13b_config_path"])
                ),
            },
            "summary": {
                "paired_seeds": len(seeds),
                "paired_cells": len(seeds) * len(updates) * len(gaps),
            },
            "claim_gate": {
                "supported": (
                    (not dry_run) if supported is None else supported
                ),
                "conditions": {
                    "all_registered_conditions_passed": (
                        not dry_run
                        and (
                            True if supported is None else bool(supported)
                        )
                    )
                },
            },
        },
    )
    _write_json(
        root / E13C_EXPERIMENT_ID / "latest.json",
        {"run_dir": e13c_run.name},
    )
    return e13c_run, provenance


def _valid_gate_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed, updates, gap in product(
        FIXED_SEEDS,
        FIXED_UPDATES,
        FIXED_GAPS,
    ):
        rows.append(
            {
                "variant": "dual",
                "training_seed": seed,
                "updates": updates,
                "gap_events": gap,
                "stale_plan_mse": 0.0002,
                "assimilated_plan_mse": 0.0001,
                "plan_correction_gain": 0.0001,
                "affected_stale_plan_mse": 0.004,
                "affected_plan_mse": 0.0025,
                "affected_plan_correction_gain": 0.0015,
                "unaffected_plan_retention_mse": 0.0001,
                "forward_seconds_per_batch": 0.01,
                "affected_entity_count": 10,
                "unaffected_entity_count": 100,
                "base_transaction_digest": (
                    f"{updates:064x}"
                ),
            }
        )
    return rows


def _rewrite_e13c_provenance(
    e13c_run: Path,
    provenance: list[dict[str, object]],
) -> None:
    _write_jsonl(
        e13c_run / "source_run_provenance.jsonl",
        provenance,
    )
    report_path = e13c_run / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["source_runs"] = provenance
    _write_json(report_path, report)


def test_e14_selects_exactly_e13c_sealed_dual_checkpoints(
    tmp_path: Path,
) -> None:
    _, provenance = _build_dependency_tree(tmp_path)
    stray = (
        tmp_path
        / E13B_EXPERIMENT_ID
        / "zzzz-unsealed"
        / "checkpoints"
        / "dual_seed999.pt"
    )
    stray.parent.mkdir(parents=True)
    stray.write_bytes(b"must-not-be-selected")

    dependency, checkpoints = resolve_e13c_dependency(
        artifact_root=tmp_path,
        config=load_config("configs/e14_plan_continuation.yaml"),
        dry_run=False,
    )

    expected_dual_hashes = {
        int(row["seed"]): str(row["checkpoint_sha256"])
        for row in provenance
        if row["variant"] == "dual"
    }
    assert [checkpoint.seed for checkpoint in checkpoints] == list(
        FIXED_SEEDS
    )
    assert all(checkpoint.variant == "dual" for checkpoint in checkpoints)
    assert {
        checkpoint.seed: checkpoint.checkpoint_sha256
        for checkpoint in checkpoints
    } == expected_dual_hashes
    assert dependency["required_seeds"] == list(FIXED_SEEDS)
    assert len(dependency["selected_checkpoints"]) == len(FIXED_SEEDS)
    assert all(checkpoint.checkpoint_path != stray for checkpoint in checkpoints)


def test_e14_blocks_unsupported_or_mutated_e13c_sources(
    tmp_path: Path,
) -> None:
    _build_dependency_tree(tmp_path, supported=False)
    config = load_config("configs/e14_plan_continuation.yaml")
    with pytest.raises(RuntimeError, match="not supported"):
        resolve_e13c_dependency(
            artifact_root=tmp_path,
            config=config,
            dry_run=False,
        )

    _, provenance = _build_dependency_tree(tmp_path, supported=True)
    dual = next(row for row in provenance if row["variant"] == "dual")
    Path(str(dual["checkpoint_path"])).write_bytes(b"mutated")
    with pytest.raises(ValueError, match="checkpoint hash mismatch"):
        resolve_e13c_dependency(
            artifact_root=tmp_path,
            config=config,
            dry_run=False,
        )


def test_e14_rejects_resealed_checkpoint_payload_mismatch(
    tmp_path: Path,
) -> None:
    e13c_run, provenance = _build_dependency_tree(tmp_path)
    dual = next(row for row in provenance if row["variant"] == "dual")
    checkpoint_path = Path(str(dual["checkpoint_path"]))
    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    payload["seed"] = 999
    torch.save(payload, checkpoint_path)
    dual["checkpoint_sha256"] = file_sha256(checkpoint_path)
    _rewrite_e13c_provenance(e13c_run, provenance)

    with pytest.raises(ValueError, match="Checkpoint seed mismatch"):
        resolve_e13c_dependency(
            artifact_root=tmp_path,
            config=load_config("configs/e14_plan_continuation.yaml"),
            dry_run=False,
        )


def test_e14_rejects_incomplete_or_escaping_resealed_provenance(
    tmp_path: Path,
) -> None:
    e13c_run, provenance = _build_dependency_tree(tmp_path)
    removed = provenance.pop()
    _rewrite_e13c_provenance(e13c_run, provenance)
    with pytest.raises(ValueError, match="provenance key mismatch"):
        resolve_e13c_dependency(
            artifact_root=tmp_path,
            config=load_config("configs/e14_plan_continuation.yaml"),
            dry_run=False,
        )

    provenance.append(removed)
    dual = next(row for row in provenance if row["variant"] == "dual")
    dual["checkpoint_path"] = "/tmp/e14-unsealed-checkpoint.pt"
    _rewrite_e13c_provenance(e13c_run, provenance)
    with pytest.raises(ValueError, match="escapes"):
        resolve_e13c_dependency(
            artifact_root=tmp_path,
            config=load_config("configs/e14_plan_continuation.yaml"),
            dry_run=False,
        )


def test_e14_dry_dependency_validates_but_never_requires_support(
    tmp_path: Path,
) -> None:
    _build_dependency_tree(tmp_path, dry_run=True, supported=False)

    dependency, checkpoints = resolve_e13c_dependency(
        artifact_root=tmp_path,
        config=load_config("configs/e14_plan_continuation.yaml"),
        dry_run=True,
    )

    assert dependency["required_seeds"] == [FIXED_SEEDS[0]]
    assert len(checkpoints) == 1
    assert checkpoints[0].variant == "dual"


def test_e14_gate_uses_affected_gain_and_requires_every_seed_cell() -> None:
    rows = _valid_gate_rows()
    result = assess_plan_gate(
        rows,
        required_seeds=FIXED_SEEDS,
        required_updates=FIXED_UPDATES,
        required_gaps=FIXED_GAPS,
        minimum_affected_gain=0.001,
        maximum_retention_mse=0.0005,
        dry_run=False,
    )

    assert result["supported"] is True
    assert result["all_cells_pass"] is True
    assert result["required_cell_count"] == 60
    assert result["seed_level_gain_sign_flip_p"] == pytest.approx(1 / 32)
    assert result["batch_or_episode_used_as_inference_unit"] is False
    # Whole-table gain remains below the historical 0.001 threshold; the
    # prospectively repaired primary estimand is affected-field gain.
    assert max(float(row["plan_correction_gain"]) for row in rows) < 0.001
    dry = assess_plan_gate(
        rows,
        required_seeds=FIXED_SEEDS,
        required_updates=FIXED_UPDATES,
        required_gaps=FIXED_GAPS,
        minimum_affected_gain=0.001,
        maximum_retention_mse=0.0005,
        dry_run=True,
    )
    assert dry["supported"] is False

    rows[0]["affected_plan_correction_gain"] = 0.0009
    failed = assess_plan_gate(
        rows,
        required_seeds=FIXED_SEEDS,
        required_updates=FIXED_UPDATES,
        required_gaps=FIXED_GAPS,
        minimum_affected_gain=0.001,
        maximum_retention_mse=0.0005,
        dry_run=False,
    )
    assert failed["supported"] is False
    assert failed["cells_passing"] == 59
    assert failed["seed_all_cells_pass"][str(FIXED_SEEDS[0])] is False


def test_e14_gate_rejects_missing_duplicate_or_empty_cells() -> None:
    rows = _valid_gate_rows()
    with pytest.raises(ValueError, match="seed/cell contract mismatch"):
        assess_plan_gate(
            rows[:-1],
            required_seeds=FIXED_SEEDS,
            required_updates=FIXED_UPDATES,
            required_gaps=FIXED_GAPS,
            minimum_affected_gain=0.001,
            maximum_retention_mse=0.0005,
            dry_run=False,
        )

    duplicate = [*rows, dict(rows[0])]
    with pytest.raises(ValueError, match="Duplicate"):
        assess_plan_gate(
            duplicate,
            required_seeds=FIXED_SEEDS,
            required_updates=FIXED_UPDATES,
            required_gaps=FIXED_GAPS,
            minimum_affected_gain=0.001,
            maximum_retention_mse=0.0005,
            dry_run=False,
        )

    rows[0]["unaffected_entity_count"] = 0
    with pytest.raises(ValueError, match="no unaffected"):
        assess_plan_gate(
            rows,
            required_seeds=FIXED_SEEDS,
            required_updates=FIXED_UPDATES,
            required_gaps=FIXED_GAPS,
            minimum_affected_gain=0.001,
            maximum_retention_mse=0.0005,
            dry_run=False,
        )


def test_e14_v2_base_transactions_are_paired_across_gaps() -> None:
    shared = {
        "batch_size": 4,
        "num_entities": 8,
        "value_vocab": 16,
        "updates": 4,
        "seed": 1701,
        "device": torch.device("cpu"),
    }
    no_gap = generate_transactional_sequence_batch_v2(
        **shared,
        gap_events=0,
    )
    long_gap = generate_transactional_sequence_batch_v2(
        **shared,
        gap_events=128,
    )

    assert base_transaction_digest_v2(no_gap) == base_transaction_digest_v2(
        long_gap
    )


def test_e14_prospective_repair_preserves_numeric_thresholds() -> None:
    config = load_config("configs/e14_plan_continuation.yaml")

    assert config["claim_gate"]["primary_estimand"] == (
        "affected_plan_correction_gain"
    )
    assert (
        config["claim_gate"]["minimum_affected_plan_correction_gain"]
        == 0.001
    )
    assert config["claim_gate"]["maximum_retention_mse"] == 0.0005
    assert config["dependency"]["required_seeds"] == list(FIXED_SEEDS)
    assert STATIC_EVIDENCE_BOUNDARY == {
        "tier": "CONTROLLED_REFERENCE",
        "proxy_type": "STRUCTURED_SYNTHETIC_ENTITY_VALUE",
        "oracle_entity_address": True,
        "oracle_old_new_candidates": True,
        "independent_plan_semantics_tested": False,
        "semantic_demand_inference_tested": False,
        "learned_addressing_tested": False,
        "language_model_transfer_claim_eligible": False,
        "general_agent_planning_claim_eligible": False,
        "official_backend_claim_eligible": False,
        "production_break_even_claim_eligible": False,
        "long_gap_persistence_claim_eligible": False,
    }
