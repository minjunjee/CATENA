from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import freeze_e21_structured_sequence_transfer as freeze


def test_write_exclusive_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "freeze.json"
    payload = {"schema_version": 1}
    freeze._write_exclusive(output, payload)
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    with pytest.raises(FileExistsError):
        freeze._write_exclusive(output, payload)


def _self_consistent_payload(tmp_path: Path, *, supported: bool) -> dict:
    source_runs = [
        {
            "seed": seed,
            "run_dir": str(
                (
                    tmp_path / freeze.SOURCE_EXPERIMENT_ID / f"20260728T0{index}0000.000000Z"
                ).resolve()
            ),
        }
        for index, seed in enumerate((113, 223, 331, 449, 557), start=1)
    ]
    status = "SUPPORTED" if supported else "NOT_SUPPORTED"
    return {
        "schema_version": 1,
        "frozen_at_utc": "2026-07-28T08:00:00Z",
        "experiment_family": "E21",
        "source_experiment_id": freeze.SOURCE_EXPERIMENT_ID,
        "original_aggregate_experiment_id": freeze.ORIGINAL_AGGREGATE_ID,
        "repair_aggregate_experiment_id": freeze.R1_AGGREGATE_ID,
        "source_runs": source_runs,
        "original_aggregate": {
            "run_dir": str(
                (tmp_path / freeze.ORIGINAL_AGGREGATE_ID / "20260728T080001.000000Z").resolve()
            ),
            "frozen_disposition": freeze.ORIGINAL_DISPOSITION,
            "claim_eligible": False,
        },
        "repair_aggregate": {
            "run_dir": str(
                (tmp_path / freeze.R1_AGGREGATE_ID / "20260728T080002.000000Z").resolve()
            ),
            "frozen_disposition": status,
            "claim_eligible": supported,
        },
        "claim_status": status,
        "evidence_tier": "CONTROLLED_REFERENCE",
        "scientific_evidence": False,
        "locks_and_configs": {},
        "claim_boundary": freeze._claim_boundary(supported),
        "immutable": True,
    }


def test_validate_existing_rebuilds_exact_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = _self_consistent_payload(tmp_path, supported=True)
    monkeypatch.setattr(
        freeze,
        "build_freeze_payload",
        lambda **_: payload,
    )
    freeze.validate_freeze(
        payload,
        repo_root=tmp_path,
        artifact_root=tmp_path,
    )


def test_original_aggregate_can_never_be_claim_eligible(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = _self_consistent_payload(tmp_path, supported=True)
    payload["original_aggregate"]["claim_eligible"] = True
    monkeypatch.setattr(
        freeze,
        "build_freeze_payload",
        lambda **_: payload,
    )
    with pytest.raises(RuntimeError, match="disposition boundary"):
        freeze.validate_freeze(
            payload,
            repo_root=tmp_path,
            artifact_root=tmp_path,
        )


def test_original_disposition_is_fixed_even_if_report_gate_was_positive() -> None:
    report = {
        "claim_gate": {"supported": True, "status": "SUPPORTED"},
    }
    assert report["claim_gate"]["supported"] is True
    assert freeze.ORIGINAL_DISPOSITION == "INCONCLUSIVE_GATE_IMPLEMENTATION"
    boundary = freeze._claim_boundary(True)
    assert boundary["original_e21b_claim_eligible"] is False
    assert boundary["e21b_r1_controlled_structured_sequence_claim_eligible"] is True


def test_validate_rejects_noncontrolled_or_external_claim_boundary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = _self_consistent_payload(tmp_path, supported=True)
    payload["scientific_evidence"] = True
    payload["claim_boundary"]["official_backend_claim_eligible"] = True
    monkeypatch.setattr(
        freeze,
        "build_freeze_payload",
        lambda **_: payload,
    )
    with pytest.raises(RuntimeError, match="scientific_evidence"):
        freeze.validate_freeze(
            payload,
            repo_root=tmp_path,
            artifact_root=tmp_path,
        )


def test_validate_detects_nonreproducible_hash_or_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = _self_consistent_payload(tmp_path, supported=False)
    rebuilt = _self_consistent_payload(tmp_path, supported=False)
    rebuilt["locks_and_configs"] = {"repair_protocol_lock": {"sha256": "a" * 64}}
    monkeypatch.setattr(
        freeze,
        "build_freeze_payload",
        lambda **_: rebuilt,
    )
    with pytest.raises(RuntimeError, match="does not reproduce"):
        freeze.validate_freeze(
            payload,
            repo_root=tmp_path,
            artifact_root=tmp_path,
        )


def _source_descriptors(tmp_path: Path) -> list[dict]:
    result = []
    for seed in (113, 223, 331, 449, 557):
        run_dir = (
            tmp_path / freeze.SOURCE_EXPERIMENT_ID / f"20260728T{seed:06d}.000000Z"
        ).resolve()
        result.append(
            {
                "seed": seed,
                "run_dir": str(run_dir),
                "hashes": {
                    "report.json": f"{seed:064x}",
                    "structured_sequence_transfer_metrics.jsonl": (f"{seed + 1:064x}"),
                    "RESULTS_SUMMARY_KO.md": f"{seed + 2:064x}",
                },
                "checkpoint_hashes": {
                    variant: f"{seed + index + 3:064x}"
                    for index, variant in enumerate(
                        ("base", "separate_address", "state_aware", "full")
                    )
                },
            }
        )
    return result


def _make_aggregate_fixture(
    tmp_path: Path,
    *,
    experiment_id: str,
    supported: bool,
) -> tuple[Path, list[dict], list[dict], dict[str, str], dict[str, str]]:
    run_dir = (tmp_path / experiment_id / "20260728T081500.000000Z").resolve()
    run_dir.mkdir(parents=True)
    sources = _source_descriptors(tmp_path)
    source_rows = [{"row": index} for index in range(3840)]
    metrics_name = "structured_sequence_paired_metrics.jsonl"
    contrasts_name = (
        "structured_sequence_seed_contrasts.jsonl"
        if experiment_id == freeze.ORIGINAL_AGGREGATE_ID
        else "structured_sequence_seed_contrasts_r1.jsonl"
    )
    provenance = [
        {
            "seed": source["seed"],
            "run_dir": source["run_dir"],
            "report_sha256": source["hashes"]["report.json"],
            "metrics_sha256": source["hashes"]["structured_sequence_transfer_metrics.jsonl"],
            "results_summary_sha256": source["hashes"]["RESULTS_SUMMARY_KO.md"],
            "checkpoint_hashes": source["checkpoint_hashes"],
        }
        for source in sources
    ]
    (run_dir / metrics_name).write_text(
        "".join(json.dumps(row) + "\n" for row in source_rows),
        encoding="utf-8",
    )
    (run_dir / contrasts_name).write_text(
        "".join(json.dumps({"seed": seed}) + "\n" for seed in (113, 223, 331, 449, 557)),
        encoding="utf-8",
    )
    (run_dir / "source_run_provenance.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in provenance),
        encoding="utf-8",
    )
    (run_dir / "RESULTS_SUMMARY_KO.md").write_text(
        "# fixture\n",
        encoding="utf-8",
    )
    source_lock = {"sha256": "a" * 64, "config_sha256": "b" * 64}
    repair_lock = {"sha256": "c" * 64, "config_sha256": "d" * 64}
    source_contract = {
        "source_runs": provenance,
    }
    summary: dict = {}
    if experiment_id == freeze.ORIGINAL_AGGREGATE_ID:
        source_contract.update(
            {
                "protocol_lock_sha256": source_lock["sha256"],
                "source_config_sha256": source_lock["config_sha256"],
            }
        )
    else:
        source_contract.update(
            {
                "source_protocol_lock_sha256": source_lock["sha256"],
                "repair_protocol_lock_sha256": repair_lock["sha256"],
                "repair_config_sha256": repair_lock["config_sha256"],
            }
        )
        summary = {
            "supported": supported,
            "repair": {"original_e21b_disposition": freeze.ORIGINAL_DISPOSITION},
        }
    artifacts = {
        "paired_metrics_sha256": freeze._sha256(run_dir / metrics_name),
        "seed_contrasts_sha256": freeze._sha256(run_dir / contrasts_name),
        "source_provenance_sha256": freeze._sha256(run_dir / "source_run_provenance.jsonl"),
        "results_summary_ko": {
            "sha256": freeze._sha256(run_dir / "RESULTS_SUMMARY_KO.md"),
            "line_count": 1,
        },
    }
    status = "SUPPORTED" if supported else "NOT_SUPPORTED"
    report = {
        "status": "PASS",
        "run_mode": "MAIN",
        "evidence_tier": "CONTROLLED_REFERENCE",
        "scientific_evidence": False,
        "source_contract": source_contract,
        "summary": summary,
        "artifacts": artifacts,
        "claim_gate": {"supported": supported, "status": status},
    }
    if experiment_id == freeze.R1_AGGREGATE_ID:
        report["original_e21b_disposition"] = freeze.ORIGINAL_DISPOSITION
    (run_dir / "report.json").write_text(
        json.dumps(report),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 2,
        "experiment_id": experiment_id,
        "run_id": run_dir.name,
        "run_mode": "MAIN",
        "report_sha256": freeze._sha256(run_dir / "report.json"),
        "source_fingerprint": {"files": 1, "sha256": "e" * 64},
        "source_fingerprint_phase": "RUN_START",
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return run_dir, sources, source_rows, source_lock, repair_lock


def test_original_aggregate_descriptor_forces_inconclusive_disposition(
    tmp_path: Path,
) -> None:
    run_dir, sources, source_rows, source_lock, _repair_lock = _make_aggregate_fixture(
        tmp_path,
        experiment_id=freeze.ORIGINAL_AGGREGATE_ID,
        supported=True,
    )
    descriptor = freeze._aggregate_descriptor(
        run_dir,
        experiment_id=freeze.ORIGINAL_AGGREGATE_ID,
        filenames=freeze.ORIGINAL_AGGREGATE_FILES,
        metrics_filename="structured_sequence_paired_metrics.jsonl",
        contrasts_filename="structured_sequence_seed_contrasts.jsonl",
        sources=sources,
        source_rows=source_rows,
        source_lock=source_lock,
        repair_lock=None,
    )
    assert descriptor["observed_report_claim_status"] == "SUPPORTED"
    assert descriptor["frozen_disposition"] == freeze.ORIGINAL_DISPOSITION
    assert descriptor["claim_eligible"] is False


def test_r1_descriptor_is_only_valid_claim_gate_and_detects_metric_tamper(
    tmp_path: Path,
) -> None:
    run_dir, sources, source_rows, source_lock, repair_lock = _make_aggregate_fixture(
        tmp_path,
        experiment_id=freeze.R1_AGGREGATE_ID,
        supported=True,
    )
    kwargs = {
        "experiment_id": freeze.R1_AGGREGATE_ID,
        "filenames": freeze.R1_AGGREGATE_FILES,
        "metrics_filename": "structured_sequence_paired_metrics.jsonl",
        "contrasts_filename": "structured_sequence_seed_contrasts_r1.jsonl",
        "sources": sources,
        "source_rows": source_rows,
        "source_lock": source_lock,
        "repair_lock": repair_lock,
    }
    descriptor = freeze._aggregate_descriptor(run_dir, **kwargs)
    assert descriptor["frozen_disposition"] == "SUPPORTED"
    assert descriptor["claim_eligible"] is True

    with (run_dir / "structured_sequence_paired_metrics.jsonl").open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write('{"tampered":true}\n')
    with pytest.raises(RuntimeError, match="hash contract"):
        freeze._aggregate_descriptor(run_dir, **kwargs)


def test_aggregate_source_provenance_rejects_summary_hash_change(
    tmp_path: Path,
) -> None:
    sources = _source_descriptors(tmp_path)
    provenance = [
        {
            "seed": source["seed"],
            "run_dir": source["run_dir"],
            "report_sha256": source["hashes"]["report.json"],
            "metrics_sha256": source["hashes"]["structured_sequence_transfer_metrics.jsonl"],
            "results_summary_sha256": source["hashes"]["RESULTS_SUMMARY_KO.md"],
            "checkpoint_hashes": source["checkpoint_hashes"],
        }
        for source in sources
    ]
    report = {"source_contract": {"source_runs": provenance}}
    provenance[0]["results_summary_sha256"] = "f" * 64
    with pytest.raises(RuntimeError, match="provenance changed"):
        freeze._validate_aggregate_source_provenance(
            report,
            provenance,
            sources=sources,
        )
