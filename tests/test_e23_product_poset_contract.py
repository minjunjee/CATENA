from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from catena.data.controller_poset import (
    CANONICAL_CONTROLLERS,
    CONTROLLER_BY_ID,
    DEMAND_FAMILIES,
    missing_required_axes,
)
from catena.data.product_poset_sequence import (
    generate_product_poset_sequence_batch,
)
from catena.post_e21.product_poset_eval import (
    E22B_PROTOCOL_LOCK_SHA256,
    resolve_e18b_freeze,
    resolve_e22b_dependency,
    resolve_e23a_screen_dependency,
    summarize_seed_predictions,
)
from catena.post_e21.product_poset_model import (
    MatchedProductPosetSequenceController,
    product_poset_parameter_count,
)

ROOT = Path(__file__).resolve().parents[1]


def _write_e22_report(
    directory: Path,
    *,
    status: str,
    safe: bool,
    execution_status: str = "PASS",
) -> Path:
    directory.mkdir()
    payload = {
        "execution_status": execution_status,
        "evidence_tier": "CONTROLLED_REFERENCE",
        "claim_eligible": safe,
        "protocol_lock": {"sha256": E22B_PROTOCOL_LOCK_SHA256},
        "parent_e21": {"inherited_thresholds": {"maximum_nontarget_degradation": 0.0005}},
        "phase_dependency": {
            "selected_method": {
                "method_id": "cvar_005",
                "objective": "cvar",
                "selection_eligible": True,
                "baseline": False,
                "tail_fraction": 0.05,
                "normalized_temperature": None,
                "active_fraction": None,
            }
        },
        "claim_gate": {
            "status": status,
            "safe_locality_supported": safe,
        },
    }
    path = directory / "report.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_e22_dependency_requires_explicit_completed_report() -> None:
    decision = resolve_e22b_dependency(e22b_run=None, dry_run=False)
    assert decision.execution_status == "BLOCKED_DEPENDENCY"
    assert decision.boundary_mode is None


def test_e18_dependency_requires_explicit_supported_freeze() -> None:
    decision = resolve_e18b_freeze(freeze_path=None, dry_run=False)
    assert decision.execution_status == "BLOCKED_DEPENDENCY"
    assert decision.freeze_sha256 is None


def test_e18_dry_fixture_never_reads_canonical_path() -> None:
    decision = resolve_e18b_freeze(
        freeze_path="/definitely/not/read/in/dry-run.json",
        dry_run=True,
    )
    assert decision.execution_status == "PASS"
    assert decision.synthetic is True
    assert decision.freeze_path is None


def test_checked_in_e18_freeze_validates_supported_report() -> None:
    freeze = Path("/data/minjun_dev/CATENA/artifacts/E18_SEQUENCE_CONTROL_LATTICE_FREEZE_V1.json")
    if not freeze.is_file():
        pytest.skip("local immutable E18 freeze is unavailable")
    decision = resolve_e18b_freeze(freeze_path=freeze, dry_run=False)
    assert decision.execution_status == "PASS"
    assert decision.claim_status == "SUPPORTED"


def test_e23a_screen_dependency_preserves_e18_provenance(
    tmp_path: Path,
) -> None:
    report = {
        "experiment_id": "e23a_product_poset_screen",
        "execution_status": "PASS",
        "run_mode": "MAIN",
        "phase": "SCREEN",
        "e18_dependency": {
            "execution_status": "PASS",
            "freeze_sha256": "a" * 64,
        },
        "claim_gate": {
            "status": "SCREEN_ONLY_NO_CONFIRMATORY_CLAIM",
            "supported": False,
        },
        # Deliberately arbitrary screen outcome: dependency validation must not
        # use it to choose the confirmatory boundary.
        "summary": {"capacity_supported": False},
    }
    path = tmp_path / "e23a_report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    decision = resolve_e23a_screen_dependency(
        screen_run=path,
        dry_run=False,
        expected_e18_freeze_sha256="a" * 64,
    )
    assert decision.execution_status == "PASS"
    assert decision.outcomes_used_for_boundary is False
    mismatch = resolve_e23a_screen_dependency(
        screen_run=path,
        dry_run=False,
        expected_e18_freeze_sha256="b" * 64,
    )
    assert mismatch.execution_status == "BLOCKED_DEPENDENCY"


def test_dry_run_dependency_is_explicitly_synthetic_non_evidence() -> None:
    decision = resolve_e22b_dependency(e22b_run=None, dry_run=True)
    assert decision.execution_status == "PASS"
    assert decision.boundary_mode == "capacity_only"
    assert decision.synthetic is True
    assert decision.dependency_claim_eligible is False


def test_safe_e22_dependency_locks_safe_boundary(tmp_path: Path) -> None:
    report = _write_e22_report(
        tmp_path / "safe",
        status="SUPPORTED_SAFE_LOCALIZED_ASSIMILATION_CONTROLLED",
        safe=True,
    )
    decision = resolve_e22b_dependency(e22b_run=report, dry_run=False)
    assert decision.execution_status == "PASS"
    assert decision.boundary_mode == "safe_minimality"
    assert decision.protocol_lock_sha256 == E22B_PROTOCOL_LOCK_SHA256
    assert decision.locality_method["method_id"] == "cvar_005"


def test_e22_dependency_rejects_stale_protocol_lock(tmp_path: Path) -> None:
    report = _write_e22_report(
        tmp_path / "stale",
        status="SUPPORTED_SAFE_LOCALIZED_ASSIMILATION_CONTROLLED",
        safe=True,
    )
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["protocol_lock"]["sha256"] = "0" * 64
    report.write_text(json.dumps(payload), encoding="utf-8")
    decision = resolve_e22b_dependency(e22b_run=report, dry_run=False)
    assert decision.execution_status == "BLOCKED_DEPENDENCY"
    assert decision.protocol_lock_sha256 == "0" * 64


def test_safe_sparse_e22_objective_blocks_without_hard_route(
    tmp_path: Path,
) -> None:
    report = _write_e22_report(
        tmp_path / "sparse",
        status="SUPPORTED_SAFE_LOCALIZED_ASSIMILATION_CONTROLLED",
        safe=True,
    )
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["phase_dependency"]["selected_method"] = {
        "method_id": "sparse_0125",
        "objective": "sparse",
        "selection_eligible": True,
        "baseline": False,
        "tail_fraction": 0.10,
        "normalized_temperature": None,
        "active_fraction": 0.125,
    }
    report.write_text(json.dumps(payload), encoding="utf-8")
    decision = resolve_e22b_dependency(e22b_run=report, dry_run=False)
    assert decision.execution_status == "BLOCKED_DEPENDENCY"
    assert decision.safe_objective_implemented is False
    assert decision.locality_method["method_id"] == "sparse_0125"


@pytest.mark.parametrize(
    "status",
    (
        "CAPACITY_SUPPORTED_LOCALITY_NOT_SUPPORTED",
        "OVERREGULARIZED_LOCALITY_TRADEOFF",
        "NOT_SUPPORTED",
    ),
)
def test_completed_non_safe_e22_dependency_locks_capacity_only(
    tmp_path: Path,
    status: str,
) -> None:
    report = _write_e22_report(
        tmp_path / status,
        status=status,
        safe=False,
    )
    decision = resolve_e22b_dependency(e22b_run=report, dry_run=False)
    assert decision.execution_status == "PASS"
    assert decision.boundary_mode == "capacity_only"
    assert decision.locality_method["method_id"] == "mean_retention"


def test_inconsistent_e22_status_is_blocked(tmp_path: Path) -> None:
    report = _write_e22_report(
        tmp_path / "inconsistent",
        status="SUPPORTED_SAFE_LOCALIZED_ASSIMILATION_CONTROLLED",
        safe=False,
    )
    decision = resolve_e22b_dependency(e22b_run=report, dry_run=False)
    assert decision.execution_status == "BLOCKED_DEPENDENCY"
    assert decision.boundary_mode is None


def test_product_poset_uses_learned_e18_compatible_tensors() -> None:
    batch = generate_product_poset_sequence_batch(
        demand_family="magnitude_value",
        intensity=0.5,
        batch_size=3,
        num_entities=32,
        value_dim=32,
        updates=2,
        gap_events=4,
        seed=17,
        device=torch.device("cpu"),
    )
    assert batch.inputs.initial_state.shape == (3, 32, 32)
    assert batch.inputs.demand_features.shape[1] == 6
    assert batch.update_mask.sum(dim=1).tolist() == [2, 2, 2]
    models = [
        MatchedProductPosetSequenceController(
            controller=CONTROLLER_BY_ID[controller_id],
            num_entities=32,
            value_dim=32,
            embedding_dim=8,
            hidden_dim=16,
        )
        for controller_id in ("c0000", "c1100", "c1111")
    ]
    assert len({product_poset_parameter_count(model) for model in models}) == 1
    output = models[1](batch.inputs)
    assert output.state.shape == batch.target_state.shape


def _absolute_adequacy_rows(*, sufficient_error: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for demand in DEMAND_FAMILIES:
        for controller in CANONICAL_CONTROLLERS:
            sufficient = not missing_required_axes(controller, demand)
            rows.append(
                {
                    "seed": 7,
                    "controller_id": controller.controller_id,
                    "demand_family": demand,
                    "intensity": 0.5,
                    "updates": 1,
                    "gap_events": 0,
                    "affected_mse": (sufficient_error if sufficient else 0.002),
                    "retention_mse": 0.0,
                    "active_nontarget_degradation": None,
                }
            )
    return rows


def test_absolute_adequacy_then_poset_minimal_recovers_theory() -> None:
    seed_rows, detail = summarize_seed_predictions(
        _absolute_adequacy_rows(sufficient_error=1.0e-6),
        seeds=[7],
        intensities=[0.5],
        updates=[1],
        gap_events=[0],
        affected_mse_tolerance=1.0e-4,
        target_margin=1.0e-4,
        retention_margin=5.0e-4,
        locality_margin=5.0e-4,
        minimum_single_axis_exact_matches=4,
        minimum_pairwise_exact_matches=5,
        incomparable_direction_margin=0.0,
        maximal_simpler_degradation_margin=5.0e-4,
        boundary_mode="capacity_only",
    )
    assert seed_rows[0]["capacity_gate_passed"] is True
    assert detail["assessment"]["minimum_single_axis_exact_match_count"] == 4
    assert detail["assessment"]["minimum_pairwise_exact_match_count"] == 6
    assert detail["assessment"]["total_false_adequate_count"] == 0
    assert detail["assessment"]["total_false_inadequate_count"] == 0


def test_absolute_adequacy_does_not_use_best_plus_epsilon() -> None:
    seed_rows, detail = summarize_seed_predictions(
        _absolute_adequacy_rows(sufficient_error=0.001),
        seeds=[7],
        intensities=[0.5],
        updates=[1],
        gap_events=[0],
        affected_mse_tolerance=1.0e-4,
        target_margin=1.0e-4,
        retention_margin=5.0e-4,
        locality_margin=5.0e-4,
        minimum_single_axis_exact_matches=4,
        minimum_pairwise_exact_matches=5,
        incomparable_direction_margin=0.0,
        maximal_simpler_degradation_margin=5.0e-4,
        boundary_mode="capacity_only",
    )
    assert seed_rows[0]["capacity_gate_passed"] is False
    assert detail["assessment"]["total_false_inadequate_count"] > 0


@pytest.mark.parametrize(
    "entrypoint,config,experiment_id,expected_raw,expected_seed",
    (
        (
            "experiments/e23a_product_poset_screen.py",
            "configs/e23a_product_poset_screen.yaml",
            "e23a_product_poset_screen",
            176,
            1,
        ),
        (
            "experiments/e23b_product_poset_confirmatory.py",
            "configs/e23b_product_poset_confirmatory.yaml",
            "e23b_product_poset_confirmatory",
            176,
            1,
        ),
    ),
)
def test_e23_cpu_dry_run_artifact_contract(
    tmp_path: Path,
    entrypoint: str,
    config: str,
    experiment_id: str,
    expected_raw: int,
    expected_seed: int,
) -> None:
    artifact_root = tmp_path / "artifacts"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / entrypoint),
            "--config",
            str(ROOT / config),
            "--device",
            "cpu",
            "--artifact-root",
            str(artifact_root),
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
    )
    latest = json.loads((artifact_root / experiment_id / "latest.json").read_text(encoding="utf-8"))
    run_dir = Path(latest["run_dir"])
    required = {
        "config.resolved.yaml",
        "environment.json",
        "run_manifest.json",
        "protocol_lock.json",
        "data_manifest.json",
        "theory_predictions.json",
        "product_poset_raw_metrics.jsonl",
        "product_poset_seed_metrics.jsonl",
        "poset_minimal_demands.jsonl",
        "product_poset_training_runs.jsonl",
        "RESULTS_SUMMARY_KO.md",
        "report.json",
    }
    assert required <= {path.name for path in run_dir.iterdir()}
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    assert report["execution_status"] == "PASS"
    assert report["run_mode"] == "DRY_RUN"
    assert report["claim_eligible"] is False
    assert report["scientific_evidence"] is False
    assert report["claim_gate"]["status"] == "DRY_RUN_ONLY"
    assert report["artifacts"]["rows"]["raw"]["rows"] == expected_raw
    assert report["artifacts"]["rows"]["seed"]["rows"] == expected_seed
    assert report["checkpoint_sha256"] is not None
    assert len(report["checkpoint_hashes"]) == 16
    first_raw = json.loads(
        (run_dir / "product_poset_raw_metrics.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert "e18_freeze_sha256" in first_raw
    if experiment_id == "e23b_product_poset_confirmatory":
        assert "e23a_screen_report_sha256" in first_raw
        assert "e22_report_sha256" in first_raw
        assert "e22_protocol_lock_sha256" in first_raw
