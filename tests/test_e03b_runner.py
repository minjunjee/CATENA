from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from catena.core.config import load_config
from catena.core.provenance_v61 import (
    ProvenanceValidationError,
    read_json_object_strict,
    sha256_file,
    write_json_strict,
)
from catena.data.graded_operator_families import (
    generate_graded_operator_family,
    tensor_sha256,
)
from catena.eval.jd_calibration import (
    AnalyticCandidate,
    validate_selected_design,
)
from experiments import e03b_graded_jd_calibration as runner
from experiments.e03b_graded_jd_calibration import (
    _AnalyticRuntime,
    _execution_contract,
    _regret_bins,
    _required_bin_boundary_clearance,
    _run_empirical_phase,
    _selection_lock_rows,
    _train_identifiability,
    _training_weights,
    _validate_config,
    _validate_initialized_config,
    _validate_registered_config_file,
    _validate_reloaded_config,
    _write_preprobe_lock,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/e03b_graded_jd_calibration.yaml"


def _registered_config() -> dict[str, Any]:
    return cast(dict[str, Any], load_config(CONFIG_PATH))


def test_registered_config_pins_exact_bins_streams_and_source() -> None:
    config = _validate_registered_config_file(CONFIG_PATH)
    _validate_config(config)
    bins = _regret_bins(config)
    assert [
        (item.lower, item.upper, item.include_upper) for item in bins
    ] == [
        (1.0e-5, 2.5e-4, False),
        (2.5e-4, 7.5e-4, False),
        (7.5e-4, 1.5e-3, False),
        (1.5e-3, 3.0e-3, False),
        (3.0e-3, 4.5e-3, False),
        (4.5e-3, 6.5e-3, True),
    ]
    registry = config["candidate_registry"]
    assert registry["alpha_schedule"][:6] == [
        0.06,
        0.12,
        0.18,
        0.26,
        0.40,
        0.55,
    ]
    alpha_count = len(registry["alpha_schedule"])
    main = registry["main"]
    dry = registry["dry_run"]
    main_seeds = set(
        range(
            main["generation_seed_start"],
            main["generation_seed_start"] + main["hard_max_candidates"],
        )
    )
    dry_seeds = set(
        range(
            dry["generation_seed_start"],
            dry["generation_seed_start"] + dry["hard_max_candidates"],
        )
    )
    pilot_seeds = set(registry["development_analytic_pilot"]["generation_seeds"])
    assert dry["generation_seed_start"] == 950001
    assert registry["development_analytic_pilot"]["empirical_probe_calls"] == 0
    assert (
        registry["development_analytic_pilot"]["artifact_sha256"]
        == "21d60d79738e5bd05034312ec630a694d6485f00be1bc824344088eebc9fe94c"
    )
    assert main["hard_max_candidates"] == alpha_count * main["replicates_per_alpha"]
    assert dry["hard_max_candidates"] == alpha_count * dry["replicates_per_alpha"]
    assert main_seeds.isdisjoint(dry_seeds)
    assert main_seeds.isdisjoint(pilot_seeds)
    assert dry_seeds.isdisjoint(pilot_seeds)
    assert config["source_e03"]["claim_status_registry"]["quantitative_status"] == "FAILED"
    assert config["source_e03"]["claim_status_registry"]["full_claim_open"] is False


def test_config_rejects_payload_and_comment_byte_mutation(tmp_path: Path) -> None:
    changed_payload = deepcopy(_registered_config())
    changed_payload["unchecked_extra_field"] = "must still fail canonical hash"
    with pytest.raises(ValueError, match="canonical protocol hash"):
        _validate_config(changed_payload)

    changed_file = tmp_path / CONFIG_PATH.name
    changed_file.write_bytes(CONFIG_PATH.read_bytes() + b"\n# byte-only mutation\n")
    with pytest.raises(ValueError, match="file bytes"):
        _validate_registered_config_file(changed_file)


def test_reloaded_config_must_match_the_validated_preview() -> None:
    preview = _registered_config()
    _validate_reloaded_config(preview, deepcopy(preview))
    changed = deepcopy(preview)
    changed["runtime"]["cpu_threads"] = 2
    with pytest.raises(ProvenanceValidationError, match="changed during initialization"):
        _validate_reloaded_config(preview, changed)


def test_initialized_config_rechecks_exact_file_bytes(tmp_path: Path) -> None:
    copied_config = tmp_path / CONFIG_PATH.name
    copied_config.write_bytes(CONFIG_PATH.read_bytes())
    preview = _validate_registered_config_file(copied_config)
    copied_config.write_bytes(copied_config.read_bytes() + b"\n# TOCTOU mutation\n")
    with pytest.raises(ValueError, match="file bytes"):
        _validate_initialized_config(
            copied_config,
            preview,
            deepcopy(preview),
            copied_config,
        )


def test_spectral_weights_are_local_and_identifiability_is_analytic_only() -> None:
    config = _registered_config()
    estimator = config["basis_estimator"]
    torch.manual_seed(8421)
    state = torch.random.get_rng_state().clone()
    weights = _training_weights(estimator, 24)
    assert torch.equal(torch.random.get_rng_state(), state)
    assert tensor_sha256(weights) == estimator["weight_vector_sha256"]

    family = generate_graded_operator_family(
        dim=32,
        rank=8,
        train_count=24,
        heldout_count=8,
        seed=300001,
        alpha=0.1,
        max_rotation_radians=float(config["data"]["max_rotation_radians"]),
    )
    diagnostic = _train_identifiability(
        family,
        weights,
        minimum_weighted_eigenvalue_gap=float(
            estimator["minimum_weighted_eigenvalue_gap"]
        ),
        maximum_zero_alpha_regret=float(
            estimator["maximum_zero_alpha_heldout_regret"]
        ),
    )
    assert diagnostic["passed"] is True
    assert diagnostic["unique_train_inclusion_signatures"] == 32
    assert diagnostic["zero_alpha_heldout_regret"] <= 1.0e-10


def test_bin_clearance_includes_pilot_saturation_uncertainty_floor() -> None:
    estimator = _registered_config()["basis_estimator"]
    required, effective = _required_bin_boundary_clearance(
        estimator,
        {"optimizer_uncertainty": 0.0},
    )
    assert effective == pytest.approx(4.61e-6)
    assert required == pytest.approx(4.61e-5)


def _synthetic_selected() -> tuple[
    dict[str, list[AnalyticCandidate]],
    dict[str, _AnalyticRuntime],
]:
    bins = _regret_bins(_registered_config())
    regrets = (1.0e-4, 5.0e-4, 1.0e-3, 2.0e-3, 3.5e-3, 5.0e-3)
    selected: dict[str, list[AnalyticCandidate]] = {
        item.label: [] for item in bins
    }
    runtimes: dict[str, _AnalyticRuntime] = {}
    for index, (regret_bin, regret) in enumerate(zip(bins, regrets, strict=True)):
        family = generate_graded_operator_family(
            dim=4,
            rank=1,
            train_count=3,
            heldout_count=1,
            seed=71000 + index,
            alpha=0.1,
        )
        candidate = AnalyticCandidate(
            candidate_id=family.realization_id,
            construction_sha256=family.realization_sha256,
            analytic_regret=regret,
            alpha=family.alpha,
            generation_seed=family.spec.seed,
        )
        basis = torch.eye(4, dtype=torch.float64)
        selected[regret_bin.label].append(candidate)
        runtimes[candidate.candidate_id] = _AnalyticRuntime(
            stream_index=index,
            alpha_schedule_index=0,
            replicate_index=index,
            family=family,
            candidate=candidate,
            basis=basis,
            basis_sha256=tensor_sha256(basis),
            train_analytic_regret=regret,
            heldout_operator_analytic_regrets=[regret],
            basis_diagnostics={
                "method": "test",
                "basis_sha256": tensor_sha256(basis),
            },
            identifiability={"passed": True},
            probe_seeds=[80000 + index],
        )
    return selected, runtimes


def test_selection_and_basis_lock_exist_before_first_empirical_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _registered_config()
    bins = _regret_bins(config)
    selected, runtimes = _synthetic_selected()
    design = validate_selected_design(
        selected,
        bins,
        families_per_bin=1,
        minimum_nonzero_range=0.004,
    )
    assert design["passed"] is True
    registry_path = tmp_path / "candidate_split_registry.json"
    write_json_strict(
        registry_path,
        {
            "test_registry": True,
            "candidate_registry_sha256": "a" * 64,
        },
    )
    selection_rows = _selection_lock_rows(
        selected,
        runtimes,
        bins,
        registry_sha256="a" * 64,
        probe_count=17,
    )
    lock = _write_preprobe_lock(
        tmp_path,
        registry_path=registry_path,
        registry_file_sha256=sha256_file(registry_path),
        registry_sha256="a" * 64,
        audit_rows=[{"candidate": "analytic-only"}],
        selection_rows=selection_rows,
        design_gate=design,
    )

    calls: list[int] = []

    def fake_empirical(
        targets: list[torch.Tensor],
        approximations: list[torch.Tensor],
        *,
        probe_count: int,
        probe_seed: int,
    ) -> float:
        assert targets and approximations
        assert lock.selection_path.is_file()
        assert sha256_file(lock.selection_path) == lock.selection_sha256
        metadata = read_json_object_strict(lock.metadata_path)
        assert metadata["status"] == "LOCKED_BEFORE_EMPIRICAL_PROBES"
        assert metadata["empirical_probe_phase_authorized"] is True
        calls.append(probe_seed)
        return float(probe_count) * 1.0e-8

    monkeypatch.setattr(runner, "_empirical_application_mse", fake_empirical)
    family_rows, operator_rows = _run_empirical_phase(
        selected,
        runtimes,
        bins,
        preprobe_lock=lock,
        probe_count=17,
    )
    assert len(calls) == 6
    assert len(family_rows) == 6
    assert len(operator_rows) == 6
    assert all("heldout_operator_sha256" in row for row in operator_rows)
    assert all("jd_basis_sha256" in row for row in operator_rows)


def test_runtime_basis_mutation_after_lock_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bins = _regret_bins(_registered_config())
    selected, runtimes = _synthetic_selected()
    design = validate_selected_design(
        selected,
        bins,
        families_per_bin=1,
        minimum_nonzero_range=0.004,
    )
    registry_path = tmp_path / "candidate_split_registry.json"
    write_json_strict(
        registry_path,
        {
            "test_registry": True,
            "candidate_registry_sha256": "c" * 64,
        },
    )
    lock = _write_preprobe_lock(
        tmp_path,
        registry_path=registry_path,
        registry_file_sha256=sha256_file(registry_path),
        registry_sha256="c" * 64,
        audit_rows=[{"candidate": "analytic-only"}],
        selection_rows=_selection_lock_rows(
            selected,
            runtimes,
            bins,
            registry_sha256="c" * 64,
            probe_count=17,
        ),
        design_gate=design,
    )
    first_runtime = next(iter(runtimes.values()))
    first_runtime.basis[0, 0] += 0.25
    called = False

    def forbidden_probe(*args: object, **kwargs: object) -> float:
        nonlocal called
        called = True
        return 0.0

    monkeypatch.setattr(runner, "_empirical_application_mse", forbidden_probe)
    with pytest.raises(ProvenanceValidationError, match="Runtime JD basis changed"):
        _run_empirical_phase(
            selected,
            runtimes,
            bins,
            preprobe_lock=lock,
            probe_count=17,
        )
    assert called is False


def test_design_failure_locks_no_probe_result_without_crashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _registered_config()
    bins = _regret_bins(config)
    registry_path = tmp_path / "candidate_split_registry.json"
    write_json_strict(registry_path, {"test_registry": True})
    lock = _write_preprobe_lock(
        tmp_path,
        registry_path=registry_path,
        registry_file_sha256=sha256_file(registry_path),
        registry_sha256="b" * 64,
        audit_rows=[],
        selection_rows=[],
        design_gate={"passed": False},
    )
    called = False

    def forbidden_probe(*args: object, **kwargs: object) -> float:
        nonlocal called
        called = True
        return 0.0

    monkeypatch.setattr(runner, "_empirical_application_mse", forbidden_probe)
    with pytest.raises(ProvenanceValidationError, match="did not authorize"):
        _run_empirical_phase(
            {item.label: [] for item in bins},
            {},
            bins,
            preprobe_lock=lock,
            probe_count=3,
        )
    assert called is False


def test_main_execution_and_claim_evaluation_are_separate() -> None:
    failed_design = _execution_contract(
        dry_run=False,
        design_passed=False,
        registry_exhausted=True,
        selected_count=0,
        selection_row_count=0,
        family_row_count=0,
        operator_row_count=0,
        expected_families=48,
        expected_operator_rows=384,
        metrics_finite=True,
    )
    assert failed_design["main_execution_complete"] is True
    assert failed_design["claim_evaluated"] is False

    complete_empirical = _execution_contract(
        dry_run=False,
        design_passed=True,
        registry_exhausted=False,
        selected_count=48,
        selection_row_count=48,
        family_row_count=48,
        operator_row_count=384,
        expected_families=48,
        expected_operator_rows=384,
        metrics_finite=True,
    )
    assert complete_empirical["main_execution_complete"] is True
    assert complete_empirical["claim_evaluated"] is True
