from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any

import pytest
import torch

from catena.core.config import load_config
from catena.core.provenance_v61 import ProvenanceValidationError
from catena.core.schema import Operation
from catena.eval.statistics_v61 import Interval
from catena.models.matched_controllers import ScalarConstraint
from experiments import e02b_prospective_absolute_supersede as e02b

CONFIG_PATH = Path("configs/e02b_prospective_absolute_supersede.yaml").resolve()


def _config() -> dict[str, Any]:
    return load_config(CONFIG_PATH)


def test_e02b_protocol_config_and_historical_dependency_are_exactly_pinned() -> None:
    config = _config()
    e02b._validate_protocol_config(config, CONFIG_PATH)
    source = e02b.validate_pinned_e02_source("artifacts", config)

    assert source.run.run_id == "20260726T153504.455509Z"
    assert source.run.main_eligible is False
    assert source.run.full_eligible is False
    assert len(source.checkpoints) == 16
    assert set(source.inherited_tuning_values) == {11, 22, 33, 44, 55, 66, 77, 88}
    assert all(value > 0.0 for value in source.inherited_tuning_values.values())
    assert {
        checkpoint.path.name for checkpoint in source.checkpoints.values()
    } == {
        f"seed{seed}_{constraint}.pt"
        for seed in (11, 22, 33, 44, 55, 66, 77, 88)
        for constraint in ("tied", "dual")
    }
    assert e02b._PINNED_AMENDMENT_SHA256 == (
        "e42b71beb4512995e621000beb7522be93c939ce74b839c54f9aee8e63a2fd03"
    )
    assert e02b._PINNED_AMENDMENT_LOCK_SHA256 == (
        "14f8a9d546d3a84b0d47039a2c29d576b277bc65e305d23c455f68b2ed7d64e6"
    )


def test_e02b_rejects_any_source_dependency_substitution() -> None:
    config = _config()
    source = dict(e02b._require_mapping(config["source_e02"], "source_e02"))
    source["run_id"] = "20260726T000000.000000Z"
    config["source_e02"] = source

    with pytest.raises(ProvenanceValidationError, match="not exactly pinned"):
        e02b.validate_pinned_e02_source("artifacts", config)


def test_e02b_main_and_dry_run_seed_ranges_are_disjoint_from_all_reserved_ranges() -> None:
    config = _config()
    main_offset, main_contract = e02b._validate_seed_namespace(
        config,
        count_per_operation=512,
        dry_run=False,
    )
    dry_offset, dry_contract = e02b._validate_seed_namespace(
        config,
        count_per_operation=16,
        dry_run=True,
    )

    assert main_offset == 62_500
    assert main_contract["active_relative_range"] == [62_500, 64_547]
    assert dry_offset == 90_000
    assert dry_contract["active_relative_range"] == [90_000, 90_063]
    assert not e02b._ranges_overlap((62_500, 64_547), (75_000, 75_127))
    assert not e02b._ranges_overlap((90_000, 90_063), (62_500, 64_547))


def test_e02b_geometry_grid_is_16_unseen_balanced_cells() -> None:
    config = _config()
    cells = e02b._geometry_cells(config)
    main_cells, main_repeats = e02b._execution_geometry_design(
        config,
        dry_run=False,
    )
    dry_cells, dry_repeats = e02b._execution_geometry_design(
        config,
        dry_run=True,
    )

    assert len(cells) == 16
    assert main_cells == cells
    assert dry_cells == cells
    assert main_repeats == 32
    assert dry_repeats == 1
    assert {cell.norm_pair_bin for cell in cells} == {0, 1, 2, 3}
    assert {cell.angle_bin for cell in cells} == {0, 1, 2, 3}
    assert all((cell.old_scale, cell.new_scale) != (1.0, 1.0) for cell in cells)
    assert all(cell.angle_degrees != 90.0 for cell in cells)
    assert len(cells) * main_repeats == 512
    assert len(dry_cells) * dry_repeats == 16


def test_e02b_dry_episode_generation_records_balanced_cell_and_repeat_ids() -> None:
    config = _config()
    cells, repeats_per_cell = e02b._execution_geometry_design(
        config,
        dry_run=True,
    )
    episodes = e02b._heldout_episodes(
        checkpoint_seed=11,
        seed_offset=90_000,
        seed_block_size=100_000,
        cells=cells,
        repeats_per_cell=repeats_per_cell,
        registered_repeats_per_cell=32,
        data=e02b._require_mapping(config["data"], "data"),
    )

    assert [item.episode.operation for item in episodes[:4]] == list(Operation)
    assert [item.geometry_seed for item in episodes] == list(
        range(1_190_000, 1_190_064)
    )
    assert len({item.episode.episode_id for item in episodes}) == len(episodes)
    assert {item.cell_id for item in episodes} == set(range(16))
    assert {item.repeat_id for item in episodes} == {0}
    assert {
        (item.cell_id, item.episode.operation)
        for item in episodes
    } == {
        (cell_id, operation)
        for cell_id in range(16)
        for operation in Operation
    }
    for item in episodes[:: len(Operation)]:
        old_value = item.episode.old_value
        new_value = item.episode.new_value
        observed_cosine = float(
            torch.dot(old_value, new_value).item()
            / (old_value.norm().item() * new_value.norm().item())
        )
        assert float(old_value.norm().item()) == pytest.approx(item.old_scale)
        assert float(new_value.norm().item()) == pytest.approx(item.new_scale)
        assert observed_cosine == pytest.approx(item.old_new_cosine, abs=1e-6)


def test_e02b_e00_dependency_requires_full_only_for_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[bool] = []

    def fake_validate(
        artifact_root: str | Path,
        *,
        require_full: bool,
    ) -> dict[str, Any]:
        assert str(artifact_root) == "artifacts"
        calls.append(require_full)
        return {"experiment_id": "e00_protocol_lock", "full": require_full}

    monkeypatch.setattr(e02b, "validate_legacy_e00", fake_validate)

    assert e02b._validate_e00_dependency("artifacts", dry_run=False)["full"] is True
    assert e02b._validate_e00_dependency("artifacts", dry_run=True)["full"] is False
    assert calls == [True, False]


def test_e02b_preserves_original_inconclusive_adjudication_separately() -> None:
    assert e02b._ORIGINAL_E02_ADJUDICATION == {
        "original_confirmatory_status": "INCONCLUSIVE",
        "original_inconclusive_reason": (
            "PREREGISTERED_SYMMETRIC_RELATIVE_GATE_UNIDENTIFIABLE"
        ),
        "original_evaluable_gates_passed": "5/5",
        "original_h2_claim_open": False,
    }


def test_e02b_loads_a_persisted_strict_pair_without_training() -> None:
    source = e02b.validate_pinned_e02_source("artifacts", _config())
    tied = e02b._load_frozen_model(
        source.checkpoints[(11, "tied")],
        constraint=ScalarConstraint.TIED,
        device=torch.device("cpu"),
    )
    dual = e02b._load_frozen_model(
        source.checkpoints[(11, "dual")],
        constraint=ScalarConstraint.DUAL,
        device=torch.device("cpu"),
    )

    assert tied.constraint is ScalarConstraint.TIED
    assert dual.constraint is ScalarConstraint.DUAL
    assert sum(parameter.numel() for parameter in tied.parameters()) == 4994
    assert sum(parameter.numel() for parameter in dual.parameters()) == 4994


def test_e02b_supersede_gate_is_raw_absolute_equivalence_at_five_e_minus_four() -> None:
    statistics = e02b._require_mapping(_config()["statistics"], "statistics")
    margin = e02b._require_finite(
        statistics["supersede_absolute_equivalence_margin"],
        "SUPERSEDE margin",
    )

    assert margin == 0.0005
    assert e02b._equivalence_within(
        Interval(estimate=0.0, low=-0.00049, high=0.00049),
        margin,
    )
    assert not e02b._equivalence_within(
        Interval(estimate=0.0, low=-0.00051, high=0.00049),
        margin,
    )


def test_e02b_gate_one_requires_full_add_and_invalidate_support_per_seed() -> None:
    seeds = [11, 22]
    complete = {
        seed: {
            operation: {
                "expected": 512,
                "eligible": 512,
                "excluded_low_headroom": 0,
            }
            for operation in ("add", "invalidate")
        }
        for seed in seeds
    }

    assert e02b._asymmetric_registered_support_complete(
        complete,
        seeds,
        expected_per_operation=512,
        main_design=True,
    )

    one_excluded = {
        seed: {
            operation: dict(counts)
            for operation, counts in per_operation.items()
        }
        for seed, per_operation in complete.items()
    }
    one_excluded[11]["add"]["eligible"] = 511
    one_excluded[11]["add"]["excluded_low_headroom"] = 1
    assert not e02b._asymmetric_registered_support_complete(
        one_excluded,
        seeds,
        expected_per_operation=512,
        main_design=True,
    )

    missing_operation = {
        seed: dict(per_operation)
        for seed, per_operation in complete.items()
    }
    del missing_operation[22]["invalidate"]
    assert not e02b._asymmetric_registered_support_complete(
        missing_operation,
        seeds,
        expected_per_operation=512,
        main_design=True,
    )
    assert not e02b._asymmetric_registered_support_complete(
        complete,
        seeds,
        expected_per_operation=512,
        main_design=False,
    )


def test_e02b_claim_requires_exactly_all_six_gates() -> None:
    passing = {name: True for name in e02b._GATE_NAMES}
    assert e02b._six_gate_supported(passing)

    for gate_name in e02b._GATE_NAMES:
        one_failed = dict(passing)
        one_failed[gate_name] = False
        assert not e02b._six_gate_supported(one_failed)

    with_extra = {**passing, "posthoc_extra": True}
    assert not e02b._six_gate_supported(with_extra)


def test_e02b_repair_adjudication_separates_unevaluable_from_failed() -> None:
    incomplete_support = e02b._repair_adjudication(
        dry_run=False,
        exact_main_execution=True,
        inference_eligible=False,
        six_gate_supported=False,
    )
    assert incomplete_support == {
        "status": "INCONCLUSIVE",
        "reason": "REGISTERED_ASYMMETRIC_SUPPORT_INCOMPLETE",
        "evaluated": False,
        "inference_eligible": False,
        "six_gate_supported": False,
    }

    evaluated_failure = e02b._repair_adjudication(
        dry_run=False,
        exact_main_execution=True,
        inference_eligible=True,
        six_gate_supported=False,
    )
    assert evaluated_failure["status"] == "NOT_SUPPORTED"
    assert evaluated_failure["evaluated"] is True
    assert evaluated_failure["reason"] == "ONE_OR_MORE_EVALUABLE_GATES_FAILED"

    supported = e02b._repair_adjudication(
        dry_run=False,
        exact_main_execution=True,
        inference_eligible=True,
        six_gate_supported=True,
    )
    assert supported["status"] == "SUPPORTED"
    assert supported["evaluated"] is True
    assert supported["reason"] is None

    dry_run = e02b._repair_adjudication(
        dry_run=True,
        exact_main_execution=False,
        inference_eligible=False,
        six_gate_supported=False,
    )
    assert dry_run["status"] == "NOT_EVALUATED_DRY_RUN"
    assert dry_run["evaluated"] is False


def test_e02b_allowed_claim_distinguishes_five_fresh_from_one_inherited_gate() -> None:
    config = _config()
    claim = e02b._require_mapping(config["claim"], "claim")
    protocol = e02b._require_mapping(config["protocol"], "protocol")
    wording = str(claim["allowed_if_supported"])

    assert protocol["geometry_design"] == (
        "frozen-controller unseen norm/angle OOD heldout extension"
    )
    assert protocol["repair_scope"] == (
        "prospective absolute-SUPERSEDE repair plus OOD geometry extension"
    )
    assert "frozen-controller unseen norm/angle OOD heldout extension" in wording
    assert "five freshly evaluated gates" in wording
    assert "one preregistered inherited E02 tuning-direction fact" in wording
    assert "six-gate E02b repair criterion" in wording
    assert (
        "prospective absolute-SUPERSEDE repair plus OOD geometry extension"
        in wording
    )


def test_e02b_module_has_no_training_or_optimizer_import_path() -> None:
    source = inspect.getsource(e02b)
    tree = ast.parse(source)
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported_modules.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )

    assert not any(module.startswith("catena.training") for module in imported_modules)
    assert "torch.optim" not in source
    assert "train_matched_controller" not in source
