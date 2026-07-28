from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

import pytest
import torch
import yaml

from catena.post_e21.contracts import validate_protocol_lock
from catena.post_e21.text_anchor import (
    FrozenHashNgramEncoder,
    MatchedTextTransactionController,
    TextController,
    TextControllerOutput,
    _evaluate_output,
    matched_parameter_count,
    oracle_rows,
    summarize_rows,
)
from catena.post_e21.text_transactions import (
    MagnitudeOperation,
    OldRuleStatus,
    TextDemand,
    TextSplit,
    TextTransaction,
    build_text_transactions,
    decode_visible_policy_candidate,
    lexical_leakage,
    shuffled_texts,
    tensor_sha256,
    visible_registry_rows,
    wrong_entity_texts,
)
from experiments import e25b_text_transaction_anchor as e25b


def _examples(split: TextSplit) -> list[TextTransaction]:
    return build_text_transactions(
        split=split,
        demand_families=list(TextDemand),
        count_per_demand=6,
        slots=8,
        value_dim=8,
        namespace_seed=25_000_000_004_101,
        semantic_value_seed=25_000_000_099_000,
        blacklist=[
            "add",
            "delete",
            "revoke",
            "invalidate",
            "replace",
            "supersede",
            "erase",
            "write",
        ],
    )


def test_text_generator_hides_registered_direct_cues() -> None:
    blacklist = ("add", "delete", "revoke", "invalidate", "replace", "supersede")
    examples = _examples(TextSplit.PRIMARY)
    assert examples
    assert all(not lexical_leakage(example.text, blacklist) for example in examples)
    assert all("rule-" not in example.text.lower() for example in examples)
    registry = visible_registry_rows(examples)
    assert all("integer_address" not in row for row in registry)
    assert all("target_state" not in row for row in registry)
    assert all(
        not any(character.isdigit() for character in entity)
        for example in examples
        for entity in example.memory_entities
    )


def test_magnitude_train_has_anchors_and_primary_adds_unseen_composition() -> None:
    train = [
        example for example in _examples(TextSplit.TRAIN) if example.demand is TextDemand.MAGNITUDE
    ]
    primary = [
        example
        for example in _examples(TextSplit.PRIMARY)
        if example.demand is TextDemand.MAGNITUDE
    ]
    assert {example.magnitude_operation for example in train} == {
        MagnitudeOperation.ADD.value,
        MagnitudeOperation.INVALIDATE.value,
    }
    assert MagnitudeOperation.SUPERSEDE.value not in {
        example.magnitude_operation for example in train
    }
    assert {example.magnitude_operation for example in primary} == {
        operation.value for operation in MagnitudeOperation
    }
    add = next(
        example
        for example in primary
        if example.magnitude_operation == MagnitudeOperation.ADD.value
    )
    invalidate = next(
        example
        for example in primary
        if example.magnitude_operation == MagnitudeOperation.INVALIDATE.value
    )
    supersede = next(
        example
        for example in primary
        if example.magnitude_operation == MagnitudeOperation.SUPERSEDE.value
    )
    assert torch.allclose(
        add.target_state[add.erase_index],
        add.old_value + add.new_value,
    )
    assert torch.count_nonzero(invalidate.target_state[invalidate.erase_index]) == 0
    assert torch.allclose(
        supersede.target_state[supersede.erase_index],
        supersede.new_value,
    )
    assert not torch.allclose(
        add.target_state[add.erase_index],
        supersede.target_state[supersede.erase_index],
    )


def test_h2_geometry_has_asymmetric_tied_residual_and_symmetric_reachability() -> None:
    examples = [
        example
        for example in _examples(TextSplit.PRIMARY)
        if example.demand is TextDemand.MAGNITUDE and example.active
    ]
    by_operation = {example.magnitude_operation: example for example in examples}
    add = by_operation[MagnitudeOperation.ADD.value]
    invalidate = by_operation[MagnitudeOperation.INVALIDATE.value]
    supersede = by_operation[MagnitudeOperation.SUPERSEDE.value]

    def tied_best_mse(example: TextTransaction) -> float:
        old = example.old_value
        new = example.new_value
        target = example.target_state[example.erase_index]
        direction = new - old
        beta = torch.dot(target - old, direction) / torch.dot(direction, direction)
        beta = beta.clamp(0.0, 1.0)
        tied = (1.0 - beta) * old + beta * new
        return float(torch.square(tied - target).mean().item())

    assert abs(float(torch.dot(add.old_value, add.new_value).item())) < 1.0e-6
    assert tied_best_mse(add) > 0.001
    assert tied_best_mse(invalidate) > 0.001
    assert tied_best_mse(supersede) < 1.0e-12
    dual_add = (1.0 - 0.0) * add.old_value + 1.0 * add.new_value
    dual_invalidate = (1.0 - 1.0) * invalidate.old_value + 0.0 * invalidate.new_value
    tied_supersede = (1.0 - 1.0) * supersede.old_value + 1.0 * supersede.new_value
    assert torch.allclose(dual_add, add.target_state[add.erase_index])
    assert torch.allclose(
        dual_invalidate,
        invalidate.target_state[invalidate.erase_index],
    )
    assert torch.allclose(
        tied_supersede,
        supersede.target_state[supersede.erase_index],
    )
    assert torch.allclose(
        add.old_value + add.new_value,
        add.target_state[add.erase_index],
    )
    assert torch.allclose(
        torch.zeros_like(invalidate.old_value),
        invalidate.target_state[invalidate.erase_index],
    )
    assert torch.allclose(
        supersede.new_value,
        supersede.target_state[supersede.erase_index],
    )


def test_visible_policy_candidate_decoder_is_fixed_and_shared() -> None:
    examples = _examples(TextSplit.PRIMARY)[:2]
    expected = [
        decode_visible_policy_candidate(
            example.text,
            dimension=8,
            semantic_value_seed=25_000_000_099_000,
        )
        for example in examples
    ]
    assert all(
        torch.equal(candidate, example.new_value)
        for candidate, example in zip(expected, examples, strict=True)
    )
    incoming: list[torch.Tensor] = []
    for variant in (TextController.TIED, TextController.DUAL):
        torch.manual_seed(17)
        model = MatchedTextTransactionController(
            variant=variant,
            encoder=FrozenHashNgramEncoder(
                output_dim=24,
                buckets=128,
                ngram_min=1,
                ngram_max=2,
                seed=99,
            ),
            slots=8,
            value_dim=8,
            hidden_dim=24,
            semantic_value_seed=25_000_000_099_000,
        )
        assert not hasattr(model, "incoming_head")
        output = model(
            texts=[example.text for example in examples],
            state=torch.stack([example.state for example in examples]),
            memory_entities=[example.memory_entities for example in examples],
        )
        incoming.append(output.incoming)
    assert torch.equal(incoming[0], incoming[1])
    assert torch.equal(incoming[0], torch.stack(expected))


def test_minimal_pair_groups_cover_all_demand_relations() -> None:
    examples = _examples(TextSplit.PRIMARY)
    groups: dict[str, list[TextTransaction]] = {}
    for example in examples:
        groups.setdefault(example.minimal_pair_id, []).append(example)
    assert groups
    for group in groups.values():
        assert {example.demand for example in group} == set(TextDemand)
        assert len({tensor_sha256(example.state) for example in group}) == 1
        assert len({tensor_sha256(example.old_value) for example in group}) == 1
        assert len({tensor_sha256(example.new_value) for example in group}) == 1
        assert len({example.day for example in group}) == 1


def test_private_active_state_has_exact_surface_form_counterpart() -> None:
    examples = [
        example
        for example in _examples(TextSplit.PRIMARY)
        if example.demand is TextDemand.STATE_CONDITIONING
    ]
    by_text: dict[str, list[TextTransaction]] = {}
    by_counterpair: dict[str, list[TextTransaction]] = {}
    for example in examples:
        by_text.setdefault(example.text, []).append(example)
        by_counterpair.setdefault(example.state_counterpair_id, []).append(example)
    assert by_text
    assert all({example.active for example in group} == {False, True} for group in by_text.values())
    assert all(len(group) == 2 for group in by_counterpair.values())
    for group in by_counterpair.values():
        assert len({example.text for example in group}) == 1
        assert len({example.entity for example in group}) == 1
        assert len({example.other_entity for example in group}) == 1
        assert len({example.new_value_label for example in group}) == 1
        assert len({example.day for example in group}) == 1
        assert len({tensor_sha256(example.state) for example in group}) == 2


def test_visible_value_tokens_keep_one_semantic_target_across_splits() -> None:
    primary = _examples(TextSplit.PRIMARY)
    paraphrase = _examples(TextSplit.PARAPHRASE)
    primary_by_demand = {example.demand: example for example in primary}
    paraphrase_by_demand = {example.demand: example for example in paraphrase}
    for demand in TextDemand:
        assert torch.equal(
            primary_by_demand[demand].new_value,
            paraphrase_by_demand[demand].new_value,
        )


def test_derived_action_is_consistent_with_materialized_target() -> None:
    for example in _examples(TextSplit.PRIMARY):
        expected = int(example.target_state[example.write_index].argmax().item() % 4)
        assert example.derived_action == expected


def test_identifier_ood_uses_disjoint_opaque_memory_keys() -> None:
    primary_keys = set(_examples(TextSplit.PRIMARY)[0].memory_entities)
    identifier_keys = set(_examples(TextSplit.IDENTIFIER)[0].memory_entities)
    assert primary_keys.isdisjoint(identifier_keys)


def test_frozen_encoder_and_maximal_parameter_surface_are_shared() -> None:
    counts = set()
    fingerprints = set()
    for variant in TextController:
        torch.manual_seed(7)
        encoder = FrozenHashNgramEncoder(
            output_dim=24,
            buckets=128,
            ngram_min=1,
            ngram_max=2,
            seed=99,
        )
        model = MatchedTextTransactionController(
            variant=variant,
            encoder=encoder,
            slots=8,
            value_dim=8,
            hidden_dim=24,
            semantic_value_seed=25_000_000_099_000,
        )
        counts.add(matched_parameter_count(model))
        fingerprints.add(encoder.fingerprint())
        assert all(not parameter.requires_grad for parameter in encoder.parameters())
    assert len(counts) == 1
    assert len(fingerprints) == 1


def test_forward_accepts_shared_text_state_and_opaque_memory_keys() -> None:
    examples = _examples(TextSplit.DEVELOPMENT)[:3]
    model = MatchedTextTransactionController(
        variant=TextController.STATE_AWARE,
        encoder=FrozenHashNgramEncoder(
            output_dim=24,
            buckets=128,
            ngram_min=1,
            ngram_max=2,
            seed=99,
        ),
        slots=8,
        value_dim=8,
        hidden_dim=24,
        semantic_value_seed=25_000_000_099_000,
    )
    result = model(
        texts=[example.text for example in examples],
        state=torch.stack([example.state for example in examples]),
        memory_entities=[example.memory_entities for example in examples],
    )
    assert result.state.shape == (3, 8, 8)
    assert torch.isfinite(result.state).all()


def test_only_state_aware_projection_changes_controls_with_state() -> None:
    examples = _examples(TextSplit.DEVELOPMENT)[:2]
    texts = [examples[0].text, examples[0].text]
    base = examples[0].state
    states = torch.stack([base, torch.zeros_like(base)])
    for variant in TextController:
        torch.manual_seed(13)
        model = MatchedTextTransactionController(
            variant=variant,
            encoder=FrozenHashNgramEncoder(
                output_dim=24,
                buckets=128,
                ngram_min=1,
                ngram_max=2,
                seed=99,
            ),
            slots=8,
            value_dim=8,
            hidden_dim=24,
            semantic_value_seed=25_000_000_099_000,
        )
        output = model(
            texts=texts,
            state=states,
            memory_entities=[
                examples[0].memory_entities,
                examples[0].memory_entities,
            ],
        )
        controls_equal = torch.allclose(
            output.erase_gate[0], output.erase_gate[1]
        ) and torch.allclose(
            output.write_gate[0],
            output.write_gate[1],
        )
        if variant is TextController.STATE_AWARE:
            assert not controls_equal
        else:
            assert controls_equal


def test_negative_text_controls_are_factorized() -> None:
    examples = _examples(TextSplit.PRIMARY)
    wrong = wrong_entity_texts(examples)
    shuffled = shuffled_texts(examples)
    for example in examples:
        wrong_text = wrong[example.example_id]
        shuffled_text = shuffled[example.example_id]
        assert wrong_text != example.text
        assert shuffled_text != example.text
        # Wrong-entity changes address strings only; policy/day stay exact.
        assert example.new_value_label in wrong_text
        assert str(example.day) in wrong_text
        assert example.entity not in wrong_text
        # Shuffling changes policy/day while preserving the recipient address.
        assert example.entity in shuffled_text
        if example.demand is TextDemand.ADDRESS:
            assert example.other_entity in shuffled_text
        assert shuffled_text != wrong_text


def test_inactive_noop_direct_fact_and_old_rule_are_nontrivial() -> None:
    example = next(
        row
        for row in _examples(TextSplit.PRIMARY)
        if row.demand is TextDemand.STATE_CONDITIONING and not row.active
    )
    predicted = example.target_state.clone()
    predicted[example.erase_index].zero_()
    erase = torch.nn.functional.one_hot(
        torch.tensor([example.erase_index]),
        num_classes=example.state.shape[0],
    ).to(torch.float32)
    output = TextControllerOutput(
        state=predicted.unsqueeze(0),
        erase_address=erase,
        write_address=erase,
        erase_gate=torch.zeros(1, example.state.shape[1]),
        write_gate=torch.zeros(1, example.state.shape[1]),
        candidate=example.old_value.unsqueeze(0),
        incoming=example.new_value.unsqueeze(0),
    )
    row = _evaluate_output(
        examples=[example],
        output=output,
        condition="test",
        seed=1,
        accuracy_mse_threshold=0.001,
    )[0]
    assert row["affected_correction_mse"] == 0.0
    assert float(row["direct_fact_mse"]) > 0.0
    assert row["direct_fact_accuracy"] == 0.0
    assert row["old_rule_accuracy"] == 0.0
    assert float(row["old_rule_residual"]) > 0.0


def test_perfect_targets_pass_categorical_old_rule_status_for_every_branch() -> None:
    examples = build_text_transactions(
        split=TextSplit.PRIMARY,
        demand_families=list(TextDemand),
        count_per_demand=12,
        slots=8,
        value_dim=8,
        namespace_seed=25_000_000_004_101,
        semantic_value_seed=25_000_000_099_000,
        blacklist=[
            "add",
            "delete",
            "revoke",
            "invalidate",
            "replace",
            "supersede",
            "erase",
            "write",
        ],
    )
    rows = oracle_rows(examples, seed=17)
    assert len(rows) == len(examples)
    assert all(row["old_rule_accuracy"] == 1.0 for row in rows)
    assert all(row["gold_old_rule_status"] == row["predicted_old_rule_status"] for row in rows)
    assert {
        (example.demand, example.magnitude_operation, example.active) for example in examples
    } >= {
        (TextDemand.MAGNITUDE, operation.value, active)
        for operation in MagnitudeOperation
        for active in (False, True)
    }
    assert {(example.demand, example.active) for example in examples} >= {
        (demand, active) for demand in TextDemand for active in (False, True)
    }
    expected = {
        MagnitudeOperation.ADD.value: OldRuleStatus.FULL,
        MagnitudeOperation.INVALIDATE.value: OldRuleStatus.NONE,
        MagnitudeOperation.SUPERSEDE.value: OldRuleStatus.NONE,
    }
    for example in examples:
        if example.demand is TextDemand.MAGNITUDE:
            assert example.old_rule_status is expected[example.magnitude_operation]
        elif example.demand is TextDemand.VALUE:
            assert example.old_rule_status is OldRuleStatus.PARTIAL
        elif example.demand is TextDemand.ADDRESS:
            assert example.old_rule_status is OldRuleStatus.NONE
        else:
            assert example.old_rule_status is (
                OldRuleStatus.NONE if example.active else OldRuleStatus.FULL
            )


def test_oracle_queries_use_real_evaluator_and_wrong_value_fails_partial() -> None:
    examples = _examples(TextSplit.PRIMARY)
    oracle = oracle_rows(examples, seed=19)
    for row in oracle:
        assert row["direct_fact_accuracy"] == 1.0
        assert row["derived_action_accuracy"] == 1.0
        assert row["old_rule_accuracy"] == 1.0
        assert row["erase_address_accuracy"] == 1.0
        assert row["write_address_accuracy"] == 1.0
        assert row["affected_correction_mse"] == 0.0
        assert row["unaffected_retention_mse"] == 0.0

    example = next(row for row in examples if row.demand is TextDemand.VALUE)
    predicted = example.target_state.clone()
    predicted[example.erase_index] = example.coordinate_mask * example.new_value
    erase = torch.nn.functional.one_hot(
        torch.tensor([example.erase_index]),
        num_classes=example.state.shape[0],
    ).to(torch.float32)
    output = TextControllerOutput(
        state=predicted.unsqueeze(0),
        erase_address=erase,
        write_address=erase,
        erase_gate=torch.zeros(1, example.state.shape[1]),
        write_gate=torch.zeros(1, example.state.shape[1]),
        candidate=example.old_value.unsqueeze(0),
        incoming=example.new_value.unsqueeze(0),
    )
    wrong = _evaluate_output(
        examples=[example],
        output=output,
        condition="wrong_value",
        seed=19,
        accuracy_mse_threshold=0.001,
    )[0]
    assert wrong["gold_old_rule_status"] == OldRuleStatus.PARTIAL.value
    assert wrong["predicted_old_rule_status"] == OldRuleStatus.NONE.value
    assert wrong["old_rule_accuracy"] == 0.0


def test_oracle_headroom_normalized_recovery_is_explicit() -> None:
    examples = _examples(TextSplit.PRIMARY)[:1]
    oracle = oracle_rows(examples, seed=7)
    base = {
        key: value
        for key, value in oracle[0].items()
        if key not in {"condition", "affected_correction_mse"}
    }
    rows = [
        {**base, "condition": "tied", "affected_correction_mse": 0.01},
        {**base, "condition": "dual", "affected_correction_mse": 0.005},
        {**oracle[0], "affected_correction_mse": 0.0},
    ]
    summary = summarize_rows(
        rows,
        minimum_identifiable_oracle_headroom=1.0e-12,
    )
    dual = next(row for row in summary if row["condition"] == "dual")
    assert dual["oracle_headroom_identifiable"] is True
    assert float(dual["oracle_headroom_normalized_recovery"]) == 0.5


def _assessment_rows() -> list[dict[str, object]]:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (root / "configs/e25b_text_transaction_anchor.yaml").read_text(encoding="utf-8")
    )
    rows: list[dict[str, object]] = []
    cells = {
        ("magnitude", MagnitudeOperation.ADD.value): {
            "tied": 0.02,
            "dual": 0.0,
            "state_aware": 0.0,
        },
        ("magnitude", MagnitudeOperation.INVALIDATE.value): {
            "tied": 0.02,
            "dual": 0.0,
            "state_aware": 0.0,
        },
        ("magnitude", MagnitudeOperation.SUPERSEDE.value): {
            "tied": 0.00001,
            "dual": 0.000012,
            "state_aware": 0.00001,
        },
        ("value", "not_applicable"): {
            "dual": 0.02,
            "diagonal": 0.0,
            "state_aware": 0.0,
        },
        ("address", "not_applicable"): {
            "diagonal": 0.02,
            "separate_address": 0.0,
            "state_aware": 0.0,
        },
        ("state_conditioning", "not_applicable"): {
            "separate_address": 0.02,
            "state_aware": 0.0,
        },
    }
    for seed in config["seeds"]:
        for (demand, operation), controllers in cells.items():
            for condition, error in controllers.items():
                rows.append(
                    {
                        "seed": int(seed),
                        "split": "primary",
                        "demand_family": demand,
                        "magnitude_operation": operation,
                        "condition": condition,
                        "affected_correction_mse": error,
                        "unaffected_retention_mse": 0.0,
                        "erase_address_accuracy": 1.0,
                        "write_address_accuracy": 1.0,
                        "oracle_headroom_identifiable": False,
                        "oracle_headroom_normalized_recovery": None,
                    }
                )
            rows.append(
                {
                    "seed": int(seed),
                    "split": "primary",
                    "demand_family": demand,
                    "magnitude_operation": operation,
                    "condition": "oracle_demand",
                    "affected_correction_mse": 0.0,
                    "unaffected_retention_mse": 0.0,
                    "erase_address_accuracy": 1.0,
                    "write_address_accuracy": 1.0,
                    "oracle_headroom_identifiable": False,
                    "oracle_headroom_normalized_recovery": None,
                }
            )
            full_error = float(controllers["state_aware"])
            for condition in (
                "shuffled_text",
                "wrong_entity",
                "transaction_only_zero_state",
                "state_only",
            ):
                rows.append(
                    {
                        "seed": int(seed),
                        "split": "primary",
                        "demand_family": demand,
                        "magnitude_operation": operation,
                        "condition": condition,
                        "affected_correction_mse": full_error + 0.002,
                        "unaffected_retention_mse": 0.0,
                        "erase_address_accuracy": 1.0,
                        "write_address_accuracy": 1.0,
                        "oracle_headroom_identifiable": False,
                        "oracle_headroom_normalized_recovery": None,
                    }
                )
    return rows


def test_assessment_separates_asymmetric_gain_from_supersede_equivalence() -> None:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (root / "configs/e25b_text_transaction_anchor.yaml").read_text(encoding="utf-8")
    )
    rows = _assessment_rows()
    result = e25b._assessment(rows, config=config, dry_run=False)
    assert result["magnitude_asymmetric_gain"]["passed"] is True
    assert result["magnitude_supersede_composition_equivalence"]["passed"] is True
    assert "magnitude" not in result["interaction_effects"]

    for row in rows:
        if (
            row["demand_family"] == "magnitude"
            and row["magnitude_operation"] == MagnitudeOperation.SUPERSEDE.value
            and row["condition"] == "dual"
        ):
            row["affected_correction_mse"] = 0.002
    failed = e25b._assessment(rows, config=config, dry_run=False)
    assert failed["magnitude_asymmetric_gain"]["passed"] is True
    assert failed["magnitude_supersede_composition_equivalence"]["passed"] is False
    assert failed["supported"] is False


def test_audit_population_has_exact_registered_size() -> None:
    examples = build_text_transactions(
        split=TextSplit.PRIMARY,
        demand_families=list(TextDemand),
        count_per_demand=75,
        slots=8,
        value_dim=8,
        namespace_seed=25_000_000_009_901,
        semantic_value_seed=25_000_000_099_000,
        blacklist=[
            "add",
            "delete",
            "revoke",
            "invalidate",
            "replace",
            "supersede",
            "erase",
            "write",
        ],
    )
    assert len(examples) == 300
    assert all(example.minimal_pair_id for example in examples)


def _reviewed_audit(tmp_path: Path) -> tuple[dict[str, object], Path, Path]:
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs/e25b_text_transaction_anchor.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    prep_dir = tmp_path / "audit-preparation"
    prep_dir.mkdir()
    artifacts = e25b._write_audit_artifacts(
        run_dir=prep_dir,
        examples=e25b._audit_examples(config),
        config=config,
        config_path=config_path,
    )
    template = Path(str(artifacts["review_template"]["path"]))
    work_dir = tmp_path / "review-work"
    work_dir.mkdir()
    reviewed = work_dir / str(config["audit"]["review_work_filename"])
    shutil.copyfile(template, reviewed)
    with reviewed.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for field in e25b._AUDIT_REVIEW_FIELDS:
            if field.startswith(("reviewer_a_", "reviewer_b_")):
                row[field] = "PASS"
    with reviewed.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(e25b._AUDIT_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    population_lock = Path(str(artifacts["population_lock"]["path"]))
    return config, reviewed, population_lock


def test_audit_is_bound_to_exact_prepared_population_and_gold(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config, reviewed, population_lock = _reviewed_audit(tmp_path)
    result = e25b._validate_human_audit(
        reviewed,
        population_lock_path=population_lock,
        config=config,
        config_path=root / "configs/e25b_text_transaction_anchor.yaml",
    )
    assert result["passed"] is True
    assert result["rows"] == 300
    assert reviewed.parent != population_lock.parent
    lock = json.loads(population_lock.read_text(encoding="utf-8"))
    assert result["population_sha256"] == lock["population_sha256"]
    with reviewed.open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["private_old_value_sha256"]
    assert row["private_new_value_sha256"]
    assert row["gold_direct_fact_answer"]
    assert row["gold_old_rule_status"] in {
        OldRuleStatus.FULL.value,
        OldRuleStatus.PARTIAL.value,
        OldRuleStatus.NONE.value,
    }
    assert "PARTIAL=" in row["gold_old_rule_status_definition"]
    vector = [float(value) for value in row["gold_direct_fact_vector_values"].split("|")]
    taxonomy = ("HOLD", "AUTHORIZE", "ESCALATE", "MONITOR")
    expected_action = taxonomy[max(range(len(vector)), key=vector.__getitem__) % 4]
    assert row["gold_derived_action"] == expected_action
    assert "0=HOLD" in row["gold_derived_action_rule"]
    template = population_lock.parent / str(config["audit"]["review_template_filename"])
    with template.open("r", encoding="utf-8", newline="") as handle:
        template_row = next(csv.DictReader(handle))
    assert all(not template_row[field] for field in e25b._AUDIT_REVIEW_FIELDS)


def test_review_work_cannot_live_in_finalized_preparation_run(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config, reviewed, population_lock = _reviewed_audit(tmp_path)
    forbidden = population_lock.parent / str(config["audit"]["review_work_filename"])
    shutil.copyfile(reviewed, forbidden)
    with pytest.raises(ValueError, match="outside"):
        e25b._validate_human_audit(
            forbidden,
            population_lock_path=population_lock,
            config=config,
            config_path=root / "configs/e25b_text_transaction_anchor.yaml",
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("audit_id", "E25B-TAMPERED"),
        ("example_id", "tampered-example"),
        ("text", "tampered transaction"),
        ("split", "tampered-split"),
        ("minimal_pair_id", "tampered-pair"),
        ("state_counterpair_id", "tampered-counterpair"),
    ],
)
def test_audit_rejects_identity_population_mutation(
    tmp_path: Path,
    field: str,
    replacement: str,
) -> None:
    root = Path(__file__).resolve().parents[1]
    config, reviewed, population_lock = _reviewed_audit(tmp_path)
    with reviewed.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0][field] = replacement
    with reviewed.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(e25b._AUDIT_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError):
        e25b._validate_human_audit(
            reviewed,
            population_lock_path=population_lock,
            config=config,
            config_path=root / "configs/e25b_text_transaction_anchor.yaml",
        )


@pytest.mark.parametrize("field", ["audit_id", "example_id"])
def test_audit_rejects_duplicate_identity(
    tmp_path: Path,
    field: str,
) -> None:
    root = Path(__file__).resolve().parents[1]
    config, reviewed, population_lock = _reviewed_audit(tmp_path)
    with reviewed.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[1][field] = rows[0][field]
    with reviewed.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(e25b._AUDIT_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="unique"):
        e25b._validate_human_audit(
            reviewed,
            population_lock_path=population_lock,
            config=config,
            config_path=root / "configs/e25b_text_transaction_anchor.yaml",
        )


def test_main_validates_audit_before_initialize_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    called = False

    def reject_audit(*args: object, **kwargs: object) -> dict[str, object]:
        raise ValueError("audit rejected before run allocation")

    def forbidden_initialize(*args: object, **kwargs: object) -> None:
        nonlocal called
        called = True
        raise AssertionError("initialize_run must not be reached")

    monkeypatch.setattr(e25b, "_validate_human_audit", reject_audit)
    monkeypatch.setattr(e25b, "initialize_run", forbidden_initialize)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "e25b_text_transaction_anchor.py",
            "--config",
            str(root / "configs/e25b_text_transaction_anchor.yaml"),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--audit-csv",
            str(tmp_path / "E25B_V4_HUMAN_AUDIT_REVIEW_WORK.csv"),
            "--audit-population-lock",
            str(tmp_path / "E25B_V4_HUMAN_AUDIT_POPULATION_LOCK.json"),
        ],
    )
    with pytest.raises(ValueError, match="audit rejected"):
        e25b.main()
    assert called is False
    assert not (tmp_path / "artifacts").exists()


def test_protocol_doc_and_lock_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    assert (root / "docs/E25B_TEXT_TRANSACTION_PROTOCOL_KO.md").is_file()
    assert (root / "docs/E25B_TEXT_TRANSACTION_LOCK.json").is_file()
    snapshot = validate_protocol_lock(
        lock_path=root / "docs/E25B_TEXT_TRANSACTION_LOCK.json",
        config_path=root / "configs/e25b_text_transaction_anchor.yaml",
        experiment_id="e25b_text_transaction_anchor",
        repo_root=root,
    )
    assert snapshot.payload["main_execution_started"] is False
    config = yaml.safe_load(
        (root / "configs/e25b_text_transaction_anchor.yaml").read_text(encoding="utf-8")
    )
    assert config["statistics"]["shared_encoder_controller_floor_ceiling"] == 0.001
    assert config["statistics"]["minimum_state_aware_address_accuracy"] == 0.95
    assert config["protocol"]["protocol_id"].endswith("_v4")
    assert config["audit"]["review_work_filename"] == "E25B_V4_HUMAN_AUDIT_REVIEW_WORK.csv"
    assert config["artifacts"]["raw_metrics_filename"] == "text_transaction_metrics.jsonl"
    assert config["artifacts"]["seed_metrics_filename"] == "text_transaction_seed_metrics.jsonl"
    assert (
        config["encoder"]["visible_candidate_decoder"]["learned_controller_incoming_head"] is False
    )
