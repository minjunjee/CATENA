from __future__ import annotations

from pathlib import Path

import torch
import yaml

from catena.data.structured_sequence_localization import (
    StructuredTransferCondition,
    StructuredTransferDemand,
    generate_structured_sequence_transfer_batch,
    make_structured_identifier_codebook,
    structured_base_transaction_digest,
    structured_event_feature_dim,
)
from catena.eval.structured_sequence_localization import (
    assess_structured_sequence_transfer,
    compute_structured_sequence_seed_contrasts,
    structured_sequence_aggregate_summary_ko,
    structured_sequence_source_summary_ko,
)
from catena.models.structured_sequence_localization import (
    MatchedStructuredSequenceController,
    StructuredSequenceFreedom,
    structured_sequence_parameter_count,
)
from catena.training.structured_sequence_localization import (
    evaluate_structured_sequence_controller,
    structured_state_dict_sha256,
    train_structured_sequence_controller,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _batch(
    *,
    family: StructuredTransferDemand,
    gap_events: int,
    seed: int = 71,
):
    codebook = make_structured_identifier_codebook(
        slots=8,
        code_dim=8,
        seed=21001,
    )
    return generate_structured_sequence_transfer_batch(
        family=family,
        batch_size=4,
        slots=8,
        value_dim=8,
        updates=2,
        gap_events=gap_events,
        state_scale=0.5,
        identifier_codebook=codebook,
        seed=seed,
        base_namespace="unit-base",
        distractor_namespace="unit-distractor",
        device=torch.device("cpu"),
    )


def test_structured_batch_hides_target_only_fields_and_pairs_base_across_gap():
    no_gap = _batch(
        family=StructuredTransferDemand.ADDRESS_DECOUPLING,
        gap_events=0,
    )
    with_gap = _batch(
        family=StructuredTransferDemand.ADDRESS_DECOUPLING,
        gap_events=5,
    )
    assert not hasattr(with_gap.inputs, "update_mask")
    assert not hasattr(with_gap.inputs, "old_candidates")
    assert not hasattr(with_gap.inputs, "erase_addresses")
    assert structured_base_transaction_digest(no_gap) == (
        structured_base_transaction_digest(with_gap)
    )
    assert with_gap.update_mask[0].nonzero().flatten().tolist() == [0, 6]
    assert torch.all(
        with_gap.inputs.verified_flags.squeeze(-1)[~with_gap.update_mask] == 0
    )
    assert with_gap.inputs.identifier_features.shape[-1] == 16
    assert structured_event_feature_dim(8, 8) == 43


def test_address_decoupling_is_the_only_different_address_family():
    for family in StructuredTransferDemand:
        batch = _batch(family=family, gap_events=0)
        verified = batch.update_mask
        different = (
            batch.erase_addresses[verified] != batch.write_addresses[verified]
        )
        if family is StructuredTransferDemand.ADDRESS_DECOUPLING:
            assert bool(different.all())
        else:
            assert not bool(different.any())


def test_oracle_information_route_reconstructs_target_when_activity_is_one():
    codebook = make_structured_identifier_codebook(
        slots=8,
        code_dim=8,
        seed=21001,
    )
    for family in StructuredTransferDemand:
        batch = generate_structured_sequence_transfer_batch(
            family=family,
            batch_size=4,
            slots=8,
            value_dim=8,
            updates=2,
            gap_events=0,
            state_scale=0.5,
            identifier_codebook=codebook,
            seed=91,
            base_namespace="oracle-test",
            distractor_namespace="oracle-test-distractor",
            device=torch.device("cpu"),
        )
        model = MatchedStructuredSequenceController(
            freedom=StructuredSequenceFreedom.BASE,
            slots=8,
            identifier_dim=8,
            value_dim=8,
            hidden_dim=16,
            address_temperature=0.2,
        )
        with torch.no_grad():
            model.activity_head.weight.zero_()
            model.activity_head.bias.fill_(30.0)
        output = model(
            batch,
            StructuredTransferCondition.A_ORACLE_ADDRESS_ORACLE_CANDIDATE,
        )
        assert torch.allclose(output.state, batch.target_state, atol=1e-6)


def test_four_freedoms_have_identical_parameter_surface_and_initialization():
    hashes = set()
    counts = set()
    for freedom in StructuredSequenceFreedom:
        torch.manual_seed(123)
        model = MatchedStructuredSequenceController(
            freedom=freedom,
            slots=8,
            identifier_dim=8,
            value_dim=8,
            hidden_dim=16,
            address_temperature=0.2,
        )
        hashes.add(structured_state_dict_sha256(model.state_dict()))
        counts.add(structured_sequence_parameter_count(model))
    assert len(hashes) == 1
    assert len(counts) == 1


def test_tiny_training_and_evaluation_are_finite_and_schema_complete():
    codebook = make_structured_identifier_codebook(
        slots=8,
        code_dim=8,
        seed=21001,
    )
    model = MatchedStructuredSequenceController(
        freedom=StructuredSequenceFreedom.FULL,
        slots=8,
        identifier_dim=8,
        value_dim=8,
        hidden_dim=16,
        address_temperature=0.2,
    )
    trace = train_structured_sequence_controller(
        model=model,
        conditions=list(StructuredTransferCondition),
        families=list(StructuredTransferDemand),
        steps=2,
        batch_size=2,
        slots=8,
        value_dim=8,
        updates=1,
        gap_events=2,
        state_scale=0.5,
        identifier_codebook=codebook,
        learning_rate=5e-4,
        address_loss_weight=0.25,
        candidate_loss_weight=1.0,
        activity_loss_weight=0.25,
        retention_weight=1.0,
        train_namespace="unit-train",
        distractor_namespace="unit-distractor",
        device=torch.device("cpu"),
        seed=777,
    )
    assert torch.isfinite(torch.tensor(trace.final_loss))
    metrics = evaluate_structured_sequence_controller(
        model=model,
        condition=StructuredTransferCondition.D_LEARNED_ADDRESS_STATE_READ_CANDIDATE,
        family=StructuredTransferDemand.ADDRESS_DECOUPLING,
        batches=1,
        batch_size=2,
        slots=8,
        value_dim=8,
        updates=1,
        gap_events=2,
        state_scale=0.5,
        identifier_codebook=codebook,
        evaluation_namespace="unit-evaluation",
        distractor_namespace="unit-distractor",
        device=torch.device("cpu"),
        seed=888,
    )
    assert set(metrics) == {
        "state_mse",
        "affected_mse",
        "retention_mse",
        "address_accuracy",
        "candidate_recovery_mse",
        "verified_activity_mean",
        "distractor_activity_mean",
        "verified_event_count",
        "distractor_event_count",
        "affected_entity_count",
        "retained_entity_count",
        "base_transaction_digest",
    }
    assert all(
        torch.isfinite(torch.tensor(float(metrics[key])))
        for key in (
            "state_mse",
            "affected_mse",
            "retention_mse",
            "address_accuracy",
            "candidate_recovery_mse",
        )
    )


def _synthetic_supported_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seeds = [113, 223, 331, 449, 557]
    variants = ["base", "separate_address", "state_aware", "full"]
    conditions = [
        "A_oracle_address_oracle_candidate",
        "B_learned_address_oracle_candidate",
        "C_oracle_address_state_read_candidate",
        "D_learned_address_state_read_candidate",
    ]
    families = [
        "magnitude_factorization",
        "value_granularity",
        "address_decoupling",
        "state_conditioning",
    ]
    for seed in seeds:
        for variant in variants:
            for condition in conditions:
                for family in families:
                    affected = 0.0
                    if (
                        condition == "B_learned_address_oracle_candidate"
                        and family == "address_decoupling"
                        and variant in {"base", "state_aware"}
                    ):
                        affected = 0.02
                    if (
                        condition == "C_oracle_address_state_read_candidate"
                        and family == "state_conditioning"
                        and variant in {"base", "separate_address"}
                    ):
                        affected = 0.02
                    if (
                        condition
                        == "D_learned_address_state_read_candidate"
                        and family == "address_decoupling"
                        and variant != "full"
                    ):
                        affected = 0.02
                    rows.append(
                        {
                            "seed": seed,
                            "variant": variant,
                            "condition": condition,
                            "demand_family": family,
                            "updates": 8,
                            "gap_events": 2048,
                            "affected_mse": affected,
                            "retention_mse": 0.0,
                            "address_accuracy": 1.0,
                            "candidate_recovery_mse": 0.0,
                            "verified_activity_mean": 1.0,
                            "distractor_activity_mean": 0.0,
                        }
                    )
    return rows


def test_registered_aggregate_pattern_and_one_page_summaries():
    rows = _synthetic_supported_rows()
    seeds = [113, 223, 331, 449, 557]
    contrasts = compute_structured_sequence_seed_contrasts(
        rows,
        seeds=seeds,
        stress_updates=8,
        stress_gap_events=2048,
    )
    config = yaml.safe_load(
        (
            REPO_ROOT
            / "configs/e21_structured_sequence_localization_transfer.yaml"
        ).read_text(encoding="utf-8")
    )
    assessment = assess_structured_sequence_transfer(
        contrasts,
        thresholds=config["claim_gate"],
        alpha=float(config["statistics"]["alpha"]),
        dry_run=False,
    )
    assert assessment["supported"] is True
    assert all(
        result["sign_flip_p"] == 0.03125
        for result in assessment["pattern"].values()
    )
    source_summary = structured_sequence_source_summary_ko(
        dry_run=True,
        seed=99121,
        rows=rows[:8],
        report_status="DRY_RUN",
        paired=True,
    )
    aggregate_summary = structured_sequence_aggregate_summary_ko(
        dry_run=False,
        assessment=assessment,
        seeds=seeds,
    )
    assert len(source_summary.splitlines()) <= 55
    assert len(aggregate_summary.splitlines()) <= 55
    assert "과학적 증거 아님" in source_summary
    assert "SUPPORTED" in aggregate_summary


def test_e18_e19_parent_locks_remain_at_recorded_hashes():
    import hashlib

    expected = {
        "docs/E18_SEQUENCE_CONTROL_LATTICE_LOCK.json": (
            "7c465ceb60b6979e717d85599533bd7c0dd884f10b191fa29c42771ccc9c9989"
        ),
        "docs/E19_LOCALIZATION_CANDIDATE_LOCK.json": (
            "8550fef23f938d84e35f584f16fd625cdb36c8422a6eefacf86f198f614dd3ec"
        ),
    }
    for relative, expected_hash in expected.items():
        digest = hashlib.sha256((REPO_ROOT / relative).read_bytes()).hexdigest()
        assert digest == expected_hash
