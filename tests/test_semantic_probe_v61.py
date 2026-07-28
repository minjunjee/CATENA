import torch

from catena.core.schema import Operation
from catena.data.semantic_controls_v61 import (
    SemanticControl,
    build_control_pairing_registry,
)
from catena.data.semantic_transactions_v61 import (
    SemanticMemorySpec,
    SemanticNamespaceRegistry,
    build_balanced_semantic_examples,
)
from catena.models.semantic_controllers_v61 import (
    MatchedSemanticControllerV61,
    SemanticRoute,
)
from catena.models.semantic_encoder_v61 import (
    FrozenSemanticFieldEncoderV61,
    SemanticFeatureConfigV61,
)
from catena.training.semantic_probe_v61 import (
    apply_batched_visible_update,
    evaluate_semantic_model,
    per_example_behavioral_metrics,
    tensorize_semantic_examples,
)


def _examples():
    registry = SemanticNamespaceRegistry(
        integer_root=5_000_000_000_000,
        split_stride=100_000_000,
        seed_stride=100_000,
        split_offsets=(("pilot", 1),),
        dry_run=True,
        prior_numeric_seed_max=100_000_000,
    )
    return build_balanced_semantic_examples(
        namespace_registry=registry,
        namespace_name="pilot",
        checkpoint_seed=101,
        seed_slot=0,
        operations=(Operation.PRESERVE, Operation.ADD, Operation.INVALIDATE),
        domains=("api",),
        templates=("record",),
        count_per_cell=2,
        memory_spec=SemanticMemorySpec(num_associations=4, key_dim=8, value_dim=8),
    )


def _encoder():
    return FrozenSemanticFieldEncoderV61(
        SemanticFeatureConfigV61(
            categorical_fields=(
                "entity_description",
                "domain",
                "current_relation",
                "incoming_evidence",
                "scope",
                "source",
                "provenance",
                "incoming_value_token",
                "template_surface",
            ),
            numeric_fields=(
                "prior_version",
                "evidence_version",
                "observation_day",
                "evidence_timestamp_day",
                "prior_valid_from_day",
                "prior_valid_to_day",
                "evidence_valid_from_day",
                "evidence_valid_to_day",
            ),
            categorical_bins_per_field=8,
            version_scale=16.0,
            day_scale=256.0,
            state_read_dim=8,
        )
    )


def test_oracle_visible_path_reaches_numerical_zero_without_private_candidate():
    examples = _examples()
    batch = tensorize_semantic_examples(
        examples,
        encoder=_encoder(),
        control=SemanticControl.FULL,
    )
    output = apply_batched_visible_update(
        batch.visible,
        batch.operation_demand[:, 0],
        batch.operation_demand[:, 1],
    )
    metrics = per_example_behavioral_metrics(output, batch)
    assert metrics["affected_read_mse"].max().item() < 1e-10
    assert metrics["unaffected_retention_mse"].max().item() < 1e-10


def test_transaction_only_and_wrong_address_use_public_control_views():
    examples = _examples()
    registry = build_control_pairing_registry(examples)
    model = MatchedSemanticControllerV61(
        _encoder().config.input_dim,
        8,
        SemanticRoute.FACTORIZED,
    )
    full_rows, full, _ = evaluate_semantic_model(
        model,
        examples,
        encoder=_encoder(),
        control=SemanticControl.FULL,
        pairing_registry=None,
        oracle_demand=False,
        device=torch.device("cpu"),
    )
    transaction_rows, transaction, _ = evaluate_semantic_model(
        model,
        examples,
        encoder=_encoder(),
        control=SemanticControl.TRANSACTION_ONLY,
        pairing_registry=registry,
        oracle_demand=False,
        device=torch.device("cpu"),
    )
    wrong_rows, wrong, _ = evaluate_semantic_model(
        model,
        examples,
        encoder=_encoder(),
        control=SemanticControl.WRONG_ADDRESS,
        pairing_registry=registry,
        oracle_demand=False,
        device=torch.device("cpu"),
    )
    assert len(full_rows) == len(transaction_rows) == len(wrong_rows) == len(examples)
    assert transaction.mean() > full.mean()
    assert wrong.mean() > full.mean()
