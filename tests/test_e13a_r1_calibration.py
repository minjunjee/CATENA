from __future__ import annotations

import torch

from catena.core.config import load_config
from catena.data.transactional_sequence import (
    generate_transactional_sequence_batch,
)
from catena.models.sequence_memory import (
    SequenceControl,
    TransactionalSequenceMemory,
)
from catena.training.sequence_training import evaluate_sequence_memory
from experiments.e13a_r1_sequence_floor_throughput import (
    _state_dict_sha256,
    measure_e13b_scale_training_feasibility,
    measure_paired_forward_throughput,
    project_e13b_schedule,
)
from experiments.e13b_transactional_sequence_memory import (
    CALIBRATION_EXPERIMENT_ID,
)


def _model(control: SequenceControl) -> TransactionalSequenceMemory:
    return TransactionalSequenceMemory(
        control=control,
        num_entities=8,
        value_vocab=16,
        embedding_dim=8,
        hidden_dim=16,
    )


def test_tied_and_dual_can_use_identical_registered_initialization() -> None:
    torch.manual_seed(101)
    tied = _model(SequenceControl.TIED)
    torch.manual_seed(101)
    dual = _model(SequenceControl.DUAL)

    assert _state_dict_sha256(tied.state_dict()) == _state_dict_sha256(
        dual.state_dict()
    )
    assert sum(parameter.numel() for parameter in tied.parameters()) == sum(
        parameter.numel() for parameter in dual.parameters()
    )


def test_paired_forward_timing_is_repeated_and_excludes_generation() -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(9)
    batch = generate_transactional_sequence_batch(
        batch_size=2,
        num_entities=8,
        value_vocab=16,
        updates=1,
        gap_events=0,
        generator=generator,
        device=torch.device("cpu"),
    )
    result = measure_paired_forward_throughput(
        models={
            "tied": _model(SequenceControl.TIED),
            "dual": _model(SequenceControl.DUAL),
        },
        batch=batch,
        device=torch.device("cpu"),
        warmup_repeats=1,
        measured_repeats=2,
    )

    assert set(result) == {"tied", "dual"}
    assert all(
        row["timing_method"] == "paired_alternating_forward_only"
        for row in result.values()
    )
    assert all(row["measured_repeats"] == 2 for row in result.values())
    assert all(row["examples_per_second"] > 0 for row in result.values())


def test_e13b_scale_projection_uses_registered_steps_batch_and_waves() -> None:
    result = project_e13b_schedule(
        examples_per_second={"tied": 128.0, "dual": 256.0},
        steps=30_000,
        batch_size=128,
        planned_wave_jobs=(4, 4, 2),
    )

    assert result["projected_seconds_per_run"]["tied"] == 30_000.0
    assert result["projected_seconds_per_run"]["dual"] == 15_000.0
    assert result["projected_seconds_per_wave"] == [
        30_000.0,
        30_000.0,
        30_000.0,
    ]
    assert result["projected_total_sequential_wave_seconds"] == 90_000.0


def test_e13b_scale_probe_measures_training_path_before_projection() -> None:
    source_config = {
        "data": {"num_entities": 8, "value_vocab": 16},
        "model": {
            "variants": ["tied", "dual"],
            "embedding_dim": 8,
            "hidden_dim": 16,
        },
        "training": {
            "steps": 10,
            "batch_size": 2,
            "updates": 1,
            "gap_events": 0,
            "learning_rate": 0.001,
            "retention_weight": 1.0,
        },
    }

    measurements, schedule = measure_e13b_scale_training_feasibility(
        source_config=source_config,
        device=torch.device("cpu"),
        seed=23,
        warmup_steps=0,
        measured_steps=1,
        planned_wave_jobs=(4, 4, 2),
    )

    assert set(measurements) == {"tied", "dual"}
    assert all(
        row["training_examples_per_second"] > 0
        for row in measurements.values()
    )
    assert schedule["planned_wave_jobs"] == [4, 4, 2]
    assert schedule["projected_slowest_run_seconds"] > 0


def test_sequence_evaluation_reports_affected_exact_and_real_denominators() -> None:
    class ExactModel(torch.nn.Module):
        def forward(self, batch):  # type: ignore[no-untyped-def]
            return batch.target_state

    metrics = evaluate_sequence_memory(
        model=ExactModel(),  # type: ignore[arg-type]
        batches=2,
        batch_size=4,
        num_entities=8,
        value_vocab=16,
        updates=2,
        gap_events=1,
        device=torch.device("cpu"),
        seed=17,
    )

    assert metrics["affected_entity_exact_match"] == 1.0
    assert metrics["affected_entity_count"] > 0
    assert metrics["unaffected_entity_count"] > 0
    assert metrics["retention_mse"] == 0.0


def test_only_prospective_r1_can_open_e13b() -> None:
    config = load_config("configs/e13a_r1_sequence_floor_throughput.yaml")

    assert CALIBRATION_EXPERIMENT_ID == "e13a_r1_sequence_floor_throughput"
    assert config["claim_gate"]["minimum_dual_affected_exact_match"] == 0.95
    assert config["claim_gate"]["maximum_dual_affected_mse"] == 0.001
    assert config["feasibility"]["source_config_path"] == (
        "configs/e13b_transactional_sequence_memory.yaml"
    )
    assert config["feasibility"]["planned_wave_jobs"] == [4, 4, 2]
