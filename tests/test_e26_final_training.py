from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from catena.lm.e26_final_training import (
    AUTOTUNE_MEASURED_STEPS,
    AUTOTUNE_WARMUP_STEPS,
    GLOBAL_BATCH_SEQUENCES,
    MICROBATCH_CANDIDATES,
    SEQUENCE_LENGTH,
    CandidateObservation,
    E26FinalTrainingError,
    deterministic_token_batch,
    registered_layout,
    select_autotune_candidate,
    telemetry_summary,
    token_plan_digest,
    validate_matched_variant_receipts,
)


def test_registered_layouts_preserve_per_run_global_token_batch() -> None:
    for microbatch in MICROBATCH_CANDIDATES:
        layout = registered_layout(microbatch)
        assert layout.physical_microbatch_sequences == microbatch
        assert layout.gradient_accumulation_steps * microbatch == GLOBAL_BATCH_SEQUENCES
        assert layout.global_batch_tokens == GLOBAL_BATCH_SEQUENCES * SEQUENCE_LENGTH
        assert layout.precision == "OFFICIAL_BF16_MIXED_AUTOCAST"
        assert layout.parameter_dtype == "torch.float32"

    with pytest.raises(E26FinalTrainingError, match="one of"):
        registered_layout(3)


def test_counter_based_tokens_are_replayable_and_cover_exact_global_batch() -> None:
    layout = registered_layout(8)
    parts = [
        deterministic_token_batch(
            seed=26_000,
            optimizer_step=7,
            microbatch_index=index,
            layout=layout,
            device="cpu",
        )
        for index in range(layout.gradient_accumulation_steps)
    ]
    replay = [
        deterministic_token_batch(
            seed=26_000,
            optimizer_step=7,
            microbatch_index=index,
            layout=layout,
            device="cpu",
        )
        for index in range(layout.gradient_accumulation_steps)
    ]
    assert torch.equal(torch.cat(parts), torch.cat(replay))
    assert torch.cat(parts).shape == (GLOBAL_BATCH_SEQUENCES, SEQUENCE_LENGTH + 1)
    assert int(torch.cat(parts).min()) >= 1
    assert int(torch.cat(parts).max()) < 32_000
    assert not torch.equal(parts[0], parts[1])


def test_token_plan_digest_changes_only_with_physical_recipe() -> None:
    layout = registered_layout(4)
    first = token_plan_digest(
        seed=26_000,
        layout=layout,
        warmup_steps=AUTOTUNE_WARMUP_STEPS,
        measured_steps=AUTOTUNE_MEASURED_STEPS,
    )
    assert first == token_plan_digest(
        seed=26_000,
        layout=layout,
        warmup_steps=AUTOTUNE_WARMUP_STEPS,
        measured_steps=AUTOTUNE_MEASURED_STEPS,
    )
    assert first != token_plan_digest(
        seed=26_001,
        layout=layout,
        warmup_steps=AUTOTUNE_WARMUP_STEPS,
        measured_steps=AUTOTUNE_MEASURED_STEPS,
    )


def _candidate(
    microbatch: int,
    throughput: float | None,
    memory: float | None,
    *,
    passed: bool = True,
) -> CandidateObservation:
    return CandidateObservation(
        microbatch_sequences=microbatch,
        passed=passed,
        disposition="PASS" if passed else "REJECTED_OOM",
        tokens_per_second=throughput,
        peak_vram_gib=memory,
        receipt_path=f"/tmp/mb{microbatch}.json",
        receipt_sha256=f"{microbatch:064x}",
    )


def test_autotune_is_throughput_only_with_conservative_tie_break() -> None:
    rows = [
        _candidate(1, 10_000.0, 40.0),
        _candidate(2, 20_000.0, 45.0),
        _candidate(4, 20_000.0, 50.0),
        _candidate(8, 30_000.0, 93.0),
        _candidate(16, None, None, passed=False),
    ]
    selected = select_autotune_candidate(rows)
    assert selected.microbatch_sequences == 2

    with pytest.raises(E26FinalTrainingError, match="exactly one"):
        select_autotune_candidate(rows[:-1])


def _matched_receipt(variant: str) -> dict[str, object]:
    return {
        "variant": variant,
        "passed": True,
        "parameter_surface_sha256": "a" * 64,
        "parameter_count": 1_450_096_416,
        "transformer_h_parameter_count": 1_302_638_112,
        "optimizer_surface_sha256": "b" * 64,
        "initialization_sha256": "c" * 64,
        "layout_sha256": "d" * 64,
        "token_plan_sha256": "e" * 64,
        "checkpoint_sha256": "f" * 64,
        "official_source_commit": "1" * 40,
        "official_runtime_source_sha256": "2" * 64,
        "precision": "BF16_AUTOCAST_FP32_PARAMETERS_OPTIMIZER_LOSS",
    }


def test_matched_variant_receipt_allows_only_policy_label_to_differ() -> None:
    tied = _matched_receipt("tied")
    dual = _matched_receipt("dual")
    assert validate_matched_variant_receipts(tied, dual)["passed"] is True

    changed = dict(dual)
    changed["layout_sha256"] = "9" * 64
    failed = validate_matched_variant_receipts(tied, changed)
    assert failed["passed"] is False
    assert failed["differing_fields"] == ["layout_sha256"]


def test_telemetry_summary_refuses_missing_or_nonfinite_samples() -> None:
    summary = telemetry_summary(
        [
            {"utilization_percent": 60.0, "power_watts": 250.0, "memory_used_mib": 1024.0},
            {"utilization_percent": 80.0, "power_watts": 350.0, "memory_used_mib": 2048.0},
        ]
    )
    assert summary["median_utilization_percent"] == 70.0
    assert summary["mean_power_watts"] == 300.0
    assert summary["peak_nvidia_smi_memory_gib"] == 2.0
    with pytest.raises(E26FinalTrainingError, match="no valid samples"):
        telemetry_summary([])
    with pytest.raises(E26FinalTrainingError, match="invalid"):
        telemetry_summary(
            [{"utilization_percent": float("nan"), "power_watts": 1.0, "memory_used_mib": 1.0}]
        )


def test_candidate_dataclass_replacement_does_not_mutate_original() -> None:
    original = _candidate(1, 1.0, 1.0)
    replaced = replace(original, tokens_per_second=2.0)
    assert original.tokens_per_second == 1.0
    assert replaced.tokens_per_second == 2.0
