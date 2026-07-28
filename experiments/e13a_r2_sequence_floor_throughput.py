from __future__ import annotations

from collections.abc import Mapping
from statistics import median
from time import perf_counter

import torch

import experiments.e13a_r1_sequence_floor_throughput as r1
from catena.core.config import load_config
from catena.data.transactional_sequence_v2 import (
    base_transaction_digest_v2,
    generate_transactional_sequence_batch_v2,
)
from catena.models.sequence_memory_v2 import (
    SequenceControlV2,
    TransactionalSequenceMemoryV2,
    sequence_parameter_count_v2,
)
from catena.training.sequence_training_v2 import (
    evaluate_sequence_memory_v2,
    train_sequence_memory_v2,
)

EXPERIMENT_ID = "e13a_r2_sequence_floor_throughput"
DEFAULT_CONFIG = "configs/e13a_r2_sequence_floor_throughput.yaml"


def measure_paired_forward_throughput_v2(
    *,
    models: Mapping[str, TransactionalSequenceMemoryV2],
    batch: object,
    device: torch.device,
    warmup_repeats: int,
    measured_repeats: int,
) -> dict[str, dict[str, float | int | str]]:
    if warmup_repeats < 0 or measured_repeats <= 0:
        raise ValueError("timing repeats must be nonnegative/positive")
    names = tuple(models)
    if not names:
        raise ValueError("models must not be empty")
    inputs = batch.inputs
    for model in models.values():
        model.to(device).eval()
    with torch.no_grad():
        for _ in range(warmup_repeats):
            for name in names:
                models[name](inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        durations: dict[str, list[float]] = {name: [] for name in names}
        for repeat in range(measured_repeats):
            order = names if repeat % 2 == 0 else tuple(reversed(names))
            for name in order:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = perf_counter()
                models[name](inputs)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                durations[name].append(perf_counter() - started)

    batch_size = int(inputs.initial_state.shape[0])
    return {
        name: {
            "timing_method": "paired_alternating_forward_only_v2",
            "warmup_repeats": warmup_repeats,
            "measured_repeats": measured_repeats,
            "median_forward_seconds": median(durations[name]),
            "examples_per_second": batch_size
            / max(median(durations[name]), 1e-12),
        }
        for name in names
    }


def _generate_from_legacy_generator(
    *,
    batch_size: int,
    num_entities: int,
    value_vocab: int,
    updates: int,
    gap_events: int,
    generator: torch.Generator,
    device: torch.device,
) -> object:
    return generate_transactional_sequence_batch_v2(
        batch_size=batch_size,
        num_entities=num_entities,
        value_vocab=value_vocab,
        updates=updates,
        gap_events=gap_events,
        seed=int(generator.initial_seed()),
        device=device,
    )


def _distractor_contract(config: dict) -> dict[str, bool | float | str]:
    updates = int(config["data"]["updates"])
    gap_events = int(config["data"]["gap_events"])
    seed = int(config["evaluation"]["seed"]) + 71
    common = {
        "batch_size": 4,
        "num_entities": int(config["data"]["num_entities"]),
        "value_vocab": int(config["data"]["value_vocab"]),
        "updates": updates,
        "seed": seed,
        "device": torch.device("cpu"),
    }
    no_gap = generate_transactional_sequence_batch_v2(
        **common,
        gap_events=0,
    )
    with_gap = generate_transactional_sequence_batch_v2(
        **common,
        gap_events=gap_events,
    )
    paired_digest = (
        base_transaction_digest_v2(no_gap)
        == base_transaction_digest_v2(with_gap)
    )
    first_row_mask = with_gap.update_mask[0].detach().cpu()
    true_positions = first_row_mask.nonzero(as_tuple=False).flatten().tolist()
    expected_positions = [0] if updates == 1 else [
        0,
        *range(gap_events + 1, gap_events + updates),
    ]
    interleaving = true_positions == expected_positions

    torch.manual_seed(seed + 1)
    model = TransactionalSequenceMemoryV2(
        control=SequenceControlV2.DUAL,
        num_entities=int(config["data"]["num_entities"]),
        value_vocab=int(config["data"]["value_vocab"]),
        embedding_dim=int(config["model"]["embedding_dim"]),
        hidden_dim=int(config["model"]["hidden_dim"]),
    ).eval()
    with torch.no_grad():
        no_gap_state = model(no_gap.inputs).state
        with_gap_state = model(with_gap.inputs).state
    path_delta = float((with_gap_state - no_gap_state).abs().max())
    minimum_delta = float(
        config["claim_gate"]["minimum_unmasked_path_delta"]
    )
    model_input_excludes_target_mask = not hasattr(
        with_gap.inputs,
        "update_mask",
    )
    protocol = config.get("protocol", {})
    protocol_fields_match = bool(
        protocol.get("verified_role") == "semantic_input_only"
        and protocol.get("update_mask_role") == "audit_metadata_only"
        and protocol.get("paired_base_transactions_across_gaps") is True
    )
    passed = bool(
        paired_digest
        and interleaving
        and path_delta > minimum_delta
        and model_input_excludes_target_mask
        and protocol_fields_match
    )
    return {
        "passed": passed,
        "base_transaction_digest_matched_across_gap": paired_digest,
        "verified_event_positions_matched_contract": interleaving,
        "model_input_excludes_update_mask": model_input_excludes_target_mask,
        "protocol_fields_matched": protocol_fields_match,
        "random_initialization_full_vs_no_gap_max_abs_delta": path_delta,
        "minimum_unmasked_path_delta": minimum_delta,
        "layout": "one_total_block_after_first_verified_update",
    }


def _install_r2_runtime() -> None:
    original_finalize = r1.finalize_run

    def finalize_r2_run(
        *,
        experiment_id: str,
        artifact_root: str,
        run_dir: object,
        report: dict,
    ) -> None:
        resolved_config = load_config(run_dir / "config.resolved.yaml")
        contract = _distractor_contract(resolved_config)
        base_go = bool(report["claim_gate"].get("go_for_e13b", False))
        report["run_scope"] = (
            "SEQUENCE_BRIDGE_LEARNED_DISTRACTOR_CALIBRATION"
        )
        report["protocol_disposition"] = {
            "original_e13a_immutable": True,
            "e13a_r1_immutable": True,
            "e13a_r1_used_as_repaired_dependency": False,
            "r2_repair_is_prospective": True,
        }
        report["distractor_path_contract"] = contract
        report["summary"]["distractor_path_contract_ok"] = bool(
            contract["passed"]
        )
        report["claim_gate"]["go_for_e13b"] = False
        report["claim_gate"]["go_for_e13b_r1"] = bool(
            base_go and contract["passed"]
        )
        report["claim_gate"]["allowed_claim"] = (
            "Prospective learned-distractor calibration only; no "
            "confirmatory sequence-memory claim."
        )
        report["claim_gate"]["forbidden_claim"] = (
            "Use of E13a-R1 to open the repaired sequence main, or any "
            "natural-language, recurrent-LM, agent, or official-backend claim."
        )
        original_finalize(
            experiment_id=experiment_id,
            artifact_root=artifact_root,
            run_dir=run_dir,
            report=report,
        )

    r1.EXPERIMENT_ID = EXPERIMENT_ID
    r1.DEFAULT_CONFIG = DEFAULT_CONFIG
    r1.SequenceControl = SequenceControlV2
    r1.TransactionalSequenceMemory = TransactionalSequenceMemoryV2
    r1.generate_transactional_sequence_batch = _generate_from_legacy_generator
    r1.train_sequence_memory = train_sequence_memory_v2
    r1.evaluate_sequence_memory = evaluate_sequence_memory_v2
    r1.sequence_parameter_count = sequence_parameter_count_v2
    r1.measure_paired_forward_throughput = (
        measure_paired_forward_throughput_v2
    )
    r1.finalize_run = finalize_r2_run


def main() -> None:
    _install_r2_runtime()
    r1.main()


if __name__ == "__main__":
    main()
