from __future__ import annotations

import json
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Any

import torch

from catena.core.config import load_config
from catena.core.io import file_sha256, write_json, write_jsonl
from catena.data.learned_rank import (
    best_rank_errors,
    make_low_rank_family,
    sample_descriptors,
)
from catena.eval.postcore_metrics import exact_sign_flip
from catena.eval.rank_saturation import (
    canonical_checkpoint_index_sha256,
    classify_pre_saturation_pairs,
    eligible_pre_saturation_monotonic_fraction,
)
from catena.eval.rank_scaling import (
    aggregate_intrinsic_rank_effects_by_seed,
    evaluate_minimum_rank_tracking,
    minimum_sufficient_rank_from_exact_target_recovery,
    oracle_normalized_rank_recovery,
    rank_cell_seed_provenance,
)
from catena.models.operator_controllers import LowRankOperatorController, parameter_count
from catena.training.postcore import evaluate_matrix_controller
from experiments.common import build_parser, finalize_run, initialize_run

EXPERIMENT_ID = "e10b_floor_aware_rank_scaling"
DEFAULT_CONFIG = "configs/e10b_floor_aware_rank_scaling.yaml"
REPO_ROOT = Path(__file__).resolve().parents[1]


def fresh_test_descriptor_seed(
    *,
    source_training_seed: int,
    intrinsic_rank: int,
    seed_offset: int,
    seed_multiplier: int,
) -> int:
    return (
        int(seed_offset)
        + int(seed_multiplier) * int(source_training_seed)
        + int(intrinsic_rank)
    )


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise TypeError(f"expected a JSON object at {path}:{line_number}")
            rows.append(payload)
    return rows


def validate_protocol_lock(config: dict[str, Any]) -> tuple[Path, str]:
    lock_path = (REPO_ROOT / str(config["protocol"]["lock_path"])).resolve()
    if lock_path.is_symlink() or not lock_path.is_file():
        raise RuntimeError(f"E10b protocol lock is missing or unsafe: {lock_path}")
    lock = _read_json(lock_path)
    if (
        lock.get("schema_version") != 1
        or lock.get("experiment_id") != EXPERIMENT_ID
        or lock.get("evaluation_started") is not False
    ):
        raise RuntimeError("E10b protocol lock metadata is invalid")
    files = lock.get("files")
    if not isinstance(files, dict) or not files:
        raise RuntimeError("E10b protocol lock has no frozen file hashes")
    for relative, expected in files.items():
        path = (REPO_ROOT / str(relative)).resolve()
        try:
            path.relative_to(REPO_ROOT)
        except ValueError as exc:
            raise RuntimeError(f"protocol-locked path escapes repository: {path}") from exc
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"protocol-locked file is missing or unsafe: {path}")
        observed = file_sha256(path)
        if observed != str(expected):
            raise RuntimeError(
                f"protocol-locked file hash mismatch: {relative}: "
                f"expected={expected}, observed={observed}"
            )
    return lock_path, file_sha256(lock_path)


def _expected_grid(config: dict[str, Any]) -> set[tuple[int, int, int]]:
    return set(
        product(
            (int(value) for value in config["seeds"]),
            (int(value) for value in config["data"]["intrinsic_ranks"]),
            (int(value) for value in config["model"]["learned_ranks"]),
        )
    )


def validate_frozen_source(
    config: dict[str, Any],
) -> tuple[Path, list[dict[str, Any]], list[dict[str, Any]]]:
    source = config["source_e10"]
    source_dir = Path(str(source["run_dir"])).resolve()
    expected_suffix = (
        Path(str(source["experiment_id"])) / str(source["run_id"])
    )
    if source_dir.is_symlink() or not source_dir.is_dir():
        raise RuntimeError(f"frozen E10 source run is missing or unsafe: {source_dir}")
    if source_dir.parts[-2:] != expected_suffix.parts:
        raise RuntimeError("frozen E10 source path does not match its registered identity")

    pinned_files = {
        "report.json": str(source["report_sha256"]),
        "run_manifest.json": str(source["run_manifest_sha256"]),
        "rank_scaling_metrics.jsonl": str(source["metrics_sha256"]),
        "config.resolved.yaml": str(source["resolved_config_sha256"]),
    }
    for name, expected in pinned_files.items():
        path = source_dir / name
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"missing or unsafe frozen E10 source file: {path}")
        observed = file_sha256(path)
        if observed != expected:
            raise RuntimeError(
                f"frozen E10 source hash mismatch for {name}: "
                f"expected={expected}, observed={observed}"
            )

    report = _read_json(source_dir / "report.json")
    manifest = _read_json(source_dir / "run_manifest.json")
    if (
        report.get("status") != "PASS"
        or report.get("run_scope") != "CONTROLLED_REFERENCE"
        or report.get("claim_gate", {}).get("supported") is not False
        or report.get("claim_gate", {})
        .get("conditions", {})
        .get("learned_rank_error_monotonicity_passed")
        is not False
    ):
        raise RuntimeError("source E10 is not the pinned PASS / NOT_OPENED outcome")
    source_conditions = report["claim_gate"]["conditions"]
    for condition in (
        "rank_tracking_fraction_passed",
        "seedwise_minimum_rank_nondecreasing_passed",
        "seed_level_low_vs_high_rank_sign_flip_passed",
    ):
        if source_conditions.get(condition) is not True:
            raise RuntimeError(f"source E10 expected condition did not pass: {condition}")
    if (
        manifest.get("experiment_id") != str(source["experiment_id"])
        or manifest.get("run_id") != str(source["run_id"])
        or manifest.get("run_mode") != "MAIN"
        or manifest.get("report_sha256") != str(source["report_sha256"])
    ):
        raise RuntimeError("source E10 manifest identity or report linkage is invalid")

    source_config = manifest.get("config")
    if not isinstance(source_config, dict):
        raise RuntimeError("source E10 manifest has no resolved config")
    comparisons = (
        (source_config.get("seeds"), config["seeds"], "seed grid"),
        (
            source_config.get("data", {}).get("dimension"),
            config["data"]["dimension"],
            "state dimension",
        ),
        (
            source_config.get("data", {}).get("descriptor_dim"),
            config["data"]["descriptor_dim"],
            "descriptor dimension",
        ),
        (
            source_config.get("data", {}).get("intrinsic_ranks"),
            config["data"]["intrinsic_ranks"],
            "intrinsic-rank grid",
        ),
        (
            source_config.get("model", {}).get("learned_ranks"),
            config["model"]["learned_ranks"],
            "learned-rank grid",
        ),
        (
            source_config.get("model", {}).get("hidden_dim"),
            config["model"]["hidden_dim"],
            "hidden dimension",
        ),
        (
            source_config.get("claim_gate", {}).get("oracle_normalized_recovery"),
            config["claim_gate"]["oracle_normalized_recovery"],
            "exact-target recovery threshold",
        ),
        (
            source_config.get("claim_gate", {}).get("max_rank_factor"),
            config["claim_gate"]["max_rank_factor"],
            "maximum rank factor",
        ),
        (
            source_config.get("claim_gate", {}).get("minimum_rank_match_fraction"),
            config["claim_gate"]["minimum_rank_match_fraction"],
            "minimum-rank match fraction",
        ),
    )
    for observed, expected, label in comparisons:
        if observed != expected:
            raise RuntimeError(
                f"E10b changed the registered {label}: "
                f"source={observed!r}, e10b={expected!r}"
            )

    metric_path = source_dir / "rank_scaling_metrics.jsonl"
    rows = _read_jsonl(metric_path)
    if len(rows) != int(source["checkpoint_count"]):
        raise RuntimeError("frozen E10 metric row count is not 240")
    observed_keys = [
        (
            int(row["seed"]),
            int(row["intrinsic_rank"]),
            int(row["learned_rank"]),
        )
        for row in rows
    ]
    if len(set(observed_keys)) != len(observed_keys):
        raise RuntimeError("frozen E10 metrics contain duplicate checkpoint identities")
    if set(observed_keys) != _expected_grid(config):
        raise RuntimeError("frozen E10 metrics do not contain the exact registered grid")
    index_sha = canonical_checkpoint_index_sha256(rows, source_run_dir=source_dir)
    if index_sha != str(source["checkpoint_index_sha256"]):
        raise RuntimeError(
            "frozen E10 checkpoint index digest mismatch: "
            f"expected={source['checkpoint_index_sha256']}, observed={index_sha}"
        )

    verification_rows: list[dict[str, Any]] = []
    for row in rows:
        seed = int(row["seed"])
        intrinsic_rank = int(row["intrinsic_rank"])
        learned_rank = int(row["learned_rank"])
        expected_name = (
            f"seed{seed}_intrinsic{intrinsic_rank}_learned{learned_rank}.pt"
        )
        checkpoint = Path(str(row["checkpoint"])).resolve()
        if (
            checkpoint.parent != (source_dir / "checkpoints").resolve()
            or checkpoint.name != expected_name
            or checkpoint.is_symlink()
            or not checkpoint.is_file()
        ):
            raise RuntimeError(f"invalid frozen checkpoint path: {checkpoint}")
        observed_sha = file_sha256(checkpoint)
        expected_sha = str(row["checkpoint_sha256"])
        if observed_sha != expected_sha:
            raise RuntimeError(
                f"frozen checkpoint hash mismatch: {checkpoint}: "
                f"expected={expected_sha}, observed={observed_sha}"
            )
        seed_provenance = rank_cell_seed_provenance(seed, intrinsic_rank)
        if int(row["family_seed"]) != seed_provenance["family_seed"]:
            raise RuntimeError("frozen source row family seed is inconsistent")
        verification_rows.append(
            {
                "seed": seed,
                "intrinsic_rank": intrinsic_rank,
                "learned_rank": learned_rank,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": observed_sha,
                "hash_verified": True,
            }
        )
    return source_dir, rows, verification_rows


def _load_frozen_model(
    *,
    row: dict[str, Any],
    descriptor_dim: int,
    dimension: int,
    hidden_dim: int,
) -> LowRankOperatorController:
    seed = int(row["seed"])
    intrinsic_rank = int(row["intrinsic_rank"])
    learned_rank = int(row["learned_rank"])
    checkpoint = Path(str(row["checkpoint"]))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or int(payload.get("seed", -1)) != seed
        or int(payload.get("intrinsic_rank", -1)) != intrinsic_rank
        or int(payload.get("learned_rank", -1)) != learned_rank
        or not isinstance(payload.get("model"), dict)
    ):
        raise RuntimeError(f"checkpoint payload identity mismatch: {checkpoint}")
    model = LowRankOperatorController(
        descriptor_dim=descriptor_dim,
        dimension=dimension,
        rank=learned_rank,
        hidden_dim=hidden_dim,
    )
    model.load_state_dict(payload["model"], strict=True)
    return model


def main() -> None:
    parser = build_parser(EXPERIMENT_ID, DEFAULT_CONFIG)
    args = parser.parse_args()
    prospective_config = load_config(args.config)
    lock_path, lock_sha256 = validate_protocol_lock(prospective_config)
    source_dir, source_rows, verification_rows = validate_frozen_source(
        prospective_config
    )

    config, run_dir, device = initialize_run(
        experiment_id=EXPERIMENT_ID,
        config_path=args.config,
        artifact_root=args.artifact_root,
        device_request=args.device,
        run_mode="DRY_RUN" if args.dry_run else "MAIN",
    )
    seeds = [int(value) for value in config["seeds"]]
    intrinsic_ranks = [int(value) for value in config["data"]["intrinsic_ranks"]]
    learned_ranks = [int(value) for value in config["model"]["learned_ranks"]]
    test_count = int(config["data"]["fresh_test_count"])
    if args.dry_run:
        seeds = seeds[:1]
        intrinsic_ranks = intrinsic_ranks[:2]
        learned_ranks = learned_ranks[:3]
        test_count = min(test_count, 64)

    dimension = int(config["data"]["dimension"])
    descriptor_dim = int(config["data"]["descriptor_dim"])
    hidden_dim = int(config["model"]["hidden_dim"])
    batch_size = int(config["evaluation"]["batch_size"])
    namespace = config["data"]["fresh_test_namespace"]
    recovery_threshold = float(config["claim_gate"]["oracle_normalized_recovery"])
    source_by_key = {
        (
            int(row["seed"]),
            int(row["intrinsic_rank"]),
            int(row["learned_rank"]),
        ): row
        for row in source_rows
    }

    rows: list[dict[str, Any]] = []
    by_seed_intrinsic: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    payload_verified_count = 0
    for seed in seeds:
        for intrinsic_rank in intrinsic_ranks:
            provenance = rank_cell_seed_provenance(seed, intrinsic_rank)
            fresh_seed = fresh_test_descriptor_seed(
                source_training_seed=seed,
                intrinsic_rank=intrinsic_rank,
                seed_offset=int(namespace["seed_offset"]),
                seed_multiplier=int(namespace["seed_multiplier"]),
            )
            source_test_seed = provenance["test_descriptor_seed"]
            if fresh_seed == source_test_seed:
                raise RuntimeError("E10b fresh descriptor namespace collides with E10")
            family = make_low_rank_family(
                dimension=dimension,
                descriptor_dim=descriptor_dim,
                intrinsic_rank=intrinsic_rank,
                seed=provenance["family_seed"],
            )
            descriptor = sample_descriptors(
                count=test_count,
                descriptor_dim=descriptor_dim,
                seed=fresh_seed,
            )
            target = family.operator(descriptor)
            baseline_error = float(target.square().mean())
            oracle_ranks = list(dict.fromkeys([*learned_ranks, intrinsic_rank]))
            oracle_by_rank = best_rank_errors(target, oracle_ranks)
            exact_target_oracle_error = float(oracle_by_rank[intrinsic_rank].mean())

            for learned_rank in learned_ranks:
                source_row = source_by_key[(seed, intrinsic_rank, learned_rank)]
                model = _load_frozen_model(
                    row=source_row,
                    descriptor_dim=descriptor_dim,
                    dimension=dimension,
                    hidden_dim=hidden_dim,
                )
                payload_verified_count += 1
                with torch.inference_mode():
                    test_error, per_example = evaluate_matrix_controller(
                        model=model,
                        descriptors=descriptor,
                        targets=target,
                        device=device,
                        batch_size=batch_size,
                    )
                oracle_error = float(oracle_by_rank[learned_rank].mean())
                reachable_floor_recovery = oracle_normalized_rank_recovery(
                    baseline_error=baseline_error,
                    model_error=test_error,
                    oracle_error=oracle_error,
                )
                exact_target_recovery = oracle_normalized_rank_recovery(
                    baseline_error=baseline_error,
                    model_error=test_error,
                    oracle_error=exact_target_oracle_error,
                )
                row: dict[str, Any] = {
                    "seed": seed,
                    "intrinsic_rank": intrinsic_rank,
                    "learned_rank": learned_rank,
                    "test_error": test_error,
                    "oracle_error": oracle_error,
                    "exact_target_oracle_error": exact_target_oracle_error,
                    "baseline_error": baseline_error,
                    "reachable_floor_recovery": reachable_floor_recovery,
                    "exact_target_recovery": exact_target_recovery,
                    "excess_over_oracle": test_error - oracle_error,
                    "per_example_error_std": float(
                        per_example.std(unbiased=False)
                    ),
                    "parameter_count": parameter_count(model),
                    "family_seed": provenance["family_seed"],
                    "fresh_test_namespace": str(namespace["id"]),
                    "fresh_test_descriptor_seed": fresh_seed,
                    "source_test_descriptor_seed_not_reused": source_test_seed,
                    "source_checkpoint": str(source_row["checkpoint"]),
                    "source_checkpoint_sha256": str(
                        source_row["checkpoint_sha256"]
                    ),
                    "source_checkpoint_hash_verified": True,
                    "source_checkpoint_payload_identity_verified": True,
                    "checkpoint_retrained": False,
                    "original_test_outcome_reused": False,
                    "oracle_method": (
                        "single_batched_svdvals_per_fresh_seed_intrinsic_cell"
                    ),
                }
                rows.append(row)
                by_seed_intrinsic[(seed, intrinsic_rank)].append(row)
                del model

    pair_rows: list[dict[str, Any]] = []
    cell_effects: dict[tuple[int, int], float] = {}
    minimum_qualifying_ranks: dict[tuple[int, int], int | None] = {}
    exact_intrinsic_matches: list[bool] = []
    for (seed, intrinsic_rank), group in sorted(by_seed_intrinsic.items()):
        ordered = sorted(group, key=lambda item: int(item["learned_rank"]))
        classified = classify_pre_saturation_pairs(
            ordered,
            recovery_threshold=recovery_threshold,
        )
        for pair in classified:
            pair_rows.append(
                {
                    "seed": seed,
                    "intrinsic_rank": intrinsic_rank,
                    "fresh_test_descriptor_seed": int(
                        ordered[0]["fresh_test_descriptor_seed"]
                    ),
                    **pair,
                }
            )
        recoveries = {
            int(item["learned_rank"]): float(item["exact_target_recovery"])
            for item in ordered
        }
        minimum = minimum_sufficient_rank_from_exact_target_recovery(
            recoveries,
            threshold=recovery_threshold,
        )
        minimum_qualifying_ranks[(seed, intrinsic_rank)] = minimum
        exact_intrinsic_matches.append(minimum == intrinsic_rank)
        cell_effects[(seed, intrinsic_rank)] = float(
            ordered[0]["test_error"]
        ) - float(ordered[-1]["test_error"])

    (
        pre_saturation_fraction,
        pre_saturation_passed,
        pre_saturation_eligible,
    ) = eligible_pre_saturation_monotonic_fraction(pair_rows)
    saturated_pair_count = sum(
        not bool(item["eligible_pre_saturation"]) for item in pair_rows
    )
    rank_tracking_cell_rows, rank_tracking_seed_rows = (
        evaluate_minimum_rank_tracking(
            minimum_qualifying_ranks,
            seeds=seeds,
            intrinsic_ranks=intrinsic_ranks,
            max_rank_factor=float(config["claim_gate"]["max_rank_factor"]),
            max_available_rank=max(learned_ranks),
        )
    )
    rank_match_fraction = sum(
        bool(item["rank_tracking_matched"]) for item in rank_tracking_cell_rows
    ) / len(rank_tracking_cell_rows)
    seedwise_nondecreasing_fraction = sum(
        bool(item["minimum_qualifying_rank_nondecreasing"])
        for item in rank_tracking_seed_rows
    ) / len(rank_tracking_seed_rows)
    seed_effect_rows = aggregate_intrinsic_rank_effects_by_seed(
        cell_effects,
        seeds=seeds,
        intrinsic_ranks=intrinsic_ranks,
    )
    seed_effects = [
        float(item["mean_low_vs_high_rank_gain"]) for item in seed_effect_rows
    ]
    sign_flip_p = exact_sign_flip(seed_effects, alternative="greater")

    full_grid_evaluated = (
        len(rows) == int(config["source_e10"]["checkpoint_count"])
        and len(seeds) == len(config["seeds"])
        and len(intrinsic_ranks) == len(config["data"]["intrinsic_ranks"])
        and len(learned_ranks) == len(config["model"]["learned_ranks"])
    )
    conditions = {
        "all_240_checkpoint_hashes_verified": (
            len(verification_rows) == int(config["source_e10"]["checkpoint_count"])
            and all(bool(item["hash_verified"]) for item in verification_rows)
        ),
        "full_fresh_grid_evaluated": full_grid_evaluated,
        "eligible_pre_saturation_pair_count_positive": (
            pre_saturation_eligible > 0
        ),
        "eligible_pre_saturation_monotonicity_passed": (
            pre_saturation_fraction
            >= float(
                config["claim_gate"][
                    "eligible_pre_saturation_monotonic_fraction"
                ]
            )
        ),
        "exact_target_minimum_rank_tracking_passed": (
            rank_match_fraction
            >= float(config["claim_gate"]["minimum_rank_match_fraction"])
        ),
        "seedwise_minimum_rank_nondecreasing_passed": (
            seedwise_nondecreasing_fraction == 1.0
        ),
        "seed_level_low_vs_high_rank_sign_flip_passed": (
            len(seed_effects) == 8
            and sign_flip_p <= float(config["statistics"]["alpha"])
        ),
    }
    supported = (not args.dry_run) and all(conditions.values())
    report = {
        "status": "PASS",
        "run_mode": "DRY_RUN" if args.dry_run else "MAIN",
        "evidence_tier": "CONTROLLED_REFERENCE",
        "scientific_evidence": False,
        "original_e10_disposition": {
            "run_id": str(config["source_e10"]["run_id"]),
            "execution_status": "PASS",
            "claim_status": "NOT_OPENED",
            "reason": str(config["protocol"]["original_e10_reason"]),
            "unchanged": True,
        },
        "summary": {
            "fresh_evaluation_rows": len(rows),
            "frozen_checkpoint_hashes_verified": len(verification_rows),
            "frozen_checkpoint_payloads_evaluated": payload_verified_count,
            "eligible_pre_saturation_pairs": pre_saturation_eligible,
            "eligible_pre_saturation_pairs_passed": pre_saturation_passed,
            "eligible_pre_saturation_monotonic_fraction": (
                pre_saturation_fraction
            ),
            "saturated_excluded_pairs": saturated_pair_count,
            "rank_match_fraction": rank_match_fraction,
            "minimum_rank_equals_intrinsic_rank_fraction": (
                sum(exact_intrinsic_matches) / len(exact_intrinsic_matches)
            ),
            "seedwise_minimum_rank_nondecreasing_fraction": (
                seedwise_nondecreasing_fraction
            ),
            "low_vs_high_rank_sign_flip_p": sign_flip_p,
            "low_vs_high_rank_seed_mean_gain": (
                sum(seed_effects) / len(seed_effects)
            ),
            "statistical_unit": str(config["statistics"]["statistical_unit"]),
            "statistical_unit_count": len(seed_effects),
        },
        "protocol": {
            "amendment_id": str(config["protocol"]["amendment_id"]),
            "protocol_lock_path": str(lock_path),
            "protocol_lock_sha256": lock_sha256,
            "checkpoint_only_evaluation": True,
            "checkpoints_retrained": False,
            "fresh_descriptor_test_namespace": namespace,
            "original_test_rows_reused": False,
            "monotonicity_estimand": (
                "adjacent rank error is non-increasing exactly when the "
                "lower-rank fresh-set exact-target recovery is below 0.95"
            ),
            "saturated_pair_disposition": (
                "retained in pair diagnostics and excluded from the "
                "pre-saturation monotonicity denominator"
            ),
            "new_error_tolerance_introduced": False,
        },
        "source_e10": {
            **config["source_e10"],
            "checkpoint_index_verified": True,
            "source_report_claim_open": False,
        },
        "claim_gate": {
            "supported": supported,
            "conditions": conditions,
            "controlled_geometry_claim_eligible": supported,
            "official_backend_claim_eligible": False,
            "pretrained_language_model_claim_eligible": False,
            "allowed_claim": (
                "On the registered smooth synthetic operator families and a "
                "fresh descriptor test namespace, exact-target minimum learned "
                "rank tracks demand intrinsic rank and every identifiable "
                "pre-saturation adjacent-rank error is non-increasing."
            ),
            "forbidden_claim": (
                "This result establishes a universal rank law for official "
                "backends, pretrained language models, or natural-language "
                "transactions."
            ),
        },
    }
    write_jsonl(run_dir / "e10b_fresh_rank_metrics.jsonl", rows)
    write_jsonl(run_dir / "e10b_pre_saturation_pairs.jsonl", pair_rows)
    write_jsonl(
        run_dir / "e10b_rank_tracking_cells.jsonl",
        rank_tracking_cell_rows,
    )
    write_jsonl(
        run_dir / "e10b_rank_tracking_seeds.jsonl",
        rank_tracking_seed_rows,
    )
    write_jsonl(run_dir / "e10b_seed_effects.jsonl", seed_effect_rows)
    write_jsonl(
        run_dir / "source_checkpoint_verification.jsonl",
        verification_rows,
    )
    write_json(
        run_dir / "source_freeze.json",
        {
            "source_run_dir": str(source_dir),
            "source_report_sha256": str(config["source_e10"]["report_sha256"]),
            "source_checkpoint_count": len(verification_rows),
            "source_checkpoint_index_sha256": str(
                config["source_e10"]["checkpoint_index_sha256"]
            ),
            "all_checkpoint_hashes_verified": True,
            "original_test_outcomes_reused": False,
        },
    )
    finalize_run(
        experiment_id=EXPERIMENT_ID,
        artifact_root=args.artifact_root,
        run_dir=run_dir,
        report=report,
    )
    print(
        f"[{EXPERIMENT_ID}] "
        f"{'DRY_RUN' if args.dry_run else 'PASS'}: {run_dir}"
    )


if __name__ == "__main__":
    main()
