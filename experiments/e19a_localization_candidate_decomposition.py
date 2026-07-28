from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from catena.core.config import load_config
from catena.core.io import file_sha256, write_jsonl
from catena.data.localization_candidate import (
    LocalizationCandidateCondition,
    address_codebook_sha256,
    make_address_codebook,
)
from catena.models.localization_candidate import (
    LocalizationCandidateFreedom,
    MatchedLocalizationCandidateController,
    localization_candidate_parameter_count,
)
from catena.training.localization_candidate import (
    evaluate_localization_candidate_controller,
    train_localization_candidate_controller,
)
from experiments.common import finalize_run, initialize_run

EXPERIMENT_ID = "e19a_localization_candidate_decomposition"
DEFAULT_CONFIG = "configs/e19a_localization_candidate_decomposition.yaml"
REPO_ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = REPO_ROOT / "docs/E19_LOCALIZATION_CANDIDATE_LOCK.json"


def _state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(b"\0")
        digest.update(repr(tuple(tensor.shape)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _read_json_object(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def validate_e19_protocol_lock(config_path: str | Path) -> str:
    if not LOCK_PATH.is_file() or LOCK_PATH.is_symlink():
        raise RuntimeError("E19 prospective protocol lock is missing")
    lock = _read_json_object(LOCK_PATH)
    if (
        lock.get("experiment_family") != "E19"
        or lock.get("main_evaluation_started") is not False
        or lock.get("gpu_main_executed") is not False
    ):
        raise RuntimeError("E19 lock does not certify a pre-main protocol")
    files = lock.get("files")
    if not isinstance(files, dict) or not files:
        raise RuntimeError("E19 lock lacks a non-empty file hash map")
    for relative, expected in files.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise RuntimeError("E19 lock contains an invalid file record")
        path = (REPO_ROOT / relative).resolve()
        try:
            path.relative_to(REPO_ROOT)
        except ValueError as error:
            raise RuntimeError("E19 lock path escapes repository root") from error
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"E19 locked file missing or unsafe: {path}")
        if file_sha256(path) != expected:
            raise RuntimeError(f"E19 locked file changed: {path}")
    resolved_config = Path(config_path).resolve()
    expected_config = files.get("configs/e19a_localization_candidate_decomposition.yaml")
    if expected_config is None or file_sha256(resolved_config) != expected_config:
        raise RuntimeError("E19a config does not match the prospective lock")
    return file_sha256(LOCK_PATH)


def build_local_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=EXPERIMENT_ID)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_local_parser().parse_args()
    pre_config = load_config(args.config)
    if pre_config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("E19a config experiment_id mismatch")
    registered_seeds = [int(value) for value in pre_config["seeds"]]
    if not args.dry_run and args.seed is None:
        raise ValueError("E19a MAIN requires one explicit --seed")
    seed = registered_seeds[0] if args.seed is None else int(args.seed)
    if seed not in registered_seeds:
        raise ValueError(f"Unregistered E19a seed: {seed}")
    protocol_lock_sha256 = validate_e19_protocol_lock(args.config)

    config, run_dir, device = initialize_run(
        experiment_id=EXPERIMENT_ID,
        config_path=args.config,
        artifact_root=args.artifact_root,
        device_request=args.device,
        run_mode="DRY_RUN" if args.dry_run else "MAIN",
    )
    conditions = [
        LocalizationCandidateCondition(value)
        for value in config["conditions"]
    ]
    variants = [
        LocalizationCandidateFreedom(value)
        for value in config["model"]["variants"]
    ]
    slots = int(config["data"]["slots"])
    value_dim = int(config["data"]["value_dim"])
    code_dim = int(config["data"]["address_code_dim"])
    descriptor_dim = 2 * code_dim + value_dim
    steps = int(config["training"]["steps"])
    train_batch_size = int(config["training"]["batch_size"])
    episodes = int(config["evaluation"]["episodes_per_condition"])
    eval_batch_size = int(config["evaluation"]["batch_size"])
    if args.dry_run:
        steps = min(steps, 4)
        train_batch_size = min(train_batch_size, 8)
        episodes = min(episodes, 32)
        eval_batch_size = min(eval_batch_size, 16)

    codebook = make_address_codebook(
        slots=slots,
        code_dim=code_dim,
        seed=int(config["data"]["address_schema_seed"]),
    )
    rows: list[dict[str, float | int | str]] = []
    initialization_hashes: dict[str, str] = {}
    parameter_counts: dict[str, int] = {}
    checkpoint_hashes: dict[str, str] = {}

    for variant in variants:
        torch.manual_seed(10_000 + seed)
        model = MatchedLocalizationCandidateController(
            freedom=variant,
            descriptor_dim=descriptor_dim,
            slots=slots,
            value_dim=value_dim,
            hidden_dim=int(config["model"]["hidden_dim"]),
            address_temperature=float(
                config["model"]["address_temperature"]
            ),
        )
        initialization_hashes[variant.value] = _state_dict_sha256(
            model.state_dict()
        )
        parameter_counts[variant.value] = localization_candidate_parameter_count(
            model
        )
        trace = train_localization_candidate_controller(
            model=model,
            conditions=conditions,
            steps=steps,
            batch_size=train_batch_size,
            slots=slots,
            value_dim=value_dim,
            state_scale=float(config["data"]["state_scale"]),
            address_codebook=codebook,
            learning_rate=float(config["training"]["learning_rate"]),
            address_loss_weight=float(
                config["training"]["address_loss_weight"]
            ),
            candidate_loss_weight=float(
                config["training"]["candidate_loss_weight"]
            ),
            retention_weight=float(config["training"]["retention_weight"]),
            device=device,
            seed=20_000 + seed,
        )
        checkpoint = (
            run_dir / "checkpoints" / f"seed{seed}_{variant.value}.pt"
        )
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "model_class": "MatchedLocalizationCandidateController",
                "variant": variant.value,
                "seed": seed,
                "config": config,
                "protocol_lock_sha256": protocol_lock_sha256,
            },
            checkpoint,
        )
        checkpoint_sha256 = file_sha256(checkpoint)
        checkpoint_hashes[variant.value] = checkpoint_sha256
        for condition in conditions:
            evaluation = evaluate_localization_candidate_controller(
                model=model,
                condition=condition,
                episodes=episodes,
                batch_size=eval_batch_size,
                slots=slots,
                value_dim=value_dim,
                state_scale=float(config["data"]["state_scale"]),
                address_codebook=codebook,
                device=device,
                seed=30_000 + seed,
            )
            rows.append(
                {
                    "seed": seed,
                    "variant": variant.value,
                    "condition": condition.value,
                    "parameter_count": parameter_counts[variant.value],
                    "initialization_sha256": initialization_hashes[
                        variant.value
                    ],
                    "train_final_loss": trace.final_loss,
                    "train_best_loss": trace.best_loss,
                    "evaluation_seed": 30_000 + seed,
                    "episodes": episodes,
                    "checkpoint": str(checkpoint.resolve()),
                    "checkpoint_sha256": checkpoint_sha256,
                    **evaluation,
                }
            )

    full_error = {
        str(row["condition"]): float(row["affected_mse"])
        for row in rows
        if row["variant"] == LocalizationCandidateFreedom.FULL.value
    }
    if set(full_error) != {condition.value for condition in conditions}:
        raise RuntimeError("E19a full-controller condition grid incomplete")
    for row in rows:
        row["architecture_extra_error"] = (
            float(row["affected_mse"]) - full_error[str(row["condition"])]
        )

    paired_initialization = len(set(initialization_hashes.values())) == 1
    matched_parameter_count = len(set(parameter_counts.values())) == 1
    if not paired_initialization or not matched_parameter_count:
        raise RuntimeError("E19a paired maximal-surface contract failed")
    metrics_path = run_dir / "localization_candidate_metrics.jsonl"
    write_jsonl(metrics_path, rows)
    report = {
        "status": "DRY_RUN" if args.dry_run else "PASS",
        "run_mode": "DRY_RUN" if args.dry_run else "MAIN",
        "run_scope": "CONTROLLED_LEARNED_LOCALIZATION_STATE_READ",
        "evidence_tier": "CONTROLLED_REFERENCE",
        "scientific_evidence": False,
        "seed": seed,
        "rows": len(rows),
        "paired_contract": {
            "same_maximal_surface": True,
            "paired_initialization": paired_initialization,
            "initialization_hashes": initialization_hashes,
            "matched_parameter_count": matched_parameter_count,
            "parameter_counts": parameter_counts,
            "paired_training_seed": 20_000 + seed,
            "paired_evaluation_seed": 30_000 + seed,
            "address_codebook_sha256": address_codebook_sha256(codebook),
        },
        "artifacts": {
            "metrics_path": str(metrics_path.resolve()),
            "metrics_sha256": file_sha256(metrics_path),
            "checkpoint_hashes": checkpoint_hashes,
        },
        "protocol": {
            "lock_path": str(LOCK_PATH),
            "lock_sha256": protocol_lock_sha256,
            "source_config_sha256": file_sha256(args.config),
        },
        "claim_gate": {
            "status": (
                "NOT_EVALUATED_DRY_RUN"
                if args.dry_run
                else "PENDING_AGGREGATE"
            ),
            "allowed_claim": (
                "Per-seed controlled localization/candidate evidence only "
                "after the prospectively locked E19b five-seed aggregate."
            ),
            "forbidden_claim": (
                "Semantic, natural-language, pretrained-model, agent, or "
                "official-backend localization transfer."
            ),
        },
    }
    finalize_run(
        experiment_id=EXPERIMENT_ID,
        artifact_root=args.artifact_root,
        run_dir=run_dir,
        report=report,
    )
    print(f"[{EXPERIMENT_ID}] {report['status']}: {run_dir}")


if __name__ == "__main__":
    main()
