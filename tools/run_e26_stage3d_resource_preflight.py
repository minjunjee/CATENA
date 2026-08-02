#!/usr/bin/env python3
"""Measure Stage-3D GO candidates under the one locked physical layout.

This is a non-evidence resource preflight.  It deliberately does not search
microbatch sizes, retry an OOM with a different layout, audit counterfactual
accumulation layouts, or start scientific E26a.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import yaml

from catena.core.provenance_v61 import (
    SHA256_PATTERN,
    read_json_object_strict,
    sha256_canonical_json,
    sha256_file,
    write_json_strict,
)
from catena.lm.artifacts import ArtifactRun
from catena.lm.audit_contract import e26_execution_source_inventory
from catena.lm.backend_lock import (
    cuda_hardware_inventory,
    observed_single_visible_cuda_device,
)
from catena.lm.e26a_executor import _candidate_config, _measure_variant
from catena.lm.e26a_gate import (
    CandidateMeasurement,
    E26AGateBlocked,
    ResourcePolicy,
    project_candidate_resources,
    select_candidate,
)
from catena.lm.general_corpus import TokenMemmap
from catena.lm.hashing import parameter_signature_hash
from catena.lm.model import assert_matched_models, build_paired_models
from catena.lm.stage3d_fixed_layout import (
    validate_stage3d_admissibility_receipt,
    validate_stage3d_protocol_lock,
    validate_stage3d_resource_preflight_receipt,
)
from catena.lm.tokenizer import ExternalScientificTokenizer
from catena.lm.trainer import compare_optimizer_signatures, make_optimizer

SCHEMA_VERSION = "catena-v8.1"
STAGE3D_RECEIPT_VERSION = "catena-e26-stage3d-fixed-layout-receipt-v1"
RESOURCE_RECEIPT_VERSION = "catena-e26-stage3d-resource-preflight-v1"
STAGE3D_MANIFEST_TYPE = "E26_STAGE3D_FIXED_LAYOUT_RECEIPT"
STAGE3D_GO = "STAGE3D_GO_FIXED_LAYOUT_BF16_ADMISSIBLE"
RESOURCE_MANIFEST_TYPE = "E26_STAGE3D_RESOURCE_PREFLIGHT_RECEIPT"
TARGET_GLOBAL_INPUT_TOKENS = 65_536
FIXED_MICROBATCH_SEQUENCES = 1
TOKEN_BUDGETS = (250_000_000, 375_000_000, 500_000_000)
MAX_MAIN_WALL_CLOCK_HOURS = 168.0
SAFETY_TIME_MULTIPLIER = 1.25
MAX_MAIN_CHECKPOINT_STORAGE_GIB = 100.0
DEADLINE_REFERENCE_HOURS = 240.0
DEADLINE_FRACTION_MAX = 0.70
MAIN_RUNS = 10
GPU_LANES = 4
SAVE_EVERY_TOKENS = 25_000_000
VARIANTS = ("projected_tied_delta_lm", "dual_delta_lm")
CANONICAL_ARTIFACT_ROOT = Path("/data/minjun_dev/CATENA/artifacts")
RESOURCE_EXPERIMENT = "e26_stage3d_resource_preflight"
RESOURCE_NAMESPACE = CANONICAL_ARTIFACT_ROOT / RESOURCE_EXPERIMENT
RESOURCE_WORKER_PREFIX = "e26_stage3d_resource_worker_"

_INPUT_NAMES = (
    "config",
    "stage3d_receipt",
    "tokenizer_manifest",
    "corpus_manifest",
)


def _yaml_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return payload


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments],
        cwd=repo,
        text=True,
        stderr=subprocess.PIPE,
    ).strip()


def _regular_file(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{label} must be a regular file: {resolved}")
    return resolved


def _require_canonical_hash(payload: Mapping[str, Any], *, label: str) -> str:
    observed = payload.get("receipt_sha256")
    if not isinstance(observed, str) or not SHA256_PATTERN.fullmatch(observed):
        raise ValueError(f"{label} lacks a canonical receipt SHA-256")
    unhashed = dict(payload)
    unhashed.pop("receipt_sha256")
    if observed != sha256_canonical_json(unhashed):
        raise ValueError(f"{label} canonical SHA-256 mismatch")
    return observed


def _candidate_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = config.get("model_candidates")
    if not isinstance(raw, list) or not raw:
        raise ValueError("E26 config lacks model candidates")
    rows: list[dict[str, Any]] = []
    for row in raw:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise ValueError("Malformed E26 model candidate")
        rows.append(row)
    return rows


def _expected_layouts(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    layouts: list[dict[str, Any]] = []
    for candidate in _candidate_rows(config):
        context_length = int(candidate["context_length"])
        if TARGET_GLOBAL_INPUT_TOKENS % context_length:
            raise ValueError(
                f"{candidate['id']}: context does not divide fixed global input tokens"
            )
        global_sequences = TARGET_GLOBAL_INPUT_TOKENS // context_length
        layouts.append(
            {
                "candidate_id": str(candidate["id"]),
                "context_length": context_length,
                "microbatch_sequences": FIXED_MICROBATCH_SEQUENCES,
                "target_global_input_tokens": TARGET_GLOBAL_INPUT_TOKENS,
                "global_batch_sequences": global_sequences,
                "accumulation_steps": global_sequences,
            }
        )
    return layouts


def _hash_binding(payload: Mapping[str, Any], key: str) -> str:
    for container_name in ("input_hashes", "locked_hashes"):
        container = payload.get(container_name)
        if isinstance(container, Mapping):
            observed = container.get(key)
            if isinstance(observed, str) and SHA256_PATTERN.fullmatch(observed):
                return observed
    protocol_binding = payload.get("protocol_lock")
    if not isinstance(protocol_binding, Mapping):
        raise ValueError(f"Stage-3D receipt lacks required data binding: {key}")
    protocol_path = _regular_file(str(protocol_binding.get("path")), "Stage-3D protocol")
    if sha256_file(protocol_path) != protocol_binding.get("sha256"):
        raise ValueError("Stage-3D protocol byte hash changed")
    protocol = _yaml_mapping(protocol_path)
    protocol_unhashed = dict(protocol)
    observed_protocol_hash = protocol_unhashed.pop("protocol_sha256", None)
    if observed_protocol_hash != protocol_binding.get(
        "protocol_sha256"
    ) or observed_protocol_hash != sha256_canonical_json(protocol_unhashed):
        raise ValueError("Stage-3D protocol canonical hash changed")
    stage3c = protocol.get("stage3c")
    if not isinstance(stage3c, Mapping):
        raise ValueError("Stage-3D protocol lacks Stage-3C binding")
    stage3c_binding = stage3c.get("protocol")
    if not isinstance(stage3c_binding, Mapping):
        raise ValueError("Stage-3D protocol lacks Stage-3C protocol binding")
    stage3c_path = _regular_file(str(stage3c_binding.get("path")), "Stage-3C prospective protocol")
    if sha256_file(stage3c_path) != stage3c_binding.get("sha256"):
        raise ValueError("Stage-3C prospective protocol byte hash changed")
    stage3c_protocol = _yaml_mapping(stage3c_path)
    execution_inputs = stage3c_protocol.get("execution_inputs")
    if not isinstance(execution_inputs, Mapping):
        raise ValueError("Stage-3C prospective protocol lacks execution inputs")
    observed = execution_inputs.get(key)
    if not isinstance(observed, str) or not SHA256_PATTERN.fullmatch(observed):
        raise ValueError(f"Stage-3C prospective protocol lacks data binding: {key}")
    return observed


def _bound_e26_inputs(payload: Mapping[str, Any]) -> dict[str, Path]:
    """Resolve the exact E26a config/tokenizer/corpus bound by Stage-3C.

    Resource measurement is downstream of Stage-3D, but the physical model and
    token stream remain the ones fixed by the immutable Stage-3C protocol.  A
    same-byte substitute at another path is deliberately rejected: both the
    registered path and its byte hash are part of this resource receipt.
    """

    stage3d_binding = payload.get("protocol_lock")
    if not isinstance(stage3d_binding, Mapping):
        raise ValueError("Stage-3D receipt lacks protocol lock binding")
    stage3d_path = _regular_file(
        str(stage3d_binding.get("path")), "Stage-3D protocol lock"
    )
    if sha256_file(stage3d_path) != stage3d_binding.get("sha256"):
        raise ValueError("Stage-3D protocol lock byte hash changed")
    stage3d_protocol = validate_stage3d_protocol_lock(_yaml_mapping(stage3d_path))
    stage3c = stage3d_protocol.get("stage3c")
    if not isinstance(stage3c, Mapping) or not isinstance(stage3c.get("protocol"), Mapping):
        raise ValueError("Stage-3D protocol lacks Stage-3C protocol binding")
    stage3c_binding = stage3c["protocol"]
    stage3c_path = _regular_file(
        str(stage3c_binding.get("path")), "Stage-3C protocol lock"
    )
    if sha256_file(stage3c_path) != stage3c_binding.get("sha256"):
        raise ValueError("Stage-3C protocol lock byte hash changed")
    stage3c_protocol = _yaml_mapping(stage3c_path)
    raw_paths = stage3c_protocol.get("execution_input_paths")
    raw_hashes = stage3c_protocol.get("execution_inputs")
    if not isinstance(raw_paths, Mapping) or not isinstance(raw_hashes, Mapping):
        raise ValueError("Stage-3C protocol lacks execution input bindings")
    result: dict[str, Path] = {}
    for name in ("config", "tokenizer_manifest", "corpus_manifest"):
        raw_path = raw_paths.get(name)
        expected_hash = raw_hashes.get(f"{name}_sha256")
        if not isinstance(raw_path, str) or raw_path.startswith("BUNDLE_RELATIVE:"):
            raise ValueError(f"Stage-3C protocol has no absolute {name} path")
        if not isinstance(expected_hash, str) or not SHA256_PATTERN.fullmatch(expected_hash):
            raise ValueError(f"Stage-3C protocol has no valid {name} SHA-256")
        path = _regular_file(raw_path, f"Stage-3C bound {name}")
        if sha256_file(path) != expected_hash:
            raise ValueError(f"Stage-3C bound {name} byte hash changed")
        result[name] = path
    return result


def _require_exact_bound_input_paths(
    paths: Mapping[str, Path], *, stage3d_receipt: Mapping[str, Any]
) -> None:
    bound = _bound_e26_inputs(stage3d_receipt)
    for name in ("config", "tokenizer_manifest", "corpus_manifest"):
        if paths[name] != bound[name]:
            raise ValueError(
                f"Resource {name} path differs from the exact Stage-3C binding: "
                f"{paths[name]} != {bound[name]}"
            )


def validate_stage3d_go_receipt(
    payload: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    config_sha256: str | None = None,
    tokenizer_manifest_sha256: str | None = None,
    corpus_manifest_sha256: str | None = None,
    verify_canonical_contract: bool = True,
) -> dict[str, Any]:
    """Validate that resource measurement is authorized by an exact Stage-3D GO."""

    _require_canonical_hash(payload, label="Stage-3D receipt")
    if payload.get("schema_version") != STAGE3D_RECEIPT_VERSION:
        raise ValueError("Stage-3D receipt schema version differs")
    if payload.get("manifest_type") != STAGE3D_MANIFEST_TYPE:
        raise ValueError("Stage-3D receipt manifest type differs")
    if payload.get("disposition") != STAGE3D_GO or payload.get("passed") is not True:
        raise ValueError("Resource preflight requires the immutable Stage-3D GO receipt")
    if payload.get("scientific_e26a_started") is not False:
        raise ValueError("Stage-3D receipt indicates scientific E26a was opened")
    if payload.get("scientific_evidence") not in (None, False):
        raise ValueError("Stage-3D numerical preflight cannot be scientific evidence")

    source = payload.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("Stage-3D receipt lacks source binding")
    for key, pattern in (("git_commit", r"[0-9a-f]{40}"), ("source_tree_sha256", None)):
        value = source.get(key)
        if not isinstance(value, str):
            raise ValueError(f"Stage-3D source lacks {key}")
        if pattern is None:
            if not SHA256_PATTERN.fullmatch(value):
                raise ValueError(f"Stage-3D source {key} is invalid")
        elif len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError(f"Stage-3D source {key} is invalid")

    protocol = payload.get("protocol_lock")
    if not isinstance(protocol, Mapping):
        raise ValueError("Stage-3D receipt lacks protocol lock binding")
    for key in ("sha256", "protocol_sha256"):
        if not isinstance(protocol.get(key), str) or not SHA256_PATTERN.fullmatch(
            str(protocol[key])
        ):
            raise ValueError(f"Stage-3D protocol lock lacks {key}")

    if verify_canonical_contract:
        protocol_path = _regular_file(str(protocol.get("path")), "Stage-3D protocol lock")
        if sha256_file(protocol_path) != protocol.get("sha256"):
            raise ValueError("Stage-3D protocol lock byte hash changed")
        validated_protocol = validate_stage3d_protocol_lock(_yaml_mapping(protocol_path))
        validated_receipt = validate_stage3d_admissibility_receipt(
            payload,
            protocol_lock=validated_protocol,
        )
        if validated_receipt.get("disposition") != STAGE3D_GO:
            raise ValueError("Canonical Stage-3D validator did not authorize resource preflight")

    if payload.get("fixed_layouts") != _expected_layouts(config):
        raise ValueError("Stage-3D fixed layouts differ from the prospective contract")

    summary = payload.get("gate_summary")
    if not isinstance(summary, Mapping):
        raise ValueError("Stage-3D receipt lacks gate summary")
    if any(summary.get(f"g{index}_passed") is not True for index in range(7)):
        raise ValueError("Stage-3D G0-G6 did not all pass")
    if (
        summary.get("g3_pass_count") != 12
        or summary.get("g3_required_count") != 12
        or summary.get("g4_pass_count") != 6
        or summary.get("g4_required_count") != 6
    ):
        raise ValueError("Stage-3D required-case counts differ from 12/12 and 6/6")

    g3_cases = payload.get("g3_cases")
    if not isinstance(g3_cases, list) or len(g3_cases) != 12:
        raise ValueError("Stage-3D receipt must contain exactly 12 G3 cases")
    for case in g3_cases:
        if not isinstance(case, Mapping) or case.get("passed") is not True:
            raise ValueError("A Stage-3D G3 case did not pass")
        comparisons = case.get("comparisons")
        if not isinstance(comparisons, Mapping):
            raise ValueError("A Stage-3D G3 case lacks comparisons")
        for comparison in (
            "compiled_bf16_vs_reference_python_bf16",
            "reference_python_bf16_vs_reference_python_fp32",
        ):
            row = comparisons.get(comparison)
            if not isinstance(row, Mapping) or row.get("passed") is not True:
                raise ValueError(f"A Stage-3D G3 {comparison} comparison did not pass")

    replays = payload.get("g4_replays")
    if not isinstance(replays, list) or len(replays) != 6:
        raise ValueError("Stage-3D receipt must contain exactly six same-layout replays")
    if any(not isinstance(row, Mapping) or row.get("passed") is not True for row in replays):
        raise ValueError("A Stage-3D same-layout replay did not pass")

    bindings = {
        "config_sha256": config_sha256,
        "tokenizer_manifest_sha256": tokenizer_manifest_sha256,
        "corpus_manifest_sha256": corpus_manifest_sha256,
    }
    for key, expected in bindings.items():
        if expected is not None and _hash_binding(payload, key) != expected:
            raise ValueError(f"Stage-3D {key} differs from the resource input")
    return dict(payload)


def fixed_resource_policy() -> ResourcePolicy:
    return ResourcePolicy(
        deadline_reference_hours=DEADLINE_REFERENCE_HOURS,
        deadline_fraction_max=DEADLINE_FRACTION_MAX,
        max_main_wall_clock_hours=MAX_MAIN_WALL_CLOCK_HOURS,
        safety_time_multiplier=SAFETY_TIME_MULTIPLIER,
        max_main_checkpoint_storage_gib=MAX_MAIN_CHECKPOINT_STORAGE_GIB,
        token_budgets=TOKEN_BUDGETS,
        main_runs=MAIN_RUNS,
        gpu_lanes=GPU_LANES,
        save_every_tokens=SAVE_EVERY_TOKENS,
    )


def resource_projections_with_gpu_hours(
    measurement: CandidateMeasurement,
    policy: ResourcePolicy,
) -> list[dict[str, Any]]:
    """Return the locked ETA rows plus explicit ten-run GPU-hour totals."""

    rows: list[dict[str, Any]] = []
    for projection in project_candidate_resources(measurement, policy):
        row = dict(projection)
        total_gpu_hours = float(row["single_run_hours"]) * policy.main_runs
        row.update(
            {
                "paired_seed_count": 5,
                "variants_per_seed": 2,
                "total_run_count": policy.main_runs,
                "total_gpu_hours": total_gpu_hours,
                "safety_adjusted_total_gpu_hours": (
                    total_gpu_hours * policy.safety_time_multiplier
                ),
            }
        )
        rows.append(row)
    return rows


def _hardware_by_physical_index(
    rows: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """Index the observed CUDA inventory using its canonical UUID field."""

    indexed: dict[int, dict[str, Any]] = {}
    for row in rows:
        index = row.get("physical_device_index")
        gpu_uuid = row.get("gpu_uuid")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("CUDA inventory has an invalid physical_device_index")
        if not isinstance(gpu_uuid, str) or not gpu_uuid.strip():
            raise ValueError("CUDA inventory lacks canonical gpu_uuid")
        if index in indexed:
            raise ValueError("CUDA inventory repeats a physical device index")
        indexed[index] = row
    return indexed


def _fresh_output_root(path: Path) -> Path:
    unresolved = path.expanduser()
    if unresolved.exists() or unresolved.is_symlink():
        raise FileExistsError(f"Stage-3D resource root must be fresh: {unresolved}")
    RESOURCE_NAMESPACE.mkdir(parents=True, exist_ok=True)
    parent = unresolved.parent.resolve(strict=True)
    resolved = parent / unresolved.name
    if resolved.parent != RESOURCE_NAMESPACE.resolve(strict=True):
        raise ValueError(
            "Stage-3D resource root must be a direct child of the canonical "
            f"namespace: {RESOURCE_NAMESPACE}"
        )
    if not resolved.name or resolved.name.startswith("."):
        raise ValueError("Stage-3D resource run ID is invalid")
    resolved.mkdir(mode=0o700)
    return resolved


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="E26 Stage-3D fixed-layout resource preflight (non-evidence)"
    )
    parser.add_argument("--repo-root", type=Path, default=Path("/home/minjun_dev/CATENA"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--stage3d-receipt", type=Path)
    parser.add_argument("--tokenizer-manifest", type=Path)
    parser.add_argument("--corpus-manifest", type=Path)
    parser.add_argument("--devices", default="0,1,2")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-spec", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if args.worker_spec is None or args.worker_output is None:
            parser.error("worker requires --worker-spec and --worker-output")
        return args
    required = ("output_root", "config", "stage3d_receipt", "tokenizer_manifest", "corpus_manifest")
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error(f"missing required arguments: {', '.join(missing)}")
    return args


def _validate_worker_spec(payload: Mapping[str, Any]) -> dict[str, Any]:
    observed = payload.get("spec_sha256")
    if not isinstance(observed, str):
        raise ValueError("Worker spec lacks hash")
    unhashed = dict(payload)
    unhashed.pop("spec_sha256")
    if observed != sha256_canonical_json(unhashed):
        raise ValueError("Worker spec canonical hash mismatch")
    required = {
        "schema_version",
        "manifest_type",
        "scientific_evidence",
        "scientific_e26a_started",
        "repo_root",
        "output_root",
        "source",
        "candidate",
        "fixed_layout",
        "physical_device_index",
        "gpu_uuid",
        "input_paths",
        "input_hashes",
        "spec_sha256",
    }
    if set(payload) != required:
        raise ValueError("Worker spec fields differ from the fixed-layout contract")
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("manifest_type") != "E26_STAGE3D_RESOURCE_WORKER_SPEC"
        or payload.get("scientific_evidence") is not False
        or payload.get("scientific_e26a_started") is not False
    ):
        raise ValueError("Worker evidence boundary is invalid")
    return dict(payload)


def _worker(args: argparse.Namespace) -> int:
    spec_path = _regular_file(args.worker_spec, "worker spec")
    spec = _validate_worker_spec(read_json_object_strict(spec_path))
    repo = Path(str(spec["repo_root"])).resolve(strict=True)
    if _git(repo, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("Stage-3D resource worker requires a clean worktree")
    source = spec["source"]
    if _git(repo, "rev-parse", "HEAD") != source["git_commit"]:
        raise ValueError("Worker source commit differs from Stage-3D")
    inventory = e26_execution_source_inventory(repo)
    if inventory["source_tree_sha256"] != source["source_tree_sha256"]:
        raise ValueError("Worker source tree differs from Stage-3D")

    input_paths = {
        name: _regular_file(spec["input_paths"][name], f"worker {name}") for name in _INPUT_NAMES
    }
    observed_hashes = {f"{name}_sha256": sha256_file(path) for name, path in input_paths.items()}
    if observed_hashes != spec["input_hashes"]:
        raise ValueError("Worker input hashes changed")
    config = _yaml_mapping(input_paths["config"])
    raw_stage3d = read_json_object_strict(input_paths["stage3d_receipt"])
    _require_exact_bound_input_paths(input_paths, stage3d_receipt=raw_stage3d)
    receipt = validate_stage3d_go_receipt(
        raw_stage3d,
        config=config,
        config_sha256=observed_hashes["config_sha256"],
        tokenizer_manifest_sha256=observed_hashes["tokenizer_manifest_sha256"],
        corpus_manifest_sha256=observed_hashes["corpus_manifest_sha256"],
    )
    candidate = spec["candidate"]
    layout = spec["fixed_layout"]
    expected_candidate = next(
        (row for row in _candidate_rows(config) if row["id"] == candidate["id"]),
        None,
    )
    if candidate != expected_candidate:
        raise ValueError("Worker candidate differs from the locked config")
    expected_layout = next(
        row for row in _expected_layouts(config) if row["candidate_id"] == candidate["id"]
    )
    if layout != expected_layout or layout not in receipt["fixed_layouts"]:
        raise ValueError("Worker layout differs from the Stage-3D fixed layout")
    if int(layout["microbatch_sequences"]) != 1:
        raise ValueError("Stage-3D resource measurement requires microbatch_sequences=1")

    output = Path(args.worker_output).expanduser()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite worker output: {output}")
    device_observation = observed_single_visible_cuda_device(
        expected_physical_index=int(spec["physical_device_index"]),
        expected_gpu_uuid=str(spec["gpu_uuid"]),
    )
    root = Path(str(spec["output_root"])).resolve(strict=True)
    if root.parent != RESOURCE_NAMESPACE.resolve(strict=True):
        raise ValueError("Worker output is outside the canonical resource namespace")
    experiment = f"{RESOURCE_WORKER_PREFIX}{candidate['id']}"
    run = ArtifactRun(
        experiment=experiment,
        artifact_root=CANONICAL_ARTIFACT_ROOT,
        run_mode="MAIN",
        dry_run=False,
        source_root=repo,
        scientific_evidence=False,
        evidence_tier="NON_EVIDENCE_NUMERICAL_PREFLIGHT",
        claim_ceiling="RESOURCE_FEASIBILITY_ONLY",
    )
    tokenizer = ExternalScientificTokenizer.from_manifest(input_paths["tokenizer_manifest"])
    corpus = TokenMemmap.from_scientific_manifest(
        input_paths["corpus_manifest"], tokenizer_manifest=tokenizer.manifest
    )

    base = _candidate_config(candidate, variant="dual_delta_lm")
    tied, dual = build_paired_models(base, seed=26_000, device="cpu")
    rows: list[dict[str, Any]] = []
    parameter_count = dual.parameter_count()
    try:
        assert_matched_models(tied, dual)
        tied_optimizer = make_optimizer(tied)
        dual_optimizer = make_optimizer(dual)
        optimizer_match = compare_optimizer_signatures(tied_optimizer, dual_optimizer).matched
        del tied_optimizer, dual_optimizer
        if not optimizer_match:
            raise ValueError("Tied/Dual optimizer signatures differ")
        initialization_digest = tied.initialization_digest()
        if initialization_digest != dual.initialization_digest():
            raise ValueError("Tied/Dual initialization digests differ")
        parameter_signature = parameter_signature_hash(tied)
        if parameter_signature != parameter_signature_hash(dual):
            raise ValueError("Tied/Dual parameter signatures differ")

        throughput = config["throughput"]
        seed = 260_100 + [row["id"] for row in _candidate_rows(config)].index(candidate["id"])
        for model in (tied, dual):
            model.to(torch.device("cuda:0"))
            row = _measure_variant(
                model,
                candidate_id=str(candidate["id"]),
                corpus=corpus,
                tokenizer=tokenizer,
                seed=seed,
                global_sequences=int(layout["global_batch_sequences"]),
                microbatch_candidates=(FIXED_MICROBATCH_SEQUENCES,),
                warmup_steps=int(throughput["warmup_steps"]),
                measured_steps=int(throughput["measured_steps"]),
                run=run,
                device=torch.device("cuda:0"),
            )
            model.to("cpu")
            torch.cuda.empty_cache()
            if (
                row["microbatch_size"] != FIXED_MICROBATCH_SEQUENCES
                or row["accumulation_steps"] != layout["accumulation_steps"]
            ):
                raise ValueError("Measured variant drifted from the Stage-3D fixed layout")
            rows.append(row)
    finally:
        tied.to("cpu")
        dual.to("cpu")
        del tied, dual
        torch.cuda.empty_cache()

    if [row["variant"] for row in rows] != list(VARIANTS):
        raise ValueError("Resource measurement variant order differs")
    if rows[0]["source_token_accounting"] != rows[1]["source_token_accounting"]:
        raise ValueError("Tied/Dual source token accounting differs")
    diagnostics = [row["diagnostics"] for row in rows]
    measurement = CandidateMeasurement(
        candidate_id=str(candidate["id"]),
        parameter_count=parameter_count,
        matching_passed=True,
        numerical_passed=True,
        tokens_per_second_by_variant={
            str(row["variant"]): float(row["tokens_per_second"]) for row in rows
        },
        checkpoint_bytes=max(int(row["checkpoint"]["bytes"]) for row in rows),
        peak_allocated_bytes=max(int(row["peak_allocated_bytes"]) for row in rows),
        peak_reserved_bytes=max(int(row["peak_reserved_bytes"]) for row in rows),
        p50_step_seconds=max(float(row["p50_step_seconds"]) for row in rows),
        p95_step_seconds=max(float(row["p95_step_seconds"]) for row in rows),
        compile_seconds=sum(float(row["compile_seconds"]) for row in rows),
        graph_break_count=max(int(row["graph_break_count"]) for row in diagnostics),
        fallback_count=max(int(row["fallback_count"]) for row in diagnostics),
        context_length=int(layout["context_length"]),
        selected_microbatch_sequences=FIXED_MICROBATCH_SEQUENCES,
        accumulation_steps=int(layout["accumulation_steps"]),
        measured_optimizer_steps=int(config["throughput"]["measured_steps"]),
        descriptive_stability_steps=0,
        model_config_sha256=sha256_canonical_json(candidate),
        parameter_signature_sha256=parameter_signature,
        paired_initialization_digest=initialization_digest,
        token_mix_bounded_discrepancy_passed=all(
            bool(row["token_mix_bounded_discrepancy_passed"]) for row in rows
        ),
    )
    paired_identity = {
        "variants": list(VARIANTS),
        "physical_layout": layout,
        "shared_data_seed": seed,
        "same_source_token_accounting": True,
        "same_parameter_signature": True,
        "same_initialization_digest": True,
        "same_optimizer_signature": True,
        "variant_specific_layout": False,
        "variant_specific_precision": False,
        "oom_layout_fallback": False,
        "alternative_layout_audit": False,
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "manifest_type": "E26_STAGE3D_RESOURCE_WORKER_REPORT",
        "run_id": run.run_id,
        "experiment": run.experiment,
        "run_mode": "MAIN",
        "scientific_evidence": False,
        "evidence_tier": "NON_EVIDENCE_NUMERICAL_PREFLIGHT",
        "claim_ceiling": "RESOURCE_FEASIBILITY_ONLY",
        "candidate_id": candidate["id"],
        "candidate_config_sha256": sha256_canonical_json(candidate),
        "fixed_layout": layout,
        "measurement": asdict(measurement),
        "variant_measurements": rows,
        "paired_recipe_identity": paired_identity,
        "stage3d_receipt_sha256": receipt["receipt_sha256"],
        "worker_spec_sha256": spec["spec_sha256"],
        "source": source,
        "input_hashes": observed_hashes,
        "execution_device": device_observation,
        "main_test_opened": False,
        "scientific_e26a_started": False,
        "scientific_e26b_started": False,
        "scientific_main_started": False,
        "passed": True,
    }
    run.finalize(
        report,
        (
            "# E26 Stage-3D fixed-layout resource measurement\n\n"
            f"- Candidate: `{candidate['id']}`\n"
            "- Physical microbatch sequences: `1`\n"
            "- Evidence: `NON_EVIDENCE_NUMERICAL_PREFLIGHT`\n"
            "- Scientific E26a started: `false`\n"
        ),
    )
    worker_receipt = {
        "schema_version": SCHEMA_VERSION,
        "manifest_type": "E26_STAGE3D_RESOURCE_WORKER_RECEIPT",
        "candidate_id": candidate["id"],
        "candidate_config_sha256": sha256_canonical_json(candidate),
        "fixed_layout": layout,
        "measurement": asdict(measurement),
        "paired_recipe_identity": paired_identity,
        "run_dir": str(run.run_dir),
        "report_sha256": sha256_file(run.run_dir / "report.json"),
        "worker_spec_sha256": spec["spec_sha256"],
        "source": source,
        "input_hashes": observed_hashes,
        "execution_device": device_observation,
        "scientific_evidence": False,
        "scientific_e26a_started": False,
    }
    worker_receipt["receipt_sha256"] = sha256_canonical_json(worker_receipt)
    write_json_strict(output, worker_receipt)
    return 0


def _validate_worker_receipt(
    payload: Mapping[str, Any],
    *,
    spec: Mapping[str, Any],
    expected_hardware: Mapping[str, Any] | None = None,
    verify_artifacts: bool = True,
) -> dict[str, Any]:
    _require_canonical_hash(payload, label="Stage-3D resource worker receipt")
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("manifest_type") != "E26_STAGE3D_RESOURCE_WORKER_RECEIPT"
        or payload.get("scientific_evidence") is not False
        or payload.get("scientific_e26a_started") is not False
    ):
        raise ValueError("Resource worker receipt evidence boundary is invalid")
    expected = {
        "candidate_id": spec["candidate"]["id"],
        "candidate_config_sha256": sha256_canonical_json(spec["candidate"]),
        "fixed_layout": spec["fixed_layout"],
        "worker_spec_sha256": spec["spec_sha256"],
        "source": spec["source"],
        "input_hashes": spec["input_hashes"],
    }
    drift = [key for key, value in expected.items() if payload.get(key) != value]
    if drift:
        raise ValueError(f"Resource worker receipt binding mismatch: {drift}")
    measurement = payload.get("measurement")
    if not isinstance(measurement, Mapping):
        raise ValueError("Resource worker receipt lacks measurement")
    if (
        measurement.get("selected_microbatch_sequences") != 1
        or measurement.get("accumulation_steps") != spec["fixed_layout"]["accumulation_steps"]
        or measurement.get("graph_break_count") != 0
        or measurement.get("fallback_count") != 0
    ):
        raise ValueError("Resource worker measurement violated the fixed-layout contract")
    identity = payload.get("paired_recipe_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("Resource worker lacks tied/Dual recipe identity")
    required_true = (
        "same_source_token_accounting",
        "same_parameter_signature",
        "same_initialization_digest",
        "same_optimizer_signature",
    )
    if any(identity.get(key) is not True for key in required_true):
        raise ValueError("Resource worker tied/Dual recipe identity failed")
    prohibited_true = (
        "variant_specific_layout",
        "variant_specific_precision",
        "oom_layout_fallback",
        "alternative_layout_audit",
    )
    if any(identity.get(key) is not False for key in prohibited_true):
        raise ValueError("Resource worker used a prohibited layout path")
    device = payload.get("execution_device")
    if not isinstance(device, Mapping):
        raise ValueError("Resource worker receipt lacks execution-device binding")
    expected_device = {
        "physical_device_index": int(spec["physical_device_index"]),
        "gpu_uuid": str(spec["gpu_uuid"]),
        "worker_visible_cuda_index": 0,
        "cuda_visible_devices": str(spec["physical_device_index"]),
        "observation": "PYTORCH_VISIBLE_DEVICE_UUID_VERIFIED",
    }
    for key, expected_value in expected_device.items():
        if device.get(key) != expected_value:
            raise ValueError(f"Resource worker device binding changed: {key}")
    if expected_hardware is not None:
        for key in (
            "physical_device_index",
            "gpu_uuid",
            "name",
            "total_memory_bytes",
            "compute_capability",
        ):
            if device.get(key) != expected_hardware.get(key):
                raise ValueError(f"Resource worker hardware inventory changed: {key}")
    if verify_artifacts:
        run_dir = Path(str(payload.get("run_dir"))).expanduser().resolve(strict=True)
        expected_experiment = f"{RESOURCE_WORKER_PREFIX}{spec['candidate']['id']}"
        if (
            run_dir.parent.name != expected_experiment
            or run_dir.parent.parent != CANONICAL_ARTIFACT_ROOT.resolve(strict=True)
        ):
            raise ValueError("Resource worker run directory is outside its canonical namespace")
        report_path = _regular_file(run_dir / "report.json", "resource worker report")
        if sha256_file(report_path) != payload.get("report_sha256"):
            raise ValueError("Resource worker report SHA-256 changed")
        report = read_json_object_strict(report_path)
        expected_report = {
            "run_id": run_dir.name,
            "experiment": expected_experiment,
            "run_mode": "MAIN",
            "scientific_evidence": False,
            "evidence_tier": "NON_EVIDENCE_NUMERICAL_PREFLIGHT",
            "claim_ceiling": "RESOURCE_FEASIBILITY_ONLY",
            "candidate_id": spec["candidate"]["id"],
            "worker_spec_sha256": spec["spec_sha256"],
            "source": spec["source"],
            "input_hashes": spec["input_hashes"],
            "main_test_opened": False,
            "scientific_e26a_started": False,
            "scientific_e26b_started": False,
            "scientific_main_started": False,
            "passed": True,
        }
        drifted_report = [
            key
            for key, expected_value in expected_report.items()
            if report.get(key) != expected_value
        ]
        if drifted_report:
            raise ValueError(f"Resource worker report binding mismatch: {drifted_report}")
        if report.get("measurement") != payload.get("measurement"):
            raise ValueError("Resource worker report/receipt measurement mismatch")
        if report.get("execution_device") != payload.get("execution_device"):
            raise ValueError("Resource worker report/receipt device mismatch")
    return dict(payload)


def _file_binding(path: Path, *, receipt_field: str | None = None) -> dict[str, Any]:
    resolved = _regular_file(path, "resource input binding")
    binding: dict[str, Any] = {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
    }
    if receipt_field is not None:
        payload = read_json_object_strict(resolved)
        value = payload.get(receipt_field)
        if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
            raise ValueError(f"Bound receipt lacks {receipt_field}: {resolved}")
        binding[receipt_field] = value
    return binding


def _artifact_rows(root: Path, *, exclude: frozenset[str] = frozenset()) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Resource artifacts cannot contain symlinks: {path}")
        if not path.is_file() or path.name in exclude:
            continue
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def _validate_resource_schema(repo: Path, payload: Mapping[str, Any]) -> None:
    try:
        import jsonschema  # type: ignore[import-untyped]
    except ModuleNotFoundError as error:  # pragma: no cover - fail-closed production guard
        raise RuntimeError("Stage-3D resource receipt validation requires jsonschema") from error
    schema_path = _regular_file(
        repo / "schemas/v8_1/e26_stage3d_resource_preflight_receipt.schema.json",
        "Stage-3D resource schema",
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(dict(payload))


def _publish_resource_latest(
    *, output_root: Path, disposition: str, report_path: Path
) -> None:
    latest = {
        "schema_version": SCHEMA_VERSION,
        "manifest_type": "E26_STAGE3D_RESOURCE_PREFLIGHT_LATEST_POINTER",
        "run_dir": str(output_root),
        "disposition": disposition,
        "report_path": str(report_path),
        "report_sha256": sha256_file(report_path),
        "status_sha256": sha256_file(output_root / "status.json"),
        "artifact_audit_sha256": sha256_file(output_root / "artifact_audit.json"),
        "scientific_e26a_started": False,
    }
    resource_receipt = output_root / "resource_preflight.json"
    if resource_receipt.is_file() and not resource_receipt.is_symlink():
        latest["resource_preflight_sha256"] = sha256_file(resource_receipt)
    latest["pointer_sha256"] = sha256_canonical_json(latest)
    temporary = RESOURCE_NAMESPACE / f".latest.{os.getpid()}.json"
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"Refusing to reuse latest temporary path: {temporary}")
    write_json_strict(temporary, latest)
    os.replace(temporary, RESOURCE_NAMESPACE / "latest.json")


def _write_resource_summary(output_root: Path, payload: Mapping[str, Any]) -> Path:
    destination = output_root / "RESULTS_SUMMARY_KO.md"
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite resource summary: {destination}")
    lines = [
        "# E26 Stage-3D Resource Preflight 결과 요약",
        "",
        f"- 판정: `{payload['disposition']}`",
        "- Evidence tier: `NON_EVIDENCE_NUMERICAL_PREFLIGHT`",
        "- Scientific E26a started: `false`",
        "",
    ]
    if payload.get("resource_feasibility_evaluated") is False:
        lines.extend(
            [
                f"- 실행 오류: `{payload.get('error_type', 'UNKNOWN')}`",
                f"- 원인: {payload.get('error', 'unavailable')}",
            ]
        )
    else:
        lines.extend(
            [
                "| Candidate | Tied tok/s | Dual tok/s | Fixed accumulation |",
                "|---|---:|---:|---:|",
            ]
        )
        candidates = payload.get("candidates")
        if isinstance(candidates, list):
            for row in candidates:
                if not isinstance(row, Mapping):
                    continue
                measurement = row.get("measurement")
                rates = (
                    measurement.get("tokens_per_second_by_variant", {})
                    if isinstance(measurement, Mapping)
                    else {}
                )
                layout = row.get("fixed_layout")
                accumulation = (
                    layout.get("accumulation_steps", "?")
                    if isinstance(layout, Mapping)
                    else "?"
                )
                lines.append(
                    "| `{}` | {:.3f} | {:.3f} | {} |".format(
                        row.get("candidate_id", "UNKNOWN"),
                        float(rates.get("projected_tied_delta_lm", float("nan"))),
                        float(rates.get("dual_delta_lm", float("nan"))),
                        accumulation,
                    )
                )
        selection = payload.get("selection")
        lines.append("")
        if isinstance(selection, Mapping):
            lines.extend(
                [
                    f"- 선택 후보: `{selection.get('candidate_id')}`",
                    f"- 선택 token budget/model: `{selection.get('token_budget')}`",
                    "- 이 선택은 처리량·메모리·checkpoint resource gate만 반영한다.",
                ]
            )
        else:
            lines.extend(
                [
                    "- 선택 후보: 없음",
                    f"- Resource gate 원인: {payload.get('selection_error')}",
                ]
            )
    destination.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return destination


def _write_resource_execution_error(output_root: Path, error: Exception) -> None:
    """Durably close a post-namespace operational failure as not evaluable."""

    if output_root.parent != RESOURCE_NAMESPACE.resolve(strict=True):
        return
    error_path = output_root / "execution_error.json"
    if error_path.exists() or error_path.is_symlink():
        return
    error_payload = {
        "schema_version": SCHEMA_VERSION,
        "manifest_type": "E26_STAGE3D_RESOURCE_PREFLIGHT_EXECUTION_ERROR",
        "execution_status": "DEPENDENCY_OR_EXECUTION_ERROR",
        "disposition": "RESOURCE_PREFLIGHT_NOT_EVALUABLE_DEPENDENCY_OR_EXECUTION_ERROR",
        "passed": False,
        "resource_feasibility_evaluated": False,
        "error_type": type(error).__name__,
        "error": str(error),
        "run_dir": str(output_root),
        "scientific_evidence": False,
        "scientific_e26a_started": False,
        "scientific_main_started": False,
    }
    error_payload["receipt_sha256"] = sha256_canonical_json(error_payload)
    write_json_strict(error_path, error_payload)
    report_path = output_root / "report.json"
    if not report_path.exists() and not report_path.is_symlink():
        write_json_strict(report_path, error_payload)
    summary_path = output_root / "RESULTS_SUMMARY_KO.md"
    if not summary_path.exists() and not summary_path.is_symlink():
        _write_resource_summary(output_root, error_payload)
    status_path = output_root / "status.json"
    if not status_path.exists() and not status_path.is_symlink():
        status = {
            "schema_version": SCHEMA_VERSION,
            "manifest_type": "E26_STAGE3D_RESOURCE_PREFLIGHT_STATUS",
            "execution_status": "DEPENDENCY_OR_EXECUTION_ERROR",
            "disposition": error_payload["disposition"],
            "passed": False,
            "resource_preflight_completed": False,
            "resource_feasibility_evaluated": False,
            "scientific_e26a_started": False,
            "scientific_evidence": False,
            "run_dir": str(output_root),
            "execution_error_sha256": sha256_file(error_path),
        }
        status["receipt_sha256"] = sha256_canonical_json(status)
        write_json_strict(status_path, status)
    audit_path = output_root / "artifact_audit.json"
    if not audit_path.exists() and not audit_path.is_symlink():
        rows = _artifact_rows(output_root, exclude=frozenset({"artifact_audit.json"}))
        audit = {
            "schema_version": SCHEMA_VERSION,
            "manifest_type": "E26_STAGE3D_RESOURCE_PREFLIGHT_ARTIFACT_AUDIT",
            "scientific_evidence": False,
            "scientific_e26a_started": False,
            "run_dir": str(output_root),
            "terminal_disposition": error_payload["disposition"],
            "file_count_excluding_self": len(rows),
            "aggregate_sha256_excluding_self": sha256_canonical_json(rows),
            "files": rows,
            "passed": True,
        }
        audit["receipt_sha256"] = sha256_canonical_json(audit)
        write_json_strict(audit_path, audit)
    _publish_resource_latest(
        output_root=output_root,
        disposition=str(error_payload["disposition"]),
        report_path=error_path,
    )


def _write_resource_terminal_artifacts(
    *,
    repo: Path,
    output_root: Path,
    payload: Mapping[str, Any],
    stage3d: Mapping[str, Any],
) -> None:
    """Write, re-read, validate, audit and atomically publish one resource run."""

    receipt_path = output_root / "resource_preflight.json"
    report_path = output_root / "report.json"
    write_json_strict(receipt_path, dict(payload))
    write_json_strict(report_path, dict(payload))
    reloaded = read_json_object_strict(receipt_path)
    _require_canonical_hash(reloaded, label="canonical Stage-3D resource receipt")
    _validate_resource_schema(repo, reloaded)
    if reloaded.get("run_dir") != str(output_root):
        raise ValueError("Canonical resource receipt run directory changed")
    inputs = reloaded.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("Canonical resource receipt lacks exact input bindings")
    for name, binding in inputs.items():
        if not isinstance(binding, Mapping):
            raise ValueError(f"Malformed canonical resource input binding: {name}")
        bound_path = _regular_file(str(binding.get("path")), f"resource input {name}")
        if sha256_file(bound_path) != binding.get("sha256"):
            raise ValueError(f"Canonical resource input changed: {name}")
    if reloaded.get("stage3d_receipt") != inputs.get("stage3d_receipt"):
        raise ValueError("Duplicated Stage-3D receipt bindings differ")
    if reloaded.get("passed") is True:
        protocol_binding = stage3d.get("protocol_lock")
        if not isinstance(protocol_binding, Mapping):
            raise ValueError("Stage-3D GO lacks protocol lock binding")
        protocol = validate_stage3d_protocol_lock(
            _yaml_mapping(_regular_file(str(protocol_binding.get("path")), "Stage-3D protocol"))
        )
        validate_stage3d_resource_preflight_receipt(
            reloaded,
            protocol_lock=protocol,
            stage3d_receipt=stage3d,
        )
    if sha256_file(receipt_path) != sha256_file(report_path):
        raise ValueError("Canonical resource receipt/report bytes differ")

    # Re-hash every bound worker artifact before issuing a terminal status.
    # Worker reports live in their own canonical sibling namespaces; worker
    # receipts are part of this aggregate run directory.
    for candidate in reloaded["candidates"]:
        report_binding = candidate["worker_report"]
        report = _regular_file(report_binding["path"], "bound worker report")
        if (
            sha256_file(report) != report_binding["sha256"]
            or report_binding["sha256"] != candidate["worker_report_sha256"]
        ):
            raise ValueError("Bound worker report changed during artifact finalization")
        receipt_binding = candidate["worker_receipt"]
        receipt = _regular_file(receipt_binding["path"], "bound worker receipt")
        worker_receipt = read_json_object_strict(receipt)
        _require_canonical_hash(worker_receipt, label="bound worker receipt")
        if (
            sha256_file(receipt) != receipt_binding["sha256"]
            or receipt_binding["receipt_sha256"] != candidate["worker_receipt_sha256"]
            or worker_receipt["receipt_sha256"] != candidate["worker_receipt_sha256"]
        ):
            raise ValueError("Bound worker receipt changed during artifact finalization")

    _write_resource_summary(output_root, reloaded)

    status = {
        "schema_version": SCHEMA_VERSION,
        "manifest_type": "E26_STAGE3D_RESOURCE_PREFLIGHT_STATUS",
        "disposition": reloaded["disposition"],
        "passed": reloaded["passed"],
        "resource_preflight_completed": True,
        "scientific_e26a_started": False,
        "scientific_evidence": False,
        "run_dir": str(output_root),
        "resource_preflight_sha256": sha256_file(receipt_path),
        "resource_receipt_sha256": reloaded["receipt_sha256"],
    }
    status["receipt_sha256"] = sha256_canonical_json(status)
    write_json_strict(output_root / "status.json", status)

    rows = _artifact_rows(output_root, exclude=frozenset({"artifact_audit.json"}))
    audit = {
        "schema_version": SCHEMA_VERSION,
        "manifest_type": "E26_STAGE3D_RESOURCE_PREFLIGHT_ARTIFACT_AUDIT",
        "scientific_evidence": False,
        "scientific_e26a_started": False,
        "run_dir": str(output_root),
        "file_count_excluding_self": len(rows),
        "aggregate_sha256_excluding_self": sha256_canonical_json(rows),
        "files": rows,
        "external_worker_artifacts": [
            {
                "candidate_id": row["candidate_id"],
                "worker_report": row["worker_report"],
            }
            for row in reloaded["candidates"]
        ],
        "passed": True,
    }
    audit["receipt_sha256"] = sha256_canonical_json(audit)
    write_json_strict(output_root / "artifact_audit.json", audit)

    _publish_resource_latest(
        output_root=output_root,
        disposition=str(reloaded["disposition"]),
        report_path=report_path,
    )


def _parent(args: argparse.Namespace) -> int:
    repo = args.repo_root.expanduser().resolve(strict=True)
    if _git(repo, "status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("Stage-3D resource preflight requires a clean committed worktree")
    paths = {
        name: _regular_file(getattr(args, name), f"Stage-3D resource {name}")
        for name in _INPUT_NAMES
    }
    input_hashes = {f"{name}_sha256": sha256_file(path) for name, path in paths.items()}
    config = _yaml_mapping(paths["config"])
    raw_stage3d = read_json_object_strict(paths["stage3d_receipt"])
    _require_exact_bound_input_paths(paths, stage3d_receipt=raw_stage3d)
    stage3d = validate_stage3d_go_receipt(
        raw_stage3d,
        config=config,
        config_sha256=input_hashes["config_sha256"],
        tokenizer_manifest_sha256=input_hashes["tokenizer_manifest_sha256"],
        corpus_manifest_sha256=input_hashes["corpus_manifest_sha256"],
    )
    source_inventory = e26_execution_source_inventory(repo)
    source = stage3d["source"]
    if source["git_commit"] != _git(repo, "rev-parse", "HEAD"):
        raise ValueError("Current commit differs from the Stage-3D GO receipt")
    if source["source_tree_sha256"] != source_inventory["source_tree_sha256"]:
        raise ValueError("Current source tree differs from the Stage-3D GO receipt")

    candidates = _candidate_rows(config)
    layouts = _expected_layouts(config)
    devices = [value.strip() for value in args.devices.split(",") if value.strip()]
    if len(devices) != len(candidates) or len(set(devices)) != len(devices):
        raise ValueError("Supply one distinct CUDA device per fixed candidate")
    hardware = cuda_hardware_inventory(devices)
    hardware_by_index = _hardware_by_physical_index(hardware)
    if len(hardware_by_index) != len(devices):
        raise ValueError("CUDA device inventory is not one-to-one")

    # A blocked/malformed Stage-3D receipt must not create even a non-evidence
    # measurement namespace.  Create the fresh root only after all admission
    # and hardware checks above have succeeded.
    output_root = _fresh_output_root(args.output_root)

    tool = Path(__file__).resolve()
    specs: dict[str, dict[str, Any]] = {}
    processes: list[tuple[str, subprocess.Popen[str], Any]] = []
    for candidate, layout, device in zip(candidates, layouts, devices, strict=True):
        index = int(device)
        if index not in hardware_by_index:
            raise ValueError(f"CUDA inventory lacks physical device {index}")
        candidate_id = str(candidate["id"])
        spec = {
            "schema_version": SCHEMA_VERSION,
            "manifest_type": "E26_STAGE3D_RESOURCE_WORKER_SPEC",
            "scientific_evidence": False,
            "scientific_e26a_started": False,
            "repo_root": str(repo),
            "output_root": str(output_root),
            "source": source,
            "candidate": candidate,
            "fixed_layout": layout,
            "physical_device_index": index,
            "gpu_uuid": hardware_by_index[index]["gpu_uuid"],
            "input_paths": {name: str(path) for name, path in paths.items()},
            "input_hashes": input_hashes,
        }
        spec["spec_sha256"] = sha256_canonical_json(spec)
        specs[candidate_id] = spec
        spec_path = output_root / f"{candidate_id}_worker_spec.json"
        result_path = output_root / f"{candidate_id}_worker_receipt.json"
        log_handle = (output_root / f"{candidate_id}_worker.log").open(
            "x", encoding="utf-8", newline="\n"
        )
        write_json_strict(spec_path, spec)
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = device
        process = subprocess.Popen(
            [
                sys.executable,
                str(tool),
                "--worker",
                "--worker-spec",
                str(spec_path),
                "--worker-output",
                str(result_path),
            ],
            cwd=repo,
            env=env,
            text=True,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((candidate_id, process, log_handle))

    failures: list[str] = []
    for candidate_id, process, log_handle in processes:
        return_code = process.wait()
        log_handle.close()
        if return_code:
            failures.append(f"{candidate_id}: exit {return_code}")
    if failures:
        write_json_strict(
            output_root / "failure_status.json",
            {
                "schema_version": SCHEMA_VERSION,
                "manifest_type": "E26_STAGE3D_RESOURCE_PREFLIGHT_FAILURE",
                "scientific_evidence": False,
                "scientific_e26a_started": False,
                "failures": failures,
            },
        )
        raise RuntimeError(f"Stage-3D resource workers failed: {failures}")

    worker_receipts = {
        candidate["id"]: _validate_worker_receipt(
            read_json_object_strict(output_root / f"{candidate['id']}_worker_receipt.json"),
            spec=specs[str(candidate["id"])],
            expected_hardware=hardware_by_index[
                int(specs[str(candidate["id"])]["physical_device_index"])
            ],
        )
        for candidate in candidates
    }
    measurements = tuple(
        CandidateMeasurement(**worker_receipts[str(candidate["id"])]["measurement"])
        for candidate in candidates
    )
    policy = fixed_resource_policy()
    selection = None
    selection_error: str | None = None
    try:
        selection = select_candidate(config=config, measurements=measurements, policy=policy)
    except E26AGateBlocked as error:
        selection_error = str(error)
    candidate_receipts = []
    for candidate, layout, measurement in zip(candidates, layouts, measurements, strict=True):
        worker = worker_receipts[str(candidate["id"])]
        worker_receipt_path = _regular_file(
            output_root / f"{candidate['id']}_worker_receipt.json",
            "resource worker receipt",
        )
        worker_report_path = _regular_file(
            Path(str(worker["run_dir"])) / "report.json", "resource worker report"
        )
        candidate_receipts.append(
            {
                "candidate_id": candidate["id"],
                "candidate_config_sha256": sha256_canonical_json(candidate),
                "fixed_layout": layout,
                "measurement": asdict(measurement),
                "resource_projections": resource_projections_with_gpu_hours(measurement, policy),
                "paired_recipe_identity": worker["paired_recipe_identity"],
                "execution_device": worker["execution_device"],
                "worker_run_dir": worker["run_dir"],
                "worker_report_sha256": worker["report_sha256"],
                "worker_receipt_sha256": worker["receipt_sha256"],
                "worker_report": {
                    "path": str(worker_report_path),
                    "sha256": sha256_file(worker_report_path),
                },
                "worker_receipt": {
                    "path": str(worker_receipt_path),
                    "sha256": sha256_file(worker_receipt_path),
                    "receipt_sha256": worker["receipt_sha256"],
                },
            }
        )
    input_bindings = {
        name: _file_binding(
            path,
            receipt_field="receipt_sha256" if name == "stage3d_receipt" else None,
        )
        for name, path in paths.items()
    }
    passed = selection is not None
    payload = {
        "schema_version": RESOURCE_RECEIPT_VERSION,
        "manifest_type": RESOURCE_MANIFEST_TYPE,
        "scientific_evidence": False,
        "evidence_tier": "NON_EVIDENCE_NUMERICAL_PREFLIGHT",
        "claim_ceiling": "RESOURCE_FEASIBILITY_ONLY",
        "run_dir": str(output_root),
        "disposition": (
            "RESOURCE_PREFLIGHT_FEASIBLE"
            if passed
            else "RESOURCE_PREFLIGHT_INFEASIBLE"
        ),
        "inputs": input_bindings,
        "stage3d_receipt": {
            "path": str(paths["stage3d_receipt"]),
            "sha256": input_hashes["stage3d_receipt_sha256"],
            "receipt_sha256": stage3d["receipt_sha256"],
        },
        "source": source,
        "fixed_layouts": layouts,
        "resource_policy": {
            "token_budgets": list(TOKEN_BUDGETS),
            "max_main_wall_clock_hours": MAX_MAIN_WALL_CLOCK_HOURS,
            "safety_time_multiplier": SAFETY_TIME_MULTIPLIER,
            "max_main_checkpoint_storage_gib": MAX_MAIN_CHECKPOINT_STORAGE_GIB,
            "deadline_reference_hours": DEADLINE_REFERENCE_HOURS,
            "deadline_fraction_max": DEADLINE_FRACTION_MAX,
            "main_runs": MAIN_RUNS,
            "gpu_lanes": GPU_LANES,
            "save_every_tokens": SAVE_EVERY_TOKENS,
        },
        "hardware_inventory": hardware,
        "candidates": candidate_receipts,
        "selection": selection.as_dict() if selection is not None else None,
        "selection_error": selection_error,
        "main_test_opened": False,
        "scientific_e26a_started": False,
        "scientific_e26b_started": False,
        "scientific_main_started": False,
        "canonical_e26_artifact_created": False,
        "passed": passed,
    }
    payload["receipt_sha256"] = sha256_canonical_json(payload)
    _write_resource_terminal_artifacts(
        repo=repo,
        output_root=output_root,
        payload=payload,
        stage3d=stage3d,
    )
    return 0 if passed else 1


def main() -> int:
    args = _parse_args()
    try:
        return _worker(args) if args.worker else _parent(args)
    except Exception as error:
        print(f"Stage-3D resource preflight dependency/execution error: {error}", file=sys.stderr)
        if not args.worker and args.output_root is not None:
            candidate = args.output_root.expanduser()
            if candidate.is_dir() and not candidate.is_symlink():
                try:
                    _write_resource_execution_error(candidate.resolve(strict=True), error)
                except Exception as terminal_error:
                    print(
                        "Could not finalize the Stage-3D resource execution-error "
                        f"artifact: {terminal_error}",
                        file=sys.stderr,
                    )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
