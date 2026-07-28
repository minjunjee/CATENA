from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from catena.core.config import load_config
from catena.core.io import file_sha256, write_jsonl
from catena.data.transactional_sequence_v2 import (
    base_transaction_digest_v2,
    generate_transactional_sequence_batch_v2,
    sequence_model_input_v2,
)
from catena.eval.postcore_metrics import exact_sign_flip
from catena.models.sequence_memory_v2 import (
    SequenceControlV2,
    TransactionalSequenceMemoryV2,
)
from experiments.common import build_parser, finalize_run, initialize_run

EXPERIMENT_ID = "e14_plan_continuation"
DEFAULT_CONFIG = "configs/e14_plan_continuation.yaml"
FIXED_SEEDS = (101, 211, 307, 401, 503)
FIXED_VARIANTS = ("tied", "dual")
FIXED_UPDATES = (1, 4, 8)
FIXED_GAPS = (0, 128, 512, 2048)
REQUIRED_VARIANT = "dual"
STATIC_EVIDENCE_BOUNDARY: dict[str, Any] = {
    "tier": "CONTROLLED_REFERENCE",
    "proxy_type": "STRUCTURED_SYNTHETIC_ENTITY_VALUE",
    "oracle_entity_address": True,
    "oracle_old_new_candidates": True,
    "independent_plan_semantics_tested": False,
    "semantic_demand_inference_tested": False,
    "learned_addressing_tested": False,
    "language_model_transfer_claim_eligible": False,
    "general_agent_planning_claim_eligible": False,
    "official_backend_claim_eligible": False,
    "production_break_even_claim_eligible": False,
    "long_gap_persistence_claim_eligible": False,
}


@dataclass(frozen=True, slots=True)
class SealedCheckpoint:
    seed: int
    variant: str
    checkpoint_path: Path
    checkpoint_sha256: str
    source_run_dir: Path
    source_report_path: Path
    source_report_sha256: str
    source_metrics_path: Path
    source_metrics_sha256: str
    source_manifest_path: Path
    source_manifest_sha256: str
    checkpoint_config: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "variant": self.variant,
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_sha256": self.checkpoint_sha256,
            "source_run_dir": str(self.source_run_dir),
            "source_report_path": str(self.source_report_path),
            "source_report_sha256": self.source_report_sha256,
            "source_metrics_path": str(self.source_metrics_path),
            "source_metrics_sha256": self.source_metrics_sha256,
            "source_manifest_path": str(self.source_manifest_path),
            "source_manifest_sha256": self.source_manifest_sha256,
        }


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise TypeError(
                    f"Expected JSON object at {path}:{line_number}"
                )
            rows.append(payload)
    return rows


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _contained_path(
    root: Path,
    reference: str | Path,
    *,
    label: str,
) -> Path:
    resolved_root = root.resolve()
    path = Path(reference)
    resolved = path.resolve() if path.is_absolute() else (resolved_root / path).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"{label} escapes {resolved_root}: {reference}")
    return resolved


def _require_hash(path: Path, expected: object, *, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    expected_digest = str(expected)
    if len(expected_digest) != 64 or any(
        character not in "0123456789abcdef"
        for character in expected_digest
    ):
        raise ValueError(f"Invalid SHA-256 for {label}: {expected_digest!r}")
    observed = file_sha256(path)
    if observed != expected_digest:
        raise ValueError(
            f"{label} hash mismatch: expected={expected_digest}, "
            f"observed={observed}"
        )
    return observed


def _resolve_latest_run(artifact_root: Path, experiment_id: str) -> Path:
    experiment_root = (artifact_root.resolve() / experiment_id).resolve()
    pointer = experiment_root / "latest.json"
    if not pointer.is_file():
        raise FileNotFoundError(f"Missing latest pointer: {pointer}")
    run_reference = str(_read_json(pointer)["run_dir"])
    run_dir = _contained_path(
        experiment_root,
        run_reference,
        label=f"{experiment_id}.latest.run_dir",
    )
    if run_dir.parent != experiment_root or not run_dir.is_dir():
        raise ValueError(
            f"{experiment_id} latest pointer is not a direct run directory: "
            f"{run_dir}"
        )
    return run_dir


def _validated_experiment_id(value: object, *, label: str) -> str:
    experiment_id = str(value)
    if (
        not experiment_id
        or Path(experiment_id).name != experiment_id
        or experiment_id in {".", ".."}
    ):
        raise ValueError(f"{label} is not a safe experiment ID: {value!r}")
    return experiment_id


def _nested_value(payload: dict[str, Any], dotted_path: str) -> Any:
    value: Any = payload
    for component in dotted_path.split("."):
        if not isinstance(value, dict) or component not in value:
            raise KeyError(f"Missing dependency field {dotted_path!r}")
        value = value[component]
    return value


def _contract_axes(
    config: dict[str, Any],
    *,
    dry_run: bool,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    dependency = config.get("dependency")
    if not isinstance(dependency, dict):
        raise ValueError("E14 config must define a dependency mapping")
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError(f"config.experiment_id must be {EXPERIMENT_ID!r}")
    _validated_experiment_id(
        dependency.get("e13c_experiment_id"),
        label="dependency.e13c_experiment_id",
    )
    _validated_experiment_id(
        dependency.get("e13b_experiment_id"),
        label="dependency.e13b_experiment_id",
    )
    _validated_experiment_id(
        dependency.get("calibration_experiment_id"),
        label="dependency.calibration_experiment_id",
    )
    calibration_gate_field = str(
        dependency.get("calibration_gate_field", "")
    )
    if not calibration_gate_field.startswith("claim_gate."):
        raise ValueError(
            "dependency.calibration_gate_field must name a claim_gate field"
        )
    if dependency.get("required_variant") != REQUIRED_VARIANT:
        raise ValueError(
            f"dependency.required_variant must be {REQUIRED_VARIANT!r}"
        )
    if config.get("model", {}).get("variant") != REQUIRED_VARIANT:
        raise ValueError(f"model.variant must be {REQUIRED_VARIANT!r}")
    seeds = tuple(int(value) for value in dependency.get("required_seeds", ()))
    updates = tuple(
        int(value) for value in dependency.get("required_updates", ())
    )
    gaps = tuple(
        int(value) for value in dependency.get("required_gap_events", ())
    )
    if seeds != FIXED_SEEDS:
        raise ValueError(
            f"dependency.required_seeds must equal {list(FIXED_SEEDS)}"
        )
    if updates != FIXED_UPDATES:
        raise ValueError(
            f"dependency.required_updates must equal {list(FIXED_UPDATES)}"
        )
    if gaps != FIXED_GAPS:
        raise ValueError(
            f"dependency.required_gap_events must equal {list(FIXED_GAPS)}"
        )
    evaluation = config.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("E14 config must define an evaluation mapping")
    evaluation_updates = tuple(
        int(value) for value in evaluation.get("updates", ())
    )
    evaluation_gaps = tuple(
        int(value) for value in evaluation.get("gap_events", ())
    )
    if evaluation_updates != FIXED_UPDATES:
        raise ValueError(
            f"evaluation.updates must equal {list(FIXED_UPDATES)}"
        )
    if evaluation_gaps != FIXED_GAPS:
        raise ValueError(
            f"evaluation.gap_events must equal {list(FIXED_GAPS)}"
        )
    claim_gate = config.get("claim_gate")
    if not isinstance(claim_gate, dict):
        raise ValueError("E14 config must define a claim_gate mapping")
    if claim_gate.get("primary_estimand") != (
        "affected_plan_correction_gain"
    ):
        raise ValueError(
            "claim_gate.primary_estimand must be "
            "'affected_plan_correction_gain'"
        )
    if dry_run:
        return seeds[:1], updates[:1], gaps[:1]
    return seeds, updates, gaps


def _validate_source_record(
    raw: dict[str, Any],
    *,
    artifact_root: Path,
    e13b_experiment_id: str,
    calibration_experiment_id: str,
    calibration_gate_field: str,
    expected_e13b_config: dict[str, Any],
    expected_e13b_config_file_sha256: str,
    expected_run_mode: str,
    dry_run: bool,
) -> tuple[tuple[int, str], SealedCheckpoint | None]:
    seed = int(raw["seed"])
    variant = str(raw["variant"])
    source_root = (
        artifact_root.resolve() / e13b_experiment_id
    ).resolve()
    run_dir = _contained_path(
        source_root,
        str(raw["run_dir"]),
        label=f"E13b seed={seed} variant={variant} run_dir",
    )
    if run_dir.parent != source_root or not run_dir.is_dir():
        raise ValueError(f"E13b source is not a direct run directory: {run_dir}")

    report_path = _contained_path(
        run_dir,
        str(raw["report_path"]),
        label=f"E13b seed={seed} variant={variant} report",
    )
    metrics_path = _contained_path(
        run_dir,
        str(raw["metrics_path"]),
        label=f"E13b seed={seed} variant={variant} metrics",
    )
    checkpoint_path = _contained_path(
        run_dir,
        str(raw["checkpoint_path"]),
        label=f"E13b seed={seed} variant={variant} checkpoint",
    )
    manifest_path = _contained_path(
        run_dir,
        "run_manifest.json",
        label=f"E13b seed={seed} variant={variant} manifest",
    )
    if report_path != run_dir / "report.json":
        raise ValueError(f"Nonstandard E13b report path: {report_path}")
    if metrics_path != run_dir / "sequence_main_metrics.jsonl":
        raise ValueError(f"Nonstandard E13b metrics path: {metrics_path}")
    if (
        checkpoint_path.parent != (run_dir / "checkpoints").resolve()
        or checkpoint_path.name != f"{variant}_seed{seed}.pt"
    ):
        raise ValueError(f"Nonstandard E13b checkpoint path: {checkpoint_path}")

    report_digest = _require_hash(
        report_path,
        raw["report_sha256"],
        label=f"E13b seed={seed} variant={variant} report",
    )
    metrics_digest = _require_hash(
        metrics_path,
        raw["metrics_sha256"],
        label=f"E13b seed={seed} variant={variant} metrics",
    )
    checkpoint_digest = _require_hash(
        checkpoint_path,
        raw["checkpoint_sha256"],
        label=f"E13b seed={seed} variant={variant} checkpoint",
    )
    manifest = _read_json(manifest_path)
    if manifest.get("experiment_id") != e13b_experiment_id:
        raise ValueError(f"Wrong E13b experiment_id in {manifest_path}")
    if manifest.get("run_mode") != expected_run_mode:
        raise ValueError(
            f"Wrong E13b run_mode in {manifest_path}: "
            f"{manifest.get('run_mode')!r}"
        )
    if manifest.get("config") != expected_e13b_config:
        raise ValueError(f"E13b resolved config mismatch in {manifest_path}")
    expected_config_hash = _canonical_sha256(expected_e13b_config)
    if raw.get("source_config_canonical_sha256") != expected_config_hash:
        raise ValueError(
            f"E13b source config hash mismatch for seed={seed}, "
            f"variant={variant}"
        )

    source_report = _read_json(report_path)
    expected_status = "DRY_RUN" if dry_run else "PASS"
    expected_gate = "DRY_RUN" if dry_run else "PENDING_AGGREGATE"
    if source_report.get("status") != expected_status:
        raise ValueError(f"Ineligible E13b status in {report_path}")
    if source_report.get("variant") != variant:
        raise ValueError(f"E13b report variant mismatch in {report_path}")
    if int(source_report.get("seed", -1)) != seed:
        raise ValueError(f"E13b report seed mismatch in {report_path}")
    if source_report.get("claim_gate", {}).get("status") != expected_gate:
        raise ValueError(f"Ineligible E13b claim gate in {report_path}")
    if not dry_run:
        calibration = source_report.get("calibration_dependency")
        if not isinstance(calibration, dict):
            raise ValueError(
                f"E13b source lacks prospective calibration dependency: "
                f"{report_path}"
            )
        if calibration.get("experiment_id") != calibration_experiment_id:
            raise ValueError(
                f"E13b source has wrong calibration dependency: {report_path}"
            )
        calibration_root = (
            artifact_root.resolve() / calibration_experiment_id
        ).resolve()
        calibration_run = _contained_path(
            calibration_root,
            str(calibration["run_dir"]),
            label=f"E13b seed={seed} calibration run",
        )
        if calibration_run.parent != calibration_root:
            raise ValueError(
                f"E13b calibration is not a direct run: {calibration_run}"
            )
        calibration_report = _contained_path(
            calibration_run,
            str(calibration["report_path"]),
            label=f"E13b seed={seed} calibration report",
        )
        if calibration_report != calibration_run / "report.json":
            raise ValueError(
                f"Nonstandard calibration report path: {calibration_report}"
            )
        _require_hash(
            calibration_report,
            calibration["report_sha256"],
            label=f"E13b seed={seed} calibration report",
        )
        calibration_manifest = _contained_path(
            calibration_run,
            str(calibration["run_manifest_path"]),
            label=f"E13b seed={seed} calibration manifest",
        )
        if calibration_manifest != calibration_run / "run_manifest.json":
            raise ValueError(
                f"Nonstandard calibration manifest path: "
                f"{calibration_manifest}"
            )
        _require_hash(
            calibration_manifest,
            calibration["run_manifest_sha256"],
            label=f"E13b seed={seed} calibration manifest",
        )
        if calibration.get("source_config_sha256") != (
            expected_e13b_config_file_sha256
        ):
            raise ValueError(
                f"E13b source cites the wrong calibrated config hash: "
                f"{report_path}"
            )
        calibration_payload = _read_json(calibration_report)
        if calibration_payload.get("status") != "PASS" or _nested_value(
            calibration_payload,
            calibration_gate_field,
        ) is not True:
            raise ValueError(
                f"E13b calibration dependency did not open E13b: "
                f"{calibration_report}"
            )

    if variant != REQUIRED_VARIANT:
        return (seed, variant), None

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if file_sha256(checkpoint_path) != checkpoint_digest:
        raise RuntimeError(
            f"Checkpoint changed while loading: {checkpoint_path}"
        )
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint mapping: {checkpoint_path}")
    if checkpoint.get("variant") != variant:
        raise ValueError(f"Checkpoint variant mismatch: {checkpoint_path}")
    if int(checkpoint.get("seed", -1)) != seed:
        raise ValueError(f"Checkpoint seed mismatch: {checkpoint_path}")
    checkpoint_config = checkpoint.get("config")
    if not isinstance(checkpoint_config, dict):
        raise TypeError(f"Checkpoint config is not a mapping: {checkpoint_path}")
    if not dry_run and checkpoint_config != expected_e13b_config:
        raise ValueError(f"Checkpoint config mismatch: {checkpoint_path}")
    if not isinstance(checkpoint.get("model"), dict):
        raise TypeError(f"Checkpoint model state is not a mapping: {checkpoint_path}")
    if not dry_run and checkpoint.get("model_class") != (
        "TransactionalSequenceMemoryV2"
    ):
        raise ValueError(f"Checkpoint is not a V2 sequence model: {checkpoint_path}")

    return (
        (seed, variant),
        SealedCheckpoint(
            seed=seed,
            variant=variant,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_digest,
            source_run_dir=run_dir,
            source_report_path=report_path,
            source_report_sha256=report_digest,
            source_metrics_path=metrics_path,
            source_metrics_sha256=metrics_digest,
            source_manifest_path=manifest_path.resolve(),
            source_manifest_sha256=file_sha256(manifest_path),
            checkpoint_config=dict(checkpoint_config),
        ),
    )


def resolve_e13c_dependency(
    *,
    artifact_root: str | Path,
    config: dict[str, Any],
    dry_run: bool,
) -> tuple[dict[str, Any], list[SealedCheckpoint]]:
    seeds, updates, gaps = _contract_axes(config, dry_run=dry_run)
    dependency = dict(config["dependency"])
    expected_e13c_config = load_config(
        str(dependency["e13c_config_path"])
    )
    expected_e13b_config = load_config(
        str(dependency["e13b_config_path"])
    )
    expected_e13b_config_file_sha256 = file_sha256(
        str(dependency["e13b_config_path"])
    )
    e13c_experiment_id = _validated_experiment_id(
        dependency["e13c_experiment_id"],
        label="dependency.e13c_experiment_id",
    )
    e13b_experiment_id = _validated_experiment_id(
        dependency["e13b_experiment_id"],
        label="dependency.e13b_experiment_id",
    )
    calibration_experiment_id = _validated_experiment_id(
        dependency["calibration_experiment_id"],
        label="dependency.calibration_experiment_id",
    )
    calibration_gate_field = str(dependency["calibration_gate_field"])
    if expected_e13c_config.get("experiment_id") != e13c_experiment_id:
        raise ValueError("E13c config experiment_id does not match E14")
    if expected_e13b_config.get("experiment_id") != e13b_experiment_id:
        raise ValueError("E13b config experiment_id does not match E14")
    source_protocol = expected_e13b_config.get("protocol")
    if not isinstance(source_protocol, dict) or not (
        source_protocol.get("repair_id")
        == "prospective_learned_distractor_path_repair"
        and source_protocol.get("verified_role") == "semantic_input_only"
        and source_protocol.get("update_mask_role") == "audit_metadata_only"
        and source_protocol.get("paired_base_transactions_across_gaps")
        is True
    ):
        raise ValueError(
            "E13b config is not the repaired learned-distractor protocol"
        )
    e13c_source = expected_e13c_config.get("source")
    if not isinstance(e13c_source, dict):
        raise ValueError("E13c config source contract is missing")
    if e13c_source.get("experiment_id") != e13b_experiment_id:
        raise ValueError("E13c config names the wrong source experiment")
    if Path(str(e13c_source.get("config_path", ""))).resolve() != Path(
        str(dependency["e13b_config_path"])
    ).resolve():
        raise ValueError("E13c and E14 name different E13b source configs")
    if tuple(int(value) for value in e13c_source.get("required_seeds", ())) != (
        FIXED_SEEDS
    ):
        raise ValueError("E13c config does not require the fixed five seeds")
    if tuple(
        str(value) for value in e13c_source.get("required_variants", ())
    ) != FIXED_VARIANTS:
        raise ValueError("E13c config does not require tied and dual sources")
    if tuple(
        int(value) for value in e13c_source.get("required_updates", ())
    ) != FIXED_UPDATES:
        raise ValueError("E13c config has the wrong update grid")
    if tuple(
        int(value) for value in e13c_source.get(
            "required_gap_events",
            (),
        )
    ) != FIXED_GAPS:
        raise ValueError("E13c config has the wrong gap grid")
    if tuple(int(value) for value in expected_e13b_config.get("seeds", ())) != (
        FIXED_SEEDS
    ):
        raise ValueError("E13b config does not contain the fixed five seeds")
    if tuple(
        str(value)
        for value in expected_e13b_config.get("model", {}).get(
            "variants",
            (),
        )
    ) != FIXED_VARIANTS:
        raise ValueError("E13b config does not contain tied and dual variants")
    root = Path(artifact_root).resolve()
    e13c_run = _resolve_latest_run(root, e13c_experiment_id)
    report_path = _contained_path(
        e13c_run,
        "report.json",
        label="E13c report",
    )
    manifest_path = _contained_path(
        e13c_run,
        "run_manifest.json",
        label="E13c manifest",
    )
    provenance_path = _contained_path(
        e13c_run,
        "source_run_provenance.jsonl",
        label="E13c source provenance",
    )
    for path in (report_path, manifest_path, provenance_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing E13c dependency file: {path}")

    report = _read_json(report_path)
    manifest = _read_json(manifest_path)
    expected_status = "DRY_RUN" if dry_run else "PASS"
    expected_run_mode = "DRY_RUN" if dry_run else "MAIN"
    if report.get("status") != expected_status:
        raise RuntimeError(
            f"E13c latest run has status={report.get('status')!r}; "
            f"expected {expected_status!r}"
        )
    supported = report.get("claim_gate", {}).get("supported")
    if dry_run:
        if supported is not False:
            raise RuntimeError("E13c dry-run must not open its claim gate")
    elif supported is not True:
        raise RuntimeError(
            "E14 is blocked because latest E13c is not supported"
        )
    if not dry_run:
        conditions = report.get("claim_gate", {}).get("conditions")
        if (
            not isinstance(conditions, dict)
            or not conditions
            or any(value is not True for value in conditions.values())
        ):
            raise RuntimeError(
                "E13c supported flag is inconsistent with its condition gates"
            )
    if manifest.get("experiment_id") != e13c_experiment_id:
        raise ValueError(f"Wrong E13c experiment_id in {manifest_path}")
    if manifest.get("run_mode") != expected_run_mode:
        raise ValueError(f"Wrong E13c run_mode in {manifest_path}")
    if manifest.get("config") != expected_e13c_config:
        raise ValueError(f"E13c resolved config mismatch in {manifest_path}")

    expected_source_contract = {
        "fixed_seeds": list(seeds),
        "variants": list(FIXED_VARIANTS),
        "updates": list(updates),
        "gap_events": list(gaps),
        "complete_grid_required": True,
        "unique_run_per_seed_variant_required": True,
    }
    observed_source_contract = report.get("source_contract")
    if not isinstance(observed_source_contract, dict) or any(
        observed_source_contract.get(key) != value
        for key, value in expected_source_contract.items()
    ):
        raise ValueError("E13c source contract does not match E14 requirements")
    if observed_source_contract.get(
        "same_base_transaction_across_gaps_required"
    ) is not True:
        raise ValueError(
            "E13c source contract did not require paired base transactions"
        )
    if Path(
        str(observed_source_contract.get("source_config_path", ""))
    ).resolve() != Path(str(dependency["e13b_config_path"])).resolve():
        raise ValueError("E13c source config path does not match E14")
    if observed_source_contract.get("source_config_sha256") != file_sha256(
        str(dependency["e13b_config_path"])
    ):
        raise ValueError("E13c source config file hash does not match E14")
    summary = report.get("summary", {})
    expected_paired_cells = len(seeds) * len(updates) * len(gaps)
    if (
        int(summary.get("paired_seeds", -1)) != len(seeds)
        or int(summary.get("paired_cells", -1)) != expected_paired_cells
    ):
        raise ValueError("E13c paired seed/cell summary is incomplete")

    provenance_rows = _read_jsonl(provenance_path)
    if report.get("source_runs") != provenance_rows:
        raise ValueError(
            "E13c report source_runs differ from sealed provenance JSONL"
        )
    aggregate_calibration = report.get("calibration_dependency")
    if dry_run:
        if aggregate_calibration is not None:
            raise ValueError(
                "E13c dry-run must not record a main calibration dependency"
            )
    elif not isinstance(aggregate_calibration, dict):
        raise ValueError("E13c aggregate lacks its calibration dependency")
    expected_keys = set(product(seeds, FIXED_VARIANTS))
    observed: dict[tuple[int, str], SealedCheckpoint | None] = {}
    for raw in provenance_rows:
        key, selected = _validate_source_record(
            raw,
            artifact_root=root,
            e13b_experiment_id=e13b_experiment_id,
            calibration_experiment_id=calibration_experiment_id,
            calibration_gate_field=calibration_gate_field,
            expected_e13b_config=expected_e13b_config,
            expected_e13b_config_file_sha256=(
                expected_e13b_config_file_sha256
            ),
            expected_run_mode=expected_run_mode,
            dry_run=dry_run,
        )
        source_run = _contained_path(
            (root / e13b_experiment_id).resolve(),
            str(raw["run_dir"]),
            label=f"E13b source {key} run",
        )
        source_report = _read_json(source_run / "report.json")
        if not dry_run and source_report.get(
            "calibration_dependency"
        ) != aggregate_calibration:
            raise ValueError(
                "E13b source calibration differs from E13c aggregate seal"
            )
        if key in observed:
            raise ValueError(f"Duplicate E13c source provenance key: {key}")
        observed[key] = selected
    if set(observed) != expected_keys:
        missing = sorted(expected_keys - set(observed))
        extra = sorted(set(observed) - expected_keys)
        raise ValueError(
            f"E13c provenance key mismatch: missing={missing}, extra={extra}"
        )

    selected_checkpoints = [
        observed[(seed, REQUIRED_VARIANT)]
        for seed in seeds
    ]
    if any(checkpoint is None for checkpoint in selected_checkpoints):
        raise ValueError("E13c provenance did not seal every dual checkpoint")
    sealed = [
        checkpoint
        for checkpoint in selected_checkpoints
        if checkpoint is not None
    ]
    if [checkpoint.seed for checkpoint in sealed] != list(seeds):
        raise ValueError("Selected dual checkpoint seed order is incomplete")

    dependency_record = {
        "e13c_experiment_id": e13c_experiment_id,
        "e13b_experiment_id": e13b_experiment_id,
        "calibration_experiment_id": calibration_experiment_id,
        "calibration_gate_field": calibration_gate_field,
        "calibration_dependency": aggregate_calibration,
        "e13c_run_dir": str(e13c_run),
        "e13c_report_path": str(report_path.resolve()),
        "e13c_report_sha256": file_sha256(report_path),
        "e13c_manifest_path": str(manifest_path.resolve()),
        "e13c_manifest_sha256": file_sha256(manifest_path),
        "e13c_source_provenance_path": str(provenance_path.resolve()),
        "e13c_source_provenance_sha256": file_sha256(provenance_path),
        "e13c_config_path": str(
            Path(str(dependency["e13c_config_path"])).resolve()
        ),
        "e13c_config_sha256": file_sha256(
            str(dependency["e13c_config_path"])
        ),
        "e13b_config_path": str(
            Path(str(dependency["e13b_config_path"])).resolve()
        ),
        "e13b_config_sha256": file_sha256(
            str(dependency["e13b_config_path"])
        ),
        "required_seeds": list(seeds),
        "required_variant": REQUIRED_VARIANT,
        "required_updates": list(updates),
        "required_gap_events": list(gaps),
        "selected_checkpoints": [
            checkpoint.as_dict() for checkpoint in sealed
        ],
    }
    return dependency_record, sealed


def assess_plan_gate(
    rows: list[dict[str, Any]],
    *,
    required_seeds: tuple[int, ...],
    required_updates: tuple[int, ...],
    required_gaps: tuple[int, ...],
    minimum_affected_gain: float,
    maximum_retention_mse: float,
    dry_run: bool,
) -> dict[str, Any]:
    if (
        not math.isfinite(minimum_affected_gain)
        or minimum_affected_gain < 0.0
        or not math.isfinite(maximum_retention_mse)
        or maximum_retention_mse < 0.0
    ):
        raise ValueError("E14 claim thresholds must be finite and nonnegative")
    expected_keys = set(
        product(required_seeds, required_updates, required_gaps)
    )
    observed: dict[tuple[int, int, int], dict[str, Any]] = {}
    metric_names = (
        "stale_plan_mse",
        "assimilated_plan_mse",
        "plan_correction_gain",
        "affected_stale_plan_mse",
        "affected_plan_mse",
        "affected_plan_correction_gain",
        "unaffected_plan_retention_mse",
        "forward_seconds_per_batch",
    )
    for row in rows:
        key = (
            int(row["training_seed"]),
            int(row["updates"]),
            int(row["gap_events"]),
        )
        if key in observed:
            raise ValueError(f"Duplicate E14 seed/cell row: {key}")
        if str(row["variant"]) != REQUIRED_VARIANT:
            raise ValueError(f"E14 row has non-dual variant: {key}")
        if int(row["affected_entity_count"]) <= 0:
            raise ValueError(f"E14 row has no affected entities: {key}")
        if int(row["unaffected_entity_count"]) <= 0:
            raise ValueError(f"E14 row has no unaffected entities: {key}")
        for metric_name in metric_names:
            if not math.isfinite(float(row[metric_name])):
                raise ValueError(
                    f"E14 row has non-finite {metric_name}: {key}"
                )
        digest = str(row["base_transaction_digest"])
        if len(digest) != 64 or any(
            character not in "0123456789abcdef"
            for character in digest
        ):
            raise ValueError(
                f"E14 row has invalid base transaction digest: {key}"
            )
        observed[key] = row
    if set(observed) != expected_keys:
        missing = sorted(expected_keys - set(observed))
        extra = sorted(set(observed) - expected_keys)
        raise ValueError(
            f"E14 seed/cell contract mismatch: missing={missing}, extra={extra}"
        )

    seed_all_cells_pass: dict[str, bool] = {}
    seed_mean_affected_gain: dict[str, float] = {}
    seed_max_retention_mse: dict[str, float] = {}
    cells_passing = 0
    for seed in required_seeds:
        seed_rows = [
            observed[(seed, updates, gap)]
            for updates, gap in product(required_updates, required_gaps)
        ]
        cell_passes = [
            float(row["affected_plan_correction_gain"])
            >= minimum_affected_gain
            and float(row["unaffected_plan_retention_mse"])
            <= maximum_retention_mse
            for row in seed_rows
        ]
        cells_passing += sum(cell_passes)
        seed_all_cells_pass[str(seed)] = all(cell_passes)
        seed_mean_affected_gain[str(seed)] = sum(
            float(row["affected_plan_correction_gain"])
            for row in seed_rows
        ) / len(seed_rows)
        seed_max_retention_mse[str(seed)] = max(
            float(row["unaffected_plan_retention_mse"])
            for row in seed_rows
        )
    all_cells_pass = cells_passing == len(expected_keys)
    gap_pairing_ok = all(
        len(
            {
                str(
                    observed[(seed, updates, gap)][
                        "base_transaction_digest"
                    ]
                )
                for seed in required_seeds
                for gap in required_gaps
            }
        )
        == 1
        for updates in required_updates
    )
    if not gap_pairing_ok:
        raise ValueError(
            "E14 base transaction digests differ across gaps/checkpoints"
        )
    seed_gain_values = [
        seed_mean_affected_gain[str(seed)] for seed in required_seeds
    ]
    return {
        "required_seed_count": len(required_seeds),
        "required_cell_count": len(expected_keys),
        "observed_seed_count": len(
            {int(row["training_seed"]) for row in rows}
        ),
        "observed_cell_count": len(rows),
        "cells_passing": cells_passing,
        "all_cells_pass": all_cells_pass,
        "seed_all_cells_pass": seed_all_cells_pass,
        "seed_mean_affected_gain": seed_mean_affected_gain,
        "seed_max_retention_mse": seed_max_retention_mse,
        "seed_level_gain_sign_flip_p": exact_sign_flip(
            seed_gain_values,
            alternative="greater",
        ),
        "seed_level_minimum_exact_p": 1.0 / (2 ** len(required_seeds)),
        "batch_or_episode_used_as_inference_unit": False,
        "all_seeds_pass": all(seed_all_cells_pass.values()),
        "base_transaction_pairing_ok": gap_pairing_ok,
        "supported": bool(not dry_run and all_cells_pass),
    }


def _evaluation_batch_seed(
    base_seed: int,
    updates: int,
    batch_index: int,
) -> int:
    return int(base_seed + 100_000 * updates + batch_index)


def _load_model(
    checkpoint: SealedCheckpoint,
    *,
    device: torch.device,
) -> TransactionalSequenceMemoryV2:
    if file_sha256(checkpoint.checkpoint_path) != (
        checkpoint.checkpoint_sha256
    ):
        raise RuntimeError(
            f"Sealed checkpoint changed before model load: "
            f"{checkpoint.checkpoint_path}"
        )
    payload = torch.load(
        checkpoint.checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if file_sha256(checkpoint.checkpoint_path) != (
        checkpoint.checkpoint_sha256
    ):
        raise RuntimeError(
            f"Sealed checkpoint changed during model load: "
            f"{checkpoint.checkpoint_path}"
        )
    if payload.get("variant") != checkpoint.variant:
        raise ValueError(
            f"Checkpoint variant changed: {checkpoint.checkpoint_path}"
        )
    if int(payload.get("seed", -1)) != checkpoint.seed:
        raise ValueError(
            f"Checkpoint seed changed: {checkpoint.checkpoint_path}"
        )
    if payload.get("config") != checkpoint.checkpoint_config:
        raise ValueError(
            f"Checkpoint config changed: {checkpoint.checkpoint_path}"
        )
    source_config = checkpoint.checkpoint_config
    model = TransactionalSequenceMemoryV2(
        control=SequenceControlV2(checkpoint.variant),
        num_entities=int(source_config["data"]["num_entities"]),
        value_vocab=int(source_config["data"]["value_vocab"]),
        embedding_dim=int(source_config["model"]["embedding_dim"]),
        hidden_dim=int(source_config["model"]["hidden_dim"]),
    )
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()
    return model


def assert_dependency_unchanged(
    dependency: dict[str, Any],
    checkpoints: list[SealedCheckpoint],
) -> None:
    dependency_files = (
        ("e13c_report_path", "e13c_report_sha256"),
        ("e13c_manifest_path", "e13c_manifest_sha256"),
        (
            "e13c_source_provenance_path",
            "e13c_source_provenance_sha256",
        ),
        ("e13c_config_path", "e13c_config_sha256"),
        ("e13b_config_path", "e13b_config_sha256"),
    )
    for path_key, hash_key in dependency_files:
        path = Path(str(dependency[path_key]))
        if not path.is_file() or file_sha256(path) != str(
            dependency[hash_key]
        ):
            raise RuntimeError(f"E14 dependency changed during run: {path}")
    for checkpoint in checkpoints:
        sealed_files = (
            (
                checkpoint.checkpoint_path,
                checkpoint.checkpoint_sha256,
            ),
            (
                checkpoint.source_report_path,
                checkpoint.source_report_sha256,
            ),
            (
                checkpoint.source_metrics_path,
                checkpoint.source_metrics_sha256,
            ),
            (
                checkpoint.source_manifest_path,
                checkpoint.source_manifest_sha256,
            ),
        )
        for path, digest in sealed_files:
            if not path.is_file() or file_sha256(path) != digest:
                raise RuntimeError(
                    f"E14 sealed source changed during run: {path}"
                )


def main() -> None:
    parser = build_parser(EXPERIMENT_ID, DEFAULT_CONFIG)
    args = parser.parse_args()
    config, run_dir, device = initialize_run(
        experiment_id=EXPERIMENT_ID,
        config_path=args.config,
        artifact_root=args.artifact_root,
        device_request=args.device,
        run_mode="DRY_RUN" if args.dry_run else "MAIN",
    )
    seeds, update_values, gap_values = _contract_axes(
        config,
        dry_run=args.dry_run,
    )
    dependency_record, checkpoints = resolve_e13c_dependency(
        artifact_root=args.artifact_root,
        config=config,
        dry_run=args.dry_run,
    )

    batches = int(config["evaluation"]["batches"])
    batch_size = int(config["evaluation"]["batch_size"])
    if args.dry_run:
        batches = 2
        batch_size = min(batch_size, 8)
    minimum_affected_gain = float(
        config["claim_gate"]["minimum_affected_plan_correction_gain"]
    )
    retention_margin = float(
        config["claim_gate"]["maximum_retention_mse"]
    )

    rows: list[dict[str, float | int | str | bool]] = []
    for checkpoint in checkpoints:
        model = _load_model(checkpoint, device=device)
        source_config = checkpoint.checkpoint_config
        for updates, gap in product(update_values, gap_values):
            stale_total = 0.0
            model_total = 0.0
            affected_stale_total = 0.0
            affected_total = 0.0
            retention_total = 0.0
            elapsed = 0.0
            count = 0
            affected_count = 0
            unaffected_count = 0
            base_digest = hashlib.sha256()
            for batch_index in range(batches):
                evaluation_seed = _evaluation_batch_seed(
                    int(config["evaluation"]["seed"]),
                    updates,
                    batch_index,
                )
                batch = generate_transactional_sequence_batch_v2(
                    batch_size=batch_size,
                    num_entities=int(
                        source_config["data"]["num_entities"]
                    ),
                    value_vocab=int(source_config["data"]["value_vocab"]),
                    updates=updates,
                    gap_events=gap,
                    seed=evaluation_seed,
                    device=device,
                )
                base_digest.update(
                    base_transaction_digest_v2(batch).encode()
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = perf_counter()
                with torch.no_grad():
                    prediction = model(
                        sequence_model_input_v2(batch)
                    ).state
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed += perf_counter() - started
                stale_error = (
                    (batch.inputs.initial_state - batch.target_state)
                    .square()
                    .mean(dim=-1)
                )
                model_error = (
                    (prediction - batch.target_state)
                    .square()
                    .mean(dim=-1)
                )
                stale_total += float(stale_error.mean()) * batch_size
                model_total += float(model_error.mean()) * batch_size
                affected_mask = batch.affected_entities
                unaffected_mask = ~affected_mask
                affected_stale_total += float(
                    stale_error[affected_mask].sum()
                )
                affected_total += float(model_error[affected_mask].sum())
                retention_total += float(model_error[unaffected_mask].sum())
                affected_count += int(affected_mask.sum().item())
                unaffected_count += int(unaffected_mask.sum().item())
                count += batch_size
            if affected_count <= 0 or unaffected_count <= 0:
                raise RuntimeError(
                    f"Empty E14 denominator for seed={checkpoint.seed}, "
                    f"updates={updates}, gap={gap}"
                )
            plan_gain = (stale_total - model_total) / count
            affected_gain = (
                affected_stale_total - affected_total
            ) / affected_count
            retention_mse = retention_total / unaffected_count
            rows.append(
                {
                    "variant": checkpoint.variant,
                    "training_seed": checkpoint.seed,
                    "evaluation_seed_base": int(
                        config["evaluation"]["seed"]
                    ),
                    "evaluation_seed_rule": (
                        "base + 100000 * updates + batch_index"
                    ),
                    "checkpoint": str(checkpoint.checkpoint_path),
                    "checkpoint_sha256": checkpoint.checkpoint_sha256,
                    "source_e13b_run_dir": str(checkpoint.source_run_dir),
                    "updates": updates,
                    "gap_events": gap,
                    "base_transaction_digest": base_digest.hexdigest(),
                    "stale_plan_mse": stale_total / count,
                    "assimilated_plan_mse": model_total / count,
                    "plan_correction_gain": plan_gain,
                    "affected_stale_plan_mse": (
                        affected_stale_total / affected_count
                    ),
                    "affected_plan_mse": affected_total / affected_count,
                    "affected_plan_correction_gain": affected_gain,
                    "unaffected_plan_retention_mse": retention_mse,
                    "affected_entity_count": affected_count,
                    "unaffected_entity_count": unaffected_count,
                    "forward_seconds_per_batch": elapsed / batches,
                    "cell_gate_pass": bool(
                        affected_gain >= minimum_affected_gain
                        and retention_mse <= retention_margin
                    ),
                    "structured_proxy_only": True,
                }
            )

    gate_summary = assess_plan_gate(
        rows,
        required_seeds=seeds,
        required_updates=update_values,
        required_gaps=gap_values,
        minimum_affected_gain=minimum_affected_gain,
        maximum_retention_mse=retention_margin,
        dry_run=args.dry_run,
    )
    assert_dependency_unchanged(dependency_record, checkpoints)
    report = {
        "status": "DRY_RUN" if args.dry_run else "PASS",
        "run_scope": "STRUCTURED_ENTITY_VALUE_PLAN_PROXY",
        "evidence_scope": {
            **STATIC_EVIDENCE_BOUNDARY,
            "structured_proxy_claim_eligible": bool(
                gate_summary["supported"]
            ),
            "structured_gap_integrity_claim_eligible": bool(
                gate_summary["supported"]
            ),
            "untouched_retention_role": (
                "LEARNED_NOOP_PATH_GUARDRAIL_WITH_ORACLE_ADDRESS"
            ),
        },
        "dependency": dependency_record,
        "design_guardrails": {
            **gate_summary,
            "same_evaluation_stream_across_training_seeds": True,
            "evaluation_seed_rule": (
                "base + 100000 * updates + batch_index; gap excluded"
            ),
            "first_batch_seed_by_updates": {
                str(updates): _evaluation_batch_seed(
                    int(config["evaluation"]["seed"]),
                    updates,
                    0,
                )
                for updates in update_values
            },
            "base_transaction_pairing_across_gaps_required": True,
            "all_five_training_seeds_required": not args.dry_run,
            "full_update_gap_grid_required": not args.dry_run,
        },
        "claim_gate": {
            "supported": bool(gate_summary["supported"]),
            "primary_estimand": "affected_plan_correction_gain",
            "whole_table_plan_correction_gain_is_descriptive": True,
            "minimum_affected_plan_correction_gain": (
                minimum_affected_gain
            ),
            "maximum_retention_mse": retention_margin,
            "allowed_claim": (
                "Across the sealed five-seed E13b dual checkpoints, the "
                "controller corrects stale affected fields while preserving "
                "untouched fields in the tested structured synthetic "
                "entity-value continuation proxy."
            ),
            "forbidden_claim": (
                "Independent plan semantics, natural-language planning, "
                "general agent planning, tool orchestration, or production "
                "system break-even."
            ),
        },
    }
    write_jsonl(run_dir / "plan_continuation_metrics.jsonl", rows)
    write_jsonl(
        run_dir / "sealed_checkpoint_provenance.jsonl",
        [checkpoint.as_dict() for checkpoint in checkpoints],
    )
    finalize_run(
        experiment_id=EXPERIMENT_ID,
        artifact_root=args.artifact_root,
        run_dir=run_dir,
        report=report,
    )
    print(f"[{EXPERIMENT_ID}] {report['status']}: {run_dir}")


if __name__ == "__main__":
    main()
