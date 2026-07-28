from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from catena.core.config import load_config
from catena.core.provenance_v61 import (
    ManifestValidationRequirements,
    ProvenanceValidationError,
    ValidatedRun,
    sha256_canonical_json,
    sha256_file,
    validate_run_manifest,
    write_jsonl_strict,
)
from catena.core.schema import CandidateMode, MemoryEpisode, Operation
from catena.data.geometry_sweep import build_geometry_episode, controller_features
from catena.eval.metrics import EpisodeMetrics, evaluate_episode
from catena.eval.seed_inference import exact_sign_flip_test
from catena.eval.statistics_v61 import (
    Interval,
    fixed_seed_operation_stratified_bootstrap,
)
from catena.models.matched_controllers import MatchedScalarController, ScalarConstraint
from catena.models.memory import apply_scalar_update
from experiments.common import build_parser
from experiments.v61_common import (
    finalize_v61_run,
    initialize_v61_run,
    validate_legacy_e00,
)

EXPERIMENT_ID = "e02b_prospective_absolute_supersede"
DEFAULT_CONFIG = "configs/e02b_prospective_absolute_supersede.yaml"
REPO_ROOT = Path(__file__).resolve().parents[1]

_GEOMETRY_PROTOCOL_LABEL = (
    "frozen-controller unseen norm/angle OOD heldout extension"
)
_REPAIR_PROTOCOL_LABEL = (
    "prospective absolute-SUPERSEDE repair plus OOD geometry extension"
)
_PINNED_CONFIG_CANONICAL_SHA256 = (
    "9b7d299a916003a9d4b5038a8db325d4609d618b3d9919ba7b603cdf8d76f781"
)
_PINNED_CONFIG_FILE_SHA256 = (
    "4e269572652f494977f0370a82c5104bd083ae4e7a4f570635d4685af15fe956"
)
_PINNED_AMENDMENT_SHA256 = (
    "e42b71beb4512995e621000beb7522be93c939ce74b839c54f9aee8e63a2fd03"
)
_PINNED_AMENDMENT_LOCK_SHA256 = (
    "14f8a9d546d3a84b0d47039a2c29d576b277bc65e305d23c455f68b2ed7d64e6"
)
_PINNED_SOURCE: dict[str, object] = {
    "experiment_id": "e02_magnitude_factorization",
    "run_id": "20260726T153504.455509Z",
    "manifest_sha256": (
        "1ea4ad867ffbc86bca3f4ee8f3eceb698089f973db250920a0eeb3dc39641d4c"
    ),
    "report_sha256": (
        "f3df03e231598d6eda11ebf71825ab418cc9a59ac9a96a299caff617291e4211"
    ),
    "episode_metrics_sha256": (
        "9a1f85d4c18366f062caceec47b83d16763dffcbdeaef305b3ecc2010878c5fd"
    ),
    "config_sha256": (
        "b3629fd646057a87c85fc4e3d305c9c80b4a40fc74f1f4567e74eb94b7229191"
    ),
    "config_file_sha256": (
        "3594de2d53173785784f4424a567c26b233219eeeeb5e08b1d3ed09176307609"
    ),
    "source_fingerprint_sha256": (
        "9a9730217cd02e55498f901477372563035fa9ca187fd83a5a28da67bf098356"
    ),
    "source_fingerprint_files": 109,
    "strict_checkpoint_contract_sha256": (
        "211792700b7ed37c45519c88c73bb4a083898b8113cc44392bf5000789666982"
    ),
    "inherited_tuning_gate_sha256": (
        "72e3b5b28e4db9a232a0ee879b8f3c5f45f804ecdf4ec23d4ad5269c213ce89f"
    ),
}
_EXPECTED_SEEDS = (11, 22, 33, 44, 55, 66, 77, 88)
_EXPECTED_EXCLUDED_RANGES = (
    ("e02_train", 0, 2047),
    ("e02_validation", 25000, 26023),
    ("e02_test", 50000, 52047),
    ("e04_heldout", 75000, 75127),
)
_EXPECTED_NORM_PAIRS = (
    (0.75, 0.90),
    (0.90, 1.25),
    (1.10, 0.80),
    (1.25, 1.10),
)
_EXPECTED_ANGLES_DEGREES = (45.0, 75.0, 105.0, 135.0)
_EXPECTED_DRY_RUN_CELL_IDS = tuple(range(16))
_EXPECTED_REPEATS_PER_CELL = 32
_GATE_NAMES = (
    "asymmetric_normalized_gain",
    "preserve_raw_equivalence",
    "supersede_raw_equivalence",
    "positive_raw_operation_interaction",
    "retention_noninferiority",
    "inherited_tuning_direction",
)
_SIGN_FLIP_GATE_NAMES = (
    "asymmetric_normalized_gain",
    "positive_raw_operation_interaction",
    "retention_noninferiority",
)
_EQUIVALENCE_GATE_NAMES = (
    "preserve_raw_equivalence",
    "supersede_raw_equivalence",
)
_ORIGINAL_E02_ADJUDICATION: dict[str, object] = {
    "original_confirmatory_status": "INCONCLUSIVE",
    "original_inconclusive_reason": (
        "PREREGISTERED_SYMMETRIC_RELATIVE_GATE_UNIDENTIFIABLE"
    ),
    "original_evaluable_gates_passed": "5/5",
    "original_h2_claim_open": False,
}


@dataclass(frozen=True, slots=True)
class FrozenCheckpoint:
    seed: int
    constraint: str
    path: Path
    file_sha256: str
    state_dict_sha256: str
    initial_state_sha256: str
    parameter_count: int
    input_dim: int
    hidden_dim: int

    def report_record(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "constraint": self.constraint,
            "source_filename": self.path.name,
            "source_file_sha256": self.file_sha256,
            "state_dict_sha256": self.state_dict_sha256,
            "initial_state_sha256": self.initial_state_sha256,
            "parameter_count": self.parameter_count,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "recipe": "strict",
        }


@dataclass(frozen=True, slots=True)
class PinnedE02Source:
    run: ValidatedRun
    checkpoints: dict[tuple[int, str], FrozenCheckpoint]
    inherited_tuning_values: dict[int, float]

    def dependency_record(self) -> dict[str, Any]:
        return {
            **self.run.dependency_record(),
            "evidence_role": "immutable_inconclusive_e02_checkpoint_source",
            "strict_checkpoint_contract_sha256": _PINNED_SOURCE[
                "strict_checkpoint_contract_sha256"
            ],
            "inherited_tuning_gate_sha256": _PINNED_SOURCE[
                "inherited_tuning_gate_sha256"
            ],
            "original_claim_supported": False,
            "checkpoint_recipe": "strict",
            "checkpoint_pair_count": len(self.checkpoints) // 2,
        }


@dataclass(frozen=True, slots=True)
class HeldoutEpisode:
    geometry_seed: int
    cell_id: int
    norm_pair_bin: int
    angle_bin: int
    repeat_id: int
    old_scale: float
    new_scale: float
    angle_degrees: float
    old_new_cosine: float
    episode: MemoryEpisode


@dataclass(frozen=True, slots=True)
class GeometryCell:
    cell_id: int
    norm_pair_bin: int
    angle_bin: int
    old_scale: float
    new_scale: float
    angle_degrees: float
    old_new_cosine: float

    @property
    def label(self) -> str:
        return f"cell_{self.cell_id:02d}"

    def report_record(self) -> dict[str, object]:
        return {
            "cell_id": self.cell_id,
            "cell_label": self.label,
            "norm_pair_bin": self.norm_pair_bin,
            "angle_bin": self.angle_bin,
            "old_scale": self.old_scale,
            "new_scale": self.new_scale,
            "angle_degrees": self.angle_degrees,
            "old_new_cosine": self.old_new_cosine,
            "unseen_vs_e02_original": True,
        }


def _require_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ProvenanceValidationError(f"{name} must be an object.")
    return value


def _require_list(value: object, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ProvenanceValidationError(f"{name} must be an array.")
    return value


def _require_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ProvenanceValidationError(f"{name} must be an integer.")
    return int(value)


def _require_finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError(f"{name} must be numeric.")
    result = float(value)
    if not np.isfinite(result):
        raise FloatingPointError(f"{name} is non-finite: {result}")
    return result


def _finite_vector(
    values: object,
    name: str,
    *,
    allow_empty: bool = False,
) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got {result.shape}.")
    if not allow_empty and len(result) == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.isfinite(result).all():
        raise FloatingPointError(f"{name} contains a non-finite value.")
    return cast(np.ndarray, result)


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


def _count_jsonl_rows(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def _direct_hashed_file(run_dir: Path, filename: object, expected_hash: object) -> Path:
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ProvenanceValidationError(f"Unsafe source artifact filename: {filename!r}.")
    if not isinstance(expected_hash, str):
        raise ProvenanceValidationError(f"Missing SHA-256 for source artifact {filename}.")
    path = run_dir / filename
    if path.is_symlink() or not path.is_file():
        raise ProvenanceValidationError(f"Missing direct source artifact: {path}.")
    if sha256_file(path) != expected_hash:
        raise ProvenanceValidationError(f"Source artifact hash mismatch: {path}.")
    return path


def _validate_protocol_config(config: Mapping[str, Any], config_path: Path) -> None:
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("E02b config experiment_id is not prospectively locked.")
    if sha256_canonical_json(dict(config)) != _PINNED_CONFIG_CANONICAL_SHA256:
        raise ProvenanceValidationError("E02b canonical config differs from the locked protocol.")
    if sha256_file(config_path) != _PINNED_CONFIG_FILE_SHA256:
        raise ProvenanceValidationError("E02b config file bytes differ from the locked protocol.")
    source = _require_mapping(config.get("source_e02"), "config.source_e02")
    if dict(source) != _PINNED_SOURCE:
        raise ProvenanceValidationError("E02b source dependency is not the pinned E02 run.")
    protocol = _require_mapping(config.get("protocol"), "config.protocol")
    if protocol.get("geometry_design") != _GEOMETRY_PROTOCOL_LABEL:
        raise ProvenanceValidationError("E02b OOD geometry protocol label differs.")
    if protocol.get("repair_scope") != _REPAIR_PROTOCOL_LABEL:
        raise ProvenanceValidationError("E02b repair-scope protocol label differs.")
    amendment = REPO_ROOT / "docs/E02B_PROTOCOL_AMENDMENT_FROZEN_KO.md"
    amendment_lock = REPO_ROOT / "docs/E02B_PROTOCOL_AMENDMENT_LOCK_KO.md"
    if (
        not amendment.is_file()
        or amendment.is_symlink()
        or sha256_file(amendment) != _PINNED_AMENDMENT_SHA256
    ):
        raise ProvenanceValidationError("E02b frozen amendment hash differs.")
    if (
        not amendment_lock.is_file()
        or amendment_lock.is_symlink()
        or sha256_file(amendment_lock) != _PINNED_AMENDMENT_LOCK_SHA256
    ):
        raise ProvenanceValidationError("E02b amendment lock hash differs.")


def _validate_original_e02_state(report: Mapping[str, Any]) -> None:
    execution = _require_mapping(report.get("execution"), "E02 report.execution")
    integrity = _require_mapping(
        report.get("execution_integrity"),
        "E02 report.execution_integrity",
    )
    claim = _require_mapping(report.get("claim_gate"), "E02 report.claim_gate")
    symmetric = _require_mapping(
        report.get("symmetric_guardrails"),
        "E02 report.symmetric_guardrails",
    )
    preserve = _require_mapping(
        symmetric.get("preserve_raw_effect"),
        "E02 report.symmetric_guardrails.preserve_raw_effect",
    )
    supersede = _require_mapping(
        symmetric.get("supersede_relative_effect"),
        "E02 report.symmetric_guardrails.supersede_relative_effect",
    )
    if (
        execution.get("dry_run") is not False
        or execution.get("row_count") != 16384
        or execution.get("strict_checkpoint_count") != 16
        or execution.get("tuning_executed") is not True
        or integrity.get("exact_eight_paired_seed_design") is not True
        or integrity.get("actual_episode_metric_rows") != 16384
        or claim.get("supported") is not False
        or claim.get("requires_asymmetric_gain") is not True
        or claim.get("requires_positive_interaction") is not True
        or claim.get("requires_retention_noninferiority") is not True
        or claim.get("requires_tuned_direction_consistency") is not True
        or claim.get("requires_symmetric_equivalence") is not False
        or preserve.get("equivalent") is not True
        or supersede.get("episode_ci_evaluable") is not False
        or supersede.get("equivalent") is not False
    ):
        raise ProvenanceValidationError(
            "Pinned E02 no longer has the expected complete-but-inconclusive state."
        )


def _validate_inherited_tuning_gate(
    report: Mapping[str, Any],
) -> dict[int, float]:
    robustness = _require_mapping(
        report.get("optimization_robustness"),
        "E02 report.optimization_robustness",
    )
    expected_hash = str(_PINNED_SOURCE["inherited_tuning_gate_sha256"])
    if sha256_canonical_json(dict(robustness)) != expected_hash:
        raise ProvenanceValidationError("E02 inherited tuning record hash mismatch.")
    values = _require_mapping(
        robustness.get("tuned_add_invalidate_normalized_gain_by_seed"),
        "E02 tuned direction values",
    )
    records = _require_mapping(
        robustness.get("equal_budget_records_by_seed"),
        "E02 equal-budget records",
    )
    expected_keys = {str(seed) for seed in _EXPECTED_SEEDS}
    if (
        robustness.get("strict_recipe_is_confirmatory") is not True
        or robustness.get("tuning_executed") is not True
        or robustness.get("direction_consistent") is not True
        or set(values) != expected_keys
        or set(records) != expected_keys
    ):
        raise ProvenanceValidationError("E02 inherited tuning gate is incomplete.")
    result = {
        seed: _require_finite(values[str(seed)], f"E02 tuned direction seed {seed}")
        for seed in _EXPECTED_SEEDS
    }
    if not all(value > 0.0 for value in result.values()):
        raise ProvenanceValidationError("E02 inherited tuning direction is not 8/8 positive.")
    return result


def _validate_checkpoint_contract(
    run: ValidatedRun,
    report: Mapping[str, Any],
) -> dict[tuple[int, str], FrozenCheckpoint]:
    contract = _require_mapping(
        report.get("strict_checkpoint_contract"),
        "E02 report.strict_checkpoint_contract",
    )
    expected_contract_hash = str(_PINNED_SOURCE["strict_checkpoint_contract_sha256"])
    if sha256_canonical_json(dict(contract)) != expected_contract_hash:
        raise ProvenanceValidationError("E02 strict checkpoint contract hash mismatch.")
    entries = _require_list(contract.get("entries"), "E02 checkpoint entries")
    expected_keys = {
        (seed, constraint)
        for seed in _EXPECTED_SEEDS
        for constraint in ("tied", "dual")
    }
    checkpoints: dict[tuple[int, str], FrozenCheckpoint] = {}
    for raw_entry in entries:
        entry = _require_mapping(raw_entry, "E02 checkpoint entry")
        seed = _require_int(entry.get("seed"), "E02 checkpoint seed")
        constraint = str(entry.get("constraint"))
        key = (seed, constraint)
        expected_filename = f"seed{seed}_{constraint}.pt"
        if (
            key not in expected_keys
            or key in checkpoints
            or entry.get("filename") != expected_filename
            or entry.get("recipe") != "strict"
            or entry.get("round_trip_match") is not True
            or entry.get("candidate_mode") != CandidateMode.ORACLE.value
            or entry.get("input_dim") != 10
            or entry.get("hidden_dim") != 64
            or entry.get("parameter_count") != 4994
        ):
            raise ProvenanceValidationError(f"Invalid E02 checkpoint entry: {entry}.")
        checkpoint_path = _direct_hashed_file(
            run.run_dir,
            entry.get("filename"),
            entry.get("sha256"),
        )
        file_hash = str(entry.get("sha256"))
        state_hash = str(entry.get("state_dict_sha256"))
        initial_hash = str(entry.get("initial_state_sha256"))
        if not all(len(value) == 64 for value in (file_hash, state_hash, initial_hash)):
            raise ProvenanceValidationError("E02 checkpoint entry has an invalid state hash.")
        checkpoints[key] = FrozenCheckpoint(
            seed=seed,
            constraint=constraint,
            path=checkpoint_path,
            file_sha256=file_hash,
            state_dict_sha256=state_hash,
            initial_state_sha256=initial_hash,
            parameter_count=int(entry["parameter_count"]),
            input_dim=int(entry["input_dim"]),
            hidden_dim=int(entry["hidden_dim"]),
        )
    if set(checkpoints) != expected_keys or len(checkpoints) != 16:
        raise ProvenanceValidationError("E02 lacks exactly eight strict checkpoint pairs.")
    actual_pt_names = sorted(path.name for path in run.run_dir.glob("*.pt"))
    expected_pt_names = sorted(checkpoint.path.name for checkpoint in checkpoints.values())
    if actual_pt_names != expected_pt_names:
        raise ProvenanceValidationError(
            "E02 checkpoint directory differs from its pinned strict-only contract."
        )
    for seed in _EXPECTED_SEEDS:
        tied = checkpoints[(seed, "tied")]
        dual = checkpoints[(seed, "dual")]
        if tied.initial_state_sha256 != dual.initial_state_sha256:
            raise ProvenanceValidationError(
                f"E02 seed {seed} tied/dual checkpoints do not share initialization."
            )
    return checkpoints


def validate_pinned_e02_source(
    artifact_root: str | Path,
    config: Mapping[str, Any],
) -> PinnedE02Source:
    source = _require_mapping(config.get("source_e02"), "config.source_e02")
    if dict(source) != _PINNED_SOURCE:
        raise ProvenanceValidationError("E02b source dependency is not exactly pinned.")
    root = Path(artifact_root).expanduser().resolve(strict=True)
    run_dir = root / str(source["experiment_id"]) / str(source["run_id"])
    validated = validate_run_manifest(
        run_dir,
        root,
        requirements=ManifestValidationRequirements(
            expected_experiment_id=str(source["experiment_id"]),
            accepted_schema_versions=frozenset({1}),
            expected_source_sha256=str(source["source_fingerprint_sha256"]),
            expected_source_files=int(source["source_fingerprint_files"]),
            expected_run_mode="main",
            require_main_eligible=False,
            require_full_eligible=False,
        ),
    )
    if (
        validated.manifest_sha256 != source["manifest_sha256"]
        or validated.report_sha256 != source["report_sha256"]
        or validated.config_sha256 != source["config_sha256"]
        or validated.config_file_sha256 != source["config_file_sha256"]
        or validated.main_eligible
        or validated.full_eligible
    ):
        raise ProvenanceValidationError("Pinned E02 core provenance fields do not match.")
    _validate_original_e02_state(validated.report)
    artifact = _require_mapping(
        _require_mapping(validated.report.get("artifacts"), "E02 report.artifacts").get(
            "episode_metrics"
        ),
        "E02 report.artifacts.episode_metrics",
    )
    episode_metrics = _direct_hashed_file(
        validated.run_dir,
        artifact.get("path"),
        source["episode_metrics_sha256"],
    )
    if (
        artifact.get("sha256") != source["episode_metrics_sha256"]
        or artifact.get("rows") != 16384
        or _count_jsonl_rows(episode_metrics) != 16384
    ):
        raise ProvenanceValidationError("Pinned E02 episode metric artifact is incomplete.")
    checkpoints = _validate_checkpoint_contract(validated, validated.report)
    tuning_values = _validate_inherited_tuning_gate(validated.report)
    return PinnedE02Source(validated, checkpoints, tuning_values)


def _validate_e00_dependency(
    artifact_root: str | Path,
    *,
    dry_run: bool,
) -> dict[str, Any]:
    return validate_legacy_e00(
        artifact_root,
        require_full=not dry_run,
    )


def _geometry_cells(config: Mapping[str, Any]) -> tuple[GeometryCell, ...]:
    grid = _require_mapping(config.get("geometry_grid"), "config.geometry_grid")
    raw_pairs = _require_list(grid.get("norm_pairs"), "geometry_grid.norm_pairs")
    norm_pairs: list[tuple[float, float]] = []
    for pair_index, raw_pair in enumerate(raw_pairs):
        pair = _require_list(raw_pair, f"geometry_grid.norm_pairs[{pair_index}]")
        if len(pair) != 2:
            raise ValueError("Every E02b norm pair must contain exactly two scales.")
        norm_pairs.append(
            (
                _require_finite(pair[0], f"norm pair {pair_index} old scale"),
                _require_finite(pair[1], f"norm pair {pair_index} new scale"),
            )
        )
    angles = tuple(
        _require_finite(value, f"geometry angle {index}")
        for index, value in enumerate(
            _require_list(
                grid.get("angles_degrees"),
                "geometry_grid.angles_degrees",
            )
        )
    )
    if tuple(norm_pairs) != _EXPECTED_NORM_PAIRS:
        raise ValueError("E02b norm-pair grid differs from the prospective lock.")
    if angles != _EXPECTED_ANGLES_DEGREES:
        raise ValueError("E02b angle grid differs from the prospective lock.")
    if len(set(norm_pairs)) != 4 or len(set(angles)) != 4:
        raise ValueError("E02b geometry axes must each contain four unique bins.")
    if any(old <= 0.0 or new <= 0.0 for old, new in norm_pairs):
        raise ValueError("E02b norm scales must be positive.")
    if any(not 0.0 < angle < 180.0 for angle in angles):
        raise ValueError("E02b old/new angles must lie strictly between 0 and 180 degrees.")
    if any(
        np.isclose(old, 1.0, rtol=0.0, atol=1e-12)
        and np.isclose(new, 1.0, rtol=0.0, atol=1e-12)
        for old, new in norm_pairs
    ):
        raise ValueError("E02b norm-pair bins must all be unseen relative to E02.")
    if any(np.isclose(angle, 90.0, rtol=0.0, atol=1e-12) for angle in angles):
        raise ValueError("E02b angle bins must all be unseen relative to E02.")
    excluded = _require_mapping(
        grid.get("excluded_original"),
        "geometry_grid.excluded_original",
    )
    if dict(excluded) != {
        "old_scale": 1.0,
        "new_scale": 1.0,
        "angle_degrees": 90.0,
    }:
        raise ValueError("E02b excluded original geometry is not locked to (1,1,90°).")

    cells: list[GeometryCell] = []
    for norm_pair_bin, (old_scale, new_scale) in enumerate(norm_pairs):
        for angle_bin, angle_degrees in enumerate(angles):
            cell_id = norm_pair_bin * len(angles) + angle_bin
            cells.append(
                GeometryCell(
                    cell_id=cell_id,
                    norm_pair_bin=norm_pair_bin,
                    angle_bin=angle_bin,
                    old_scale=old_scale,
                    new_scale=new_scale,
                    angle_degrees=angle_degrees,
                    old_new_cosine=float(math.cos(math.radians(angle_degrees))),
                )
            )
    if len(cells) != 16 or len({cell.cell_id for cell in cells}) != 16:
        raise AssertionError("E02b must contain exactly 16 unique geometry cells.")
    return tuple(cells)


def _execution_geometry_design(
    config: Mapping[str, Any],
    *,
    dry_run: bool,
) -> tuple[tuple[GeometryCell, ...], int]:
    all_cells = _geometry_cells(config)
    grid = _require_mapping(config.get("geometry_grid"), "config.geometry_grid")
    repeats_per_cell = _require_int(
        grid.get("repeats_per_cell"),
        "geometry_grid.repeats_per_cell",
    )
    dry_repeats = _require_int(
        grid.get("dry_run_repeats_per_cell"),
        "geometry_grid.dry_run_repeats_per_cell",
    )
    dry_ids = tuple(
        _require_int(value, "dry-run geometry cell id")
        for value in _require_list(
            grid.get("dry_run_cell_ids"),
            "geometry_grid.dry_run_cell_ids",
        )
    )
    if repeats_per_cell != _EXPECTED_REPEATS_PER_CELL:
        raise ValueError("E02b main repeats_per_cell differs from the prospective lock.")
    if dry_repeats != 1 or dry_ids != _EXPECTED_DRY_RUN_CELL_IDS:
        raise ValueError("E02b dry-run geometry subset differs from the prospective lock.")
    if dry_run:
        selected = tuple(all_cells[cell_id] for cell_id in dry_ids)
        norm_counts = Counter(cell.norm_pair_bin for cell in selected)
        angle_counts = Counter(cell.angle_bin for cell in selected)
        if set(norm_counts.values()) != {4} or set(angle_counts.values()) != {4}:
            raise AssertionError("E02b dry-run subset must balance both grid axes.")
        return selected, dry_repeats

    expected_count = _require_int(
        _require_mapping(config.get("data"), "config.data").get("count_per_operation"),
        "data.count_per_operation",
    )
    if len(all_cells) * repeats_per_cell != expected_count:
        raise ValueError("E02b grid does not produce 512 main episodes per operation.")
    return all_cells, repeats_per_cell


def _relative_seed_interval(offset: int, count_per_operation: int) -> tuple[int, int]:
    if offset < 0:
        raise ValueError("Seed offset must be non-negative.")
    if count_per_operation <= 0:
        raise ValueError("count_per_operation must be positive.")
    return offset, offset + len(Operation) * count_per_operation - 1


def _ranges_overlap(first: tuple[int, int], second: tuple[int, int]) -> bool:
    return first[0] <= second[1] and second[0] <= first[1]


def _validate_seed_namespace(
    config: Mapping[str, Any],
    *,
    count_per_operation: int,
    dry_run: bool,
) -> tuple[int, dict[str, object]]:
    namespace = _require_mapping(config.get("seed_namespace"), "config.seed_namespace")
    block_size = _require_int(namespace.get("block_size"), "seed block size")
    prospective_offset = _require_int(
        namespace.get("prospective_holdout_offset"),
        "prospective seed offset",
    )
    dry_run_offset = _require_int(namespace.get("dry_run_offset"), "dry-run seed offset")
    excluded_rows = _require_list(
        namespace.get("excluded_ranges"),
        "config.seed_namespace.excluded_ranges",
    )
    excluded: list[tuple[str, int, int]] = []
    for raw in excluded_rows:
        item = _require_mapping(raw, "excluded seed range")
        excluded.append(
            (
                str(item.get("name")),
                _require_int(item.get("start"), "excluded range start"),
                _require_int(item.get("end"), "excluded range end"),
            )
        )
    if tuple(excluded) != _EXPECTED_EXCLUDED_RANGES:
        raise ValueError("E02b excluded seed ranges differ from the locked namespace.")
    if block_size != 100000 or prospective_offset != 62500 or dry_run_offset != 90000:
        raise ValueError("E02b seed offsets differ from the locked namespace.")
    active_offset = dry_run_offset if dry_run else prospective_offset
    active_range = _relative_seed_interval(active_offset, count_per_operation)
    if active_range[1] >= block_size:
        raise ValueError("E02b active seed range leaves its checkpoint-seed block.")
    comparison_ranges = [(name, (start, end)) for name, start, end in excluded]
    if dry_run:
        comparison_ranges.append(
            (
                "e02b_prospective_holdout",
                _relative_seed_interval(
                    prospective_offset,
                    _require_int(
                        _require_mapping(config.get("data"), "config.data").get(
                            "count_per_operation"
                        ),
                        "configured count_per_operation",
                    ),
                ),
            )
        )
    collisions = [
        name
        for name, seed_range in comparison_ranges
        if _ranges_overlap(active_range, seed_range)
    ]
    if collisions:
        raise ValueError(f"E02b active seed range collides with: {collisions}.")
    return active_offset, {
        "block_size": block_size,
        "active_offset": active_offset,
        "active_relative_range": list(active_range),
        "prospective_holdout_offset": prospective_offset,
        "dry_run_offset": dry_run_offset,
        "excluded_ranges": [
            {"name": name, "start": start, "end": end}
            for name, start, end in excluded
        ],
        "disjoint": True,
    }


def _heldout_episodes(
    *,
    checkpoint_seed: int,
    seed_offset: int,
    seed_block_size: int,
    cells: tuple[GeometryCell, ...],
    repeats_per_cell: int,
    registered_repeats_per_cell: int,
    data: Mapping[str, Any],
) -> list[HeldoutEpisode]:
    if not cells or repeats_per_cell <= 0 or registered_repeats_per_cell <= 0:
        raise ValueError("E02b held-out geometry design must be nonempty.")
    if repeats_per_cell > registered_repeats_per_cell:
        raise ValueError("Executed repeats cannot exceed the registered main repeats.")
    rows: list[HeldoutEpisode] = []
    cursor = 0
    for cell in cells:
        for repeat_id in range(repeats_per_cell):
            episode_index = cell.cell_id * registered_repeats_per_cell + repeat_id
            for operation in Operation:
                geometry_seed = (
                    checkpoint_seed * seed_block_size + seed_offset + cursor
                )
                rows.append(
                    HeldoutEpisode(
                        geometry_seed=geometry_seed,
                        cell_id=cell.cell_id,
                        norm_pair_bin=cell.norm_pair_bin,
                        angle_bin=cell.angle_bin,
                        repeat_id=repeat_id,
                        old_scale=cell.old_scale,
                        new_scale=cell.new_scale,
                        angle_degrees=cell.angle_degrees,
                        old_new_cosine=cell.old_new_cosine,
                        episode=build_geometry_episode(
                            seed=geometry_seed,
                            operation=operation,
                            candidate_mode=CandidateMode.ORACLE,
                            key_dim=int(data["key_dim"]),
                            value_dim=int(data["value_dim"]),
                            num_associations=int(data["num_associations"]),
                            key_correlation=float(data["key_correlation"]),
                            old_scale=cell.old_scale,
                            new_scale=cell.new_scale,
                            old_new_cosine=cell.old_new_cosine,
                            episode_index=episode_index,
                        ),
                    )
                )
                cursor += 1
    return rows


def _load_frozen_model(
    checkpoint: FrozenCheckpoint,
    *,
    constraint: ScalarConstraint,
    device: torch.device,
) -> MatchedScalarController:
    if checkpoint.constraint != constraint.value:
        raise ProvenanceValidationError("Checkpoint constraint does not match requested model.")
    if sha256_file(checkpoint.path) != checkpoint.file_sha256:
        raise ProvenanceValidationError(
            f"Checkpoint bytes changed before load: {checkpoint.path}."
        )
    raw_state = torch.load(checkpoint.path, map_location="cpu", weights_only=True)
    if sha256_file(checkpoint.path) != checkpoint.file_sha256:
        raise ProvenanceValidationError(
            f"Checkpoint bytes changed during load: {checkpoint.path}."
        )
    if not isinstance(raw_state, dict) or not raw_state:
        raise ProvenanceValidationError(f"Invalid checkpoint state: {checkpoint.path}.")
    state: dict[str, torch.Tensor] = {}
    for name, value in raw_state.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ProvenanceValidationError(f"Invalid state entry in {checkpoint.path}.")
        if not bool(torch.isfinite(value).all().item()):
            raise FloatingPointError(f"Non-finite state entry {name} in {checkpoint.path}.")
        state[name] = value
    if _state_dict_sha256(state) != checkpoint.state_dict_sha256:
        raise ProvenanceValidationError(f"State-dict hash mismatch: {checkpoint.path}.")
    model = MatchedScalarController(
        checkpoint.input_dim,
        checkpoint.hidden_dim,
        constraint,
    )
    model.load_state_dict(state, strict=True)
    if sum(parameter.numel() for parameter in model.parameters()) != checkpoint.parameter_count:
        raise ProvenanceValidationError(f"Parameter count mismatch: {checkpoint.path}.")
    model.to(device)
    model.eval()
    return model


def _evaluate_model(
    model: MatchedScalarController,
    episode: MemoryEpisode,
    device: torch.device,
) -> EpisodeMetrics:
    with torch.no_grad():
        local = episode.to(device)
        features = controller_features(episode).to(device).unsqueeze(0)
        gates = model(features)
        output = apply_scalar_update(
            local,
            gates.erase.squeeze(0),
            gates.write.squeeze(0),
        )
    if not bool(torch.isfinite(output).all().item()):
        raise FloatingPointError(f"Non-finite output for {episode.episode_id}.")
    return evaluate_episode(output.cpu(), episode)


def _bootstrap_seed_means(
    seed_effects: Mapping[int, np.ndarray],
    seed_strata: Mapping[int, np.ndarray],
    *,
    samples: int,
    seed: int,
    confidence: float,
) -> Interval:
    if set(seed_effects) != set(seed_strata) or not seed_effects:
        raise ValueError("Bootstrap effects and strata must cover the same seeds.")
    effects = {
        training_seed: _finite_vector(
            values,
            f"bootstrap effects seed {training_seed}",
        )
        for training_seed, values in seed_effects.items()
    }
    strata = {
        training_seed: np.asarray(seed_strata[training_seed]).astype(str)
        for training_seed in effects
    }
    if any(
        len(effects[training_seed]) != len(strata[training_seed])
        for training_seed in effects
    ):
        raise ValueError("Bootstrap stratum labels are not aligned.")

    def statistic(indices_by_seed: Mapping[int, np.ndarray]) -> float:
        return float(
            np.mean(
                [
                    effects[training_seed][indices].mean()
                    for training_seed, indices in indices_by_seed.items()
                ]
            )
        )

    return fixed_seed_operation_stratified_bootstrap(
        strata,
        statistic,
        samples=samples,
        seed=seed,
        confidence=confidence,
    )


def _bootstrap_raw_interaction(
    seed_effects: Mapping[int, np.ndarray],
    seed_operations: Mapping[int, np.ndarray],
    seed_strata: Mapping[int, np.ndarray],
    *,
    samples: int,
    seed: int,
    confidence: float,
) -> Interval:
    effects = {
        training_seed: _finite_vector(values, f"DID effects seed {training_seed}")
        for training_seed, values in seed_effects.items()
    }
    operations = {
        training_seed: np.asarray(seed_operations[training_seed]).astype(str)
        for training_seed in effects
    }
    strata = {
        training_seed: np.asarray(seed_strata[training_seed]).astype(str)
        for training_seed in effects
    }
    if (
        set(effects) != set(operations)
        or set(effects) != set(strata)
        or not effects
    ):
        raise ValueError("DID effects, operations, and strata must cover the same seeds.")
    if any(
        len(effects[training_seed]) != len(operations[training_seed])
        or len(effects[training_seed]) != len(strata[training_seed])
        for training_seed in effects
    ):
        raise ValueError("DID operation or stratum labels are not aligned.")

    def statistic(indices_by_seed: Mapping[int, np.ndarray]) -> float:
        per_seed: list[float] = []
        for training_seed, indices in indices_by_seed.items():
            values = effects[training_seed][indices]
            labels = operations[training_seed][indices]
            asymmetric = np.isin(
                labels,
                [Operation.ADD.value, Operation.INVALIDATE.value],
            )
            per_seed.append(float(values[asymmetric].mean() - values[~asymmetric].mean()))
        return float(np.mean(per_seed))

    return fixed_seed_operation_stratified_bootstrap(
        strata,
        statistic,
        samples=samples,
        seed=seed,
        confidence=confidence,
    )


def _equivalence_within(interval: Interval, margin: float) -> bool:
    return bool(interval.low >= -margin and interval.high <= margin)


def _interval_fields(interval: Interval | None) -> dict[str, object]:
    if interval is None:
        return {"estimate": None, "ci95": None}
    return {
        "estimate": interval.estimate,
        "ci95": [interval.low, interval.high],
    }


def _asymmetric_registered_support_complete(
    counts_by_seed: Mapping[int, Mapping[str, Mapping[str, int]]],
    seeds: list[int] | tuple[int, ...],
    *,
    expected_per_operation: int,
    main_design: bool,
) -> bool:
    if (
        not main_design
        or expected_per_operation <= 0
        or set(counts_by_seed) != set(seeds)
    ):
        return False
    for seed in seeds:
        per_operation = counts_by_seed[seed]
        for operation in (Operation.ADD, Operation.INVALIDATE):
            counts = per_operation.get(operation.value)
            if (
                counts is None
                or counts.get("expected") != expected_per_operation
                or counts.get("eligible") != expected_per_operation
                or counts.get("excluded_low_headroom") != 0
            ):
                return False
    return True


def _repair_adjudication(
    *,
    dry_run: bool,
    exact_main_execution: bool,
    inference_eligible: bool,
    six_gate_supported: bool,
) -> dict[str, object]:
    if six_gate_supported and not inference_eligible:
        raise ValueError("A supported E02b repair must be inference-eligible.")
    if inference_eligible and not exact_main_execution:
        raise ValueError("E02b inference eligibility requires exact main execution.")
    if dry_run:
        status = "NOT_EVALUATED_DRY_RUN"
        reason = "DRY_RUN"
    elif not exact_main_execution:
        status = "INCONCLUSIVE"
        reason = "MAIN_EXECUTION_INCOMPLETE"
    elif not inference_eligible:
        status = "INCONCLUSIVE"
        reason = "REGISTERED_ASYMMETRIC_SUPPORT_INCOMPLETE"
    elif six_gate_supported:
        status = "SUPPORTED"
        reason = None
    else:
        status = "NOT_SUPPORTED"
        reason = "ONE_OR_MORE_EVALUABLE_GATES_FAILED"
    return {
        "status": status,
        "reason": reason,
        "evaluated": inference_eligible,
        "inference_eligible": inference_eligible,
        "six_gate_supported": six_gate_supported,
    }


def _six_gate_supported(gates: Mapping[str, bool]) -> bool:
    return set(gates) == set(_GATE_NAMES) and all(gates[name] for name in _GATE_NAMES)


def _string_seed_keys(values: Mapping[int, object]) -> dict[str, object]:
    return {str(seed): values[seed] for seed in sorted(values)}


def main() -> None:
    parser = build_parser(EXPERIMENT_ID, DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = Path(args.config).expanduser().resolve(strict=True)
    preview = load_config(config_path)
    _validate_protocol_config(preview, config_path)
    e00_dependency = _validate_e00_dependency(
        args.artifact_root,
        dry_run=args.dry_run,
    )
    source = validate_pinned_e02_source(args.artifact_root, preview)
    dependencies = [e00_dependency, source.dependency_record()]
    config, run_dir, device, context = initialize_v61_run(
        experiment_id=EXPERIMENT_ID,
        config_path=str(config_path),
        artifact_root=args.artifact_root,
        device_request=args.device,
        dry_run=args.dry_run,
        dependencies=dependencies,
    )
    if config != preview:
        raise ProvenanceValidationError("E02b config changed during initialization.")
    _validate_protocol_config(config, context.config_path)

    configured_seeds = [int(value) for value in config["seeds"]]
    if tuple(configured_seeds) != _EXPECTED_SEEDS:
        raise ValueError("E02b requires the eight pinned E02 checkpoint seeds.")
    seeds = list(configured_seeds)
    data = _require_mapping(config.get("data"), "config.data")
    registered_count_per_operation = _require_int(
        data.get("count_per_operation"),
        "data.count_per_operation",
    )
    geometry_cells, repeats_per_cell = _execution_geometry_design(
        config,
        dry_run=args.dry_run,
    )
    count_per_operation = len(geometry_cells) * repeats_per_cell
    if args.dry_run:
        seeds = seeds[:1]
    seed_offset, seed_contract = _validate_seed_namespace(
        config,
        count_per_operation=count_per_operation,
        dry_run=args.dry_run,
    )
    seed_block_size = int(
        _require_mapping(config.get("seed_namespace"), "config.seed_namespace")[
            "block_size"
        ]
    )
    stats = _require_mapping(config.get("statistics"), "config.statistics")
    bootstrap_seeds = _require_mapping(
        stats.get("bootstrap_seeds"),
        "config.statistics.bootstrap_seeds",
    )
    alpha = _require_finite(stats["alpha"], "alpha")
    bootstrap_samples = int(stats["bootstrap_samples"])
    confidence = _require_finite(stats["bootstrap_confidence"], "bootstrap confidence")
    minimum_headroom = _require_finite(
        stats["minimum_tied_oracle_headroom"],
        "minimum tied-to-oracle headroom",
    )
    sesoi = _require_finite(
        stats["asymmetric_normalized_gain_sesoi"],
        "asymmetric normalized gain SESOI",
    )
    preserve_margin = _require_finite(
        stats["preserve_absolute_equivalence_margin"],
        "PRESERVE equivalence margin",
    )
    supersede_margin = _require_finite(
        stats["supersede_absolute_equivalence_margin"],
        "SUPERSEDE equivalence margin",
    )
    retention_margin = _require_finite(
        stats["retention_noninferiority_margin"],
        "retention noninferiority margin",
    )

    rows: list[dict[str, object]] = []
    asym_by_seed: dict[int, np.ndarray] = {}
    asym_strata_by_seed: dict[int, np.ndarray] = {}
    preserve_by_seed: dict[int, np.ndarray] = {}
    preserve_strata_by_seed: dict[int, np.ndarray] = {}
    supersede_by_seed: dict[int, np.ndarray] = {}
    supersede_strata_by_seed: dict[int, np.ndarray] = {}
    retention_by_seed: dict[int, np.ndarray] = {}
    retention_strata_by_seed: dict[int, np.ndarray] = {}
    did_effects_by_seed: dict[int, np.ndarray] = {}
    did_operations_by_seed: dict[int, np.ndarray] = {}
    did_strata_by_seed: dict[int, np.ndarray] = {}
    did_by_seed: dict[int, float] = {}
    operation_counts_by_seed: dict[int, dict[str, int]] = {}
    cell_operation_counts_by_seed: dict[int, dict[str, dict[str, int]]] = {}
    geometry_ranges_by_seed: dict[int, dict[str, int]] = {}
    asymmetric_eligibility_counts_by_seed: dict[
        int,
        dict[str, dict[str, int]],
    ] = {}

    for checkpoint_seed in seeds:
        tied_checkpoint = source.checkpoints[(checkpoint_seed, "tied")]
        dual_checkpoint = source.checkpoints[(checkpoint_seed, "dual")]
        tied = _load_frozen_model(
            tied_checkpoint,
            constraint=ScalarConstraint.TIED,
            device=device,
        )
        dual = _load_frozen_model(
            dual_checkpoint,
            constraint=ScalarConstraint.DUAL,
            device=device,
        )
        heldout = _heldout_episodes(
            checkpoint_seed=checkpoint_seed,
            seed_offset=seed_offset,
            seed_block_size=seed_block_size,
            cells=geometry_cells,
            repeats_per_cell=repeats_per_cell,
            registered_repeats_per_cell=_EXPECTED_REPEATS_PER_CELL,
            data=data,
        )
        geometry_seeds = [item.geometry_seed for item in heldout]
        if len(set(geometry_seeds)) != len(geometry_seeds):
            raise AssertionError(f"Duplicate E02b geometry seed for checkpoint {checkpoint_seed}.")
        geometry_ranges_by_seed[checkpoint_seed] = {
            "first": min(geometry_seeds),
            "last": max(geometry_seeds),
            "count": len(geometry_seeds),
        }

        operation_improvements: dict[Operation, list[float]] = defaultdict(list)
        operation_strata: dict[Operation, list[str]] = defaultdict(list)
        asym_values: list[float] = []
        asym_strata: list[str] = []
        retention_values: list[float] = []
        retention_strata: list[str] = []
        asymmetric_eligibility_counts = {
            operation.value: {
                "expected": count_per_operation,
                "eligible": 0,
                "excluded_low_headroom": 0,
            }
            for operation in (Operation.ADD, Operation.INVALIDATE)
        }
        operation_counts = {operation.value: 0 for operation in Operation}
        cell_operation_counts = {
            cell.label: {operation.value: 0 for operation in Operation}
            for cell in geometry_cells
        }

        for item in heldout:
            episode = item.episode
            cell_label = f"cell_{item.cell_id:02d}"
            stratum = f"{episode.operation.value}|{cell_label}"
            tied_metrics = _evaluate_model(tied, episode, device)
            dual_metrics = _evaluate_model(dual, episode, device)
            noop_metrics = evaluate_episode(episode.state, episode)
            oracle_metrics = evaluate_episode(episode.target_state, episode)
            operation_counts[episode.operation.value] += 1
            cell_operation_counts[cell_label][episode.operation.value] += 1

            improvement = _require_finite(
                tied_metrics.affected_read_mse - dual_metrics.affected_read_mse,
                f"{episode.episode_id} tied-minus-dual affected MSE",
            )
            tied_headroom = _require_finite(
                tied_metrics.affected_read_mse - oracle_metrics.affected_read_mse,
                f"{episode.episode_id} tied-to-oracle headroom",
            )
            normalized_gain: float | None = None
            normalized_eligible = False
            normalized_exclusion_reason: str | None
            if not episode.operation.is_asymmetric:
                normalized_exclusion_reason = "operation_not_add_or_invalidate"
            elif tied_headroom < minimum_headroom:
                normalized_exclusion_reason = "tied_to_oracle_headroom_below_minimum"
                asymmetric_eligibility_counts[episode.operation.value][
                    "excluded_low_headroom"
                ] += 1
            else:
                normalized_exclusion_reason = None
                normalized_eligible = True
                normalized_gain = _require_finite(
                    improvement / tied_headroom,
                    f"{episode.episode_id} normalized gain",
                )
                asym_values.append(normalized_gain)
                asym_strata.append(stratum)
                asymmetric_eligibility_counts[episode.operation.value]["eligible"] += 1

            retention_effect = _require_finite(
                dual_metrics.unaffected_retention_mse
                - tied_metrics.unaffected_retention_mse,
                f"{episode.episode_id} retention effect",
            )
            operation_improvements[episode.operation].append(improvement)
            operation_strata[episode.operation].append(stratum)
            retention_values.append(retention_effect)
            retention_strata.append(stratum)
            rows.append(
                {
                    "checkpoint_seed": checkpoint_seed,
                    "geometry_seed": item.geometry_seed,
                    "split": (
                        "development_dry_run"
                        if args.dry_run
                        else "prospective_holdout"
                    ),
                    "candidate_mode": CandidateMode.ORACLE.value,
                    "episode_id": episode.episode_id,
                    "episode_index": (
                        item.cell_id * _EXPECTED_REPEATS_PER_CELL + item.repeat_id
                    ),
                    "operation": episode.operation.value,
                    "cell_id": item.cell_id,
                    "cell_label": cell_label,
                    "norm_pair_bin": item.norm_pair_bin,
                    "angle_bin": item.angle_bin,
                    "repeat_id": item.repeat_id,
                    "old_scale": item.old_scale,
                    "new_scale": item.new_scale,
                    "angle_degrees": item.angle_degrees,
                    "old_new_cosine": item.old_new_cosine,
                    "unseen_vs_e02_original": True,
                    "tied_affected_mse": _require_finite(
                        tied_metrics.affected_read_mse,
                        f"{episode.episode_id} tied affected MSE",
                    ),
                    "dual_affected_mse": _require_finite(
                        dual_metrics.affected_read_mse,
                        f"{episode.episode_id} dual affected MSE",
                    ),
                    "noop_affected_mse": _require_finite(
                        noop_metrics.affected_read_mse,
                        f"{episode.episode_id} noop affected MSE",
                    ),
                    "oracle_affected_mse": _require_finite(
                        oracle_metrics.affected_read_mse,
                        f"{episode.episode_id} oracle affected MSE",
                    ),
                    "tied_minus_dual_affected_mse": improvement,
                    "tied_to_oracle_headroom": tied_headroom,
                    "normalized_gain": normalized_gain,
                    "normalized_gain_eligible": normalized_eligible,
                    "normalized_gain_exclusion_reason": normalized_exclusion_reason,
                    "tied_retention_mse": _require_finite(
                        tied_metrics.unaffected_retention_mse,
                        f"{episode.episode_id} tied retention MSE",
                    ),
                    "dual_retention_mse": _require_finite(
                        dual_metrics.unaffected_retention_mse,
                        f"{episode.episode_id} dual retention MSE",
                    ),
                    "dual_minus_tied_retention_mse": retention_effect,
                }
            )

        if set(operation_counts.values()) != {count_per_operation}:
            raise AssertionError(f"E02b checkpoint {checkpoint_seed} is operation-unbalanced.")
        if any(
            count != repeats_per_cell
            for per_cell_counts in cell_operation_counts.values()
            for count in per_cell_counts.values()
        ):
            raise AssertionError(
                f"E02b checkpoint {checkpoint_seed} is not balanced by cell and operation."
            )
        if any(
            counts["eligible"] + counts["excluded_low_headroom"]
            != counts["expected"]
            for counts in asymmetric_eligibility_counts.values()
        ):
            raise AssertionError(
                f"E02b checkpoint {checkpoint_seed} asymmetric eligibility is incomplete."
            )
        operation_counts_by_seed[checkpoint_seed] = operation_counts
        cell_operation_counts_by_seed[checkpoint_seed] = cell_operation_counts
        asymmetric_eligibility_counts_by_seed[checkpoint_seed] = (
            asymmetric_eligibility_counts
        )
        asym_by_seed[checkpoint_seed] = _finite_vector(
            asym_values,
            f"asymmetric gains seed {checkpoint_seed}",
            allow_empty=True,
        )
        asym_strata_by_seed[checkpoint_seed] = np.asarray(asym_strata)
        preserve_by_seed[checkpoint_seed] = _finite_vector(
            operation_improvements[Operation.PRESERVE],
            f"PRESERVE raw effects seed {checkpoint_seed}",
        )
        preserve_strata_by_seed[checkpoint_seed] = np.asarray(
            operation_strata[Operation.PRESERVE]
        )
        supersede_by_seed[checkpoint_seed] = _finite_vector(
            operation_improvements[Operation.SUPERSEDE],
            f"SUPERSEDE raw effects seed {checkpoint_seed}",
        )
        supersede_strata_by_seed[checkpoint_seed] = np.asarray(
            operation_strata[Operation.SUPERSEDE]
        )
        retention_by_seed[checkpoint_seed] = _finite_vector(
            retention_values,
            f"retention effects seed {checkpoint_seed}",
        )
        retention_strata_by_seed[checkpoint_seed] = np.asarray(retention_strata)
        did_effects_by_seed[checkpoint_seed] = np.concatenate(
            [
                np.asarray(operation_improvements[operation], dtype=np.float64)
                for operation in Operation
            ]
        )
        did_operations_by_seed[checkpoint_seed] = np.concatenate(
            [
                np.full(count_per_operation, operation.value)
                for operation in Operation
            ]
        )
        did_strata_by_seed[checkpoint_seed] = np.concatenate(
            [
                np.asarray(operation_strata[operation])
                for operation in Operation
            ]
        )
        asymmetric_raw = np.concatenate(
            [
                np.asarray(operation_improvements[Operation.ADD]),
                np.asarray(operation_improvements[Operation.INVALIDATE]),
            ]
        )
        symmetric_raw = np.concatenate(
            [
                np.asarray(operation_improvements[Operation.PRESERVE]),
                np.asarray(operation_improvements[Operation.SUPERSEDE]),
            ]
        )
        did_by_seed[checkpoint_seed] = _require_finite(
            asymmetric_raw.mean() - symmetric_raw.mean(),
            f"raw operation interaction seed {checkpoint_seed}",
        )

    expected_rows = len(seeds) * len(Operation) * count_per_operation
    if len(rows) != expected_rows:
        raise AssertionError(f"Expected {expected_rows} E02b rows, got {len(rows)}.")

    asym_evaluable = _asymmetric_registered_support_complete(
        asymmetric_eligibility_counts_by_seed,
        seeds,
        expected_per_operation=registered_count_per_operation,
        main_design=bool(
            not args.dry_run
            and seeds == configured_seeds
            and count_per_operation == registered_count_per_operation
        ),
    )
    asym_interval = (
        _bootstrap_seed_means(
            asym_by_seed,
            asym_strata_by_seed,
            samples=bootstrap_samples,
            seed=int(bootstrap_seeds["asymmetric_gain"]),
            confidence=confidence,
        )
        if asym_evaluable
        else None
    )
    preserve_interval = _bootstrap_seed_means(
        preserve_by_seed,
        preserve_strata_by_seed,
        samples=bootstrap_samples,
        seed=int(bootstrap_seeds["preserve_equivalence"]),
        confidence=confidence,
    )
    supersede_interval = _bootstrap_seed_means(
        supersede_by_seed,
        supersede_strata_by_seed,
        samples=bootstrap_samples,
        seed=int(bootstrap_seeds["supersede_equivalence"]),
        confidence=confidence,
    )
    retention_interval = _bootstrap_seed_means(
        retention_by_seed,
        retention_strata_by_seed,
        samples=bootstrap_samples,
        seed=int(bootstrap_seeds["retention_noninferiority"]),
        confidence=confidence,
    )
    did_interval = _bootstrap_raw_interaction(
        did_effects_by_seed,
        did_operations_by_seed,
        did_strata_by_seed,
        samples=bootstrap_samples,
        seed=int(bootstrap_seeds["operation_interaction"]),
        confidence=confidence,
    )

    asym_seed_values = (
        _finite_vector(
            [asym_by_seed[seed].mean() for seed in seeds],
            "asymmetric seed values",
        )
        if asym_evaluable
        else None
    )
    did_seed_values = _finite_vector(
        [did_by_seed[seed] for seed in seeds],
        "DID seed values",
    )
    retention_seed_values = _finite_vector(
        [retention_by_seed[seed].mean() for seed in seeds],
        "retention seed values",
    )
    asym_sign_flip_p = (
        exact_sign_flip_test(asym_seed_values - sesoi, "greater")
        if asym_seed_values is not None
        else None
    )
    did_sign_flip_p = exact_sign_flip_test(did_seed_values, "greater")
    retention_sign_flip_p = exact_sign_flip_test(
        retention_seed_values - retention_margin,
        "less",
    )

    gate_values = {
        "asymmetric_normalized_gain": bool(
            asym_interval is not None
            and asym_sign_flip_p is not None
            and asym_interval.low > sesoi
            and asym_sign_flip_p <= alpha
        ),
        "preserve_raw_equivalence": _equivalence_within(
            preserve_interval,
            preserve_margin,
        ),
        "supersede_raw_equivalence": _equivalence_within(
            supersede_interval,
            supersede_margin,
        ),
        "positive_raw_operation_interaction": bool(
            did_interval.low > 0.0 and did_sign_flip_p <= alpha
        ),
        "retention_noninferiority": bool(
            retention_interval.high <= retention_margin
            and retention_sign_flip_p <= alpha
        ),
        "inherited_tuning_direction": bool(
            set(source.inherited_tuning_values) == set(_EXPECTED_SEEDS)
            and all(value > 0.0 for value in source.inherited_tuning_values.values())
        ),
    }
    exact_main_execution = bool(
        not args.dry_run
        and seeds == configured_seeds
        and len(source.checkpoints) == 16
        and len(rows) == 16384
        and len(geometry_cells) == 16
        and repeats_per_cell == 32
    )
    repair_inference_eligible = bool(exact_main_execution and asym_evaluable)
    supported = bool(
        repair_inference_eligible and _six_gate_supported(gate_values)
    )
    repair_adjudication = _repair_adjudication(
        dry_run=bool(args.dry_run),
        exact_main_execution=exact_main_execution,
        inference_eligible=repair_inference_eligible,
        six_gate_supported=supported,
    )

    metrics_path = run_dir / "prospective_episode_metrics.jsonl"
    write_jsonl_strict(metrics_path, rows)
    claim = _require_mapping(config.get("claim"), "config.claim")
    evidence = _require_mapping(config.get("evidence_scope"), "config.evidence_scope")
    report: dict[str, Any] = {
        "status": "PASS",
        "scientific_evidence": False,
        "execution": {
            "dry_run": bool(args.dry_run),
            "checkpoint_only": True,
            "training_executed": False,
            "optimizer_created": False,
            "source_checkpoint_pair_count": len(source.checkpoints) // 2,
            "source_checkpoint_file_count": len(source.checkpoints),
            "row_count": len(rows),
            "expected_row_count": expected_rows,
            "configured_seeds": configured_seeds,
            "executed_seeds": seeds,
            "count_per_operation": count_per_operation,
            "geometry_cell_count": len(geometry_cells),
            "repeats_per_cell": repeats_per_cell,
            "exact_main_execution": exact_main_execution,
        },
        "source_e02": {
            "run_id": source.run.run_id,
            "run_dir": str(source.run.run_dir),
            "manifest_sha256": source.run.manifest_sha256,
            "report_sha256": source.run.report_sha256,
            "original_status": source.run.status,
            "original_eligibility": {
                "main": source.run.main_eligible,
                "full": source.run.full_eligible,
            },
            "original_claim_supported": False,
            **_ORIGINAL_E02_ADJUDICATION,
            "checkpoint_recipe": "strict",
            "checkpoint_contract_sha256": _PINNED_SOURCE[
                "strict_checkpoint_contract_sha256"
            ],
            "checkpoints": [
                source.checkpoints[key].report_record()
                for key in sorted(source.checkpoints)
            ],
        },
        "fresh_geometry_contract": {
            **seed_contract,
            "protocol_label": _GEOMETRY_PROTOCOL_LABEL,
            "relationship_to_original_e02": (
                "OOD geometry extension, not a same-cell gate repair"
            ),
            "same_registered_geometry_cell_as_e02": False,
            "unseen_random_episode_realizations": True,
            "unseen_geometry_hyperparameter_values": True,
            "original_e02_geometry": {
                "old_scale": 1.0,
                "new_scale": 1.0,
                "angle_degrees": 90.0,
            },
            "norm_pairs": [list(pair) for pair in _EXPECTED_NORM_PAIRS],
            "angles_degrees": list(_EXPECTED_ANGLES_DEGREES),
            "registered_cell_count": 16,
            "registered_repeats_per_cell": _EXPECTED_REPEATS_PER_CELL,
            "executed_cell_ids": [cell.cell_id for cell in geometry_cells],
            "executed_repeats_per_cell": repeats_per_cell,
            "cells": [cell.report_record() for cell in _geometry_cells(config)],
            "all_cells_unseen": True,
            "balanced_by_checkpoint_cell_operation": True,
            "candidate_mode": CandidateMode.ORACLE.value,
            "operation_schedule": "geometry_cell_repeat_operation_interleaved",
            "operation_counts_by_seed": _string_seed_keys(operation_counts_by_seed),
            "cell_operation_counts_by_seed": _string_seed_keys(
                cell_operation_counts_by_seed
            ),
            "absolute_geometry_seed_ranges_by_checkpoint_seed": _string_seed_keys(
                geometry_ranges_by_seed
            ),
        },
        "inference_contract": {
            "episode_uncertainty": {
                "method": "fixed_checkpoint_episode_bootstrap_ci",
                "stratification": (
                    "within_checkpoint_seed_operation_by_geometry_cell"
                ),
                "applies_to_fresh_gates": list(_GATE_NAMES[:5]),
            },
            "checkpoint_replicate_uncertainty": {
                "method": "eight_seed_exact_sign_flip",
                "applies_only_to_gates": list(_SIGN_FLIP_GATE_NAMES),
            },
            "equivalence_gate_inference": {
                "gates": list(_EQUIVALENCE_GATE_NAMES),
                "method": "fixed_checkpoint_episode_bootstrap_ci",
                "separate_checkpoint_seed_sign_flip": False,
            },
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_confidence": confidence,
            "seed_sign_patterns": 2 ** len(seeds),
            "claim_inference_eligible": repair_inference_eligible,
        },
        "gates": {
            "asymmetric_normalized_gain": {
                "fresh": True,
                "metric": "fraction_of_tied_to_oracle_gap_closed",
                **_interval_fields(asym_interval),
                "sesoi": sesoi,
                "registered_required_operations": [
                    Operation.ADD.value,
                    Operation.INVALIDATE.value,
                ],
                "registered_required_eligible_per_operation_per_seed": (
                    registered_count_per_operation
                ),
                "registered_support_complete": asym_evaluable,
                "seed_exact_sign_flip_p_greater_than_sesoi": asym_sign_flip_p,
                "episode_interval_method": (
                    "fixed_checkpoint_episode_bootstrap_ci"
                ),
                "checkpoint_seed_test": "eight_seed_exact_sign_flip",
                "per_operation_eligibility_counts_by_seed": _string_seed_keys(
                    asymmetric_eligibility_counts_by_seed
                ),
                "supported": gate_values["asymmetric_normalized_gain"],
            },
            "preserve_raw_equivalence": {
                "fresh": True,
                "metric": "tied_minus_dual_affected_read_mse",
                **_interval_fields(preserve_interval),
                "equivalence_margin": preserve_margin,
                "inference_method": "fixed_checkpoint_episode_bootstrap_ci",
                "separate_checkpoint_seed_sign_flip": False,
                "supported": gate_values["preserve_raw_equivalence"],
            },
            "supersede_raw_equivalence": {
                "fresh": True,
                "metric": "tied_minus_dual_affected_read_mse",
                "normalization": None,
                **_interval_fields(supersede_interval),
                "equivalence_margin": supersede_margin,
                "inference_method": "fixed_checkpoint_episode_bootstrap_ci",
                "separate_checkpoint_seed_sign_flip": False,
                "supported": gate_values["supersede_raw_equivalence"],
            },
            "positive_raw_operation_interaction": {
                "fresh": True,
                "metric": (
                    "mean_raw_add_invalidate_improvement_minus_"
                    "mean_raw_preserve_supersede_improvement"
                ),
                **_interval_fields(did_interval),
                "seed_values": _string_seed_keys(did_by_seed),
                "seed_exact_sign_flip_p_greater_than_zero": did_sign_flip_p,
                "episode_interval_method": (
                    "fixed_checkpoint_episode_bootstrap_ci"
                ),
                "checkpoint_seed_test": "eight_seed_exact_sign_flip",
                "supported": gate_values["positive_raw_operation_interaction"],
            },
            "retention_noninferiority": {
                "fresh": True,
                "metric": "dual_minus_tied_unaffected_retention_mse",
                **_interval_fields(retention_interval),
                "margin": retention_margin,
                "seed_exact_sign_flip_p_less_than_margin": retention_sign_flip_p,
                "episode_interval_method": (
                    "fixed_checkpoint_episode_bootstrap_ci"
                ),
                "checkpoint_seed_test": "eight_seed_exact_sign_flip",
                "supported": gate_values["retention_noninferiority"],
            },
            "inherited_tuning_direction": {
                "fresh": False,
                "source": "immutable_e02_report.optimization_robustness",
                "source_subtree_sha256": _PINNED_SOURCE[
                    "inherited_tuning_gate_sha256"
                ],
                "tuned_checkpoints_replayed": False,
                "reason_not_replayed": (
                    "E02 persisted strict checkpoints only; no complete selected "
                    "tuned checkpoint pair exists."
                ),
                "seed_values": _string_seed_keys(source.inherited_tuning_values),
                "supported": gate_values["inherited_tuning_direction"],
            },
        },
        "repair_adjudication": {
            **repair_adjudication,
            "prospective_protocol": _REPAIR_PROTOCOL_LABEL,
            "does_not_relabel_original_e02": True,
        },
        "claim_gate": {
            "evaluated": repair_inference_eligible,
            "inference_eligible": repair_inference_eligible,
            "supported": supported,
            "repair_status": repair_adjudication["status"],
            "repair_inconclusive_reason": repair_adjudication["reason"],
            "six_gate_conjunction": gate_values,
            "allowed_claim": claim.get("allowed_if_supported") if supported else None,
            "forbidden_claims": claim.get("forbidden"),
            "original_e02_remains_inconclusive": True,
            "original_e02_h2_claim_open": False,
        },
        "evidence_scope": {
            **dict(evidence),
            "controlled_geometry_claim_eligible": supported,
        },
        "artifacts": {
            "prospective_episode_metrics": {
                "path": metrics_path.name,
                "rows": len(rows),
                "sha256": sha256_file(metrics_path),
            }
        },
        "dependency_lineage": dependencies,
    }
    finalize_v61_run(
        context=context,
        report=report,
        main_eligible=repair_inference_eligible,
        full_eligible=repair_inference_eligible,
    )
    print(
        f"[{EXPERIMENT_ID}] PASS: {run_dir} "
        f"(H2_REPAIR={repair_adjudication['status']})"
    )


if __name__ == "__main__":
    main()
