from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
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
    read_json_object_strict,
    sha256_canonical_json,
    sha256_file,
    validate_run_manifest,
    write_json_strict,
    write_jsonl_strict,
)
from catena.core.schema import CandidateMode, MemoryEpisode, Operation
from catena.data.geometry_sweep import build_geometry_episode, controller_features
from catena.eval.functional_mediation import (
    GateChannel,
    NormMatchResult,
    dose_gate,
    exact_feasible_l2_norm_match,
    gate_vector,
    monotonic_nonincreasing_fraction,
    recovery_fraction_from_means,
    restore_relevant,
    scalarize_gate,
)
from catena.eval.metrics import EpisodeMetrics, evaluate_episode
from catena.eval.seed_inference import exact_sign_flip_test
from catena.eval.statistics_v61 import (
    Interval,
    fixed_seed_operation_stratified_bootstrap,
)
from catena.models.matched_controllers import MatchedScalarController, ScalarConstraint
from catena.models.memory import GateOutput, apply_scalar_update
from experiments.common import build_parser
from experiments.e02b_prospective_absolute_supersede import (
    FrozenCheckpoint,
    PinnedE02Source,
    validate_pinned_e02_source,
)
from experiments.v61_common import (
    finalize_v61_run,
    initialize_v61_run,
    validate_legacy_e00,
)

EXPERIMENT_ID = "e04_functional_mediation"
DEFAULT_CONFIG = "configs/e04_functional_mediation.yaml"
REPO_ROOT = Path(__file__).resolve().parents[1]

_SEED_BLOCK_SIZE = 100_000
_EXPECTED_SEEDS = (11, 22, 33, 44, 55, 66, 77, 88)
_EXPECTED_MAIN_BASE_COUNT = 128
_EXPECTED_DRY_BASE_COUNT = 8
_EXPECTED_MAIN_OFFSET = 75_000
_EXPECTED_DRY_OFFSET = 92_500
_ROW_KEY_FIELDS = ("seed", "base_index", "operation", "intervention", "dose")
_CONFIRMATORY_OPERATIONS = (Operation.ADD, Operation.INVALIDATE)
_CROSS_OPERATION = {
    Operation.ADD: Operation.INVALIDATE,
    Operation.INVALIDATE: Operation.ADD,
}
_PINNED_CONFIG_CANONICAL_SHA256 = (
    "8db019198e4c8a46f29a83c57a5eed4646d841da840ff67a45f997bde8854f1b"
)
_PINNED_CONFIG_FILE_SHA256 = (
    "f6d7a18c2ec831c8847236de46f443056dedbfc68198123fa82d104615fbc5ea"
)
_PINNED_PROTOCOL_SHA256 = (
    "06e98f7c1449181cc98be051d7dfc9a90ec6d66b93dae9de14d8f9da34f2a4c9"
)
_PINNED_PROTOCOL_LOCK_SHA256 = (
    "e419c49b808510f4d7199f3610ca0f45403fc1b37ed6c69e418454b62d3a355b"
)
_PINNED_IMPLEMENTATION_AMENDMENT_SHA256 = (
    "54272d3814c3fce5bcdca79d44566df0e818104a463ea3a693f9aa09466e984f"
)
_PINNED_IMPLEMENTATION_AMENDMENT_LOCK_SHA256 = (
    "6ad7420a3c3eaf0f56b1e86a0ac49b9b58b6d1b9c641ccf775c1acbff3335f29"
)
_PINNED_E02B: dict[str, object] = {
    "experiment_id": "e02b_prospective_absolute_supersede",
    "run_id": "20260726T180207.055493Z",
    "manifest_sha256": (
        "c6d06a6e174e3162c6cec9ee03048a3a0f725f71de829fa83811527152f8cfdd"
    ),
    "report_sha256": (
        "032c1b015851b44555ce666ed1d50908332b13f0ad65608355559c000f1d3a52"
    ),
    "prospective_episode_metrics_sha256": (
        "ee544ac88ca632f1412e54a7c3f9e0f8e235ad8721cbc9abffb333b2b6a17171"
    ),
    "config_sha256": (
        "9b7d299a916003a9d4b5038a8db325d4609d618b3d9919ba7b603cdf8d76f781"
    ),
    "config_file_sha256": (
        "4e269572652f494977f0370a82c5104bd083ae4e7a4f570635d4685af15fe956"
    ),
    "source_fingerprint_sha256": (
        "5c25ab0ee5f1aa14f5fe854e74834d633d8a856de8e258463927a795673544e7"
    ),
    "source_fingerprint_files": 125,
    "claim_registry_filename": "E02B_CLAIM_STATUS.json",
    "claim_registry_sha256": (
        "ed065cd52688d63c14fc6d59671aaeddae3841c3dadab7507ffd6938813e54b1"
    ),
}


@dataclass(frozen=True, slots=True)
class PinnedE02bRepair:
    run: ValidatedRun
    claim_registry_sha256: str

    def dependency_record(self) -> dict[str, Any]:
        return {
            **self.run.dependency_record(),
            "evidence_role": "prospective_supported_h2_repair_gate",
            "repair_status": "SUPPORTED",
            "original_e02_remains_inconclusive": True,
            "claim_registry_sha256": self.claim_registry_sha256,
        }


@dataclass(slots=True)
class SeedDesign:
    checkpoint_seed: int
    quartets: list[dict[Operation, MemoryEpisode]]
    quartet_hashes: list[str]
    dual_gates: dict[tuple[int, Operation], GateOutput]
    tied_gates: dict[tuple[int, Operation], GateOutput]
    donor_matches: dict[tuple[int, Operation, str], NormMatchResult]


class DonorNormMatchUnidentifiableError(RuntimeError):
    """The prospectively fixed donor cannot be matched inside the gate box."""


class BootstrapRatioHeadroomError(RuntimeError):
    """A registered ratio is unstable in at least one bootstrap replicate."""


def _require_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ProvenanceValidationError(f"{name} must be an object.")
    return value


def _require_list(value: object, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ProvenanceValidationError(f"{name} must be an array.")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError(f"{name} must be numeric.")
    result = float(value)
    if not np.isfinite(result):
        raise FloatingPointError(f"{name} is non-finite.")
    return result


def _integer(value: object, name: str) -> int:
    numeric = _finite(value, name)
    if not numeric.is_integer():
        raise ValueError(f"{name} must be an integer.")
    return int(numeric)


def _finite_vector(values: object, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or len(result) == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional vector.")
    if not np.isfinite(result).all():
        raise FloatingPointError(f"{name} contains a non-finite value.")
    return cast(np.ndarray, result)


def _count_jsonl_rows(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def _direct_hashed_file(run_dir: Path, filename: object, expected_hash: object) -> Path:
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ProvenanceValidationError(f"Unsafe artifact filename: {filename!r}.")
    if not isinstance(expected_hash, str):
        raise ProvenanceValidationError(f"Missing artifact hash for {filename}.")
    path = run_dir / filename
    if path.is_symlink() or not path.is_file():
        raise ProvenanceValidationError(f"Missing direct artifact: {path}.")
    if sha256_file(path) != expected_hash:
        raise ProvenanceValidationError(f"Artifact hash mismatch: {path}.")
    return path


def _validate_protocol_config(config: Mapping[str, Any], config_path: Path) -> None:
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("E04 config experiment_id differs from the frozen protocol.")
    if sha256_canonical_json(dict(config)) != _PINNED_CONFIG_CANONICAL_SHA256:
        raise ProvenanceValidationError("E04 canonical config differs from the frozen protocol.")
    if sha256_file(config_path) != _PINNED_CONFIG_FILE_SHA256:
        raise ProvenanceValidationError("E04 config bytes differ from the frozen protocol.")
    if tuple(int(value) for value in _require_list(config.get("seeds"), "config.seeds")) != (
        _EXPECTED_SEEDS
    ):
        raise ValueError("E04 checkpoint seeds differ from the frozen protocol.")
    if dict(_require_mapping(config.get("source_e02b"), "config.source_e02b")) != (
        _PINNED_E02B
    ):
        raise ProvenanceValidationError("E04 E02b dependency differs from the pinned repair.")

    protocol = REPO_ROOT / "docs/E04_PROTOCOL_PREREGISTRATION_FROZEN_KO.md"
    protocol_lock = REPO_ROOT / "docs/E04_PROTOCOL_PREREGISTRATION_LOCK_KO.md"
    implementation_amendment = (
        REPO_ROOT / "docs/E04_IMPLEMENTATION_AMENDMENT_01_FROZEN_KO.md"
    )
    implementation_amendment_lock = (
        REPO_ROOT / "docs/E04_IMPLEMENTATION_AMENDMENT_01_LOCK_KO.md"
    )
    if (
        not protocol.is_file()
        or protocol.is_symlink()
        or sha256_file(protocol) != _PINNED_PROTOCOL_SHA256
    ):
        raise ProvenanceValidationError("E04 frozen protocol hash differs.")
    if (
        not protocol_lock.is_file()
        or protocol_lock.is_symlink()
        or sha256_file(protocol_lock) != _PINNED_PROTOCOL_LOCK_SHA256
    ):
        raise ProvenanceValidationError("E04 protocol lock hash differs.")
    if (
        not implementation_amendment.is_file()
        or implementation_amendment.is_symlink()
        or sha256_file(implementation_amendment)
        != _PINNED_IMPLEMENTATION_AMENDMENT_SHA256
    ):
        raise ProvenanceValidationError("E04 implementation amendment hash differs.")
    if (
        not implementation_amendment_lock.is_file()
        or implementation_amendment_lock.is_symlink()
        or sha256_file(implementation_amendment_lock)
        != _PINNED_IMPLEMENTATION_AMENDMENT_LOCK_SHA256
    ):
        raise ProvenanceValidationError(
            "E04 implementation amendment lock hash differs."
        )


def _validate_e02b_repair(
    artifact_root: str | Path,
    config: Mapping[str, Any],
    e02: PinnedE02Source,
) -> PinnedE02bRepair:
    source = _require_mapping(config.get("source_e02b"), "config.source_e02b")
    if dict(source) != _PINNED_E02B:
        raise ProvenanceValidationError("E04 E02b source is not exactly pinned.")
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
            require_main_eligible=True,
            require_full_eligible=True,
        ),
    )
    if (
        validated.manifest_sha256 != source["manifest_sha256"]
        or validated.report_sha256 != source["report_sha256"]
        or validated.config_sha256 != source["config_sha256"]
        or validated.config_file_sha256 != source["config_file_sha256"]
    ):
        raise ProvenanceValidationError("Pinned E02b core provenance differs.")

    report = validated.report
    claim = _require_mapping(report.get("claim_gate"), "E02b report.claim_gate")
    repair = _require_mapping(
        report.get("repair_adjudication"),
        "E02b report.repair_adjudication",
    )
    execution = _require_mapping(report.get("execution"), "E02b report.execution")
    evidence = _require_mapping(report.get("evidence_scope"), "E02b report.evidence_scope")
    conjunction = _require_mapping(
        claim.get("six_gate_conjunction"),
        "E02b six-gate conjunction",
    )
    if (
        report.get("status") != "PASS"
        or claim.get("supported") is not True
        or claim.get("repair_status") != "SUPPORTED"
        or claim.get("inference_eligible") is not True
        or claim.get("original_e02_remains_inconclusive") is not True
        or claim.get("original_e02_h2_claim_open") is not False
        or set(conjunction.values()) != {True}
        or len(conjunction) != 6
        or repair.get("status") != "SUPPORTED"
        or repair.get("evaluated") is not True
        or repair.get("does_not_relabel_original_e02") is not True
        or execution.get("dry_run") is not False
        or execution.get("row_count") != 16_384
        or execution.get("source_checkpoint_pair_count") != 8
        or execution.get("training_executed") is not False
        or execution.get("optimizer_created") is not False
        or evidence.get("evidence_tier") != "CONTROLLED_REFERENCE"
        or evidence.get("scientific_evidence") is not False
    ):
        raise ProvenanceValidationError("Pinned E02b is not the supported repair artifact.")

    artifact = _require_mapping(
        _require_mapping(report.get("artifacts"), "E02b report.artifacts").get(
            "prospective_episode_metrics"
        ),
        "E02b prospective metrics artifact",
    )
    metric_path = _direct_hashed_file(
        validated.run_dir,
        artifact.get("path"),
        source["prospective_episode_metrics_sha256"],
    )
    if (
        artifact.get("rows") != 16_384
        or artifact.get("sha256") != source["prospective_episode_metrics_sha256"]
        or _count_jsonl_rows(metric_path) != 16_384
    ):
        raise ProvenanceValidationError("Pinned E02b metric artifact is incomplete.")

    source_e02 = _require_mapping(report.get("source_e02"), "E02b report.source_e02")
    if (
        source_e02.get("run_id") != e02.run.run_id
        or source_e02.get("manifest_sha256") != e02.run.manifest_sha256
        or source_e02.get("report_sha256") != e02.run.report_sha256
        or source_e02.get("original_confirmatory_status") != "INCONCLUSIVE"
        or source_e02.get("original_h2_claim_open") is not False
        or source_e02.get("checkpoint_contract_sha256")
        != config["source_e02"]["strict_checkpoint_contract_sha256"]
    ):
        raise ProvenanceValidationError("E02b does not point to the pinned E02 checkpoint source.")
    checkpoint_records = _require_list(
        source_e02.get("checkpoints"),
        "E02b source checkpoint records",
    )
    expected_checkpoint_hashes = {
        (checkpoint.seed, checkpoint.constraint): checkpoint.file_sha256
        for checkpoint in e02.checkpoints.values()
    }
    observed_checkpoint_hashes: dict[tuple[int, str], str] = {}
    for raw_record in checkpoint_records:
        record = _require_mapping(raw_record, "E02b source checkpoint record")
        key = (int(record["seed"]), str(record["constraint"]))
        observed_checkpoint_hashes[key] = str(record["source_file_sha256"])
    if observed_checkpoint_hashes != expected_checkpoint_hashes:
        raise ProvenanceValidationError("E02b checkpoint lineage differs from pinned E02.")

    registry_name = str(source["claim_registry_filename"])
    registry_path = root / registry_name
    if registry_path.is_symlink() or not registry_path.is_file():
        raise ProvenanceValidationError("E02b additive claim registry is missing.")
    registry_hash = sha256_file(registry_path)
    if registry_hash != source["claim_registry_sha256"]:
        raise ProvenanceValidationError("E02b additive claim registry hash differs.")
    registry = read_json_object_strict(registry_path)
    registry_repair = _require_mapping(
        registry.get("e02b_repair"),
        "E02b claim registry repair",
    )
    registry_original = _require_mapping(
        registry.get("original_e02"),
        "E02b claim registry original E02",
    )
    if (
        registry.get("source_run") != validated.run_id
        or registry.get("source_manifest_sha256") != validated.manifest_sha256
        or registry.get("source_report_sha256") != validated.report_sha256
        or registry_repair.get("status") != "SUPPORTED"
        or registry_repair.get("prospective_repair_claim_open") is not True
        or registry_original.get("confirmatory_status") != "INCONCLUSIVE"
        or registry_original.get("h2_claim_open") is not False
    ):
        raise ProvenanceValidationError("E02b additive claim registry is inconsistent.")
    return PinnedE02bRepair(validated, registry_hash)


def _heldout_base_seed(
    *,
    seed: int,
    base_index: int,
    heldout_seed_offset: int,
) -> int:
    if base_index < 0:
        raise ValueError("base_index must be non-negative.")
    if heldout_seed_offset < 0:
        raise ValueError("heldout_seed_offset must be non-negative.")
    if heldout_seed_offset + base_index >= _SEED_BLOCK_SIZE:
        raise ValueError(
            "heldout_seed_offset + base_index must remain inside one seed block."
        )
    return seed * _SEED_BLOCK_SIZE + heldout_seed_offset + base_index


def _is_baseline_dose(dose: float) -> bool:
    return bool(np.isclose(float(dose), 1.0, rtol=0.0, atol=1e-12))


def _row_key(row: Mapping[str, object]) -> tuple[int, int, str, str, float]:
    missing = [field for field in _ROW_KEY_FIELDS if field not in row]
    if missing:
        raise ValueError(f"Intervention row is missing key fields: {missing}")
    return (
        _integer(row["seed"], "row seed"),
        _integer(row["base_index"], "row base_index"),
        str(row["operation"]),
        str(row["intervention"]),
        _finite(row["dose"], "row dose"),
    )


def _validate_row_identities(
    rows: Sequence[Mapping[str, object]],
    *,
    require_full_schema: bool = False,
) -> None:
    seen: dict[tuple[int, int, str, str, float], int] = {}
    required = {
        *_ROW_KEY_FIELDS,
        "schema_version",
        "pair_block",
        "geometry_seed",
        "quartet_sha256",
        "applied_erase",
        "applied_write",
        "affected_read_mse",
        "unaffected_retention_mse",
        "target_state_mse",
        "old_association_residual",
        "new_write_mse",
    }
    full_schema = all(required.issubset(row) for row in rows)
    if require_full_schema and not full_schema:
        incomplete = [
            (index, sorted(required - set(row)))
            for index, row in enumerate(rows)
            if not required.issubset(row)
        ]
        raise ValueError(f"Intervention rows have incomplete schemas: {incomplete[:3]}.")
    for index, row in enumerate(rows):
        key = _row_key(row)
        if key in seen:
            raise ValueError(
                f"Duplicate intervention row key {key} at rows {seen[key]} and {index}."
            )
        seen[key] = index
        if key[3] == "relevant_dose" and _is_baseline_dose(key[4]):
            raise ValueError(
                "relevant_dose at dose=1 duplicates the canonical baseline row."
            )
        if full_schema:
            if row["schema_version"] != 1:
                raise ValueError(f"Unexpected intervention schema at row {index}.")
            if row["operation"] not in {operation.value for operation in Operation}:
                raise ValueError(f"Unknown intervention operation at row {index}.")
            if _integer(row["pair_block"], f"row {index} pair_block") != key[1] // 2:
                raise ValueError(f"Incorrect donor pair block at row {index}.")
            numeric_fields = (
                "applied_erase",
                "applied_write",
                "affected_read_mse",
                "unaffected_retention_mse",
                "target_state_mse",
                "old_association_residual",
                "new_write_mse",
            )
            for field in numeric_fields:
                _finite(row[field], f"row {index} {field}")
            if not (
                -1e-7
                <= _finite(row["applied_erase"], f"row {index} applied_erase")
                <= 1.0 + 1e-7
                and -1e-7
                <= _finite(row["applied_write"], f"row {index} applied_write")
                <= 1.0 + 1e-7
            ):
                raise ValueError(f"Applied gate leaves [0,1] at row {index}.")


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


def _tensor_digest(named_tensors: Sequence[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in named_tensors:
        local = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(local.dtype).encode("utf-8"))
        digest.update(b"\0")
        digest.update(repr(tuple(local.shape)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(local.numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _quartet_sha256(quartet: Mapping[Operation, MemoryEpisode]) -> str:
    reference = quartet[Operation.PRESERVE]
    return _tensor_digest(
        (
            ("keys", reference.keys),
            ("values", reference.values),
            ("state", reference.state),
            ("old_value", reference.old_value),
            ("new_value", reference.new_value),
            ("erase_candidate", reference.erase_candidate),
            ("write_candidate", reference.write_candidate),
            ("unaffected_indices", reference.unaffected_indices),
        )
    )


def _validate_counterfactual_quartet(
    quartet: Mapping[Operation, MemoryEpisode],
) -> str:
    if set(quartet) != set(Operation):
        raise ValueError("A counterfactual quartet must contain all four operations.")
    reference = quartet[Operation.PRESERVE]
    tensor_fields = (
        "keys",
        "values",
        "state",
        "unaffected_indices",
        "old_value",
        "new_value",
        "erase_candidate",
        "write_candidate",
    )
    metadata_fields = (
        "seed",
        "candidate_mode",
        "key_correlation",
        "state_load",
        "old_scale",
        "new_scale",
        "old_new_cosine",
        "candidate_contamination",
    )
    for operation, episode in quartet.items():
        if episode.operation is not operation:
            raise ValueError("Quartet dictionary key does not match episode operation.")
        if episode.affected_index != reference.affected_index:
            raise ValueError("Counterfactual quartet affected indices differ.")
        for field in tensor_fields:
            if not torch.equal(getattr(episode, field), getattr(reference, field)):
                raise ValueError(f"Counterfactual quartet field {field} differs.")
        for field in metadata_fields:
            if episode.metadata[field] != reference.metadata[field]:
                raise ValueError(f"Counterfactual quartet metadata {field} differs.")
        expected_features = torch.tensor(
            [float(operation is candidate) for candidate in Operation],
            dtype=torch.float32,
        )
        if not torch.equal(episode.operation_features.cpu(), expected_features):
            raise ValueError("Counterfactual operation one-hot is invalid.")
        tensors = (
            episode.state,
            episode.target_state,
            episode.erase_candidate,
            episode.write_candidate,
        )
        if not all(bool(torch.isfinite(tensor).all().item()) for tensor in tensors):
            raise FloatingPointError("Counterfactual quartet contains non-finite tensors.")
    return _quartet_sha256(quartet)


def _donor_base_index(base_index: int, count: int) -> int:
    if count < 2 or count % 2:
        raise ValueError("Adjacent two-cycle donor pairing requires an even count.")
    if not 0 <= base_index < count:
        raise ValueError("base_index lies outside the donor-pairing range.")
    return base_index ^ 1


def _load_frozen_model(
    checkpoint: FrozenCheckpoint,
    *,
    constraint: ScalarConstraint,
    device: torch.device,
) -> MatchedScalarController:
    if checkpoint.constraint != constraint.value:
        raise ProvenanceValidationError("Checkpoint constraint differs from requested model.")
    if sha256_file(checkpoint.path) != checkpoint.file_sha256:
        raise ProvenanceValidationError(f"Checkpoint bytes changed: {checkpoint.path}.")
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
            raise FloatingPointError(f"Non-finite checkpoint tensor {name}.")
        state[name] = value
    if _state_dict_sha256(state) != checkpoint.state_dict_sha256:
        raise ProvenanceValidationError(f"Checkpoint state hash differs: {checkpoint.path}.")
    model = MatchedScalarController(
        checkpoint.input_dim,
        checkpoint.hidden_dim,
        constraint,
    )
    model.load_state_dict(state, strict=True)
    if sum(parameter.numel() for parameter in model.parameters()) != (
        checkpoint.parameter_count
    ):
        raise ProvenanceValidationError("Checkpoint parameter count differs.")
    model.to(device)
    model.eval()
    return model


def _infer_gate_map(
    model: MatchedScalarController,
    quartets: Sequence[Mapping[Operation, MemoryEpisode]],
    *,
    device: torch.device,
) -> dict[tuple[int, Operation], GateOutput]:
    ordered = [
        (base_index, operation, quartet[operation])
        for base_index, quartet in enumerate(quartets)
        for operation in Operation
    ]
    features = torch.stack([controller_features(episode) for _, _, episode in ordered]).to(
        device
    )
    with torch.no_grad():
        output = model(features)
    erase = output.erase.detach()
    write = output.write.detach()
    if erase.shape != (len(ordered),) or write.shape != (len(ordered),):
        raise ValueError("Controller gate batch has an unexpected shape.")
    result: dict[tuple[int, Operation], GateOutput] = {}
    for index, (base_index, operation, _) in enumerate(ordered):
        gates = GateOutput(erase=erase[index], write=write[index])
        vector = gate_vector(gates)
        if bool(((vector < 0.0) | (vector > 1.0)).any().item()):
            raise ValueError("Frozen controller emitted a gate outside [0,1].")
        result[(base_index, operation)] = gates
    return result


def _build_seed_design(
    *,
    checkpoint_seed: int,
    count: int,
    seed_offset: int,
    data: Mapping[str, Any],
    source: PinnedE02Source,
    device: torch.device,
    norm_tolerance: float,
) -> SeedDesign:
    quartets: list[dict[Operation, MemoryEpisode]] = []
    hashes: list[str] = []
    for base_index in range(count):
        geometry_seed = _heldout_base_seed(
            seed=checkpoint_seed,
            base_index=base_index,
            heldout_seed_offset=seed_offset,
        )
        quartet = {
            operation: build_geometry_episode(
                seed=geometry_seed,
                operation=operation,
                candidate_mode=CandidateMode.ORACLE,
                key_dim=int(data["key_dim"]),
                value_dim=int(data["value_dim"]),
                num_associations=int(data["num_associations"]),
                key_correlation=float(data["key_correlation"]),
                old_scale=float(data["old_scale"]),
                new_scale=float(data["new_scale"]),
                old_new_cosine=float(data["old_new_cosine"]),
                episode_index=base_index,
            )
            for operation in Operation
        }
        hashes.append(_validate_counterfactual_quartet(quartet))
        quartets.append(quartet)
    if len(set(hashes)) != count:
        raise ValueError("E04 quartet tensor hashes are not unique within a seed.")

    dual = _load_frozen_model(
        source.checkpoints[(checkpoint_seed, "dual")],
        constraint=ScalarConstraint.DUAL,
        device=device,
    )
    tied = _load_frozen_model(
        source.checkpoints[(checkpoint_seed, "tied")],
        constraint=ScalarConstraint.TIED,
        device=device,
    )
    dual_gates = _infer_gate_map(dual, quartets, device=device)
    tied_gates = _infer_gate_map(tied, quartets, device=device)
    donor_matches: dict[tuple[int, Operation, str], NormMatchResult] = {}
    for base_index in range(count):
        donor_index = _donor_base_index(base_index, count)
        for operation in _CONFIRMATORY_OPERATIONS:
            recipient = dual_gates[(base_index, operation)]
            same_raw = dual_gates[(donor_index, operation)]
            cross_raw = dual_gates[(donor_index, _CROSS_OPERATION[operation])]
            try:
                same_match = exact_feasible_l2_norm_match(
                    same_raw,
                    recipient,
                    tolerance=norm_tolerance,
                )
                cross_match = exact_feasible_l2_norm_match(
                    cross_raw,
                    recipient,
                    tolerance=norm_tolerance,
                )
            except ValueError as error:
                raise DonorNormMatchUnidentifiableError(
                    "Prospectively paired donor norm match is infeasible: "
                    f"checkpoint_seed={checkpoint_seed}, base_index={base_index}, "
                    f"operation={operation.value}, donor_base_index={donor_index}."
                ) from error
            donor_matches[(base_index, operation, "same")] = same_match
            donor_matches[(base_index, operation, "cross")] = cross_match
    return SeedDesign(
        checkpoint_seed=checkpoint_seed,
        quartets=quartets,
        quartet_hashes=hashes,
        dual_gates=dual_gates,
        tied_gates=tied_gates,
        donor_matches=donor_matches,
    )


def _gate_payload(gates: GateOutput) -> dict[str, float]:
    vector = gate_vector(gates)
    return {
        "erase": _finite(vector[0].item(), "erase gate"),
        "write": _finite(vector[1].item(), "write gate"),
        "l2_norm": _finite(torch.linalg.vector_norm(vector).item(), "gate norm"),
    }


def _norm_match_payload(match: NormMatchResult) -> dict[str, object]:
    return {
        "matched_gate": _gate_payload(match.gates),
        "donor_norm": match.donor_norm,
        "recipient_norm": match.recipient_norm,
        "matched_norm": match.matched_norm,
        "absolute_mismatch": match.absolute_mismatch,
        "scale": match.scale,
        "clipping_used": False,
    }


def _design_lock_rows(designs: Sequence[SeedDesign]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for design in designs:
        count = len(design.quartets)
        for base_index in range(count):
            donor_index = _donor_base_index(base_index, count)
            for operation in _CONFIRMATORY_OPERATIONS:
                cross_operation = _CROSS_OPERATION[operation]
                rows.append(
                    {
                        "schema_version": 1,
                        "checkpoint_seed": design.checkpoint_seed,
                        "base_index": base_index,
                        "pair_block": base_index // 2,
                        "geometry_seed": int(
                            design.quartets[base_index][operation].metadata["seed"]
                        ),
                        "quartet_sha256": design.quartet_hashes[base_index],
                        "donor_base_index": donor_index,
                        "donor_geometry_seed": int(
                            design.quartets[donor_index][operation].metadata["seed"]
                        ),
                        "donor_quartet_sha256": design.quartet_hashes[donor_index],
                        "recipient_operation": operation.value,
                        "cross_operation": cross_operation.value,
                        "recipient_gate": _gate_payload(
                            design.dual_gates[(base_index, operation)]
                        ),
                        "same_raw_gate": _gate_payload(
                            design.dual_gates[(donor_index, operation)]
                        ),
                        "cross_raw_gate": _gate_payload(
                            design.dual_gates[(donor_index, cross_operation)]
                        ),
                        "same_norm_match": _norm_match_payload(
                            design.donor_matches[(base_index, operation, "same")]
                        ),
                        "cross_norm_match": _norm_match_payload(
                            design.donor_matches[(base_index, operation, "cross")]
                        ),
                        "donor_pairing": "adjacent_two_cycle",
                        "donor_selection_uses_outcomes": False,
                    }
                )
    return rows


def _quartet_registry_rows(designs: Sequence[SeedDesign]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for design in designs:
        count = len(design.quartets)
        for base_index, quartet in enumerate(design.quartets):
            rows.append(
                {
                    "schema_version": 1,
                    "checkpoint_seed": design.checkpoint_seed,
                    "base_index": base_index,
                    "pair_block": base_index // 2,
                    "donor_base_index": _donor_base_index(base_index, count),
                    "geometry_seed": int(
                        quartet[Operation.PRESERVE].metadata["seed"]
                    ),
                    "quartet_sha256": design.quartet_hashes[base_index],
                    "candidate_mode": CandidateMode.ORACLE.value,
                    "operation_identity_only_difference_verified": True,
                    "dual_gates": {
                        operation.value: _gate_payload(
                            design.dual_gates[(base_index, operation)]
                        )
                        for operation in Operation
                    },
                    "tied_gates": {
                        operation.value: _gate_payload(
                            design.tied_gates[(base_index, operation)]
                        )
                        for operation in Operation
                    },
                }
            )
    return rows


def _evaluate_gates(episode: MemoryEpisode, gates: GateOutput) -> EpisodeMetrics:
    output = apply_scalar_update(episode, gates.erase, gates.write)
    if not bool(torch.isfinite(output).all().item()):
        raise FloatingPointError("Intervention output contains a non-finite value.")
    return evaluate_episode(output, episode)


def _condition_row(
    *,
    seed: int,
    base_index: int,
    geometry_seed: int,
    quartet_sha256: str,
    operation: Operation,
    intervention: str,
    condition_role: str,
    dose: float,
    gates: GateOutput,
    metrics: EpisodeMetrics,
    donor_base_index: int | None = None,
    donor_operation: str | None = None,
    norm_match: NormMatchResult | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "seed": seed,
        "base_index": base_index,
        "pair_block": base_index // 2,
        "geometry_seed": geometry_seed,
        "quartet_sha256": quartet_sha256,
        "candidate_mode": CandidateMode.ORACLE.value,
        "operation": operation.value,
        "intervention": intervention,
        "condition_role": condition_role,
        "dose": dose,
        "applied_erase": _finite(gates.erase.item(), "applied erase gate"),
        "applied_write": _finite(gates.write.item(), "applied write gate"),
        "donor_base_index": donor_base_index,
        "donor_operation": donor_operation,
        "donor_raw_norm": None,
        "recipient_norm": None,
        "matched_norm": None,
        "norm_mismatch": None,
        "norm_scale": None,
        **metrics.to_dict(),
    }
    if norm_match is not None:
        payload.update(
            {
                "donor_raw_norm": norm_match.donor_norm,
                "recipient_norm": norm_match.recipient_norm,
                "matched_norm": norm_match.matched_norm,
                "norm_mismatch": norm_match.absolute_mismatch,
                "norm_scale": norm_match.scale,
            }
        )
    return payload


def _expected_rows(seed_count: int, base_count: int) -> int:
    # Per quartet: 44 common rows + 4 SUPERSEDE joint-dose rows +
    # 4 ADD/INVALIDATE transplant rows + 8 ADD/INVALIDATE rescue rows.
    return seed_count * base_count * 60


def _pair_block_mean(values: np.ndarray, name: str) -> np.ndarray:
    vector = _finite_vector(values, name)
    if len(vector) % 2:
        raise ValueError(f"{name} cannot be grouped into adjacent donor pairs.")
    return cast(np.ndarray, vector.reshape(-1, 2).mean(axis=1))


def _bootstrap_interval(
    values_by_seed: Mapping[int, np.ndarray],
    *,
    samples: int,
    seed: int,
    confidence: float,
) -> Interval:
    values = {
        checkpoint_seed: _finite_vector(vector, f"bootstrap seed {checkpoint_seed}")
        for checkpoint_seed, vector in values_by_seed.items()
    }
    strata = {
        checkpoint_seed: np.full(len(vector), "donor_pair_block")
        for checkpoint_seed, vector in values.items()
    }

    def statistic(indices: Mapping[int, np.ndarray]) -> float:
        return float(
            np.mean(
                [
                    values[checkpoint_seed][indices[checkpoint_seed]].mean()
                    for checkpoint_seed in sorted(values)
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


def _bootstrap_ratio_interval(
    numerator_by_seed: Mapping[int, np.ndarray],
    denominator_by_seed: Mapping[int, np.ndarray],
    *,
    minimum_denominator: float,
    samples: int,
    seed: int,
    confidence: float,
) -> Interval:
    if set(numerator_by_seed) != set(denominator_by_seed) or not numerator_by_seed:
        raise ValueError("Ratio numerator and denominator seeds differ.")
    numerator = {
        checkpoint_seed: _finite_vector(vector, f"ratio numerator {checkpoint_seed}")
        for checkpoint_seed, vector in numerator_by_seed.items()
    }
    denominator = {
        checkpoint_seed: _finite_vector(vector, f"ratio denominator {checkpoint_seed}")
        for checkpoint_seed, vector in denominator_by_seed.items()
    }
    if any(
        len(numerator[checkpoint_seed]) != len(denominator[checkpoint_seed])
        for checkpoint_seed in numerator
    ):
        raise ValueError("Ratio numerator and denominator lengths differ.")
    strata = {
        checkpoint_seed: np.full(len(vector), "donor_pair_block")
        for checkpoint_seed, vector in numerator.items()
    }

    def statistic(indices: Mapping[int, np.ndarray]) -> float:
        ratios: list[float] = []
        for checkpoint_seed in sorted(numerator):
            selected = indices[checkpoint_seed]
            denominator_mean = float(denominator[checkpoint_seed][selected].mean())
            if denominator_mean <= minimum_denominator:
                raise BootstrapRatioHeadroomError(
                    "Bootstrap scalarization denominator lacks registered headroom."
                )
            ratios.append(
                float(numerator[checkpoint_seed][selected].mean())
                / denominator_mean
            )
        return float(np.mean(ratios))

    return fixed_seed_operation_stratified_bootstrap(
        strata,
        statistic,
        samples=samples,
        seed=seed,
        confidence=confidence,
    )


def _interval_payload(interval: Interval) -> dict[str, object]:
    return {
        "estimate": interval.estimate,
        "ci95": [interval.low, interval.high],
    }


def _positive_gate(
    values_by_seed: Mapping[int, np.ndarray],
    *,
    threshold: float,
    alpha: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
    confidence: float,
    inference_eligible: bool,
) -> dict[str, object]:
    interval = _bootstrap_interval(
        values_by_seed,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
        confidence=confidence,
    )
    seed_values = {
        checkpoint_seed: float(values.mean())
        for checkpoint_seed, values in sorted(values_by_seed.items())
    }
    shifted = np.asarray(list(seed_values.values()), dtype=np.float64) - threshold
    p_value = exact_sign_flip_test(shifted, "greater")
    passed = bool(interval.low > threshold and p_value <= alpha)
    return {
        **_interval_payload(interval),
        "seed_values": {str(seed): value for seed, value in seed_values.items()},
        "threshold": threshold,
        "bootstrap_lower_above_threshold": bool(interval.low > threshold),
        "seed_exact_sign_flip_p_above_threshold": p_value,
        "inference_eligible": inference_eligible,
        "supported": bool(inference_eligible and passed),
    }


def _equivalence_gate(
    values_by_seed: Mapping[int, np.ndarray],
    *,
    margin: float,
    alpha: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
    confidence: float,
    inference_eligible: bool,
) -> dict[str, object]:
    interval = _bootstrap_interval(
        values_by_seed,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
        confidence=confidence,
    )
    seed_values = {
        checkpoint_seed: float(values.mean())
        for checkpoint_seed, values in sorted(values_by_seed.items())
    }
    vector = np.asarray(list(seed_values.values()), dtype=np.float64)
    lower_p = exact_sign_flip_test(vector + margin, "greater")
    upper_p = exact_sign_flip_test(vector - margin, "less")
    ci_within = bool(interval.low >= -margin and interval.high <= margin)
    tost_passed = bool(lower_p <= alpha and upper_p <= alpha)
    return {
        **_interval_payload(interval),
        "seed_values": {str(seed): value for seed, value in seed_values.items()},
        "margin": margin,
        "ci_within_margin": ci_within,
        "seed_exact_tost": {
            "lower_p": lower_p,
            "upper_p": upper_p,
            "passed": tost_passed,
        },
        "inference_eligible": inference_eligible,
        "supported": bool(inference_eligible and ci_within and tost_passed),
    }


def _noninferiority_gate(
    values_by_seed: Mapping[int, np.ndarray],
    *,
    margin: float,
    alpha: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
    confidence: float,
    inference_eligible: bool,
) -> dict[str, object]:
    interval = _bootstrap_interval(
        values_by_seed,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
        confidence=confidence,
    )
    seed_values = {
        checkpoint_seed: float(values.mean())
        for checkpoint_seed, values in sorted(values_by_seed.items())
    }
    vector = np.asarray(list(seed_values.values()), dtype=np.float64)
    p_value = exact_sign_flip_test(vector - margin, "less")
    return {
        **_interval_payload(interval),
        "seed_values": {str(seed): value for seed, value in seed_values.items()},
        "margin": margin,
        "ci_upper_within_margin": bool(interval.high <= margin),
        "seed_exact_sign_flip_p_below_margin": p_value,
        "inference_eligible": inference_eligible,
        "supported": bool(
            inference_eligible and interval.high <= margin and p_value <= alpha
        ),
    }


def _status(
    *,
    dry_run: bool,
    inference_eligible: bool,
    supported: bool,
) -> tuple[str, str | None]:
    if dry_run:
        return "NOT_EVALUATED_DRY_RUN", "DRY_RUN"
    if not inference_eligible:
        return "INCONCLUSIVE", "MAIN_EXECUTION_OR_DESIGN_INCOMPLETE"
    if supported:
        return "SUPPORTED", None
    return "NOT_SUPPORTED", "ONE_OR_MORE_EVALUABLE_GATES_FAILED"


def _evaluate_design(
    designs: Sequence[SeedDesign],
    *,
    doses: tuple[float, ...],
    minimum_headroom: float,
    monotonic_tolerance: float,
    self_tolerance: float,
) -> tuple[
    list[dict[str, object]],
    dict[str, dict[int, np.ndarray]],
    list[dict[str, object]],
    dict[str, object],
]:
    rows: list[dict[str, object]] = []
    effects: dict[str, dict[int, np.ndarray]] = {}
    seed_rows: list[dict[str, object]] = []
    maximum_self_restoration_error = 0.0

    def put_effect(name: str, seed: int, values: np.ndarray) -> None:
        effects.setdefault(name, {})[seed] = _pair_block_mean(values, name)

    for design in designs:
        seed = design.checkpoint_seed
        count = len(design.quartets)
        condition_metrics: dict[
            tuple[int, Operation, str, float],
            EpisodeMetrics,
        ] = {}

        def record(
            *,
            base_index: int,
            operation: Operation,
            intervention: str,
            role: str,
            dose: float,
            gates: GateOutput,
            donor_base_index: int | None = None,
            donor_operation: str | None = None,
            norm_match: NormMatchResult | None = None,
            _design: SeedDesign = design,
            _condition_metrics: dict[
                tuple[int, Operation, str, float],
                EpisodeMetrics,
            ] = condition_metrics,
            _seed: int = seed,
        ) -> EpisodeMetrics:
            episode_cpu = _design.quartets[base_index][operation]
            episode = episode_cpu.to(gates.erase.device)
            metrics = _evaluate_gates(episode, gates)
            _condition_metrics[
                (base_index, operation, intervention, dose)
            ] = metrics
            rows.append(
                _condition_row(
                    seed=_seed,
                    base_index=base_index,
                    geometry_seed=int(episode_cpu.metadata["seed"]),
                    quartet_sha256=_design.quartet_hashes[base_index],
                    operation=operation,
                    intervention=intervention,
                    condition_role=role,
                    dose=dose,
                    gates=gates,
                    metrics=metrics,
                    donor_base_index=donor_base_index,
                    donor_operation=donor_operation,
                    norm_match=norm_match,
                )
            )
            return metrics

        for base_index in range(count):
            donor_index = _donor_base_index(base_index, count)
            for operation in Operation:
                baseline_gate = design.dual_gates[(base_index, operation)]
                baseline_metrics = record(
                    base_index=base_index,
                    operation=operation,
                    intervention="baseline",
                    role="dual_frozen_baseline",
                    dose=1.0,
                    gates=baseline_gate,
                )
                record(
                    base_index=base_index,
                    operation=operation,
                    intervention="trained_tied",
                    role="architecture_total_effect_comparator",
                    dose=1.0,
                    gates=design.tied_gates[(base_index, operation)],
                )
                record(
                    base_index=base_index,
                    operation=operation,
                    intervention="posthoc_scalarized",
                    role="architecture_mediated_effect",
                    dose=1.0,
                    gates=scalarize_gate(baseline_gate),
                )
                for channel in (GateChannel.ERASE, GateChannel.WRITE):
                    for dose in doses:
                        if _is_baseline_dose(dose):
                            continue
                        record(
                            base_index=base_index,
                            operation=operation,
                            intervention=f"{channel.value}_dose",
                            role="physical_channel_dose",
                            dose=dose,
                            gates=dose_gate(baseline_gate, channel, dose),
                        )
                if operation is Operation.SUPERSEDE:
                    for dose in doses:
                        if _is_baseline_dose(dose):
                            continue
                        record(
                            base_index=base_index,
                            operation=operation,
                            intervention="joint_dose",
                            role="secondary_compositional_dose",
                            dose=dose,
                            gates=dose_gate(baseline_gate, GateChannel.JOINT, dose),
                        )
                if operation not in _CONFIRMATORY_OPERATIONS:
                    continue

                same_match = design.donor_matches[(base_index, operation, "same")]
                cross_match = design.donor_matches[(base_index, operation, "cross")]
                record(
                    base_index=base_index,
                    operation=operation,
                    intervention="same_operation_transplant",
                    role="confirmatory_transplant",
                    dose=1.0,
                    gates=same_match.gates,
                    donor_base_index=donor_index,
                    donor_operation=operation.value,
                    norm_match=same_match,
                )
                record(
                    base_index=base_index,
                    operation=operation,
                    intervention="cross_operation_transplant",
                    role="confirmatory_transplant",
                    dose=1.0,
                    gates=cross_match.gates,
                    donor_base_index=donor_index,
                    donor_operation=_CROSS_OPERATION[operation].value,
                    norm_match=cross_match,
                )
                relevant_channel = (
                    GateChannel.WRITE
                    if operation is Operation.ADD
                    else GateChannel.ERASE
                )
                damaged_gate = dose_gate(baseline_gate, relevant_channel, 0.0)
                self_gate = restore_relevant(damaged_gate, baseline_gate, operation)
                same_rescue_gate = restore_relevant(
                    damaged_gate,
                    same_match.gates,
                    operation,
                )
                cross_rescue_gate = restore_relevant(
                    damaged_gate,
                    cross_match.gates,
                    operation,
                )
                demand = operation.demand
                oracle_gate = GateOutput(
                    erase=torch.tensor(
                        demand[0],
                        device=baseline_gate.erase.device,
                        dtype=baseline_gate.erase.dtype,
                    ),
                    write=torch.tensor(
                        demand[1],
                        device=baseline_gate.write.device,
                        dtype=baseline_gate.write.dtype,
                    ),
                )
                oracle_rescue_gate = restore_relevant(
                    damaged_gate,
                    oracle_gate,
                    operation,
                )
                self_metrics = record(
                    base_index=base_index,
                    operation=operation,
                    intervention="self_relevant_restoration_sanity",
                    role="sanity_only_not_confirmatory_rescue",
                    dose=1.0,
                    gates=self_gate,
                    donor_base_index=base_index,
                    donor_operation=operation.value,
                )
                record(
                    base_index=base_index,
                    operation=operation,
                    intervention="same_operation_relevant_rescue",
                    role="confirmatory_independent_donor_rescue",
                    dose=1.0,
                    gates=same_rescue_gate,
                    donor_base_index=donor_index,
                    donor_operation=operation.value,
                    norm_match=same_match,
                )
                record(
                    base_index=base_index,
                    operation=operation,
                    intervention="cross_operation_relevant_rescue",
                    role="confirmatory_cross_donor_control",
                    dose=1.0,
                    gates=cross_rescue_gate,
                    donor_base_index=donor_index,
                    donor_operation=_CROSS_OPERATION[operation].value,
                    norm_match=cross_match,
                )
                record(
                    base_index=base_index,
                    operation=operation,
                    intervention="oracle_relevant_rescue",
                    role="assay_positive_control",
                    dose=1.0,
                    gates=oracle_rescue_gate,
                    donor_operation="oracle_demand",
                )
                maximum_self_restoration_error = max(
                    maximum_self_restoration_error,
                    abs(
                        self_metrics.affected_read_mse
                        - baseline_metrics.affected_read_mse
                    ),
                    abs(
                        self_metrics.unaffected_retention_mse
                        - baseline_metrics.unaffected_retention_mse
                    ),
                )

        def metric(
            base_index: int,
            operation: Operation,
            intervention: str,
            dose: float = 1.0,
            _condition_metrics: dict[
                tuple[int, Operation, str, float],
                EpisodeMetrics,
            ] = condition_metrics,
        ) -> EpisodeMetrics:
            return _condition_metrics[(base_index, operation, intervention, dose)]

        def dose_metric(
            base_index: int,
            operation: Operation,
            channel: GateChannel,
            dose: float,
        ) -> EpisodeMetrics:
            if _is_baseline_dose(dose):
                return metric(base_index, operation, "baseline")
            return metric(base_index, operation, f"{channel.value}_dose", dose)

        component_damage: dict[tuple[Operation, GateChannel], np.ndarray] = {}
        for operation in Operation:
            for channel in (GateChannel.ERASE, GateChannel.WRITE):
                component_damage[(operation, channel)] = np.asarray(
                    [
                        dose_metric(base, operation, channel, 0.0).affected_read_mse
                        - metric(base, operation, "baseline").affected_read_mse
                        for base in range(count)
                    ],
                    dtype=np.float64,
                )

        primary = 0.5 * (
            component_damage[(Operation.ADD, GateChannel.WRITE)]
            - component_damage[(Operation.ADD, GateChannel.ERASE)]
            + component_damage[(Operation.INVALIDATE, GateChannel.ERASE)]
            - component_damage[(Operation.INVALIDATE, GateChannel.WRITE)]
        )
        put_effect("primary_specificity", seed, primary)
        component_names = {
            (Operation.ADD, GateChannel.WRITE): "add_write_damage",
            (Operation.INVALIDATE, GateChannel.ERASE): "invalidate_erase_damage",
            (Operation.ADD, GateChannel.ERASE): "add_erase_damage",
            (Operation.INVALIDATE, GateChannel.WRITE): "invalidate_write_damage",
            (Operation.SUPERSEDE, GateChannel.ERASE): "supersede_erase_damage",
            (Operation.SUPERSEDE, GateChannel.WRITE): "supersede_write_damage",
            (Operation.PRESERVE, GateChannel.ERASE): "preserve_erase_damage",
            (Operation.PRESERVE, GateChannel.WRITE): "preserve_write_damage",
        }
        for key, name in component_names.items():
            put_effect(name, seed, component_damage[key])

        dose_curves: dict[str, list[float]] = {}
        monotonic_scores: dict[str, float] = {}
        for operation in Operation:
            for channel in (GateChannel.ERASE, GateChannel.WRITE):
                label = f"{operation.value}/{channel.value}"
                curve = [
                    float(
                        np.mean(
                            [
                                dose_metric(base, operation, channel, dose).affected_read_mse
                                for base in range(count)
                            ]
                        )
                    )
                    for dose in doses
                ]
                dose_curves[label] = curve
                monotonic_scores[label] = monotonic_nonincreasing_fraction(
                    np.asarray(curve),
                    tolerance=monotonic_tolerance,
                )
        joint_curve = [
            float(
                np.mean(
                    [
                        (
                            metric(base, Operation.SUPERSEDE, "baseline")
                            if _is_baseline_dose(dose)
                            else metric(base, Operation.SUPERSEDE, "joint_dose", dose)
                        ).affected_read_mse
                        for base in range(count)
                    ]
                )
            )
            for dose in doses
        ]
        dose_curves["supersede/joint"] = joint_curve
        monotonic_scores["supersede/joint"] = monotonic_nonincreasing_fraction(
            np.asarray(joint_curve),
            tolerance=monotonic_tolerance,
        )

        transplant_gap = np.asarray(
            [
                np.mean(
                    [
                        metric(
                            base,
                            operation,
                            "cross_operation_transplant",
                        ).affected_read_mse
                        - metric(
                            base,
                            operation,
                            "same_operation_transplant",
                        ).affected_read_mse
                        for operation in _CONFIRMATORY_OPERATIONS
                    ]
                )
                for base in range(count)
            ],
            dtype=np.float64,
        )
        transplant_baseline = np.asarray(
            [
                np.mean(
                    [
                        metric(
                            base,
                            operation,
                            "same_operation_transplant",
                        ).affected_read_mse
                        - metric(base, operation, "baseline").affected_read_mse
                        for operation in _CONFIRMATORY_OPERATIONS
                    ]
                )
                for base in range(count)
            ],
            dtype=np.float64,
        )
        transplant_retention = np.asarray(
            [
                np.mean(
                    [
                        metric(
                            base,
                            operation,
                            "same_operation_transplant",
                        ).unaffected_retention_mse
                        - metric(base, operation, "baseline").unaffected_retention_mse
                        for operation in _CONFIRMATORY_OPERATIONS
                    ]
                )
                for base in range(count)
            ],
            dtype=np.float64,
        )
        put_effect("transplant_gap", seed, transplant_gap)
        put_effect("transplant_baseline_difference", seed, transplant_baseline)
        put_effect("transplant_retention_difference", seed, transplant_retention)

        rescue_gap = np.asarray(
            [
                np.mean(
                    [
                        metric(
                            base,
                            operation,
                            "cross_operation_relevant_rescue",
                        ).affected_read_mse
                        - metric(
                            base,
                            operation,
                            "same_operation_relevant_rescue",
                        ).affected_read_mse
                        for operation in _CONFIRMATORY_OPERATIONS
                    ]
                )
                for base in range(count)
            ],
            dtype=np.float64,
        )
        rescue_baseline = np.asarray(
            [
                np.mean(
                    [
                        metric(
                            base,
                            operation,
                            "same_operation_relevant_rescue",
                        ).affected_read_mse
                        - metric(base, operation, "baseline").affected_read_mse
                        for operation in _CONFIRMATORY_OPERATIONS
                    ]
                )
                for base in range(count)
            ],
            dtype=np.float64,
        )
        rescue_raw_recovery = np.asarray(
            [
                np.mean(
                    [
                        dose_metric(
                            base,
                            operation,
                            (
                                GateChannel.WRITE
                                if operation is Operation.ADD
                                else GateChannel.ERASE
                            ),
                            0.0,
                        ).affected_read_mse
                        - metric(
                            base,
                            operation,
                            "same_operation_relevant_rescue",
                        ).affected_read_mse
                        for operation in _CONFIRMATORY_OPERATIONS
                    ]
                )
                for base in range(count)
            ],
            dtype=np.float64,
        )
        oracle_rescue_baseline = np.asarray(
            [
                np.mean(
                    [
                        metric(
                            base,
                            operation,
                            "oracle_relevant_rescue",
                        ).affected_read_mse
                        - metric(base, operation, "baseline").affected_read_mse
                        for operation in _CONFIRMATORY_OPERATIONS
                    ]
                )
                for base in range(count)
            ],
            dtype=np.float64,
        )
        rescue_retention = np.asarray(
            [
                np.mean(
                    [
                        metric(
                            base,
                            operation,
                            "same_operation_relevant_rescue",
                        ).unaffected_retention_mse
                        - metric(base, operation, "baseline").unaffected_retention_mse
                        for operation in _CONFIRMATORY_OPERATIONS
                    ]
                )
                for base in range(count)
            ],
            dtype=np.float64,
        )
        put_effect("rescue_gap", seed, rescue_gap)
        put_effect("rescue_baseline_difference", seed, rescue_baseline)
        put_effect("rescue_raw_recovery", seed, rescue_raw_recovery)
        put_effect("oracle_rescue_baseline_difference", seed, oracle_rescue_baseline)
        put_effect("rescue_retention_difference", seed, rescue_retention)

        total_effect = np.asarray(
            [
                np.mean(
                    [
                        metric(base, operation, "trained_tied").affected_read_mse
                        - metric(base, operation, "baseline").affected_read_mse
                        for operation in _CONFIRMATORY_OPERATIONS
                    ]
                )
                for base in range(count)
            ],
            dtype=np.float64,
        )
        mediated_effect = np.asarray(
            [
                np.mean(
                    [
                        metric(base, operation, "posthoc_scalarized").affected_read_mse
                        - metric(base, operation, "baseline").affected_read_mse
                        for operation in _CONFIRMATORY_OPERATIONS
                    ]
                )
                for base in range(count)
            ],
            dtype=np.float64,
        )
        symmetric_scalarization = np.asarray(
            [
                np.mean(
                    [
                        metric(base, operation, "posthoc_scalarized").affected_read_mse
                        - metric(base, operation, "baseline").affected_read_mse
                        for operation in (Operation.PRESERVE, Operation.SUPERSEDE)
                    ]
                )
                for base in range(count)
            ],
            dtype=np.float64,
        )
        put_effect("total_effect", seed, total_effect)
        put_effect("mediated_effect", seed, mediated_effect)
        put_effect(
            "symmetric_scalarization_difference",
            seed,
            symmetric_scalarization,
        )

        damaged_mean = float(
            np.mean(
                [
                    dose_metric(
                        base,
                        operation,
                        (
                            GateChannel.WRITE
                            if operation is Operation.ADD
                            else GateChannel.ERASE
                        ),
                        0.0,
                    ).affected_read_mse
                    for base in range(count)
                    for operation in _CONFIRMATORY_OPERATIONS
                ]
            )
        )
        same_rescue_mean = float(
            np.mean(
                [
                    metric(
                        base,
                        operation,
                        "same_operation_relevant_rescue",
                    ).affected_read_mse
                    for base in range(count)
                    for operation in _CONFIRMATORY_OPERATIONS
                ]
            )
        )
        baseline_mean = float(
            np.mean(
                [
                    metric(base, operation, "baseline").affected_read_mse
                    for base in range(count)
                    for operation in _CONFIRMATORY_OPERATIONS
                ]
            )
        )
        try:
            donor_recovery_fraction: float | None = recovery_fraction_from_means(
                damaged_error=damaged_mean,
                rescued_error=same_rescue_mean,
                baseline_error=baseline_mean,
                minimum_headroom=minimum_headroom,
            )
        except ValueError:
            donor_recovery_fraction = None
        seed_rows.append(
            {
                "schema_version": 1,
                "checkpoint_seed": seed,
                "base_count": count,
                "pair_block_count": count // 2,
                "metric_seed_means": {
                    name: float(by_seed[seed].mean())
                    for name, by_seed in effects.items()
                    if seed in by_seed
                },
                "dose_curve_seed_means": dose_curves,
                "dose_monotonicity_scores": monotonic_scores,
                "same_donor_recovery_fraction_from_seed_means": (
                    donor_recovery_fraction
                ),
                "maximum_self_restoration_absolute_error": (
                    maximum_self_restoration_error
                ),
            }
        )

    design_diagnostics: dict[str, object] = {
        "maximum_self_restoration_absolute_error": maximum_self_restoration_error,
        "self_restoration_tolerance": self_tolerance,
        "self_restoration_passed": bool(
            maximum_self_restoration_error <= self_tolerance
        ),
    }
    return rows, effects, seed_rows, design_diagnostics


def main() -> None:
    parser = build_parser(EXPERIMENT_ID, DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = Path(args.config).resolve(strict=True)
    preloaded = load_config(config_path)
    _validate_protocol_config(preloaded, config_path)
    source_e02 = validate_pinned_e02_source(args.artifact_root, preloaded)
    source_e02b = _validate_e02b_repair(
        args.artifact_root,
        preloaded,
        source_e02,
    )
    e00 = validate_legacy_e00(args.artifact_root, require_full=not args.dry_run)
    dependencies = [
        e00,
        source_e02b.dependency_record(),
        source_e02.dependency_record(),
    ]
    config, run_dir, device, context = initialize_v61_run(
        experiment_id=EXPERIMENT_ID,
        config_path=str(config_path),
        artifact_root=args.artifact_root,
        device_request=args.device,
        dry_run=args.dry_run,
        dependencies=dependencies,
    )
    _validate_protocol_config(config, config_path)

    execution_config = _require_mapping(config.get("execution"), "config.execution")
    cpu_threads = int(execution_config["cpu_threads"])
    if cpu_threads != 1:
        raise ValueError("E04 cpu_threads differs from the frozen protocol.")
    torch.set_num_threads(cpu_threads)
    data = _require_mapping(config.get("data"), "config.data")
    interventions = _require_mapping(
        config.get("interventions"),
        "config.interventions",
    )
    statistics = _require_mapping(config.get("statistics"), "config.statistics")
    bootstrap_seeds = _require_mapping(
        statistics.get("bootstrap_seeds"),
        "config.statistics.bootstrap_seeds",
    )
    seeds = [int(value) for value in _require_list(config.get("seeds"), "config.seeds")]
    count = int(data["base_count"])
    seed_namespace = _require_mapping(
        config.get("seed_namespace"),
        "config.seed_namespace",
    )
    seed_offset = int(seed_namespace["main_offset"])
    if args.dry_run:
        seeds = seeds[:1]
        count = int(data["dry_run_base_count"])
        seed_offset = int(seed_namespace["dry_run_offset"])
    if (
        int(seed_namespace["block_size"]) != _SEED_BLOCK_SIZE
        or int(data["base_count"]) != _EXPECTED_MAIN_BASE_COUNT
        or int(data["dry_run_base_count"]) != _EXPECTED_DRY_BASE_COUNT
        or int(seed_namespace["main_offset"]) != _EXPECTED_MAIN_OFFSET
        or int(seed_namespace["dry_run_offset"]) != _EXPECTED_DRY_OFFSET
        or count % 2
    ):
        raise ValueError("E04 data or seed namespace differs from the frozen protocol.")
    if seed_offset + count > _SEED_BLOCK_SIZE:
        raise ValueError("E04 seed namespace leaves its checkpoint-seed block.")
    doses = tuple(
        _finite(value, "intervention dose")
        for value in _require_list(interventions.get("doses"), "intervention doses")
    )
    if doses != (0.0, 0.25, 0.5, 0.75, 1.0):
        raise ValueError("E04 dose schedule differs from the frozen protocol.")
    if interventions.get("donor_pairing") != "adjacent_two_cycle":
        raise ValueError("E04 donor pairing differs from the frozen protocol.")

    norm_tolerance = _finite(
        interventions["norm_match_tolerance"],
        "norm match tolerance",
    )
    monotonic_tolerance = _finite(
        interventions["dose_monotonic_tolerance"],
        "dose monotonic tolerance",
    )
    alpha = _finite(statistics["alpha"], "alpha")
    bootstrap_samples = int(statistics["bootstrap_samples"])
    confidence = _finite(statistics["bootstrap_confidence"], "bootstrap confidence")
    positive_sesoi = _finite(statistics["positive_effect_sesoi"], "positive SESOI")
    equivalence_margin = _finite(
        statistics["equivalence_margin"],
        "equivalence margin",
    )
    minimum_headroom = _finite(
        statistics["minimum_rescue_headroom"],
        "minimum rescue headroom",
    )
    minimum_monotonic = _finite(
        statistics["minimum_monotonic_fraction"],
        "minimum monotonic fraction",
    )
    minimum_scalarization = _finite(
        statistics["minimum_scalarization_fraction"],
        "minimum scalarization fraction",
    )
    self_tolerance = _finite(
        statistics["self_restoration_absolute_tolerance"],
        "self restoration tolerance",
    )

    try:
        designs = [
            _build_seed_design(
                checkpoint_seed=checkpoint_seed,
                count=count,
                seed_offset=seed_offset,
                data=data,
                source=source_e02,
                device=device,
                norm_tolerance=norm_tolerance,
            )
            for checkpoint_seed in seeds
        ]
    except DonorNormMatchUnidentifiableError as error:
        failure_path = run_dir / "design_validity_failure.json"
        write_json_strict(
            failure_path,
            {
                "schema_version": 1,
                "status": "DONOR_NORM_MATCH_UNIDENTIFIABLE",
                "detail": str(error),
                "intervention_outcomes_evaluated": False,
            },
        )
        failure_hash = sha256_file(failure_path)
        _validate_protocol_config(config, config_path)
        rechecked_e00 = validate_legacy_e00(
            args.artifact_root,
            require_full=not args.dry_run,
        )
        rechecked_e02 = validate_pinned_e02_source(args.artifact_root, config)
        rechecked_e02b = _validate_e02b_repair(
            args.artifact_root,
            config,
            rechecked_e02,
        )
        if (
            rechecked_e00 != e00
            or rechecked_e02.run.manifest_sha256
            != source_e02.run.manifest_sha256
            or rechecked_e02.run.report_sha256 != source_e02.run.report_sha256
            or rechecked_e02b.run.manifest_sha256
            != source_e02b.run.manifest_sha256
            or rechecked_e02b.run.report_sha256
            != source_e02b.run.report_sha256
            or sha256_file(failure_path) != failure_hash
        ):
            raise ProvenanceValidationError(
                "E04 dependency or design-failure record changed before finalize."
            ) from error
        for checkpoint in rechecked_e02.checkpoints.values():
            if sha256_file(checkpoint.path) != checkpoint.file_sha256:
                raise ProvenanceValidationError(
                    "E04 source checkpoint changed before design adjudication."
                ) from error
        design_status = (
            "NOT_EVALUATED_DRY_RUN" if args.dry_run else "INCONCLUSIVE"
        )
        design_reason = (
            "DRY_RUN"
            if args.dry_run
            else "DONOR_NORM_MATCH_UNIDENTIFIABLE"
        )
        finalize_v61_run(
            context=context,
            report={
                "status": "PASS",
                "execution": {
                    "dry_run": args.dry_run,
                    "checkpoint_only": True,
                    "training_executed": False,
                    "optimizer_created": False,
                    "requested_seeds": seeds,
                    "requested_base_count_per_seed": count,
                    "protocol_execution_complete": False,
                    "intervention_outcome_rows": 0,
                },
                "dependency_lineage": dependencies,
                "design_validity": {
                    "status": "UNIDENTIFIABLE",
                    "reason": "DONOR_NORM_MATCH_UNIDENTIFIABLE",
                    "intervention_outcomes_evaluated": False,
                },
                "claim_gate": {
                    "evaluated": False,
                    "inference_eligible": False,
                    "functional_specificity_status": design_status,
                    "architecture_gap_mediation_status": design_status,
                    "status": design_status,
                    "reason": design_reason,
                    "supported": False,
                    "full_h4_claim_open": False,
                    "allowed_claim": None,
                    "forbidden_claims": config["claim"]["forbidden"],
                    "original_e02_remains_inconclusive": True,
                    "e02b_repair_remains_supported": True,
                },
                "artifacts": {
                    "design_validity_failure": {
                        "path": failure_path.name,
                        "sha256": failure_hash,
                    }
                },
                "protocol_lineage": {
                    "implementation_amendment_01": {
                        "path": (
                            "docs/E04_IMPLEMENTATION_AMENDMENT_01_FROZEN_KO.md"
                        ),
                        "sha256": _PINNED_IMPLEMENTATION_AMENDMENT_SHA256,
                    },
                    "implementation_amendment_01_lock": {
                        "path": (
                            "docs/E04_IMPLEMENTATION_AMENDMENT_01_LOCK_KO.md"
                        ),
                        "sha256": _PINNED_IMPLEMENTATION_AMENDMENT_LOCK_SHA256,
                    },
                },
                "evidence_scope": dict(config["evidence_scope"]),
                "limitations": [
                    "The prospectively fixed independent donor pairing was "
                    "not exactly norm-matchable inside the feasible gate box.",
                    "No intervention outcome was evaluated.",
                ],
            },
            main_eligible=False,
            full_eligible=False,
        )
        print(
            f"[{EXPERIMENT_ID}] PASS: {run_dir} "
            f"(H4={design_status}, reason={design_reason})"
        )
        return
    quartet_rows = _quartet_registry_rows(designs)
    design_rows = _design_lock_rows(designs)
    expected_quartet_rows = len(seeds) * count
    expected_design_rows = len(seeds) * count * len(_CONFIRMATORY_OPERATIONS)
    if len(quartet_rows) != expected_quartet_rows:
        raise AssertionError("E04 quartet registry shape differs.")
    if len(design_rows) != expected_design_rows:
        raise AssertionError("E04 intervention design lock shape differs.")
    maximum_norm_mismatch = max(
        float(
            cast(Mapping[str, Any], row["same_norm_match"])[
                "absolute_mismatch"
            ]
        )
        for row in design_rows
    )
    maximum_norm_mismatch = max(
        maximum_norm_mismatch,
        max(
            float(
                cast(Mapping[str, Any], row["cross_norm_match"])[
                    "absolute_mismatch"
                ]
            )
            for row in design_rows
        ),
    )
    if maximum_norm_mismatch > norm_tolerance:
        raise AssertionError("E04 donor norm match exceeds its frozen tolerance.")

    quartet_path = run_dir / "quartet_registry.jsonl"
    design_path = run_dir / "intervention_design_lock.jsonl"
    write_jsonl_strict(quartet_path, quartet_rows)
    write_jsonl_strict(design_path, design_rows)
    quartet_hash = sha256_file(quartet_path)
    design_hash = sha256_file(design_path)
    checkpoint_hashes = {
        checkpoint.path.name: checkpoint.file_sha256
        for checkpoint in source_e02.checkpoints.values()
    }
    prelock_path = run_dir / "preintervention_lock.json"
    write_json_strict(
        prelock_path,
        {
            "schema_version": 1,
            "status": "LOCKED_BEFORE_INTERVENTION_OUTCOME_EVALUATION",
            "run_id": context.run_id,
            "run_mode": context.run_mode,
            "config_sha256": _PINNED_CONFIG_CANONICAL_SHA256,
            "config_file_sha256": _PINNED_CONFIG_FILE_SHA256,
            "protocol_sha256": _PINNED_PROTOCOL_SHA256,
            "protocol_lock_sha256": _PINNED_PROTOCOL_LOCK_SHA256,
            "implementation_amendment_01_sha256": (
                _PINNED_IMPLEMENTATION_AMENDMENT_SHA256
            ),
            "implementation_amendment_01_lock_sha256": (
                _PINNED_IMPLEMENTATION_AMENDMENT_LOCK_SHA256
            ),
            "source_e02_manifest_sha256": source_e02.run.manifest_sha256,
            "source_e02_report_sha256": source_e02.run.report_sha256,
            "source_e02b_manifest_sha256": source_e02b.run.manifest_sha256,
            "source_e02b_report_sha256": source_e02b.run.report_sha256,
            "checkpoint_hashes": dict(sorted(checkpoint_hashes.items())),
            "quartet_registry": {
                "path": quartet_path.name,
                "rows": len(quartet_rows),
                "sha256": quartet_hash,
            },
            "intervention_design_lock": {
                "path": design_path.name,
                "rows": len(design_rows),
                "sha256": design_hash,
            },
            "maximum_norm_mismatch": maximum_norm_mismatch,
            "norm_match_tolerance": norm_tolerance,
            "donor_selection_uses_outcomes": False,
        },
    )
    prelock_hash = sha256_file(prelock_path)

    rows, effects, seed_rows, design_diagnostics = _evaluate_design(
        designs,
        doses=doses,
        minimum_headroom=minimum_headroom,
        monotonic_tolerance=monotonic_tolerance,
        self_tolerance=self_tolerance,
    )
    expected_rows = _expected_rows(len(seeds), count)
    _validate_row_identities(rows, require_full_schema=True)
    if len(rows) != expected_rows:
        raise AssertionError(
            f"E04 intervention row count {len(rows)} differs from {expected_rows}."
        )
    if sha256_file(quartet_path) != quartet_hash:
        raise ProvenanceValidationError("E04 quartet registry changed after lock.")
    if sha256_file(design_path) != design_hash:
        raise ProvenanceValidationError("E04 intervention design changed after lock.")
    if sha256_file(prelock_path) != prelock_hash:
        raise ProvenanceValidationError("E04 pre-intervention lock changed.")

    exact_main = bool(
        not args.dry_run
        and tuple(seeds) == _EXPECTED_SEEDS
        and count == _EXPECTED_MAIN_BASE_COUNT
        and len(rows) == 61_440
        and len(quartet_rows) == 1_024
        and len(design_rows) == 2_048
    )
    design_valid = bool(
        maximum_norm_mismatch <= norm_tolerance
        and design_diagnostics["self_restoration_passed"] is True
    )
    inference_eligible = bool(exact_main and design_valid)

    def positive(
        name: str,
        bootstrap_key: str | None = None,
    ) -> dict[str, object]:
        return _positive_gate(
            effects[name],
            threshold=positive_sesoi,
            alpha=alpha,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=int(bootstrap_seeds[bootstrap_key or name]),
            confidence=confidence,
            inference_eligible=inference_eligible,
        )

    def equivalent(
        name: str,
        bootstrap_key: str | None = None,
    ) -> dict[str, object]:
        return _equivalence_gate(
            effects[name],
            margin=equivalence_margin,
            alpha=alpha,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=int(bootstrap_seeds[bootstrap_key or name]),
            confidence=confidence,
            inference_eligible=inference_eligible,
        )

    def noninferior(
        name: str,
        bootstrap_key: str | None = None,
    ) -> dict[str, object]:
        return _noninferiority_gate(
            effects[name],
            margin=equivalence_margin,
            alpha=alpha,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=int(bootstrap_seeds[bootstrap_key or name]),
            confidence=confidence,
            inference_eligible=inference_eligible,
        )

    primary_gate = positive("primary_specificity")
    component_gates = {
        "add_write_damage": positive("add_write_damage"),
        "invalidate_erase_damage": positive("invalidate_erase_damage"),
        "add_erase_equivalence": equivalent(
            "add_erase_damage",
            "add_erase_equivalence",
        ),
        "invalidate_write_equivalence": equivalent(
            "invalidate_write_damage",
            "invalidate_write_equivalence",
        ),
        "supersede_erase_damage": positive("supersede_erase_damage"),
        "supersede_write_damage": positive("supersede_write_damage"),
        "preserve_erase_equivalence": equivalent(
            "preserve_erase_damage",
            "preserve_erase_equivalence",
        ),
        "preserve_write_equivalence": equivalent(
            "preserve_write_damage",
            "preserve_write_equivalence",
        ),
    }
    relevant_dose_labels = (
        "add/write",
        "invalidate/erase",
        "supersede/erase",
        "supersede/write",
    )
    monotonic_by_seed = {
        str(_integer(row["checkpoint_seed"], "seed-level checkpoint_seed")): {
            label: float(
                cast(Mapping[str, Any], row["dose_monotonicity_scores"])[label]
            )
            for label in relevant_dose_labels
        }
        for row in seed_rows
    }
    dose_monotonic_supported = bool(
        inference_eligible
        and all(
            score >= minimum_monotonic
            for per_seed in monotonic_by_seed.values()
            for score in per_seed.values()
        )
    )
    physical_dose_supported = bool(
        dose_monotonic_supported
        and all(gate["supported"] is True for gate in component_gates.values())
    )

    transplant_gates = {
        "cross_minus_same": positive("transplant_gap"),
        "same_minus_baseline_equivalence": equivalent(
            "transplant_baseline_difference",
            "transplant_baseline_equivalence",
        ),
    }
    rescue_gates = {
        "cross_minus_same": positive("rescue_gap"),
        "same_minus_baseline_equivalence": equivalent(
            "rescue_baseline_difference",
            "rescue_baseline_equivalence",
        ),
        "raw_same_donor_recovery": positive("rescue_raw_recovery"),
        "oracle_minus_baseline_equivalence": equivalent(
            "oracle_rescue_baseline_difference",
            "oracle_rescue_equivalence",
        ),
    }
    functional_conjunction = {
        "primary_specificity": primary_gate["supported"] is True,
        "physical_dose": physical_dose_supported,
        "transplant_specificity": all(
            gate["supported"] is True for gate in transplant_gates.values()
        ),
        "independent_donor_rescue": all(
            gate["supported"] is True for gate in rescue_gates.values()
        ),
    }
    functional_supported = bool(all(functional_conjunction.values()))

    total_effect_gate = positive("total_effect", "tied_dual_total_effect")
    mediated_effect_gate = positive(
        "mediated_effect",
        "scalarization_mediated_effect",
    )
    total_effect_seed_values = {
        checkpoint_seed: float(values.mean())
        for checkpoint_seed, values in sorted(effects["total_effect"].items())
    }
    scalarization_headroom_complete = bool(
        all(value > minimum_headroom for value in total_effect_seed_values.values())
    )
    ratio_interval: Interval | None = None
    ratio_seed_values: dict[str, float | None] = {}
    ratio_evaluable = False
    ratio_unidentifiable_reason: str | None = None
    if scalarization_headroom_complete:
        ratio_seed_values = {
            str(checkpoint_seed): float(
                effects["mediated_effect"][checkpoint_seed].mean()
            )
            / float(effects["total_effect"][checkpoint_seed].mean())
            for checkpoint_seed in sorted(effects["total_effect"])
        }
        try:
            ratio_interval = _bootstrap_ratio_interval(
                effects["mediated_effect"],
                effects["total_effect"],
                minimum_denominator=minimum_headroom,
                samples=bootstrap_samples,
                seed=int(bootstrap_seeds["scalarization_fraction"]),
                confidence=confidence,
            )
        except BootstrapRatioHeadroomError:
            ratio_p = None
            ratio_supported = False
            ratio_unidentifiable_reason = (
                "BOOTSTRAP_TOTAL_EFFECT_HEADROOM_INCOMPLETE"
            )
        else:
            ratio_evaluable = True
            ratio_p = exact_sign_flip_test(
                np.asarray(list(ratio_seed_values.values()), dtype=np.float64)
                - minimum_scalarization,
                "greater",
            )
            ratio_supported = bool(
                inference_eligible
                and ratio_interval.low >= minimum_scalarization
                and ratio_p <= alpha
            )
    else:
        ratio_seed_values = {
            str(checkpoint_seed): None
            for checkpoint_seed in sorted(effects["total_effect"])
        }
        ratio_p = None
        ratio_supported = False
        ratio_unidentifiable_reason = "SEED_TOTAL_EFFECT_HEADROOM_INCOMPLETE"
    ratio_inference_eligible = bool(inference_eligible and ratio_evaluable)
    scalarization_ratio_gate = {
        "estimate": None if ratio_interval is None else ratio_interval.estimate,
        "ci95": (
            None
            if ratio_interval is None
            else [ratio_interval.low, ratio_interval.high]
        ),
        "seed_values": ratio_seed_values,
        "minimum_fraction": minimum_scalarization,
        "all_seed_total_effect_headroom_complete": scalarization_headroom_complete,
        "bootstrap_ratio_evaluable": ratio_evaluable,
        "unidentifiable_reason": ratio_unidentifiable_reason,
        "seed_exact_sign_flip_p_above_fraction": ratio_p,
        "inference_eligible": ratio_inference_eligible,
        "supported": ratio_supported,
    }
    symmetric_scalarization_gate = equivalent(
        "symmetric_scalarization_difference",
        "symmetric_scalarization_equivalence",
    )
    transplant_retention_gate = noninferior(
        "transplant_retention_difference",
        "transplant_retention_noninferiority",
    )
    rescue_retention_gate = noninferior(
        "rescue_retention_difference",
        "rescue_retention_noninferiority",
    )
    architecture_conjunction = {
        "fresh_tied_dual_total_effect": total_effect_gate["supported"] is True,
        "posthoc_scalarization_mediated_effect": (
            mediated_effect_gate["supported"] is True
        ),
        "minimum_gap_recreation_fraction": scalarization_ratio_gate["supported"]
        is True,
        "symmetric_scalarization_equivalence": (
            symmetric_scalarization_gate["supported"] is True
        ),
        "transplant_retention_noninferiority": (
            transplant_retention_gate["supported"] is True
        ),
        "rescue_retention_noninferiority": (
            rescue_retention_gate["supported"] is True
        ),
    }
    architecture_supported = bool(all(architecture_conjunction.values()))
    full_supported = bool(functional_supported and architecture_supported)
    architecture_inference_eligible = bool(
        inference_eligible and ratio_inference_eligible
    )
    full_inference_eligible = bool(
        inference_eligible and architecture_inference_eligible
    )
    functional_status, functional_reason = _status(
        dry_run=args.dry_run,
        inference_eligible=inference_eligible,
        supported=functional_supported,
    )
    architecture_status, architecture_reason = _status(
        dry_run=args.dry_run,
        inference_eligible=architecture_inference_eligible,
        supported=architecture_supported,
    )
    full_status, full_reason = _status(
        dry_run=args.dry_run,
        inference_eligible=full_inference_eligible,
        supported=full_supported,
    )
    if (
        not args.dry_run
        and inference_eligible
        and not ratio_evaluable
        and ratio_unidentifiable_reason is not None
    ):
        architecture_reason = ratio_unidentifiable_reason
        full_reason = ratio_unidentifiable_reason

    metrics_path = run_dir / "intervention_metrics.jsonl"
    seed_path = run_dir / "seed_level_effects.jsonl"
    write_jsonl_strict(metrics_path, rows)
    write_jsonl_strict(seed_path, seed_rows)
    metrics_hash = sha256_file(metrics_path)
    seed_hash = sha256_file(seed_path)
    if _count_jsonl_rows(metrics_path) != len(rows):
        raise AssertionError("E04 metric artifact row count differs after write.")
    if _count_jsonl_rows(seed_path) != len(seed_rows):
        raise AssertionError("E04 seed-effect artifact row count differs after write.")

    _validate_protocol_config(config, config_path)
    rechecked_e00 = validate_legacy_e00(
        args.artifact_root,
        require_full=not args.dry_run,
    )
    if rechecked_e00 != e00:
        raise ProvenanceValidationError("E04 E00 dependency changed before finalize.")
    rechecked_e02 = validate_pinned_e02_source(args.artifact_root, config)
    rechecked_e02b = _validate_e02b_repair(
        args.artifact_root,
        config,
        rechecked_e02,
    )
    for checkpoint in rechecked_e02.checkpoints.values():
        if sha256_file(checkpoint.path) != checkpoint.file_sha256:
            raise ProvenanceValidationError("E04 source checkpoint changed before finalize.")
    if (
        rechecked_e02.run.manifest_sha256 != source_e02.run.manifest_sha256
        or rechecked_e02.run.report_sha256 != source_e02.run.report_sha256
        or rechecked_e02b.run.manifest_sha256 != source_e02b.run.manifest_sha256
        or rechecked_e02b.run.report_sha256 != source_e02b.run.report_sha256
        or sha256_file(quartet_path) != quartet_hash
        or sha256_file(design_path) != design_hash
        or sha256_file(prelock_path) != prelock_hash
        or sha256_file(metrics_path) != metrics_hash
        or sha256_file(seed_path) != seed_hash
    ):
        raise ProvenanceValidationError("E04 dependency or output changed before finalize.")

    report: dict[str, Any] = {
        "status": "PASS",
        "execution": {
            "dry_run": args.dry_run,
            "checkpoint_only": True,
            "training_executed": False,
            "optimizer_created": False,
            "configured_seeds": list(_EXPECTED_SEEDS),
            "executed_seeds": seeds,
            "base_count_per_seed": count,
            "counterfactual_quartet_count": len(quartet_rows),
            "base_operation_episode_count": len(quartet_rows) * len(Operation),
            "intervention_metric_rows": len(rows),
            "expected_intervention_metric_rows": expected_rows,
            "design_lock_rows": len(design_rows),
            "pair_blocks_per_seed": count // 2,
            "cpu_threads": cpu_threads,
            "exact_main_execution": exact_main,
            "protocol_execution_complete": True,
        },
        "dependency_lineage": dependencies,
        "original_e02_disposition": {
            "run_id": source_e02.run.run_id,
            "execution_status": "PASS",
            "confirmatory_status": "INCONCLUSIVE",
            "reason": "PREREGISTERED_SYMMETRIC_RELATIVE_GATE_UNIDENTIFIABLE",
            "original_h2_claim_open": False,
            "status": "immutable_not_relabelled",
        },
        "e02b_repair_disposition": {
            "run_id": source_e02b.run.run_id,
            "repair_status": "SUPPORTED",
            "prospective_repair_claim_open": True,
            "checkpoint_source": source_e02.run.run_id,
        },
        "counterfactual_design": {
            "same_base_fields_verified": [
                "state",
                "address_and_keys",
                "values",
                "old_value",
                "new_value",
                "erase_candidate",
                "write_candidate",
            ],
            "only_operation_demand_changes": True,
            "main_relative_seed_range": [
                _EXPECTED_MAIN_OFFSET,
                _EXPECTED_MAIN_OFFSET + _EXPECTED_MAIN_BASE_COUNT - 1,
            ],
            "dry_relative_seed_range": [
                _EXPECTED_DRY_OFFSET,
                _EXPECTED_DRY_OFFSET + _EXPECTED_DRY_BASE_COUNT - 1,
            ],
            "active_relative_seed_range": [
                seed_offset,
                seed_offset + count - 1,
            ],
            "candidate_mode": CandidateMode.ORACLE.value,
            "geometry": {
                "old_scale": float(data["old_scale"]),
                "new_scale": float(data["new_scale"]),
                "old_new_cosine": float(data["old_new_cosine"]),
                "key_correlation": float(data["key_correlation"]),
            },
        },
        "preintervention_lock": {
            "status": "LOCKED_BEFORE_INTERVENTION_OUTCOME_EVALUATION",
            "quartet_registry_sha256": quartet_hash,
            "intervention_design_lock_sha256": design_hash,
            "metadata_sha256": prelock_hash,
            "donor_pairing": "adjacent_two_cycle",
            "same_and_cross_share_independent_donor_base": True,
            "donor_selection_uses_outcomes": False,
            "norm_matching_uses_clipping": False,
            "maximum_norm_mismatch": maximum_norm_mismatch,
            "norm_match_tolerance": norm_tolerance,
        },
        "inference_contract": {
            "checkpoint_seed_is_inference_unit": True,
            "checkpoint_seed_count": len(seeds),
            "episode_rows_are_not_independent_replicates": True,
            "bootstrap_unit": "adjacent_donor_pair_block_within_fixed_checkpoint_seed",
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_confidence": confidence,
            "seed_test": "eight_seed_exact_sign_flip",
            "alpha": alpha,
            "positive_effect_sesoi": positive_sesoi,
            "equivalence_and_noninferiority_margin": equivalence_margin,
            "intersection_union_global_claim": True,
            "standalone_component_claims_require_holm": True,
        },
        "functional_specificity": {
            "primary_operation_by_relevance_interaction": primary_gate,
            "physical_component_gates": component_gates,
            "dose_monotonicity": {
                "doses": list(doses),
                "relevant_cells": list(relevant_dose_labels),
                "seed_cell_scores": monotonic_by_seed,
                "minimum_fraction": minimum_monotonic,
                "all_relevant_seed_cells_passed": dose_monotonic_supported,
                "endpoint_gates_prevent_flat_curve_pass": True,
                "supported": physical_dose_supported,
            },
            "counterfactual_transplant": {
                **transplant_gates,
                "operations": [item.value for item in _CONFIRMATORY_OPERATIONS],
                "same_and_cross_share_donor_base": True,
                "supported": all(
                    gate["supported"] is True for gate in transplant_gates.values()
                ),
            },
            "nontrivial_independent_donor_rescue": {
                **rescue_gates,
                "episodewise_same_or_oracle_max_used": False,
                "oracle_is_positive_control_only": True,
                "supported": all(
                    gate["supported"] is True for gate in rescue_gates.values()
                ),
            },
            "self_restoration_sanity": design_diagnostics,
            "conjunction": functional_conjunction,
            "status": functional_status,
            "reason": functional_reason,
            "supported": functional_supported,
        },
        "architecture_gap_mediation": {
            "fresh_tied_dual_total_effect": total_effect_gate,
            "posthoc_scalarization_mediated_effect": mediated_effect_gate,
            "gap_recreation_fraction": scalarization_ratio_gate,
            "symmetric_scalarization_equivalence": symmetric_scalarization_gate,
            "transplant_retention_noninferiority": transplant_retention_gate,
            "rescue_retention_noninferiority": rescue_retention_gate,
            "conjunction": architecture_conjunction,
            "status": architecture_status,
            "reason": architecture_reason,
            "supported": architecture_supported,
        },
        "secondary_diagnostics": {
            "seed_level_rows": seed_rows,
            "supersede_joint_dose_is_confirmatory": False,
            "same_donor_recovery_fraction_is_ratio_of_seed_means": True,
        },
        "claim_gate": {
            "evaluated": full_inference_eligible,
            "inference_eligible": full_inference_eligible,
            "functional_specificity_status": functional_status,
            "architecture_gap_mediation_status": architecture_status,
            "status": full_status,
            "reason": full_reason,
            "supported": full_supported,
            "full_h4_claim_open": full_supported,
            "allowed_claim": (
                config["claim"]["allowed_if_supported"] if full_supported else None
            ),
            "forbidden_claims": config["claim"]["forbidden"],
            "original_e02_remains_inconclusive": True,
            "e02b_repair_remains_supported": True,
        },
        "protocol_lineage": {
            "frozen_protocol": {
                "path": "docs/E04_PROTOCOL_PREREGISTRATION_FROZEN_KO.md",
                "sha256": _PINNED_PROTOCOL_SHA256,
            },
            "frozen_protocol_lock": {
                "path": "docs/E04_PROTOCOL_PREREGISTRATION_LOCK_KO.md",
                "sha256": _PINNED_PROTOCOL_LOCK_SHA256,
            },
            "implementation_amendment_01": {
                "path": "docs/E04_IMPLEMENTATION_AMENDMENT_01_FROZEN_KO.md",
                "sha256": _PINNED_IMPLEMENTATION_AMENDMENT_SHA256,
            },
            "implementation_amendment_01_lock": {
                "path": "docs/E04_IMPLEMENTATION_AMENDMENT_01_LOCK_KO.md",
                "sha256": _PINNED_IMPLEMENTATION_AMENDMENT_LOCK_SHA256,
            },
            "config_canonical_sha256": _PINNED_CONFIG_CANONICAL_SHA256,
            "config_file_sha256": _PINNED_CONFIG_FILE_SHA256,
        },
        "artifacts": {
            "quartet_registry": {
                "path": quartet_path.name,
                "rows": len(quartet_rows),
                "sha256": quartet_hash,
            },
            "intervention_design_lock": {
                "path": design_path.name,
                "rows": len(design_rows),
                "sha256": design_hash,
            },
            "preintervention_lock": {
                "path": prelock_path.name,
                "sha256": prelock_hash,
            },
            "intervention_metrics": {
                "path": metrics_path.name,
                "rows": len(rows),
                "sha256": metrics_hash,
            },
            "seed_level_effects": {
                "path": seed_path.name,
                "rows": len(seed_rows),
                "sha256": seed_hash,
            },
        },
        "evidence_scope": dict(config["evidence_scope"]),
        "limitations": [
            "Oracle candidate and oracle address are supplied.",
            "Operation identity is supplied to the frozen controller.",
            "The donor intervention is a controlled synthetic mediation assay.",
            "No official backend, pretrained language model, or semantic input is evaluated.",
        ],
    }
    finalize_v61_run(
        context=context,
        report=report,
        main_eligible=full_inference_eligible,
        full_eligible=full_inference_eligible,
    )
    print(
        f"[{EXPERIMENT_ID}] PASS: {run_dir} "
        f"(H4={full_status}, functional={functional_status}, "
        f"architecture={architecture_status})"
    )


if __name__ == "__main__":
    main()
