from __future__ import annotations

import copy
import random
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Any

import torch

from catena.core.provenance_v61 import SHA256_PATTERN, sha256_canonical_json

from .audit_contract import validate_e26_audit_locked_hashes
from .config import ModelConfig
from .hashing import state_dict_digest, tensor_tree_digest
from .model import CatenaLM, RuntimeState, cross_entropy_loss
from .trainer import make_optimizer, optimizer_step_microbatches


@dataclass(frozen=True, slots=True)
class NumericalTolerances:
    relative_l2_max: float
    max_abs_max: float | None

    def __post_init__(self) -> None:
        if self.relative_l2_max <= 0:
            raise ValueError("relative_l2_max must be positive")
        if self.max_abs_max is not None and self.max_abs_max <= 0:
            raise ValueError("max_abs_max must be positive when supplied")


@dataclass(frozen=True, slots=True)
class TensorError:
    relative_l2: float
    max_abs: float

    def passes(self, tolerances: NumericalTolerances) -> bool:
        return self.relative_l2 <= tolerances.relative_l2_max and (
            tolerances.max_abs_max is None or self.max_abs <= tolerances.max_abs_max
        )


@dataclass(frozen=True, slots=True)
class RuntimeStateError:
    recurrent: TensorError
    attention_key: TensorError
    attention_value: TensorError
    positions_equal: bool
    lengths_equal: bool
    write_indices_equal: bool
    position_equal: bool

    def passes(self, tolerances: NumericalTolerances) -> bool:
        return (
            self.recurrent.passes(tolerances)
            and self.attention_key.passes(tolerances)
            and self.attention_value.passes(tolerances)
            and self.positions_equal
            and self.lengths_equal
            and self.write_indices_equal
            and self.position_equal
        )


@dataclass(frozen=True, slots=True)
class PartitionAuditRow:
    partition: tuple[int, ...]
    logits: TensorError
    runtime_state: RuntimeStateError
    gradients: TensorError
    gradients_worst_leaf: TensorError
    gradients_finite: bool
    passed: bool

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["partition"] = list(self.partition)
        return payload


@dataclass(frozen=True, slots=True)
class PartitionAuditReport:
    precision: str
    partitions: tuple[tuple[int, ...], ...]
    reference_logits: TensorError
    reference_runtime_state: RuntimeStateError
    reference_gradients: TensorError
    reference_gradients_worst_leaf: TensorError
    reference_gradients_finite: bool
    rope_offset_contract: str
    rows: tuple[PartitionAuditRow, ...]
    passed: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "precision": self.precision,
            "partitions": [list(partition) for partition in self.partitions],
            "reference_logits": asdict(self.reference_logits),
            "reference_runtime_state": asdict(self.reference_runtime_state),
            "reference_gradients": asdict(self.reference_gradients),
            "reference_gradients_worst_leaf": asdict(self.reference_gradients_worst_leaf),
            "reference_gradients_finite": self.reference_gradients_finite,
            "rope_offset_contract": self.rope_offset_contract,
            "rows": [row.as_dict() for row in self.rows],
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class PartitionedOutput:
    logits: torch.Tensor
    runtime_state: RuntimeState


@dataclass(frozen=True, slots=True)
class GradAccumulationAuditRow:
    microbatch_sizes: tuple[int, ...]
    loss_error: float
    gradient_error: TensorError
    gradient_worst_leaf_error: TensorError
    parameter_error: TensorError
    optimizer_error: TensorError
    optimizer_worst_leaf_error: TensorError
    optimizer_structure_equal: bool
    gradient_norm_relative_error: float
    clip_coefficient_absolute_error: float
    optimizer_digest_equal: bool
    scheduler_digest_equal: bool
    token_exposure_equal: bool
    passed: bool

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["microbatch_sizes"] = list(self.microbatch_sizes)
        return payload


def fixed_partition_suite(
    total_length: int,
    *,
    random_seeds: Sequence[int],
) -> tuple[tuple[int, ...], ...]:
    """Build the preregisterable mandatory plus fixed-random partition suite."""

    if total_length <= 415:
        raise ValueError("total_length must exceed 415 for the mandatory partition suite")
    if len(random_seeds) < 8:
        raise ValueError("At least eight fixed random partition seeds are required")
    partitions: list[tuple[int, ...]] = [
        (total_length,),
        (1, total_length - 1),
        (3, 5, 7, total_length - 15),
        (31, 127, 257, total_length - 415),
    ]
    for seed in random_seeds:
        rng = random.Random(int(seed))
        pieces = 2 + rng.randrange(min(15, total_length - 1))
        cuts = sorted(rng.sample(range(1, total_length), pieces - 1))
        boundaries = [0, *cuts, total_length]
        partition = tuple(
            boundaries[index + 1] - boundaries[index] for index in range(len(boundaries) - 1)
        )
        partitions.append(partition)
    if any(not partition or any(piece <= 0 for piece in partition) for partition in partitions):
        raise AssertionError("Partition suite contains a non-positive piece")
    if any(sum(partition) != total_length for partition in partitions):
        raise AssertionError("Partition suite does not cover the full sequence")
    if len(set(partitions)) != len(partitions):
        raise ValueError("Fixed random partition seeds produced duplicate partitions")
    return tuple(partitions)


def forward_partitioned(
    model: CatenaLM,
    input_ids: torch.Tensor,
    partition: Sequence[int],
    *,
    initial_state: RuntimeState | None = None,
    autocast_dtype: torch.dtype | None = None,
) -> PartitionedOutput:
    if not partition or any(piece <= 0 for piece in partition):
        raise ValueError("partition must contain positive lengths")
    if sum(partition) != input_ids.shape[1]:
        raise ValueError("partition does not cover input_ids")
    state = None if initial_state is None else initial_state.clone(detach=True)
    logits: list[torch.Tensor] = []
    offset = 0
    for piece in partition:
        context = (
            torch.autocast(device_type=input_ids.device.type, dtype=autocast_dtype)
            if autocast_dtype is not None
            else nullcontext()
        )
        with context:
            output = model(input_ids[:, offset : offset + piece].contiguous(), state)
        logits.append(output.logits)
        state = output.runtime_state
        offset += piece
    assert state is not None
    return PartitionedOutput(logits=torch.cat(logits, dim=1), runtime_state=state)


def _tensor_error(left: torch.Tensor, right: torch.Tensor) -> TensorError:
    if left.shape != right.shape:
        return TensorError(relative_l2=float("inf"), max_abs=float("inf"))
    difference = left.detach().float() - right.detach().float()
    denominator = right.detach().float().norm()
    if float(denominator.item()) == 0.0:
        relative = 0.0 if float(difference.norm().item()) == 0.0 else float("inf")
    else:
        relative = float((difference.norm() / denominator).item())
    maximum = float(difference.abs().max().item()) if difference.numel() else 0.0
    return TensorError(relative_l2=relative, max_abs=maximum)


def _max_tensor_error(errors: Sequence[TensorError]) -> TensorError:
    if not errors:
        return TensorError(relative_l2=0.0, max_abs=0.0)
    return TensorError(
        relative_l2=max(error.relative_l2 for error in errors),
        max_abs=max(error.max_abs for error in errors),
    )


def runtime_state_error(observed: RuntimeState, expected: RuntimeState) -> RuntimeStateError:
    if len(observed.recurrent) != len(expected.recurrent) or len(observed.attention) != len(
        expected.attention
    ):
        infinite = TensorError(float("inf"), float("inf"))
        return RuntimeStateError(
            recurrent=infinite,
            attention_key=infinite,
            attention_value=infinite,
            positions_equal=False,
            lengths_equal=False,
            write_indices_equal=False,
            position_equal=False,
        )
    return RuntimeStateError(
        recurrent=_max_tensor_error(
            [
                _tensor_error(left.matrix, right.matrix)
                for left, right in zip(observed.recurrent, expected.recurrent, strict=True)
            ]
        ),
        attention_key=_max_tensor_error(
            [
                _tensor_error(left.key, right.key)
                for left, right in zip(observed.attention, expected.attention, strict=True)
            ]
        ),
        attention_value=_max_tensor_error(
            [
                _tensor_error(left.value, right.value)
                for left, right in zip(observed.attention, expected.attention, strict=True)
            ]
        ),
        positions_equal=all(
            torch.equal(left.positions, right.positions)
            for left, right in zip(observed.attention, expected.attention, strict=True)
        ),
        lengths_equal=all(
            left.length == right.length
            for left, right in zip(observed.attention, expected.attention, strict=True)
        ),
        write_indices_equal=all(
            left.write_index == right.write_index
            for left, right in zip(observed.attention, expected.attention, strict=True)
        ),
        position_equal=observed.position == expected.position,
    )


def _gradient_snapshot(model: CatenaLM) -> tuple[dict[str, torch.Tensor], bool]:
    gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    complete = len(gradients) == sum(1 for _ in model.parameters())
    finite = complete and all(
        bool(torch.isfinite(value).all().item()) for value in gradients.values()
    )
    return gradients, finite


def _gradient_errors(
    observed: Mapping[str, torch.Tensor],
    expected: Mapping[str, torch.Tensor],
) -> tuple[TensorError, TensorError]:
    """Return the registered global-tree error and worst-leaf diagnostic."""

    aggregate, worst_leaf, structure_equal = _state_tree_tensor_errors(
        observed,
        expected,
    )
    if not structure_equal:
        infinite = TensorError(float("inf"), float("inf"))
        return infinite, infinite
    return aggregate, worst_leaf


def _state_tree_tensor_errors(
    observed: Any,
    expected: Any,
) -> tuple[TensorError, TensorError, bool]:
    """Return aggregate and worst-leaf errors for a numerical state tree.

    The registered tolerance is a relative-L2/max-absolute contract, so the
    primary value is computed over the complete floating tensor tree. A
    leafwise relative maximum is retained as a diagnostic: it is not a stable
    primary gate when a second-moment leaf has a norm close to zero. Scalar
    bookkeeping tensors (for example Adam's step counter) must match exactly
    and are excluded from the aggregate denominator.
    """

    tensor_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []

    def collect(left: Any, right: Any) -> bool:
        if torch.is_tensor(left) and torch.is_tensor(right):
            if left.shape != right.shape or left.dtype != right.dtype:
                return False
            if left.ndim == 0 or not (left.is_floating_point() or left.is_complex()):
                return bool(torch.equal(left, right))
            tensor_pairs.append((left, right))
            return True
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            if set(left) != set(right):
                return False
            return all(
                collect(left[key], right[key])
                for key in sorted(
                    right,
                    key=lambda value: (type(value).__qualname__, repr(value)),
                )
            )
        if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
            return len(left) == len(right) and all(
                collect(left_child, right_child)
                for left_child, right_child in zip(left, right, strict=True)
            )
        return bool(left == right)

    structure_equal = collect(observed, expected)
    if not structure_equal:
        infinite = TensorError(float("inf"), float("inf"))
        return infinite, infinite, False
    if not tensor_pairs:
        zero = TensorError(0.0, 0.0)
        return zero, zero, True

    difference_squared = 0.0
    expected_squared = 0.0
    maximum_absolute = 0.0
    leaf_errors: list[TensorError] = []
    for left, right in tensor_pairs:
        left_float = torch.view_as_real(left).float() if left.is_complex() else left.float()
        right_float = torch.view_as_real(right).float() if right.is_complex() else right.float()
        difference = left_float - right_float
        difference_squared += float(difference.pow(2).sum().item())
        expected_squared += float(right_float.pow(2).sum().item())
        maximum_absolute = max(
            maximum_absolute,
            float(difference.abs().max().item()) if difference.numel() else 0.0,
        )
        leaf_errors.append(_tensor_error(left, right))
    if expected_squared == 0.0:
        relative_l2 = 0.0 if difference_squared == 0.0 else float("inf")
    else:
        relative_l2 = (difference_squared / expected_squared) ** 0.5
    return (
        TensorError(relative_l2=relative_l2, max_abs=maximum_absolute),
        _max_tensor_error(leaf_errors),
        True,
    )


def _run_gradient(
    model: CatenaLM,
    input_ids: torch.Tensor,
    partition: Sequence[int],
    *,
    initial_state: RuntimeState | None,
    autocast_dtype: torch.dtype | None,
) -> tuple[dict[str, torch.Tensor], bool]:
    model.zero_grad(set_to_none=True)
    result = forward_partitioned(
        model,
        input_ids,
        partition,
        initial_state=initial_state,
        autocast_dtype=autocast_dtype,
    )
    loss = cross_entropy_loss(result.logits, input_ids)
    loss.backward()  # type: ignore[no-untyped-call]
    return _gradient_snapshot(model)


def monolithic_reference_model(model: CatenaLM) -> CatenaLM:
    mapping = model.config.to_dict()
    mapping["backend_id"] = "reference_python"
    mapping["backend_scientific_main_capable"] = False
    reference = CatenaLM(ModelConfig.from_mapping(mapping))
    reference.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
    return reference.to(next(model.parameters()).device)


def audit_arbitrary_partitions(
    model: CatenaLM,
    input_ids: torch.Tensor,
    *,
    partitions: Sequence[Sequence[int]],
    tolerances: NumericalTolerances,
    autocast_dtype: torch.dtype | None,
    initial_state: RuntimeState | None = None,
) -> PartitionAuditReport:
    """Audit external partitions against optimized full and monolithic reference."""

    if not partitions:
        raise ValueError("At least one partition is required")
    normalized = tuple(tuple(int(piece) for piece in partition) for partition in partitions)
    if normalized[0] != (input_ids.shape[1],):
        raise ValueError("The first partition must be the monolithic full sequence")
    model.eval()
    reference = monolithic_reference_model(model).eval()
    full = forward_partitioned(
        model,
        input_ids,
        normalized[0],
        initial_state=initial_state,
        autocast_dtype=autocast_dtype,
    )
    reference_full = forward_partitioned(
        reference,
        input_ids,
        normalized[0],
        initial_state=initial_state,
        autocast_dtype=autocast_dtype,
    )
    reference_logits = _tensor_error(full.logits, reference_full.logits)
    reference_state = runtime_state_error(full.runtime_state, reference_full.runtime_state)
    baseline_gradients, baseline_finite = _run_gradient(
        model,
        input_ids,
        normalized[0],
        initial_state=initial_state,
        autocast_dtype=autocast_dtype,
    )
    reference_gradients, reference_finite = _run_gradient(
        reference,
        input_ids,
        normalized[0],
        initial_state=initial_state,
        autocast_dtype=autocast_dtype,
    )
    (
        reference_gradient_error,
        reference_gradient_worst_leaf,
    ) = _gradient_errors(baseline_gradients, reference_gradients)
    rows: list[PartitionAuditRow] = []
    for partition in normalized:
        observed = forward_partitioned(
            model,
            input_ids,
            partition,
            initial_state=initial_state,
            autocast_dtype=autocast_dtype,
        )
        logits_error = _tensor_error(observed.logits, full.logits)
        state_error = runtime_state_error(observed.runtime_state, full.runtime_state)
        gradients, finite = _run_gradient(
            model,
            input_ids,
            partition,
            initial_state=initial_state,
            autocast_dtype=autocast_dtype,
        )
        gradients_error, gradients_worst_leaf = _gradient_errors(
            gradients,
            baseline_gradients,
        )
        passed = (
            logits_error.passes(tolerances)
            and state_error.passes(tolerances)
            and gradients_error.passes(tolerances)
            and baseline_finite
            and finite
        )
        rows.append(
            PartitionAuditRow(
                partition=partition,
                logits=logits_error,
                runtime_state=state_error,
                gradients=gradients_error,
                gradients_worst_leaf=gradients_worst_leaf,
                gradients_finite=baseline_finite and finite,
                passed=passed,
            )
        )
    reference_pass = (
        reference_logits.passes(tolerances)
        and reference_state.passes(tolerances)
        and reference_gradient_error.passes(tolerances)
        and baseline_finite
        and reference_finite
    )
    precision = "fp32" if autocast_dtype is None else str(autocast_dtype).removeprefix("torch.")
    return PartitionAuditReport(
        precision=precision,
        partitions=normalized,
        reference_logits=reference_logits,
        reference_runtime_state=reference_state,
        reference_gradients=reference_gradient_error,
        reference_gradients_worst_leaf=reference_gradient_worst_leaf,
        reference_gradients_finite=baseline_finite and reference_finite,
        rope_offset_contract=(
            "N/A_NO_ROPE_IN_MODEL; RuntimeState.position and local-attention absolute "
            "positions audited exactly"
        ),
        rows=tuple(rows),
        passed=reference_pass and all(row.passed for row in rows),
    )


def audit_gradient_accumulation(
    model: CatenaLM,
    global_batch: torch.Tensor,
    *,
    accumulation_layouts: Sequence[Sequence[int]],
    tolerances: NumericalTolerances,
    autocast_dtype: torch.dtype | None,
    optimizer_factory: Callable[[CatenaLM], torch.optim.Optimizer] = make_optimizer,
    scheduler_factory: Callable[[torch.optim.Optimizer], Any] | None = None,
    grad_clip_norm: float = 1.0,
) -> tuple[GradAccumulationAuditRow, ...]:
    if not accumulation_layouts:
        raise ValueError("At least one accumulation layout is required")
    batch_size = global_batch.shape[0]
    normalized = tuple(tuple(int(value) for value in layout) for layout in accumulation_layouts)
    if any(
        any(value <= 0 for value in layout) or sum(layout) != batch_size for layout in normalized
    ):
        raise ValueError("Every accumulation layout must partition the global batch")

    baseline_model = copy.deepcopy(model)
    baseline_optimizer = optimizer_factory(baseline_model)
    baseline_scheduler = (
        None if scheduler_factory is None else scheduler_factory(baseline_optimizer)
    )
    baseline_batches: list[torch.Tensor] = []
    baseline_start = 0
    for size in normalized[0]:
        baseline_batches.append(global_batch[baseline_start : baseline_start + size])
        baseline_start += size
    baseline_step = optimizer_step_microbatches(
        baseline_model,
        baseline_batches,
        optimizer=baseline_optimizer,
        scheduler=baseline_scheduler,
        grad_clip_norm=grad_clip_norm,
        autocast_dtype=autocast_dtype,
        capture_gradients=True,
    )
    assert baseline_step.gradients_before_clip is not None
    rows: list[GradAccumulationAuditRow] = []
    for layout in normalized:
        candidate = copy.deepcopy(model)
        optimizer = optimizer_factory(candidate)
        scheduler = None if scheduler_factory is None else scheduler_factory(optimizer)
        batches: list[torch.Tensor] = []
        start = 0
        for size in layout:
            batches.append(global_batch[start : start + size])
            start += size
        step = optimizer_step_microbatches(
            candidate,
            batches,
            optimizer=optimizer,
            scheduler=scheduler,
            grad_clip_norm=grad_clip_norm,
            autocast_dtype=autocast_dtype,
            capture_gradients=True,
        )
        assert step.gradients_before_clip is not None
        gradient_error, gradient_worst_leaf_error = _gradient_errors(
            step.gradients_before_clip,
            baseline_step.gradients_before_clip,
        )
        parameter_errors = [
            _tensor_error(observed, expected)
            for (_, observed), (_, expected) in zip(
                candidate.named_parameters(),
                baseline_model.named_parameters(),
                strict=True,
            )
        ]
        parameter_error = _max_tensor_error(parameter_errors)
        loss_error = abs(step.loss - baseline_step.loss) / max(
            abs(baseline_step.loss),
            torch.finfo(torch.float32).eps,
        )
        optimizer_equal = tensor_tree_digest(optimizer.state_dict()) == tensor_tree_digest(
            baseline_optimizer.state_dict()
        )
        (
            optimizer_error,
            optimizer_worst_leaf_error,
            optimizer_structure_equal,
        ) = _state_tree_tensor_errors(
            optimizer.state_dict(),
            baseline_optimizer.state_dict(),
        )
        scheduler_equal = (
            scheduler is None
            and baseline_scheduler is None
            or (
                scheduler is not None
                and baseline_scheduler is not None
                and tensor_tree_digest(scheduler.state_dict())
                == tensor_tree_digest(baseline_scheduler.state_dict())
            )
        )
        token_equal = (
            step.valid_prediction_tokens == baseline_step.valid_prediction_tokens
            and step.exposed_input_tokens == baseline_step.exposed_input_tokens
        )
        gradient_norm_error = abs(
            step.gradient_norm_before_clip - baseline_step.gradient_norm_before_clip
        ) / max(
            abs(baseline_step.gradient_norm_before_clip),
            torch.finfo(torch.float32).eps,
        )
        clip_coefficient_error = abs(step.clip_coefficient - baseline_step.clip_coefficient)
        passed = (
            gradient_error.passes(tolerances)
            and parameter_error.passes(tolerances)
            and loss_error <= tolerances.relative_l2_max
            and gradient_norm_error <= tolerances.relative_l2_max
            and clip_coefficient_error <= tolerances.relative_l2_max
            and optimizer_error.passes(tolerances)
            and optimizer_structure_equal
            and scheduler_equal
            and token_equal
        )
        rows.append(
            GradAccumulationAuditRow(
                microbatch_sizes=layout,
                loss_error=loss_error,
                gradient_error=gradient_error,
                gradient_worst_leaf_error=gradient_worst_leaf_error,
                parameter_error=parameter_error,
                optimizer_error=optimizer_error,
                optimizer_worst_leaf_error=optimizer_worst_leaf_error,
                optimizer_structure_equal=optimizer_structure_equal,
                gradient_norm_relative_error=gradient_norm_error,
                clip_coefficient_absolute_error=clip_coefficient_error,
                optimizer_digest_equal=optimizer_equal,
                scheduler_digest_equal=scheduler_equal,
                token_exposure_equal=token_equal,
                passed=passed,
            )
        )
    return tuple(rows)


def model_and_optimizer_digests(
    model: CatenaLM,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
) -> dict[str, str | None]:
    return {
        "model": state_dict_digest(model),
        "optimizer": tensor_tree_digest(optimizer.state_dict()),
        "scheduler": None if scheduler is None else tensor_tree_digest(scheduler.state_dict()),
    }


def numerical_audit_receipt(
    *,
    fp32_partition: PartitionAuditReport,
    bf16_partition: PartitionAuditReport,
    fp32_grad_accumulation: Sequence[GradAccumulationAuditRow],
    bf16_grad_accumulation: Sequence[GradAccumulationAuditRow],
    locked_hashes: Mapping[str, str],
    source_inventory: Mapping[str, Any],
) -> dict[str, Any]:
    """Create the non-evidence aggregate consumed by the fail-closed E26a runner."""

    normalized_hashes = validate_e26_audit_locked_hashes(locked_hashes)
    if source_inventory.get("source_tree_sha256") != normalized_hashes["source_tree_sha256"]:
        raise ValueError("Numerical receipt source inventory differs from locked source hash")
    all_passed = (
        fp32_partition.passed
        and bf16_partition.passed
        and bool(fp32_grad_accumulation)
        and bool(bf16_grad_accumulation)
        and all(row.passed for row in fp32_grad_accumulation)
        and all(row.passed for row in bf16_grad_accumulation)
    )
    payload: dict[str, Any] = {
        "schema_version": "catena-v8.1",
        "manifest_type": "E26_NUMERICAL_AUDIT",
        "scientific_evidence": False,
        "evidence_tier": "NON_EVIDENCE_VALIDATION",
        "all_passed": all_passed,
        "passed": all_passed,
        "main_test_opened": False,
        "locked_hashes": normalized_hashes,
        "source_inventory": dict(source_inventory),
        "arbitrary_partitions": {
            "fp32": fp32_partition.as_dict(),
            "bf16": bf16_partition.as_dict(),
        },
        "gradient_accumulation": {
            "fp32": [row.as_dict() for row in fp32_grad_accumulation],
            "bf16": [row.as_dict() for row in bf16_grad_accumulation],
        },
        "rope_offset_contract": fp32_partition.rope_offset_contract,
    }
    payload["receipt_sha256"] = sha256_canonical_json(payload)
    return payload


def candidate_matrix_numerical_audit_receipt(
    *,
    candidate_audits: Mapping[str, Mapping[str, Any]],
    expected_candidate_ids: Sequence[str],
    locked_hashes: Mapping[str, str],
    source_inventory: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind numerical coverage to every locked candidate and both variants."""

    normalized_ids = tuple(str(value) for value in expected_candidate_ids)
    if len(normalized_ids) != len(set(normalized_ids)) or not normalized_ids:
        raise ValueError("expected_candidate_ids must be non-empty and unique")
    if set(candidate_audits) != set(normalized_ids):
        raise ValueError("Numerical audit does not cover the exact locked candidate grid")
    normalized_hashes = validate_e26_audit_locked_hashes(locked_hashes)
    if source_inventory.get("source_tree_sha256") != normalized_hashes["source_tree_sha256"]:
        raise ValueError("Numerical receipt source inventory differs from locked source hash")

    variants = {"dual_delta_lm", "projected_tied_delta_lm"}
    candidate_passes: list[bool] = []
    normalized_candidates: dict[str, Any] = {}
    for candidate_id in normalized_ids:
        row = dict(candidate_audits[candidate_id])
        config_hash = row.get("model_config_sha256")
        variant_rows = row.get("variants")
        if (
            not isinstance(config_hash, str)
            or not SHA256_PATTERN.fullmatch(config_hash)
            or not isinstance(variant_rows, Mapping)
            or set(variant_rows) != variants
        ):
            raise ValueError(f"Malformed candidate numerical audit: {candidate_id}")
        variant_passes: list[bool] = []
        for variant in sorted(variants):
            variant_row = variant_rows[variant]
            if not isinstance(variant_row, Mapping):
                raise ValueError(f"Malformed variant numerical audit: {candidate_id}/{variant}")
            partitions = variant_row.get("arbitrary_partitions")
            accumulation = variant_row.get("gradient_accumulation")
            if (
                not isinstance(partitions, Mapping)
                or set(partitions) != {"zero_state", "prefilled_state"}
                or not isinstance(accumulation, Mapping)
                or set(accumulation) != {"fp32", "bf16"}
            ):
                raise ValueError(f"Incomplete precision/state coverage: {candidate_id}/{variant}")
            partition_passes = [
                isinstance(state_row, Mapping)
                and set(state_row) == {"fp32", "bf16"}
                and all(
                    isinstance(state_row[precision], Mapping)
                    and state_row[precision].get("passed") is True
                    for precision in ("fp32", "bf16")
                )
                for state_row in partitions.values()
            ]
            accumulation_passes = [
                isinstance(accumulation[precision], list)
                and bool(accumulation[precision])
                and all(
                    isinstance(item, Mapping) and item.get("passed") is True
                    for item in accumulation[precision]
                )
                for precision in ("fp32", "bf16")
            ]
            observed_pass = (
                all(partition_passes)
                and all(accumulation_passes)
                and variant_row.get("passed") is True
            )
            variant_passes.append(observed_pass)
        candidate_pass = all(variant_passes) and row.get("passed") is True
        candidate_passes.append(candidate_pass)
        normalized_candidates[candidate_id] = row

    all_passed = all(candidate_passes)
    payload: dict[str, Any] = {
        "schema_version": "catena-v8.1",
        "manifest_type": "E26_CANDIDATE_MATRIX_NUMERICAL_AUDIT",
        "scientific_evidence": False,
        "evidence_tier": "NON_EVIDENCE_VALIDATION",
        "claim_ceiling": "PROTOCOL_IDENTIFIABILITY_ONLY",
        "all_passed": all_passed,
        "passed": all_passed,
        "main_test_opened": False,
        "expected_candidate_ids": list(normalized_ids),
        "locked_hashes": normalized_hashes,
        "source_inventory": dict(source_inventory),
        "candidate_audits": normalized_candidates,
    }
    payload["receipt_sha256"] = sha256_canonical_json(payload)
    return payload
