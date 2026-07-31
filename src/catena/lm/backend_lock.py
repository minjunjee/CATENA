from __future__ import annotations

import hashlib
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from catena.core.provenance_v61 import (
    DEFAULT_EXCLUDED_DIRECTORY_NAMES,
    DEFAULT_EXCLUDED_DIRECTORY_PREFIXES,
    SHA256_PATTERN,
    sha256_canonical_json,
    sha256_file,
)

from .audit_contract import (
    E26_EXECUTION_SOURCE_SUFFIXES,
    e26_execution_source_inventory,
)


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments],
        cwd=repo,
        text=True,
        stderr=subprocess.PIPE,
    ).strip()


def _require_ancestor_commit(
    repo: Path,
    commit: Any,
    *,
    label: str,
) -> str:
    if not isinstance(commit, str) or len(commit) != 40:
        raise ValueError(f"{label} must be a full 40-character Git commit")
    try:
        _git(repo, "rev-parse", "--verify", f"{commit}^{{commit}}")
        _git(repo, "merge-base", "--is-ancestor", commit, "HEAD")
    except subprocess.CalledProcessError as error:
        raise ValueError(f"{label} is not an existing ancestor of the current HEAD") from error
    return commit


def _execution_source_path(relative_path: str) -> bool:
    path = Path(relative_path)
    if path.suffix.lower() not in E26_EXECUTION_SOURCE_SUFFIXES:
        return False
    return not any(
        part in DEFAULT_EXCLUDED_DIRECTORY_NAMES
        or any(part.startswith(prefix) for prefix in DEFAULT_EXCLUDED_DIRECTORY_PREFIXES)
        for part in path.parts[:-1]
    )


def _execution_source_inventory_at_commit(
    repo: Path,
    commit: str,
) -> dict[str, Any]:
    """Reconstruct the execution inventory from immutable Git blob bytes."""

    output = subprocess.check_output(
        ["git", "ls-tree", "-r", "-z", commit],
        cwd=repo,
        stderr=subprocess.PIPE,
    )
    selected: list[tuple[str, str]] = []
    for raw_entry in output.split(b"\0"):
        if not raw_entry:
            continue
        metadata, raw_path = raw_entry.split(b"\t", maxsplit=1)
        _mode, object_type, object_id = metadata.decode("ascii").split()
        if object_type != "blob":
            continue
        relative_path = raw_path.decode("utf-8")
        if _execution_source_path(relative_path):
            selected.append((relative_path, object_id))

    digest = hashlib.sha256()
    rows: list[dict[str, Any]] = []
    for relative_path, object_id in sorted(selected):
        content = subprocess.check_output(
            ["git", "cat-file", "blob", object_id],
            cwd=repo,
            stderr=subprocess.PIPE,
        )
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
        rows.append(
            {
                "path": relative_path,
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    return {
        "algorithm": "relative_path_nul_file_bytes_nul_v1",
        "suffixes": sorted(E26_EXECUTION_SOURCE_SUFFIXES),
        "files": len(rows),
        "rows": rows,
        "source_tree_sha256": digest.hexdigest(),
    }


def _require_exact_commit_inventory(
    repo: Path,
    commit: str,
    recorded_inventory: Any,
    *,
    label: str,
) -> None:
    if not isinstance(recorded_inventory, Mapping):
        raise ValueError(f"{label} source inventory must be a mapping")
    commit_inventory = _execution_source_inventory_at_commit(repo, commit)
    if dict(recorded_inventory) != commit_inventory:
        raise ValueError(
            f"{label} source_commit does not contain the recorded execution-source bytes"
        )


def _capabilities_false(payload: Mapping[str, Any]) -> bool:
    return all(
        payload.get(field) is False
        for field in (
            "e26a_candidate_capable",
            "e26a_gate_capable",
            "scientific_main_capable",
            "parity_verified",
        )
    )


def backend_candidate_lock_payload(
    *,
    repo_root: str | Path,
    config_path: str | Path,
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Create the protocol-bound, pre-audit backend node of the provenance DAG."""

    repo = Path(repo_root).expanduser().resolve(strict=True)
    config = Path(config_path).expanduser().resolve(strict=True)
    inventory = e26_execution_source_inventory(repo)
    candidate_rows = [
        {
            "candidate_id": str(candidate["id"]),
            "model_config_sha256": sha256_canonical_json(dict(candidate)),
        }
        for candidate in candidates
    ]
    payload: dict[str, Any] = {
        "schema_version": "catena-v8.1",
        "manifest_type": "E26_BACKEND_CANDIDATE_LOCK",
        "scientific_evidence": False,
        "evidence_tier": "NON_EVIDENCE_VALIDATION",
        "claim_ceiling": "PROTOCOL_IDENTIFIABILITY_ONLY",
        "main_test_opened": False,
        "source_commit": _git(repo, "rev-parse", "HEAD"),
        "source_inventory": inventory,
        "config_sha256": sha256_file(config),
        "candidate_configs": candidate_rows,
        "backend_id": "torch_compile_fixed_chunk_scan_v1",
        "algorithm": "static_chunk_unrolled_delta_recurrence",
        "compiler": "inductor",
        "fullgraph": True,
        "dynamic_shapes": False,
        "silent_fallback_allowed": False,
        "e26a_candidate_capable": False,
        "e26a_gate_capable": False,
        "scientific_main_capable": False,
        "parity_verified": False,
    }
    payload["manifest_sha256"] = sha256_canonical_json(payload)
    return payload


def validate_backend_candidate_lock(
    payload: Mapping[str, Any],
    *,
    repo_root: str | Path,
    config_path: str | Path,
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    normalized = dict(payload)
    observed_hash = normalized.pop("manifest_sha256", None)
    if not isinstance(observed_hash, str) or observed_hash != sha256_canonical_json(normalized):
        raise ValueError("Backend candidate lock canonical SHA-256 mismatch")
    repo = Path(repo_root).expanduser().resolve(strict=True)
    recorded_commit = _require_ancestor_commit(
        repo,
        normalized.get("source_commit"),
        label="Backend candidate lock source_commit",
    )
    _require_exact_commit_inventory(
        repo,
        recorded_commit,
        normalized.get("source_inventory"),
        label="Backend candidate lock",
    )
    expected = backend_candidate_lock_payload(
        repo_root=repo,
        config_path=config_path,
        candidates=candidates,
    )
    expected.pop("manifest_sha256")
    # A documentation-only descendant commit is allowed after the pre-audit
    # candidate lock. Executable source drift is still rejected by the exact
    # source inventory and config/candidate comparisons below.
    expected["source_commit"] = recorded_commit
    if normalized != expected:
        raise ValueError("Backend candidate lock differs from current execution inputs")
    if not _capabilities_false(normalized):
        raise ValueError("Backend candidate lock improperly opens an E26 capability")
    normalized["manifest_sha256"] = observed_hash
    return normalized


def cuda_hardware_inventory(device_indices: Sequence[str]) -> list[dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA inventory requested without CUDA")
    driver = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=driver_version,uuid",
            "--format=csv,noheader",
            "-i",
            ",".join(device_indices),
        ],
        text=True,
    ).splitlines()
    rows: list[dict[str, Any]] = []
    for position, device_index in enumerate(device_indices):
        properties = torch.cuda.get_device_properties(int(device_index))
        driver_version, gpu_uuid = (
            value.strip() for value in driver[position].split(",", maxsplit=1)
        )
        rows.append(
            {
                "physical_device_index": int(device_index),
                "name": properties.name,
                "total_memory_bytes": properties.total_memory,
                "compute_capability": f"{properties.major}.{properties.minor}",
                "driver_version": driver_version,
                "gpu_uuid": gpu_uuid,
            }
        )
    return rows


def observed_single_visible_cuda_device(
    *,
    expected_physical_index: int,
    expected_gpu_uuid: str,
) -> dict[str, Any]:
    """Bind an isolated worker to the CUDA device it actually observes.

    Parent-side ``CUDA_VISIBLE_DEVICES`` assignment is not evidence that a
    worker used the intended physical device.  The worker therefore checks the
    environment, requires one visible CUDA device, and compares the UUID
    reported by PyTorch with the parent's independently inventoried UUID.
    """

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(expected_physical_index):
        raise RuntimeError(
            "Worker CUDA_VISIBLE_DEVICES differs from the locked physical index: "
            f"{visible!r} != {expected_physical_index!r}"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Isolated worker requires exactly one visible CUDA device")
    properties = torch.cuda.get_device_properties(0)
    observed_uuid = str(getattr(properties, "uuid", ""))
    if observed_uuid and not observed_uuid.startswith("GPU-"):
        observed_uuid = f"GPU-{observed_uuid}"
    if observed_uuid != expected_gpu_uuid:
        raise RuntimeError(
            "Worker-observed CUDA UUID differs from the locked physical device: "
            f"{observed_uuid!r} != {expected_gpu_uuid!r}"
        )
    return {
        "physical_device_index": expected_physical_index,
        "gpu_uuid": observed_uuid,
        "worker_visible_cuda_index": 0,
        "cuda_visible_devices": visible,
        "name": properties.name,
        "total_memory_bytes": int(properties.total_memory),
        "compute_capability": f"{properties.major}.{properties.minor}",
        "observation": "PYTORCH_VISIBLE_DEVICE_UUID_VERIFIED",
    }


def _normalize_hardware_inventory(
    records: Sequence[Mapping[str, Any]],
    *,
    label: str,
) -> list[dict[str, Any]]:
    if not records:
        raise ValueError(f"{label} must contain at least one CUDA device")
    required = {
        "physical_device_index",
        "name",
        "total_memory_bytes",
        "compute_capability",
        "driver_version",
        "gpu_uuid",
    }
    normalized: list[dict[str, Any]] = []
    for position, raw in enumerate(records):
        row = dict(raw)
        if set(row) != required:
            raise ValueError(
                f"{label}[{position}] fields differ from the registered hardware schema"
            )
        physical_index = row["physical_device_index"]
        total_memory = row["total_memory_bytes"]
        if (
            isinstance(physical_index, bool)
            or not isinstance(physical_index, int)
            or physical_index < 0
        ):
            raise ValueError(f"{label}[{position}] has an invalid physical device index")
        if isinstance(total_memory, bool) or not isinstance(total_memory, int) or total_memory <= 0:
            raise ValueError(f"{label}[{position}] has invalid total memory")
        for field in ("name", "compute_capability", "driver_version", "gpu_uuid"):
            if not isinstance(row[field], str) or not row[field]:
                raise ValueError(f"{label}[{position}] has invalid {field}")
        if not row["gpu_uuid"].startswith("GPU-"):
            raise ValueError(f"{label}[{position}] GPU UUID is not canonical")
        normalized.append(row)
    physical_indices = [int(row["physical_device_index"]) for row in normalized]
    gpu_uuids = [str(row["gpu_uuid"]) for row in normalized]
    if len(physical_indices) != len(set(physical_indices)):
        raise ValueError(f"{label} repeats a physical device index")
    if len(gpu_uuids) != len(set(gpu_uuids)):
        raise ValueError(f"{label} repeats a GPU UUID")
    return sorted(normalized, key=lambda row: int(row["physical_device_index"]))


def _compiled_diagnostics(
    candidate_rows: Mapping[str, Any],
    *,
    expected_candidate_ids: Sequence[str],
) -> tuple[dict[str, dict[str, dict[str, Any]]], bool]:
    expected_ids = tuple(str(value) for value in expected_candidate_ids)
    if set(candidate_rows) != set(expected_ids):
        raise ValueError("Compiled diagnostics do not cover the locked candidate set")
    expected_variants = {"dual_delta_lm", "projected_tied_delta_lm"}
    diagnostics: dict[str, dict[str, dict[str, Any]]] = {}
    all_candidates_proved = True
    for candidate_id in expected_ids:
        candidate = candidate_rows[candidate_id]
        if not isinstance(candidate, Mapping):
            raise ValueError(f"Malformed candidate audit row: {candidate_id}")
        variant_rows = candidate.get("variants")
        if not isinstance(variant_rows, Mapping) or set(variant_rows) != expected_variants:
            raise ValueError(f"Compiled diagnostics variant coverage mismatch: {candidate_id}")
        candidate_diagnostics: dict[str, dict[str, Any]] = {}
        candidate_compilations = 0
        candidate_code_hashes = 0
        for variant in sorted(expected_variants):
            variant_row = variant_rows[variant]
            if not isinstance(variant_row, Mapping):
                raise ValueError(f"Malformed candidate variant audit: {candidate_id}/{variant}")
            raw_diagnostics = variant_row.get("compiled_backend_diagnostics")
            if not isinstance(raw_diagnostics, Mapping):
                raise ValueError(f"Missing compiled backend diagnostics: {candidate_id}/{variant}")
            row = dict(raw_diagnostics)
            integer_fields = (
                "graph_compilations",
                "graph_invocations",
                "optimized_calls",
                "chunks_executed",
                "padded_tokens",
                "fallback_count",
                "graph_break_count",
                "last_graph_node_count",
            )
            if any(
                isinstance(row.get(field), bool)
                or not isinstance(row.get(field), int)
                or int(row[field]) < 0
                for field in integer_fields
            ):
                raise ValueError(f"Malformed compiled counter: {candidate_id}/{variant}")
            compilation_count = int(row["graph_compilations"])
            code_hash = row.get("last_graph_code_sha256")
            if compilation_count > 0:
                if (
                    int(row["last_graph_node_count"]) <= 0
                    or not isinstance(code_hash, str)
                    or not SHA256_PATTERN.fullmatch(code_hash)
                ):
                    raise ValueError(f"Compiled graph identity is absent: {candidate_id}/{variant}")
                candidate_code_hashes += 1
            elif code_hash is not None or int(row["last_graph_node_count"]) != 0:
                raise ValueError(
                    f"Cached-graph diagnostics are internally inconsistent: "
                    f"{candidate_id}/{variant}"
                )
            variant_proved = (
                int(row["graph_invocations"]) > 0
                and int(row["optimized_calls"]) > 0
                and int(row["chunks_executed"]) > 0
                and int(row["fallback_count"]) == 0
                and int(row["graph_break_count"]) == 0
            )
            all_candidates_proved = all_candidates_proved and variant_proved
            candidate_compilations += compilation_count
            candidate_diagnostics[variant] = row
        candidate_proved = candidate_compilations > 0 and candidate_code_hashes > 0
        all_candidates_proved = all_candidates_proved and candidate_proved
        diagnostics[candidate_id] = candidate_diagnostics
    return diagnostics, all_candidates_proved


def _validate_execution_device_bindings(
    *,
    candidate_rows: Mapping[str, Any],
    restart_receipt: Mapping[str, Any],
    hardware_inventory: Sequence[Mapping[str, Any]],
) -> None:
    hardware_by_index = {int(row["physical_device_index"]): dict(row) for row in hardware_inventory}

    def validate_observation(device: Any, *, label: str) -> None:
        if not isinstance(device, Mapping):
            raise ValueError(f"{label} lacks an observed execution-device mapping")
        expected_fields = {
            "physical_device_index",
            "gpu_uuid",
            "worker_visible_cuda_index",
            "cuda_visible_devices",
            "name",
            "total_memory_bytes",
            "compute_capability",
            "observation",
        }
        if set(device) != expected_fields:
            raise ValueError(f"{label} execution-device binding is malformed")
        physical_index = device.get("physical_device_index")
        if isinstance(physical_index, bool) or not isinstance(physical_index, int):
            raise ValueError(f"{label} physical device index is malformed")
        hardware = hardware_by_index.get(physical_index)
        if hardware is None:
            raise ValueError(f"{label} execution device is absent from hardware inventory")
        expected_values = {
            "gpu_uuid": hardware["gpu_uuid"],
            "worker_visible_cuda_index": 0,
            "cuda_visible_devices": str(physical_index),
            "name": hardware["name"],
            "total_memory_bytes": hardware["total_memory_bytes"],
            "compute_capability": hardware["compute_capability"],
            "observation": "PYTORCH_VISIBLE_DEVICE_UUID_VERIFIED",
        }
        mismatched = [
            field for field, expected in expected_values.items() if device.get(field) != expected
        ]
        if mismatched:
            raise ValueError(
                f"{label} execution-device observation differs from hardware "
                f"inventory: {mismatched}"
            )

    for candidate_id, raw_candidate in candidate_rows.items():
        if not isinstance(raw_candidate, Mapping):
            raise ValueError(f"Malformed candidate audit row: {candidate_id}")
        validate_observation(
            raw_candidate.get("execution_device"),
            label=f"Candidate {candidate_id}",
        )
    restart_cases = restart_receipt.get("resume_cases")
    if not isinstance(restart_cases, Mapping) or not restart_cases:
        raise ValueError("Restart receipt lacks physical execution-device cases")
    for case_id, raw_case in restart_cases.items():
        if not isinstance(raw_case, Mapping):
            raise ValueError(f"Malformed restart case: {case_id}")
        execution_device = raw_case.get("execution_device")
        validate_observation(
            execution_device,
            label=f"Restart case {case_id}",
        )
        if not isinstance(execution_device, Mapping):
            raise AssertionError("Execution-device validation did not narrow mapping")
        if raw_case.get("physical_device_index") != execution_device.get(
            "physical_device_index"
        ) or raw_case.get("gpu_uuid") != execution_device.get("gpu_uuid"):
            raise ValueError(
                f"Restart case top-level device binding conflicts with its observed "
                f"execution device: {case_id}"
            )


def backend_preflight_manifest(
    *,
    candidate_lock_path: str | Path,
    candidate_lock: Mapping[str, Any],
    numerical_receipt_path: str | Path,
    numerical_receipt: Mapping[str, Any],
    restart_receipt_path: str | Path,
    restart_receipt: Mapping[str, Any],
    hardware_inventory: Sequence[Mapping[str, Any]],
    source_inventory: Mapping[str, Any],
    source_commit: str,
) -> dict[str, Any]:
    all_passed = numerical_receipt.get("passed") is True and restart_receipt.get("passed") is True
    candidate_rows = numerical_receipt.get("candidate_audits")
    if not isinstance(candidate_rows, Mapping) or not candidate_rows:
        raise ValueError("Numerical receipt lacks candidate audit rows")
    candidate_configs = candidate_lock.get("candidate_configs")
    if not isinstance(candidate_configs, list) or not candidate_configs:
        raise ValueError("Backend candidate lock lacks candidate configs")
    expected_candidate_ids = [
        str(row.get("candidate_id")) for row in candidate_configs if isinstance(row, Mapping)
    ]
    if len(expected_candidate_ids) != len(candidate_configs):
        raise ValueError("Backend candidate lock candidate configs are malformed")
    diagnostics, codegen_proved = _compiled_diagnostics(
        candidate_rows,
        expected_candidate_ids=expected_candidate_ids,
    )
    if not codegen_proved:
        raise ValueError("Backend candidate diagnostics do not prove compiled codegen capability")
    fallback_count = sum(
        int(value.get("fallback_count", 0))
        for candidate in diagnostics.values()
        for value in candidate.values()
    )
    graph_break_count = sum(
        int(value.get("graph_break_count", 0))
        for candidate in diagnostics.values()
        for value in candidate.values()
    )
    all_passed = all_passed and codegen_proved and fallback_count == 0 and graph_break_count == 0
    normalized_hardware = _normalize_hardware_inventory(
        hardware_inventory,
        label="Backend preflight hardware inventory",
    )
    _validate_execution_device_bindings(
        candidate_rows=candidate_rows,
        restart_receipt=restart_receipt,
        hardware_inventory=normalized_hardware,
    )
    payload: dict[str, Any] = {
        "schema_version": "catena-v8.1",
        "manifest_type": "E26_BACKEND_PREFLIGHT_MANIFEST",
        "scientific_evidence": False,
        "evidence_tier": "NON_EVIDENCE_VALIDATION",
        "claim_ceiling": "PROTOCOL_IDENTIFIABILITY_ONLY",
        "main_test_opened": False,
        "source_commit": source_commit,
        "source_inventory": dict(source_inventory),
        "candidate_lock": {
            "path": str(Path(candidate_lock_path).resolve()),
            "sha256": sha256_file(candidate_lock_path),
            "manifest_sha256": candidate_lock["manifest_sha256"],
        },
        "numerical_audit": {
            "path": str(Path(numerical_receipt_path).resolve()),
            "sha256": sha256_file(numerical_receipt_path),
            "receipt_sha256": numerical_receipt["receipt_sha256"],
        },
        "restart_audit": {
            "path": str(Path(restart_receipt_path).resolve()),
            "sha256": sha256_file(restart_receipt_path),
            "receipt_sha256": restart_receipt["receipt_sha256"],
        },
        "backend_id": "torch_compile_fixed_chunk_scan_v1",
        "backend_type": "TORCH_COMPILED",
        "algorithm": "static_chunk_unrolled_delta_recurrence",
        "compiler": "inductor",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "hardware_inventory": normalized_hardware,
        "candidate_audits": diagnostics,
        "fallback_count": fallback_count,
        "graph_break_count": graph_break_count,
        "candidate_codegen_capable": codegen_proved,
        "e26a_candidate_capable": all_passed,
        "e26a_gate_capable": False,
        "scientific_main_capable": False,
        "parity_verified": False,
    }
    payload["manifest_sha256"] = sha256_canonical_json(payload)
    return payload


def validate_backend_preflight_manifest(
    payload: Mapping[str, Any],
    *,
    repo_root: str | Path,
    candidate_lock_path: str | Path,
    candidate_lock: Mapping[str, Any],
    numerical_receipt_path: str | Path,
    numerical_receipt: Mapping[str, Any],
    restart_receipt_path: str | Path,
    restart_receipt: Mapping[str, Any],
    expected_hardware_inventory: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate the post-audit backend promotion without reopening E26a MAIN.

    The promotion may be consumed from a documentation-only descendant commit.
    Its recorded commit must be an ancestor, while the execution-source
    inventory remains byte-exact.
    """

    repo = Path(repo_root).expanduser().resolve(strict=True)
    normalized = dict(payload)
    observed_hash = normalized.pop("manifest_sha256", None)
    if not isinstance(observed_hash, str) or observed_hash != sha256_canonical_json(normalized):
        raise ValueError("Backend preflight manifest canonical SHA-256 mismatch")
    recorded_commit = _require_ancestor_commit(
        repo,
        normalized.get("source_commit"),
        label="Backend preflight source_commit",
    )
    _require_exact_commit_inventory(
        repo,
        recorded_commit,
        normalized.get("source_inventory"),
        label="Backend preflight",
    )
    current_inventory = e26_execution_source_inventory(repo)
    if normalized.get("source_inventory") != current_inventory:
        raise ValueError(
            "Backend preflight source inventory differs from current execution sources"
        )

    bindings = (
        (
            "candidate_lock",
            candidate_lock_path,
            candidate_lock.get("manifest_sha256"),
            "manifest_sha256",
        ),
        (
            "numerical_audit",
            numerical_receipt_path,
            numerical_receipt.get("receipt_sha256"),
            "receipt_sha256",
        ),
        (
            "restart_audit",
            restart_receipt_path,
            restart_receipt.get("receipt_sha256"),
            "receipt_sha256",
        ),
    )
    for label, path, embedded_hash, embedded_name in bindings:
        observed = normalized.get(label)
        if not isinstance(observed, Mapping):
            raise ValueError(f"Backend preflight lacks {label} binding")
        expected = {
            "path": str(Path(path).expanduser().resolve(strict=True)),
            "sha256": sha256_file(path),
            embedded_name: embedded_hash,
        }
        if dict(observed) != expected:
            raise ValueError(f"Backend preflight {label} binding mismatch")

    candidate_rows = numerical_receipt.get("candidate_audits")
    if not isinstance(candidate_rows, Mapping) or not candidate_rows:
        raise ValueError("Numerical receipt lacks candidate audit rows")
    candidate_configs = candidate_lock.get("candidate_configs")
    if not isinstance(candidate_configs, list) or not candidate_configs:
        raise ValueError("Backend candidate lock lacks candidate configs")
    expected_candidate_ids = [
        str(row.get("candidate_id")) for row in candidate_configs if isinstance(row, Mapping)
    ]
    if len(expected_candidate_ids) != len(candidate_configs):
        raise ValueError("Backend candidate lock candidate configs are malformed")
    expected_diagnostics, codegen_proved = _compiled_diagnostics(
        candidate_rows,
        expected_candidate_ids=expected_candidate_ids,
    )
    if normalized.get("candidate_audits") != expected_diagnostics:
        raise ValueError("Backend preflight compiled diagnostics mismatch")
    recorded_hardware = normalized.get("hardware_inventory")
    if not isinstance(recorded_hardware, list):
        raise ValueError("Backend preflight hardware inventory is not a list")
    normalized_recorded_hardware = _normalize_hardware_inventory(
        recorded_hardware,
        label="Backend preflight recorded hardware inventory",
    )
    normalized_expected_hardware = _normalize_hardware_inventory(
        expected_hardware_inventory,
        label="Current expected hardware inventory",
    )
    if normalized_recorded_hardware != normalized_expected_hardware:
        raise ValueError(
            "Backend preflight hardware inventory differs from current expected hardware"
        )
    _validate_execution_device_bindings(
        candidate_rows=candidate_rows,
        restart_receipt=restart_receipt,
        hardware_inventory=normalized_recorded_hardware,
    )
    expected_fallback_count = sum(
        int(value.get("fallback_count", 0))
        for candidate in expected_diagnostics.values()
        for value in candidate.values()
    )
    expected_graph_break_count = sum(
        int(value.get("graph_break_count", 0))
        for candidate in expected_diagnostics.values()
        for value in candidate.values()
    )
    if normalized.get("fallback_count") != expected_fallback_count:
        raise ValueError("Backend preflight fallback count mismatch")
    if normalized.get("graph_break_count") != expected_graph_break_count:
        raise ValueError("Backend preflight graph-break count mismatch")
    expected_backend_values = {
        "backend_id": "torch_compile_fixed_chunk_scan_v1",
        "backend_type": "TORCH_COMPILED",
        "algorithm": "static_chunk_unrolled_delta_recurrence",
        "compiler": "inductor",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    mismatched_backend = [
        key for key, expected in expected_backend_values.items() if normalized.get(key) != expected
    ]
    if mismatched_backend:
        raise ValueError(f"Backend preflight runtime/backend fields changed: {mismatched_backend}")

    if not codegen_proved or normalized.get("candidate_codegen_capable") is not True:
        raise ValueError("Backend preflight did not preserve candidate codegen capability")
    all_audits_passed = (
        numerical_receipt.get("passed") is True
        and restart_receipt.get("passed") is True
        and normalized.get("fallback_count") == 0
        and normalized.get("graph_break_count") == 0
    )
    if normalized.get("e26a_candidate_capable") is not all_audits_passed:
        raise ValueError("Backend preflight candidate capability contradicts audits")
    if any(
        normalized.get(field) is not False
        for field in (
            "e26a_gate_capable",
            "scientific_main_capable",
            "parity_verified",
        )
    ):
        raise ValueError("Backend preflight improperly opens a downstream capability")
    if normalized.get("scientific_evidence") is not False:
        raise ValueError("Backend preflight must remain non-evidence")
    if normalized.get("main_test_opened") is not False:
        raise ValueError("Backend preflight must not open the main test")

    normalized["manifest_sha256"] = observed_hash
    return normalized
