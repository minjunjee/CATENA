"""Pinned official-only minimal replications for CATENA E25a.

This module is intentionally an orchestration layer over the exact official
``chunk_gdn2`` and ``chunk_kda`` entry points loaded by the separately pinned
E25a gate adapter.  It contains no reference recurrence and no operator
fallback.  The public entry point is ``run_minimal_replications(config)``.
"""

from __future__ import annotations

import importlib
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from catena.core.config import load_config
from catena.core.io import file_sha256

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LOCK_PATH = _REPO_ROOT / "docs/E25A_OFFICIAL_GDN2_LOCK.json"
_SELF_RELATIVE = "src/catena_official_plugins/e25a_replication.py"

_GDN2_COMMIT = "95709fc250357c2dd109361c353192f2aa5913f9"
_FLA_COMMIT = "4b02d15d6a68700181b180235be62a9fb95d2a38"
_GATE_ADAPTER_SHA256 = "e5643656de1a9ba164f78c4bdd46a66b971acedf0f84a8114a7cf3b38ba575a3"
_SOURCE_CONFIG_SHA256 = {
    "e02b": "4e269572652f494977f0370a82c5104bd083ae4e7a4f570635d4685af15fe956",
    "e18": "12fab86ea863aec7ec19e7393374d70afbf93aac8e7a4c7d1170ce2f8c3164bf",
    "e22": "49c719dcf93ae8b6ecd5bd6c5c7b91bf8c855b99eb0348799b86fc5da73d724f",
}
_E22B_PROTOCOL_LOCK_SHA256 = "e19dfd26018e53d7ab601d1bd1b0e94c3bd922e1849c35cdcffec7ae38474598"
_REQUIRED_SUBSETS = (
    "e02b_magnitude_factorization",
    "e18_magnitude_sequence",
    "e22_locality_if_supported",
)


class OfficialReplicationNotConfigured(RuntimeError):
    """The exact official-only replication runtime is incomplete."""


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a list")
    return value


def _resolve_repo_file(value: object, label: str) -> Path:
    raw = Path(str(value))
    path = (_REPO_ROOT / raw).resolve() if not raw.is_absolute() else raw.resolve()
    try:
        path.relative_to(_REPO_ROOT)
    except ValueError as error:
        raise ValueError(f"{label} escapes the CATENA repository") from error
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"{label} is missing or unsafe: {path}")
    return path


def _require_self_lock() -> dict[str, Any]:
    if not _LOCK_PATH.is_file() or _LOCK_PATH.is_symlink():
        raise OfficialReplicationNotConfigured("E25a static protocol lock is unavailable")
    payload = json.loads(_LOCK_PATH.read_text(encoding="utf-8"))
    files = _mapping(payload.get("files"), "E25a lock files")
    expected = str(files.get(_SELF_RELATIVE, ""))
    actual = file_sha256(Path(__file__).resolve())
    if expected != actual:
        raise RuntimeError("official replication source differs from the frozen E25a protocol lock")
    if payload.get("scientific_replication_started") is not False:
        raise RuntimeError("E25a lock does not authorize a fresh first replication")
    return cast(dict[str, Any], payload)


def validate_source_contracts(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the immutable E02b/E18/E22 source tensor/metric contracts."""

    replication = _mapping(config.get("replication"), "replication")
    subsets = tuple(str(value) for value in _list(replication.get("subsets"), "subsets"))
    if subsets != _REQUIRED_SUBSETS:
        raise ValueError("E25a replication subset order differs from the protocol")
    tensor_contract = _mapping(
        replication.get("source_tensor_contract"),
        "replication.source_tensor_contract",
    )
    paths = {
        "e02b": _resolve_repo_file(tensor_contract.get("e02b_config"), "E02b config"),
        "e18": _resolve_repo_file(tensor_contract.get("e18_config"), "E18 config"),
        "e22": _resolve_repo_file(tensor_contract.get("e22_config"), "E22 config"),
    }
    for name, path in paths.items():
        actual = file_sha256(path)
        if actual != _SOURCE_CONFIG_SHA256[name]:
            raise ValueError(f"{name} source config hash mismatch")
    e02b = load_config(paths["e02b"])
    e18 = load_config(paths["e18"])
    e22 = load_config(paths["e22"])
    if e02b.get("experiment_id") != "e02b_prospective_absolute_supersede":
        raise ValueError("unexpected E02b source experiment")
    if list(e02b.get("seeds", [])) != [11, 22, 33, 44, 55, 66, 77, 88]:
        raise ValueError("E02b seed contract changed")
    if dict(e02b["data"]) != {
        "count_per_operation": 512,
        "num_associations": 16,
        "key_dim": 32,
        "value_dim": 32,
        "key_correlation": 0.15,
    }:
        raise ValueError("E02b data dimensions changed")
    if list(e02b["geometry_grid"]["norm_pairs"]) != [
        [0.75, 0.90],
        [0.90, 1.25],
        [1.10, 0.80],
        [1.25, 1.10],
    ]:
        raise ValueError("E02b norm grid changed")
    if list(e02b["geometry_grid"]["angles_degrees"]) != [45.0, 75.0, 105.0, 135.0]:
        raise ValueError("E02b angle grid changed")
    if int(e02b["geometry_grid"]["repeats_per_cell"]) != 32:
        raise ValueError("E02b cell count changed")
    if e18.get("experiment_id") != "e18a_sequence_control_lattice":
        raise ValueError("unexpected E18 source experiment")
    if list(e18.get("seeds", [])) != [101, 211, 307, 401, 503]:
        raise ValueError("E18 seed contract changed")
    if (
        int(e18["data"]["num_entities"]) != 32
        or int(e18["data"]["value_dim"]) != 32
        or list(e18["evaluation"]["updates"]) != [1, 4, 8]
        or list(e18["evaluation"]["gap_events"]) != [0, 128, 512, 2048]
    ):
        raise ValueError("E18 tensor/grid contract changed")
    if list(e18["model"]["variants"])[:2] != ["tied_scalar", "dual_scalar"]:
        raise ValueError("E18 tied/dual source variants changed")
    if e22.get("experiment_id") != str(tensor_contract.get("e22_experiment_id")):
        raise ValueError("unexpected E22 source experiment")
    if (
        list(e22["confirmatory_seeds"]) != [1301, 1319, 1327, 1361, 1381, 1409, 1423, 1451]
        or int(e22["data"]["slots"]) != 32
        or int(e22["data"]["value_dim"]) != 32
    ):
        raise ValueError("E22 tensor/seed contract changed")
    return {
        "paths": {name: str(path) for name, path in paths.items()},
        "sha256": dict(_SOURCE_CONFIG_SHA256),
        "configs": {"e02b": e02b, "e18": e18, "e22": e22},
    }


def validate_safe_e22_report(
    path: Path | None,
    *,
    expected_experiment_id: str = "e22b_active_path_locality",
) -> dict[str, Any]:
    """Return an explicit include/skip decision for the optional locality subset."""

    if path is None:
        return {
            "include": False,
            "reason": "EXPLICIT_SAFE_E22B_REPORT_NOT_PROVIDED",
        }
    resolved = path.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise OfficialReplicationNotConfigured(
            f"explicit E22b report is missing or unsafe: {resolved}"
        )
    report = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("E22b report must be a JSON object")
    manifest_path = resolved.parent / "run_manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("explicit E22b report lacks a safe sibling run_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("E22b run manifest must be a JSON object")
    experiment_id = manifest.get("experiment_id", report.get("experiment_id"))
    summary = _mapping(report.get("summary"), "E22b report.summary")
    claim_gate = _mapping(report.get("claim_gate"), "E22b report.claim_gate")
    report_protocol_lock = _mapping(
        report.get("protocol_lock"),
        "E22b report.protocol_lock",
    )
    locality_comparison = _mapping(
        summary.get("selected_vs_mean_locality"),
        "E22b report.summary.selected_vs_mean_locality",
    )
    explicitly_safe = bool(
        experiment_id == expected_experiment_id
        and report.get("experiment_id") == expected_experiment_id
        and report.get("status", report.get("execution_status")) == "PASS"
        and report.get("execution_status") == "PASS"
        and report.get("run_mode") == "MAIN"
        and report.get("run_scope") == "E22B_ACTIVE_PATH_LOCALITY_CONFIRMATORY"
        and report.get("evidence_tier") == "CONTROLLED_REFERENCE"
        and report.get("claim_eligible") is True
        and report_protocol_lock.get("sha256") == _E22B_PROTOCOL_LOCK_SHA256
        and summary.get("status") == "SUPPORTED_SAFE_LOCALIZED_ASSIMILATION_CONTROLLED"
        and summary.get("supported") is True
        and summary.get("dry_run_non_evidence") is False
        and summary.get("seed_count") == 8
        and summary.get("recovery_pattern_gate_passed") is True
        and summary.get("capacity_gate_passed") is True
        and summary.get("recovery_capacity_gate_passed") is True
        and summary.get("absolute_locality_gate_passed") is True
        and summary.get("retention_gate_passed") is True
        and summary.get("locality_retention_gate_passed") is True
        and locality_comparison.get("passed") is True
        and claim_gate.get("status") == "SUPPORTED_SAFE_LOCALIZED_ASSIMILATION_CONTROLLED"
        and claim_gate.get("supported") is True
    )
    if not explicitly_safe:
        raise ValueError("explicit E22b report does not open the safe locality subset")
    return {
        "include": True,
        "reason": "EXPLICIT_SAFE_E22B_REPORT_VALIDATED",
        "path": str(resolved),
        "sha256": file_sha256(resolved),
        "run_manifest_path": str(manifest_path.resolve()),
        "run_manifest_sha256": file_sha256(manifest_path),
        "protocol_lock_sha256": _E22B_PROTOCOL_LOCK_SHA256,
    }


def _official_context(config: Mapping[str, Any]) -> dict[str, Any]:
    backend = _mapping(config.get("backend"), "backend")
    if str(backend.get("expected_commit")) != _GDN2_COMMIT:
        raise ValueError("replication plugin GDN2 commit mismatch")
    if str(backend.get("fla_expected_commit")) != _FLA_COMMIT:
        raise ValueError("replication plugin FLA commit mismatch")
    adapter = importlib.import_module(str(backend.get("plugin_module")))
    raw_origin = getattr(adapter, "__file__", None)
    if raw_origin is None or file_sha256(Path(raw_origin).resolve()) != _GATE_ADAPTER_SHA256:
        raise RuntimeError("pinned official gate adapter provenance mismatch")
    required = (
        "_require_pinned_repo",
        "_load_official_apis",
        "_resolve_device",
        "_strict_fp32",
    )
    if any(not callable(getattr(adapter, name, None)) for name in required):
        raise AttributeError("pinned official gate adapter API contract changed")
    try:
        torch = importlib.import_module("torch")
    except ModuleNotFoundError as error:
        raise OfficialReplicationNotConfigured("PyTorch is unavailable") from error
    gdn2_repo = adapter._require_pinned_repo(  # noqa: SLF001
        Path(str(backend["repo_path"])),
        expected_commit=_GDN2_COMMIT,
        label="GDN2",
    )
    fla_repo = adapter._require_pinned_repo(  # noqa: SLF001
        Path(str(backend["fla_repo_path"])),
        expected_commit=_FLA_COMMIT,
        label="Flash Linear Attention",
    )
    device = adapter._resolve_device(torch)  # noqa: SLF001
    chunk_gdn2, chunk_kda, _ = adapter._load_official_apis(  # noqa: SLF001
        gdn2_repo=gdn2_repo,
        fla_repo=fla_repo,
    )
    return {
        "torch": torch,
        "device": device,
        "chunk_gdn2": chunk_gdn2,
        "chunk_kda": chunk_kda,
        "strict_fp32": adapter._strict_fp32,  # noqa: SLF001
        "gdn2_repo": str(gdn2_repo),
        "fla_repo": str(fla_repo),
    }


def _gdn2(chunk_gdn2: Any, probe: Mapping[str, Any]) -> Any:
    _, final_state = chunk_gdn2(
        probe["q"],
        probe["k"],
        probe["v"],
        probe["g"],
        probe["b"],
        probe["w"],
        scale=1.0,
        initial_state=probe["initial_state"],
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
        use_gate_in_kernel=False,
        safe_gate=False,
        disable_recompute=False,
    )
    if final_state is None:
        raise RuntimeError("official GDN2 did not return a final state")
    return final_state


def _kda(chunk_kda: Any, probe: Mapping[str, Any]) -> Any:
    _, final_state = chunk_kda(
        probe["q"],
        probe["k"],
        probe["v"],
        probe["g"],
        probe["beta"],
        scale=1.0,
        initial_state=probe["initial_state"],
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
        use_gate_in_kernel=False,
        safe_gate=False,
        disable_recompute=False,
    )
    if final_state is None:
        raise RuntimeError("official KDA did not return a final state")
    return final_state


def _unit_rows(torch: Any, batch: int, dimension: int, offset: int = 0) -> Any:
    indices = (torch.arange(batch, dtype=torch.long) + int(offset)) % dimension
    return torch.nn.functional.one_hot(indices, num_classes=dimension).to(torch.float32)


def _paired_values(
    torch: Any,
    *,
    batch: int,
    dimension: int,
    old_scale: float,
    new_scale: float,
    cosine: float,
    seed: int,
) -> tuple[Any, Any]:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    old = torch.randn(batch, dimension, generator=generator)
    old = old / old.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
    orthogonal = torch.randn(batch, dimension, generator=generator)
    orthogonal -= (orthogonal * old).sum(dim=-1, keepdim=True) * old
    orthogonal /= orthogonal.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
    sine = math.sqrt(max(0.0, 1.0 - float(cosine) ** 2))
    new = float(cosine) * old + sine * orthogonal
    return float(old_scale) * old, float(new_scale) * new


def _optimal_tied_beta(torch: Any, old: Any, new: Any, erase: Any, write: Any) -> Any:
    direction = new - old
    target = -erase.unsqueeze(-1) * old + write.unsqueeze(-1) * new
    numerator = (direction * target).sum(dim=-1)
    denominator = direction.square().sum(dim=-1).clamp_min(1.0e-12)
    return (numerator / denominator).clamp(0.0, 1.0)


def _state_metrics(torch: Any, actual: Any, target: Any, affected: Any) -> dict[str, float]:
    state = actual[:, 0].float()
    target_state = target.float()
    entity_error = (state - target_state).square().mean(dim=-1)
    unaffected = ~affected
    return {
        "state_mse": float(entity_error.mean().item()),
        "affected_mse": float(entity_error[affected].mean().item()),
        "retention_mse": float(entity_error[unaffected].mean().item()),
    }


def _e02_probe(
    torch: Any,
    *,
    seed: int,
    operation: str,
    config: Mapping[str, Any],
    device: Any,
) -> tuple[dict[str, Any], Any]:
    dimension = int(config["data"]["value_dim"])
    pairs = list(config["geometry_grid"]["norm_pairs"])
    angles = list(config["geometry_grid"]["angles_degrees"])
    repeats = int(config["geometry_grid"]["repeats_per_cell"])
    old_values: list[Any] = []
    new_values: list[Any] = []
    for pair_index, (old_scale, new_scale) in enumerate(pairs):
        for angle_index, angle in enumerate(angles):
            old, new = _paired_values(
                torch,
                batch=repeats,
                dimension=dimension,
                old_scale=float(old_scale),
                new_scale=float(new_scale),
                cosine=math.cos(math.radians(float(angle))),
                seed=seed * 100_000 + pair_index * 1_000 + angle_index,
            )
            old_values.append(old)
            new_values.append(new)
    old = torch.cat(old_values, dim=0)
    new = torch.cat(new_values, dim=0)
    batch = int(old.shape[0])
    key = _unit_rows(torch, batch, dimension)
    retained_key = _unit_rows(torch, batch, dimension, offset=1)
    generator = torch.Generator(device="cpu").manual_seed(seed * 100_000 + 99_001)
    retained_value = torch.randn(batch, dimension, generator=generator)
    retained_value /= retained_value.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
    initial_state = torch.einsum("bk,bv->bkv", key, old)
    initial_state += torch.einsum("bk,bv->bkv", retained_key, retained_value)
    erase_value = 0.0 if operation == "ADD" else 1.0
    write_value = 1.0 if operation == "ADD" else 0.0
    erase = torch.full((batch,), erase_value)
    write = torch.full((batch,), write_value)
    target = initial_state - erase[:, None, None] * torch.einsum("bk,bv->bkv", key, old)
    target += write[:, None, None] * torch.einsum("bk,bv->bkv", key, new)
    beta = _optimal_tied_beta(torch, old, new, erase, write)
    qk = key[:, None, None, :]
    probe = {
        "q": qk,
        "k": qk,
        "v": new[:, None, None, :],
        "g": torch.zeros(batch, 1, 1, dimension),
        "b": erase[:, None, None, None].expand(batch, 1, 1, dimension),
        "w": write[:, None, None, None].expand(batch, 1, 1, dimension),
        "beta": beta[:, None, None],
        "initial_state": initial_state[:, None],
    }
    moved = {name: value.to(device=device).contiguous() for name, value in probe.items()}
    metadata = {
        "target": target.to(device=device),
        "affected": torch.nn.functional.one_hot(
            torch.arange(batch) % dimension,
            num_classes=dimension,
        )
        .to(device=device)
        .bool(),
        "batch_size": batch,
        "projection_scope": "operator_compatible_orthogonal_address_projection",
    }
    return moved, metadata


def _run_e02(
    context: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    torch = context["torch"]
    rows: list[dict[str, Any]] = []
    statistics = config["statistics"]
    with context["strict_fp32"](torch):
        for seed in config["seeds"]:
            operation_rows: dict[str, dict[str, Any]] = {}
            for operation in ("ADD", "INVALIDATE"):
                probe, metadata = _e02_probe(
                    torch,
                    seed=int(seed),
                    operation=operation,
                    config=config,
                    device=context["device"],
                )
                dual_state = _gdn2(context["chunk_gdn2"], probe)
                tied_state = _kda(context["chunk_kda"], probe)
                dual = _state_metrics(torch, dual_state, metadata["target"], metadata["affected"])
                tied = _state_metrics(torch, tied_state, metadata["target"], metadata["affected"])
                headroom = tied["affected_mse"]
                gain = (
                    (tied["affected_mse"] - dual["affected_mse"]) / headroom
                    if headroom >= float(statistics["minimum_tied_oracle_headroom"])
                    else None
                )
                operation_rows[operation] = {
                    "tied_affected_mse": tied["affected_mse"],
                    "dual_affected_mse": dual["affected_mse"],
                    "dual_retention_mse": dual["retention_mse"],
                    "tied_retention_mse": tied["retention_mse"],
                    "normalized_gain": gain,
                }
            gains = [
                float(item["normalized_gain"])
                for item in operation_rows.values()
                if item["normalized_gain"] is not None
            ]
            rows.append(
                {
                    "subset": "e02b_magnitude_factorization",
                    "seed": int(seed),
                    "operations": operation_rows,
                    "mean_asymmetric_normalized_gain": sum(gains) / len(gains),
                    "maximum_retention_degradation": max(
                        float(item["dual_retention_mse"]) - float(item["tied_retention_mse"])
                        for item in operation_rows.values()
                    ),
                    "episodes_per_operation": int(config["data"]["count_per_operation"]),
                    "tensor_contract": {
                        "key_dim": int(config["data"]["key_dim"]),
                        "value_dim": int(config["data"]["value_dim"]),
                        "norm_angle_grid_exact": True,
                        "oracle_candidate_projection_exact": False,
                        "projection_scope": metadata["projection_scope"],
                    },
                }
            )
    passed = bool(
        all(
            float(row["mean_asymmetric_normalized_gain"])
            >= float(statistics["asymmetric_normalized_gain_sesoi"])
            and float(row["maximum_retention_degradation"])
            <= float(statistics["retention_noninferiority_margin"])
            for row in rows
        )
    )
    return rows, {
        "passed": passed,
        "seed_count": len(rows),
        "minimum_seed_normalized_gain": min(
            float(row["mean_asymmetric_normalized_gain"]) for row in rows
        ),
        "maximum_retention_degradation": max(
            float(row["maximum_retention_degradation"]) for row in rows
        ),
    }


def _sequence_probe(torch: Any, batch: Any, *, device: Any) -> tuple[dict[str, Any], Any]:
    initial = batch.inputs.initial_state.detach().cpu()
    erase_ids = batch.inputs.erase_entity_ids.detach().cpu()
    candidates = batch.inputs.candidate_values.detach().cpu()
    features = batch.inputs.demand_features.detach().cpu()
    mask = batch.update_mask.detach().cpu()
    operations = features[:, :, 4:8].argmax(dim=-1)
    batch_size, events = erase_ids.shape
    dimension = int(initial.shape[-1])
    keys = torch.nn.functional.one_hot(erase_ids, num_classes=dimension).to(torch.float32)
    erase = (((operations == 2) | (operations == 3)) & mask).to(torch.float32)
    write = (((operations == 1) | (operations == 3)) & mask).to(torch.float32)
    beta = torch.zeros(batch_size, events)
    target_cursor = initial.clone()
    row_index = torch.arange(batch_size)
    for event in range(events):
        verified = mask[:, event]
        if not bool(verified.any()):
            continue
        entity = erase_ids[:, event]
        old = target_cursor[row_index, entity].clone()
        new = candidates[:, event]
        beta[:, event] = _optimal_tied_beta(torch, old, new, erase[:, event], write[:, event])
        updated = old - erase[:, event, None] * old + write[:, event, None] * new
        target_cursor[row_index[verified], entity[verified]] = updated[verified]
    if not torch.allclose(
        target_cursor,
        batch.target_state.detach().cpu(),
        atol=1.0e-6,
        rtol=0.0,
    ):
        raise RuntimeError("official sequence translation changed the E18 target")
    probe = {
        "q": keys[:, :, None, :],
        "k": keys[:, :, None, :],
        "v": candidates[:, :, None, :],
        "g": torch.zeros(batch_size, events, 1, dimension),
        "b": erase[:, :, None, None].expand(batch_size, events, 1, dimension),
        "w": write[:, :, None, None].expand(batch_size, events, 1, dimension),
        "beta": beta[:, :, None],
        "initial_state": initial[:, None],
    }
    return (
        {name: value.to(device=device).contiguous() for name, value in probe.items()},
        batch.target_state.to(device=device),
    )


def _run_sequence_cell(
    context: Mapping[str, Any],
    *,
    source_config: Mapping[str, Any],
    seed: int,
    updates: int,
    gap_events: int,
    batches: int,
    batch_size: int,
) -> dict[str, float]:
    torch = context["torch"]
    data_module = importlib.import_module("catena.data.sequence_control_lattice")
    training_module = importlib.import_module("catena.training.sequence_control_lattice")
    family = data_module.SequenceDemandFamily.MAGNITUDE
    totals = {
        "tied_affected_mse": 0.0,
        "dual_affected_mse": 0.0,
        "tied_retention_mse": 0.0,
        "dual_retention_mse": 0.0,
    }
    for batch_index in range(batches):
        evaluation_seed = 100_000 + 10_000 * int(seed) + int(updates)
        batch = data_module.generate_sequence_control_lattice_batch(
            family=family,
            batch_size=batch_size,
            num_entities=int(source_config["data"]["num_entities"]),
            value_dim=int(source_config["data"]["value_dim"]),
            updates=int(updates),
            gap_events=int(gap_events),
            seed=training_module.indexed_sequence_lattice_seed(
                evaluation_seed,
                "evaluation-batch",
                batch_index,
            ),
            device=torch.device("cpu"),
        )
        probe, target = _sequence_probe(torch, batch, device=context["device"])
        dual_state = _gdn2(context["chunk_gdn2"], probe)
        tied_state = _kda(context["chunk_kda"], probe)
        affected = batch.affected_entities.to(device=context["device"])
        dual = _state_metrics(torch, dual_state, target, affected)
        tied = _state_metrics(torch, tied_state, target, affected)
        totals["tied_affected_mse"] += tied["affected_mse"]
        totals["dual_affected_mse"] += dual["affected_mse"]
        totals["tied_retention_mse"] += tied["retention_mse"]
        totals["dual_retention_mse"] += dual["retention_mse"]
    return {name: value / batches for name, value in totals.items()}


def _run_e18(
    context: Mapping[str, Any],
    config: Mapping[str, Any],
    replication: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    batches = int(replication["e18_evaluation_batches"])
    batch_size = int(replication["e18_evaluation_batch_size"])
    with context["strict_fp32"](context["torch"]):
        for seed in config["seeds"]:
            cell_rows: list[dict[str, Any]] = []
            for updates in config["evaluation"]["updates"]:
                for gap_events in config["evaluation"]["gap_events"]:
                    metrics = _run_sequence_cell(
                        context,
                        source_config=config,
                        seed=int(seed),
                        updates=int(updates),
                        gap_events=int(gap_events),
                        batches=batches,
                        batch_size=batch_size,
                    )
                    cell_rows.append(
                        {
                            "updates": int(updates),
                            "gap_events": int(gap_events),
                            **metrics,
                            "affected_gain": (
                                metrics["tied_affected_mse"] - metrics["dual_affected_mse"]
                            ),
                            "retention_degradation": (
                                metrics["dual_retention_mse"] - metrics["tied_retention_mse"]
                            ),
                        }
                    )
            rows.append(
                {
                    "subset": "e18_magnitude_sequence",
                    "seed": int(seed),
                    "mean_affected_gain": sum(float(cell["affected_gain"]) for cell in cell_rows)
                    / len(cell_rows),
                    "stress_affected_gain": next(
                        float(cell["affected_gain"])
                        for cell in cell_rows
                        if cell["updates"] == 8 and cell["gap_events"] == 2048
                    ),
                    "maximum_retention_degradation": max(
                        float(cell["retention_degradation"]) for cell in cell_rows
                    ),
                    "cell_metrics": cell_rows,
                    "registered_grid_exact": True,
                    "official_subset_batches": batches,
                    "official_subset_batch_size": batch_size,
                    "source_evaluation_batches": int(config["evaluation"]["batches"]),
                    "source_evaluation_batch_size": int(config["evaluation"]["batch_size"]),
                }
            )
    gate = config["claim_gate"]
    passed = bool(
        sum(float(row["mean_affected_gain"]) for row in rows) / len(rows)
        >= float(gate["minimum_corresponding_demand_gain"])
        and max(float(row["maximum_retention_degradation"]) for row in rows)
        <= float(gate["maximum_retention_degradation"])
        and all(float(row["stress_affected_gain"]) > 0.0 for row in rows)
    )
    return rows, {
        "passed": passed,
        "seed_count": len(rows),
        "mean_affected_gain": sum(float(row["mean_affected_gain"]) for row in rows) / len(rows),
        "maximum_retention_degradation": max(
            float(row["maximum_retention_degradation"]) for row in rows
        ),
        "stress_positive_seed_fraction": sum(
            float(row["stress_affected_gain"]) > 0.0 for row in rows
        )
        / len(rows),
    }


def run_minimal_replications(config: dict[str, Any]) -> dict[str, Any]:
    """Run registered official subsets after the entry-point gate authorizes them."""

    if not isinstance(config, dict):
        raise TypeError("E25a replication config must be a dictionary")
    lock = _require_self_lock()
    contracts = validate_source_contracts(config)
    replication = _mapping(config.get("replication"), "replication")
    report_env = str(replication["explicit_safe_e22b_report_env"])
    report_text = os.environ.get(report_env, "").strip()
    e22_decision = validate_safe_e22_report(
        Path(report_text) if report_text else None,
        expected_experiment_id=str(replication["source_tensor_contract"]["e22_experiment_id"]),
    )
    if e22_decision["include"]:
        # E22b identifies a selected learned locality objective and routing
        # policy.  The pinned official GDN2/KDA operator API exposes no
        # implementation of that learned objective/route.  A plain retention
        # check would not replicate E22 and is therefore forbidden.
        return {
            "passed": False,
            "configured": True,
            "scientific_evidence": False,
            "blocked_dependency": True,
            "rows": [],
            "seed_rows": [],
            "checks": {
                "e22_locality_if_supported": {
                    "passed": False,
                    "status": "NOT_IMPLEMENTED",
                    "reason": (
                        "SELECTED_E22_LOCALITY_OBJECTIVE_AND_ROUTE_ARE_NOT_"
                        "IMPLEMENTED_ON_THE_PINNED_OFFICIAL_OPERATOR_PATH"
                    ),
                }
            },
            "metrics": {},
            "subset_decisions": {
                "e02b_magnitude_factorization": {
                    "include": False,
                    "reason": "PREEMPTED_BY_REGISTERED_E22_DEPENDENCY_BLOCK",
                },
                "e18_magnitude_sequence": {
                    "include": False,
                    "reason": "PREEMPTED_BY_REGISTERED_E22_DEPENDENCY_BLOCK",
                },
                "e22_locality_if_supported": {
                    **e22_decision,
                    "implemented": False,
                    "status": "BLOCKED_DEPENDENCY",
                },
            },
            "provenance": {
                "protocol_lock_sha256": file_sha256(_LOCK_PATH),
                "protocol_lock": lock,
                "source_contracts": {
                    "paths": contracts["paths"],
                    "sha256": contracts["sha256"],
                },
                "gdn2_commit": _GDN2_COMMIT,
                "fla_commit": _FLA_COMMIT,
                "gate_adapter_sha256": _GATE_ADAPTER_SHA256,
                "replication_plugin_sha256": file_sha256(Path(__file__).resolve()),
                "official_api_only": True,
                "reference_or_mock_fallback": False,
            },
        }
    context = _official_context(config)
    e02_rows, e02 = _run_e02(context, contracts["configs"]["e02b"])
    e18_rows, e18 = _run_e18(
        context,
        contracts["configs"]["e18"],
        replication,
    )
    rows = [*e02_rows, *e18_rows]
    checks: dict[str, dict[str, Any]] = {
        "e02b_magnitude_factorization": {
            "passed": bool(e02["passed"]),
            "metrics": e02,
        },
        "e18_magnitude_sequence": {
            "passed": bool(e18["passed"]),
            "metrics": e18,
        },
    }
    checks["e22_locality_if_supported"] = {
        "passed": True,
        "skipped": True,
        "reason": e22_decision["reason"],
    }
    passed = all(bool(item["passed"]) for item in checks.values())
    seed_rows = [
        {
            "subset": str(row["subset"]),
            "seed": int(row["seed"]),
            "passed": bool(checks[str(row["subset"])]["passed"]),
        }
        for row in rows
    ]
    return {
        "passed": passed,
        "configured": True,
        "scientific_evidence": passed,
        "rows": rows,
        "seed_rows": seed_rows,
        "checks": checks,
        "metrics": {
            "e02b_mean_normalized_gain": e02["minimum_seed_normalized_gain"],
            "e18_mean_affected_gain": e18["mean_affected_gain"],
        },
        "subset_decisions": {
            "e02b_magnitude_factorization": {"include": True},
            "e18_magnitude_sequence": {"include": True},
            "e22_locality_if_supported": e22_decision,
        },
        "provenance": {
            "protocol_lock_sha256": file_sha256(_LOCK_PATH),
            "protocol_lock": lock,
            "source_contracts": {
                "paths": contracts["paths"],
                "sha256": contracts["sha256"],
            },
            "gdn2_commit": _GDN2_COMMIT,
            "fla_commit": _FLA_COMMIT,
            "gate_adapter_sha256": _GATE_ADAPTER_SHA256,
            "replication_plugin_sha256": file_sha256(Path(__file__).resolve()),
            "official_api_only": True,
            "reference_or_mock_fallback": False,
        },
    }
